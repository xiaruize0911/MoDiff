"""Fake-quant a_hat storage vs fp32 a_hat, on the telescoping simulation path.

NOT the kernel path. The model stays fp16; a forward_pre_hook replaces each live
ResBlock conv's input with the MoDiff-reconstructed a_hat (act_fake_quant.py's
linearity trick: o_hat_t = A(a_hat_t) + b). The only extra knob is whether that
a_hat is snapped onto an N-bit grid after every update, which is what storing
a_hat as int8/int4 and dequantizing it would do.

Because the conv sees the snapped a_hat, this is the OPTIMISTIC simulation:
storage error goes through the linear map. The earlier kernel monkey-patches
(docs/int8_ahat_cache_2026-08-26) were pessimistic — o_hat never saw the extra
rounding, so a_hat and o_hat diverged.

Arms, one row each, same seed / same noise per column:

  fp16
  W8A8 MoDiff fake-quant, a_hat fp32
  W8A8 MoDiff fake-quant, a_hat int8  (dynamic absmax, never clips)
  W4A4 MoDiff fake-quant, a_hat fp32
  W4A4 MoDiff fake-quant, a_hat int4

Activation and delta use a dynamic (per-call absmax) quantizer — no calibration
file required, and it is the paper's Theorem 4.3 setting. Weights are fake-quant
in place: per-output-channel absmax at 8 bits, MSE-clip search at 4 bits (the
shipped _int4_weight_scale rule).

Run: python docs/ahat_fake_quant_2026-08-27/scripts/ahat_fake_quant_grid.py
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts")]

import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
from ldm.models.diffusion.ddim import DDIMSampler                          # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402
import act_fake_quant as A                                                 # noqa: E402

OUT = "docs/ahat_fake_quant_2026-08-27/plots/ahat_fake_quant_grid.png"
JSON = "docs/ahat_fake_quant_2026-08-27/data/ahat_fake_quant.json"
WARMUP_ROUNDS = A.WARMUP_ROUNDS
DELTA_REFRESH = A.DELTA_REFRESH


def q(v, s, qmax):
    return torch.clamp(torch.round(v * s), -qmax, qmax) / s


def int4_mse_weight_scale(w_flat, Q=7.0):
    """Same search as integration.kernels.int4_optimized._int4_weight_scale (mse)."""
    am = w_flat.abs().max(dim=1).values
    best_err = best_scale = None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4):
        sc = torch.clamp(am * r / Q, min=1e-8)
        e = (((w_flat / sc[:, None]).round().clamp(-Q, Q) * sc[:, None] - w_flat) ** 2).sum(1)
        if best_err is None:
            best_err, best_scale = e, sc
        else:
            m = e < best_err
            best_err = torch.where(m, e, best_err)
            best_scale = torch.where(m, sc, best_scale)
    return best_scale


class AhatHook:
    """MoDiff delta recursion with an optional a_hat storage snap.

    mode 'modiff': t=T warmup then t<T delta, matching act_fake_quant.DeltaHook.
    quantize_ahat: after every a_hat write, snap onto a dynamic qmax-grid.
    """

    def __init__(self, qmax, quantize_ahat=False, refresh=DELTA_REFRESH):
        self.qmax = float(qmax)
        self.quantize_ahat = quantize_ahat
        self.refresh = refresh
        self.a_hat = None
        self.step = 0
        self._held = None
        self.snap_l2 = 0.0
        self.snaps = 0

    def reset(self):
        self.a_hat, self.step, self._held = None, 0, None

    def _snap(self, a):
        if not self.quantize_ahat:
            return a
        s = self.qmax / a.abs().max().clamp_min(1e-6)
        snapped = q(a, s, self.qmax)
        den = a.norm().clamp_min(1e-12)
        self.snap_l2 += float((snapped - a).norm() / den)
        self.snaps += 1
        return snapped

    def __call__(self, mod, args):
        x = args[0].float()
        if self.a_hat is None or self.a_hat.shape != x.shape:
            s_act = self.qmax / x.abs().max().clamp_min(1e-6)
            a = q(x, s_act, self.qmax)
            for _ in range(WARMUP_ROUNDS - 1):
                r = x - a
                rs = self.qmax / r.abs().max().clamp_min(1e-6)
                a = a + q(r, rs, self.qmax)
            self.a_hat = self._snap(a)
            self.step = 1
            self._held = None
        else:
            d = x - self.a_hat
            if self._held is None or (self.step - 1) % self.refresh == 0:
                self._held = float(self.qmax / d.abs().max().clamp_min(1e-6))
            self.a_hat = self._snap(self.a_hat + q(d, self._held, self.qmax))
            self.step += 1
        return (self.a_hat.to(args[0].dtype),) + args[1:]


def quantize_weights_(convs, bits):
    Q = 127.0 if bits == 8 else 7.0
    saved = {}
    for name, m in convs.items():
        w = m.weight.data
        saved[name] = w.clone()
        wf = w.reshape(w.shape[0], -1).float()
        sc = (torch.clamp(wf.abs().max(dim=1).values / Q, min=1e-8) if bits == 8
              else int4_mse_weight_scale(wf, Q))
        wq = (wf / sc[:, None]).round().clamp(-Q, Q) * sc[:, None]
        m.weight.data = wq.reshape_as(w).to(w.dtype)
    return saved


def restore_weights_(convs, saved):
    for name, m in convs.items():
        m.weight.data = saved[name]


SHAPE = (4, 32, 32)


def sample_latent(model, sampler, steps, batch, seed):
    """One DDIM pass. Churches is unconditional; seed is reset so columns pair."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=batch, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def decode(model, lat, chunk=8):
    lat = lat.to("cuda", torch.float16)
    out = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for i in range(0, lat.shape[0], chunk):
            d = model.decode_first_stage(lat[i:i + chunk])
            out.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0).permute(0, 2, 3, 1).cpu())
            del d
    return (torch.cat(out, 0).numpy() * 255).round().astype("uint8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"
    os.environ["MODIFF_QUANT_LINEAR"] = "0"

    for p in ("models/ldm/lsun_churches256/model.ckpt",
              "models/first_stage_models/kl-f8/model.ckpt"):
        if not os.path.exists(p) or os.path.getsize(p) < 10_000:
            print(f"FAIL: need a real checkpoint at {p}")
            return 1

    print(f"building fp16 model (batch {a.n}, {a.steps} steps) ...", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_fake_quant_2026-08-27/tmp_out",
        batch_size=a.n, steps=a.steps, shape=SHAPE, calibration_path=None,
        auto_delta_table=False)
    model, sampler = runner._setup_model("fp16")
    convs = A.target_convs(model.model.diffusion_model)
    print(f"  {len(convs)} FusedResBlock conv hook targets", flush=True)
    if len(convs) < 50:
        print(f"FAIL: expected ~70 live convs, got {len(convs)}")
        return 1

    # (label, bits or 0, MoDiff, quantize_ahat)
    ARMS = [
        ("fp16 reference", 0, False, False),
        ("W8A8 MoDiff  a_hat fp32", 8, True, False),
        ("W8A8 MoDiff  a_hat int8", 8, True, True),
        ("W4A4 MoDiff  a_hat fp32", 4, True, False),
        ("W4A4 MoDiff  a_hat int4", 4, True, True),
    ]

    rows, quality, ref = [], {}, None
    for label, bits, modiff, qahat in ARMS:
        saved = quantize_weights_(convs, bits) if bits else None
        hooks, handles = [], []
        if modiff:
            qmax = 127.0 if bits == 8 else 7.0
            for _key, mod in convs.items():
                h = AhatHook(qmax=qmax, quantize_ahat=qahat)
                hooks.append(h)
                handles.append(mod.register_forward_pre_hook(h))
        sample_latent(model, sampler, a.steps, a.n, a.seed)   # discard: cudnn warmup
        for h in hooks:
            h.reset()
        lat = sample_latent(model, sampler, a.steps, a.n, a.seed)
        for hd in handles:
            hd.remove()
        if saved is not None:
            restore_weights_(convs, saved)
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        mean_snap = (sum(h.snap_l2 for h in hooks) / max(1, sum(h.snaps for h in hooks))
                     if qahat else 0.0)
        quality[label] = {"relL2_vs_fp16": rel, "mean_ahat_snap_relL2": mean_snap,
                          "bits": bits, "quantize_ahat": qahat}
        print(f"  {label:36s} relL2 {rel:.4f}"
              + (f"  a_hat snap {mean_snap:.4f}" if qahat else ""), flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label, decode(model, lat)))

    cell, pad, lab = a.cell, 6, 26
    W = pad + a.n * (cell + pad)
    Hh = len(rows) * (cell + lab + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for label, arr in rows:
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += lab
        for i in range(min(a.n, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT, "PNG")
    os.makedirs(os.path.dirname(JSON), exist_ok=True)
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n,
               "warmup_rounds": WARMUP_ROUNDS, "delta_refresh": DELTA_REFRESH,
               "relL2": quality, "out": OUT}, open(JSON, "w"), indent=1)
    print(f"\nwrote {OUT}  ({W}x{Hh})\nwrote {JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
