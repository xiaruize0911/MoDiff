"""All fake-quant is group quant: weights, activations, delta, and a_hat.

Same group size G along:
  weights  — flattened Cin·kH·kW, per output channel (pad K if needed)
  acts/δ   — channels C at each (n,h,w) (pad C if needed)
  I-MoDiff a_hat — same channel groups, s* refreshed every step

No per-tensor act scale, no per-output-channel weight scale.

  W: [Cout, K] -> [Cout, nG, G], scale = absmax/G_qmax over the G dim
  x: [N,C,H,W] -> [N, nG, G, H, W], scale per (n, group, h, w)

I-MoDiff (integer a_hat, refresh every step):
  s*_g = qmax / max(|x|_g)
  a    = sat(round((a / s_old) * s*_g))
  q    = sat(round(x * s*_g) − a)
  a   += q ;  conv input = a / s*_g

MoDiff (fp32 a_hat): group-wise Q on x (t=T + warmup) and on (x−a_hat).

G ∈ {128, 64, 32}. W8A8 qmax=127, W4A4 qmax=7.

Run: source setup_cuda_env.sh
     python docs/cache_schemes_report_2026-08-28/scripts/imode_blockwise_samples.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts")]

os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_QUANT_ATTN"] = "0"
os.environ["MODIFF_QUANT_LINEAR"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="imode_blockwise_samples.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
import act_fake_quant as A  # noqa: E402
import ahat_fake_quant_grid as G  # noqa: E402

OUT_PNG = "docs/cache_schemes_report_2026-08-28/plots/imode_group_quant_grid.png"
OUT_DIR = "docs/cache_schemes_report_2026-08-28/plots/imode_group_quant"
JSON_OUT = "docs/cache_schemes_report_2026-08-28/data/imode_group_quant.json"
SHAPE = (4, 32, 32)
GROUPS = (128, 64, 32)
WARMUP = G.WARMUP_ROUNDS


def _group_nchw(x, gsize):
    n, c, h, w = x.shape
    pad = (gsize - c % gsize) % gsize
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
    ng = x.shape[1] // gsize
    return x.reshape(n, ng, gsize, h, w), c


def _ungroup_nchw(xg, c):
    n, ng, gs, h, w = xg.shape
    return xg.reshape(n, ng * gs, h, w)[:, :c]


def q_group(x, qmax, gsize):
    """Fake-quant x with one scale per channel-group at each spatial location."""
    xg, c = _group_nchw(x, gsize)
    s = qmax / xg.abs().amax(dim=2, keepdim=True).clamp_min(1e-6)
    y = (xg * s).round().clamp(-qmax, qmax) / s
    return _ungroup_nchw(y, c)


def quantize_weights_group_(convs, bits, gsize):
    """Group-quant weights along flattened Cin·kH·kW. One scale per (Cout, group)."""
    qmax = 127.0 if bits == 8 else 7.0
    saved = {}
    for name, m in convs.items():
        w = m.weight.data
        saved[name] = w.clone()
        shape = w.shape
        wf = w.reshape(w.shape[0], -1).float()
        k = wf.shape[1]
        pad = (gsize - k % gsize) % gsize
        if pad:
            wf = F.pad(wf, (0, pad))
        ng = wf.shape[1] // gsize
        wg = wf.reshape(wf.shape[0], ng, gsize)
        sc = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / qmax
        wq = (wg / sc).round().clamp(-qmax, qmax) * sc
        wq = wq.reshape(wf.shape[0], -1)[:, :k].reshape(shape)
        m.weight.data = wq.to(w.dtype)
    return saved


class GroupMoDiffHook:
    """MoDiff with fp32 a_hat; act and delta are group-quantized every step."""

    def __init__(self, qmax, gsize):
        self.qmax = float(qmax)
        self.gsize = int(gsize)
        self.a_hat = None

    def reset(self):
        self.a_hat = None

    def _q(self, v):
        return q_group(v, self.qmax, self.gsize)

    def __call__(self, mod, args):
        x = args[0].float()
        if self.a_hat is None or self.a_hat.shape != x.shape:
            a = self._q(x)
            for _ in range(WARMUP - 1):
                a = a + self._q(x - a)
            self.a_hat = a
        else:
            self.a_hat = self.a_hat + self._q(x - self.a_hat)
        return (self.a_hat.to(args[0].dtype),) + args[1:]


class GroupIModeHook:
    """Integer a_hat, group s* refreshed every step."""

    def __init__(self, qmax, gsize):
        self.qmax = float(qmax)
        self.gsize = int(gsize)
        self.a = None
        self.s = None
        self.C0 = None
        self.max_sat = 0.0
        self.n_sat = 0
        self.n_elem = 0

    def reset(self):
        self.a = self.s = self.C0 = None
        self.max_sat = 0.0
        self.n_sat = self.n_elem = 0

    def _note(self, a):
        self.max_sat = max(self.max_sat, float(a.abs().max()) / self.qmax)
        self.n_sat += int((a.abs() >= self.qmax - 0.5).sum())
        self.n_elem += a.numel()

    def __call__(self, mod, args):
        x = args[0].float()
        xg, c = _group_nchw(x, self.gsize)
        self.C0 = c
        amax = xg.abs().amax(dim=2, keepdim=True).clamp_min(1e-6)
        s_new = self.qmax / amax
        if self.a is None or self.a.shape != xg.shape:
            self.a = (xg * s_new).round().clamp(-self.qmax, self.qmax)
        else:
            a = (self.a / self.s * s_new).round().clamp(-self.qmax, self.qmax)
            xi = (xg * s_new).round().clamp(-self.qmax, self.qmax)
            q = (xi - a).clamp(-self.qmax, self.qmax)
            self.a = (a + q).clamp(-self.qmax, self.qmax)
        self.s = s_new
        self._note(self.a)
        recon = _ungroup_nchw(self.a / self.s, self.C0)
        return (recon.to(args[0].dtype),) + args[1:]


def _stats(hooks):
    imode = [h for h in hooks if isinstance(h, GroupIModeHook)]
    if not imode:
        return None
    return {
        "max_sat": max(h.max_sat for h in imode),
        "sat_frac": (sum(h.n_sat for h in imode)
                     / max(1, sum(h.n_elem for h in imode))),
        "n_hooks": len(imode),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  n={a.n} steps={a.steps}  "
          f"groups={list(GROUPS)}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(OUT_DIR, "_tmp"),
        batch_size=a.n, steps=a.steps, shape=SHAPE, calibration_path=None,
        auto_delta_table=False)
    model, sampler = runner._setup_model("fp16")
    convs = A.target_convs(model.model.diffusion_model)
    print(f"  {len(convs)} FusedResBlock conv hook targets", flush=True)
    if len(convs) < 50:
        print(f"FAIL: expected ~70 live convs, got {len(convs)}")
        return 1

    # (label, bits, gsize, imode)
    arms = [("fp16", 0, None, False)]
    for bits, tag in ((8, "W8A8"), (4, "W4A4")):
        for g in GROUPS:
            arms.append((f"{tag} MoDiff G={g}  w/act/δ group", bits, g, False))
            arms.append((f"{tag} I-MoDiff G={g}  refresh", bits, g, True))

    rows, quality, ref = [], {}, None

    def add(label, lat, arr, extra=None):
        nonlocal ref
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / (ref.norm() + 1e-12))
        rec = {"relL2_vs_fp16": rel}
        if extra:
            rec.update(extra)
        quality[label] = rec
        tag = label if rel == 0.0 else f"{label}    relL2 {rel:.4f}"
        extra_s = ""
        if extra and "max_sat" in extra:
            extra_s = f"  sat={extra['max_sat']:.3f} frac={extra['sat_frac']:.4f}"
        print(f"  {label:40s} relL2 {rel:.4f}{extra_s}", flush=True)
        rows.append((tag, arr))
        folder = os.path.join(OUT_DIR, label.replace(" ", "_").replace("=", "").replace("/", "-"))
        os.makedirs(folder, exist_ok=True)
        for i in range(arr.shape[0]):
            Image.fromarray(arr[i]).save(os.path.join(folder, f"{i:06d}.png"))

    for label, bits, gsize, imode in arms:
        saved = quantize_weights_group_(convs, bits, gsize) if bits else None
        hooks, handles = [], []
        if bits:
            qmax = 127.0 if bits == 8 else 7.0
            cls = GroupIModeHook if imode else GroupMoDiffHook
            for _key, mod in convs.items():
                h = cls(qmax=qmax, gsize=gsize)
                hooks.append(h)
                handles.append(mod.register_forward_pre_hook(h))
        G.sample_latent(model, sampler, a.steps, a.n, a.seed)
        for h in hooks:
            h.reset()
        lat = G.sample_latent(model, sampler, a.steps, a.n, a.seed)
        for hd in handles:
            hd.remove()
        if saved is not None:
            G.restore_weights_(convs, saved)
        add(label, lat, G.decode(model, lat), _stats(hooks))

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
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    canvas.save(OUT_PNG, "PNG")
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "groups": list(GROUPS),
               "note": "group quant everywhere: weights (Cin·kH·kW), act/delta and "
                       "I-MoDiff a_hat (along C). Same G per arm. s* refresh every step. "
                       "fake-quant; o_hat not quantized",
               "relL2": quality, "out": OUT_PNG}, open(JSON_OUT, "w"), indent=1)
    print(f"\nwrote {OUT_PNG}  ({W}x{Hh})\nwrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
