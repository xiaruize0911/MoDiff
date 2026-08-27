"""Skip-K a_hat / o_hat cache updates, on the fake-quant telescoping path.

NOT the kernel path. Same linearity trick as ahat_fake_quant_grid.py: the conv sees
reconstructed a_hat, so o_hat = A(a_hat) + b for that forward.

K = cache commit period:
  t=T: always seed a_hat (warmup rounds, same as MoDiff).
  t<T: residual d = Q(x - a_hat_cache) every step; conv input = a_hat_cache + d.
       Commit a_hat_cache <- that reconstruction only every K steps.
       The K-1 steps after a commit leave the cache frozen, so the next residual
       is still x - a_hat_stale. That is the scheme: same residual formula, no
       cache write.

This is NOT skip2-exact (deferred DRAM write + reconstruct the missed a_hat from
the pending code). Skip2-exact is bit-identical to K=1. This one is the stale-
reference skip the kernel path would use if it simply stopped writing a_hat/o_hat.

Residual scale is per-call absmax (cannot clip from a stale scale) so the grid
isolates cache-freeze error. Weights: W8 per-channel absmax, W4 MSE-clip search.

Run: python docs/ahat_fake_quant_2026-08-27/scripts/ahat_skip_k_grid.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts")]

import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402
import act_fake_quant as A                                                 # noqa: E402
import ahat_fake_quant_grid as G                                           # noqa: E402

OUT = "docs/ahat_fake_quant_2026-08-27/plots/ahat_skip_k_grid.png"
JSON = "docs/ahat_fake_quant_2026-08-27/data/ahat_skip_k.json"
KS = (1, 2, 4, 8)


def q(v, s, qmax):
    return torch.clamp(torch.round(v * s), -qmax, qmax) / s


class SkipKHook:
    """MoDiff residual every step; a_hat cache written only every K steps."""

    def __init__(self, qmax, skip_k=1):
        self.qmax = float(qmax)
        self.skip_k = int(skip_k)
        self.a_hat = None
        self.step = 0
        self.n_commit = 0
        self.n_skip = 0
        self.delta_rel = 0.0
        self.n_delta = 0

    def reset(self):
        self.a_hat, self.step = None, 0
        self.n_commit = self.n_skip = 0
        self.delta_rel = 0.0
        self.n_delta = 0

    def __call__(self, mod, args):
        x = args[0].float()
        if self.a_hat is None or self.a_hat.shape != x.shape:
            s_act = self.qmax / x.abs().max().clamp_min(1e-6)
            a = q(x, s_act, self.qmax)
            for _ in range(G.WARMUP_ROUNDS - 1):
                r = x - a
                rs = self.qmax / r.abs().max().clamp_min(1e-6)
                a = a + q(r, rs, self.qmax)
            self.a_hat = a
            self.step = 1
            self.n_commit += 1
            used = a
        else:
            d = x - self.a_hat
            s = self.qmax / d.abs().max().clamp_min(1e-6)
            dq = q(d, s, self.qmax)
            den = x.norm().clamp_min(1e-12)
            self.delta_rel += float(d.norm() / den)
            self.n_delta += 1
            used = self.a_hat + dq
            # t=T already committed. Next commit on step K, 2K, ...
            if self.step % self.skip_k == 0:
                self.a_hat = used
                self.n_commit += 1
            else:
                self.n_skip += 1
            self.step += 1
        return (used.to(args[0].dtype),) + args[1:]


SHAPE = (4, 32, 32)


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

    # (label, bits or 0, skip_k or 0)
    ARMS = [("fp16 reference", 0, 0)]
    for bits, tag in ((8, "W8A8"), (4, "W4A4")):
        for k in KS:
            ARMS.append((f"{tag} MoDiff  skip-K={k}", bits, k))

    rows, quality, ref = [], {}, None
    for label, bits, skip_k in ARMS:
        saved = G.quantize_weights_(convs, bits) if bits else None
        hooks, handles = [], []
        if skip_k:
            qmax = 127.0 if bits == 8 else 7.0
            for _key, mod in convs.items():
                h = SkipKHook(qmax=qmax, skip_k=skip_k)
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
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        n_commit = sum(h.n_commit for h in hooks)
        n_skip = sum(h.n_skip for h in hooks)
        mean_d = (sum(h.delta_rel for h in hooks) / max(1, sum(h.n_delta for h in hooks))
                  if hooks else 0.0)
        quality[label] = {"relL2_vs_fp16": rel, "bits": bits, "skip_k": skip_k,
                          "n_commit": n_commit, "n_skip": n_skip,
                          "mean_delta_relL2": mean_d}
        print(f"  {label:32s} relL2 {rel:.4f}  commit {n_commit} skip {n_skip}  "
              f"|d|/|x| {mean_d:.4f}", flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label,
                     G.decode(model, lat)))

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
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "K": list(KS),
               "warmup_rounds": G.WARMUP_ROUNDS, "relL2": quality, "out": OUT},
              open(JSON, "w"), indent=1)
    print(f"\nwrote {OUT}  ({W}x{Hh})\nwrote {JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
