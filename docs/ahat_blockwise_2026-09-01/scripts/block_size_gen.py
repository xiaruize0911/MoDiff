"""n=6 contact sheet: fake-quant a_hat at the block sizes the sweep ranked.

Mutates the live fp16 cache after every write (same contract as storing
int8 codes and dequantizing on the next load). Does not change the delta
code path. W8A8 kernels, static delta table.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/ahat_blockwise_2026-09-01/scripts/block_size_gen.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="block_size_gen.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.kernels.int8_optimized import OptimizedInt8Conv2d  # noqa: E402
import ahat_fake_quant_grid as G  # noqa: E402

SHAPE = (4, 32, 32)
QMAX = 127.0
OUT_PNG = "docs/ahat_blockwise_2026-09-01/plots/block_size_grid.png"
OUT_JSON = "docs/ahat_blockwise_2026-09-01/data/block_size_gen.json"

# (label, kind, B)  kind: none | tensor | token | c | hw | channel
ARMS = (
    ("W8A8  a_hat fp16", "none", 0),
    ("W8A8  per-tensor int8", "tensor", 0),
    ("W8A8  tokenwise", "token", 0),
    ("W8A8  along-C B=8", "c", 8),
    ("W8A8  along-C B=16", "c", 16),
    ("W8A8  along-C B=32", "c", 32),
    ("W8A8  per-channel", "channel", 0),
)

SCHEME = {"kind": "none", "B": 0}
ORIG = OptimizedInt8Conv2d._after_ahat_write


def _snap_along_c_(a, B):
    x = a.permute(0, 2, 3, 1).contiguous().float()
    n, h, w, c = x.shape
    B = min(int(B), c)
    pad = (B - c % B) % B
    xp = F.pad(x, (0, pad)) if pad else x
    g = xp.shape[-1] // B
    blk = xp.reshape(n, h, w, g, B)
    amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    s = amax / QMAX
    q = (blk / s).round().clamp(-QMAX, QMAX)
    recon = (q * s).reshape(n, h, w, -1)[..., :c]
    a.copy_(recon.permute(0, 3, 1, 2).to(a.dtype))


def _snap_along_hw_(a, B):
    n, c, h, w = a.shape
    hw = h * w
    B = min(int(B), hw)
    y = a.float().reshape(n, c, hw)
    pad = (B - hw % B) % B
    yp = F.pad(y, (0, pad)) if pad else y
    g = yp.shape[-1] // B
    blk = yp.reshape(n, c, g, B)
    amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    s = amax / QMAX
    q = (blk / s).round().clamp(-QMAX, QMAX)
    recon = (q * s).reshape(n, c, -1)[..., :hw].reshape(n, c, h, w)
    a.copy_(recon.to(a.dtype))


def _snap_(a):
    kind, B = SCHEME["kind"], SCHEME["B"]
    if kind == "none":
        return
    if kind == "tensor":
        xf = a.float()
        amax = xf.abs().amax().clamp_min(1e-12)
        s = amax / QMAX
        a.copy_(((xf / s).round().clamp(-QMAX, QMAX) * s).to(a.dtype))
        return
    if kind == "token":
        _snap_along_c_(a, a.shape[1])
        return
    if kind == "c":
        _snap_along_c_(a, B)
        return
    if kind == "channel":
        _snap_along_hw_(a, a.shape[2] * a.shape[3])
        return
    if kind == "hw":
        _snap_along_hw_(a, B)


def _hook(self, out):
    ret = ORIG(self, out)
    a = self.a_hat_cache
    if a is None or a.numel() == 0 or a.dtype not in (torch.float16, torch.float32):
        return ret
    _snap_(a)
    return ret


def sample(runner, model, sampler, n, seed, steps, quantized):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  n={a.n} steps={a.steps}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_blockwise_2026-09-01/tmp_gen",
        batch_size=a.n, steps=a.steps, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)

    rows, quality, ref = [], {}, None

    def add(label, lat):
        nonlocal ref
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        quality[label] = {"relL2_vs_fp16": rel}
        print(f"  {label:32s} relL2 {rel:.4f}", flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label,
                     G.decode(model, lat)))

    print("===== fp16 =====", flush=True)
    model, sampler = runner._setup_model("fp16")
    sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=False)
    lat = sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=False)
    add("fp16 reference", lat)
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int8 =====", flush=True)
    runner.calibration_path = B._default_calibration_path("int8")
    model, sampler = runner._setup_model("int8")
    OptimizedInt8Conv2d._after_ahat_write = _hook
    try:
        for label, kind, Bsz in ARMS:
            SCHEME["kind"], SCHEME["B"] = kind, Bsz
            sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=True)
            lat = sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=True)
            add(label, lat)
    finally:
        OptimizedInt8Conv2d._after_ahat_write = ORIG
        SCHEME["kind"] = "none"

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
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "qmax": QMAX,
               "relL2": quality, "out": OUT_PNG}, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_PNG}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        OptimizedInt8Conv2d._after_ahat_write = ORIG
