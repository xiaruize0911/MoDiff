"""Fake-quant a_hat along C, B=32. fp16 cache, qmax=127, dynamic per-block absmax.

Uses the production hook (MODIFF_AHAT_BLOCK=32), not a monkey-patch.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/ahat_blockwise_2026-09-01/scripts/block32_fake_gen.py
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
os.environ["MODIFF_AHAT_BLOCK"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="block32_fake_gen.py")

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
import ahat_fake_quant_grid as G  # noqa: E402

SHAPE = (4, 32, 32)
OUT_PNG = "docs/ahat_blockwise_2026-09-01/plots/block32_fake.png"
OUT_JSON = "docs/ahat_blockwise_2026-09-01/data/block32_fake.json"


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
        output_dir="docs/ahat_blockwise_2026-09-01/tmp_block32",
        batch_size=a.n, steps=a.steps, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)

    rows, quality, ref = [], {}, None

    def add(model, label, lat):
        nonlocal ref
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        quality[label] = {"relL2_vs_fp16": rel}
        print(f"  {label:36s} relL2 {rel:.4f}", flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label,
                     G.decode(model, lat)))

    print("===== fp16 =====", flush=True)
    os.environ["MODIFF_AHAT_BLOCK"] = "0"
    model, sampler = runner._setup_model("fp16")
    sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=False)
    add(model, "fp16 reference",
        sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=False))
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int8 =====", flush=True)
    runner.calibration_path = B._default_calibration_path("int8")
    os.environ["MODIFF_AHAT_BLOCK"] = "0"
    model, sampler = runner._setup_model("int8")

    sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=True)
    add(model, "W8A8  a_hat fp16",
        sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=True))

    os.environ["MODIFF_AHAT_BLOCK"] = "32"
    sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=True)
    add(model, "W8A8  a_hat along-C B=32 fake",
        sample(runner, model, sampler, a.n, a.seed, a.steps, quantized=True))
    os.environ["MODIFF_AHAT_BLOCK"] = "0"

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
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "block": 32,
               "relL2": quality, "out": OUT_PNG}, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_PNG}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
