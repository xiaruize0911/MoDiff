"""a_hat bit-width hyperparameter on the three existing scheme classes.

Schemes (unchanged classification):
  full        — compute every step, store every step  (SKIP=1, REPLAY=1)
  skip-K=4    — compute every step, freeze cache 3/4  (MODIFF_CACHE_SKIP_K=4)
  replay-K=4  — skip GN+conv 3/4, reuse o_hat         (MODIFF_REPLAY_K=4)

MODIFF_AHAT_BITS:
  16 — fp16 a_hat (shipped)
  8  — real int8 a_hat storage: kernels dequant on load / quant on store
  4  — real int4-grid a_hat storage (unpacked int8 codes, qmax=7)

MODIFF_AHAT_REFRESH:
  0  — hold the t=T per-tensor scale (default)
  1  — unpack to fp16 on commit, pack with a fresh absmax (fake-quant grid)

Run: python docs/ahat_bits_2026-08-27/scripts/ahat_bits_gen.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts")]

import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402
import ahat_fake_quant_grid as G                                           # noqa: E402

OUT = "docs/ahat_bits_2026-08-27/plots/ahat_bits_grid.png"
JSON_OUT = "docs/ahat_bits_2026-08-27/data/ahat_bits_gen.json"
SHAPE = (4, 32, 32)
MODE_TAG = {"int8": "W8A8", "int4": "W4A4"}
SCHEMES = (
    ("full", 1, 1),
    ("skip-K=4", 4, 1),
    ("replay-K=4", 1, 4),
)


def _apply(skip_k, replay_k, bits, refresh=0):
    os.environ["MODIFF_CACHE_SKIP_K"] = str(skip_k)
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_AHAT_REFRESH"] = str(refresh)


def _reset(model, mode):
    unet = model.model.diffusion_model
    if "int8" in mode:
        B.reset_modiff_state_int8(unet)
    elif "int4" in mode:
        B.reset_modiff_state_int4(unet)
    B._reset_wxax_modiff_safe(model)


def sample(model, sampler, steps, batch, seed, mode):
    _reset(model, mode)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=batch, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def bits_tag(bits, refresh=0):
    if bits >= 16:
        return "fp16"
    kind = f"int{bits}"
    return f"{kind} refresh" if refresh else f"{kind} held"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()
    os.environ.setdefault("MODIFF_DELTA_MODE", "static")
    os.environ["MODIFF_LINEAR"] = "0"
    _apply(1, 1, 16, 0)

    print(f"GPU: {torch.cuda.get_device_name()}  n={a.n} steps={a.steps}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_bits_2026-08-27/tmp_gen",
        batch_size=a.n, steps=a.steps, shape=SHAPE, calibration_path=None,
        auto_delta_table=True)

    rows, quality, ref = [], {}, None

    def add_arm(label, lat, extra=None):
        nonlocal ref
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        rec = {"relL2_vs_fp16": rel}
        if extra:
            rec.update(extra)
        quality[label] = rec
        print(f"  {label:32s} relL2 {rel:.4f}", flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label, G.decode(model, lat)))

    print("===== fp16 =====", flush=True)
    model, sampler = runner._setup_model("fp16")
    sample(model, sampler, a.steps, a.n, a.seed, "fp16")
    lat = sample(model, sampler, a.steps, a.n, a.seed, "fp16")
    add_arm("fp16 reference", lat)
    del model, sampler
    torch.cuda.empty_cache()

    mode_bits = {"int8": (16, 8, 4), "int4": (16, 4)}
    for mode, bit_list in mode_bits.items():
        tag = MODE_TAG[mode]
        print(f"===== {mode} =====", flush=True)
        runner.calibration_path = B._default_calibration_path(mode)
        _apply(1, 1, 16, 0)
        model, sampler = runner._setup_model(mode)
        for scheme, skip_k, replay_k in SCHEMES:
            for bits in bit_list:
                refreshes = (0,) if bits >= 16 else (0, 1)
                for refresh in refreshes:
                    _apply(skip_k, replay_k, bits, refresh)
                    sample(model, sampler, a.steps, a.n, a.seed, mode)
                    lat = sample(model, sampler, a.steps, a.n, a.seed, mode)
                    label = f"{tag} {scheme}  a_hat {bits_tag(bits, refresh)}"
                    add_arm(label, lat, {"scheme": scheme, "skip_k": skip_k,
                                         "replay_k": replay_k, "ahat_bits": bits,
                                         "refresh": refresh})
        _apply(1, 1, 16, 0)
        del model, sampler
        torch.cuda.empty_cache()

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
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "relL2": quality, "out": OUT},
              open(JSON_OUT, "w"), indent=1)
    print(f"\nwrote {OUT}  ({W}x{Hh})\nwrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _apply(1, 1, 16, 0)
