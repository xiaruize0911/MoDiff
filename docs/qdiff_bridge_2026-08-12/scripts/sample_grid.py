"""Decoded samples from each arm, same seed, side by side -- the perceptual check relL2 cannot give.

relL2 and FID both say the Q-Diffusion activation scales fix a real problem in the W8A8 PTQ baseline
(0.2564 -> 0.1119). This renders it, so the claim can be looked at rather than taken on trust.

FOUR ARMS, one row each, identical seed and identical latent noise per column, so a column is the
SAME image generated four ways and differences are attributable:

  fp16                          the reference
  W8A8 PTQ, shipped scales      the arm with the bad calibration (relL2 0.2564, FID 16.37)
  W8A8 PTQ, qdiff scales        the same arm, corrected            (relL2 0.1119)
  W8A8 MoDiff conv-only, qdiff  the configuration Stage D made the default

Warm-up discipline is the usual one and is not optional: the quantized attention blocks self-calibrate
over their first MODIFF_ATTN_CALIB_STEPS forwards, so a first sample is several x worse than steady
state. Each arm discards one batch before the one that gets rendered.

Run: python docs/qdiff_bridge_2026-08-12/scripts/sample_grid.py [--n 6] [--steps 50]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

SHIPPED = "integration/calibration/int8_calibration_realckpt.pt"
QDIFF = "integration/calibration/int8_calibration_qdiff.pt"
OUT = "docs/qdiff_bridge_2026-08-12/plots/sample_grid.png"

#: (row label, mode, calibration, MODIFF_LINEAR)
ARMS = [("fp16 reference", "fp16", None, "0"),
        ("W8A8 PTQ - shipped scales (relL2 0.2564)", "int8_baseline", SHIPPED, "0"),
        ("W8A8 PTQ - qdiff scales (relL2 0.1119)", "int8_baseline", QDIFF, "0"),
        ("W8A8 MoDiff conv-only - qdiff (new default)", "int8", QDIFF, "0")]


def decode(model, lat, chunk=8):
    """Latent -> uint8 RGB. Chunked: the VAE takes 32x32 -> 256x256, and one big decode OOM'd once."""
    lat = lat.to("cuda", torch.float16)
    out = []
    # The VAE holds fp32 params, so an fp16 latent only works under the SAME autocast the sampler
    # ran in -- generate_fid_samples.py:144 wraps sampling and decoding in one context and that is
    # load-bearing, not incidental. Without it: "Input type (c10::Half) and bias type (float)".
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
    H.STEPS, H.BATCH, H.SEED = a.steps, a.n, a.seed

    rows = []
    for label, mode, calib, lin in ARMS:
        os.environ["MODIFF_LINEAR"] = lin
        cal = calib or (None if mode == "fp16" else H.CALIB["int8"])
        print(f"=== {label}", flush=True)
        r, m, s = H.build(mode, cal, "static" if mode in ("fp16", "int8_baseline") else "dynamic")
        H.SEED = a.seed
        H.latent(r, m, s)                       # discard: attention self-calibration + clock ramp
        H.SEED = a.seed
        lat, _ = H.latent(r, m, s)
        rows.append((label, decode(m, lat)))
        del r, m, s
        torch.cuda.empty_cache()
    os.environ.pop("MODIFF_LINEAR", None)

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
    print(f"\nwrote {OUT}  ({W}x{Hh})")

    # Per-arm strips too: the grid is wide, and a single row is easier to look at closely.
    for label, arr in rows:
        tag = label.split(" -")[0].replace(" ", "_").replace("/", "_")
        strip = Image.new("RGB", (W, cell + pad * 2), (252, 252, 251))
        for i in range(min(a.n, arr.shape[0])):
            strip.paste(Image.fromarray(arr[i]), (pad + i * (cell + pad), pad))
        strip.save(f"docs/qdiff_bridge_2026-08-12/plots/samples_{tag}.png", "PNG")
    print("wrote per-arm strips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
