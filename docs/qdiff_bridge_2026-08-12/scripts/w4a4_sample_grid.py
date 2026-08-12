"""W4A4 samples across every scale file, same seed -- what those relL2 numbers actually look like.

W4A4 is 4-bit WEIGHTS as well as activations, and integration documents its own 4-bit weight
reconstruction error at 0.1254 median relative Frobenius. So a large part of the degradation here is
not reachable by any activation calibration -- expect blur regardless. The question this answers is
narrower: does the 31% relL2 win from dropping SmoothQuant (0.7119 -> 0.4885) show up perceptually,
and does the qdiff failure (1.1945) look like the clipping it is?

Six rows, identical seed and identical noise per column:

  fp16                             the reference
  W4A4 PTQ, shipped                relL2 0.7119  the current default
  W4A4 PTQ, shipped no-smooth      relL2 0.4885  best PTQ, SmoothQuant OFF
  W4A4 PTQ, qdiff                  relL2 1.1945  the failed bridge
  W4A4 MoDiff, shipped             relL2 0.4200  the current MoDiff default
  W4A4 MoDiff, qdiff               relL2 0.3398  best MoDiff

Run: python docs/qdiff_bridge_2026-08-12/scripts/w4a4_sample_grid.py [--n 6]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
from PIL import Image, ImageDraw                                            # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

D = "docs/qdiff_bridge_2026-08-12/data"
SHIPPED = "integration/calibration/int4_calibration_realckpt.pt"
OUT = "docs/qdiff_bridge_2026-08-12/plots/w4a4_sample_grid.png"

#: (label, mode, scale file)
ARMS = [("fp16 reference", "fp16", None),
        ("W4A4 PTQ - shipped (relL2 0.7119, current default)", "int4_baseline", SHIPPED),
        ("W4A4 PTQ - SmoothQuant OFF (relL2 0.4885, best PTQ)", "int4_baseline",
         f"{D}/int4_shipped_nosmooth.pt"),
        ("W4A4 PTQ - qdiff (relL2 1.1945, the failed bridge)", "int4_baseline",
         f"{D}/qdiff_w4a4_sym.pt"),
        ("W4A4 MoDiff - shipped (relL2 0.4200, current default)", "int4", SHIPPED),
        ("W4A4 MoDiff - qdiff (relL2 0.3398, best MoDiff)", "int4", f"{D}/qdiff_w4a4_sym.pt")]


def decode(model, lat, chunk=8):
    """The VAE holds fp32 params, so an fp16 latent only decodes under the SAME autocast the sampler
    ran in. Without it: "Input type (c10::Half) and bias type (float)"."""
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
    H.STEPS, H.BATCH, H.SEED = a.steps, a.n, a.seed

    rows = []
    for label, mode, cal in ARMS:
        os.environ["MODIFF_LINEAR"] = "0"
        if cal:
            H.CALIB["int4"] = cal
        print(f"=== {label}", flush=True)
        dm = "dynamic" if mode == "int4" else "static"
        r, m, s = H.build(mode, cal, dm)
        H.SEED = a.seed
        H.latent(r, m, s)                    # discard: attention self-calibration
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
            canvas.paste(Image.fromarray(arr[i]), (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT, "PNG")
    print(f"\nwrote {OUT}  ({W}x{Hh})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
