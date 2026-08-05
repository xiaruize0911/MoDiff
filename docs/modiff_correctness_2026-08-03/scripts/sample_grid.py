"""Generate and decode real samples per mode, plus a labelled side-by-side comparison grid.

Every quality claim in FINDINGS so far is a latent relative-L2 number. Those are the right metric for
tracking a quantizer, but they are not something you can look at. This decodes actual images through
the VAE so the differences are visible, and puts the same seed's sample from every mode next to each
other in one PNG.

Discipline carried over from the rest of the harness, because both matter here:
  * the REAL-checkpoint calibration (integration/calibration/*_realckpt.pt). The un-suffixed files
    were fitted to the 856-byte stub's random weights and give latent relL2 0.88/3.02 with real
    weights, so images generated on them would look broken for the wrong reason.
  * one warm-up sampling run per mode, discarded. The quantized attention blocks self-calibrate over
    their first forwards, so run 1 is several x worse than run 2 (FINDINGS 2026-08-04) -- a grid built
    from first runs would make every quantized mode look worse than it is.
  * one shared seed, so row-to-row differences are the mode and nothing else.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.dirname(os.path.abspath(__file__))]

import torch
from PIL import Image, ImageDraw

from dynamic_delta_ab import CALIB, build, latent

OUT = "docs/modiff_correctness_2026-08-03/samples"
N_IMG = int(os.environ.get("GRID_N", "6"))
MODES = [("fp16", "fp16", "static", "fp16 (reference)"),
         ("int8_baseline", "int8_baseline", "static", "W8A8 baseline (MoDiff off)"),
         ("int8_modiff", "int8", "dynamic", "W8A8 + MoDiff (dynamic)"),
         ("int4_baseline", "int4_baseline", "static", "W4A4 baseline (MoDiff off)"),
         ("int4_modiff", "int4", "dynamic", "W4A4 + MoDiff (dynamic)")]


def decode(model, lat):
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        img = model.decode_first_stage(lat.to("cuda", torch.float16))
    img = torch.clamp((img.float() + 1.0) / 2.0, 0.0, 1.0)
    return (img.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, meta, ref = [], {}, None
    for tag, mode, dm, label in MODES:
        calib = None if mode == "fp16" else CALIB["int4" if "int4" in mode else "int8"]
        r, m, s = build(mode, calib, dm)
        latent(r, m, s)                       # warm-up: discard, see module docstring
        lat, ms = latent(r, m, s)
        if ref is None:
            ref = lat
        rel = float((lat - ref).norm() / ref.norm())
        arr = decode(m, lat[:N_IMG])
        d = os.path.join(OUT, tag)
        os.makedirs(d, exist_ok=True)
        for i in range(arr.shape[0]):
            Image.fromarray(arr[i]).save(os.path.join(d, f"{i:02d}.png"))
        rows.append((label, arr, rel, ms))
        meta[tag] = {"label": label, "rel_l2_vs_fp16": rel, "ms_per_step_batch8": ms,
                     "dir": d, "n_images": int(arr.shape[0])}
        print(f"  {label:28s} relL2 vs fp16 {rel:.4f}   {ms:6.2f} ms/step   -> {d}", flush=True)
        del m, s, r
        torch.cuda.empty_cache()

    # --- comparison grid: one row per mode, labelled ---
    H = W = rows[0][1].shape[1]
    pad, lab = 4, 26
    gw = pad + N_IMG * (W + pad)
    gh = sum(lab + H + pad for _ in rows) + pad
    grid = Image.new("RGB", (gw, gh), (18, 18, 18))
    draw = ImageDraw.Draw(grid)
    y = pad
    for label, arr, rel, ms in rows:
        draw.text((pad, y + 7), f"{label}    latent relL2 vs fp16 = {rel:.4f}", fill=(235, 235, 235))
        y += lab
        for i in range(min(N_IMG, arr.shape[0])):
            grid.paste(Image.fromarray(arr[i]), (pad + i * (W + pad), y))
        y += H + pad
    gp = os.path.join(OUT, "comparison_grid.png")
    grid.save(gp)
    print(f"\n  comparison grid -> {gp}  ({gw}x{gh})")

    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump({"seed": 1234, "steps": 50, "n_images": N_IMG,
                   "grid": gp, "modes": meta}, f, indent=2)
    print(f"  index -> {os.path.join(OUT, 'index.json')}")


if __name__ == "__main__":
    main()
