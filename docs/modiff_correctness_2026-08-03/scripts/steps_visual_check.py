"""Does MoDiff @16 steps actually LOOK as good as the baseline @50? Decide by looking.

steps_equal_quality.py found MoDiff @16 steps at latent distance 0.2242 against the baseline @50's
0.2505, implying a 2.9x speedup at equal quality. That number needs a visual check before it can be
believed, for a specific reason: MoDiff's distance curve is NON-MONOTONE in step count (32 steps
scored better than 50), which means the metric is not measuring what it looks like it measures. DDIM
picks a different timestep subset for every S, so distance-to-fp16@200 jumps around with S
independently of quality, and latent L2 does not capture the perceptual signature of under-sampling
(loss of fine detail, over-smoothing) that a 16-step sample is expected to show.

Rows, all from one seed:
  fp16 @ 50            what the model can do
  int8_baseline @ 50   the bar, 3547 ms/sample at batch 128
  int8 + MoDiff @ 50   1.08x the bar's time
  int8 + MoDiff @ 25   1.86x
  int8 + MoDiff @ 16   2.91x  <- the claim
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.dirname(os.path.abspath(__file__))]

import torch
from PIL import Image, ImageDraw

from steps_equal_quality import CALIB, MS, build, sample

OUT = "docs/modiff_correctness_2026-08-03/samples_steps"
N_IMG = 5
ROWS = [("fp16", None, 50, "fp16 @ 50 steps"),
        ("int8_baseline", "int8", 50, "int8_baseline @ 50   (the bar)"),
        ("int8", "int8", 50, "int8 + MoDiff @ 50"),
        ("int8", "int8", 25, "int8 + MoDiff @ 25   (1.86x the bar's time)"),
        ("int8", "int8", 16, "int8 + MoDiff @ 16   (2.91x the bar's time)")]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for mode, cal, steps, label in ROWS:
        r, m, s = build(mode, None if cal is None else CALIB[cal])
        sample(r, m, s, 50)                      # warm-up
        lat = sample(r, m, s, steps)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            img = m.decode_first_stage(lat[:N_IMG].to("cuda", torch.float16))
        img = torch.clamp((img.float() + 1.0) / 2.0, 0.0, 1.0)
        arr = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
        ms = steps * MS.get(mode, 102.4)
        rows.append((f"{label}    {ms:.0f} ms/sample", arr))
        print(f"  {label:44s} {ms:6.0f} ms/sample", flush=True)
        del m, s, r
        torch.cuda.empty_cache()

    H = W = rows[0][1].shape[1]
    pad, lab = 4, 26
    grid = Image.new("RGB", (pad + N_IMG * (W + pad), sum(lab + H + pad for _ in rows) + pad),
                     (18, 18, 18))
    draw = ImageDraw.Draw(grid)
    y = pad
    for label, arr in rows:
        draw.text((pad, y + 7), label, fill=(235, 235, 235))
        y += lab
        for i in range(min(N_IMG, arr.shape[0])):
            grid.paste(Image.fromarray(arr[i]), (pad + i * (W + pad), y))
        y += H + pad
    gp = os.path.join(OUT, "steps_comparison.png")
    grid.save(gp)
    print(f"\n  -> {gp}")


if __name__ == "__main__":
    main()
