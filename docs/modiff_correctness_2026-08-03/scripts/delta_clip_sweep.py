"""Sweep the dynamic delta quantizer's clipping ratio, W8A8 and W4A4.

Why a sweep rather than a mode choice. The 2026-08-04 A/B (dynamic_delta_ab.py) came out split:
pure-absmax dynamic beat the static activation grid by 1.77x at W4A4 but LOST at W8A8
(0.1777 -> 0.2313). Both facts have the same cause -- max|delta| reaches roughly the full
activation range on outlier elements, so Q/max|delta| is a COARSER step than the activation-grid
scale, just one that never clips. Which side of that trade wins depends on how many levels you
have to spend: at 127 you would rather clip the tail and resolve the bulk, at 15 you cannot afford
to clip at all.

So "static vs dynamic" is really one knob, not two modes:

    scale = Q / (ratio * max|delta|)

    ratio = 1.0   pure absmax, cannot clip, coarsest
    ratio < 1.0   finer grid, clips everything above `ratio` of the observed range
    ratio -> 0    approaches (and passes) the static activation-grid scale

This finds the optimum on each side instead of assuming it. Same fp16 reference, seed, activation
calibration and process for every point.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamic_delta_ab import CALIB, build, latent  # one definition of the harness

RATIOS = [float(r) for r in os.environ.get(
    "SWEEP_RATIOS", "1.0,0.7,0.5,0.35,0.25,0.15,0.1").split(",")]
# Static reference points, re-measured in-process at steady state. The 2026-08-04 first-run
# figures (int8 baseline 0.2798 / static 0.1777, int4 baseline 0.7648 / static 0.7606) are NOT
# usable as references: they were taken on run 1, before the quantized attention blocks finished
# self-calibrating, which inflates every mode by several x.
STATIC = {}


def main():
    r, m, s = build("fp16", None, "static")
    latent(r, m, s)                 # warm-up, see dynamic_delta_ab.measure
    ref, _ = latent(r, m, s)
    del m, s, r
    torch.cuda.empty_cache()
    print(f"fp16 reference |x|max {float(ref.abs().max()):.4f}\n", flush=True)

    out = {}
    for bits in ("int8", "int4"):
        print(f"{'=' * 74}\n{bits}\n{'=' * 74}", flush=True)
        # Steady-state static references, same process and warm-up discipline as the sweep points.
        STATIC[bits] = {}
        for tag, mode, dm in ((f"baseline", f"{bits}_baseline", "static"),
                              (f"static", bits, "static")):
            r, m, s = build(mode, CALIB[bits], dm)
            latent(r, m, s)
            lat, ms = latent(r, m, s)
            STATIC[bits][tag] = float((lat - ref).norm() / ref.norm())
            print(f"  {tag:14s} relL2 {STATIC[bits][tag]:.4f}   {ms:7.2f} ms/step", flush=True)
            del m, s, r
            torch.cuda.empty_cache()
        out[bits] = {}
        for ratio in RATIOS:
            os.environ["MODIFF_DELTA_CLIP"] = str(ratio)
            r, m, s = build(bits, CALIB[bits], "dynamic")
            latent(r, m, s)         # discard run 1: attention is still self-calibrating
            lat, ms = latent(r, m, s)
            rel = float((lat - ref).norm() / ref.norm())
            out[bits][str(ratio)] = {"rel_l2_vs_fp16": rel, "ms_per_step": ms}
            print(f"  ratio {ratio:5.2f}   relL2 {rel:.4f}   {ms:7.2f} ms/step", flush=True)
            del m, s, r
            torch.cuda.empty_cache()
        os.environ["MODIFF_DELTA_CLIP"] = "1.0"

        best = min(out[bits].items(), key=lambda kv: kv[1]["rel_l2_vs_fp16"])
        b = best[1]["rel_l2_vs_fp16"]
        print(f"\n  best ratio {best[0]}: relL2 {b:.4f}   "
              f"vs static {STATIC[bits]['static']:.4f} ({STATIC[bits]['static'] / b:.3f}x)   "
              f"vs baseline {STATIC[bits]['baseline']:.4f} "
              f"({STATIC[bits]['baseline'] / b:.3f}x)\n", flush=True)

    with open("docs/modiff_correctness_2026-08-03/data/delta_clip_sweep.json", "w") as f:
        json.dump({"ratios": RATIOS, "static_reference": STATIC, "results": out}, f, indent=2)
    print("wrote docs/modiff_correctness_2026-08-03/data/delta_clip_sweep.json")


if __name__ == "__main__":
    main()
