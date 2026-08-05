"""Calibrate the W4A4 per-step delta table and test the paper's central claim.

MoDiff's headline claim is that it lets you DROP BITS: the paper's Table 2 has W8A4 going from
355.85 FID unmodulated to 3.97 modulated, i.e. a 4-bit activation model made competitive with an
8-bit one. Reproducing that needs int4 to have a delta-fitted quantizer grid, and until now it did
not -- the int4 path had no per-step delta table at all (Stage 1 was int8-only), so it quantized the
temporal delta on the ACTIVATION grid with only 15 levels. Measured consequence: int4 MoDiff 0.7772
vs int4_baseline 0.7830 -- MoDiff bought essentially nothing, exactly as Theorem 4.3 predicts when
the step size is unchanged.

The int8 table is a 4.4x improvement (0.1850 -> 0.0422). If int4 gains similarly it lands near 0.18,
which would be BETTER than int8_baseline's 0.2376 while running at 62.6 vs 70.9 ms/step -- MoDiff
int4 beating the int8 baseline on both accuracy and speed. That is the claim; this measures it.

All rows share one fp16 reference, one seed, one process, and discard run 1 as warm-up.
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

from dynamic_delta_ab import CALIB, build, latent
from integration.kernels.int4_optimized import (
    apply_int4_delta_scales, begin_delta_calibration_int4, delta_calibration_report_int4,
    end_delta_calibration_int4, export_int4_delta_scales)

TABLE = "integration/calibration/int4_delta_calibration.pt"


def warm(r, m, s):
    latent(r, m, s)
    return latent(r, m, s)


def main():
    # Calibration must run on the STATIC path: the table is recovered by inverting
    # max|code| / scale_used, which only carries information when the scale is known. Dynamic mode
    # drives max|code| to exactly Q by construction, so it would observe nothing.
    os.environ["MODIFF_DELTA_MODE"] = "static"
    os.environ["MODIFF_DELTA_REFRESH"] = "1"

    r, m, s = build("fp16", None, "static")
    ref = warm(r, m, s)[0]
    del m, s, r
    torch.cuda.empty_cache()
    print(f"fp16 reference |x|max {float(ref.abs().max()):.4f}\n", flush=True)

    # --- calibrate ---
    r, m, s = build("int4", CALIB["int4"], "static")
    n_armed = begin_delta_calibration_int4(m, reset=True)
    print(f"armed {n_armed} int4 conv layers; observing at act_scale/4 (provably cannot clip)",
          flush=True)
    latent(r, m, s)
    n_set = end_delta_calibration_int4(m)
    rep = delta_calibration_report_int4(m)
    gains = sorted(x["step_gain_tail"] for x in rep if x.get("step_gain_tail"))
    clipped = [x["obs_clipped_frac"] for x in rep if x.get("obs_clipped_frac") is not None]
    print(f"  {n_set} tables set; step gain median {gains[len(gains)//2]:.2f}x max {gains[-1]:.2f}x; "
          f"observation clipped on {sum(1 for c in clipped if c > 0)}/{len(clipped)} layers",
          flush=True)
    table = export_int4_delta_scales(m)
    torch.save(table, TABLE)
    print(f"  wrote {TABLE} ({len(table)} layers)\n", flush=True)
    del m, s, r
    torch.cuda.empty_cache()

    # --- A/B ---
    out = {}
    rows = [("int4_baseline (MoDiff off)", "int4_baseline", "static", False),
            ("int4 MoDiff, table OFF", "int4", "static", False),
            ("int4 MoDiff, table ON", "int4", "static", True),
            ("int4 MoDiff dynamic (ref)", "int4", "dynamic", False),
            ("int8_baseline (the bar to beat)", "int8_baseline", "static", False)]
    for label, mode, dm, use_table in rows:
        calib = CALIB["int8"] if "int8" in mode else CALIB["int4"]
        r, m, s = build(mode, calib, dm)
        n = apply_int4_delta_scales(m, torch.load(TABLE, weights_only=True)) if use_table else 0
        lat, ms = warm(r, m, s)
        rel = float((lat - ref).norm() / ref.norm())
        out[label] = {"rel_l2_vs_fp16": rel, "ms_per_step": ms, "table_layers": n}
        print(f"  {label:34s} relL2 {rel:.4f}   {ms:7.2f} ms/step"
              + (f"   (table on {n} layers)" if n else ""), flush=True)
        del m, s, r
        torch.cuda.empty_cache()

    off = out["int4 MoDiff, table OFF"]["rel_l2_vs_fp16"]
    on = out["int4 MoDiff, table ON"]["rel_l2_vs_fp16"]
    i8 = out["int8_baseline (the bar to beat)"]["rel_l2_vs_fp16"]
    print(f"\n{'=' * 76}")
    print(f"  int4 delta table effect: {off:.4f} -> {on:.4f}  ({off / on:.2f}x better)")
    print(f"  vs int8_baseline ({i8:.4f}): int4+table is "
          f"{'BETTER' if on < i8 else 'still worse'} by {abs(i8 - on) / i8 * 100:.0f}%")
    print(f"  If better, MoDiff W4A4 beats the W8A8 baseline on accuracy AND on speed")
    print(f"  (62.6 vs 70.9 ms/step at batch 128) -- the paper's drop-a-bit claim, reproduced.")

    with open("docs/modiff_correctness_2026-08-03/data/int4_delta_table.json", "w") as f:
        json.dump({"results": out, "median_step_gain": gains[len(gains) // 2]}, f, indent=2)


if __name__ == "__main__":
    main()
