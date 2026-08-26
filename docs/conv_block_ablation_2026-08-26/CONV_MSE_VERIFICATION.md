# Conv kernel numerical verification (MSE vs fp64)

**Date** 2026-08-26 · **GPU** NVIDIA A40 · **Batch** 8

[`bench_conv_block_ablation.py`](scripts/bench_conv_block_ablation.py) times
`conv2d_int{8,4}_evt_bias_residual_fp16` and `conv2d_int{8,4}_evt_o_hat` but feeds them random
codes and never checks the output is a correct convolution. No existing test in
`integration/tests/` covers these four kernels' numerics either (`grep -rl conv2d_int8
integration/` finds nothing). [`verify_conv_mse.py`](scripts/verify_conv_mse.py) closes that gap.

## Method

Independent float64 reference: dequantize (`code * dequant_multiplier`, the convention confirmed
from `integration/kernels/int8_optimized.py:682` and `:1198` — `weight_scale_channel` and `alpha`
are both **multipliers**, not divisors), run `F.conv2d` in float64, then compare against the real
CUDA kernel's output by MSE and max-abs-error. Baseline is scored against `conv_ref + bias +
residual` (both empty here); the MoDiff/o_hat arm is scored against `o_hat_before + conv_ref`,
checked after the kernel's in-place accumulate.

## Result: correct, at fp16-rounding precision

| | int8 | int4 |
|---|--:|--:|
| relative max-abs-error, all shapes/arms | 2.3e-4 – 4.4e-4 | 2.6e-4 – 4.3e-4 |

Every one of 24 (shape × precision × arm) checks lands in this narrow band — no outliers, no
per-shape anomalies, and `base`/`o_hat` agree closely within each shape (expected: `o_hat` is
scored against the same `conv_ref` plus one extra float64 addition). The int32 dot product and the
`alpha * weight_scale[k]` multiply are exact; the only lossy step is the kernel's final round to
fp16 on store, which the reference does not itself replicate (it stays in float64 throughout) — so
a residual at fp16's ~1e-3 relative precision is the expected floor, and what was measured (2–4e-4)
is comfortably inside it, not at it.

## Files

- [`scripts/verify_conv_mse.py`](scripts/verify_conv_mse.py) — self-contained

## Scope and limitations

- 6 of the 20 real shapes, batch 8 — a numerics check, not a timing one.
- Weight/activation codes are independently randomized per-tensor, not read from a real
  calibration file; the dequantization convention is what is being checked, not calibration
  quality.
- Does not check `conv2d_int{8,4}_evt_bias_residual_fp16`'s `bias`/`residual` epilogue terms
  (passed empty here) or the `_residual` o_hat variant — only the two entry points
  `bench_conv_block_ablation.py` actually times.
