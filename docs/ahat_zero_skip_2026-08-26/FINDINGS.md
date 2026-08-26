# Warp-granularity zero-code skip on a_hat's write: dead, and the real data shows why

**Status: measured, refuted.** No CUDA code was written or changed for this idea.

## The idea

`a_hat += q/scale`; when the quantized delta code `q == 0` that add is a no-op, so skipping the
write is **exact**, not an approximation — unlike every other idea in this session, this one
carries zero numerics risk if it works. GPUs move memory in fixed-size sectors, not per element,
so the write can only actually be skipped at **warp granularity**:
`gn_apply_delta_quantize_flat_vec2_kernel` gives one warp (32 threads) 64 consecutive flat-NHWC
elements (2 per thread, vec2) — that is the unit a predicated skip would have to operate on.

## Measurement

[`scripts/measure_zero_code_rate.py`](scripts/measure_zero_code_rate.py) hooks
`modiff_cutlass.group_norm_silu_delta_quantize_nhwc` / `_pack_nhwc` at the Python attribute level
(both are called as `modiff_cutlass.<name>(...)`, confirmed at
[`int8_optimized.py:963`](../../integration/kernels/int8_optimized.py:963), a plain
module-attribute lookup that a pre-run monkeypatch intercepts) and captures **every real delta
code from an actual 20-step int8 generation**, batch 4 — not synthetic data.

| | elem zero-rate | warp-all-zero rate |
|---|--:|--:|
| all 2356 calls | 52.98% | **1.70%** |
| first half (early steps) | 51.45% | 1.65% |
| second half (late steps) | 54.50% | 1.76% |

**Over half of individual codes are exactly zero — but only 1.7% of 64-element warps have EVERY
code zero.** The zeros are scattered, not clustered. A skip that only fires on 1.7% of warps has
to beat the cost of the warp-vote check (`__all_sync` or equivalent) it would need on the *other*
98.3% of warps that do not qualify — on any reasonable cost model that is a net loss, not a win.
**Refuted without needing to write the kernel.**

## Why the early/late split did not show the pattern the calibration data predicted

This run had **no delta calibration file**, so every layer fell back to the
full-activation-scale path (`_delta_scale_args`'s warning fired on every layer — see the captured
log). That is a different quantization regime from the shipped per-step delta table, and is the
likely reason the early/late gap here (51.45% -> 54.50%) is far smaller than the 12.45x-124.5x
`step_gain_tail` swing measured on calibrated layers
([int8_ahat_cache_2026-08-26](../int8_ahat_cache_2026-08-26/FINDINGS.md)). The **structural**
finding — zeros are scattered rather than clustered at 64-element granularity — is a property of
which channels/positions in the activation are correlated, not of which scale quantizes them, so
it is unlikely to reverse under calibration; but the specific 53%/1.7% numbers should be read as
"this regime," not as the calibrated deployment's numbers. A re-run with real calibration files
would tighten this if the question ever needs a precise answer.

## What survives

Nothing to build. This closes one candidate from the list without spending any CUDA-development
budget on it — the measurement was the whole cost.

## Files

- [`scripts/measure_zero_code_rate.py`](scripts/measure_zero_code_rate.py) — self-contained,
  reproduces the capture against a real (if uncalibrated) generation run
