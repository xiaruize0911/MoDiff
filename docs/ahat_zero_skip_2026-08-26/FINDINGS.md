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

## Follow-up 2026-08-26: is a_hat's SHAPE sparse — checked by channel and by spatial position, not just by warp

Prompted by "a_hat的shape可以做sparse吗" (can a_hat's shape support a sparse representation). The
original measurement above only scored zeros at the kernel's actual memory-access unit (64
consecutive flat-NHWC elements = up to 64 consecutive channels at one spatial position) — it never
asked whether zeros cluster along a *structural* axis (a channel that is reliably near-zero, or a
spatial position that is), which a coarser skip (per-channel or per-tile) could exploit even if
the fine-grained warp view looks scattered.

[`scripts/measure_zero_code_structure.py`](scripts/measure_zero_code_structure.py) captures the
real `(N, C, H, W)` delta-code tensor for one representative layer (192 channels, 32×32, same
20-step/batch-4 uncalibrated regime as the original measurement, 266 calls) and computes zero-rate
and whole-slice-zero rate along both axes:

| axis | zero-rate mean ± std (range) | whole-slice-zero-this-call rate: mean / **max** |
|---|---|---|
| per-channel (192 channels) | 47.2% ± 2.8% (37.8–53.6%) | 1.5% / **4.5%** |
| per-spatial-position (1024 positions) | 47.2% ± 0.6% (45.6–48.8%) | 0.5% / **1.1%** |

Both axes are close to flat — the small per-channel spread (std 2.8%, vs. 0.6% spatially) is the
only hint of structure, and it does not translate into anything actionable: **no single channel is
ever fully zero more than 4.5% of the 266 calls**, and spatially the ceiling is 1.1%. Neither comes
anywhere close to a rate a per-channel or per-tile skip could use.

### Verdict

**Refuted at every granularity now checked**: element (the original 64-wide warp, 1.7%), channel
(4.5% ceiling), and spatial position (1.1% ceiling). The ~50% zero rate is spread almost perfectly
evenly across every axis this data supports grouping by — there is no structural axis along which
reshaping or regrouping a_hat's traffic would concentrate zeros into skippable blocks. Any sparse
encoding (mask, index list, run-length) would be paying overhead against a density that is not
sparse enough (~50%, not the 90%+ regime sparse formats need to pay for themselves) while also
losing the coalesced-access bandwidth the dense write already gets (73–81% of peak, §1 above). "Make
a_hat's shape sparse" is dead, on the same uncalibrated-regime caveat as the parent measurement.

### Files

- [`scripts/measure_zero_code_structure.py`](scripts/measure_zero_code_structure.py)
- [`data/zero_code_structure.npz`](data/zero_code_structure.npz)
