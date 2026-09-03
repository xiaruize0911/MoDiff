# Kernel-1 accuracy: the metric I proposed as primary is the wrong one

W8A8, LSUN-churches LDM-KL-8, batch 4, 49 of 50 DDIM steps, 5 conv layers spanning
CPG in {6,12,18,24,48} and spatial 4..32. `scripts/capture.py` -> `scripts/measure.py` ->
`data/accuracy_int8.json`, curves in `plots/kernel1_accuracy.png`.

## Method

`capture.py` monkeypatches `group_norm_silu_delta_quantize_nhwc` during a live sampling run and
records every argument except a_hat -- x_t, GN weight/bias, num_groups, eps, apply_silu, the
per-step delta scale, smooth_inv, mod_scale/shift -- then calls through. `measure.py` replays
those captured inputs through the REAL CUDA kernel once per arm, so the **only** thing that
differs between arms is a_hat storage. Reference = the same recurrence in fp32 with a_hat held
exactly (including the kernel's `__half2float(__float2half(n))` round before SiLU, so that is not
part of the measured difference). Open loop by construction: every arm sees the same x_t.

Only the plain entry point is covered: 62 of 70 conv layers. The 8 updown ResBlocks go through
`group_norm_silu_delta_quantize_resize_nhwc`.

## Result: `consumed` self-corrects and is nearly blind to a_hat precision

Median over t>=5 (after the transient), averaged over the 5 layers, `x floor` = multiple of the
fp16-a_hat arm:

| arm | consumed | x floor | state | x floor | codes | sat |
|---|---|---|---|---|---|---|
| fp16 (floor) | 3.82e-03 | 1.00x | 3.83e-03 | 1.00x | 0.105 | 0.000 |
| i8 B=16 | 8.72e-03 | 2.28x | 1.11e-02 | 2.91x | 0.449 | 0.063 |
| i8 B=32 | 8.82e-03 | 2.31x | 1.21e-02 | 3.16x | 0.490 | 0.032 |
| i8 B=64 | 8.86e-03 | 2.32x | 1.31e-02 | 3.42x | 0.523 | 0.016 |
| **i4 B=32** | **9.41e-03** | **2.46x** | **1.33e-01** | **34.58x** | **0.916** | 0.040 |

**i4 stores a_hat 10x more coarsely than i8 (state 1.33e-01 vs 1.21e-02) and yet reconstructs the
activation to within 7% of i8's error (9.41e-03 vs 8.82e-03).** That is not a measurement bug, it
is MoDiff's structure: the delta is `o - dequant(a_hat_stored)`, measured against the SAME coarse
value the reconstruction then adds back, so `consumed = dequant(a_hat_stored) + q/s` cancels the
a_hat error to first order and leaves only the delta quantizer's own error. Coarsening a_hat makes
the delta larger, and the delta quantizer absorbs it.

So the metric I nominated as primary cannot rank a_hat storage schemes. Corrected: the
discriminating metrics are

1. **`codes`** -- fraction of delta codes differing from the reference. fp16 0.105, i8 ~0.45-0.52,
   **i4 0.916**. Discrete and directly interpretable: it is what the GEMM consumes differently.
2. **`state`** -- a_hat drift. Orders i8 B=16 < B=32 < B=64 exactly as granularity predicts
   (2.91x / 3.16x / 3.42x the floor), and separates i4 by 10x.

`sat` (a_hat codes pinned at +-limit) rules out saturation as i4's mechanism: 4.0%, between
i8 B=32's 3.2% and B=16's 6.3%. i4 fails by plain coarseness, not by pinning.

## Nothing compounds in open loop

Every metric saturates by t~4 and is flat for the remaining 45 steps (see the curves). i4's state
error is already 1.44e-01 at t=0 and *declines* slightly to 1.30e-01 by t=48. So the E2E collapse
in Addendum 9/10 is **not** open-loop error accumulation inside this kernel -- it has to be the
closed loop: the a_hat state error perturbs the codes, the conv output moves, and the model's own
next-step input moves with it. Open-loop kernel accuracy therefore cannot predict the E2E
outcome, which is the same lesson `docs/act_budget_2026-09-02` reached from the other direction.

Practical consequence: use `codes` and `state` to RANK storage schemes cheaply, and keep decoded
samples as the only accept/reject gate.

## Ordering, for the record

By `state` and `codes`, finer blocks are strictly better and bits dominate blocks:
B=16 < B=32 < B=64 at 8 bits, all four int8 configurations far ahead of i4 B=32. Combined with
Addendum 10's speed (i8 B=32 fastest, i4 no faster than fp16 a_hat) and memory (i4 3.2x smaller),
i8 B=32 remains the only configuration that is not dominated on some axis.


---

## Addendum: the metric that DOES predict E2E — accumulated storage error

The per-step metrics above rank the arms but cannot explain the E2E collapse, because all of them
saturate by t~4. The quantity that does not saturate was missing from the first pass.

**Where a_hat storage error actually goes.** Write `eta_t = dequant(a_hat_t) - consumed_t`, the
storage rounding at step t. The activation reconstruction cancels a_hat exactly:

    consumed_t = dequant(a_hat_{t-1}) + q_t/s = o_t + eps      (eps = delta rounding only)

but `o_hat` does not, because it is written from the CODES while a_hat is written from the ROUNDED
value. Unrolling `o_hat_t = o_hat_{t-1} + conv(consumed_t - dequant(a_hat_{t-1}))`:

    out_T = conv(consumed_T) - conv(eta_1 + eta_2 + ... + eta_{T-1})

The two caches part by `eta_t` once per step, and the conv output carries the **running sum**,
uncorrected. So the predictive metric is `||sum_k eta_k|| / ||consumed||`.

Measured (5-layer mean, `eta_cum` in `data/accuracy_int8.json`):

| arm | eta / step | sum eta @ t=48 | growth | consumed | codes |
|---|---|---|---|---|---|
| fp16 | 0.0002 | 0.0015 | 7.9x | 3.82e-03 | 0.105 |
| i8 B=16 | 0.0059 | 0.0412 | 7.0x | 8.72e-03 | 0.449 |
| i8 B=32 | 0.0072 | **0.0508** | 7.1x | 8.82e-03 | 0.490 |
| i8 B=64 | 0.0086 | 0.0607 | 7.1x | 8.86e-03 | 0.523 |
| **i4 B=32** | 0.1314 | **1.9247** | 14.7x | 9.41e-03 | 0.916 |

**i4's accumulated storage error exceeds the signal itself** (1.92 vs 1.0) -- by t~15 on the
C=384 layer it is already past 1.0. i8 B=32 tops out at 0.051. That is the whole difference
between usable images and the shard texture in `docs/ahat_only_conv_2026-09-02/samples`, and it
is visible from a cheap open-loop replay.

Note the growth column: i8 accumulates at ~sqrt(48) = 6.9x, i.e. an uncorrelated random walk.
i4 accumulates at 14.7x, more than twice that -- its rounding errors are correlated step to step
(a coarse grid rounds the same way repeatedly), so they add rather than cancel. Coarser storage is
worse than the per-step numbers suggest.

**Revised recommendation for this kernel:** `eta_cum` is the primary metric. `state` and `codes`
rank correctly but saturate; `consumed` is structurally blind. A scheme is safe if `eta_cum` stays
well under 1.0 over the full trajectory.

---

## Addendum: single layer, random input, 4-bit down to B=2 — finer blocks do NOT rescue it

`scripts/single_layer_sweep.py` -> `data/single_layer_sweep.json`, `plots/ahat_pareto.png`.
C=384, 16x16, batch 4, 49 steps, one layer, synthetic input.

The real i4 kernel is B=32 only (`ahat_b32_update2_i4` hardcodes `gi = i>>5`, a 16-lane reduction
and `threadIdx.x & 15`), so the sweep uses a PyTorch model of the storage quantizer, VALIDATED
against the real kernel at B=32 first -- final-a_hat relL2 1.2e-02 (8-bit) / 7.9e-03 (4-bit) on
the walk trajectory. That agreement is the same order as the per-step storage error itself
(~7e-03), because GN's fp32 reduction order differs in the last bits and code-boundary flips get
amplified by the accumulator. So absolute values carry ~1e-02 of slop; the B-dependence below
spans 0.87 -> 5.05, two orders larger, so the ordering and magnitudes are safe.

Two input trajectories, since a_hat is a temporal accumulator: `iid` (independent each step) and
`walk` (`x_{t+1} = sqrt(1-a) x_t + sqrt(a) eps`, a=0.02). The real captured trajectory's value for
4-bit B=32 (1.92, previous addendum) sits between the two synthetic ones (1.73 iid, 3.99 walk),
which is a useful consistency check on both.

| a_hat | B/elem | eta_cum @48 (walk) | (iid) | state | codes | **sat** |
|---|---|---|---|---|---|---|
| fp16 | 2.000 | 0.0000 | 0.0000 | 0 | 0.000 | 0.000 |
| **8-bit B=32** | 1.125 | **0.0469** | **0.0552** | 5.0e-02 | 0.186 | 0.032 |
| 4-bit B=2 | **2.500** | 0.8665 | 0.3518 | 5.7e-02 | 0.179 | **0.525** |
| 4-bit B=4 | 1.500 | 1.4948 | 0.5883 | 8.1e-02 | 0.310 | 0.280 |
| 4-bit B=8 | 1.000 | 2.1642 | 0.9215 | 1.1e-01 | 0.441 | 0.139 |
| 4-bit B=16 | 0.750 | 2.9585 | 1.2530 | 1.3e-01 | 0.574 | 0.070 |
| 4-bit B=32 | 0.625 | 3.9873 | 1.7280 | 1.6e-01 | 0.700 | 0.037 |
| 4-bit B=64 | 0.562 | 5.0498 | 2.4972 | 1.9e-01 | 0.785 | 0.020 |

**Finer blocks improve 4-bit monotonically -- B=2 is 4.6x better than B=32 -- and it still never
gets close.** Even B=2 is 18x worse than 8-bit B=32 (0.87 vs 0.047), and B=2 costs **2.5 B/elem,
more than fp16's 2.0**. Every 4-bit configuration that saves memory (B>=8) has accumulated error
at or above the signal itself. There is no crossing point: the two curves never meet, and the
8-bit point sits below and to the left of the entire 4-bit frontier on the memory axis that
matters (1.125 vs 1.000 for 4-bit B=8, a 12% memory difference for a 46x error difference).

## Why fine blocks cannot help: the amax code is spent, not earned

`sat` explains it. With symmetric per-block amax scaling, **at least 1 code in every B is exactly
at the limit by construction** -- it is the element that defined the amax. Measured: B=2 pins
52.5% of codes, B=4 28%, B=8 14%, B=32 3.7%, i.e. exactly ~1/B. At 4 bits there are only 15
levels, so at B=2 half the stored codes carry no information at all, and the scale storage that
bought that granularity costs 2 B/elem on its own.

So shrinking B trades information-free codes and scale bytes for resolution, and at 4 bits the
trade never pays. This is a property of amax block scaling, not of this implementation.

**Conclusion: 4-bit a_hat is dead at every block size.** Combined with Addendum 10 (i4 is also no
faster than fp16 a_hat), int8 B=32 remains the recommendation, and the remaining cheap memory win
is fp16 block scales (1.125 -> 1.0625 B/elem) rather than fewer bits.

---

## Addendum: the metrics, calibrated — where each one's threshold actually is

The earlier "keep eta_cum well under 1.0" was a guess from a 2-point bracket (8-bit fine, 4-bit
broken). `scripts/bits_grid.py` fills in 5/6/7 bits x B=16/32/64 on the REAL captured inputs, and
three more E2E sim runs (`int4_ahat32_sim5/6/7`) locate the visual boundary. Crossing the two
turns the metric into a threshold. Grid in `data/bits_grid.json`, samples in
`docs/ahat_only_conv_2026-09-02/samples/labeled_bits_boundary.png`.

### The error curve is smooth and regular

Median over the 5 layers, eta_cum at t=48 (min/max across layers in the JSON):

| bits | B=16 | B=32 | B=64 | eta_step (B=32) | codes (B=32) | sat |
|---|---|---|---|---|---|---|
| 4 | 1.498 | 1.982 | 2.543 | 0.135 | 0.916 | ~1/B |
| 5 | 0.506 | 0.657 | 0.828 | 0.063 | 0.847 | ~1/B |
| 6 | 0.200 | 0.254 | 0.305 | 0.030 | 0.716 | ~1/B |
| 7 | 0.090 | 0.110 | 0.131 | 0.015 | 0.535 | ~1/B |
| 8 | 0.043 | 0.053 | 0.063 | 0.007 | 0.399 | ~1/B |

**Each extra bit divides eta_cum by ~2.45** (halving the step would give 2.0; the extra factor is
reduced step-to-step correlation of the rounding). **Each doubling of B multiplies it by ~1.13.**
So one bit is worth about seven doublings of B -- bits dominate blocks, decisively.

### Calibrated bands, anchored to the decoded samples

| eta_cum | verdict | anchor |
|---|---|---|
| <= 0.15 | indistinguishable from fp16 | 8-bit (0.053), 7-bit (0.110) — both clean |
| 0.15 - 0.30 | safe, mild softening | 6-bit (0.254) — recognizable, slightly hazier |
| 0.30 - 0.70 | **marginal, visible degradation** | 5-bit (0.657) — structure survives, texture erodes |
| > 1 | broken | 4-bit (1.982) — shard texture; 3-bit (~6) — noise |

So the usable threshold is **eta_cum < 0.3**, not "< 1.0" -- by 1.0 it is already past saving.
Equivalently on the isolated storage term, **eta_step < 0.03**.

`codes` spans only 0.40 -> 0.92 over the same range from clean to destroyed, so it is a fine
ranking signal and a bad threshold. `consumed` moves 8.8e-03 -> 9.4e-03 across that whole range
and is useless for either. `sat` is ~1/B regardless of bit width, so it measures the block scheme,
not the quality.

### Memory/quality Pareto frontier

| B/elem | config | eta_cum | band |
|---|---|---|---|
| 0.5625 | 4-bit B=64 | 2.543 | BROKEN |
| 0.6250 | 4-bit B=32 | 1.982 | BROKEN |
| 0.6875 | 5-bit B=64 | 0.828 | BROKEN |
| 0.7500 | 5-bit B=32 | 0.657 | MARGINAL |
| 0.8125 | 6-bit B=64 | 0.305 | MARGINAL |
| **0.8750** | **6-bit B=32** | **0.254** | **SAFE (cheapest safe point)** |
| 0.9375 | 7-bit B=64 | 0.131 | SAFE |
| 1.0000 | 7-bit B=32 | 0.110 | SAFE |
| 1.0625 | 8-bit B=64 | 0.063 | SAFE |
| 1.1250 | 8-bit B=32 | 0.053 | SAFE (what is built) |
| 1.2500 | 8-bit B=16 | 0.043 | SAFE |
| 2.0000 | fp16 | 0.001 | SAFE |

**Every B=16 configuration is dominated**: at the same B/elem, one more bit at B=32 is strictly
better (7-bit B=32 0.110 beats 6-bit B=16 0.200 at exactly 1.000 B/elem; 8-bit B=32 0.053 beats
7-bit B=16 0.090 at 1.125). This is the same "spend bytes on bits, not on scales" conclusion, now
as a frontier rather than an argument.

### What this changes

- **6-bit B=32 (0.875 B/elem) is the cheapest configuration that still samples cleanly**, a
  further 1.29x below the 1.125 that ships. Not built, and 6 bits does not pack into bytes
  (4 channels x 6 bit = 3 bytes), so the implementation cost is real for a 1.29x memory win with
  measurably worse error. Recorded, not recommended.
- **fp16 block scales remain the better move.** int8 B=32 with fp16 scales is 1.0625 B/elem at
  eta_cum ~0.053, which strictly dominates 8-bit B=64 (1.0625 at 0.063) -- same memory, less
  error, and no new datapath. That is the one item on the frontier that is both free and unbuilt.

---

## Addendum: MSE, and the calibration that makes eta_cum a predictor

`scripts/mse.py` -> `data/mse.json`, `plots/mse_vs_bits.png`.

### (A) Tensor level — absolute MSE of the a_hat storage error, B=32, per step (median over t>=5)

| bits | C=192 32² | C=384 16² | C=1536 4² | C=768 16² | C=576 32² | NMSE(step) | NMSE(cum) |
|---|---|---|---|---|---|---|---|
| 3 | 1.34e-02 | 1.09e-03 | 1.88e-02 | 1.55e-02 | 1.11e-02 | 1.04e-01 | 5.04e+01 |
| 4 | 2.60e-03 | 2.36e-04 | 3.27e-03 | 2.66e-03 | 1.99e-03 | 1.82e-02 | 3.85e+00 |
| 5 | 5.92e-04 | 5.61e-05 | 7.19e-04 | 5.79e-04 | 4.29e-04 | 4.00e-03 | 4.46e-01 |
| 6 | 1.40e-04 | 1.41e-05 | 1.67e-04 | 1.35e-04 | 9.59e-05 | 9.02e-04 | 6.34e-02 |
| 7 | 3.39e-05 | 3.03e-06 | 4.06e-05 | 3.28e-05 | 2.17e-05 | 2.02e-04 | 1.22e-02 |
| **8** | **8.33e-06** | **7.52e-07** | **1.00e-05** | **7.96e-06** | **5.25e-06** | **4.89e-05** | **2.87e-03** |
| 9 | 2.07e-06 | 1.84e-07 | 2.48e-06 | 1.96e-06 | 1.33e-06 | 1.24e-05 | 7.05e-04 |
| 10 | 5.09e-07 | 4.68e-08 | 6.12e-07 | 4.90e-07 | 3.33e-07 | 3.09e-06 | 1.72e-04 |
| 12 | 3.19e-08 | 2.99e-09 | 3.82e-08 | 3.05e-08 | 2.07e-08 | 1.92e-07 | 1.07e-05 |
| signal power | 1.71e-01 | 1.56e-02 | 1.80e-01 | 1.40e-01 | 1.08e-01 | | |

Absolute MSE spans **20x across layers at the same bit width** (7.5e-07 to 1.0e-05 at 8 bits),
purely because activation power differs 11x (1.6e-02 vs 1.8e-01). So absolute MSE is not
comparable across layers -- NMSE (= relL2²) is, and it collapses all five onto one line.
Each bit divides MSE by ~4 (amplitude by 2), as it must.

### (B) Image level — decoded samples vs the fp16-a_hat reference, same seed, pixel-aligned

| a_hat | image MSE | RMSE | PSNR | eta_cum |
|---|---|---|---|---|
| fp16 | 0 | 0 | inf | 0.001 |
| **i8 B=32 (REAL)** | **8.58e-04** | **0.0293** | **30.66 dB** | 0.053 |
| 7-bit | 1.84e-03 | 0.0429 | 27.35 | 0.110 |
| 6-bit | 3.71e-03 | 0.0609 | 24.30 | 0.254 |
| 5-bit | 9.29e-03 | 0.0964 | 20.32 | 0.657 |
| 4-bit | 2.65e-02 | 0.1628 | 15.77 | 1.982 |
| 3-bit | 1.17e-01 | 0.3424 | 9.31 | 7.330 |

### (C) The calibration — image MSE is PROPORTIONAL to eta_cum

Log-log fit over the six arms:

    image MSE = 0.0149 * eta_cum ^ 0.981          slope 0.981, i.e. linear

    simplified:  image MSE ~= 0.0152 * eta_cum    (coefficient range 0.0134-0.0167, +-12%)
    equivalently PSNR ~= 18.2 - 10*log10(eta_cum)  dB

Predicted vs measured MSE agrees to 0.91-1.11x across **two decades** of eta_cum. So a single
cheap open-loop kernel replay -- no sampling, no decoding -- predicts the decoded-image MSE to
about +-12%. That upgrades eta_cum from a ranking signal to a calibrated predictor, and it is what
the per-step metrics could never do (consumed moves 8.8e-03 -> 9.4e-03 across this entire range).

Thresholds in PSNR terms, from the earlier sample-anchored bands:

| eta_cum | PSNR | verdict |
|---|---|---|
| 0.15 | 26.4 dB | indistinguishable from fp16 |
| **0.30** | **23.4 dB** | **safe limit** |
| 0.70 | 19.7 dB | marginal limit |

The shipped int8 B=32 sits at eta_cum 0.053 / 30.7 dB, comfortably inside. 6-bit would be
24.3 dB (just inside), 5-bit 20.3 dB (outside).

---

## Addendum: testing the real kernels — and a correction to the image-MSE calibration

`scripts/validate_kernel.py` -> `data/validate_kernel.json`.

### (1) The real kernel and the PyTorch model agree to 1e-4 relative

Single step with a_hat = 0 (no trajectory divergence possible, so any difference is the GN
reduction order or the quantizers). Fraction of delta codes disagreeing:

| conv path | disagreement |
|---|---|
| int4 (`..._pack_nhwc`) | **0.00e+00 on all five layers** — bit-identical |
| int8 (`..._nhwc`) | 0 to 1.02e-04 |

int8 disagrees on at most 1 code in 10,000 and int4 not at all, because int4's DLIM=7 grid has
~18x fewer code boundaries for a tie to land on. The disagreement is identical across all five
a_hat variants, which is the expected sanity check: at t=T with a_hat=0 the storage scheme cannot
affect the delta codes.

Over the full 49-step trajectory, eta_cum from the real kernel vs from the model:

| a_hat | kernel | model | rel. diff |
|---|---|---|---|
| fp16 | 0.0015310330 | 0.0015310056 | 1.8e-05 |
| i8 B=16 | 0.0411779419 | 0.0411648557 | 3.2e-04 |
| i8 B=32 | 0.0507599212 | 0.0507607803 | 1.7e-05 |
| i8 B=64 | 0.0607382938 | 0.0607127614 | 4.2e-04 |
| i4 B=32 | 1.9247386456 | 1.9246665716 | 3.7e-05 |

Two to four orders below any effect being measured, so **every curve derived from the PyTorch
model applies to the shipped kernels**. The AhatI4 path built in Addendum 10 is included and
lands on the model's value.

### (2) CORRECTION — the decoded-image MSE has a reproducibility floor of 1.7e-03

Two runs of the SAME arm at the SAME seed differ by image MSE **1.705e-03**, which is LARGER than
the arm-vs-reference MSE of ~1.0e-03 the int8 arms produce. The sampling path is not
bit-deterministic run to run (autotuned conv algo selection and/or a non-deterministic reduction);
`torch.manual_seed` fixes the initial latent, not the kernels.

Consequences, all corrections to the previous addendum:

| arm | eta_cum | image MSE | /floor | resolvable? |
|---|---|---|---|---|
| i8 B=32 (REAL) | 0.053 | 8.58e-04 | 0.50x | **no — at the floor** |
| 7-bit | 0.110 | 1.84e-03 | 1.08x | **no** |
| 6-bit | 0.254 | 3.71e-03 | 2.18x | **no** |
| 5-bit | 0.657 | 9.29e-03 | 5.45x | yes |
| 4-bit | 1.982 | 2.65e-02 | 15.6x | yes |
| 3-bit | 7.330 | 1.17e-01 | 68.8x | yes |

- The three W8A8 real-kernel calibration points (i8 B=16/32/64, image MSE 9.1e-04 / 1.0e-03 /
  1.7e-03) are **entirely inside the noise floor**. Their apparent coefficients 0.0221 / 0.0198 /
  0.0283 are non-monotone and are noise; I should not have quoted them.
- Refitting on only the three points at least 3x above the floor:
  `image MSE = 0.0139 * eta_cum^1.054`, linearized coefficient **0.0145 (0.0134-0.0160)**.
  The linear law survives; the coefficient moves from 0.0152 to 0.0145 and the fit is now
  supported rather than floor-contaminated.
- **Image MSE cannot resolve anything below eta_cum ~ 0.118.** So every int8 configuration
  (eta_cum 0.041-0.061) is indistinguishable from fp16 a_hat AND from every other int8
  configuration in image space -- which is exactly what the sample grids show, and why the PSNR
  separations I reported for those rows (30.7 / 30.4 / 27.7 dB) were not real separations.

This makes eta_cum more useful, not less: it is deterministic and reproducible to 1e-4, so it
resolves configurations that the images provably cannot. Ranking by eta_cum, accept/reject by
samples only when eta_cum is above ~0.12.

---

## Addendum: full 2D grid — bit width x block size

`scripts/grid_2d.py` -> `data/grid_2d.json`, `scripts/plot_grid_2d.py` -> `plots/grid_2d.png`.
Real captured kernel-1 inputs, 5 layers (median), 49 steps. The model is validated against the
real kernels to 1e-4 relative, so the whole grid stands for kernel behaviour.

### eta_cum @ t=48

| bits | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 | B=128 | per-tensor |
|---|---|---|---|---|---|---|---|---|
| 3 | 1.6014 | 2.6821 | 3.9194 | 5.4744 | 7.3299 | 9.4367 | 11.2919 | 16.3151 |
| 4 | 0.4511 | 0.7617 | 1.0953 | 1.4983 | 1.9822 | 2.5425 | 3.2142 | 5.7719 |
| 5 | 0.1617 | 0.2656 | 0.3727 | 0.5060 | 0.6573 | 0.8284 | 0.9773 | 1.6889 |
| 6 | 0.0681 | 0.1095 | 0.1517 | 0.2000 | 0.2536 | 0.3050 | 0.3638 | 0.6044 |
| 7 | 0.0313 | 0.0498 | 0.0693 | 0.0898 | 0.1103 | 0.1314 | 0.1535 | 0.2411 |
| **8** | 0.0151 | 0.0242 | 0.0334 | 0.0432 | **0.0531** | 0.0625 | 0.0721 | 0.1104 |
| 9 | 0.0075 | 0.0121 | 0.0166 | 0.0214 | 0.0262 | 0.0311 | 0.0356 | 0.0541 |
| 10 | 0.0037 | 0.0060 | 0.0083 | 0.0107 | 0.0130 | 0.0154 | 0.0178 | 0.0269 |
| 12 | 0.0009 | 0.0015 | 0.0021 | 0.0027 | 0.0033 | 0.0039 | 0.0044 | 0.0066 |

Storage cost B/elem = bits/8 + 4/B, e.g. 8-bit B=32 = 1.125, 6-bit B=32 = 0.875, 4-bit B=2 = 2.5.

### Scaling law (least squares in log space, numpy)

| regime | fit | per +1 bit | per B doubling | residual (median / max) |
|---|---|---|---|---|
| 3-12 bit | 11.73 · B^0.274 · exp(-0.829·bits) | ÷2.29 | x1.209 | 20.5% / 57.5% |
| **7-12 bit** (where real configs live) | 3.921 · B^0.252 · exp(-0.702·bits) | **÷2.02** | **x1.191** | **9.5% / 21.3%** |
| 3-5 bit (coarse) | 48.7 · B^0.311 · exp(-1.188·bits) | ÷3.28 | x1.241 | 8.3% / 27.3% |

A single power law does not cover the whole range (20.5% residual) because the per-bit factor is
not constant: 3.28 in the coarse regime, converging to 2.02 (pure step-size scaling) at high bit
width. The extra factor at low bits is step-to-step correlation of the rounding -- a coarse grid
rounds the same way repeatedly, so errors add instead of cancelling. Split by regime the fit is
tight (9.5% / 8.3%).

**One bit is worth a 16x reduction in block size** in the asymptotic regime (45x in the coarse
one). That is the whole design conclusion in one number: bits dominate blocks completely.

Sanity check of the asymptotic fit at real configs: 8bit/B=32 predicts 0.0502 vs 0.0531 measured
(0.95x), 8bit/B=16 0.903x, 8bit/B=64 1.034x, 12bit/B=32 0.924x.

### Trends

- **vs bits** (panel a): straight lines on log-y across nine bit widths and seven block sizes,
  the seven B curves merely offset -- confirming the factorization.
- **vs block** (panel b): the same data transposed. Every line is nearly flat compared to (a):
  going B=2 -> B=128 (64x more memory-efficient scales) costs only 7.1x error, while going
  8-bit -> 3-bit (1.6x fewer bits) costs 138x.
- **per-tensor** (last column) is 2.1x worse than B=32 at 8 bits and 2.9x worse at 4 bits, so
  blockwise buys progressively more as bits shrink -- which is why blockwise a_hat matters at all,
  and why it matters more at low precision.

### Pareto frontier

The frontier is **always the highest bit width at the coarsest block** for a given budget:

| B/elem | config | eta_cum | band |
|---|---|---|---|
| 0.6250 | 4-bit B=32 | 1.9822 | BROKEN |
| 0.7500 | 5-bit B=32 | 0.6573 | MARGINAL |
| 0.8125 | 6-bit B=64 | 0.3050 | MARGINAL |
| **0.8750** | **6-bit B=32** | **0.2536** | **SAFE, cheapest** |
| 1.0000 | 7-bit B=32 | 0.1103 | SAFE |
| 1.0625 | 8-bit B=64 | 0.0625 | SAFE |
| **1.1250** | **8-bit B=32** | **0.0531** | **SAFE — what ships** |
| 1.2500 | 9-bit B=32 | 0.0262 | SAFE |

Every B<=16 configuration is dominated: at equal memory, spending the bytes on bits beats
spending them on finer scales. B=2 and B=4 only appear on the frontier at 12 bits, where the
error is already 50x below any threshold and the memory is worse than fp16.

---

## Addendum: is the (bits x block) grid shape-dependent? No — it factorizes out.

`scripts/shape_grid.py` -> `data/shape_grid.json`, `plots/shape_grid.png`.
One axis at a time on synthetic input with content held fixed, default N=4 C=384 H=W=16,
49 steps, 3 bit widths x 3 block sizes at each of 25 shape points.

### eta_cum is flat in every shape axis

Range across each swept axis, B=32:

| axis | swept | 4 bit | 6 bit | 8 bit |
|---|---|---|---|---|
| N (batch) | 1 -> 16 | 1.00x | 1.10x | 1.01x |
| C (channels) | 128 -> 1536 | 1.08x | 1.23x | 1.06x |
| H | 2 -> 64 | 1.01x | 1.11x | 1.01x |
| W | 2 -> 64 | 1.01x | 1.11x | 1.01x |

H and W give byte-identical numbers, as they must: NHWC makes the spatial extent one flat
dimension and a_hat quantization is per-pixel along C only.

### The bit and block effects are shape-invariant

| effect | N | C | H | W |
|---|---|---|---|---|
| eta(B=64)/eta(B=16) at 8 bit | 1.543-1.549 | 1.514-1.561 | 1.532-1.548 | 1.532-1.548 |
| eta(4bit)/eta(8bit) at B=32 | 79.8-80.5 | 79.4-82.3 | 79.8-80.4 | 79.8-80.4 |

Both are flat to within a couple of percent across a 16x range in batch, 12x in channels and 32x
in each spatial extent. **So the whole 2D grid measured at one shape transfers to every shape**,
which is what justifies quoting a single median grid and makes the design choice shape-free.

### The 6-bit wobble is seed noise, not shape

The 6-bit rows wander outside +-10% in the plot while 4-bit and 8-bit do not. Five different
seeds at the SAME default shape:

| config | seed spread | rel. std |
|---|---|---|
| 8 bit B=32 | 1.076x | 3.2% |
| 6 bit B=32 | 1.469x | **17.0%** |
| 6 bit B=64 | 1.694x | **23.4%** |
| 4 bit B=32 | 1.121x | 4.8% |

The seed noise at 6 bits exceeds the entire shape variation, so those lines carry no shape
information. The non-monotone pattern (low at 8 bit, peak at 6 bit, low again at 4 bit) is
consistent with the correlation finding earlier -- at 8 bits the rounding is essentially
uncorrelated step to step (endpoint is a well-averaged random walk, low variance), at 4 bits it is
strongly correlated (endpoint dominated by a systematic drift, again low RELATIVE variance), and
the mixed regime in between has the highest variance. That is a hypothesis consistent with the
measurements, not something these measurements prove.

**Practical consequence:** single-seed eta_cum carries 3-23% uncertainty depending on bit width,
worst in the 5-7 bit band. The grid's trends span 100x+ and are far above it, and 8-bit block
comparisons (1.23x per doubling vs 3.2% noise) are resolvable. But a single-seed comparison inside
the 6-bit region -- e.g. 6bit B=32 0.254 vs B=64 0.305, a 1.20x gap -- is NOT resolvable. The
real-data grid averages 5 layers, which cuts this by about sqrt(5).

### Real layers spread more than shape can explain

| config | real 5-layer spread | synthetic C axis | synthetic N/H/W |
|---|---|---|---|
| 4 bit B=32 | 1.66x | 1.08x | 1.02x |
| 6 bit B=32 | 1.36x | 1.23x | 1.19x |
| 8 bit B=32 | 1.52x | 1.06x | 1.01x |

At 4 and 8 bits the real spread (1.5-1.7x) is well beyond what shape contributes (<=1.08x) and
beyond seed noise (1.08-1.12x), so the residual is the layers' activation statistics -- content,
not geometry. With only five layers the correlation with any single statistic is not identifiable
(r(logC) swings from +0.18 at 4 bits to +0.60 at 8 bits), so no attribution is claimed beyond
"not shape".
