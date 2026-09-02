# `MODIFF_DELTA_MODE=dynamic` does not beat calibrated static, and is not free

LSUN-churches LDM-KL-8, W8A8, A40, batch 128, 50 DDIM. `MODIFF_LINEAR=0 MODIFF_AHAT_BITS=16
MODIFF_IMODE=0 MODIFF_CACHE_SKIP_K=1 MODIFF_REPLAY_K=1 MODIFF_AHAT_REFRESH=0`,
`int8_calibration_realckpt.pt`. Timing median of 2 after 1 warmup; quality n=6, seed 20260805,
latent relL2 vs the fp16 arm. `scripts/delta_dynamic.py`, `data/delta_dynamic.json`.

## Result

| arm | ms/step | vs fp16 | vs shipped | relL2 |
|---|---:|---:|---:|---:|
| fp16 | 101.924 | 1.000x | 0.801x | — |
| **W8A8 MoDiff, delta static (ships)** | **81.652** | 1.248x | 1.000x | **0.0999** |
| W8A8 MoDiff, delta dynamic | 83.929 | 1.214x | **0.973x** | 0.1091 |
| W8A8 MoDiff, dynamic + `a_hat` B=32 | 82.030 | 1.243x | 0.995x | 0.0836 |

**Dynamic is 2.8% slower and its quality is a wash.** It was predicted to be ~3.5x better at zero
cost. Both halves of that prediction are wrong:

*Not free.* Dynamic derives the delta step size per call, so it pays a runtime absmax the static
table does not. 2.28 ms/step.

*Not better.* 0.0999 -> 0.1091. This tree does not reproduce relL2 to better than about +-0.03
and this run has no repeats, so the honest statement is "indistinguishable", not "worse". Either
way it is not 3.5x.

The flag was live: `_load_delta_table` prints for 70 layers in the static arm and correctly
prints nothing in the dynamic arm (it returns 0 under dynamic by design), and the 2.8% time
delta is the runtime absmax. This is not a no-op measurement.

## Why this does not contradict the 2026-08-04 result

`int8_optimized.py:166-180` records static 0.1878 -> dynamic 0.0393 and calls dynamic
"decisively" better. That measurement **predates the delta table being loadable at all**: until
2026-08-12 `apply_int8_delta_scales` had zero call sites, so "static" there meant an
*uncalibrated* grid. Today's static loads `int8_delta_calibration.pt` for 70 layers and lands at
0.0999. Calibrated static is a different arm from the one dynamic beat, and it ties.

The comment block should be read as superseded, not as disagreeing with this run.

## The consequence for blockwise, which is the reason this was run

`docs/act_blockwise_2026-09-01` prices granularities by **per-layer quantizer error**,
`||dequant(Q(v)) - v|| / ||v||`, and reports static 0.1838 vs dynamic 0.0451 -- a 3.5x gap. That
gap is real as a property of the quantizer. It bought **zero** end-to-end latent relL2.

So on this evidence that metric does not predict E2E quality on this path, and **the entire
accuracy case for the conv blockwise mainloop rests on the same metric**: B=64 at 0.0102 is
"16x better than static" in exactly the units that just failed to transfer. There is no
remaining reason to expect the blockwise conv's 15-40% time cost to buy visible quality.

Two readings, not yet separated:

1. The conv-input quantizer is not the dominant error term at W8A8. relL2 ~0.10 may be set by W8
   weights, quantized attention, or `a_hat`, in which case a 3.5x (or 16x) better conv-input
   quantizer moves the total by little regardless of how good it is.
2. relL2-vs-fp16 saturates and cannot resolve the difference at all.

**The experiment that separates them** is an error-budget ablation: hold everything else exact
and quantize only the conv inputs, then sweep granularity. If that sweep is also flat, blockwise
is dead on quality grounds and the mainloop is a negative result worth writing up as one. If it
is steep, the conv-input quantizer is being masked by a larger term and blockwise only pays off
after that term is fixed. Either outcome decides the conv work; the current numbers decide
nothing, which is the actual state.

## Side results

`a_hat` B=32 on top of dynamic recovers 1.90 ms/step (2.3%), consistent with the 1.4-1.8%
in `docs/ahat_blockwise_2026-09-01`. It also has the lowest relL2 of the three int8 arms
(0.0836), though again inside noise. Static + B=32 was not measured and is the config most
likely to be the best available today.

## An unresolved harness discrepancy

This harness reads **81.652 ms/step** for the production W8A8 MoDiff path.
`docs/pipeline_profile_2026-08-31` reads **72.20 ms/step** for what it also calls the production
path, with a 37.81 ms/step GEMM/conv bucket. 13% apart. This env block was copied from
`docs/ahat_blockwise_2026-09-01/scripts/block32_real.py`, which reads 81.30-81.74 for its fp16
`a_hat` arm -- so 81.65 is consistent with that harness and the disagreement predates this work.
The fp16 arms agree (101.92 here vs 102.23 there), so the gap is in the int8 path only.

This matters beyond bookkeeping: any projection of the blockwise conv's E2E cost that scales the
37.81 ms conv bucket is using a denominator from the *other* harness. The conv bucket has not
been profiled at 81.65, so those projections carry that error.
