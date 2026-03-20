# Static vs Dynamic Baseline and MoDiff Benchmark Report

**Date**: 2026-03-19 14:57:31
**GPU**: NVIDIA A40
**Batch Size**: 32
**Timesteps**: 200
**Timed Samples per Mode**: 128
**Quality Samples per Mode**: 4

## Progress and debugging notes

- Built a dedicated benchmark to compare static vs dynamic quantization for both baseline and MoDiff variants.
- Generated fresh per-experiment calibration scales for all static modes instead of reusing shared legacy calibration files.
- Debug note: the current INT8 conv path does not populate built-in conv calibration scales on its own because it uses the GPU-only scale path; this benchmark compensates with an explicit hook-based INT8 conv-scale calibration pass so the static INT8 rows are real static runs.
- Kept decode/save outside the timed region so the numbers isolate denoising throughput rather than PNG I/O.
- Used the same initial latent noise (`x_T`) for all quality comparisons so visual differences come from the quantization mode rather than random sampling drift.
- Important implementation detail: the current MoDiff path in this repo keeps residual error compensation dynamically quantized by design. The `static` MoDiff rows therefore represent the repository's actual static-capable MoDiff configuration: static calibrated standard/first-step path + static linear scales, while residual compensation stays dynamic.
- Follow-up fairness check for the INT4 MoDiff pair: reran only `INT4 dynamic MoDiff` and `INT4 static MoDiff` with identical pre-generated timed latents, 3 timing repeats, and the same fused CUTLASS kernel path. That controlled rerun produced effectively identical throughput: **42.31 ± 0.05 s** dynamic vs **42.36 ± 0.06 s** static for 128 samples.

## Timing and memory results

| Mode | Time / sample (s) | Time / step (ms) | Allocated after setup (MB) | Peak memory (MB) | Loaded conv scales | Loaded linear scales |
| --- | --- | --- | --- | --- | --- | --- |
| INT8 dynamic baseline | 0.365 | 1.82 | 3868 | 5838 | 0 | 0 |
| INT8 static baseline | 0.326 | 1.63 | 6132 | 7929 | 70 | 0 |
| INT8 dynamic MoDiff | 0.377 | 1.88 | 3878 | 7655 | 0 | 0 |
| INT8 static MoDiff | 0.374 | 1.87 | 6625 | 10402 | 70 | 37 |
| INT4 dynamic baseline | 0.468 | 2.34 | 3653 | 5450 | 0 | 0 |
| INT4 static baseline | 0.323 | 1.61 | 3653 | 5450 | 70 | 0 |
| INT4 dynamic MoDiff | 0.347 | 1.73 | 3653 | 7430 | 0 | 0 |
| INT4 static MoDiff | 0.364 | 1.82 | 3653 | 7430 | 70 | 37 |

## INT8 static vs dynamic summary

- Baseline static vs dynamic speedup: **1.12x** (0.365s → 0.326s).
- MoDiff static vs dynamic speedup: **1.01x** (0.377s → 0.374s).
- Baseline peak-memory delta (static - dynamic): **+2092 MB**.
- MoDiff peak-memory delta (static - dynamic): **+2746 MB**.

### INT8 image-quality comparison against FP32

| Mode | MAE vs FP32 | Max abs diff | PSNR vs FP32 (dB) |
| --- | --- | --- | --- |
| INT8 dynamic baseline | 0.0081 | 0.3711 | 36.43 |
| INT8 static baseline | 0.0584 | 0.9422 | 20.15 |
| INT8 dynamic MoDiff | 0.0047 | 0.4007 | 40.46 |
| INT8 static MoDiff | 0.0048 | 0.4112 | 40.36 |

Quality figure: `integration/results/static_dynamic/quality/int8_quality_comparison.png`

### Visual inspection notes

_Pending manual visual review._

## INT4 static vs dynamic summary

- Baseline static vs dynamic speedup: **1.45x** (0.468s → 0.323s).
- MoDiff static vs dynamic speedup: **0.95x** (0.347s → 0.364s).
- Baseline peak-memory delta (static - dynamic): **+0 MB**.
- MoDiff peak-memory delta (static - dynamic): **+0 MB**.

### Controlled fairness rerun for INT4 MoDiff

- The original `0.347s` vs `0.364s` gap came from a single-pass run. That was good enough to smoke-test the experiment, but not good enough to call a real regression.
- I reran the pair with a stricter methodology: identical timed `x_T` batches for both modes, 3 repeats per mode, and unchanged batch size / timestep / sample count (`32 / 200 / 128`).
- Controlled rerun result: **INT4 dynamic MoDiff = 42.31 ± 0.05 s total = 0.3306 s/sample**, **INT4 static MoDiff = 42.36 ± 0.06 s total = 0.3309 s/sample**.
- That is a **0.11% delta**, which is within normal run-to-run noise here. In other words: under a fair comparison, INT4 static MoDiff is **effectively the same speed** as dynamic, not materially slower.
- Kernel audit confirmed the comparison is apples-to-apples on the hot path: both modes use the same fused CUTLASS MoDiff conv path on **140 / 140** converted conv layers. The static variant simply adds loaded calibration state on **70 conv** layers and **37 linear** layers.
- Why no real speedup? Because in this repository the MoDiff residual path remains dynamic by design. After the first step, both INT4 MoDiff variants execute the same fused residual kernels, so static calibration only changes the first-step / standard path and does not accelerate the dominant per-timestep hot loop.

### INT4 image-quality comparison against FP32

| Mode | MAE vs FP32 | Max abs diff | PSNR vs FP32 (dB) |
| --- | --- | --- | --- |
| INT4 dynamic baseline | 0.1812 | 1.0000 | 12.60 |
| INT4 static baseline | 0.1656 | 0.9902 | 13.40 |
| INT4 dynamic MoDiff | 0.0642 | 0.8818 | 20.19 |
| INT4 static MoDiff | 0.0641 | 0.8696 | 20.20 |

Quality figure: `integration/results/static_dynamic/quality/int4_quality_comparison.png`

### Visual inspection notes

_Pending manual visual review._

## Initial conclusions

- Static calibration should reduce per-sample time whenever the benchmark can reuse loaded activation scales instead of recomputing them at runtime.
- The static-vs-dynamic gain is expected to be strongest in the baseline path because that path can fully replace repeated activation-scale discovery with cached scales.
- MoDiff quality should remain closer to FP32 than the baseline variants because temporal residual compensation is still active.