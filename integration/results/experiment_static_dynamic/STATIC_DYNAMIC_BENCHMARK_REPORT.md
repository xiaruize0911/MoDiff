# Static vs Dynamic Baseline and MoDiff Benchmark Report

**Date**: 2026-03-22 17:04:09
**GPU**: NVIDIA A40
**Batch Size**: 32
**Timesteps**: 200
**Timed Samples per Mode**: 64
**Quality Samples per Mode**: 4

## Progress and debugging notes

- Built a dedicated benchmark to compare static vs dynamic quantization for both baseline and MoDiff variants.
- Generated fresh per-experiment calibration scales for all static modes instead of reusing shared legacy calibration files.
- Debug note: the current INT8 conv path does not populate built-in conv calibration scales on its own because it uses the GPU-only scale path; this benchmark compensates with an explicit hook-based INT8 conv-scale calibration pass so the static INT8 rows are real static runs.
- Kept decode/save outside the timed region so the numbers isolate denoising throughput rather than PNG I/O.
- Used the same initial latent noise (`x_T`) for all quality comparisons so visual differences come from the quantization mode rather than random sampling drift.
- Important implementation detail: the current MoDiff path in this repo keeps residual error compensation dynamically quantized by design. The `static` MoDiff rows therefore represent the repository's actual static-capable MoDiff configuration: static calibrated standard/first-step path + static linear scales, while residual compensation stays dynamic.

## Timing and memory results

| Mode | Time / sample (s) | Time / step (ms) | Allocated after setup (MB) | Peak memory (MB) | Loaded conv scales | Loaded linear scales |
| --- | --- | --- | --- | --- | --- | --- |
| INT8 dynamic baseline | 0.344 | 1.72 | 3870 | 5840 | 0 | 0 |
| INT8 static baseline | 0.312 | 1.56 | 6134 | 7931 | 70 | 0 |
| INT8 dynamic MoDiff | 0.355 | 1.78 | 3880 | 7657 | 0 | 0 |
| INT8 static MoDiff | 0.328 | 1.64 | 6626 | 10400 | 70 | 37 |
| INT4 dynamic baseline | 0.406 | 2.03 | 3655 | 5452 | 0 | 0 |
| INT4 static baseline | 0.287 | 1.43 | 3655 | 5452 | 70 | 0 |
| INT4 dynamic MoDiff | 0.330 | 1.65 | 3655 | 7432 | 0 | 0 |
| INT4 static MoDiff | 0.302 | 1.51 | 3655 | 6727 | 70 | 37 |

## INT8 static vs dynamic summary

- Baseline static vs dynamic speedup: **1.10x** (0.344s → 0.312s).
- MoDiff static vs dynamic speedup: **1.08x** (0.355s → 0.328s).
- Baseline peak-memory delta (static - dynamic): **+2092 MB**.
- MoDiff peak-memory delta (static - dynamic): **+2743 MB**.
- Baseline timing repeat std-dev: **0.01s** dynamic vs **0.03s** static.
- MoDiff timing repeat std-dev: **0.01s** dynamic vs **0.05s** static.
- Timed runs reuse the same pre-generated initial latents (`x_T`) across compared modes so static vs dynamic timing is measured on identical denoising workloads.

### INT8 image-quality comparison against FP32

| Mode | MAE vs FP32 | Max abs diff | PSNR vs FP32 (dB) |
| --- | --- | --- | --- |
| INT8 dynamic baseline | 0.0081 | 0.3712 | 36.43 |
| INT8 static baseline | 0.0462 | 0.8962 | 22.21 |
| INT8 dynamic MoDiff | 0.0047 | 0.4007 | 40.46 |
| INT8 static MoDiff | 0.0134 | 0.6721 | 31.66 |

Quality figure: `integration/results/experiment_static_dynamic/quality/int8_quality_comparison.png`

### Visual inspection notes

_Pending manual visual review._

## INT4 static vs dynamic summary

- Baseline static vs dynamic speedup: **1.42x** (0.406s → 0.287s).
- MoDiff static vs dynamic speedup: **1.09x** (0.330s → 0.302s).
- Baseline peak-memory delta (static - dynamic): **+0 MB**.
- MoDiff peak-memory delta (static - dynamic): **-705 MB**.
- Baseline timing repeat std-dev: **0.03s** dynamic vs **0.03s** static.
- MoDiff timing repeat std-dev: **0.01s** dynamic vs **0.02s** static.
- Timed runs reuse the same pre-generated initial latents (`x_T`) across compared modes so static vs dynamic timing is measured on identical denoising workloads.

### INT4 image-quality comparison against FP32

| Mode | MAE vs FP32 | Max abs diff | PSNR vs FP32 (dB) |
| --- | --- | --- | --- |
| INT4 dynamic baseline | 0.1812 | 1.0000 | 12.60 |
| INT4 static baseline | 0.1376 | 0.9917 | 14.71 |
| INT4 dynamic MoDiff | 0.0642 | 0.8818 | 20.19 |
| INT4 static MoDiff | 0.0807 | 0.9955 | 18.43 |

Quality figure: `integration/results/experiment_static_dynamic/quality/int4_quality_comparison.png`

### Visual inspection notes

_Pending manual visual review._

## Initial conclusions

- Static calibration should reduce per-sample time whenever the benchmark can reuse loaded activation scales instead of recomputing them at runtime.
- The static-vs-dynamic gain is expected to be strongest in the baseline path because that path can fully replace repeated activation-scale discovery with cached scales.
- MoDiff quality should remain closer to FP32 than the baseline variants because temporal residual compensation is still active.