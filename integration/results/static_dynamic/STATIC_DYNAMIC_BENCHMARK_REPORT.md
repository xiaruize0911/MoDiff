# Static vs Dynamic Baseline and MoDiff Benchmark Report

**Date**: 2026-03-20 06:18:11
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

## Timing and memory results

| Mode | Time / sample (s) | Time / step (ms) | Allocated after setup (MB) | Peak memory (MB) | Loaded conv scales | Loaded linear scales |
| --- | --- | --- | --- | --- | --- | --- |
| INT8 dynamic baseline | 0.343 | 1.72 | 3871 | 5841 | 0 | 0 |
| INT8 static baseline | 0.312 | 1.56 | 6135 | 7932 | 70 | 0 |
| INT8 dynamic MoDiff | 0.355 | 1.78 | 3881 | 7658 | 0 | 0 |
| INT8 static MoDiff | 0.327 | 1.64 | 6627 | 10401 | 70 | 37 |
| INT4 dynamic baseline | 0.411 | 2.06 | 3656 | 5453 | 0 | 0 |
| INT4 static baseline | 0.287 | 1.44 | 3656 | 5453 | 70 | 0 |
| INT4 dynamic MoDiff |0.331 | 1.65 | 3656 | 7433 | 0 | 0 |
| INT4 static MoDiff | 0.302 | 1.51 | 3656 | 6728 | 70 | 37 |

## INT8 static vs dynamic summary

- Baseline static vs dynamic speedup: **1.10x** (0.343s → 0.312s).
- MoDiff static vs dynamic speedup: **1.09x** (0.355s → 0.327s).
- Baseline peak-memory delta (static - dynamic): **+2092 MB**.
- MoDiff peak-memory delta (static - dynamic): **+2743 MB**.
- Baseline timing repeat std-dev: **0.05s** dynamic vs **0.15s** static.
- MoDiff timing repeat std-dev: **0.04s** dynamic vs **0.10s** static.
- Timed runs reuse the same pre-generated initial latents (`x_T`) across compared modes so static vs dynamic timing is measured on identical denoising workloads.

### INT8 image-quality comparison against FP32

| Mode | MAE vs FP32 | Max abs diff | PSNR vs FP32 (dB) |
| --- | --- | --- | --- | 
| INT8 dynamic baseline | 0.0081 | 0.3687 | 36.43 |
| INT8 static baseline | 0.0713 | 0.9614 | 19.34 |
| INT8 dynamic MoDiff | 0.0047 | 0.4024 | 40.47 |
| INT8 static MoDiff | 0.0098 | 0.6645 | 34.18 |

Quality figure: `integration/results/static_dynamic/quality/int8_quality_comparison.png`

### Visual inspection notes

_Pending manual visual review._

## INT4 static vs dynamic summary

- Baseline static vs dynamic speedup: **1.43x** (0.411s → 0.287s).
- MoDiff static vs dynamic speedup: **1.10x** (0.331s → 0.302s).
- Baseline peak-memory delta (static - dynamic): **+0 MB**.
- MoDiff peak-memory delta (static - dynamic): **-705 MB**.
- Baseline timing repeat std-dev: **0.13s** dynamic vs **0.10s** static.
- MoDiff timing repeat std-dev: **0.10s** dynamic vs **0.02s** static.
- Timed runs reuse the same pre-generated initial latents (`x_T`) across compared modes so static vs dynamic timing is measured on identical denoising workloads.

### INT4 image-quality comparison against FP32

| Mode | MAE vs FP32 | Max abs diff | PSNR vs FP32 (dB) |
| --- | --- | --- | --- |
| INT4 dynamic baseline | 0.1812 | 1.0000 | 12.60 |
| INT4 static baseline | 0.2000 | 0.9917 | 12.07 |
| INT4 dynamic MoDiff | 0.0642 | 0.8813 | 20.19 |
| INT4 static MoDiff | 0.0711 | 0.8927 | 19.37 |

Quality figure: `integration/results/static_dynamic/quality/int4_quality_comparison.png`

### Visual inspection notes

_Pending manual visual review._

## Initial conclusions

- Static calibration should reduce per-sample time whenever the benchmark can reuse loaded activation scales instead of recomputing them at runtime.
- The static-vs-dynamic gain is expected to be strongest in the baseline path because that path can fully replace repeated activation-scale discovery with cached scales.
- MoDiff quality should remain closer to FP32 than the baseline variants because temporal residual compensation is still active.