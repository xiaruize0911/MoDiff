# Fused vs Separate MoDiff Benchmark Report (static quantization)

**Date**: 2026-04-07 03:18:17
**GPU**: NVIDIA A40
**Torch**: 2.6.0+cu124 (CUDA 12.4)
**Conv backend**: cudnn
**Quant Mode**: static

## Sources

- Layerwise source dir: `/workspace/MoDiff/analysis_fused_vs_separate/static_quant/layerwise_results`
- Modelwise source dir: `/workspace/MoDiff/analysis_fused_vs_separate/static_quant/modelwise_results`

## Benchmark settings

### Layerwise

- batch size: **32**
- warmup iterations: **5**
- timed iterations per repeat: **20**
- timed repeats: **3**
- unique Conv2d shapes enumerated: **33**
- supported shapes benchmarked: **20**
- excluded shapes reported: **13**

### Whole model

- batch size: **32**
- DDIM steps: **50**
- warmup runs: **1**
- timed calls per mode: **6**
- quantization mode: **static**
- timed region covers full DDIM denoising and excludes decode/save
- sampler console output is suppressed during timing

## Headline results

- **Layerwise INT8 weighted fusion speedup**: **2.24x** (36.632 ms → 16.352 ms)
- **Layerwise INT4 weighted fusion speedup**: **2.49x** (31.061 ms → 12.451 ms)
- **Whole-model INT8 fusion speedup**: **1.50x** (4041.66 ms → 2701.89 ms)
- **Whole-model INT4 fusion speedup**: **1.45x** (3604.53 ms → 2493.33 ms)

## Layerwise observations

- Across the **20 supported shapes**, fused was never slower than separate for either INT8 or INT4.
- The weakest supported INT8 shape still showed **1.17x** speedup on `2x2 | 1536->768 | k3x3 s1x1 p1x1`.
- The weakest supported INT4 shape still showed **1.48x** speedup on `4x4 | 1152->768 | k3x3 s1x1 p1x1`.
- Excluded shapes are not benchmark failures; they are layers the repository's quantized conversion path does not replace in practice (e.g. skip-path 1x1 convs, final output conv, or very small input-channel cases).

## Whole-model observations

- Fused INT8 reduced mean full-model call time from **4041.66 ms** to **2701.89 ms** at batch size 32 and 50 DDIM steps.
- Fused INT4 reduced mean full-model call time from **3604.53 ms** to **2493.33 ms** under the same workload.
- Static quantization mode reuses cached per-layer activation scales; when a supplied scale file had no matching conv keys, fresh scales were auto-calibrated from representative DDIM sampling calls.
- Ready memory before timed sampling is INT8 fused **4152 MB** vs separate **3453 MB**, and INT4 fused **3209 MB** vs separate **3214 MB**.
- Timed-region peak memory is INT8 fused **4932 MB** vs separate **4065 MB**, and INT4 fused **3680 MB** vs separate **3871 MB**.
- The earlier inflated fused-memory readings were caused by setup-time artifacts: benchmark-side buffer-pool preallocation and calibration-only FP32 `_orig_weight` clones inside fused modules. The rebuilt benchmark disables the former, releases the latter, and measures peak memory only after warmup.
- After those fixes, the remaining INT8 fused post-warmup gap appears to be mostly backend/workspace retention rather than another reporting bug: Python-visible extra fused state is only the persistent `_residual_buf` (~44 MB on this UNet), while roughly 0.5 GB stays allocated until the fused INT8 model is destroyed. INT4 does not show the same lingering footprint.
- INT8 scale source: **generated-calibration** with **70** applied scales (status: **generated-calibration**, path: `/workspace/MoDiff/analysis_fused_vs_separate/static_quant/modelwise_results/int8_generated_static_scales.pt`).
- INT4 scale source: **loaded-file** with **70** applied scales (status: **loaded-file**, path: `/workspace/MoDiff/integration/calibration/int4_calibration.pt`).

## Figures

- `plot_01_layerwise_weighted_totals.png`
- `plot_02_layerwise_weighted_breakdown.png`
- `plot_03_layerwise_top_shapes_speedup.png`
- `plot_04_modelwise_call_times.png`
- `plot_05_modelwise_speedup_memory.png`

## Output tables

- `table_layerwise_weighted_summary.csv`
- `table_layerwise_supported_shapes.csv`
- `table_layerwise_excluded_shapes.csv`
- `table_modelwise_summary.csv`
- `table_modelwise_speedup.csv`
- `table_overall_summary.csv`

## Key takeaway

Kernel fusion clearly matters in this codebase. The rebuilt per-layer hot-path benchmark shows **2.24x** weighted speedup for INT8 and **2.49x** for INT4, and the effect survives at the whole-model level as **1.50x** for INT8 and **1.45x** for INT4 under the rebuilt batch-32 / 50-step workload.
