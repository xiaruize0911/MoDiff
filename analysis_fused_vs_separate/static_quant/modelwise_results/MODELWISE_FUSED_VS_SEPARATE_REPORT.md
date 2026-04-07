# Model-wise fused-vs-separate MoDiff benchmark (static quantization)

**Date**: 2026-04-07 03:18:11
**GPU**: NVIDIA A40
**Config**: `/workspace/MoDiff/configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml`
**Checkpoint**: `/workspace/MoDiff/models/ldm/lsun_churches256/model.ckpt`
**Batch Size**: 32
**Timesteps**: 50
**Quant Mode**: static

Timing notes:
- Each number is the mean full-sampling-call latency over 3 timed repeats × 2 iterations.
- Timed region covers the full DDIM denoising call and excludes decode / image save.
- MoDiff state is reset before every timed call, outside the timed region.
- Sampler stdout/stderr and progress bars are suppressed during warmup and timed calls so console I/O does not pollute the timing.
- Peak GPU memory is reset **after warmup** and measured only over the timed region, so one-off setup allocations do not distort the comparison.
- The benchmark intentionally leaves the global buffer pool disabled because the current pool pre-allocates oversized residual buffers for fused layers and inflates memory without benefiting these kernels.
- Fused layers also drop their calibration-only FP32 `_orig_weight` clones after setup, because those buffers are only needed for later SmoothQuant calibration and otherwise exaggerate inference memory.
- The same pre-generated latent tensors are reused across compared modes so fused and separate paths denoise identical workloads.
- Static mode applies one fixed activation scale per quantized layer; when a supplied scale file has no matching conv keys, the benchmark auto-calibrates fresh scales from representative DDIM sampling calls.

## Timing summary

| Mode | Mean call (ms) | Std over repeat means (ms) | Time/sample (ms) | Time/step (ms) | Ready memory (MB) | Timed peak (MB) | Peak Δ (MB) | Loaded scales | Scale status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 fused MoDiff | 2701.89 | 32.27 | 84.43 | 1.6887 | 4152 | 4932 | 780 | 70 | generated-calibration |
| INT8 separate MoDiff | 4041.66 | 30.71 | 126.30 | 2.5260 | 3453 | 4065 | 613 | 70 | generated-calibration |
| INT4 fused MoDiff | 2493.33 | 92.28 | 77.92 | 1.5583 | 3209 | 3680 | 471 | 70 | loaded-file |
| INT4 separate MoDiff | 3604.53 | 9.56 | 112.64 | 2.2528 | 3214 | 3871 | 657 | 70 | loaded-file |

## Fusion speedup

- **INT8 fused vs separate**: 1.50x faster (4041.66 ms → 2701.89 ms).
- **INT4 fused vs separate**: 1.45x faster (3604.53 ms → 2493.33 ms).

## Calibration notes

- Static activation scales are shared between the fused and separate variants of each precision so the comparison isolates kernel fusion rather than calibration drift.

## Memory notes

- Reported peak memory is the **timed-region peak after warmup**, not the whole-process peak since process start.
- The earlier inflated fused-memory readings were caused by two setup-time artifacts: benchmark-side buffer-pool preallocation and retained calibration-only `_orig_weight` clones inside fused modules.
- The rebuilt benchmark disables the former and releases the latter before timing.
- After those artifacts are removed, the remaining INT8 fused post-warmup gap is mostly backend/workspace retention rather than model-owned tensors: Python-visible extra fused state is only the persistent `_residual_buf` (~44 MB on this UNet), while roughly 0.5 GB remains allocated until the fused INT8 model is destroyed. INT4 does not show the same lingering footprint.
- INT8 fused MoDiff: scale source **generated-calibration**, status **generated-calibration**, applied scales **70**, path `/workspace/MoDiff/analysis_fused_vs_separate/static_quant/modelwise_results/int8_generated_static_scales.pt`.
- INT8 separate MoDiff: scale source **generated-calibration**, status **generated-calibration**, applied scales **70**, path `/workspace/MoDiff/analysis_fused_vs_separate/static_quant/modelwise_results/int8_generated_static_scales.pt`.
- INT4 fused MoDiff: scale source **loaded-file**, status **loaded-file**, applied scales **70**, path `/workspace/MoDiff/integration/calibration/int4_calibration.pt`.
- INT4 separate MoDiff: scale source **loaded-file**, status **loaded-file**, applied scales **70**, path `/workspace/MoDiff/integration/calibration/int4_calibration.pt`.
