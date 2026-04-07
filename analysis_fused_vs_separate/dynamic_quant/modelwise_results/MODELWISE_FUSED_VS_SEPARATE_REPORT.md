# Model-wise fused-vs-separate MoDiff benchmark (dynamic quantization)

**Date**: 2026-04-07 03:13:58
**GPU**: NVIDIA A40
**Config**: `/workspace/MoDiff/configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml`
**Checkpoint**: `/workspace/MoDiff/models/ldm/lsun_churches256/model.ckpt`
**Batch Size**: 32
**Timesteps**: 50
**Quant Mode**: dynamic

Timing notes:
- Each number is the mean full-sampling-call latency over 3 timed repeats × 2 iterations.
- Timed region covers the full DDIM denoising call and excludes decode / image save.
- MoDiff state is reset before every timed call, outside the timed region.
- Sampler stdout/stderr and progress bars are suppressed during warmup and timed calls so console I/O does not pollute the timing.
- Peak GPU memory is reset **after warmup** and measured only over the timed region, so one-off setup allocations do not distort the comparison.
- The benchmark intentionally leaves the global buffer pool disabled because the current pool pre-allocates oversized residual buffers for fused layers and inflates memory without benefiting these kernels.
- Fused layers also drop their calibration-only FP32 `_orig_weight` clones after setup, because those buffers are only needed for later SmoothQuant calibration and otherwise exaggerate inference memory.
- The same pre-generated latent tensors are reused across compared modes so fused and separate paths denoise identical workloads.
- Dynamic mode intentionally disables static activation scales so each call recomputes its own per-tensor activation scale.

## Timing summary

| Mode | Mean call (ms) | Std over repeat means (ms) | Time/sample (ms) | Time/step (ms) | Ready memory (MB) | Timed peak (MB) | Peak Δ (MB) | Loaded scales | Scale status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 fused MoDiff | 2882.56 | 7.32 | 90.08 | 1.8016 | 4152 | 4933 | 781 | 0 | disabled-dynamic |
| INT8 separate MoDiff | 4304.48 | 78.49 | 134.52 | 2.6903 | 3454 | 4069 | 615 | 0 | disabled-dynamic |
| INT4 fused MoDiff | 2989.09 | 190.24 | 93.41 | 1.8682 | 3913 | 4380 | 467 | 0 | disabled-dynamic |
| INT4 separate MoDiff | 3777.44 | 228.02 | 118.05 | 2.3609 | 3213 | 3872 | 660 | 0 | disabled-dynamic |

## Fusion speedup

- **INT8 fused vs separate**: 1.49x faster (4304.48 ms → 2882.56 ms).
- **INT4 fused vs separate**: 1.26x faster (3777.44 ms → 2989.09 ms).

## Calibration notes

- Dynamic mode ignores static calibration files by design; both fused and separate paths recompute activation scales online.

## Memory notes

- Reported peak memory is the **timed-region peak after warmup**, not the whole-process peak since process start.
- The earlier inflated fused-memory readings were caused by two setup-time artifacts: benchmark-side buffer-pool preallocation and retained calibration-only `_orig_weight` clones inside fused modules.
- The rebuilt benchmark disables the former and releases the latter before timing.
- After those artifacts are removed, the remaining INT8 fused post-warmup gap is mostly backend/workspace retention rather than model-owned tensors: Python-visible extra fused state is only the persistent `_residual_buf` (~44 MB on this UNet), while roughly 0.5 GB remains allocated until the fused INT8 model is destroyed. INT4 does not show the same lingering footprint.
- INT8 fused MoDiff: scale source **dynamic-disabled**, status **disabled-dynamic**, applied scales **0**, path `n/a`.
- INT8 separate MoDiff: scale source **dynamic-disabled**, status **disabled-dynamic**, applied scales **0**, path `n/a`.
- INT4 fused MoDiff: scale source **dynamic-disabled**, status **disabled-dynamic**, applied scales **0**, path `n/a`.
- INT4 separate MoDiff: scale source **dynamic-disabled**, status **disabled-dynamic**, applied scales **0**, path `n/a`.
