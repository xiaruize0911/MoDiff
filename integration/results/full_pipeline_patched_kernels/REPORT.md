# Whole Pipeline Patched-Kernel Benchmark

Date: 2026-06-30
GPU: NVIDIA A40
Model: `configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml`
Checkpoint: `models/ldm/lsun_churches256/model.ckpt`
Run shape: 200 DDIM steps, 168 samples, batch size 42
Calibration: `integration/calibration/int8_calibration.pt`

## What Was Tested

This run benchmarks the full LDM sampling pipeline after adding and then refining the AWQ-style kernel paths:

- Linear layers: verified square attention projections dispatch to AWQ W8A8 fused quantized GEMM for sizes 512, 2048, and 4096.
- 1x1 Conv2d layers: an AWQ W8A8 GEMM path exists for eligible pointwise Conv2d, but this LSUN LDM UNet has its Conv2d pointwise layers under `skip_connection`, which the converter skips.
- Conv1d-1x1 attention projections: the first full AWQ run converted all 42 projections; the optimized run uses a shape gate and converts only projections with `in_channels >= 1024`.
- Other Conv2d layers: still use the local CUTLASS INT8 Conv2d path, including existing fused quantize and output-buffer reuse paths.

Important comparison note: for this model, `skip_pointwise=False` does not add Conv2d pointwise layers because those layers are skipped by the `skip_connection` guard. The meaningful difference in the first AWQ-full run was the blanket Conv1d-1x1 AWQ conversion.

## Commands

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode fp16 \
  --steps 200 \
  --num_samples 168 \
  --batch_size 42 \
  --output_dir integration/results/full_pipeline_patched_kernels/fp16_s200_n168_b42
```

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_baseline \
  --steps 200 \
  --num_samples 168 \
  --batch_size 42 \
  --skip_calibration \
  --calibration integration/calibration/int8_calibration.pt \
  --output_dir integration/results/full_pipeline_patched_kernels/int8_baseline_s200_n168_b42
```

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_awq_full_baseline \
  --steps 200 \
  --num_samples 168 \
  --batch_size 42 \
  --skip_calibration \
  --calibration integration/calibration/int8_calibration.pt \
  --output_dir integration/results/full_pipeline_patched_kernels/int8_awq_full_s200_n168_b42
```

After adding the Conv1d profitability gate and AWQ preallocation, the optimized run was:

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_awq_full_baseline \
  --steps 200 \
  --num_samples 168 \
  --batch_size 42 \
  --skip_calibration \
  --calibration integration/calibration/int8_calibration.pt \
  --output_dir integration/results/full_pipeline_patched_kernels/int8_awq_full_selective_s200_n168_b42
```

## Results

| Mode | Kernel coverage | Total time | Time/sample | Time/step | Vs FP16 | Vs INT8 baseline |
|---|---:|---:|---:|---:|---:|---:|
| `fp16` | FP16 PyTorch/cuDNN | 50.77 s | 0.302 s | 1.511 ms | 1.000x | 0.963x |
| `int8_baseline` | local INT8 Conv2d + FP16 linear backend, pointwise Conv2d skipped | 48.91 s | 0.291 s | 1.456 ms | 1.038x | 1.000x |
| `int8_awq_full_baseline` original | local INT8 Conv2d + AWQ linear dispatch + all 42 Conv1d-1x1 projections converted to AWQ | 52.11 s | 0.310 s | 1.551 ms | 0.974x | 0.938x |
| `int8_awq_full_baseline` selective | local INT8 Conv2d + AWQ linear dispatch + profitable Conv1d AWQ gate | 50.02 s | 0.298 s | 1.489 ms | 1.015x | 0.978x |

The selective AWQ-kernel path recovers most of the regression: it is 4.18% faster than the original blanket AWQ-full run and 1.47% faster than FP16. It is still 2.28% slower than the current INT8 baseline.

## Layer/Kernel Split

The measured `int8_awq_full_baseline` run converted:

| Component | Count | Kernel path |
|---|---:|---|
| `OptimizedInt8Conv2d` | 140 | local CUTLASS INT8 Conv2d for this model; Conv2d pointwise skip connections are not converted |
| `OptimizedInt8Linear` | 37 | AWQ W8A8 GEMM for verified square attention projections, local fallback otherwise |
| Conv1d 1x1 projections, original run | 42/42 converted | blanket AWQ-backed projection path |
| Conv1d 1x1 projections, selective run | 0/42 converted | FP16 kept because all actual LSUN LDM shapes are below the profitable gate |

Tracked quantized memory after warmup for INT8 modes was 533.3 MiB:

| Bucket | MiB |
|---|---:|
| Conv quantized weights | 450.6 |
| Linear FP16 weights retained | 54.3 |
| Linear quantized weights | 27.1 |
| Conv scale state | 0.9 |
| Conv bias | 0.3 |
| Linear bias | 0.1 |

## Interpretation

The kernel-level changes are real, but only some of them help this complete pipeline:

- Linear microbenchmarks previously matched AWQ direct performance for the verified square projection shapes.
- 1x1 Conv2d microbenchmarks improved by about 1.2x to 2.0x versus the existing CUTLASS Conv2d path.
- The LSUN LDM Conv2d pointwise layers are all `skip_connection` layers, so the 1x1 Conv2d AWQ path is not exercised by this whole-pipeline run.
- Blanket AWQ Conv1d conversion was the real regression: actual model shapes are 192, 384, and 768 input channels, and microbenchmarks showed AWQ Conv1d was slower than FP16 for those shapes.
- The selective Conv1d gate avoids those losing AWQ calls and improves total time from 52.11 s to 50.02 s.

The current best whole-pipeline mode remains `int8_baseline` at 48.91 s total. The optimized AWQ path is now faster than FP16, but not faster than the INT8 baseline.

## Next Work

The next optimization target should be reducing pointwise Conv2d overhead rather than replacing only the GEMM body:

- Keep the Conv1d profitability gate for this model; converting all attention projections to AWQ is slower.
- Investigate why AWQ linear dispatch still leaves the selective path 2.28% slower than `int8_baseline`.
- If we want Conv1d AWQ to win for 192/384/768-channel attention, the missing piece is a native strided/layout-aware Conv1d projection kernel. The current wrapper pays input transpose/flatten and output transpose/contiguous costs around AWQ GEMM.
- Keep `skip_pointwise=True` for production baseline unless a model has non-skip 1x1 Conv2d shapes that benchmark faster under AWQ.

Raw results:

- `fp16_s200_n168_b42/results.json`
- `int8_baseline_s200_n168_b42/results.json`
- `int8_awq_full_s200_n168_b42/results.json`
- `int8_awq_full_selective_s200_n168_b42/results.json`
