# Extended MoDiff Benchmark Report

**Date**: 2026-03-15 07:23:32
**GPU**: NVIDIA A40
**Batch Size**: 32
**Timesteps**: 200
**Latent Shape**: (4, 32, 32)

## Pipeline Timing Results

| Mode | Time/Sample (s) | Speedup vs FP32 | Peak Memory (MB) | Graphs | Captures | Replays |
| --- | --- | --- | --- | --- | --- | --- |
| int8 | 0.498 | - | 13314 | - | - | - |
| int8_baseline | 0.312 | - | 11106 | - | - | - |
| int8_cudagraph | 0.215 | - | 17668 | 2 | 2 | 198 |
| int8_cudagraph_baseline | 0.241 | - | 20280 | 1 | 1 | 199 |

## CUDA Graph Replay Stats

| Mode | Graphs | Captures | Replays | Replays / Captures |
| --- | --- | --- | --- | --- |
| int8_cudagraph | 2 | 2 | 198 | 99.0 |
| int8_cudagraph_baseline | 1 | 1 | 199 | 199.0 |

## Mode Implementation Details

### Baseline floating-point modes

- **`fp32`**: original LDM inference path with standard PyTorch/CUDA kernels and no quantization.
- **`fp16`**: original LDM inference path under autocast. This reduces memory traffic, but does not change the algorithmic structure.

### CUTLASS fused MoDiff modes

- **`int8`**: `OptimizedInt8Conv2d` from `integration/kernels/int8_optimized.py`.
  - weights are quantized once to per-channel INT8
  - activation residual path uses the fused `step1_quantize_fprop` kernel
  - conv-side uses the fused `conv2d_int8_fprop_o_hat` kernel
  - MoDiff caches (`a_hat_cache`, `o_hat_cache`) are updated inside the optimized hot path
- **`int8_baseline`**: same CUTLASS INT8 backend, but MoDiff temporal modulation is disabled.
- **`int4`**: `OptimizedInt4Conv2d` from `integration/kernels/int4_optimized.py`.
  - same structure as INT8, but using packed INT4 activations/weights and fused INT4 kernels
- **`int4_baseline`**: same CUTLASS INT4 backend, with MoDiff disabled.

### CUTLASS INT8 + CUDA Graph modes

- **`int8_cudagraph`**: `OptimizedInt8Conv2d` from `integration/kernels/int8_optimized.py` plus real per-step UNet CUDA graph replay.
  - the underlying convolution and modulation path use the same CUTLASS INT8 kernels as fused `int8`
  - activations stay on the CUTLASS quantized path instead of dequantizing back to FP16 `F.conv2d`
  - DDIM remains the outer Python loop, but each UNet step is replayed from captured CUDA graphs
  - two graphs are used: one for the first MoDiff step and one for all modulated steps
- **`int8_cudagraph_baseline`**: same backend, but with MoDiff disabled, so only a single per-step graph is needed.

### Separate-kernel baselines

- **`int8_separate`**: `SeparateKernelInt8Conv2d` from `integration/kernels/fused_baseline.py`.
  - residual computation, absmax, scale computation, quantization, dequantization, cache update, conv, dequant-by-weight-scale, and `o_hat` accumulation are split into separate operations
  - this preserves the same math as MoDiff but intentionally removes kernel fusion
- **`int8_separate_baseline`**: same separate INT8 backend with MoDiff disabled.
- **`int4_separate`** / **`int4_separate_baseline`**: same idea for INT4.

## Analysis

### Task 1: CUTLASS INT8 + CUDA Graph

CUDA Graphs reduce Python/kernel-launch overhead by replaying captured UNet executions.
In this implementation, the graph replay is real and is exercised in the benchmark:
- `int8_cudagraph_baseline` captures 1 graph and replays it 199 times
- `int8_cudagraph` captures 2 graphs (first/modulated) and replays them 198 times

`int8_cudagraph` now uses the same CUTLASS INT8 kernels as fused `int8` for the conv/modulation path.
This isolates the effect of CUDA Graph replay on top of the optimized backend instead of benchmarking a different FP16 fallback backend.
Any remaining gap between `int8_cudagraph` and eager fused `int8` therefore reflects graph-capture constraints, first-step capture cost, and the interaction between static replay buffers and the MoDiff execution schedule rather than a backend mismatch.

### Task 2: Fused vs Separate Kernels

The current MoDiff implementation fuses multiple operations into fewer kernel launches:
- **Fused Step1**: sub_absmax_scale + scale_quantize + dequant_accumulate
- **Fused Conv**: conv + weight_scale + o_hat_accumulate

The separate baseline breaks these into individual PyTorch/CUTLASS calls.
Fusion benefit is primarily from reduced kernel launch overhead and memory bandwidth savings.

### Effect of CUDA Graph replay on CUTLASS INT8

- **`int8_cudagraph` vs `int8`**: 0.215s vs 0.498s.
  - CUDA Graph replay reduces the eager CUTLASS INT8 path by 2.32x on this run
  - both modes use the same CUTLASS INT8 kernels on the hot path, so this speedup comes from reducing Python/kernel-launch overhead
  - the remaining trade-off is memory: graph replay keeps large static buffers alive, which raises peak memory usage