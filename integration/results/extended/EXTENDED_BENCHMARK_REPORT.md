# Extended MoDiff Benchmark Report

**Date**: 2026-03-15 06:43:07
**GPU**: NVIDIA A40
**Batch Size**: 32
**Timesteps**: 200
**Latent Shape**: (4, 32, 32)

## Pipeline Timing Results

| Mode | Time/Sample (s) | Speedup vs FP32 | Peak Memory (MB) | Graphs | Captures | Replays |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 0.610 | - | 39051 | - | - | - |
| fp16 | 0.719 | 0.85x | 9992 | - | - | - |
| int8 | 0.349 | 1.75x | 13075 | - | - | - |
| int8_baseline | 0.312 | 1.95x | 11106 | - | - | - |
| int4 | 0.323 | 1.89x | 12850 | - | - | - |
| int4_baseline | 0.285 | 2.14x | 10881 | - | - | - |
| int8_cudagraph | 0.507 | 1.20x | 17321 | 2 | 2 | 198 |
| int8_cudagraph_baseline | 0.371 | 1.65x | 19308 | 1 | 1 | 199 |
| int8_separate | 0.482 | 1.27x | 10579 | - | - | - |
| int8_separate_baseline | 0.360 | 1.69x | 9312 | - | - | - |
| int4_separate | 0.444 | 1.37x | 10354 | - | - | - |
| int4_separate_baseline | 0.333 | 1.83x | 9087 | - | - | - |

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

### PyTorch INT8 + CUDA Graph modes

- **`int8_cudagraph`**: `PyTorchInt8Conv2d` from `integration/kernels/int8_cudagraph.py` plus real per-step UNet CUDA graph replay.
	- activations are quantized and immediately dequantized
	- convolution is still executed by PyTorch `F.conv2d` using FP16 weights
	- DDIM remains the outer Python loop, but each UNet step is replayed from captured CUDA graphs
	- two graphs are used: one for the first MoDiff step and one for all modulated steps
- **`int8_cudagraph_baseline`**: same backend, but with MoDiff disabled, so only a single per-step graph is needed.

### Separate-kernel baselines

- **`int8_separate`**: `SeparateKernelInt8Conv2d` from `integration/kernels/fused_baseline.py`.
	- residual computation, absmax, scale computation, quantization, dequantization, cache update, conv, dequant-by-weight-scale, and `o_hat` accumulation are split into separate operations
	- this preserves the same math as MoDiff but intentionally removes kernel fusion
- **`int8_separate_baseline`**: same separate INT8 backend with MoDiff disabled.
- **`int4_separate`** / **`int4_separate_baseline`**: same idea for INT4.

## Kernel Timing: Fused vs Separate

| Shape | Fused Total (ms) | Separate Total (ms) | Fusion Speedup |
| --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.647 | 1.541 | 2.38x |
| INT4_32x192x32x32 | 0.530 | 1.196 | 2.26x |
| INT8_32x384x16x16 | 0.345 | 0.809 | 2.34x |
| INT4_32x384x16x16 | 0.286 | 0.634 | 2.22x |
| INT8_32x768x8x8 | 0.251 | 0.488 | 1.94x |
| INT4_32x768x8x8 | 0.181 | 0.362 | 2.00x |

### Detailed Kernel Breakdown

| Shape | Fused Step1 (ms) | Fused Conv (ms) | Sep Step1 (ms) | Sep Conv (ms) |
| --- | --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.286 | 0.361 | 1.083 | 0.458 |
| INT4_32x192x32x32 | 0.279 | 0.251 | 0.848 | 0.348 |
| INT8_32x384x16x16 | 0.147 | 0.198 | 0.561 | 0.248 |
| INT4_32x384x16x16 | 0.144 | 0.142 | 0.442 | 0.192 |
| INT8_32x768x8x8 | 0.072 | 0.179 | 0.284 | 0.204 |
| INT4_32x768x8x8 | 0.070 | 0.111 | 0.226 | 0.137 |

## Kernel Timing: Compute+DQ vs Compute+DQ+Update o_hat

| Shape | Compute+DQ (ms) | Compute+DQ+Update o_hat (ms) | Overhead |
| --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.320 | 0.361 | +0.041ms (+12.8%) |
| INT4_32x192x32x32 | 0.211 | 0.251 | +0.040ms (+19.1%) |
| INT8_32x384x16x16 | 0.177 | 0.198 | +0.021ms (+11.8%) |
| INT4_32x384x16x16 | 0.121 | 0.142 | +0.020ms (+16.8%) |
| INT8_32x768x8x8 | 0.170 | 0.179 | +0.009ms (+5.6%) |
| INT4_32x768x8x8 | 0.101 | 0.111 | +0.010ms (+10.0%) |

## Analysis

### Task 1: PyTorch INT8 + CUDA Graph

CUDA Graphs reduce Python/kernel-launch overhead by replaying captured UNet executions.
In this implementation, the graph replay is real and is exercised in the benchmark:
- `int8_cudagraph_baseline` captures 1 graph and replays it 199 times
- `int8_cudagraph` captures 2 graphs (first/modulated) and replays them 198 times

However, `int8_cudagraph` is still slower than CUTLASS `int8` because the backend is different:
- CUTLASS `int8` uses true INT8 tensor-core convolution kernels
- `int8_cudagraph` quantizes activations, dequantizes them, and then runs FP16 `F.conv2d`
- graph replay removes launch overhead, but it does **not** convert the PyTorch backend into a true INT8 conv backend
- the result is better than the old eager pseudo-cudagraph path, but still slower than the fused CUTLASS INT8 implementation

### Task 2: Fused vs Separate Kernels

The current MoDiff implementation fuses multiple operations into fewer kernel launches:
- **Fused Step1**: sub_absmax_scale + scale_quantize + dequant_accumulate
- **Fused Conv**: conv + weight_scale + o_hat_accumulate

The separate baseline breaks these into individual PyTorch/CUTLASS calls.
Fusion benefit is primarily from reduced kernel launch overhead and memory bandwidth savings.

### Why `int8_cudagraph` and `int8_separate` do not beat fused `int8`

- **`int8_cudagraph` vs `int8`**: `0.507s` vs `0.349s` per sample.
	- CUDA Graph replay helps on the control-plane side
	- but the compute kernel is still PyTorch FP16 conv after quantize/dequantize
	- fused CUTLASS `int8` wins because it keeps the hot path in dedicated INT8 kernels end-to-end

- **`int8_separate` vs `int8`**: `0.482s` vs `0.349s` per sample.
	- the separate path performs the same MoDiff math, but it explodes the fused hot path into many kernels
	- the microbenchmark shows the main gap is in **Step1 fusion**, not in `o_hat` accumulation
	- fused INT8 total kernel time is `0.647ms` vs `1.541ms` on the representative `32x192x32x32` case
	- the extra cost of `compute+DQ+update_o_hat` over `compute+DQ` is only `+0.041ms (+12.8%)`
	- so the missing speedup is mostly due to unfused Step1 work and extra global-memory traffic, not because `o_hat` update is too expensive