# Extended MoDiff Benchmark Report

**Date**: 2026-03-18 04:16:10
**GPU**: NVIDIA A40
**Batch Size**: 32
**Timesteps**: 200
**Latent Shape**: (4, 32, 32)

## Pipeline Timing Results

| Mode | Time/Sample (s) | Speedup vs FP32 | Peak Memory (MB) | Graphs | Captures | Replays |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 0.912 | - | 39051 | - | - | - |
| fp16 | 0.407 | 2.24x | 9992 | - | - | - |
| int8 | 0.350 | 2.60x | 13099 | - | - | - |
| int8_baseline | 0.348 | 2.62x | 11130 | - | - | - |
| int4 | 0.327 | 2.79x | 12874 | - | - | - |
| int4_baseline | 0.292 | 3.12x | 10905 | - | - | - |
| int8_cudagraph | 0.337 | 2.70x | 23325 | 2 | 2 | 800 |
| int8_cudagraph_baseline | 0.330 | 2.77x | 22770 | 1 | 1 | 800 |
| int8_separate | 0.483 | 1.89x | 10622 | - | - | - |
| int8_separate_baseline | 0.487 | 1.87x | 9354 | - | - | - |
| int4_separate | 0.449 | 2.03x | 10396 | - | - | - |
| int4_separate_baseline | 0.337 | 2.71x | 9129 | - | - | - |

## CUDA Graph Replay Stats

| Mode | Graphs | Captures | Replays | Replays / Captures |
| --- | --- | --- | --- | --- |
| int8_cudagraph | 2 | 2 | 800 | 400.0 |
| int8_cudagraph_baseline | 1 | 1 | 800 | 800.0 |

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
  - the benchmark pre-captures those graphs on a short valid DDIM schedule before timed sampling, so graph construction cost does not leak into the steady-state timing
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
| INT8_32x192x32x32 | 0.648 | 1.541 | 2.38x |
| INT4_32x192x32x32 | 0.530 | 1.197 | 2.26x |
| INT8_32x384x16x16 | 0.346 | 0.810 | 2.34x |
| INT4_32x384x16x16 | 0.285 | 0.635 | 2.22x |
| INT8_32x768x8x8 | 0.253 | 0.490 | 1.94x |
| INT4_32x768x8x8 | 0.181 | 0.364 | 2.01x |

### Detailed Kernel Breakdown

| Shape | Fused Step1 (ms) | Fused Conv (ms) | Sep Step1 (ms) | Sep Conv (ms) |
| --- | --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.287 | 0.361 | 1.083 | 0.458 |
| INT4_32x192x32x32 | 0.279 | 0.251 | 0.849 | 0.348 |
| INT8_32x384x16x16 | 0.148 | 0.198 | 0.562 | 0.248 |
| INT4_32x384x16x16 | 0.144 | 0.142 | 0.443 | 0.192 |
| INT8_32x768x8x8 | 0.073 | 0.179 | 0.285 | 0.205 |
| INT4_32x768x8x8 | 0.070 | 0.111 | 0.227 | 0.137 |

## Kernel Timing: Compute+DQ vs Compute+DQ+Update o_hat

| Shape | Compute+DQ (ms) | Compute+DQ+Update o_hat (ms) | Overhead |
| --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.321 | 0.361 | +0.041ms (+12.7%) |
| INT4_32x192x32x32 | 0.210 | 0.251 | +0.041ms (+19.6%) |
| INT8_32x384x16x16 | 0.178 | 0.198 | +0.020ms (+11.3%) |
| INT4_32x384x16x16 | 0.121 | 0.142 | +0.021ms (+17.2%) |
| INT8_32x768x8x8 | 0.170 | 0.179 | +0.009ms (+5.2%) |
| INT4_32x768x8x8 | 0.100 | 0.111 | +0.011ms (+10.6%) |

## Layer-level Quantization Timing

These measurements compare the current dynamic activation quantization path against a static-scale path using the same quantization kernels.

### Dynamic vs Static Quantization

| Shape | Dynamic (ms) | Static (ms) | Dynamic overhead | Absmax+scale (ms) | IO proxy (ms) | Compute est. (ms) | Dominant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.196 | 0.056 | +0.141ms (+251.5%) | 0.150 | 0.094 | 0.000 | io |
| INT4_32x192x32x32 | 0.190 | 0.050 | +0.140ms (+277.2%) | 0.150 | 0.094 | 0.000 | io |
| INT8_32x384x16x16 | 0.104 | 0.030 | +0.074ms (+242.1%) | 0.089 | 0.049 | 0.000 | io |
| INT4_32x384x16x16 | 0.101 | 0.028 | +0.073ms (+262.3%) | 0.087 | 0.049 | 0.000 | io |
| INT8_32x768x8x8 | 0.055 | 0.022 | +0.032ms (+146.2%) | 0.077 | 0.027 | 0.000 | io |
| INT4_32x768x8x8 | 0.052 | 0.021 | +0.031ms (+144.9%) | 0.078 | 0.027 | 0.000 | io |

### Quantization interpretation

- **Dynamic quantization** includes the per-tensor absmax reduction and scale computation in the hot path.
- **Static quantization** removes that scale-discovery step and directly quantizes with a cached scale.
- **IO proxy** is a tensor copy lower bound for memory traffic.
- **Compute estimate** is `static_quant_ms - io_proxy_ms`, clipped at zero; if it stays small, the quantization kernel is predominantly IO-limited.

## Analysis

### Fairness configuration

- TF32 is disabled for both matmul and cuDNN paths in this benchmark, so the `fp32` baseline stays true FP32 rather than silently using TensorFloat-32 acceleration.
- The baseline quantized modes use the same optimized kernels and static scales as the MoDiff modes; the only intended difference is temporal caching.

### Task 1: CUTLASS INT8 + CUDA Graph

CUDA Graphs reduce Python/kernel-launch overhead by replaying captured UNet executions.
In this implementation, the graph replay is real and is exercised in the benchmark:
- `int8_cudagraph_baseline` captures 1 graph and replays it 800 times
- `int8_cudagraph` captures 2 graphs (first/modulated) and replays them 800 times

`int8_cudagraph` now uses the same CUTLASS INT8 kernels as fused `int8` for the conv/modulation path.
This isolates the effect of CUDA Graph replay on top of the optimized backend instead of benchmarking a different FP16 fallback backend.
Any remaining gap between `int8_cudagraph` and eager fused `int8` therefore reflects graph-capture constraints, first-step capture cost, and the interaction between static replay buffers and the MoDiff execution schedule rather than a backend mismatch.

### Why the earlier graph numbers looked slow

The slow `int8_cudagraph` / `int8_cudagraph_baseline` run was primarily a benchmarking bug rather than a pure kernel regression.
The pre-capture path used an invalid short DDIM schedule (`S=3`), but DDIM uniform discretization only works cleanly for step counts that divide the 1000-step base schedule.
That could either fail outright or prevent the graphs from being fully pre-captured before the timed sample, which makes the first measured batch pay graph-construction cost.
The benchmark now pre-captures with the minimum valid schedule needed for each mode:
- `int8_cudagraph`: 2 steps (captures both first-step and modulated graphs)
- `int8_cudagraph_baseline`: 1 step (captures the single baseline graph)

### Task 2: Fused vs Separate Kernels

The current MoDiff implementation fuses multiple operations into fewer kernel launches:
- **Fused Step1**: sub_absmax_scale + scale_quantize + dequant_accumulate
- **Fused Conv**: conv + weight_scale + o_hat_accumulate

The separate baseline breaks these into individual PyTorch/CUTLASS calls.
Fusion benefit is primarily from reduced kernel launch overhead and memory bandwidth savings.

### Effect of CUDA Graph replay on CUTLASS INT8

- **`int8_cudagraph` vs `int8`**: 0.337s vs 0.350s.
  - CUDA Graph replay reduces the eager CUTLASS INT8 path by 1.04x on this run
  - both modes use the same CUTLASS INT8 kernels on the hot path, so this speedup comes from reducing Python/kernel-launch overhead
  - the remaining trade-off is memory: graph replay keeps large static buffers alive, which raises peak memory usage

- **`int8_separate` vs `int8`**: 0.483s vs 0.350s.
  - the separate path performs the same MoDiff math, but it explodes the fused hot path into many kernels
  - the microbenchmark shows the main gap is in **Step1 fusion**, not in `o_hat` accumulation
  - fused INT8 total kernel time is 0.648ms vs 1.541ms on the representative 32x192x32x32 case
  - the extra cost of `compute+DQ+update_o_hat` over `compute+DQ` is only +0.041ms (+12.7%)
  - so the missing speedup is mostly due to unfused Step1 work and extra global-memory traffic, not because `o_hat` update is too expensive