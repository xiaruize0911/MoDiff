# Extended MoDiff Benchmark Report

**Date**: 2026-03-19 14:11:00
**GPU**: NVIDIA A40
**Batch Size**: 32
**Timesteps**: 200
**Latent Shape**: (4, 32, 32)

## Pipeline Timing Results

| Mode | Time/Sample (s) | Speedup vs FP32 | Peak Memory (MB) | Graphs | Captures | Replays |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 0.917 | - | 37541 | - | - | - |
| fp16 | 0.406 | 2.26x | 4539 | - | - | - |
| int8 | 0.353 | 2.60x | 7698 | - | - | - |
| int8_baseline | 0.360 | 2.55x | 5729 | - | - | - |
| int4 | 0.330 | 2.78x | 7473 | - | - | - |
| int4_baseline | 0.299 | 3.06x | 5504 | - | - | - |
| int8_cudagraph | 0.338 | 2.72x | 23325 | 2 | 2 | 800 |
| int8_cudagraph_baseline | 0.329 | 2.78x | 22770 | 1 | 1 | 800 |
| int8_separate | 0.511 | 1.79x | 5221 | - | - | - |
| int8_separate_baseline | 0.497 | 1.84x | 3954 | - | - | - |
| int4_separate | 0.453 | 2.03x | 4996 | - | - | - |
| int4_separate_baseline | 0.377 | 2.43x | 3729 | - | - | - |

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
| INT8_32x192x32x32 | 0.651 | 1.546 | 2.38x |
| INT4_32x192x32x32 | 0.531 | 1.205 | 2.27x |
| INT8_32x384x16x16 | 0.347 | 0.812 | 2.34x |
| INT4_32x384x16x16 | 0.286 | 0.637 | 2.22x |
| INT8_32x768x8x8 | 0.254 | 0.491 | 1.93x |
| INT4_32x768x8x8 | 0.182 | 0.367 | 2.01x |

### Detailed Kernel Breakdown

| Shape | Fused Step1 (ms) | Fused Conv (ms) | Sep Step1 (ms) | Sep Conv (ms) |
| --- | --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.288 | 0.363 | 1.085 | 0.461 |
| INT4_32x192x32x32 | 0.280 | 0.251 | 0.855 | 0.350 |
| INT8_32x384x16x16 | 0.148 | 0.199 | 0.563 | 0.249 |
| INT4_32x384x16x16 | 0.144 | 0.142 | 0.444 | 0.193 |
| INT8_32x768x8x8 | 0.074 | 0.180 | 0.285 | 0.206 |
| INT4_32x768x8x8 | 0.071 | 0.112 | 0.230 | 0.137 |

## Kernel Timing: Compute+DQ vs Compute+DQ+Update o_hat

| Shape | Compute+DQ (ms) | Compute+DQ+Update o_hat (ms) | Overhead |
| --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.323 | 0.363 | +0.040ms (+12.5%) |
| INT4_32x192x32x32 | 0.210 | 0.251 | +0.041ms (+19.3%) |
| INT8_32x384x16x16 | 0.178 | 0.199 | +0.021ms (+11.6%) |
| INT4_32x384x16x16 | 0.122 | 0.142 | +0.021ms (+17.1%) |
| INT8_32x768x8x8 | 0.171 | 0.180 | +0.009ms (+5.2%) |
| INT4_32x768x8x8 | 0.101 | 0.112 | +0.011ms (+10.4%) |

## Layer-level Quantization Timing

These measurements compare the current dynamic activation quantization path against a static-scale path using the same quantization kernels.

### Dynamic vs Static Quantization

| Shape | Dynamic (ms) | Static (ms) | Dynamic overhead | Absmax+scale (ms) | IO proxy (ms) | Compute est. (ms) | Dominant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.197 | 0.056 | +0.141ms (+249.6%) | 0.151 | 0.095 | 0.000 | io |
| INT4_32x192x32x32 | 0.190 | 0.051 | +0.140ms (+275.5%) | 0.151 | 0.095 | 0.000 | io |
| INT8_32x384x16x16 | 0.105 | 0.030 | +0.074ms (+245.0%) | 0.093 | 0.050 | 0.000 | io |
| INT4_32x384x16x16 | 0.101 | 0.031 | +0.070ms (+227.8%) | 0.094 | 0.050 | 0.000 | io |
| INT8_32x768x8x8 | 0.055 | 0.024 | +0.031ms (+131.1%) | 0.089 | 0.036 | 0.000 | io |
| INT4_32x768x8x8 | 0.080 | 0.034 | +0.046ms (+134.9%) | 0.122 | 0.031 | 0.003 | io |

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

- **`int8_cudagraph` vs `int8`**: 0.338s vs 0.353s.
  - CUDA Graph replay reduces the eager CUTLASS INT8 path by 1.04x on this run
  - both modes use the same CUTLASS INT8 kernels on the hot path, so this speedup comes from reducing Python/kernel-launch overhead
  - the remaining trade-off is memory: graph replay keeps large static buffers alive, which raises peak memory usage

- **`int8_separate` vs `int8`**: 0.511s vs 0.353s.
  - the separate path performs the same MoDiff math, but it explodes the fused hot path into many kernels
  - the microbenchmark shows the main gap is in **Step1 fusion**, not in `o_hat` accumulation
  - fused INT8 total kernel time is 0.651ms vs 1.546ms on the representative 32x192x32x32 case
  - the extra cost of `compute+DQ+update_o_hat` over `compute+DQ` is only +0.040ms (+12.5%)
  - so the missing speedup is mostly due to unfused Step1 work and extra global-memory traffic, not because `o_hat` update is too expensive