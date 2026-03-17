# Extended MoDiff Benchmark Report

**Date**: 2026-03-17 03:14:41
**GPU**: NVIDIA A40
**Batch Size**: 32
**Timesteps**: 200
**Latent Shape**: (4, 32, 32)

## Pipeline Timing Results

| Mode | Time/Sample (s) | Speedup vs FP32 | Peak Memory (MB) | Graphs | Captures | Replays |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 0.622 | - | 39051 | - | - | - |
| fp16 | 0.408 | 1.52x | 9992 | - | - | - |
| int8 | 0.354 | 1.76x | 13099 | - | - | - |
| int8_baseline | 0.350 | 1.78x | 11130 | - | - | - |
| int4 | 0.329 | 1.89x | 12874 | - | - | - |
| int4_baseline | 0.418 | 1.49x | 10905 | - | - | - |
| int8_cudagraph | 0.341 | 1.82x | 23325 | 2 | 2 | 800 |
| int8_cudagraph_baseline | 0.334 | 1.87x | 22770 | 1 | 1 | 800 |
| int8_separate | 0.485 | 1.28x | 10622 | - | - | - |
| int8_separate_baseline | 0.473 | 1.32x | 9354 | - | - | - |
| int4_separate | 0.445 | 1.40x | 10396 | - | - | - |
| int4_separate_baseline | 0.453 | 1.37x | 9129 | - | - | - |

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
| INT8_32x192x32x32 | 0.646 | 1.542 | 2.39x |
| INT4_32x192x32x32 | 0.531 | 1.204 | 2.27x |
| INT8_32x384x16x16 | 0.347 | 0.811 | 2.34x |
| INT4_32x384x16x16 | 0.287 | 0.636 | 2.22x |
| INT8_32x768x8x8 | 0.253 | 0.491 | 1.94x |
| INT4_32x768x8x8 | 0.182 | 0.366 | 2.01x |

### Detailed Kernel Breakdown

| Shape | Fused Step1 (ms) | Fused Conv (ms) | Sep Step1 (ms) | Sep Conv (ms) |
| --- | --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.286 | 0.361 | 1.084 | 0.458 |
| INT4_32x192x32x32 | 0.280 | 0.251 | 0.855 | 0.349 |
| INT8_32x384x16x16 | 0.148 | 0.199 | 0.562 | 0.249 |
| INT4_32x384x16x16 | 0.145 | 0.142 | 0.444 | 0.192 |
| INT8_32x768x8x8 | 0.073 | 0.180 | 0.286 | 0.206 |
| INT4_32x768x8x8 | 0.071 | 0.112 | 0.229 | 0.137 |

## Kernel Timing: Compute+DQ vs Compute+DQ+Update o_hat

| Shape | Compute+DQ (ms) | Compute+DQ+Update o_hat (ms) | Overhead |
| --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.320 | 0.361 | +0.041ms (+13.0%) |
| INT4_32x192x32x32 | 0.211 | 0.251 | +0.041ms (+19.3%) |
| INT8_32x384x16x16 | 0.178 | 0.199 | +0.021ms (+12.0%) |
| INT4_32x384x16x16 | 0.122 | 0.142 | +0.021ms (+17.0%) |
| INT8_32x768x8x8 | 0.170 | 0.180 | +0.010ms (+5.7%) |
| INT4_32x768x8x8 | 0.101 | 0.112 | +0.010ms (+10.1%) |

## Analysis

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

- **`int8_cudagraph` vs `int8`**: 0.341s vs 0.354s.
  - CUDA Graph replay reduces the eager CUTLASS INT8 path by 1.04x on this run
  - both modes use the same CUTLASS INT8 kernels on the hot path, so this speedup comes from reducing Python/kernel-launch overhead
  - the remaining trade-off is memory: graph replay keeps large static buffers alive, which raises peak memory usage

- **`int8_separate` vs `int8`**: 0.485s vs 0.354s.
  - the separate path performs the same MoDiff math, but it explodes the fused hot path into many kernels
  - the microbenchmark shows the main gap is in **Step1 fusion**, not in `o_hat` accumulation
  - fused INT8 total kernel time is 0.646ms vs 1.542ms on the representative 32x192x32x32 case
  - the extra cost of `compute+DQ+update_o_hat` over `compute+DQ` is only +0.041ms (+13.0%)

---

## Bottleneck Analysis: Per-Component Time Breakdown

**Setup**: INT8 MoDiff + Fused ResBlocks + Triton GN+SiLU, Batch=32, 20 DDIM steps, A40 GPU.

Wall-clock per-step: **60.0 ms**

Profiling method: CUDA event hooks on every leaf module, classified by type. Events captured from the final DDIM step (steady state), summed across all layers.

| Component | Per-Step (ms) | % of Step | Layer Count |
| --- | --- | --- | --- |
| INT8 Convolutions (CUTLASS) | 23.4 | 39.1% | 70 |
| Attention (Naive BMM) | 17.8 | 29.7% | 42 |
| Fused GN+SiLU (Triton) | 6.2 | 10.3% | 35 |
| GroupNorm (standalone) | 2.7 | 4.5% | 22 |
| FP32 Convolutions | 2.2 | 3.6% | 19 |
| Resample (Up/Downsample) | 0.7 | 1.2% | 16 |
| Linear | 0.5 | 0.8% | 37 |
| SiLU (standalone) | 0.3 | 0.5% | 37 |
| Time Embedding | 0.2 | 0.4% | 1 |
| **Hooked Total** | **54.0** | **90.0%** | |
| Overhead / Other | 6.0 | 10.0% | |
| **Wall-clock Step** | **60.0** | **100.0%** | |

### Key Findings

**Top 3 bottlenecks account for 79% of step time:**
1. INT8 CUTLASS convolutions dominate at 39%, expected since the UNet is conv-heavy (70 quantized conv layers)
2. Attention blocks consume 30% — currently using naive batch-matrix-multiply (BMM) with explicit attention matrix materialization
3. Fused GN+SiLU takes 10% — the Triton kernel handles GroupNorm and SiLU in a single pass

### Experiment 2: Attention — Naive BMM vs Flash Attention (SDPA)

Replacing the naive `einsum('b i d, b j d -> b i j')` + softmax + `einsum('b i j, b j d -> b i d')` path
with `torch.nn.functional.scaled_dot_product_attention` (SDPA / Flash Attention):

| Config | Naive BMM (ms) | SDPA (ms) | Speedup | Memory Saved |
| --- | --- | --- | --- | --- |
| B32, C=192, 32×32 (seq=1024) | 5.866 | 0.445 | **13.2×** | 512 MiB |
| B32, C=384, 16×16 (seq=256) | 0.445 | 0.063 | **7.0×** | 32 MiB |
| B32, C=384, 8×8 (seq=64) | 0.059 | 0.028 | **2.1×** | 2 MiB |
| B32, C=768, 4×4 (seq=16) | 0.057 | 0.027 | **2.1×** | 0.1 MiB |

Flash Attention eliminates the O(n²) attention matrix materialization.
At the highest resolution (32×32, seq_len=1024), this saves 5.4ms per attention layer call and 512 MiB of memory.
With 42 attention layers across the UNet (multiple at the high-resolution levels), the model-level impact is substantial.

**Estimated model-level savings**: ~14 ms/step (80% reduction in attention time, weighted by resolution distribution).

### Experiment 3: Triton Fused GroupNorm+SiLU

| Shape | Separate (ms) | Triton Fused (ms) | Speedup |
| --- | --- | --- | --- |
| 32×192×32×32 | 0.356 | 1.570 | 0.23× (slower) |
| 32×384×16×16 | 0.188 | 0.134 | **1.41×** |
| 32×384×8×8 | 0.071 | 0.042 | **1.68×** |
| 32×768×4×4 | 0.068 | 0.038 | **1.80×** |
| 32×768×2×2 | 0.067 | 0.037 | **1.82×** |

The Triton kernel is slower at 32×32 spatial (likely suboptimal block tiling for large spatial dims) but provides 1.4–1.8× speedup at smaller resolutions. Since the UNet spends more compute at higher resolutions, the net benefit is mixed — the 32×32 regression partially offsets gains at lower resolutions.

**Recommendation**: Use resolution-adaptive dispatch — standard `F.group_norm + F.silu` at 32×32, Triton kernel at 16×16 and below.

### Experiment 4: FP16 Cache Accumulation

| Shape | FP32 Cache (ms) | FP16 Cache (ms) | Speedup | Memory Saved |
| --- | --- | --- | --- | --- |
| 32×192×32×32 | 0.287 | 0.913 | 0.31× (slower) | 24 MiB |
| 32×384×16×16 | 0.147 | 0.477 | 0.31× | 12 MiB |
| 32×384×8×8 | 0.033 | 0.135 | 0.25× | 3 MiB |

Converting caches to FP16 with separate PyTorch ops is **slower** than the fused FP32 CUTLASS kernel. The fused kernel accesses cache memory as part of the compute pipeline without extra memory round-trips. The manual FP16 path introduces 3 extra global memory operations (dequant → accumulate → requant) that outweigh the bandwidth savings from smaller data types.

**Recommendation**: FP16 cache savings require native CUTLASS kernel support (accumulate in FP16 within the fused kernel), not a separate PyTorch wrapper.

### Experiment 5: torch.compile on GN+SiLU+Conv Pipeline

| Shape | Eager (ms) | Compiled (ms) | Speedup |
| --- | --- | --- | --- |
| 32×192×32×32 | 2.060 | 1.565 | **1.32×** |
| 32×384×16×16 | 1.431 | 1.367 | **1.05×** |
| 32×384×8×8 | 0.439 | 0.848 | 0.52× (slower) |

`torch.compile` provides meaningful speedup at large spatial resolutions by fusing the GN+SiLU+Conv pipeline into optimized Triton kernels. At small spatial dims, the compilation overhead dominates.

**Recommendation**: Selectively apply torch.compile to high-resolution blocks only.

## Optimization Roadmap

Based on the bottleneck analysis, the following optimizations can reduce per-step time from **60 ms** to an estimated **40 ms** (1.5× improvement):

| Optimization | Estimated Savings | Difficulty | Status |
| --- | --- | --- | --- |
| Flash Attention (SDPA) | -14.3 ms (24%) | Low | Ready — drop-in replacement |
| CUDA Graph replay | -3.0 ms (5%) | Medium | Implemented, needs tuning |
| Torch.compile (high-res blocks) | -1.2 ms (2%) | Low | Requires selective application |
| Triton GN+SiLU (small spatial) | -1.2 ms (2%) | Low | Already implemented, needs dispatch fix |
| INT4 quantization (future) | -7.0 ms (12%) | High | Experimental, needs accuracy validation |

**Combined actionable savings (first 4 items)**: ~19.7 ms → projected **40.3 ms/step** (1.49× vs current INT8)

Including INT4 quantization: ~26.7 ms → projected **33.3 ms/step** (1.80× vs current, 3.22× vs FP32)
  - so the missing speedup is mostly due to unfused Step1 work and extra global-memory traffic, not because `o_hat` update is too expensive