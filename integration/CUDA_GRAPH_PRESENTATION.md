---
marp: true
theme: default
paginate: true
backgroundColor: "#ffffff"
color: "#111827"
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px;
    color: #111827;
    background: #ffffff;
  }
  h1 { color: #1d4ed8; border-bottom: 2px solid #bfdbfe; padding-bottom: 8px; }
  h2 { color: #1e40af; }
  h3 { color: #2563eb; margin: 6px 0; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { background: #eff6ff; color: #1e40af; padding: 5px 10px;
       border: 1px solid #bfdbfe; font-weight: 600; }
  td { padding: 4px 10px; border: 1px solid #dbeafe; color: #1f2937; }
  tr:nth-child(even) { background: #f0f9ff; }
  tr:nth-child(odd)  { background: #ffffff; }
  code { background: #f1f5f9; color: #be185d; padding: 2px 5px;
         border-radius: 4px; font-size: 0.88em; }
  pre  { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
         padding: 10px 14px; font-size: 0.72em; line-height: 1.35; }
  .highlight { color: #15803d; font-weight: bold; }
  .warn      { color: #dc2626; font-weight: bold; }
  .accent    { color: #d97706; font-weight: bold; }
  blockquote { border-left: 4px solid #3b82f6; background: #eff6ff;
               padding: 6px 14px; color: #1e3a8a; border-radius: 0 6px 6px 0;
               margin: 8px 0; font-size: 0.9em; }
  img { border-radius: 6px; max-width: 100%; max-height: 58vh;
        object-fit: contain; display: block; margin: 0 auto; }
  section.dense { font-size: 16px; }
  section.dense table { font-size: 12px; }
  section.dense th { padding: 4px 8px; }
  section.dense td { padding: 3px 8px; }
  section.dense pre { font-size: 0.68em; }
  section.small-img img { max-height: 45vh; }
---

# CUDA Graphs + INT8 Quantization in MoDiff

### Accelerating Latent Diffusion Inference with Kernel Fusion & Graph Replay

---
GPU: **NVIDIA A40** · Batch size: **32** · Steps: **200** · Date: **2026-03-17**

---

## Agenda

1. **What is a CUDA Graph?** — concept, terminology, why it matters
2. **How CUDA Graphs work** — capture → replay flow diagram
3. **CUDA Graphs in MoDiff** — two-phase graph strategy
4. **Software architecture** — kernel, fused-op, and graph layers
5. **All benchmark modes** — what each one measures
6. **End-to-end results** — speed, memory, speedup charts
7. **Kernel-level analysis** — fused vs separate, cache overhead
8. **Bottleneck analysis** — per-component profiling & optimization opportunities
9. **Key takeaways & optimization roadmap**

---

## 1 · What is a CUDA Graph?

Normal CUDA execution launches **one kernel at a time** from the CPU. Each launch pays:

- **CPU → driver overhead** — microseconds of Python / C++ call cost
- **Command queue serialization** — every `cudaLaunchKernel` is a round-trip
- **Synchronization bubbles** — Python GIL and `torch.autograd` bookkeeping

> **CUDA Graphs** let you record a sequence of operations into a static graph object, then **replay** that graph with a single `cudaGraphLaunch` call.

```
Without CUDA Graphs                With CUDA Graphs
──────────────────                 ────────────────
CPU: launch kernel A  ←──────────  CPU: replay graph  ─────┐
GPU: run A            CPU wait     GPU: A → B → C → D  ←───┘
CPU: launch kernel B               (one round-trip, zero bubbles)
GPU: run B
CPU: launch kernel C
...
```

**When this helps most:** models with hundreds of small kernel launches per step — exactly what UNet-based diffusion models look like.

---

<!-- _class: dense -->

## 2 · CUDA Graph: Capture → Replay Flow

```
┌──────────────────────────────────────────────────────────────┐
│  CAPTURE PHASE  (runs once, outside timed loop)              │
│  1. Allocate static input/output tensors (fixed addresses)   │
│  2. Warm-up: run the model once on a side stream             │
│  3. cudaStreamBeginCapture()                                 │
│       kernel A  →  kernel B  →  kernel C  →  …              │
│       (all GPU work is recorded, not executed)               │
│  4. cudaStreamEndCapture()  →  CUDAGraph object              │
└──────────────────────────────────────────────────────────────┘
              ↓  graph reused for every diffusion step  ↓
┌──────────────────────────────────────────────────────────────┐
│  REPLAY PHASE  (runs N times per sample)                     │
│  1. Copy live x_t into static_x    (memcpy, not a launch)    │
│  2. cudaGraphLaunch(graph)         (single round-trip)       │
│  3. Read output from static_output (zero copy)               │
└──────────────────────────────────────────────────────────────┘
```

**Constraint:** all tensor shapes and GPU memory addresses must be **identical** across replays — this is why static buffers and fixed batch sizes are required.

---

<!-- _class: dense -->

## 3 · CUDA Graphs in MoDiff: Two-Phase Strategy

MoDiff uses **error-compensated temporal modulation**. The UNet behaves differently at step `T=0` vs all subsequent steps — so two graphs must be captured:

```
DDIM outer loop (Python, 200 steps)
│
├── Step t = T  (first denoising step)
│     └── graph_manager("first")   ──→  capture once, replay once
│         • resets a_hat, o_hat caches to zero
│         • warms up MoDiff state for 3 iterations
│
└── Step t < T  (all 199 subsequent steps)
      └── graph_manager("modulated")  ──→  capture once, replay 199×
          • residual = x - a_hat_cache
          • quantize residual → INT8
          • conv(residual) + o_hat_cache
          • update a_hat_cache, o_hat_cache in-place
```

**Baseline mode** (MoDiff disabled): state is independent of step order  
→ only **one graph** is needed, captured once and replayed 200×.

| Mode | Graphs | Captures | Replays |
|------|--------|----------|---------|
| `int8_cudagraph` (MoDiff on) | 2 | 2 | 800 (128 samples) |
| `int8_cudagraph_baseline` (MoDiff off) | 1 | 1 | 800 (128 samples) |

---

<!-- _class: dense -->

## 4 · Software Architecture

```
benchmark_extended.py
  └─▶ DDIMSampler  (ldm/models/diffusion/ddim.py)
        └─▶ DiffusionWrapper  (ldm/models/diffusion/ddpm.py)
              ├─▶ UNetCudaGraphManager
              │     (integration/kernels/int8_cudagraph.py)
              │     • capture_phase / replay_phase
              │     • manages static buffers & graph records
              └─▶ UNet  (ldm/modules/diffusionmodules/openaimodel.py)
                    └─▶ FusedResBlock
                          (integration/fused_ops/fused_resblock.py)
                          ├─▶ in_conv / out_conv — OptimizedInt8Conv2d
                          │     (integration/kernels/int8_optimized.py)
                          │     • CUTLASS INT8 implicit GEMM
                          │     • fused step1_quantize_fprop kernel
                          │     • fused conv2d_int8_fprop_o_hat kernel
                          │     • MoDiff a_hat / o_hat cache update
                          └─▶ fused GroupNorm + SiLU
                                (integration/fused_ops/fused_gn_silu.py)
```

---

<!-- _class: dense -->

## 5 · Benchmark Modes — What Each Measures

| Mode | Backend | MoDiff | Graph | Purpose |
|------|---------|:------:|:-----:|---------|
| `fp32` | PyTorch FP32 | ✗ | ✗ | Baseline reference |
| `fp16` | PyTorch FP16 autocast | ✗ | ✗ | Standard autocast |
| `int8` | CUTLASS INT8 fused | ✓ | ✗ | Fused INT8 + MoDiff |
| `int8_baseline` | CUTLASS INT8 fused | ✗ | ✗ | Fused INT8, no MoDiff |
| `int4` | CUTLASS INT4 fused | ✓ | ✗ | Fused INT4 + MoDiff |
| `int4_baseline` | CUTLASS INT4 fused | ✗ | ✗ | Fused INT4, no MoDiff |
| `int8_cudagraph` | CUTLASS INT8 + graph | ✓ | ✓ 2 graphs | Graph on top of fused INT8 |
| `int8_cudagraph_baseline` | CUTLASS INT8 + graph | ✗ | ✓ 1 graph | Graph on top of fused INT8 |
| `int8_separate` | Separate INT8 kernels | ✓ | ✗ | Cost of kernel fusion |
| `int8_separate_baseline` | Separate INT8 kernels | ✗ | ✗ | Unfused INT8, no MoDiff |
| `int4_separate` | Separate INT4 kernels | ✓ | ✗ | Cost of fusion, INT4 |
| `int4_separate_baseline` | Separate INT4 kernels | ✗ | ✗ | Unfused INT4, no MoDiff |

Three independent axes: **precision** (FP32/FP16/INT8/INT4) · **fusion** (fused/separate) · **graph replay** (eager/graph).

---

## 6 · End-to-End Speedup

**NVIDIA A40 · Batch 32 · 200 DDIM steps · 128 total samples**

![Speedup bar chart](results/extended/charts/speedup.png)

---

## 6b · Peak Memory Usage

![Peak memory bar chart](results/extended/charts/memory.png)

---

## 6c · Speedup vs Memory Trade-off

![Scatter plot: speedup vs memory](results/extended/charts/tradeoff_scatter.png)

---

<!-- _class: dense -->

## 6d · End-to-End Results Table

| Mode | Time / Sample (s) | Speedup vs FP32 | Peak Memory (GB) |
|------|:-----------------:|:---------------:|:----------------:|
| `fp32` | 0.622 | 1.00× (ref) | 38.1 |
| `fp16` | 0.408 | 1.52× | 9.8 |
| `int8_separate` | 0.485 | 1.28× | 10.4 |
| `int8_separate_baseline` | 0.473 | 1.32× | 9.1 |
| `int4_separate` | 0.445 | 1.40× | 10.2 |
| `int4_separate_baseline` | 0.453 | 1.37× | 8.9 |
| `int4_baseline` | 0.418 | 1.49× | 10.6 |
| **`int8`** | **0.354** | **1.76×** | 12.8 |
| **`int8_baseline`** | **0.350** | **1.78×** | 10.9 |
| **`int8_cudagraph`** | **0.341** | **1.82×** | 22.8 |
| **`int8_cudagraph_baseline`** | **0.334** | **1.87×** | 22.2 |
| **`int4`** | **0.329** | **1.89×** | 12.6 |

> Best fused INT8+graph: **1.87×** speedup · Best overall: **INT4 fused at 1.89×**

---

<!-- _class: small-img -->

## 6e · CUDA Graph Overhead vs Speedup Trade-off

![CUDA Graph overhead comparison](results/extended/charts/cudagraph_overhead.png)

| | `int8` | `int8_cudagraph` | Delta |
|---|:---:|:---:|:---:|
| Time / sample (s) | 0.354 | 0.341 | **−0.013 s (−3.7%)** |
| Peak memory (GB) | 12.8 | 22.8 | **+10.0 GB (+78%)** |
| Graphs captured | — | 2 | — |

> Graph replay saves ~13 ms/sample on top of already-fused INT8 kernels, but nearly doubles peak memory due to static replay buffers kept resident.

---

<!-- _class: small-img -->

## 7 · Kernel Timing: Fused vs Separate

![Kernel timing grouped bar chart](results/extended/charts/kernel_timing.png)

Fusion benefit is **highest at large spatial resolution** — more global memory traffic to save, more launch overhead amortized.

---

## 7b · Kernel Breakdown: Step1 vs Conv

| Shape | Fused Step1 (ms) | Fused Conv (ms) | Sep Step1 (ms) | Sep Conv (ms) |
|-------|:----------------:|:---------------:|:--------------:|:-------------:|
| INT8 · 32×192×32×32 | 0.286 | 0.361 | 1.084 | 0.458 |
| INT4 · 32×192×32×32 | 0.280 | 0.251 | 0.855 | 0.349 |
| INT8 · 32×384×16×16 | 0.148 | 0.199 | 0.562 | 0.249 |
| INT4 · 32×384×16×16 | 0.145 | 0.142 | 0.444 | 0.192 |
| INT8 · 32×768×8×8   | 0.073 | 0.180 | 0.286 | 0.206 |
| INT4 · 32×768×8×8   | 0.071 | 0.112 | 0.229 | 0.137 |

**Key insight:** the *separate* Step1 path costs **3–4× more** than fused Step1. The Conv improvement is smaller because conv compute dominates regardless of fusion.

---

<!-- _class: small-img -->

## 7c · MoDiff Cache Update Overhead

![Cache update overhead](results/extended/charts/cache_overhead.png)

**Step1 (a_hat write) adds ~45–50%** because it reads + writes the full input activation tensor (8 bytes/element extra).  
**Conv (o_hat read) adds ~7–21%** — only a read of the existing `o_hat_cache`.

---

<!-- _class: dense -->

## 7d · Memory-IO Model for Cache Updates

```
Step1 (a_hat)                         Conv (o_hat)
─────────────────────────────         ─────────────────────────
FP32 read  N×C×H×W×4 bytes            FP32 read  N×K×H'×W'×4 bytes
FP32 write N×C×H×W×4 bytes            Extra IO:  N×K×H'×W'×4 bytes
Extra IO:  N×C×H×W×8 bytes

32×192×32×32 → +50 MiB extra IO       32×192×32×32 → +25 MiB extra IO
```

- **Step1** reads *and* writes the cache tensor → **8 bytes/element extra** → ~50% overhead
- **Conv** only reads `o_hat_cache` → **4 bytes/element extra** → ~13–21% overhead
- Combined extra IO per layer: **~75 MiB** at the largest spatial resolution

---

<!-- _class: small-img -->

## 8 · Bottleneck Analysis: Where is the Time Going?

**INT8 MoDiff + Fused ResBlocks + Triton GN+SiLU · Batch=32 · A40**

![Component breakdown](results/extended/charts/component_breakdown.png)

Wall-clock: **60 ms/step**. Top 3 components consume **79% of step time**.

---

<!-- _class: small-img -->

## 8b · Attention: The Biggest Optimization Target

Current attention uses naive BMM with O(n²) memory. **Flash Attention (SDPA)** eliminates this:

![Attention SDPA speedup](results/extended/charts/attention_sdpa.png)

At 32×32 spatial (seq=1024): **13.2× speedup**, saving **512 MiB** of attention matrix memory per layer.

---

## 8c · Attention Results Table

| Resolution | Seq Len | Naive BMM (ms) | SDPA (ms) | Speedup | Memory Saved |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 32×32 | 1024 | 5.866 | 0.445 | **13.2×** | 512 MiB |
| 16×16 | 256 | 0.445 | 0.063 | **7.0×** | 32 MiB |
| 8×8 | 64 | 0.059 | 0.028 | **2.1×** | 2 MiB |
| 4×4 | 16 | 0.057 | 0.027 | **2.1×** | 0.1 MiB |

> Attention is 30% of step time. SDPA could save **~14 ms/step** (80% of attention cost).

---

<!-- _class: small-img -->

## 8d · Triton Fused GN+SiLU: Resolution Matters

![Triton GN+SiLU comparison](results/extended/charts/triton_gn_silu.png)

Triton kernel is **slower at 32×32** (suboptimal tiling) but **1.4–1.8× faster** at smaller resolutions.

**Solution**: Resolution-adaptive dispatch — use `F.group_norm + F.silu` at 32×32, Triton below.

---

<!-- _class: dense -->

## 8e · torch.compile and FP16 Cache Results

### torch.compile on GN+SiLU+Conv pipeline

| Shape | Eager (ms) | Compiled (ms) | Speedup |
|-------|:---:|:---:|:---:|
| 32×192×32×32 | 2.060 | 1.565 | **1.32×** |
| 32×384×16×16 | 1.431 | 1.367 | 1.05× |
| 32×384×8×8 | 0.439 | 0.848 | 0.52× (slower) |

Useful at high resolution; overhead dominates at small spatial dims.

### FP16 cache accumulation

Manual FP16 path is **slower** than fused FP32 CUTLASS (0.31× at largest shape). The fused kernel accesses cache in-pipeline without extra memory round-trips. **FP16 savings require native CUTLASS kernel support.**

---

<!-- _class: dense small-img -->

## 8f · Optimization Roadmap

![Optimization roadmap](results/extended/charts/optimization_roadmap.png)

| Optimization | Est. Savings | Difficulty | Status |
|---|:---:|:---:|---|
| Flash Attention (SDPA) | **−14.3 ms** | Low | Drop-in replacement |
| CUDA Graph replay | −3.0 ms | Medium | Implemented |
| Torch.compile (hi-res) | −1.2 ms | Low | Selective application |
| Triton GN+SiLU dispatch | −1.2 ms | Low | Needs resolution check |
| INT4 quantization | −7.0 ms | High | Needs accuracy validation |

**Projected**: 60 ms → **40 ms/step** (1.5×) · With INT4: **33 ms/step** (1.8×, 3.2× vs FP32)

---

<!-- _class: dense -->

## 9 · Key Takeaways

### Performance (NVIDIA A40, 32-batch, 200-step DDIM)

| Optimization | Best speedup | Key trade-off |
|---|:---:|---|
| FP16 autocast | 1.52× | minimal; free with autocast |
| INT8 fused kernels + MoDiff | **1.76×** | +3 GB memory vs FP16 |
| INT4 fused kernels + MoDiff | **1.89×** | quality risk at INT4 |
| INT8 + CUDA Graph replay | **1.82×** | +10 GB memory (static buffers) |

### Bottleneck summary: where to focus next

| Component | % of Step | Key Optimization | Projected Gain |
|---|:---:|---|:---:|
| INT8 Convolutions | 39% | INT4 quantization | −7 ms |
| Attention (Naive BMM) | 30% | Flash Attention (SDPA) | −14.3 ms |
| Fused GN+SiLU | 10% | Resolution-adaptive dispatch | −1.2 ms |
| Framework overhead | 10% | CUDA Graph replay | −3.0 ms |

---

<!-- _class: dense -->

## 9b · Key Findings

1. **Kernel fusion is the dominant win** — 2.2–2.4× per-layer speedup; separate kernels cost 3–4× more.
2. **Attention is the #1 target** — 30% of step time, naive O(n²) BMM. SDPA gives **13.2×** at 32×32.
3. **CUDA Graph gives incremental gain (~4%)** — eliminates Python/launch overhead (~13 ms/sample).
4. **MoDiff cache overhead is small end-to-end** — Step1 a_hat costs ~50% per-kernel but only ~1% of total.
5. **Triton GN+SiLU needs resolution dispatch** — slower at 32×32, 1.4–1.8× faster at smaller dims.
6. **FP16 cache needs native CUTLASS support** — separate PyTorch ops are 3× slower than fused FP32.
7. **Memory doubles with CUDA Graph** — static replay buffers are a real trade-off on constrained GPUs.

---

<!-- _class: dense -->

## Appendix A · File Reference

| File | Role |
|------|------|
| `integration/kernels/int8_optimized.py` | `OptimizedInt8Conv2d`: CUTLASS INT8 + MoDiff fused path |
| `integration/kernels/int4_optimized.py` | `OptimizedInt4Conv2d`: CUTLASS INT4 + MoDiff fused path |
| `integration/kernels/int8_cudagraph.py` | `UNetCudaGraphManager`: two-phase graph capture + replay |
| `integration/kernels/fused_baseline.py` | `SeparateKernelInt8Conv2d`: unfused baseline |
| `integration/fused_ops/fused_resblock.py` | `FusedResBlock`: wires conv layers into UNet blocks |
| `integration/fused_ops/fused_gn_silu.py` | Triton fused GroupNorm + SiLU |
| `integration/benchmarks/benchmark_extended.py` | All-mode benchmark driver |
| `integration/benchmarks/benchmark_bottleneck.py` | Per-component bottleneck profiling |
| `integration/results/extended/bottleneck_results.json` | Bottleneck experiment data |
| `integration/results/extended/extended_results.json` | Raw benchmark numbers |
| `integration/results/extended/EXTENDED_BENCHMARK_REPORT.md` | Full auto-generated report |
| `integration/results/extended/FUSED_CACHE_UPDATE_REPORT.md` | Cache overhead microbenchmarks |
| `integration/results/extended/charts/` | Generated chart images used in this deck |

---

## Appendix B · MoDiff Error-Compensated Modulation Equations

The INT8 path implements the MoDiff paper (Gao et al., ICML 2025):

**First step (t = T):**
```
a_hat_T = Q(a_T)                    -- quantize activation, store as cache
o_hat_T = A(a_hat_T) + bias         -- conv on quantized input
```

**Modulated steps (t < T):**
```
a_hat_t = Q(a_t − a_hat_{t+1}) + a_hat_{t+1}     -- quantize residual, update cache
o_hat_t = A(Q(a_t − a_hat_{t+1})) + o_hat_{t+1}   -- conv residual, accumulate output cache
```

Where `Q(·)` is per-tensor symmetric INT8 quantization and `A(·)` is the convolution operator.

This allows INT8 inference across timesteps by accumulating **quantization error corrections** rather than re-quantizing the full activation at each step — the key insight of MoDiff.

---

*Presentation generated from live benchmark data · NVIDIA A40 · MoDiff + CUTLASS Integration · 2026-03-17*
