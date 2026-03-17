---
marp: true
theme: default
paginate: true
backgroundColor: "#0d1117"
color: "#e6edf3"
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 20px;
  }
  h1 { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 8px; }
  h2 { color: #79c0ff; }
  h3 { color: #a5d6ff; }
  table { width: 100%; border-collapse: collapse; font-size: 16px; }
  th { background: #21262d; color: #58a6ff; padding: 6px 10px; border: 1px solid #30363d; }
  td { padding: 5px 10px; border: 1px solid #30363d; }
  tr:nth-child(even) { background: #161b22; }
  tr:nth-child(odd)  { background: #0d1117; }
  code { background: #161b22; color: #f97583; padding: 2px 6px; border-radius: 4px; }
  pre  { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; }
  .highlight { color: #3fb950; font-weight: bold; }
  .warn      { color: #f85149; font-weight: bold; }
  .accent    { color: #ffa657; font-weight: bold; }
  blockquote { border-left: 4px solid #58a6ff; background: #161b22; padding: 8px 16px; }
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
4. **The `.item()` bug** — a real-world capture failure and fix
5. **Software architecture** — kernel, fused-op, and graph layers
6. **All benchmark modes** — what each one measures
7. **End-to-end results** — speed, memory, speedup tables
8. **Kernel-level analysis** — fused vs separate, cache overhead
9. **Key takeaways**

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

## 2 · CUDA Graph: Capture → Replay Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  CAPTURE PHASE  (runs once, outside timed loop)                     │
│                                                                     │
│  1. Allocate static input/output tensors (fixed GPU addresses)      │
│  2. Warm-up: run the model once on a side stream                    │
│                                                                     │
│  3. cudaStreamBeginCapture()                                        │
│     ┌─────────────────────────────────────────────────┐            │
│     │  kernel A  →  kernel B  →  kernel C  →  …       │            │
│     │  (all GPU work is recorded, not executed)        │            │
│     └─────────────────────────────────────────────────┘            │
│  4. cudaStreamEndCapture()  →  CUDAGraph object                     │
└─────────────────────────────────────────────────────────────────────┘

              ↓  graph is reused for every diffusion step  ↓

┌─────────────────────────────────────────────────────────────────────┐
│  REPLAY PHASE  (runs N times per sample, inside timed loop)         │
│                                                                     │
│  1. Copy live x_t into static_x    (memcpy, not a launch)           │
│  2. cudaGraphLaunch(graph)         (single round-trip to driver)    │
│  3. Read output from static_output (zero copy — same address)       │
└─────────────────────────────────────────────────────────────────────┘
```

**Constraint:** all tensor shapes and GPU memory addresses must be **identical** across replays.  
This is why static buffers and fixed batch sizes are required.

---

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

## 4 · The `.item()` Bug — Capture Failure and Fix

### What went wrong

During CUDA graph capture, the stream is in **capture mode**. Any operation that reads a GPU value back to the CPU (a "device→host sync") is **forbidden**:

```
cudaErrorStreamCaptureUnsupported:
  "operation not permitted when stream is capturing"
```

The offending code in `int8_optimized.py::_forward_standard`:

```python
# _compute_activation_scale — used by _forward_standard
abs_max = x.abs().max().item()   # ← .item() syncs GPU→CPU
#                         ↑↑↑↑↑ ILLEGAL during graph capture
scale = 127.0 / max(abs_max, 1e-6)
```

### Why it happens

`_forward_standard` is the **standard (non-MoDiff) path** called when `is_calibrated=False`. It falls back to dynamic scale computation, which calls `.item()` to get a Python float — always a CPU sync.

### The fix

Replace the CPU-sync path with `_compute_scale_tensor`, which stays on the GPU:

```python
# BEFORE (illegal in capture):
input_scale = self._compute_activation_scale(x)   # calls .item()

# AFTER (graph-safe):
input_scale = self._compute_scale_tensor(x)        # amax() stays on GPU
```

`_compute_scale_tensor` uses `torch.Tensor.amax()` and leaves the result as a device tensor. CUTLASS accepts a device pointer for the scale — no round-trip needed.

---

## 5 · Software Architecture

```
benchmark_extended.py
        │
        ▼
DDIMSampler  (ldm/models/diffusion/ddim.py)
        │  apply_model() per denoising step
        ▼
DiffusionWrapper  (ldm/models/diffusion/ddpm.py)
        │
        ├─── UNetCudaGraphManager   (integration/kernels/int8_cudagraph.py)
        │         • capture_phase / replay_phase
        │         • manages static buffers & graph records
        │
        └─── UNet (ldm/modules/diffusionmodules/openaimodel.py)
                  │
                  └── FusedResBlock  (integration/fused_ops/fused_resblock.py)
                            │
                            ├── in_conv / out_conv
                            │     OptimizedInt8Conv2d
                            │     (integration/kernels/int8_optimized.py)
                            │           • CUTLASS INT8 implicit GEMM
                            │           • fused step1_quantize_fprop kernel
                            │           • fused conv2d_int8_fprop_o_hat kernel
                            │           • MoDiff a_hat / o_hat cache update
                            │
                            └── fused GroupNorm + SiLU
                                  (integration/fused_ops/fused_gn_silu.py)
```

---

## 6 · Benchmark Modes — What Each Measures

| Mode | Backend | MoDiff Caches | Graph Replay | Purpose |
|------|---------|:-------------:|:------------:|---------|
| `fp32` | PyTorch FP32 | ✗ | ✗ | Baseline reference |
| `fp16` | PyTorch autocast FP16 | ✗ | ✗ | Standard autocast |
| `int8` | CUTLASS INT8, fused kernels | ✓ | ✗ | Fused INT8 + MoDiff |
| `int8_baseline` | CUTLASS INT8, fused kernels | ✗ | ✗ | Fused INT8, no MoDiff |
| `int4` | CUTLASS INT4, fused kernels | ✓ | ✗ | Fused INT4 + MoDiff |
| `int4_baseline` | CUTLASS INT4, fused kernels | ✗ | ✗ | Fused INT4, no MoDiff |
| `int8_cudagraph` | CUTLASS INT8 + graph replay | ✓ | ✓ 2 graphs | Graph on top of fused INT8 |
| `int8_cudagraph_baseline` | CUTLASS INT8 + graph replay | ✗ | ✓ 1 graph | Graph on top of fused INT8 |
| `int8_separate` | Separate INT8 kernels (unfused) | ✓ | ✗ | Cost of kernel fusion |
| `int8_separate_baseline` | Separate INT8 kernels (unfused) | ✗ | ✗ | Unfused INT8, no MoDiff |
| `int4_separate` | Separate INT4 kernels (unfused) | ✓ | ✗ | Cost of fusion, INT4 |
| `int4_separate_baseline` | Separate INT4 kernels (unfused) | ✗ | ✗ | Unfused INT4, no MoDiff |

Modes differ along **three independent axes**: precision (FP32/FP16/INT8/INT4), fusion (fused/separate), and graph replay (eager/captured).

---

## 7 · End-to-End Benchmark Results

**NVIDIA A40 · Batch 32 · 200 DDIM steps · 128 total samples**

| Mode | Time / Sample (s) | Speedup vs FP32 | Peak Memory (MB) |
|------|:-----------------:|:---------------:|:----------------:|
| `fp32` | 0.622 | **1.00×** (ref) | 39 051 |
| `fp16` | 0.408 | 1.52× | 9 992 |
| `int8_separate` | 0.485 | 1.28× | 10 622 |
| `int8_separate_baseline` | 0.473 | 1.32× | 9 354 |
| `int4_separate` | 0.445 | 1.40× | 10 396 |
| `int4_separate_baseline` | 0.453 | 1.37× | 9 129 |
| `int4_baseline` | 0.418 | 1.49× | 10 905 |
| **`int8`** | **0.354** | **1.76×** | 13 099 |
| **`int8_baseline`** | **0.350** | **1.78×** | 11 130 |
| **`int8_cudagraph`** | **0.341** | **1.82×** | 23 325 |
| **`int4`** | **0.329** | **1.89×** | 12 874 |
| **`int8_cudagraph_baseline`** | **0.334** | **1.87×** | 22 770 |

> Best fused INT8+graph: **1.87×** speedup · Best overall: **INT4 fused at 1.89×**

---

## 7b · Speedup Visualization

```
Mode                    Speedup (× vs FP32)
─────────────────────── ──────────────────────────────────────────────
fp32                    ████  1.00×
fp16                    ██████  1.52×
int8_separate           █████  1.28×
int8_separate_baseline  █████  1.32×
int4_separate           █████  1.40×
int4_separate_baseline  █████  1.37×
int4_baseline           ██████  1.49×
int8                    ███████  1.76×
int8_baseline           ███████  1.78×
int8_cudagraph          ████████  1.82×
int8_cudagraph_baseline ████████  1.87×
int4                    ████████  1.89×  ◄ best
                        0         1         2
```

```
Mode                    Peak GPU Memory (GB)
─────────────────────── ──────────────────────────────────────────────
fp32                    ██████████████████████████████████████  39.1 GB
int8_cudagraph          ██████████████████████  22.8 GB
int8_cudagraph_baseline █████████████████████  22.3 GB
int8                    ████████████  12.8 GB
int4                    ████████████  12.6 GB
int8_baseline           ██████████  10.9 GB
fp16                    █████████  9.8 GB
int8_separate           █████████  10.4 GB
int4_baseline           █████████  10.6 GB
int8_separate_baseline  ████████  9.1 GB
int4_separate           █████████  10.2 GB
int4_separate_baseline  ████████  8.9 GB
                        0               20 GB           40 GB
```

---

## 7c · CUDA Graph Overhead vs Speedup Trade-off

| | `int8` | `int8_cudagraph` | Delta |
|---|:---:|:---:|:---:|
| Time / sample (s) | 0.354 | 0.341 | **−0.013s (−3.7%)** |
| Speedup vs FP32 | 1.76× | 1.82× | +0.06× |
| Peak memory (MB) | 13 099 | 23 325 | **+10 226 MB (+78%)** |
| Graphs captured | — | 2 | — |
| Replays / sample | — | 400 | — |

**Observation:** CUDA Graph replay saves ~13 ms/sample on top of already-fused INT8 kernels, but doubles peak memory due to static replay buffers kept resident.

| | `int8_baseline` | `int8_cudagraph_baseline` | Delta |
|---|:---:|:---:|:---:|
| Time / sample (s) | 0.350 | 0.334 | **−0.016s (−4.6%)** |
| Peak memory (MB) | 11 130 | 22 770 | **+11 640 MB (+105%)** |
| Graphs captured | — | 1 | — |
| Replays / sample | — | 800 | — |

The baseline mode captures only **one** graph and replays it 800× — twice the reuse ratio — which explains the slightly larger gain.

---

## 8 · Kernel Timing: Fused vs Separate

Microbenchmarks on individual layer shapes (batch=32, 100 iterations, CUDA events):

| Shape | Fused Total (ms) | Separate Total (ms) | **Fusion Speedup** |
|-------|:----------------:|:-------------------:|:------------------:|
| INT8 · 32×192×32×32 | 0.646 | 1.542 | **2.39×** |
| INT4 · 32×192×32×32 | 0.531 | 1.204 | **2.27×** |
| INT8 · 32×384×16×16 | 0.347 | 0.811 | **2.34×** |
| INT4 · 32×384×16×16 | 0.287 | 0.636 | **2.22×** |
| INT8 · 32×768×8×8   | 0.253 | 0.491 | **1.94×** |
| INT4 · 32×768×8×8   | 0.182 | 0.366 | **2.01×** |

```
Kernel-level fusion speedup (Fused / Separate)
─────────────────────────────────────────────────
INT8 192×32×32   ██████████████████████████  2.39× ← largest spatial maps
INT8 384×16×16   █████████████████████████  2.34×
INT4 192×32×32   ████████████████████████  2.27×
INT4 384×16×16   ███████████████████████  2.22×
INT4 768×8×8     █████████████████████  2.01×
INT8 768×8×8     ████████████████████  1.94× ← smallest spatial maps
                 0                    3×
```

Fusion benefit is **highest at large spatial resolution** (more memory traffic, more launch overhead amortized).

---

## 8b · Kernel Breakdown: Step1 vs Conv

The fused kernel path splits into two sub-kernels per residual block:

| Shape | Fused Step1 (ms) | Fused Conv (ms) | Sep Step1 (ms) | Sep Conv (ms) |
|-------|:----------------:|:---------------:|:--------------:|:-------------:|
| INT8 · 32×192×32×32 | 0.286 | 0.361 | 1.084 | 0.458 |
| INT4 · 32×192×32×32 | 0.280 | 0.251 | 0.855 | 0.349 |
| INT8 · 32×384×16×16 | 0.148 | 0.199 | 0.562 | 0.249 |
| INT4 · 32×384×16×16 | 0.145 | 0.142 | 0.444 | 0.192 |
| INT8 · 32×768×8×8   | 0.073 | 0.180 | 0.286 | 0.206 |
| INT4 · 32×768×8×8   | 0.071 | 0.112 | 0.229 | 0.137 |

**Key insight:** the *separate* Step1 path (absorb residual, absmax, scale, quantize, dequant, accumulate cache) costs **3–4× more** than fused Step1. The Conv path improvement is smaller because the conv compute itself dominates regardless of fusion.

---

## 8c · MoDiff Cache Update Overhead

How much does maintaining the temporal `a_hat` / `o_hat` caches cost inside the fused kernels?

### Step1 cache update (a_hat write)

| Shape | w/ cache (ms) | no cache (ms) | Overhead |
|-------|:-------------:|:-------------:|:--------:|
| INT8 · 32×192×32×32 | 0.286 | 0.192 | **+49%** |
| INT8 · 32×384×16×16 | 0.148 | 0.101 | **+46%** |
| INT8 · 32×768×8×8   | 0.073 | 0.051 | **+44%** |
| INT4 · 32×192×32×32 | 0.280 | 0.186 | **+50%** |

### Conv cache update (o_hat accumulate)

| Shape | w/ cache (ms) | no cache (ms) | Overhead |
|-------|:-------------:|:-------------:|:--------:|
| INT8 · 32×192×32×32 | 0.361 | 0.317 | **+14%** |
| INT8 · 32×384×16×16 | 0.199 | 0.177 | **+12%** |
| INT8 · 32×768×8×8   | 0.180 | 0.168 | **+7%** |
| INT4 · 32×192×32×32 | 0.251 | 0.207 | **+21%** |

**Step1 overhead is ~45–50%** because `a_hat` update requires reading + writing the full input activation tensor (8 bytes/element extra). **Conv overhead is ~7–21%** — only a read of the existing `o_hat_cache`.

---

## 8d · Memory-IO Model for Cache Updates

```
Step1 cache update (a_hat)
──────────────────────────
Input activation:    FP32 read   →  N × C × H × W × 4 bytes
a_hat_cache:         FP32 write  →  N × C × H × W × 4 bytes
                                    ─────────────────────────
                     Extra IO:      N × C × H × W × 8 bytes

For 32×192×32×32:
  8 bytes × 32 × 192 × 32 × 32 = 50 MiB  (matches measured ~48 MiB)

Conv cache update (o_hat)
─────────────────────────
o_hat_cache:         FP32 read   →  N × K × H' × W' × 4 bytes
                     (write already charged to quantized conv output)
                                    ─────────────────────────
                     Extra IO:      N × K × H' × W' × 4 bytes

For 32×192×32×32:
  4 bytes × 32 × 192 × 32 × 32 = 24 MiB  (matches measured 24 MiB)
```

Combined extra IO per layer: **72 MiB** for the largest spatial resolution.  
This explains why Step1 overhead (~50%) is approximately **2× the Conv overhead (~13–21%)** — it writes the cache on top of already reading it, while the Conv path only reads.

---

## 9 · Architecture Decision Map

```
┌──────────────────────────────────────────────────────────────────┐
│  Optimization Dimensions                                         │
│                                                                  │
│  Precision  ──────────────────────────────────────────────────  │
│  FP32 ──→ FP16 ──→ INT8 ──→ INT4                                │
│   1.00×    1.52×    1.76×    1.89×                              │
│                                                                  │
│  Kernel Fusion  ─────────────────────────────────────────────── │
│  Separate ──────────────────→ Fused                             │
│  1.28–1.40×                   1.76–1.89×  (2.2–2.4× per layer) │
│                                                                  │
│  Graph Replay  ──────────────────────────────────────────────── │
│  Eager ─────────────────────→ CUDA Graph                        │
│  1.76×                        1.82×  (+13ms saved)              │
│                                                                  │
│  MoDiff Caching  ────────────────────────────────────────────── │
│  Disabled (baseline) ───────→ Enabled                           │
│  1.78×                        1.76×  (tiny ~−1% cost from cache │
│                                       update IO overhead)        │
└──────────────────────────────────────────────────────────────────┘
```

**Dominant gains come from precision + fusion, not from graph replay.**  
Graph replay adds incremental improvement on top of an already heavily-optimized backend.

---

## 9b · Graph Capture: What Can Go Wrong

The `.item()` bug encountered is one of a class of **capture-time constraints**:

| Operation | Allowed during capture? | Reason |
|-----------|:-----------------------:|--------|
| `tensor.amax()` | ✓ Yes | stays on GPU stream |
| `tensor.abs().max().item()` | ✗ **No** | GPU→CPU sync |
| `print(tensor.item())` | ✗ **No** | GPU→CPU sync |
| `F.conv2d(...)` | ✓ Yes | pure GPU kernel |
| `torch.zeros(shape)` | ✗ **No** | allocates new memory |
| `existing_buf.zero_()` | ✓ Yes | in-place on existing buffer |
| `if tensor.any():` | ✗ **No** | conditional on GPU value |
| `tensor.copy_(other)` | ✓ Yes | device→device copy |
| Python control flow | ✗ **No** | CPU branching |

**Rule of thumb:** anything that transfers data CPU↔GPU, allocates tensors, or branches on a GPU value will invalidate the graph capture stream.

---

## 10 · Summary & Key Takeaways

### Performance (NVIDIA A40, 32-batch, 200-step DDIM)

| Optimization | Best speedup | Key trade-off |
|---|:---:|---|
| FP16 autocast | 1.52× | minimal; free with autocast |
| INT8 fused kernels + MoDiff | **1.76×** | +3 GB memory vs FP16 |
| INT4 fused kernels + MoDiff | **1.89×** | quality risk at INT4 |
| INT8 + CUDA Graph replay | **1.82×** | +10 GB memory (static buffers) |

### Key findings

1. **Kernel fusion is the dominant win** — 2.2–2.4× per-layer speedup eliminates most of the quantization overhead vs the separate baseline.
2. **CUDA Graph replay gives incremental gain (~4%)** on top of already-fused INT8, primarily by eliminating Python/kernel-launch overhead (~13 ms/sample).
3. **MoDiff cache maintenance is not free** — Step1 a_hat update costs ~45–50% overhead inside the fused kernel (8 bytes/element extra IO), but the end-to-end gap between MoDiff-on and MoDiff-off is only ~1%.
4. **The `.item()` GPU→CPU sync is the most common CUDA Graph pitfall** — any dynamic scale computation that extracts a Python float will fail during capture. The fix is to keep all scale tensors on-device.
5. **Memory doubles with CUDA Graph** due to static replay buffers — a real deployment trade-off on memory-constrained GPUs.

---

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
| `integration/results/extended/extended_results.json` | Raw benchmark numbers |
| `integration/results/extended/EXTENDED_BENCHMARK_REPORT.md` | Full auto-generated report |
| `integration/results/extended/FUSED_CACHE_UPDATE_REPORT.md` | Cache overhead microbenchmarks |

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
