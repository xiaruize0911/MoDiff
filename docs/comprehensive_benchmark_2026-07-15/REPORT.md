# MoDiff comprehensive benchmark — kernels & pipeline (2026-07-15)

Kernel-level and pipeline-level speed + IO + profile of the LSUN-churches latent-diffusion UNet.
**Hardware:** NVIDIA A40 (Ampere sm_86; fp16 149.7 TFLOP/s, int8 299 TOP/s, int4 599 TOP/s, DRAM 696 GB/s).
**Config:** batch 32, DDIM, per-step numbers. Raw data in [`data/`](data/), scripts in [`scripts/`](scripts/).

> ### ⚙️ Config (2026-07-15c)
> - **Attention runs on the math (non-flash) SDPA backend**
>   (`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`) in every attention block. This is the permanent
>   design of `TokenMajorAttentionBlock`, not a toggle: the math backend keeps the QKᵀ / AV products as plain
>   cuBLAS batched GEMMs that can be intercepted and quantized, unlike the opaque fused-flash kernel. The
>   custom **fused GN→qkv kernel stays on for all modes**.
> - **fp32 (full-precision, no quantization) is included as the baseline mode** alongside fp16 so the
>   quantization and MoDiff modes can be read against a true unquantized reference, not just fp16.

**The 6 pipeline modes:** `fp32` (full precision, autocast off), `fp16`, `int8 base` (`int8_baseline`),
`int8 modiff` (`int8`, temporal caching), `int4 base` (`int4_baseline`), `int4 modiff` (`int4`). "base" =
quantized kernels, no temporal caching; "modiff" = MoDiff error-compensated temporal caching (`o_hat`/`a_hat`
deltas) across DDIM steps.

### Methodology

The A40 idles at 210 MHz and boosts to 1740 MHz, and clock-locking is not permitted here, so **warmup
dominates measurement quality**. Kernels: 30 warmup + 60 timed iters in a tight loop. Pipeline: **≥6 s
sustained `sample()` warmup + 12 back-to-back timed runs**, reporting median/min/stdev; measured pipeline
**stdev is 0.04–0.11 ms/step (<0.2%)** for every mode except int8 modiff (0.73). GPU-busy = `torch.profiler`
device self-time.

---

## 1. Kernel speed benchmark

### Conv (top shapes by cost) — ![conv speed](04_kernel_conv_speed.png)

int8 conv beats fp16 cuDNN on compute-heavy **3×3** convs (`384→384 3×3 16²`: fp16 195 → int8 104 µs;
`576→192 3×3 32²`: 586 → 329 µs); int4 goes further only when channels are large enough to amortise its
weight-unpack. Full table: [`data/kernel_conv_speed.csv`](data/kernel_conv_speed.csv).

### Conv base vs MoDiff (and why int4 ≷ int8) — ![conv modiff](05_kernel_conv_modiff.png)

**MoDiff's temporal path** (`conv2d_int{8,4}_fprop_o_hat` + `step1` delta-quantize + `sub`/`accumulate`) adds
work and skips no convolution, so it is **~2–2.5× slower per conv** (`384→384 3×3 32²`: int8 base 442 →
int8 MoDiff 1047 µs). It buys temporal *accuracy*, not speed.

**int4-vs-int8 base is shape-dependent:** int4 **wins on large-channel 3×3** (`384→384 3×3 32²` 442→400 µs;
`768→384 3×3 16²` 205→158 µs) but **loses at ≤192 channels / 1×1** (`192→192 3×3 32²` 118→194 µs) where the
int4 weight-unpack + lower 4-bit tensor-core occupancy aren't amortised. In aggregate int4 is still faster
(profile §3: conv bucket 7.6 vs 9.8 ms).

### Attention (GN→qkv fusion + math SDPA) — ![attn](07_kernel_attn.png)

- **GN→qkv fused** (custom CUTLASS): **1.11×** (C192/T1024, 236 → 213 µs), **1.26×** (C384/T256, 169 → 134 µs)
  vs GroupNorm+cuBLAS. Only fires where tokens `T` is a multiple of the conv tile (T=1024, 256); smaller
  blocks fall back to GroupNorm + cuBLAS.
- **SDPA on the math backend** is the single most expensive kernel in the UNet: on the O(T²) high-res block
  it materializes the full `[N,heads,T,T]` score matrix and costs **4060 µs** (C192/T1024), 360 µs at
  C384/T256. This is the deliberate cost of an interceptable/quantizable attention core.
- **Linear** (qkv/proj): int8 is **2–3× slower than fp16 cuBLAS** at these channel counts (K below the int8
  crossover ~2048), so int8 linear is not used.

---

## 2. Kernel IO benchmark

Effective DRAM bandwidth = (bytes in + weights + bytes out) / time, vs the 696 GB/s peak.
![kernel IO](06_kernel_io.png) — data: [`kernel_conv_io.csv`](data/kernel_conv_io.csv), [`kernel_linear_io.csv`](data/kernel_linear_io.csv)

- **3×3 convs are mid-bandwidth** (64–163 GB/s) — not DRAM-saturated; quantization helps by cutting *compute*.
- **1×1 convs and the qkv GEMM are memory-bound and near-efficient** (fp16 up to 473 GB/s ≈ 68% of peak); int8/int4
  move fewer bytes at lower effective bandwidth, so they don't win there.
- **Math attention is extremely IO-heavy**: the T=1024 block writes+reads a ~512 MB score matrix per call —
  the reason SDPA dominates the step and the pipeline peak memory sits ~0.9 GiB above a flash-based config.

---

## 3. Kernel profile (per-operation GPU time, by mode)

![kernel profile](03_kernel_profile.png) — data: [`data/kernel_profile.csv`](data/kernel_profile.csv)

| bucket | fp32 | fp16 | int8 base | int8 modiff | int4 base | int4 modiff |
|---|--:|--:|--:|--:|--:|--:|
| conv (GEMM) | 27.16 | 13.67 | 9.75 | 12.14 | **7.64** | **7.65** |
| GEMM (qkv/proj + attn QKᵀ·AV) | 30.79 | 13.37 | 13.37 | 13.36 | 13.38 | 13.37 |
| attention (softmax + SDPA) | 22.34 | 11.38 | 11.37 | 11.38 | 11.37 | 11.40 |
| GroupNorm | 7.51 | 5.91 | 5.81 | 5.67 | 5.50 | 5.67 |
| conv store epilogue | 0 | 1.82 | 1.60 | 1.35 | 2.49 | 1.35 |
| quantize / MoDiff delta | 0 | 0 | 0.20 | **2.98** | 0.18 | **2.62** |
| elementwise / copy | 11.96 | 7.55 | 5.64 | 6.74 | 5.13 | 6.72 |
| upsample / concat | 1.78 | 1.07 | 1.32 | 1.06 | 1.32 | 1.06 |
| other | 0.35 | 0.30 | 0.28 | 2.57 | 0.28 | 2.56 |
| **GPU-busy total** | **101.89** | **55.07** | **49.35** | **57.24** | **47.29** | **52.40** |

- **fp32 is ~1.85× the fp16 GPU-busy** — every compute bucket roughly doubles (conv 13.7→27.2, the GEMM bucket
  13.4→30.8, attention softmax 11.4→22.3), as expected when the tensor cores drop from fp16 to fp32 throughput.
- **Attention dominates the fp16/quantized modes (~42% of the step).** The softmax bucket is **11.4 ms** and the
  QKᵀ/AV matmuls (~11.8 ms) live in the GEMM bucket (13.4 ms total; qkv/proj alone are ~1.6 ms). Combined
  attention cost ≈ 23 ms/step.
- *Profile caveat:* with math SDPA the QKᵀ/AV matmuls are cuBLAS GEMMs **indistinguishable by kernel name from
  qkv/proj**, so they merge into the GEMM bucket — which is why that bucket reads 13.4 ms rather than ~1.6 ms.
- **Quantization only moves the conv bucket** (13.7 → 9.8 int8 → 7.6 int4). GroupNorm (~5.5 ms) + the
  attention buckets are dtype-invariant, so quantization's relative win is bounded by Amdahl (attention is
  the large dtype-invariant remainder).
- **MoDiff adds ~2.6–3.0 ms `quantize / MoDiff delta` + ~2.3 ms `other`** — the temporal machinery.

---

## 4. Pipeline speed benchmark

![pipeline speed](01_pipeline_speed.png) — data: [`data/pipeline_speed.csv`](data/pipeline_speed.csv)

| mode | wall med | wall min | stdev | GPU-busy | wall speedup vs fp16 | vs fp32 |
|---|--:|--:|--:|--:|--:|--:|
| fp32 | 102.71 | 102.60 | 0.11 | 101.89 | 0.55× | 1.00× |
| fp16 | 56.10 | 56.04 | 0.04 | 55.07 | 1.00× | 1.83× |
| int8 base | 50.33 | 50.26 | 0.04 | 49.35 | 1.11× | 2.04× |
| int8 modiff | 58.56 | 58.43 | 0.73 | 57.24 | 0.96× | 1.75× |
| **int4 base** | **48.17** | **48.10** | 0.06 | **47.29** | **1.16×** | **2.13×** |
| int4 modiff | 53.70 | 53.65 | 0.12 | 52.40 | 1.04× | 1.91× |

- **`int4 base` is the fastest mode (1.16× vs fp16, 2.13× vs fp32)**, `int8 base` next (1.11× / 2.04×). The
  quantization speedup over fp16 is bounded because the now-dominant, dtype-invariant attention (~23 ms)
  dilutes the conv savings (Amdahl: only ~28% of the step is quantizable).
- **fp16 is 1.83× faster than fp32** — the plain-precision headline. Everything below fp16 is the
  quantization win on top of that.
- **MoDiff modes:** int8 modiff is slower than fp16 (0.96×); int4 modiff just above (1.04×). The temporal
  delta machinery (§3) is real, verified GPU work — an accuracy mechanism, not a speed one.

---

## 5. Pipeline IO benchmark

![pipeline io](02_pipeline_io.png) — data: [`data/pipeline_io.csv`](data/pipeline_io.csv)

| mode | peak allocated MiB | peak reserved MiB | MoDiff cache MiB |
|---|--:|--:|--:|
| fp32 | 4920 | 6084 | 0 |
| fp16 | 4369 | 4964 | 0 |
| int8 base | 4551 | 4834 | 0 |
| int8 modiff | 4959 | 5524 | **634** |
| int4 base | **4295** | 4668 | 0 |
| int4 modiff | 4707 | 5356 | **634** |

- **fp32 has the largest footprint** (4920 MiB allocated, 6084 reserved): fp32 activations + the fp32 T×T
  attention score tensors. fp16 halves most of that.
- Base modes sit near each other (activations dominate at batch 32); **int4 base has the smallest footprint**
  (4295 MiB). **MoDiff adds a fixed +634 MiB temporal cache** (`a_hat`/`o_hat`) on top.
- Math attention is the heaviest single IO consumer across all modes — the `[N,heads,T,T]` score tensor on the
  T=1024 block (~512 MB in fp16, ~1 GB in fp32).

---

## Bottom line

- **Speed:** `int4 base` is fastest (**1.16× wall vs fp16, 2.13× vs fp32**), `int8 base` next (1.11× / 2.04×).
  fp16 alone is **1.83× faster than fp32**. Quantization's win over fp16 is Amdahl-bounded because attention
  is now ~42% of the step and dtype-invariant.
- **Attention is the dominant cost (~23 ms/step)** — the math SDPA backend's T×T score materialization on the
  high-res block (4060 µs) is the single biggest kernel. It is kept on the math backend by design so the
  QKᵀ/AV products stay as interceptable/quantizable cuBLAS GEMMs.
- **int4 vs int8:** int4 wins on large-channel 3×3 convs (aggregate conv 7.6 vs 9.8 ms), loses on ≤192-channel /
  1×1 — net faster.
- **MoDiff:** slower than fp16 (temporal delta machinery + 634 MiB cache) — accuracy, not speed.
- **IO:** compute-bound 3×3 convs are where precision helps; math attention is the heaviest IO consumer, and
  fp32 carries the largest overall footprint.

*Scripts (in [`scripts/`](scripts/)): `pipeline.py`, `kernel.py`, `mkplots.py`. Re-run with
`PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass`. Attention: math SDPA backend via
`TokenMajorAttentionBlock`; fused GN→qkv on all modes.*
