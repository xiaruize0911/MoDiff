# MoDiff comprehensive benchmark — kernels & pipeline (2026-07-15)

Kernel-level and pipeline-level speed + IO + profile of the LSUN-churches latent-diffusion UNet.
**Hardware:** NVIDIA A40 (Ampere sm_86; fp16 149.7 TFLOP/s, int8 299 TOP/s, int4 599 TOP/s, DRAM 696 GB/s).
**Config:** batch 32, DDIM, per-step numbers. Raw data in [`data/`](data/), scripts in [`scripts/`](scripts/).

> ### ⚙️ Config change (2026-07-15b): flash attention removed
> Per request, **flash attention is disabled** — SDPA now runs on the **math (non-flash) backend**
> (`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`) in every attention block, and the custom **fused
> GN→qkv kernel stays on for all modes**. The math backend **materializes the full `[N, heads, T, T]` score
> matrix** that flash was built to avoid, so **every mode is ~1.6–1.85× slower and uses ~0.9 GiB more memory**
> than with flash. All numbers below are with this config. (For reference, the flash-attention numbers are in
> git history / the `../profiling_report_2026-07-14/` report.)

**The 5 pipeline modes:** `fp16`, `int8 base` (`int8_baseline`), `int8 modiff` (`int8`, temporal caching),
`int4 base` (`int4_baseline`), `int4 modiff` (`int4`). "base" = quantized kernels, no temporal caching;
"modiff" = MoDiff error-compensated temporal caching (`o_hat`/`a_hat` deltas) across DDIM steps.

### Methodology

The A40 idles at 210 MHz and boosts to 1740 MHz, and clock-locking is not permitted here, so **warmup
dominates measurement quality**. Kernels: 30 warmup + 60 timed iters in a tight loop. Pipeline: **≥6 s
sustained `sample()` warmup + 12 back-to-back timed runs**, reporting median/min/stdev; measured pipeline
**stdev is 0.03–0.10 ms/step (<0.2%)**. GPU-busy = `torch.profiler` device self-time.

---

## 1. Kernel speed benchmark

### Conv (top shapes by cost) — ![conv speed](04_kernel_conv_speed.png)

int8 conv beats fp16 cuDNN on compute-heavy **3×3** convs (`384→384 3×3 16²`: fp16 197 → int8 104 µs;
`576→192 3×3 32²`: 580 → 318 µs); int4 goes further only when channels are large enough to amortise its
weight-unpack. Full table: [`data/kernel_conv_speed.csv`](data/kernel_conv_speed.csv).

### Conv base vs MoDiff (and why int4 ≷ int8) — ![conv modiff](05_kernel_conv_modiff.png)

**MoDiff's temporal path** (`conv2d_int{8,4}_fprop_o_hat` + `step1` delta-quantize + `sub`/`accumulate`) adds
work and skips no convolution, so it is **~2–2.5× slower per conv** (`384→384 3×3 32²`: int8 base 425 →
int8 MoDiff 1041 µs). It buys temporal *accuracy*, not speed.

**int4-vs-int8 base is shape-dependent:** int4 **wins on large-channel 3×3** (`384→384 3×3 32²` 425→390 µs;
`768→384 3×3 16²` 203→156 µs) but **loses at ≤192 channels / 1×1** (`192→192 3×3 32²` 116→187 µs) where the
int4 weight-unpack + lower 4-bit tensor-core occupancy aren't amortised. In aggregate int4 is still faster
(profile §3: conv bucket 7.4 vs 9.5 ms).

### Attention (flash removed → math) — ![attn](07_kernel_attn.png)

- **GN→qkv fused** (custom CUTLASS): **1.11×** (C192/T1024), **1.25×** (C384/T256) vs GroupNorm+cuBLAS.
- **SDPA flash→math is the dominant regression:** on the O(T²) high-res block the math backend is **9.1×
  slower** than flash (447 → **4063 µs**), 6.0× at C384/T256, 3.4–3.7× at low-res — because it materializes
  the full T×T score matrix. This is now the single most expensive kernel in the UNet.
- **Linear** (qkv/proj): int8 is **2–3× slower than fp16 cuBLAS** at these channel counts (K below the int8
  crossover ~2048), so int8 linear is not used.

---

## 2. Kernel IO benchmark

Effective DRAM bandwidth = (bytes in + weights + bytes out) / time, vs the 696 GB/s peak.
![kernel IO](06_kernel_io.png) — data: [`kernel_conv_io.csv`](data/kernel_conv_io.csv), [`kernel_linear_io.csv`](data/kernel_linear_io.csv)

- **3×3 convs are mid-bandwidth** (65–170 GB/s) — not DRAM-saturated; quantization helps by cutting *compute*.
- **1×1 convs and the qkv GEMM are memory-bound and near-efficient** (fp16 up to 475 GB/s ≈ 68% of peak); int8/int4
  move fewer bytes at lower effective bandwidth, so they don't win there.
- **Math attention is extremely IO-heavy**: the T=1024 block writes+reads a ~512 MB score matrix per call —
  the reason SDPA math is 9× slower and pipeline peak memory jumps ~0.9 GiB (§5).

---

## 3. Kernel profile (per-operation GPU time, by mode)

![kernel profile](03_kernel_profile.png) — data: [`data/kernel_profile.csv`](data/kernel_profile.csv)

| bucket | fp16 | int8 base | int8 modiff | int4 base | int4 modiff |
|---|--:|--:|--:|--:|--:|
| conv (GEMM) | 13.42 | 9.51 | 11.79 | **7.35** | **7.41** |
| GEMM (qkv/proj + attn QKᵀ·AV) | 13.39 | 13.40 | 13.41 | 13.40 | 13.41 |
| attention (softmax + SDPA) | 11.38 | 11.37 | 11.39 | 11.37 | 11.38 |
| GroupNorm | 5.69 | 5.57 | 5.51 | 5.28 | 5.51 |
| conv store epilogue | 1.81 | 1.58 | 1.33 | 2.47 | 1.33 |
| quantize / MoDiff delta | 0 | 0.20 | **2.96** | 0.18 | **2.62** |
| elementwise / copy | 7.43 | 5.56 | 6.66 | 5.04 | 6.66 |
| upsample / concat | 1.04 | 1.29 | 1.03 | 1.28 | 1.03 |
| other | 0.30 | 0.27 | 2.55 | 0.27 | 2.55 |
| **GPU-busy total** | **54.45** | **48.75** | **56.63** | **46.65** | **51.89** |

- **Attention now dominates (~42% of the step).** With flash removed, the softmax bucket is **11.4 ms** and the
  QKᵀ/AV matmuls (~11.8 ms) live in the GEMM bucket (13.4 ms total; qkv/proj alone were ~1.6 ms). Math attention
  also inflates the elementwise bucket (score scaling + fp16↔fp32 casts: fp16 4.0 → 7.4 ms). Combined attention
  cost ≈ 23 ms/step, up from ~3.4 ms with flash.
- *Profile caveat:* with math SDPA the QKᵀ/AV matmuls are cuBLAS GEMMs **indistinguishable by kernel name from
  qkv/proj**, so they merge into the GEMM bucket — which is why that bucket ballooned from 1.6 to 13.4 ms.
- **Quantization still only moves the conv bucket** (13.4 → 9.5 int8 → 7.4 int4). GroupNorm (~5.5 ms) + the
  attention buckets are dtype-invariant — now an even larger share, so quantization's relative win shrinks.
- **MoDiff adds ~2.6–3.0 ms `quantize / MoDiff delta` + ~2.3 ms `other`** — the temporal machinery.

---

## 4. Pipeline speed benchmark

![pipeline speed](01_pipeline_speed.png) — data: [`data/pipeline_speed.csv`](data/pipeline_speed.csv)

| mode | wall med | wall min | stdev | GPU-busy | wall speedup vs fp16 |
|---|--:|--:|--:|--:|--:|
| fp16 | 55.93 | 55.83 | 0.08 | 54.45 | 1.00× |
| int8 base | 49.99 | 49.76 | 0.10 | 48.75 | 1.12× |
| int8 modiff | 58.03 | 57.97 | 0.04 | 56.63 | 0.96× |
| **int4 base** | **47.64** | **47.58** | 0.03 | **46.65** | **1.17×** |
| int4 modiff | 53.23 | 53.19 | 0.06 | 51.89 | 1.05× |

- **`int4 base` remains the fastest mode (1.17× vs fp16)**, `int8 base` next (1.12×). But the quantization
  speedup **shrank vs the flash config (1.25× / 1.18×)** — because the now-huge, dtype-invariant attention
  (~23 ms) dilutes the conv savings further (Amdahl: the quantizable fraction dropped from ~40% to ~28% of
  the step).
- **MoDiff modes:** int8 modiff is slower than fp16 (0.96×); int4 modiff just above (1.05×). The temporal
  delta machinery (§3) is real, verified GPU work — an accuracy mechanism, not a speed one.
- **Every mode is ~1.6–1.85× slower than with flash** (int4 base 25.7 → 47.6 ms): the cost of removing flash.

---

## 5. Pipeline IO benchmark

![pipeline io](02_pipeline_io.png) — data: [`data/pipeline_io.csv`](data/pipeline_io.csv)

| mode | peak allocated MiB | peak reserved MiB | MoDiff cache MiB |
|---|--:|--:|--:|
| fp16 | 4365 | 4990 | 0 |
| int8 base | 4551 | 4834 | 0 |
| int8 modiff | 4957 | 5536 | **634** |
| int4 base | **4296** | 4668 | 0 |
| int4 modiff | 4705 | 5320 | **634** |

- **Peak memory is up ~0.9 GiB vs the flash config** (fp16 3404 → 4365 MiB): math attention materializes the
  `[N,heads,T,T]` score tensors (~512 MB for the T=1024 block alone). This is the memory price of removing flash.
- Base modes still sit near each other (activations dominate at batch 32); **int4 base has the smallest
  footprint** (4296 MiB). **MoDiff adds a fixed +634 MiB temporal cache** on top.

---

## Bottom line (flash-removed / math-attention config)

- **Speed:** `int4 base` is fastest (**1.17× wall vs fp16**), `int8 base` next (1.12×). Removing flash made
  **every mode ~1.6–1.85× slower** and shrank the quantization speedups (attention is now ~42% of the step and
  dtype-invariant, so Amdahl bites harder).
- **Attention is now the dominant cost (~23 ms/step, up from ~3.4 with flash)** — the math backend's T×T score
  materialization on the high-res block (9.1× slower than flash) is the single biggest kernel.
- **int4 vs int8:** int4 wins on large-channel 3×3 convs (aggregate conv 7.4 vs 9.5 ms), loses on ≤192-channel /
  1×1 — net faster.
- **MoDiff:** slower than fp16 (temporal delta machinery + 634 MiB cache) — accuracy, not speed.
- **IO:** compute-bound 3×3 convs are where precision helps; math attention is now the heaviest IO consumer
  (+0.9 GiB peak). If speed/memory matter, flash is strongly preferable — this config trades ~1.7× speed and
  ~0.9 GiB for a non-flash (regular) attention core.

*Scripts (in [`scripts/`](scripts/)): `pipeline.py`, `kernel.py`, `mkplots.py`. Re-run with
`PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass`. Attention: math SDPA backend
(flash disabled) via `TokenMajorAttentionBlock`; fused GN→qkv on all modes.*
