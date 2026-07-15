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
**stdev is 0.02–0.20 ms/step (<0.4%)** for every mode. GPU-busy = `torch.profiler` device self-time.

**Speed is measured before the profiler runs** (12 timed runs first, then a separate 3-run
`torch.profiler` pass for the bucket breakdown), so profiling does not touch the reported wall time — see
the profiler-effect check in §4. **Total IO usage** (§5) is computed *analytically* (`scripts/io_analytic.py`):
the sum over every conv / qkv-proj linear / attention-SDPA op of (bytes_in + bytes_weight + bytes_out) at each
op's real operand dtype — the same DRAM-bytes model as the §2 kernel-IO section. (The CUDA allocator's
`allocated_bytes.all.allocated` counter was tried first and rejected: quantized convs keep fp16 outputs, so
allocator traffic barely moves with dtype and misrepresents the quantization effect.)

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
| conv (GEMM) | 27.31 | 13.71 | 9.76 | 12.10 | **7.61** | **7.65** |
| GEMM (qkv/proj + attn QKᵀ·AV) | 30.83 | 13.38 | 13.37 | 13.37 | 13.37 | 13.37 |
| attention (softmax + SDPA) | 22.34 | 11.38 | 11.37 | 11.39 | 11.37 | 11.40 |
| GroupNorm | 7.54 | 5.94 | 5.80 | 5.65 | 5.47 | 5.67 |
| conv store epilogue | 0 | 1.82 | 1.60 | 1.35 | 2.48 | 1.35 |
| quantize / MoDiff delta | 0 | 0 | 0.20 | **2.98** | 0.18 | **2.62** |
| elementwise / copy | 11.98 | 7.57 | 5.65 | 6.73 | 5.12 | 6.73 |
| upsample / concat | 1.78 | 1.08 | 1.32 | 1.06 | 1.31 | 1.06 |
| other | 0.34 | 0.30 | 0.28 | 2.57 | 0.28 | 2.56 |
| **GPU-busy total** | **102.11** | **55.18** | **49.35** | **57.19** | **47.19** | **52.41** |

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
| fp32 | 102.92 | 102.66 | 0.20 | 102.11 | 0.55× | 1.00× |
| fp16 | 56.20 | 56.11 | 0.06 | 55.18 | 1.00× | 1.83× |
| int8 base | 50.43 | 50.34 | 0.07 | 49.36 | 1.11× | 2.04× |
| int8 modiff | 58.48 | 58.44 | 0.07 | 57.19 | 0.96× | 1.76× |
| **int4 base** | **48.10** | **48.07** | 0.02 | **47.19** | **1.17×** | **2.14×** |
| int4 modiff | 53.60 | 53.56 | 0.11 | 52.41 | 1.05× | 1.92× |

- **`int4 base` is the fastest mode (1.17× vs fp16, 2.14× vs fp32)**, `int8 base` next (1.11× / 2.04×).
- **fp16 is 1.83× faster than fp32** — the plain-precision headline. Everything below fp16 is the
  quantization win on top of that.
- **MoDiff modes:** int8 modiff is slower than fp16 (0.96×); int4 modiff just above (1.05×). The temporal
  delta machinery (§3) is real, verified GPU work — an accuracy mechanism, not a speed one.

#### Why the int8/int4 base speedup is "only" 1.1–1.17× (and it is *not* a measurement artifact)

The speedup looks modest next to the raw TOPS ratio (int8 2×, int4 4× the fp16 tensor-core rate). Three
real reasons, none of them the profiler:

1. **Amdahl — quantization only speeds the conv bucket.** The conv (GEMM) bucket is ~13.7 ms of the 55 ms
   fp16 step (~25%); everything else — attention (math SDPA + QKᵀ/AV GEMMs ≈ 23 ms), GroupNorm (~5.5 ms),
   qkv/proj, elementwise — is **dtype-invariant**. Even a *free* conv would cap the step at 55 − 13.7 ≈
   42 ms, i.e. a **1.32× ceiling**. int4 base reaches 47.2 ms GPU-busy, already ~80% of the way to that
   ceiling; there is very little headroom left to win.
2. **These convs are partly memory-bound, not compute-bound.** Kernel IO (§2) shows the 3×3 convs at
   64–163 GB/s — well under the 696 GB/s peak but not compute-saturated either, so int8/int4 realise only a
   fraction of their compute-ratio advantage. int4 additionally pays a weight-unpack overhead and loses on
   ≤192-channel / 1×1 convs (§1), which eats into its aggregate win.
3. **The profiler does not affect the measured speed.** Speed is timed *before* any profiling. A direct
   check (identical warmup, batch 32) confirms it:

   | mode | no profiler | after a profiler pass | inside a live profiler |
   |---|--:|--:|--:|
   | fp16 | 55.89 | 55.92 | 56.68 |
   | int8 base | 50.32 | 50.34 | 50.82 |
   | int4 base | 48.09 | 48.16 | 48.65 |

   "No profiler" and "after a profiler pass" (what the report actually measures) are identical to within
   ±0.07 ms (noise). Even timing *inside* a live profiler — which the report never does — adds only ~1%.
   The 1.11×/1.17× speedups are genuine and Amdahl-bound, not depressed by instrumentation.

---

## 5. Pipeline total IO usage

**Total IO usage** = analytical DRAM bytes moved per DDIM step = Σ over every conv / qkv-proj linear /
attention-SDPA op of (bytes_in + bytes_weight + bytes_out) at each op's real operand dtype (see Methodology).
IO usage depends on *precision*, not on the base-vs-MoDiff temporal scheme, so it is reported per precision.
![pipeline io](02_pipeline_io.png) — data: [`data/pipeline_io_analytic.csv`](data/pipeline_io_analytic.csv)

| precision | conv MiB/step | qkv/proj linear MiB/step | attention SDPA MiB/step | **total MiB/step** | conv share | total vs fp16 |
|---|--:|--:|--:|--:|--:|--:|
| fp32 | 2298 | 1297 | 11734 | **15329** | 15% | 1.95× |
| fp16 | 1330 | 648 | 5867 | **7846** | 17% | 1.00× |
| int8 | **847** | 648 | 5867 | **7362** | 12% | 0.94× |
| int4 | **605** | 648 | 5867 | **7120** | 8% | 0.91× |

- **Conv IO drops exactly as quantization should:** int8 conv IO is **0.64×** and int4 **0.45×** of fp16
  (1330 → 847 → 605 MiB/step). Quantization is doing its job on the operands it touches (int8 activations/
  weights are 1 B, int4 0.5 B; the conv output stays fp16 by design, which is why it's 0.64×/0.45× rather
  than 0.5×/0.25×).
- **But the total barely moves (int8 0.94×, int4 0.91× of fp16)** because the **attention path carries no
  quantization / Q·DQ at all — it is fp16 in *every* mode.** Verified on the built int8/int4-baseline models:
  qkv and proj are plain fp16 `nn.Linear` (the 1×1 pointwise convs are explicitly skipped by
  `convert_model_to_optimized_int8(skip_pointwise=True)`, and int8 linear is not applied — it is 2–3× slower
  at these K, §1), and the SDPA Q / `[N,heads,T,T]` score matrix are fp16 at runtime. Since the attention
  score traffic is ~75% of DRAM IO and dtype-invariant, conv quantization can only shrink the ~17% conv
  slice — the same Amdahl wall that bounds the speedup (§4). **To cut attention IO you must quantize the
  attention core itself** (the qkv/proj GEMMs + the QKᵀ/AV matmuls the math backend now exposes) — the flagged
  next step, not something this pipeline does today.
- **This is why the earlier allocator-traffic metric looked flat** (int8/int4 ≈ fp16): it was dominated by the
  fp16 attention/activation volume and never credited the conv operand shrink. The analytical split above
  makes both facts explicit — conv IO *does* fall, the total is attention-bound.
- **fp32 moves ~1.95× the fp16 total** (15.3 vs 7.8 GiB/step): every operand, including the huge attention
  scores, doubles to 4 B.

**Peak memory footprint** (measured, per mode) — a separate quantity from total IO:
[`data/pipeline_io.csv`](data/pipeline_io.csv)

| mode | peak allocated MiB | peak reserved MiB | MoDiff cache MiB |
|---|--:|--:|--:|
| fp32 | 4920 | 6084 | 0 |
| fp16 | 4369 | 4964 | 0 |
| int8 base | 4551 | 4834 | 0 |
| int8 modiff | 4958 | 5524 | **634** |
| int4 base | **4295** | 4668 | 0 |
| int4 modiff | 4707 | 5356 | **634** |

Base modes sit near each other (fp16 activations dominate the live footprint at batch 32); **int4 base has the
smallest peak** (4295 MiB), and MoDiff adds a fixed **+634 MiB** `a_hat`/`o_hat` temporal cache.

---

## Bottom line

- **Speed:** `int4 base` is fastest (**1.17× wall vs fp16, 2.14× vs fp32**), `int8 base` next (1.11× / 2.04×).
  fp16 alone is **1.83× faster than fp32**. The 1.1–1.17× quantization win over fp16 is **Amdahl-bounded, not
  a profiler artifact** (§4): quantization only speeds the ~25% conv bucket — a free conv would cap the step at
  a 1.32× ceiling — and these convs are partly memory-bound.
- **Attention is the dominant cost (~23 ms/step)** — the math SDPA backend's T×T score materialization on the
  high-res block (4060 µs) is the single biggest kernel. It is kept on the math backend by design so the
  QKᵀ/AV products stay as interceptable/quantizable cuBLAS GEMMs.
- **int4 vs int8:** int4 wins on large-channel 3×3 convs (aggregate conv 7.6 vs 9.8 ms), loses on ≤192-channel /
  1×1 — net faster.
- **MoDiff:** slower than fp16 (temporal delta machinery + 634 MiB cache) — accuracy, not speed.
- **Total IO usage:** conv IO drops as expected under quantization (int8 0.64×, int4 0.45× of fp16), but the
  *total* only falls to 0.94×/0.91× because the fp16 attention-score traffic (~75% of DRAM IO) is
  dtype-invariant — the same Amdahl wall as the speedup. fp32 moves ~1.95× the fp16 total.

*Scripts (in [`scripts/`](scripts/)): `pipeline.py`, `kernel.py`, `mkplots.py`. Re-run with
`PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass`. Attention: math SDPA backend via
`TokenMajorAttentionBlock`; fused GN→qkv on all modes.*
