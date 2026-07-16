# MoDiff comprehensive benchmark — kernels & pipeline (2026-07-15, re-run 2026-07-16)

Kernel-level and pipeline-level speed + IO + profile of the LSUN-churches latent-diffusion UNet.
**Hardware:** NVIDIA A40 (Ampere sm_86; fp16 149.7 TFLOP/s, int8 299 TOP/s, int4 599 TOP/s, DRAM 696 GB/s).
**Config:** batch 32, DDIM, per-step numbers. Raw data in [`data/`](data/), scripts in [`scripts/`](scripts/).
All numbers below are from a full re-run (rebuild + every script) on 2026-07-16.

> ### ⚙️ Config
> - **Attention** runs on the math (non-flash) SDPA backend in every attention block (the QKᵀ/AV products stay
>   as interceptable cuBLAS GEMMs), with the custom **fused GN→qkv** kernel on for all modes.
> - **fp32** (full precision, autocast off) is the baseline mode alongside fp16.
> - Two **opt-in quantization extensions** (§6 attention-score int8 flash, §7 Linear W8A8/W4A4) are measured
>   separately; the default 6-mode pipeline (§1–§5) has them off.

**The 6 pipeline modes:** `fp32`, `fp16`, `int8 base` (`int8_baseline`), `int8 modiff` (`int8`, temporal
caching), `int4 base` (`int4_baseline`), `int4 modiff` (`int4`). "base" = quantized kernels, no temporal
caching; "modiff" = MoDiff error-compensated temporal caching (`o_hat`/`a_hat` deltas) across DDIM steps.

### Methodology

The A40 idles at 210 MHz and boosts to 1740 MHz, and clock-locking is not permitted here, so **warmup
dominates measurement quality**. Kernels: 30 warmup + 60 timed iters. Pipeline: **≥6 s sustained `sample()`
warmup + 12 back-to-back timed runs** (median/min/stdev). **GPU-busy** = `torch.profiler` device self-time
(clock-throttle-robust); it is the primary speed metric here because A40 wall-clock still shows occasional
clock-ramp noise (this run, `int4 base` wall stdev was 1.9 ms — median 52.3 but min 48.6; its GPU-busy 47.0 is
clean and used below). Speed is measured before the profiler runs, so profiling never inflates it.
**Total IO usage** (§5) is analytical DRAM bytes (`scripts/io_analytic.py`), same model as the §2 kernel-IO.

---

## 1. Kernel speed benchmark

### Conv (top shapes by cost) — ![conv speed](04_kernel_conv_speed.png)

int8 conv beats fp16 cuDNN on compute-heavy **3×3** convs (`384→384 3×3 16²`: fp16 196 → int8 104 µs;
`384→384 3×3 32²`: 759 → 447 µs); int4 goes further only when channels are large enough to amortise its
weight-unpack (`768→384 3×3 16²`: 380 → int8 206 → **int4 158**). Full table:
[`data/kernel_conv_speed.csv`](data/kernel_conv_speed.csv).

### Conv base vs MoDiff (and why int4 ≷ int8) — ![conv modiff](05_kernel_conv_modiff.png)

**MoDiff's temporal path** adds work and skips no convolution, so it is **~2–2.5× slower per conv**
(`384→384 3×3 32²`: int8 base 447 → int8 MoDiff 1052 µs). It buys temporal *accuracy*, not speed.
**int4-vs-int8 base is shape-dependent:** int4 wins on large-channel 3×3 (`384→384 3×3 32²` 447→402 µs;
`768→384 3×3 16²` 206→158 µs) but loses at ≤192-channel (`192→192 3×3 32²` 117→190 µs). In aggregate int4 is
still faster (profile §3: conv bucket 7.5 vs 9.7 ms).

### Attention (GN→qkv fusion + math SDPA) — ![attn](07_kernel_attn.png)

- **GN→qkv fused** (custom CUTLASS): **1.12×** (C192/T1024, 236 → 211 µs), **1.26×** (C384/T256, 166 → 133 µs)
  vs GroupNorm+cuBLAS.
- **SDPA (math backend)** is the single most expensive kernel: on the O(T²) C192/T1024 block it materializes
  the full `[N,heads,T,T]` score matrix and costs **4059 µs** (360 µs at C384/T256). §6 quantizes this path.
- **Linear** (qkv/proj): int8 is 2–3× slower than fp16 cuBLAS at these K in the naive path (see §7 for the
  tuned kernel), so int8 linear is off by default here.

---

## 2. Kernel IO benchmark

Effective DRAM bandwidth = (bytes in + weights + bytes out) / time, vs 696 GB/s peak.
![kernel IO](06_kernel_io.png) — data: [`kernel_conv_io.csv`](data/kernel_conv_io.csv), [`kernel_linear_io.csv`](data/kernel_linear_io.csv)

- **3×3 convs are mid-bandwidth** (~64–163 GB/s) — not DRAM-saturated; quantization helps by cutting *compute*.
- **1×1 convs and the qkv GEMM are memory-bound and near-efficient** (fp16 up to ~473 GB/s ≈ 68% of peak).
- **Math attention is extremely IO-heavy**: the T=1024 block reads+writes a ~512 MB score matrix per call —
  the reason SDPA dominates and pipeline peak sits ~0.9 GiB above a flash-based config.

---

## 3. Kernel profile (per-operation GPU time, by mode) — ![kernel profile](03_kernel_profile.png)

data: [`data/kernel_profile.csv`](data/kernel_profile.csv). GPU-busy ms/step.

| bucket | fp32 | fp16 | int8 base | int8 modiff | int4 base | int4 modiff |
|---|--:|--:|--:|--:|--:|--:|
| conv (GEMM) | 27.13 | 13.65 | 9.69 | 12.00 | **7.48** | **7.62** |
| GEMM (qkv/proj + attn QKᵀ·AV) | 30.80 | 13.41 | 13.42 | 13.40 | 13.40 | 13.41 |
| attention (softmax + SDPA) | 22.34 | 11.40 | 11.37 | 11.38 | 11.37 | 11.41 |
| GroupNorm | 7.40 | 5.83 | 5.68 | 5.55 | 5.40 | 5.59 |
| conv store epilogue | 0 | 1.82 | 1.59 | 1.35 | 2.48 | 1.35 |
| quantize / MoDiff delta | 0 | 0 | 0.20 | **2.98** | 0.18 | **2.62** |
| elementwise / copy | 11.97 | 7.56 | 5.66 | 6.73 | 5.09 | 6.74 |
| upsample / concat | 1.78 | 1.07 | 1.31 | 1.05 | 1.30 | 1.06 |
| other | 0.34 | 0.30 | 0.28 | 2.57 | 0.27 | 2.57 |
| **GPU-busy total** | **101.76** | **55.04** | **49.20** | **56.99** | **46.99** | **52.37** |

- **fp32 is ~1.85× the fp16 GPU-busy** — every compute bucket ~doubles (fp32 tensor-core throughput).
- **Attention dominates the fp16/quant modes (~42% of the step)** — softmax 11.4 ms + the QKᵀ/AV matmuls
  (~11.8 ms) in the 13.4 ms GEMM bucket. Combined attention ≈ 23 ms/step.
- *Profile caveat:* with math SDPA the QKᵀ/AV matmuls are cuBLAS GEMMs indistinguishable by name from qkv/proj,
  so they merge into the GEMM bucket (13.4 ms, not ~1.6 ms).
- **Quantization only moves the conv bucket** (13.7 → 9.7 int8 → 7.5 int4). GroupNorm + attention are
  dtype-invariant. **MoDiff adds ~2.6–3.0 ms delta + ~2.3 ms other.**

---

## 4. Pipeline speed benchmark — ![pipeline speed](01_pipeline_speed.png)

data: [`data/pipeline_speed.csv`](data/pipeline_speed.csv). Speedups from GPU-busy (throttle-robust).

| mode | wall med | wall min | GPU-busy | speedup vs fp16 | vs fp32 |
|---|--:|--:|--:|--:|--:|
| fp32 | 102.69 | 102.51 | 101.76 | 0.54× | 1.00× |
| fp16 | 55.97 | 55.86 | 55.04 | 1.00× | 1.85× |
| int8 base | 50.51 | 50.28 | 49.20 | 1.12× | 2.07× |
| int8 modiff | 58.69 | 58.55 | 56.99 | 0.97× | 1.79× |
| **int4 base** | 52.27† | 48.61 | **46.99** | **1.17×** | **2.17×** |
| int4 modiff | 53.58 | 53.50 | 52.37 | 1.05× | 1.94× |

†`int4 base` wall median was clock-throttled this run (stdev 1.9 ms); min 48.61 and GPU-busy 46.99 are the
reliable values (consistent with prior runs).

- **`int4 base` is the fastest mode (1.17× vs fp16, 2.17× vs fp32)**, `int8 base` next (1.12× / 2.07×). fp16
  alone is **1.85× faster than fp32**.
- The 1.1–1.17× quantization win over fp16 is **Amdahl-bounded**: quantization only speeds the ~25% conv
  bucket — a free conv caps the step at a ~1.32× ceiling — and these convs are partly memory-bound.
- **MoDiff modes:** int8 modiff 0.97×, int4 modiff 1.05× — the temporal delta machinery is real GPU work, an
  accuracy mechanism, not a speed one.

---

## 5. Pipeline total IO usage — ![pipeline io](02_pipeline_io.png)

**Total IO usage** = analytical DRAM bytes/step = Σ over conv / qkv-proj linear / attention-SDPA ops of
(in+weight+out) at each op's real operand dtype. Depends on *precision*, not the base/modiff scheme.
data: [`data/pipeline_io_analytic.csv`](data/pipeline_io_analytic.csv)

| precision | conv MiB | qkv/proj MiB | attention MiB | **total MiB** | total vs fp16 |
|---|--:|--:|--:|--:|--:|
| fp32 | 2298 | 1297 | 11734 | **15329** | 1.95× |
| fp16 | 1330 | 648 | 5867 | **7846** | 1.00× |
| int8 | **847** | 648 | 5867 | **7362** | 0.94× |
| int4 | **605** | 648 | 5867 | **7120** | 0.91× |

- **Conv IO drops as quantization should** (int8 0.64×, int4 0.45× of fp16) — but the **total barely moves**
  (0.94×/0.91×) because the **fp16 attention-score traffic (5867 MiB, ~75%) is dtype-invariant** and the
  qkv/proj linears stay fp16 by default. Same Amdahl wall as the speedup.

**Peak memory footprint** (measured, per mode) — [`data/pipeline_io.csv`](data/pipeline_io.csv):

| mode | peak allocated MiB | peak reserved MiB | MoDiff cache MiB |
|---|--:|--:|--:|
| fp32 | 4920 | 6084 | 0 |
| fp16 | 4369 | 4964 | 0 |
| int8 base | 4552 | 4834 | 0 |
| int8 modiff | 4958 | 5524 | **634** |
| int4 base | **4296** | 4668 | 0 |
| int4 modiff | 4705 | 5356 | **634** |

---

## 6. Quantized attention-score path (opt-in: `MODIFF_QUANT_ATTN=1`)

A fused, tensor-core **int8 flash-attention** kernel (`csrc/kernels/flash_attn_int8.cu`, `mma.m16n8k32.s8`)
replaces the math-SDPA score path and **never materializes the `[N,H,T,T]` score matrix**. Measured in
`int8_baseline` (batch 32) — [`data/attn_quant.csv`](data/attn_quant.csv):

| config | ms/step | peak MiB | latent rel-err |
|---|--:|--:|--:|
| fp16 attention (default) | 46.5 | 4550 | — |
| int8 flash, large-T only (default) | 50.4 | **3602 (−20.8%)** | 0.0033 |
| int8 flash, all attn blocks | 50.5 | 3603 (−20.8%) | 0.0033 |

- **A memory win (−21% peak), not a speed win** (0.92×): the −21% comes from avoiding the fp16 T×T score
  matrix (~512 MB on the C192/T1024 block, ~94% of the win) — so the default gates int8 flash to the large-T
  block; quantizing the tiny-T blocks adds nothing. Speed doesn't improve because attention is dtype-invariant
  and cuBLAS-backed SDPA is near-optimal. Details: [`../attention_quantization_plan.md`](../attention_quantization_plan.md).

## 7. Quantized Linear layers — W8A8 / W4A4 (opt-in: `MODIFF_QUANT_LINEAR=1`)

AWQ-referenced custom int8/int4 tensor-core GEMM (`csrc/kernels/gemm_wxax.cu`, `cp.async` pipeline +
shape-adaptive tiles) quantizes the Linear-equivalent layers (attention qkv/proj, ResBlock emb_layers,
time_embed) — weight+activation, static scales. **Exact vs AWQ**; beats fp16 on memory-bound shapes, 1.1–1.85×
of AWQ. Full analysis: [`../linear_quantization_results.md`](../linear_quantization_results.md).

End-to-end (batch 32, heavy-warmup; [`data/linear_quant_speed.csv`](data/linear_quant_speed.csv)):

| mode | fp16-lin ms | quant-lin ms | speed | peak MiB (off→on) | rel-err† |
|---|--:|--:|--:|--:|--:|
| int8 base | 50.17 | 54.97 | 0.913× | 4549 → 4464 | **0.007** |
| int8 modiff | 58.79 | 63.13 | 0.931× | 4969 → 4869 | 0.057 |
| int4 base | 48.45 | 51.75 | 0.936× | 4310 → 4193 | **0.228** |
| int4 modiff | 54.70 | 57.41 | 0.953× | 4717 → 4604 | **0.456** |

†rel-err vs same-mode fp16-linear (batch 16; batch-invariant).

- **e2e-neutral (0.91–0.95×), not a speed win** — the profiler shows the quantized-Linear work is only
  **~5 ms of the ~55 ms step (~9%)** (int GEMM ~4.3 ms + act-quant ~0.8 ms), dwarfed by the conv/attention
  GEMMs (~20 ms) and elementwise (~10 ms). The Linears are a small fraction of this conv-dominated UNet.
- **int8 Linear is quality-safe** (rel 0.007–0.057); **int4 is too lossy** (0.23–0.46). Memory win small
  (−84…−117 MiB). MoDiff temporal-delta on Linear activations was tried and reverted (diverges).

---

## Bottom line

- **Speed:** `int4 base` fastest (**1.17× vs fp16, 2.17× vs fp32**), `int8 base` next (1.12× / 2.07×); fp16 is
  **1.85× faster than fp32**. Quantization's win over fp16 is Amdahl-bounded (only the ~25% conv bucket).
- **Attention is the dominant cost (~23 ms/step)** — the math-SDPA T×T materialization (4059 µs) is the single
  biggest kernel.
- **int4 vs int8:** int4 wins large-channel 3×3 convs (aggregate conv 7.5 vs 9.7 ms), loses ≤192-channel — net faster.
- **MoDiff:** slower than fp16 (temporal delta + 634 MiB cache) — accuracy, not speed.
- **Total IO:** conv IO drops (int8 0.64×, int4 0.45×) but total only 0.94×/0.91× — dtype-invariant fp16
  attention traffic (~75%) dominates.
- **Opt-in quantization (§6, §7):** attention-score int8 flash = **−21% peak memory**, not speed; Linear
  W8A8/W4A4 = **e2e-neutral** (~0.91–0.95×, ~9% of the step), int8 quality-safe / int4 too lossy. Recurring
  theme: **the compute lives in the convs**, so quantizing the non-conv ops buys memory/footprint, not
  step-time, for this UNet.

*Scripts (in [`scripts/`](scripts/)): `pipeline.py`, `kernel.py`, `io_analytic.py`, `linear_quant.py`,
`attn_quant.py`, `mkplots.py`. Re-run with `PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass`
(`pip install ninja` first). Default pipeline: math SDPA via `TokenMajorAttentionBlock`, fused GN→qkv on all
modes; §6/§7 quantization opt-in via the env flags above.*
