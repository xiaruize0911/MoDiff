# MoDiff comprehensive benchmark — kernels & pipeline (2026-07-15)

Kernel-level and pipeline-level speed + IO + profile of the LSUN-churches latent-diffusion UNet.
**Hardware:** NVIDIA A40 (Ampere sm_86; fp16 149.7 TFLOP/s, int8 299 TOP/s, int4 599 TOP/s, DRAM 696 GB/s).
**Config:** batch 32, DDIM, per-step numbers. Raw data in [`data/`](data/), scripts in [`scripts/`](scripts/).

**The 5 pipeline modes:** `fp16`, `int8 base` (`int8_baseline`), `int8 modiff` (`int8`, temporal caching),
`int4 base` (`int4_baseline`), `int4 modiff` (`int4`). "base" = quantized kernels, no temporal caching;
"modiff" = MoDiff error-compensated temporal caching (`o_hat`/`a_hat` deltas) across DDIM steps.

### Methodology (measurement rigor)

The A40 **idles at 210 MHz and boosts to 1740 MHz**, and clock-locking is not permitted in this container,
so **warmup dominates measurement quality**. All numbers here use:
- **Kernels:** 30 warmup iters + 60 timed iters, CUDA-event timed in a tight loop (sustains boost clock).
- **Pipeline:** ≥6 s of sustained `sample()` warmup (ramps and holds boost), then 12 back-to-back timed runs;
  we report **median, min, and stdev**. Measured stdev is **0.09–0.26 ms/step** (<1%), i.e. the numbers are
  clock-stable, not noise. GPU-busy = `torch.profiler` device self-time.

> ⚠️ An earlier draft used only 2 warmup `sample()` calls and measured partly during clock ramp; that
> **depressed the int4 and MoDiff modes by 2–5 ms/step** (e.g. int4 base read 28.5 ms instead of the true
> 25.7 ms). The numbers below are the re-measured, warmed, low-variance values.

---

## 1. Kernel speed benchmark

### Conv (top shapes by cost) — ![conv speed](04_kernel_conv_speed.png)

int8 conv beats fp16 cuDNN on the compute-heavy **3×3** convs (e.g. `384→384 3×3 16²`: fp16 197 → int8 104 µs;
`576→192 3×3 32²`: 580 → 318 µs). int4 goes further **only when channels are large enough to amortise its
weight-unpack overhead** — see the base-vs-MoDiff figure below and the note on int4-vs-int8.
Full table: [`data/kernel_conv_speed.csv`](data/kernel_conv_speed.csv).

### Conv base vs MoDiff (and why int4 ≷ int8) — ![conv modiff](05_kernel_conv_modiff.png)

**MoDiff's temporal path** (`conv2d_int{8,4}_fprop_o_hat` + `step1` delta-quantize + `sub`/`accumulate`)
**adds work and skips no convolution**, so it is **~2–2.5× slower per conv** than the base quantized kernel
(e.g. `384→384 3×3 32²`: int8 base 425 → int8 MoDiff 1041 µs). MoDiff buys temporal *accuracy* on correlated
denoising sequences, not kernel speed.

**Why int4 base is sometimes slower, sometimes faster than int8 base — it is shape-dependent:**
- **int4 WINS on large-channel 3×3 convs** where its 2× arithmetic advantage is realised:
  `384→384 3×3 32²` (int8 425 → int4 390 µs), `768→384 3×3 16²` (203 → 156 µs).
- **int4 LOSES at low channel counts and on 1×1 / tiny-spatial convs**: `192→192 3×3 32²` (int8 116 → int4 187 µs),
  `384→192 1×1 32²` (int8 63 → int4 147 µs). At ≤192 channels the **int4 weight-unpack + the 4-bit GEMM's lower
  tensor-core occupancy are not amortised**, so the fixed overhead dominates a small problem.
- **In aggregate across the whole UNet int4 is still faster** (profile §3: conv bucket 7.3 ms int4 vs 9.4 ms
  int8) because the expensive convs are the large-channel 3×3s where int4 wins.
- *(The earlier draft's plot showed only the 6 highest-parameter-count convs, which were all tiny-spatial
  768/1536-channel 2²/4² convs — the exact regime where int4 loses — making int4 look uniformly slower. This
  version selects the 6 highest-**FLOP** convs, the real cost drivers.)*

### Linear & attention — ![attn](07_kernel_attn.png)

- **Linear** (qkv/proj, [`data/kernel_linear_speed.csv`](data/kernel_linear_speed.csv)): int8 is **2–3× slower
  than fp16 cuBLAS** at these channel counts (192–768) — K is below the int8 tensor-core crossover (~2048), so
  int8 linear is *not* used for attention.
- **GN→qkv fused** (custom CUTLASS per-sample mainloop fusion, `fused_gn_qkv`): **1.13×** on C192/T1024,
  **1.26×** on C384/T256 vs GroupNorm+cuBLAS; falls back where T is not a multiple of 128.
- **flash SDPA** dominates the high-res block (C192/T1024 = 446 µs, O(T²)).

---

## 2. Kernel IO benchmark

Effective DRAM bandwidth = (bytes read in + weights + bytes written out) / measured time, vs the 696 GB/s peak.

![kernel IO](06_kernel_io.png) — data: [`data/kernel_conv_io.csv`](data/kernel_conv_io.csv), [`data/kernel_linear_io.csv`](data/kernel_linear_io.csv)

- **3×3 convs are mid-bandwidth** (65–170 GB/s effective) — not DRAM-saturated; quantization helps by cutting
  *compute*, not traffic.
- **1×1 convs and the qkv GEMM are memory-bound and near-efficient**: fp16 1×1 convs reach **320–475 GB/s**
  (up to 68% of peak), the fp16 qkv GEMM **474 GB/s**. int8/int4 move fewer bytes but at lower effective
  bandwidth on these shapes — which is exactly why they don't win on 1×1s.
- Takeaway: the quantizable win is on the **compute-bound 3×3 convs**; the memory-bound 1×1s and GEMMs are
  already near their roofline.

---

## 3. Kernel profile (per-operation GPU time, by mode)

![kernel profile](03_kernel_profile.png) — data: [`data/kernel_profile.csv`](data/kernel_profile.csv)

| bucket | fp16 | int8 base | int8 modiff | int4 base | int4 modiff |
|---|--:|--:|--:|--:|--:|
| conv (GEMM) | 13.30 | 9.43 | 11.70 | **7.28** | **7.36** |
| GroupNorm | 5.64 | 5.52 | 5.47 | 5.23 | 5.47 |
| attention (flash SDPA) | 3.37 | 3.39 | 3.39 | 3.39 | 3.37 |
| linear/qkv/proj (GEMM) | 1.56 | 1.56 | 1.57 | 1.56 | 1.57 |
| conv store epilogue | 1.81 | 1.57 | 1.32 | 2.47 | 1.33 |
| quantize / MoDiff delta | 0 | 0.20 | **2.96** | 0.18 | **2.61** |
| elementwise / copy | 4.04 | 2.18 | 3.28 | 1.65 | 3.28 |
| upsample / concat | 1.03 | 1.28 | 1.03 | 1.27 | 1.03 |
| other | 0.29 | 0.27 | 2.55 | 0.27 | 2.54 |
| **GPU-busy total** | **31.05** | **25.41** | **33.27** | **23.30** | **28.56** |

- **Quantization attacks the conv bucket** (13.3 → 9.4 int8 → 7.3 int4) — the only bucket it meaningfully moves.
- **GroupNorm (~5.5 ms) + flash SDPA (~3.4 ms) are dtype-invariant** memory-bound fp16 work int8/int4 cannot
  touch (~28% of the step) — the standing bottleneck.
- **MoDiff adds ~2.6–3.0 ms of `quantize / MoDiff delta` + ~2.3 ms of `other`** (temporal sub/accumulate/
  delta-quantize) — why MoDiff is slower on GPU time than its base counterpart despite identical conv work.

---

## 4. Pipeline speed benchmark

![pipeline speed](01_pipeline_speed.png) — data: [`data/pipeline_speed.csv`](data/pipeline_speed.csv)

| mode | wall med | wall min | stdev | GPU-busy | wall speedup vs fp16 | GPU-busy speedup |
|---|--:|--:|--:|--:|--:|--:|
| fp16 | 32.14 | 32.03 | 0.09 | 31.05 | 1.00× | 1.00× |
| int8 base | 27.24 | 26.91 | 0.24 | 25.41 | 1.18× | 1.22× |
| int8 modiff | 36.00 | 35.68 | 0.26 | 33.27 | 0.89× | 0.93× |
| **int4 base** | **25.75** | **25.54** | 0.23 | **23.30** | **1.25×** | **1.33×** |
| int4 modiff | 33.41 | 33.07 | 0.21 | 28.56 | 0.96× | 1.09× |

- **`int4 base` is the fastest mode with clean measurement — 25.75 ms/step wall (1.25× vs fp16)**, best GPU-busy
  (1.33×). `int8 base` is close behind at 1.18× wall. (Before proper warmup, int4's overhead tail read larger
  and int8 appeared to win — the corrected measurement reverses that.)
- **Why the speedup is "only" ~1.2–1.3×, not the 2×/4× of the raw arithmetic ratios:** Amdahl. Only the conv
  bucket is quantizable (~40% of fp16 GPU time), and even that is ~1.4–1.8× faster; GroupNorm + SDPA +
  elementwise (~half the step) are dtype-invariant. This is the fundamental ceiling, quantified in §3 and in
  `../profiling_report_2026-07-14/DIFFUSION_UNET_BOTTLENECK.md`.
- **MoDiff modes are slower than fp16 at the wall (0.89×, 0.96×)** — this is a real, verified result (the
  profile shows the +2.6–3.0 ms delta-quantize bucket is genuinely executing, not a fallback). MoDiff trades
  speed for temporal accuracy, which a pure speed benchmark does not reward. int4 modiff (0.96× wall / **1.09×
  GPU-busy**) is the least-penalised because its base convs are cheapest.

---

## 5. Pipeline IO benchmark

![pipeline io](02_pipeline_io.png) — data: [`data/pipeline_io.csv`](data/pipeline_io.csv)

| mode | peak allocated MiB | peak reserved MiB | MoDiff cache MiB |
|---|--:|--:|--:|
| fp16 | 3404 | 3688 | 0 |
| int8 base | 3603 | 3686 | 0 |
| int8 modiff | 4058 | 4778 | **634** |
| int4 base | **3364** | 3418 | 0 |
| int4 modiff | 3797 | 4526 | **634** |

- **Base modes sit near fp16's footprint** (~3.4–3.6 GiB) — quantized *weights* are smaller but activations
  dominate at batch 32, so peak memory barely moves. **int4 base has the smallest footprint** (3364 MiB).
- **MoDiff modes carry a fixed +634 MiB temporal cache** (`a_hat` + `o_hat` per quantized conv) plus a larger
  reserved pool (up to +1.1 GiB) — the memory price of the accuracy mechanism.

---

## Bottom line

- **Speed:** with clean, warmed, low-variance measurement, **`int4 base` is fastest (1.25× wall / 1.33×
  GPU-busy vs fp16)**, `int8 base` close behind (1.18× / 1.22×). The ~1.2–1.3× ceiling is Amdahl: only the 3×3
  conv bucket is quantizable; GroupNorm + SDPA (~28%) are dtype-invariant.
- **int4 vs int8:** int4 wins on the large-channel 3×3 convs that dominate cost (aggregate conv time 7.3 vs
  9.4 ms) but loses on ≤192-channel and 1×1 convs (unpack overhead) — net win for int4.
- **MoDiff:** slower than fp16 (temporal delta machinery adds ~5–8 ms GPU work + a 634 MiB cache) — an
  accuracy mechanism, not a speed one. Verified real via the profile, not a measurement artifact.
- **IO:** compute-bound 3×3 convs are where precision helps; 1×1 convs and the qkv GEMM are already near their
  memory roofline. MoDiff's price is +634 MiB.

*Scripts (in [`scripts/`](scripts/)): `pipeline.py`, `kernel.py`, `mkplots.py`. Re-run with
`PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass`. Methodology: heavy warmup + multi-run
median/min/stdev (see top); measured pipeline stdev < 1%.*
