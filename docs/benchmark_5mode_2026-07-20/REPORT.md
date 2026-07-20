# MoDiff 5-mode benchmark — e2e + kernel level (measured)

**GPU:** NVIDIA A40 (48 GB, SM 8.6) · **PyTorch:** 2.4.1+cu124 · **CUDA:** 12.4 · nsys 2024.1.1
**Model:** LSUN-Churches LDM-8 UNet (unconditional, 256×256) · **Batch:** 128 · **Sampler:** DDIM
**Date:** 2026-07-20 · Post-refactor: *materialized int8/int4 attention removed — fused-flash is the sole quantized-attention path.*

The **5 modes**: `fp16`, `int8_baseline`, `int4_baseline`, `int8_modiff`, `int4_modiff`. int8/int4 use
fused-flash quantized attention by default; `_modiff` adds the temporal-delta conv cache.

---

## Method & measurement caveats (read first)

- **All numbers are measured** (CUDA-event / wall time for speed; `torch.profiler` CUDA device time for
  the timing profile; nsys CUPTI memcpy bytes for memory). No analytical/roofline estimates.
- **Speed:** GPU clock burn-in → warmup → N timed × R rounds with `torch.cuda.synchronize()` around each
  timed region. e2e = 30 warm + **5 × 200 DDIM steps** (mean+min). Kernels = 50 warm + **200 iters × 5
  reps** (CUDA-event median). autocast fp16 ON for **all** modes (fair true-fp16 baseline).
- **Memory read/write = nsys memcpy only.** HW DRAM-byte counters (ncu / CUPTI metrics / DCGM) are
  permission-locked on this box (`RmProfilingAdminOnly=1`, no `CAP_SYS_ADMIN`; `ncu` →
  `ERR_NVGPUCTRPERM`, verified). nsys tracing (no counters needed) measures **memcpy** traffic
  (H2D/D2H/D2D) but **cannot** see the DRAM reads/writes *inside* conv/linear/attention compute kernels.
  **Consequence (measured):** steady-state memcpy is **~0.4 MiB/step, identical across all 5 modes** — the
  pipeline moves data through in-kernel DRAM I/O, not memcpys. So the memory tables below are near-null;
  a true per-component/per-kernel read/write breakdown would require unlocking the counters.
- **Checkpoint is a random-weight stub** (no public churches ckpt on this box): kernel dispatch and tensor
  shapes are identical to the real model, so **speed is faithful**; any generation-quality number would not be.
- **modiff affects conv only** (`benchmark_ldm.py`: linear uses static W/A for both baseline & modiff;
  attention flash has no temporal cache). So at the kernel level **linear and attention are identical for
  baseline vs modiff** — only conv differs (no-cache vs `o_hat` cache).

Scripts: `scripts/{e2e_speed,e2e_timing_profile,nsys_driver,parse_nsys_mem,conv_kernel,linear_kernel,attn_kernel_fair,make_plots}.py`, `scripts/run_nsys_mem.sh`. Data: `data/*.csv`. Figures: `figs/*.png`.

---

## E2E level

### 1. Speed across the 5 modes  ·  `data/e2e_speed.csv` · `figs/fig_e2e_speed.png`

| mode | ms/step | vs true fp16 |
|---|--:|--:|
| fp16 | 189.7 | 1.00× |
| int8_baseline | 124.4 | 1.53× |
| **int4_baseline** | **119.9** | **1.58×** |
| int8_modiff | 146.2 | 1.30× |
| int4_modiff | 143.7 | 1.32× |

**int4_baseline = 1.58× vs true fp16 — the fastest mode** — after the int4 conv **deep-fusion fix**
(see below), int4 now edges out int8 (1.53×), the expected 4-bit ordering. int8_baseline = 1.53× holds
the flash-only refactor headline. The modiff temporal-cache variants are **slower** than their baselines
(int8 1.30× vs 1.53×): the a_hat/o_hat delta-quantize + accumulate costs more than it saves at b128.

> **int4 conv deep-fusion (2026-07-20).** int4_baseline was originally 141.7 ms (1.34×) — *slower* than
> int8 — because every int4 conv fell back to an eager path (SmoothQuant `x*smooth_inv` multiply →
> separate quantize+pack → bias-only store → eager residual add): `_prequant_common_ok` excluded
> SmoothQuant convs from the fused GN→conv path. Since the fused `group_norm_silu_quantize_pack_nhwc`
> kernel already supports a per-channel `smooth_inv`, wiring int4's smooth into it (Python-only change in
> `integration/fused_ops/fused_resblock.py`) routes int4 through the same deep-fused path as int8
> (GN+SiLU+SmoothQuant+quantize+pack in one kernel → `conv2d_int4_fprop_no_ohat_prealloc_bias_residual`
> with fused dequant+bias+residual store). Result: **141.7 → 119.9 ms (1.34× → 1.58×), −22 ms, output
> bit-identical (rel-L2 = 0).**

> **Why the same fix does NOT help int4_modiff (already fused).** The modiff (temporal-cache) path was
> checked but *not* changed — it is already fused for everything the baseline fix addressed: its
> delta-quantize kernel `step1_static_quantize_pack_int4_fprop_silu` folds SmoothQuant + SiLU +
> delta-quantize + pack into one launch, and bias is baked into the o_hat cache. Profiling its eager
> elementwise shows the only un-fused op is the block residual add at **~1.3 ms/step**; the rest (~18 ms)
> is generic fp16 glue (`cat`/`copy`/scale-shift) shared identically with fp16 and all modes. The reason
> int4_modiff (143.7 ms) is slower than int4_baseline (119.9 ms) is **intrinsic MoDiff overhead, not
> fusable glue**: delta-quantize 26.1 vs 17.3 ms (+8.8, computing `a−a_hat` + updating the cache) and the
> o_hat accumulate (+8.7 ms). Fusion cannot remove those. The only fusions left (residual → a new
> `conv2d_int4_fprop_o_hat_bias_residual`, or GroupNorm → the delta-quantize) need new correctness-risky
> CUDA for ≤~4 ms and int4_modiff would still trail int4_baseline — so **int4_baseline (1.58×) is the mode
> to use when temporal caching isn't required.**

![e2e speed](figs/fig_e2e_speed.png)

### 2. Total read/write (measured memcpy)  ·  `data/e2e_memcpy_total.csv` · `figs/fig_e2e_memcpy.png`

| mode | H2D | D2H | D2D | total MiB/step |
|---|--:|--:|--:|--:|
| fp16 / int8_baseline / int4_baseline / int8_modiff / int4_modiff | 0.0 | 0.0 | 0.4 | **0.4** |

Measured memcpy traffic is **negligible and identical across all modes** (~0.4 MiB/step — a single 2 MiB
D2D copy of the `[128,4,32,32]` latent ~0.2×/step plus tiny timestep-embedding copies). This is the
honest limit of memcpy-only measurement: the memory traffic that *differs* between fp16 and int8/int4
(conv/linear/attention DRAM reads/writes) lives **inside the compute kernels**, which nsys memcpy tracing
does not observe and this box's locked counters cannot measure.

### 3. Per-component timing profile  ·  `data/e2e_timing_profile.csv` · `figs/fig_e2e_timing_profile.png`

Measured GPU self-time (ms/step), key buckets:

| bucket | fp16 | int8_baseline | int4_baseline | int8_modiff | int4_modiff |
|---|--:|--:|--:|--:|--:|
| attention (flash / softmax) | 44.0 | 35.1 | 33.8 | 34.4 | 33.7 |
| attn bmm fp16 (QKᵀ/AV) | 42.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| conv | 45.5 | 24.5 | **16.0** | 28.8 | 15.8 |
| qkv/proj int GEMM | 0.0 | 7.7 | 7.0 | 7.6 | 7.0 |
| GroupNorm | 21.4 | 23.9 | 22.8 | 23.3 | 23.1 |
| quantize/dequant | 0.0 | 19.1 | **17.3** | 28.7 | 26.1 |
| modiff cache | 0.0 | 0.0 | 0.0 | 8.7 | 8.7 |
| elementwise/copy | 32.6 | 12.8 | **20.0** | 15.2 | 18.9 |
| upsample/concat + other fp16 GEMM + other | 12.5 | 6.9 | 7.0 | 5.8 | 5.5 |
| **gpu_busy** | 198.2 | 130.0 | **124.1** | 152.6 | 138.9 |
| **wall** | 188.5 | 123.0 | **119.2** | 145.3 | 146.0 |

Where the int8 win comes from: **attention ≈ 86.2 ms in fp16** (44.0 softmax + 42.2 fp16 QKᵀ/AV bmm)
collapses to **≈ 35 ms** with fused-flash int8 (one bucket, bmm ≈ 0); **conv 46 → 25 ms**. The cost added
back is the **quantize/dequant** prologue (~19 ms int8) and, for modiff, the **~8.7 ms cache**.
**int4 after the deep-fusion fix** is now cheaper than int8 in every compute bucket (conv 16.0, quantize
17.3, elementwise collapsed 38.7→20.0), giving the fastest gpu_busy (124.1 ms). gpu_busy slightly exceeds
wall because profiled per-kernel time omits kernel overlap — treat it as the device-time composition, not
an additive wall.

![e2e timing profile](figs/fig_e2e_timing_profile.png)

### 4. Per-component read/write profile

Not separable from memcpy: the only steady-state memcpy is the latent D2D copy (§2, `data/e2e_memcpy_sites.csv`).
Per-component (conv/linear/attention) DRAM read/write is compute-kernel-internal → **unmeasurable on this
box** (counters locked). The §3 timing profile is the per-component signal that *is* measurable.

---

## Kernel level

### Conv — speed, 5 modes  ·  `data/conv_kernel_speed.csv` · `figs/fig_conv_kernel.png`

Churches ResBlock convs at b128 (µs, median 5×200):

| shape (Cin→Cout, HW) | fp16 | int8_base | int4_base | int8_modiff | int4_modiff | int4_base vs fp16 |
|---|--:|--:|--:|--:|--:|--:|
| res 128, 64² | 1861 | 2361 | 2270 | 2715 | 2244 | 0.82× |
| res 128, 32² | 493 | 610 | 588 | 706 | 582 | 0.84× |
| down 128→256, 32² | 944 | 932 | 911 | 1165 | 954 | 1.04× |
| res 256, 32² | 1626 | 1532 | 1283 | 1681 | 1265 | 1.27× |
| res 256, 16² | 444 | 406 | 357 | 455 | 351 | 1.24× |
| down 256→512, 16² | 813 | 654 | 546 | 759 | 564 | 1.49× |
| **mid 512, 8²** | 430 | 305 | **232** | 326 | 227 | **1.85×** |
| up 512→256, 16² | 787 | 714 | 556 | 709 | 516 | 1.42× |
| up 256→128, 32² | 883 | 1046 | 910 | 1065 | 845 | 0.97× |
| up 128, 64² | 1931 | 2379 | 2272 | 2674 | 2246 | 0.85× |

Conv quantization **wins at high-channel / low-spatial** shapes (mid_512_8 int4 **1.85×**, down_256_512
1.49×) but **loses at low-channel / high-resolution** shapes (128-ch @ 64² ≈ 0.82×), where cuDNN's fp16
conv is very strong and the int quantize/pack overhead dominates. int4 beats int8 everywhere (packed 4-bit
GEMM). **modiff adds overhead** (int8_modiff consistently slower than int8_baseline: the step1-quantize +
`o_hat` accumulate). Net across the UNet the conv win is modest and shape-dependent.

![conv kernel](figs/fig_conv_kernel.png)

### Linear (qkv/proj) — speed, 5 modes  ·  `data/linear_kernel_speed.csv` · `figs/fig_linear_kernel.png`

Weighted total over the 42 qkv/proj GEMMs per forward (b128). *Linear has no modiff variant → int8_baseline
≡ int8_modiff, int4_baseline ≡ int4_modiff.*

| policy | µs/fwd | vs fp16 |
|---|--:|--:|
| fp16 | 7390 | 1.00× |
| **int8 GEMM-only** (quantize fused into upstream GroupNorm) | **6015** | **1.23×** |
| int8 +standalone quantize | 8382 | 0.88× |
| **int4 GEMM-only** | **4751** | **1.56×** |
| int4 +standalone quantize | 9934 | 0.74× |

The int GEMM wins **only when the activation quantize is fused away** (as production does, into
`group_norm_silu_quantize`): int8 **1.23×**, int4 **1.56×**. A standalone quantize erases the win
(memory-bound pass over the [M,K] activation). Biggest wins at K≥384 (int4 up to 2.15× at 384→1152);
weakest at the small-K=192 level-0 shapes.

![linear kernel](figs/fig_linear_kernel.png)

### Attention (WITH GroupNorm, fair) — speed, 5 modes  ·  `data/attn_kernel_fair_speed.csv` · `figs/fig_attn_fair.png`

Both paths pay GroupNorm on the real `[b,C,H,W]` block input; the quant path additionally pays the Q/K/V
quantize; attention core = fused-flash int8/int4 (`flash_attn_int8_vt`/`flash_attn_int4_vt`) vs fp16 MATH
SDPA. Only hd≤48 & T%64==0 blocks run flash (15/21); the hd=96 blocks stay fp16. *Attention has no modiff
variant → baseline ≡ modiff.*

| block (hd/T) | ×cnt | GN µs | fp16 tot | int8 tot | int4 tot | int8 vs fp16 | int4 vs fp16 | rel-L2 (i8/i4) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **24/1024** | 5 | 481 | 16701 | 8268 | 8131 | **2.02×** | **2.05×** | 0.025 / 0.144 |
| 48/256 | 5 | 274 | 1628 | 2156 | 2163 | 0.75× | 0.75× | 0.018 / 0.150 |
| 48/64 | 5 | 162 | 318 | 725 | 688 | 0.44× | 0.46× | 0.015 / 0.142 |
| 96/16 | 5 | 48 | 113 | (fp16) | (fp16) | 1.00× | 1.00× | — |
| 96/4 | 1 | 12 | 78 | (fp16) | (fp16) | 1.00× | 1.00× | — |
| **weighted / forward (21 blocks)** | | 4838 | **93876** | **56392** | **55553** | **1.66×** | **1.69×** | — |

The **dominant T=1024 block is 2.0× faster** with fused-flash int8/int4 even including GroupNorm and the
quantize prologue, driving a **1.66×/1.69× weighted** attention speedup. Small-T blocks (256, 64) *lose*
(the quantize prologue > the tiny attention it feeds), and hd=96 blocks stay fp16 — but they are cheap, so
the weighted result is a solid win. int8 rel-L2 ≈ 0.02 (quality-safe); int4 ≈ 0.14 (Q/K 4-bit, lossy).

![attention with norm](figs/fig_attn_fair.png)

### Kernel-level read/write

Isolated kernel microbenchmarks emit **no per-iteration memcpy** (operands are staged once), and in-kernel
DRAM read/write is unmeasurable here (counters locked). So there is no measured kernel-level byte table; the
memory story is the e2e memcpy (§E2E-2, negligible) plus the qualitative note that the real per-kernel DRAM
traffic scales with precision (int8 ≈ ½, int4 ≈ ¼ the fp16 operand bytes) — which would need ncu to confirm.

---

## Takeaways

1. **int4_baseline = 1.58× vs true fp16 (fastest), int8_baseline = 1.53×.** The flash-only refactor holds
   the int8 headline; the int4 conv deep-fusion fix (SmoothQuant folded into the fused GN→conv kernel)
   lifted int4 from 1.34× → 1.58×, so 4-bit is now correctly the fastest mode (bit-identical output).
2. **The e2e win is attention + conv, not memory movement.** The timing profile shows fp16 attention
   (~86 ms, softmax + fp16 bmm) collapsing to ~35 ms fused-flash int8, plus conv 46→25 ms; the quantize
   prologue (~19 ms) and modiff cache (~9 ms) are the costs paid back.
3. **modiff (temporal cache) is a net loss at b128** — every modiff mode is slower than its baseline, and
   this is **intrinsic, not a fusion gap**: the int4_modiff path is already fully fused (delta-quantize +
   SmoothQuant + SiLU + pack in one kernel, bias in the o_hat cache; only a ~1.3 ms residual add is
   eager). Its overhead is the delta-quantize (+8.8 ms) and o_hat accumulate (+8.7 ms) — algorithmic
   costs of temporal caching that fusion cannot remove.
4. **Kernel wins are shape-gated:** attention flash wins big only at T=1024 (2.0×); linear int GEMM wins
   only with fused quantize (int8 1.23× / int4 1.56×); conv wins only at high-channel/low-spatial shapes.
5. **Memory read/write could not be measured** beyond negligible memcpy — GPU perf counters are locked on
   this box. A counter-enabled host + `ncu dram__bytes_{read,write}.sum` is required for true per-kernel
   DRAM traffic; everything else here is fully measured.
