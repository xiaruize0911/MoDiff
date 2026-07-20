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
- **Memory read/write — two measured signals.** HW DRAM-byte counters (ncu / CUPTI metrics / DCGM /
  `nsys --gpu-metrics`) are all permission-locked here (`RmProfilingAdminOnly=1`, no `CAP_SYS_ADMIN`;
  `ERR_NVGPUCTRPERM`, verified). So memory IO is measured two counter-free ways: (a) **e2e nsys memcpy**
  (copy traffic only — negligible here), and (b) **per-kernel via NVBit SASS instrumentation**
  (§Kernel-level read/write) — real per-kernel GLOBAL read/write bytes incl. CUTLASS conv, no counters.
  nsys memcpy alone **cannot** see the DRAM reads/writes *inside* compute kernels, which is why the e2e
  memcpy is near-null (~0.4 MiB/step, identical across modes — the pipeline moves data via in-kernel
  DRAM I/O, not memcpys); the NVBit table captures that in-kernel traffic. So the e2e memory table is near-null;
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

> **Where "total IO" is (and isn't) measured — one-stop answer.** The *only* total-IO number in this
> report is this table: **e2e memcpy traffic**, measured via nsys (`scripts/run_nsys_mem.sh` →
> `nsys_driver.py` → `parse_nsys_mem.py`, parsing `CUPTI_ACTIVITY_KIND_MEMCPY`) → `data/e2e_memcpy_total.csv`
> + `data/e2e_memcpy_sites.csv` + `figs/fig_e2e_memcpy.png`. There is **no per-kernel / per-component DRAM
> IO** (conv/linear/attention read+write bytes) — those need `ncu dram__bytes.sum`, and GPU perf counters
> are permission-locked here (`ERR_NVGPUCTRPERM`; see §Method caveats and §Kernel-level read/write). So
> at the **e2e** level total IO = memcpy (≈0.4 MiB/step, negligible). The **per-kernel** compute DRAM IO
> is instead measured via **NVBit** (SASS instrumentation, no counters) — see §Kernel-level read/write.

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
Per-component (conv/linear/attention) DRAM read/write is compute-kernel-internal, so it's not visible to
memcpy tracing — but it **is measured per kernel** via NVBit SASS instrumentation (no counters); see
**§Kernel-level read/write**. The §3 timing profile is the complementary per-component time signal.

---

## Kernel level

### Conv — speed, 5 modes  ·  `data/conv_kernel_speed.csv` · `figs/fig_conv_kernel.png`

Churches ResBlock convs at b128 (µs, median 5×200):

| shape (Cin→Cout, HW) | fp16 | int8_base | int4_base | int8_modiff | int4_modiff | **int8 vs fp16** | int4 vs fp16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| res 128, 64² | 1861 | 2361 | 2270 | 2715 | 2244 | **0.79×** | 0.82× |
| res 128, 32² | 493 | 610 | 588 | 706 | 582 | **0.81×** | 0.84× |
| down 128→256, 32² | 944 | 932 | 911 | 1165 | 954 | **1.01×** | 1.04× |
| res 256, 32² | 1626 | 1532 | 1283 | 1681 | 1265 | **1.06×** | 1.27× |
| res 256, 16² | 444 | 406 | 357 | 455 | 351 | **1.09×** | 1.24× |
| down 256→512, 16² | 813 | 654 | 546 | 759 | 564 | **1.24×** | 1.49× |
| **mid 512, 8²** | 430 | 305 | **232** | 326 | 227 | **1.41×** | **1.85×** |
| up 512→256, 16² | 787 | 714 | 556 | 709 | 516 | **1.10×** | 1.42× |
| up 256→128, 32² | 883 | 1046 | 910 | 1065 | 845 | **0.84×** | 0.97× |
| up 128, 64² | 1931 | 2379 | 2272 | 2674 | 2246 | **0.81×** | 0.85× |

**int8_baseline vs fp16**: spans **0.79×–1.41×** — the same shape pattern as int4 but milder. Conv
quantization **wins at high-channel / low-spatial** shapes (mid_512_8 int8 **1.41×** / int4 **1.85×**,
down_256_512 1.24× / 1.49×) but **loses at low-channel / high-resolution** shapes (128-ch @ 64² ≈ 0.79×
int8 / 0.82× int4), where cuDNN's fp16 conv is very strong and the int quantize/pack overhead dominates.
int4 beats int8 at every shape (packed 4-bit GEMM). **modiff adds overhead** (int8_modiff consistently
slower than int8_baseline: the step1-quantize + `o_hat` accumulate). Net across the UNet the conv win is
modest and shape-dependent.

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

**5 most-frequently-called qkv/proj shapes** (by per-forward `count`; M = b·T). The four `count=5`
T-shapes dominate call volume — the two `768→…, M=512` shapes (`count=1`) are the least frequent:

| shape (K→N) | M | count/fwd | int8 GEMM × | int4 GEMM × |
|---|--:|--:|--:|--:|
| qkv 192→576 | 131072 | 5 | 1.04× | 1.20× |
| proj 192→192 | 131072 | 5 | 1.18× | 1.36× |
| qkv 384→1152 | 32768 | 5 | 1.57× | 2.16× |
| proj 384→384 | 32768 | 5 | 1.23× | 1.64× |
| qkv 384→1152 | 8192 | 5 | 1.13× | 1.55× |

**5 shapes with the most speedup** (GEMM-only vs fp16, ranked by int4):

| shape (K→N, M) | int4 GEMM × | int8 GEMM × |
|---|--:|--:|
| qkv 384→1152, M=32768 | **2.16×** | 1.57× |
| proj 384→384, M=8192 | 1.99× | 1.56× |
| qkv 768→2304, M=2048 | 1.90× | 1.25× |
| proj 384→384, M=32768 | 1.64× | 1.23× |
| qkv 768→2304, M=512 | 1.60× | 1.01× |

**Inverse relationship:** the *most-frequent* shapes (K=192, M=131072) have the *weakest* speedup
(int8 1.04–1.18×), while the biggest speedups are the less-frequent K≥384 shapes — so the weighted
total (int8 1.23×, int4 1.56×) is pulled down by the high-frequency small-K level-0 blocks.

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

### Kernel-level read/write — MEASURED via NVBit (no perf counters)  ·  `data/nvbit_io_{total,perkernel}.csv`

HW DRAM counters (ncu/CUPTI/DCGM/`nsys --gpu-metrics`) are all locked here (`ERR_NVGPUCTRPERM`), but
**NVBit binary instrumentation** measures per-kernel GLOBAL read/write bytes by instrumenting SASS at
runtime — no counter permission, and it covers **every** kernel incl. CUTLASS conv & cuDNN. Custom tool
`scripts/nvbit_mem_bytes/` (counts `active_threads × access_size` per global ld/st, opcode-split
read/write), driven by `scripts/{nvbit_io_driver.py, run_nvbit_io.sh, parse_nvbit_io.py}`, one config per
`cuProfilerStart/Stop` range. **Validated byte-exact** (fp16 `add_` on 8192² → read=write=134217728 =
8192²×2). Measured DRAM read/write (MiB) per op at the dominant shapes, b128:

int8/int4 columns = **baseline** (modiff differs only for conv — separate table below; linear/attention
kernels are identical for baseline & modiff). rd / wr (total):

| family / shape | fp16 rd / wr | int8_base rd / wr | int4_base rd / wr |
|---|--:|--:|--:|
| **attn hd24/T1024** | 8800 / 4240 (**13040**) | 2864 / 48 (**2912**) | 2864 / 48 (2912) |
| attn hd48/T256 | 736 / 328 (1064) | 244 / 24 (268) | 236 / 24 (260) |
| conv res_128_64 (128ch,64²) | 256 / 256 (512) | 708 / 576 (**1284**) | 1090 / 672 (1762) |
| conv mid_512_8 (512ch,8²) | 16 / 16 (32) | 44 / 36 (80) | 68 / 42 (110) |
| linear qkv 192→576 M131072 | 644 / 144 (788) | 20 / 144 (164) | 20 / 144 (164) |
| linear qkv 384→1152 M8192 | 127 / 18 (145) | 2 / 18 (20) | 2 / 18 (20) |

**Conv baseline vs modiff** (total DRAM MiB) — modiff adds the a_hat/o_hat temporal-cache traffic:

| conv shape | fp16 | int8_base | int8_modiff | int4_base | int4_modiff |
|---|--:|--:|--:|--:|--:|
| res_128_64 | 512 | 1284 | **1540** | 1762 | **1506** |
| res_256_32 | 256 | 641 | **770** | 881 | **753** |
| down_256_512_16 | 128 | 241 | **321** | 364 | **316** |
| mid_512_8 | 32 | 80 | **96** | 110 | **94** |
| up_512_256_16 | 64 | 240 | **256** | 296 | **248** |

Three measured findings:

1. **Attention: flash moves 4.5× less DRAM (13040 → 2912 MiB).** Per-kernel breakdown (fp16): the
   `[BH,T,T]` softmax round-trips HBM at **2048 rd + 2048 wr MiB** and the QKᵀ/AV bmm reads 6656 MiB;
   int8/int4 flash is **one kernel, 2864 rd / 48 wr** — scores never leave SRAM, so no T×T round-trip and
   only the fp16 output is written. This is the measured memory-traffic proof of the fused-flash win.
2. **Conv int8/int4 move MORE DRAM than fp16** (res_128_64: 512 → 1284/1762 MiB) — the extra
   quantize/pack + a_hat-zero + dequant-store + residual traffic outweighs the int-operand shrink at
   these low-channel/high-res shapes, which is exactly why int8/int4 conv *loses* there (0.79×/0.82×).
   At high-channel/low-spatial (mid_512_8) the overhead is proportionally smaller. **modiff vs baseline
   (conv only):** int8_modiff reads ~256 MiB more than int8_baseline (res_128_64: 964 vs 708 rd) — the
   **a_hat temporal-cache read** for the delta; int4_modiff is slightly *lower* (its delta-quantize moves
   less than int4's full re-quantize+pack). Linear/attention are byte-identical for baseline vs modiff.
3. **Linear (GEMM-only): int8/int4 read far less** (qkv 192→576: 644 → 20 MiB read) — the int GEMM reads
   packed int8/int4 operands with far less tile-reload traffic than the fp16 GEMM. (Write ≈ equal: the
   fp16 output dominates and is the same size.) This is the IO basis of the GEMM-only linear win.

(NVBit counts *requested* global bytes — an upper bound on post-L2 DRAM, but for these large-footprint
kernels L2 reuse is small; the byte-exact `add_` check bounds the method error. Full 49-config × 93-kernel
data in the CSVs. An `ncu dram__bytes.sum` cross-check would need the counter unlock — harness also
provided in `scripts/{ncu_io_driver,run_ncu_io,parse_ncu_io}.py`.)

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
5. **Per-kernel memory read/write IS measured — via NVBit**, despite locked HW counters. The headline:
   **attention flash moves 4.5× less DRAM than fp16** (13040 → 2912 MiB at hd24/T1024), because the
   `[BH,T,T]` softmax round-trip (2048 rd + 2048 wr MiB) is eliminated (scores stay in SRAM). Conv
   int8/int4 move *more* DRAM than fp16 (quantize/pack/store overhead) — matching their conv-speed losses;
   linear int8/int4 GEMM read far less. e2e memcpy is negligible (~0.4 MiB/step). An `ncu dram__bytes.sum`
   cross-check would need a counter unlock, but NVBit (validated byte-exact) already gives the numbers.
