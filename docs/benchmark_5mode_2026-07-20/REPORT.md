# MoDiff 5-mode benchmark — measured data

**GPU:** NVIDIA A40 (48 GB, SM 8.6) · **PyTorch:** 2.4.1+cu124 · **CUDA:** 12.4 · nsys 2024.1.1
**Model:** LSUN-Churches LDM-8 UNet (unconditional, 256×256) · **Batch:** 128 · **Sampler:** DDIM
**Date:** 2026-07-20 · **all timing/kernel/IO data re-measured 2026-07-21** at the current code state
(fused-flash attention; a_hat-drop; residual→o_hat fusion ON; int4 conv deep-fuse dequant ON;
GN→delta-quantize fusion OFF).

**5 modes:** `fp16`, `int8_baseline`, `int4_baseline`, `int8_modiff`, `int4_modiff`. int8/int4 use
fused-flash quantized attention; `_modiff` adds the temporal-delta conv cache.

**Method (measured only):** speed = CUDA-event / wall time, GPU clock burn-in → warmup → N×R timed with
`torch.cuda.synchronize()` (e2e 30 warm + 5×200 steps; kernels 50 warm + 200×5). Timing profile =
`torch.profiler` CUDA self-time, bucketed by kernel name, ms/step. Per-kernel DRAM read/write = NVBit
SASS instrumentation (`scripts/nvbit_mem_bytes`, validated byte-exact; HW counters are locked here —
`ERR_NVGPUCTRPERM`). e2e memcpy = nsys `CUPTI_ACTIVITY_KIND_MEMCPY`. Checkpoint is a random-weight stub
(dispatch/shapes faithful → **speed faithful, generation quality meaningless**). autocast fp16 on for all
modes. Data: `data/*.csv` · figures: `figs/*.png`.

---

## E2E

### 1. Speed · `data/e2e_speed.csv` · `figs/fig_e2e_speed.png`

| mode | ms/step | min ms | vs fp16 |
|---|--:|--:|--:|
| fp16 | 190.0 | 188.8 | 1.00× |
| int8_baseline | 124.3 | 124.1 | 1.53× |
| **int4_baseline** | **117.3** | **117.2** | **1.62×** |
| int8_modiff | 140.9 | 140.9 | 1.35× |
| int4_modiff | 140.7 | 140.1 | 1.35× |

![e2e speed](figs/fig_e2e_speed.png)

### 2. Total memcpy (nsys) · `data/e2e_memcpy_total.csv` · `figs/fig_e2e_memcpy.png`

| mode | H2D | D2H | D2D | total MiB/step |
|---|--:|--:|--:|--:|
| fp16 / int8_baseline / int4_baseline / int8_modiff / int4_modiff | 0.0 | 0.0 | 0.4 | 0.4 |

Copy-traffic only; per-kernel compute DRAM IO is in §Kernel/IO (NVBit).

![e2e memcpy](figs/fig_e2e_memcpy.png)

### 3. Per-component timing profile (GPU self-time, ms/step) · `data/e2e_timing_profile.csv` · `figs/fig_e2e_timing_profile.png`

| bucket | fp16 | int8_baseline | int4_baseline | int8_modiff | int4_modiff |
|---|--:|--:|--:|--:|--:|
| attention (flash / softmax) | 44.0 | 35.1 | 33.9 | 34.5 | 33.7 |
| attn bmm fp16 (QKᵀ/AV) | 42.4 | 0.2 | 0.2 | 0.2 | 0.2 |
| conv (int GEMM) | 46.3 | 24.5 | 14.3 | 28.8 | 15.8 |
| qkv/proj int GEMM | 0.0 | 7.7 | 7.1 | 7.5 | 7.0 |
| GroupNorm | 21.6 | 23.9 | 22.9 | 22.4 | 22.2 |
| quantize/dequant | 0.0 | 19.0 | 17.3 | 28.3 | 25.6 |
| modiff cache (o_hat) | 0.0 | 0.0 | 0.0 | 11.0 | 11.0 |
| elementwise/copy | 32.6 | 12.7 | 18.2 | 7.6 | 11.4 |
| upsample/concat + other | 12.6 | 6.9 | 7.0 | 6.8 | 6.4 |
| **gpu_busy** | 199.5 | 130.0 | 120.9 | 147.1 | 133.1 |
| **wall** | 190.0 | 123.4 | 115.8 | 140.1 | 142.4 |

![e2e timing profile](figs/fig_e2e_timing_profile.png)

---

## Kernel

### Conv — speed, 5 modes (µs, b128) · `data/conv_kernel_speed.csv` · `figs/fig_conv_kernel.png`

| shape (Cin→Cout, HW) | fp16 | int8_base | int4_base | int8_modiff | int4_modiff | int8 vs fp16 | int4 vs fp16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| res 128, 64² | 1876 | 1668 | 1146 | 2676 | 2246 | 1.13× | 1.64× |
| res 128, 32² | 488 | 435 | 303 | 704 | 582 | 1.12× | 1.61× |
| down 128→256, 32² | 938 | 763 | 521 | 1175 | 954 | 1.23× | 1.80× |
| res 256, 32² | 1623 | 1210 | 755 | 1677 | 1265 | 1.34× | 2.15× |
| res 256, 16² | 440 | 321 | 209 | 453 | 351 | 1.37× | 2.11× |
| down 256→512, 16² | 805 | 571 | 358 | 753 | 562 | 1.41× | 2.25× |
| mid 512, 8² | 427 | 264 | 157 | 326 | 227 | 1.62× | 2.71× |
| up 512→256, 16² | 779 | 547 | 338 | 703 | 515 | 1.42× | 2.30× |
| up 256→128, 32² | 874 | 707 | 459 | 1059 | 844 | 1.24× | 1.90× |
| up 128, 64² | 1916 | 1691 | 1162 | 2688 | 2247 | 1.13× | 1.65× |

![conv kernel](figs/fig_conv_kernel.png)

### Linear (qkv/proj) — speed, weighted per forward (42 GEMMs, b128) · `data/linear_kernel_speed.csv` · `figs/fig_linear_kernel.png`

| policy | µs/fwd | vs fp16 |
|---|--:|--:|
| fp16 | 7383 | 1.00× |
| int8 GEMM-only (quantize fused into GroupNorm) | 5985 | 1.23× |
| int8 + standalone quantize | 8345 | 0.88× |
| int4 GEMM-only | 4720 | 1.56× |
| int4 + standalone quantize | 9920 | 0.74× |

5 most-frequent qkv/proj shapes (GEMM-only vs fp16; M = b·T):

| shape (K→N) | M | count/fwd | int8 × | int4 × |
|---|--:|--:|--:|--:|
| qkv 192→576 | 131072 | 5 | 1.03× | 1.20× |
| proj 192→192 | 131072 | 5 | 1.18× | 1.37× |
| qkv 384→1152 | 32768 | 5 | 1.57× | 2.16× |
| proj 384→384 | 32768 | 5 | 1.23× | 1.64× |
| qkv 384→1152 | 8192 | 5 | 1.12× | 1.56× |

![linear kernel](figs/fig_linear_kernel.png)

### Attention (WITH GroupNorm, fair) — speed, 5 modes (µs, b128) · `data/attn_kernel_fair_speed.csv` · `figs/fig_attn_fair.png`

int8/int4 attention core = `flash_attn_int8_vt` / `flash_attn_int4_vt`; fp16 = MATH SDPA. Only hd≤48 &
T%64==0 blocks run flash; hd=96 blocks stay fp16. Attention has no modiff variant (baseline ≡ modiff).

| block (hd/T) | ×cnt | GN µs | fp16 tot | int8 tot | int4 tot | int8 vs fp16 | int4 vs fp16 | rel-L2 (i8/i4) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 24/1024 | 5 | 494 | 16646 | 8417 | 8227 | 1.98× | 2.02× | 0.025 / 0.144 |
| 48/256 | 5 | 277 | 1636 | 2190 | 2185 | 0.75× | 0.75× | 0.018 / 0.150 |
| 48/64 | 5 | 163 | 319 | 727 | 689 | 0.44× | 0.46× | 0.015 / 0.142 |
| 96/16 | 5 | 48 | 113 | (fp16) | (fp16) | 1.00× | 1.00× | — |
| 96/4 | 1 | 12 | 78 | (fp16) | (fp16) | 1.00× | 1.00× | — |
| **weighted / forward (21 blocks)** | | 4921 | **93649** | **57309** | **56151** | **1.63×** | **1.67×** | — |

![attention with norm](figs/fig_attn_fair.png)

### Per-kernel DRAM read/write — NVBit (MiB, b128) · `data/nvbit_io_{total,perkernel}.csv` · `figs/fig_conv_io.png`

int8/int4 columns = baseline; rd / wr (total):

| family / shape | fp16 rd/wr | int8_base rd/wr | int4_base rd/wr |
|---|--:|--:|--:|
| attn hd24/T1024 | 8800 / 4240 (13040) | 2864 / 48 (2912) | 2864 / 48 (2912) |
| attn hd48/T256 | 736 / 328 (1064) | 244 / 24 (268) | 236 / 24 (260) |
| conv res_128_64 (128ch,64²) | 256 / 256 (512) | 580 / 320 (900) | 706 / 288 (994) |
| conv mid_512_8 (512ch,8²) | 16 / 16 (32) | 36 / 20 (56) | 44 / 18 (62) |
| linear qkv 192→576 M131072 | 644 / 144 (788) | 20 / 144 (164) | 20 / 144 (164) |
| linear qkv 384→1152 M8192 | 127 / 18 (145) | 2 / 18 (20) | 2 / 18 (20) |

Conv total DRAM IO, baseline vs modiff (MiB):

| conv shape | fp16 | int8_base | int8_modiff | int4_base | int4_modiff |
|---|--:|--:|--:|--:|--:|
| res_128_64 | 512 | 900 | 1540 | 994 | 1506 |
| res_256_32 | 256 | 449 | 770 | 497 | 753 |
| down_256_512_16 | 128 | 193 | 321 | 221 | 316 |
| mid_512_8 | 32 | 56 | 96 | 62 | 94 |
| up_512_256_16 | 64 | 144 | 256 | 152 | 248 |

![conv DRAM IO](figs/fig_conv_io.png)

---

## Optimization A/B (measured; bit-exact unless noted)

**a_hat-drop** — baseline conv, cache-free static quantize (bit-identical, rel-L2=0). Conv DRAM IO, MiB ·
`data/conv_io_ahat_drop.csv`:

| shape | int8_base before→after | int4_base before→after |
|---|--:|--:|
| res_128_64 | 1284 → 900 | 1762 → 1378 |
| res_256_32 | 641 → 450 | 881 → 689 |
| mid_512_8 | 80 → 56 | 110 → 86 |

**residual→o_hat fusion** — modiff, skip-add folded into the o_hat conv (bit-identical: 0 o_hat / 0 a_hat
/ 0 output diff over 30 e2e calls). e2e ms/step, ON by default · `data/o_hat_residual_fusion_e2e.csv`:

| mode | off | on | Δ |
|---|--:|--:|--:|
| int8_modiff | 160.9 | 156.7 | −4.2 |
| int4_modiff | 158.8 | 153.5 | −5.3 |

**int4 conv deep-fuse dequant** — weight_scale folded into the CUTLASS int4 epilogue + from_half store
(vs fp32 conv_out + scale_store; output rel-L2 4.3e-4 vs fp32 path). ON by default ·
`data/int4_deepfuse_e2e.csv`:

| metric | off | on | Δ |
|---|--:|--:|--:|
| int4_baseline e2e ms/step | 131.9 | 129.7 | −2.1 |
| int4_baseline conv elementwise ms | 20.0 | 18.2 | −1.8 |
| int4_base conv IO res_128_64 (MiB) | 1378 | 994 | −384 |

**GN→delta-quantize fusion** — modiff, GroupNorm folded into the delta-quantize (bit-identical). e2e
regression; kept OFF (opt-in `MODIFF_ENABLE_GN_MODIFF_FUSION=1`) · `data/gn_modiff_fusion_{e2e,kernel}.csv`:

| metric | off | on | Δ |
|---|--:|--:|--:|
| int8_modiff e2e ms/step | 161.1 | 164.3 | +3.2 |
| int4_modiff e2e ms/step | 158.7 | 160.7 | +2.0 |
| fused-kernel vs 2-kernel (res_128_64) | 1.00× | 0.72× | — |
