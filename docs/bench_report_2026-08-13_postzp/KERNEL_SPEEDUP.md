# Kernel speedups: fp16 → int8 → int4

`NVIDIA A40`, batch 128. Replay medians at the shapes captured from a live sample; `speedup` is fp16 µs/call ÷ quantized µs/call **at equal work**. PTQ arms (`int8_baseline`, `int4_baseline`), so no MoDiff temporal kernels are mixed in.

## 1. Suite totals (ms/sample)

> **The unit is not the e2e table's unit.** `calls_per_sample` counts calls over the capture window, which is **5 steps** of the whole batch (`capture_steps=5`), so `ms/sample` here is ~5x a ms/step for batch 128. REPORT.md section 1's `ms/sample` is per **image** (batch time / 128). fp16 is 488.31 in this table and 160.9 in that one and both are right. Every ratio below is within one unit and so unaffected, but the two columns must not be combined -- 63.34/160.9 would read attention as 39% of the run when the profile says 12.3%. The `ms/step` column is the one to compare with REPORT.md section 1 and section 1a.

Totals as captured, with `fused_gn_qkv` routed to `linear` (see below). **The `speedup` columns are printed so they can be dismissed** — not one of them is a speedup. Read §2 and REPORT.md §1a instead; the paragraphs after the table say why.

| suite | fp16 | int8 | int4 | fp16 ms/step | fp16/int8 | fp16/int4 | is this a speedup? |
|---|--:|--:|--:|--:|--:|--:|---|
| attention | 63.34 | 51.08 | 50.01 | 12.67 | 1.24× | 1.27× | **yes** — same work, same suite, all three arms |
| conv | 265.72 | 149.09 | 85.59 | 53.14 | 1.78× | 3.10× | **yes** — 33/33 records matched three ways, nothing moves in or out |
| linear | 60.92 | 47.15 | 43.45 | 12.18 | 1.29× | 1.40× | **no** — holds fp16's fused GroupNorm, whose quantized counterpart is in `norm_quantize` |
| norm_quantize | 92.23 | 143.35 | 144.44 | 18.45 | 0.64× | 0.64× | **no** — the mirror of `linear`; also absorbs quantize launches that replace work the fp16 arm pays as separate elementwise kernels |
| other | 6.10 | 10.11 | 10.11 | 1.22 | 0.60× | 0.60× | **no** — `cat2` capture coverage differs between arms (see below) |
| **all five** | **488.31** | **400.78** | **333.61** | **97.66** | **1.22×** | **1.46×** | **no** — see the third paragraph |

**Where fp16's qkv projections live.** fp16 runs the T=1024 and T=256 qkv through `fused_gn_qkv` — one kernel doing GroupNorm and the projection — worth **31.96 ms/sample**. The gate is `T % 128 == 0 and c % 8 == 0`, which is why the T=64/16/4 qkv is an ordinary `torch_linear_fp16` in this same suite. The quantized arms have no fused counterpart at all — they split the same work into a `group_norm_silu_quantize_nhwc` in `norm_quantize` plus an AWQ GEMM here. Until 2026-08-16 the capture's `suite_of()` matched name keywords and `fused_gn_qkv` contains none of them, so these two records sat in `other` — and this table published `linear` at 0.61×, `other` at 3.77× and a `conv + linear` row claiming the two cancel. They do not: the move is `other` → `linear`, and it never touched conv at all.

**Conv closes exactly, which is what kills the old story.** All 33 conv records match three ways — fp16's suite total equals its matched total to the cent, in every arm (asserted on every run of this script). Nothing moves between conv and linear in either direction. The 1×1 convs in §2 are ResBlock skip connections, present in all three arms at 1.00×, and fp16's attention out-proj is already a Linear.

**`other` is now all `cat2_channels_last_fp16`, and it does not compare either — for an unrelated reason.** The capture is asymmetric: fp16 is missing two signatures both quantized arms have (`[128,384,32,32]+[128,192,32,32]` at 2.64 ms and `[128,768,8,8]+[128,384,8,8]` at 0.34 ms), and it recorded 5 calls where int8 recorded 10 on `[128,384,16,16]²`. A concat is arm-independent by construction, so that is a coverage gap in the capture, not a real difference. It needs a GPU re-capture to close — docs/OPEN_ITEMS.md A15.

**No regrouping of these five suites is clean, including the sum.** fp16's `fused_gn_qkv` does the GroupNorm too, and that GroupNorm cannot be in `linear` and in `norm_quantize` at once. Worse for the sum: the quantized arms' fused epilogues *delete* tensors, so the elementwise kernels fp16 pays for them do not exist as records to be credited — the full-run profile in REPORT.md §1a sees that as 2.78 s saved, and a replay suite cannot see it at all. That is why the all-five row reads 1.22× against the wall clock's 1.45×.

## 2. Per conv layer — the strict comparison

Matched on the weight normalized to `(K, C, R, S)`, so the same layer is compared across all three arms despite three different operand layouts. `calls` is per sample.

> **One correction to read the fp16 column with.** The replay runs under `autocast(fp16)`, exactly as production does, and the captured arguments are what the caller passed *before* autocast cast them. For 12 of the fp16-arm conv records the activation arrives as **fp32**, so autocast's fp32→fp16 conversion of that activation is inside the timed region while the arithmetic is fp16 either way. Checked on the clearest case: `[128,1152,8,8]` fp32 is 37.7 MB read + 18.9 MB written ≈ 94 µs at ~600 GB/s, and that row's fp16-vs-quantized gap is 102.5 µs on a kernel that is `torch_conv2d_fp16` in all three arms. So rows marked `fp32-in` measure a conversion plus a conv, and their speedups are not arithmetic-only. The summary below is split on this.

| K | C | R×S | HxW | fp16 in | calls | fp16 µs | int8 µs | int4 µs | int8 | int4 | int8→int4 |
|--:|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| 384 | 384 | 3×3 | 32×32 | **fp32** | 10 | 3742.4 | 1668.6 | 825.9 | **2.24×** | **4.53×** | 2.02× |
| 192 | 576 | 3×3 | 32×32 | **fp32** | 5 | 3250.4 | 1663.6 | 834.3 | **1.95×** | **3.90×** | 1.99× |
| 384 | 768 | 3×3 | 16×16 | **fp32** | 10 | 1876.9 | 804.0 | 399.9 | **2.33×** | **4.69×** | 2.01× |
| 192 | 384 | 3×3 | 32×32 | fp16 | 10 | 1784.2 | 1055.2 | 523.5 | **1.69×** | **3.41×** | 2.02× |
| 192 _(unquantized)_ | 576 | 1×1 | 32×32 | **fp32** | 5 | 1380.8 | 580.0 | 579.3 | **2.38×** | **2.38×** | 1.00× |
| 384 | 576 | 3×3 | 16×16 | fp16 | 5 | 1246.9 | 679.1 | 344.5 | **1.84×** | **3.62×** | 1.97× |
| 192 | 192 | 3×3 | 32×32 | fp16 | 35 | 1025.0 | 749.1 | 352.5 | **1.37×** | **2.91×** | 2.12× |
| 768 | 768 | 3×3 | 8×8 | **fp32** | 10 | 919.4 | 403.0 | 194.1 | **2.28×** | **4.74×** | 2.08× |
| 384 | 384 | 3×3 | 16×16 | fp16 | 40 | 866.9 | 440.4 | 225.0 | **1.97×** | **3.85×** | 1.96× |
| 4 _(unquantized)_ | 192 | 3×3 | 32×32 | **fp32** | 5 | 826.8 | 827.8 | 827.2 | **1.00×** | **1.00×** | 1.00× |
| 384 | 1152 | 3×3 | 8×8 | **fp32** | 5 | 749.6 | 335.3 | 162.1 | **2.24×** | **4.62×** | 2.07× |
| 768 | 1536 | 3×3 | 4×4 | **fp32** | 10 | 598.8 | 275.1 | 138.8 | **2.18×** | **4.31×** | 1.98× |
| 384 _(unquantized)_ | 768 | 1×1 | 16×16 | **fp32** | 10 | 559.6 | 299.1 | 299.8 | **1.87×** | **1.87×** | 1.00× |
| 384 | 192 | 3×3 | 16×16 | fp16 | 5 | 496.6 | 309.8 | 145.8 | **1.60×** | **3.41×** | 2.13× |
| 192 _(unquantized)_ | 384 | 1×1 | 32×32 | fp16 | 10 | 492.1 | 492.7 | 493.6 | **1.00×** | **1.00×** | 1.00× |
| 384 | 768 | 3×3 | 8×8 | fp16 | 10 | 452.1 | 229.3 | 112.0 | **1.97×** | **4.04×** | 2.05× |
| 768 | 1152 | 3×3 | 4×4 | fp16 | 5 | 431.2 | 211.5 | 107.2 | **2.04×** | **4.02×** | 1.97× |
| 192 _(unquantized)_ | 4 | 3×3 | 32×32 | **fp32** | 5 | 308.1 | 308.3 | 307.9 | **1.00×** | **1.00×** | 1.00× |
| 768 | 768 | 3×3 | 4×4 | fp16 | 40 | 294.3 | 147.6 | 75.7 | **1.99×** | **3.89×** | 1.95× |
| 192 | 192 | 3×3 | 16×16 | fp16 | 10 | 279.6 | 209.4 | 102.8 | **1.34×** | **2.72×** | 2.04× |
| 384 _(unquantized)_ | 576 | 1×1 | 16×16 | fp16 | 5 | 257.2 | 258.3 | 257.8 | **1.00×** | **1.00×** | 1.00× |
| 384 | 384 | 3×3 | 8×8 | fp16 | 45 | 253.4 | 127.5 | 66.2 | **1.99×** | **3.83×** | 1.93× |
| 768 | 1536 | 3×3 | 2×2 | fp16 | 15 | 228.6 | 137.5 | 71.5 | **1.66×** | **3.20×** | 1.92× |
| 384 _(unquantized)_ | 1152 | 1×1 | 8×8 | **fp32** | 5 | 221.5 | 119.0 | 117.7 | **1.86×** | **1.88×** | 1.01× |
| 384 _(unquantized)_ | 192 | 1×1 | 16×16 | fp16 | 5 | 182.8 | 183.2 | 183.6 | **1.00×** | **1.00×** | 1.00× |
| 768 | 384 | 3×3 | 4×4 | fp16 | 5 | 160.6 | 79.6 | 41.1 | **2.02×** | **3.91×** | 1.94× |
| 768 | 768 | 3×3 | 2×2 | fp16 | 65 | 137.1 | 74.1 | 39.4 | **1.85×** | **3.48×** | 1.88× |
| 768 _(unquantized)_ | 1536 | 1×1 | 4×4 | **fp32** | 10 | 125.4 | 86.7 | 87.3 | **1.45×** | **1.44×** | 0.99× |
| 384 | 384 | 3×3 | 4×4 | fp16 | 10 | 96.8 | 41.2 | 25.2 | **2.35×** | **3.84×** | 1.64× |
| 384 _(unquantized)_ | 768 | 1×1 | 8×8 | fp16 | 10 | 95.8 | 94.4 | 94.1 | **1.01×** | **1.02×** | 1.00× |
| 768 _(unquantized)_ | 1152 | 1×1 | 4×4 | fp16 | 5 | 81.7 | 82.1 | 82.1 | **0.99×** | **0.99×** | 1.00× |
| 768 _(unquantized)_ | 384 | 1×1 | 4×4 | fp16 | 5 | 75.6 | 75.0 | 74.5 | **1.01×** | **1.01×** | 1.01× |
| 768 _(unquantized)_ | 1536 | 1×1 | 2×2 | fp16 | 15 | 71.6 | 71.3 | 71.8 | **1.00×** | **1.00×** | 0.99× |

33 layers matched in all three arms: 20 quantized, 13 unquantized controls.

| subset | n | int8 speedup | int4 speedup |
|---|--:|---|---|
| quantized, fp16 baseline also fp16-in — **the arithmetic-only number** | 14 | 1.34×–2.35× (median 1.97×) | 2.72×–4.04× (median 3.83×) |
| quantized, fp16 baseline fp32-in (includes an autocast cast) | 6 | 1.95×–2.33× (median 2.24×) | 3.90×–4.74× (median 4.62×) |
| all quantized | 20 | 1.34×–2.35× (median 1.99×) | 2.72×–4.74× (median 3.89×) |

**The controls work.** Of the 13 unquantized rows, the 7 whose activation dtype also matches across arms come out at 1.00×, 1.00×, 1.00×, 1.01×, 0.99×, 1.01×, 1.00× — the same kernel on the same input times the same in every arm, which is what shows the layout normalization is matching real layers and not coincidentally-shaped ones. The remaining 6 are the `fp32-in` rows above.

**13 of these rows are the control**: convs this pipeline does not quantize, so all three arms run the same `torch_conv2d_fp16` and the speedup must come out ≈1.00×. They do (2.38×, 1.00×, 1.87×, 1.00×, 1.00×, 1.00×, …), which is what shows the layout normalization is matching the right layers rather than coincidentally-shaped ones.

## 3. Per attention block

Matched on `T`, read from **K** (`[N,H,T,hd_pad]`), with the single-tensor packed-Q records assigned by their own `T`. N=128 and H=8 in every record, so they carry no information. Every record in the suite is assigned to exactly one row and the rows sum to the suite totals exactly — checked on every run. `ms/sample` is what the row costs; `µs/call` is **call-weighted** across the kernels in the row, because a row mixes dynamic and static variants at 10 and 5 calls.

| T | hd_pad fp16→int8/int4 | calls f/8/4 | fp16 ms | int8 ms | int4 ms | µs/call fp16→int8→int4 | int8 | int4 | noise |
|--:|---|---|--:|--:|--:|---|--:|--:|---|
| 1024 | 24→32/64 | 25/25/25 | 50.92 | 42.05 | 42.16 | 2036.9→1682.0→1686.4 | **1.21×** | **1.21×** | — |
| 256 | 48→64/64 | 25/25/25 | 8.72 | 6.41 | 5.25 | 348.7→256.3→210.1 | **1.36×** | **1.66×** | **int4 NOISY** |
| 64 | 48→64/64 | 25/25/25 | 2.27 | 1.08 | 1.03 | 90.7→43.1→41.3 | **2.10×** | **2.20×** | — |
| 16 | 96→96/96 | 25/25/25 | 1.19 | 1.36 | 1.38 | 47.8→54.6→55.4 | **0.88×** | **0.86×** | — |
| 4 | 96→96/96 | 5/5/5 | 0.24 | 0.18 | 0.18 | 47.8→36.9→36.5 | **1.30×** | **1.31×** | **fp16 NOISY** |
| **total** | | | **63.34** | **51.08** | **50.01** | | **1.24×** | **1.27×** | |

All 5 blocks matched in all three arms, all records assigned.

**int4 does not have an int4 attention datapath, and that is the whole story of the int4 column.** Three things in the data, none of which is about bit width:

1. Every operand in the int4 arm's attention is `torch.int8`.
2. The dominant T=1024 route runs `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24`, whose profiled CUDA kernel is literally `flash_attn_int8_mma_kernel_t` — the int8 MMA.
3. V stays int8 in both quantized arms. The qkv GEMM that feeds it says so in its name: `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` — int4 for Q and K, int8 for V.

So the only thing int4 can win in attention is Q/K bytes, and whether it wins any depends entirely on how `hd` pads:

| | true hd | int8 pads to | int8 B/row | int4 pads to | int4 B/row | so |
|---|--:|--:|--:|--:|--:|---|
| T=1024 | 24 | 32 values | **32** | 64 values, 2/byte | **32** | identical traffic → 1.21× vs 1.21×, no int4 gain at the route that owns 80% of the suite |
| T=256, T=64 | 48 | 64 values | **64** | 64 values, 2/byte | **32** | int4 halves Q/K traffic → 1.66× vs 1.36× |

That is also the honest form of the padding argument for **int8**: at T=1024 it moves 32 values per row where fp16 moves 24, a third more bytes, and nets 1.21×. The padding is structural to the MMA fragment layout, not a missing optimization, and the hand-written `_hd24` specialization plus a refuted 8-byte loader are what has already been tried.

**T=16 is a regression, and a DELIBERATE one — correcting an earlier claim in this file.** Only 15 of its 25 calls fall back to `torch_sdpa_fp16` (49.4 µs); the other 10 run `flash_attn_int8_qi8packed_small_qout` at 65.8 µs, which is what puts the row below 1.00×. An earlier version of this paragraph called that a sign error. It is not: `quantized_std_attention.py:484` documents it, with its own measurements — *"T16 is a KNOWN performance loss: the dp4a kernel costs ~T^2 while PyTorch's flash is launch-bound and flat … Taken deliberately for structural uniformity — one INT4 dataflow, no shape-dependent fallback to reason about. It also removes the separate `quant_attn_out_int4_pack` pass those blocks needed."* So the ~0.16 ms/step visible here is not free to reclaim: routing T=16 back to sdpa restores that separate pass, whose cost has to be counted against it. T=4 is the other way round and wins decisively (20.0 µs against sdpa's 49.2), which is why the same gate is right there.

## 4. Linear — why there is no per-layer table

**One reason, and it is not the one this section used to give.** The quantized arms pad K differently for the AWQ layout: the same projection is `[131072, 192]` with `K=192` in int8 and `[131072, 128]` with `K=256` in int4, so no printed shape means the same thing in both, and there is no fp16 shape that means it either.

The retracted reason was *"fp16 has no counterpart for the projections — they are 1×1 convs there."* fp16 has counterparts for all of them. The out-projections at every `T`, and the qkv at T=64/16/4, are `torch_linear_fp16` records in this same suite; only the T=1024 and T=256 qkv are elsewhere, and they are `fused_gn_qkv`, not convs. §1 has the accounting.

The int8→int4 comparison is still available per layer, keyed by `(M, n_out)` which both arms report:

| M | N | calls | int8 µs | int4 µs | int8→int4 | kernel |
|--:|--:|--:|--:|--:|--:|---|
| 131072 | 768 | 10 | 565.2 | 558.0 | **1.01×** | `gemm_w8a8_awq_qkv_i8_layouts` |
| 131072 | 640 | 15 | 439.2 | 382.0 | **1.15×** | `gemm_w8a8_awq_bias_res` |
| 131072 | 256 | 25 | 343.6 | 326.2 | **1.05×** | `gemm_w8a8_awq_bias_res` |
| 32768 | 1152 | 25 | 294.9 | 199.2 | **1.48×** | `gemm_w8a8_awq_bias_res` |
| 32768 | 384 | 25 | 189.7 | 169.9 | **1.12×** | `gemm_w8a8_awq_bias_res` |
| 8192 | 1152 | 25 | 84.9 | 59.6 | **1.42×** | `gemm_w8a8_awq_bias_res` |
| 8192 | 384 | 25 | 67.2 | 62.5 | **1.07×** | `gemm_w8a8_awq_bias_res` |
| 2048 | 2304 | 25 | 58.8 | 40.2 | **1.46×** | `gemm_w8a8_awq_bias_res` |
| 2048 | 768 | 25 | 46.0 | 37.0 | **1.24×** | `gemm_w8a8_awq_bias_res` |
| 512 | 768 | 5 | 29.0 | 25.9 | **1.12×** | `gemm_w8a8_awq_bias_res` |
| 512 | 2304 | 5 | 25.8 | 22.2 | **1.16×** | `gemm_w8a8_awq_bias_res` |

