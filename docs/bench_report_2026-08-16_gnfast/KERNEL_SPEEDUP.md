# Kernel speedups: fp16 → int8 → int4

`NVIDIA A40`, batch 128. Replay medians at the shapes captured from a live sample; `speedup` is fp16 µs/call ÷ quantized µs/call **at equal work**. PTQ arms (`int8_baseline`, `int4_baseline`), so no MoDiff temporal kernels are mixed in.

## 1. Suite totals (ms/sample)

> **The unit is not the e2e table's unit.** `calls_per_sample` counts calls over the capture window, which is **5 steps** of the whole batch (`capture_steps=5`), so `ms/sample` here is ~5x a ms/step for batch 128. REPORT.md section 1's `ms/sample` is per **image** (batch time / 128). fp16's five suites sum to 492.29 here against that table's ~161 per image, and both are right. Every ratio below is within one unit and so unaffected, but the two columns must not be combined -- dividing an attention total by a per-image number reads attention as ~39% of the run where the profile says 12.3%. The `ms/step` column is the one to compare with REPORT.md section 1 and section 1a.

Totals as captured, with `fused_gn_qkv` routed to `linear` (see below). **The `speedup` columns are printed so they can be dismissed** — not one of them is a speedup. Read §2 and REPORT.md §1a instead; the paragraphs after the table say why.

| suite | fp16 | int8 | int4 | fp16 ms/step | fp16/int8 | fp16/int4 | is this a speedup? |
|---|--:|--:|--:|--:|--:|--:|---|
| attention | 63.79 | 51.38 | 50.20 | 12.76 | 1.24× | 1.27× | **yes** — same work, same suite, all three arms |
| conv | 268.10 | 150.19 | 86.69 | 53.62 | 1.79× | 3.09× | **yes** — 33/33 records matched three ways, nothing moves in or out |
| linear | 61.85 | 47.46 | 42.83 | 12.37 | 1.30× | 1.44× | **no** — holds fp16's fused GroupNorm, whose quantized counterpart is in `norm_quantize` |
| norm_quantize | 92.43 | 110.85 | 108.31 | 18.49 | 0.83× | 0.85× | **no** — the mirror of `linear`; also absorbs quantize launches that replace work the fp16 arm pays as separate elementwise kernels |
| other | 6.11 | 10.11 | 10.24 | 1.22 | 0.60× | 0.60× | **no** — fp16's `th.cat` fallback was outside the capture's wrapper list (see below) |
| **all five** | **492.29** | **370.00** | **298.28** | **98.46** | **1.33×** | **1.65×** | **no** — see the third paragraph |

**Where fp16's qkv projections live.** fp16 runs the T=1024 and T=256 qkv through `fused_gn_qkv` — one kernel doing GroupNorm and the projection — worth **32.00 ms/sample**. The gate is `T % 128 == 0 and c % 8 == 0`, which is why the T=64/16/4 qkv is an ordinary `torch_linear_fp16` in this same suite. The quantized arms have no fused counterpart at all — they split the same work into a `group_norm_silu_quantize_nhwc` in `norm_quantize` plus an AWQ GEMM here. Until 2026-08-16 the capture's `suite_of()` matched name keywords and `fused_gn_qkv` contains none of them, so these two records sat in `other` — and this table published `linear` at 0.61×, `other` at 3.77× and a `conv + linear` row claiming the two cancel. They do not: the move is `other` → `linear`, and it never touched conv at all.

**Conv closes exactly, which is what kills the old story.** All 33 conv records match three ways — fp16's suite total equals its matched total to the cent, in every arm (asserted on every run of this script). Nothing moves between conv and linear in either direction. The 1×1 convs in §2 are ResBlock skip connections, present in all three arms at 1.00×, and fp16's attention out-proj is already a Linear.

**`other` reads fp16 as CHEAPER at concatenation, and that was an instrument artifact.** `openaimodel._skip_concat` takes `cat2_channels_last_fp16` only when both halves are fp16 **and** channels_last, and falls back to `th.cat` otherwise. About 3 skip-concats/step take that fallback in the fp16 arm; the quantized arms put more layers in channels_last fp16, so more of theirs qualify for the wrapped kernel. The capture wrapped `mc.*` plus three `F.*` functions and **not `torch.cat`**, so it recorded theirs and not fp16's. Two independent captures reproduced the gap to the call, which is what ruled out flakiness and pointed at the wrapper list. `torch.cat` is wrapped as of 2026-08-16; THIS capture predates that, so the row is still short and is not a comparison. docs/OPEN_ITEMS.md A15.

**No regrouping of these five suites is clean, including the sum.** fp16's `fused_gn_qkv` does the GroupNorm too, and that GroupNorm cannot be in `linear` and in `norm_quantize` at once. Worse for the sum: the quantized arms' fused epilogues *delete* tensors, so the elementwise kernels fp16 pays for them do not exist as records to be credited — the full-run profile in REPORT.md §1a sees that as 2.78 s saved, and a replay suite cannot see it at all. That is why the all-five row reads 1.33× against the wall clock's 1.45×.

## 2. Per conv layer — the strict comparison

Matched on the weight normalized to `(K, C, R, S)`, so the same layer is compared across all three arms despite three different operand layouts. `calls` is per sample.

> **One correction to read the fp16 column with.** The replay runs under `autocast(fp16)`, exactly as production does, and the captured arguments are what the caller passed *before* autocast cast them. For 12 of the fp16-arm conv records the activation arrives as **fp32**, so autocast's fp32→fp16 conversion of that activation is inside the timed region while the arithmetic is fp16 either way. Checked on the clearest case: `[128,1152,8,8]` fp32 is 37.7 MB read + 18.9 MB written ≈ 94 µs at ~600 GB/s, and that row's fp16-vs-quantized gap is 102.5 µs on a kernel that is `torch_conv2d_fp16` in all three arms. So rows marked `fp32-in` measure a conversion plus a conv, and their speedups are not arithmetic-only. The summary below is split on this.

| K | C | R×S | HxW | fp16 in | calls | fp16 µs | int8 µs | int4 µs | int8 | int4 | int8→int4 |
|--:|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| 384 | 384 | 3×3 | 32×32 | **fp32** | 10 | 3797.4 | 1689.2 | 834.6 | **2.25×** | **4.55×** | 2.02× |
| 192 | 576 | 3×3 | 32×32 | **fp32** | 5 | 3280.9 | 1674.8 | 837.5 | **1.96×** | **3.92×** | 2.00× |
| 384 | 768 | 3×3 | 16×16 | **fp32** | 10 | 1891.4 | 811.5 | 403.7 | **2.33×** | **4.69×** | 2.01× |
| 192 | 384 | 3×3 | 32×32 | fp16 | 10 | 1783.6 | 1055.3 | 524.6 | **1.69×** | **3.40×** | 2.01× |
| 192 _(unquantized)_ | 576 | 1×1 | 32×32 | **fp32** | 5 | 1381.5 | 580.9 | 580.6 | **2.38×** | **2.38×** | 1.00× |
| 384 | 576 | 3×3 | 16×16 | fp16 | 5 | 1269.7 | 678.8 | 345.1 | **1.87×** | **3.68×** | 1.97× |
| 192 | 192 | 3×3 | 32×32 | fp16 | 35 | 1030.6 | 753.9 | 354.5 | **1.37×** | **2.91×** | 2.13× |
| 768 | 768 | 3×3 | 8×8 | **fp32** | 10 | 926.2 | 406.9 | 194.4 | **2.28×** | **4.76×** | 2.09× |
| 384 | 384 | 3×3 | 16×16 | fp16 | 40 | 881.8 | 444.8 | 226.2 | **1.98×** | **3.90×** | 1.97× |
| 4 _(unquantized)_ | 192 | 3×3 | 32×32 | **fp32** | 5 | 829.8 | 830.2 | 829.9 | **1.00×** | **1.00×** | 1.00× |
| 384 | 1152 | 3×3 | 8×8 | **fp32** | 5 | 753.5 | 336.4 | 165.8 | **2.24×** | **4.55×** | 2.03× |
| 768 | 1536 | 3×3 | 4×4 | **fp32** | 10 | 600.7 | 277.3 | 139.4 | **2.17×** | **4.31×** | 1.99× |
| 384 _(unquantized)_ | 768 | 1×1 | 16×16 | **fp32** | 10 | 561.5 | 302.7 | 300.1 | **1.86×** | **1.87×** | 1.01× |
| 384 | 192 | 3×3 | 16×16 | fp16 | 5 | 497.2 | 310.6 | 148.5 | **1.60×** | **3.35×** | 2.09× |
| 192 _(unquantized)_ | 384 | 1×1 | 32×32 | fp16 | 10 | 493.3 | 494.0 | 493.0 | **1.00×** | **1.00×** | 1.00× |
| 384 | 768 | 3×3 | 8×8 | fp16 | 10 | 457.2 | 230.0 | 113.9 | **1.99×** | **4.01×** | 2.02× |
| 768 | 1152 | 3×3 | 4×4 | fp16 | 5 | 435.2 | 213.0 | 107.2 | **2.04×** | **4.06×** | 1.99× |
| 192 _(unquantized)_ | 4 | 3×3 | 32×32 | **fp32** | 5 | 308.2 | 308.2 | 308.4 | **1.00×** | **1.00×** | 1.00× |
| 768 | 768 | 3×3 | 4×4 | fp16 | 40 | 299.0 | 148.4 | 75.9 | **2.01×** | **3.94×** | 1.95× |
| 192 | 192 | 3×3 | 16×16 | fp16 | 10 | 278.5 | 211.5 | 104.7 | **1.32×** | **2.66×** | 2.02× |
| 384 _(unquantized)_ | 576 | 1×1 | 16×16 | fp16 | 5 | 258.1 | 259.2 | 259.4 | **1.00×** | **1.00×** | 1.00× |
| 384 | 384 | 3×3 | 8×8 | fp16 | 45 | 255.9 | 127.7 | 66.8 | **2.00×** | **3.83×** | 1.91× |
| 768 | 1536 | 3×3 | 2×2 | fp16 | 15 | 228.7 | 137.8 | 71.7 | **1.66×** | **3.19×** | 1.92× |
| 384 _(unquantized)_ | 1152 | 1×1 | 8×8 | **fp32** | 5 | 222.8 | 118.5 | 119.7 | **1.88×** | **1.86×** | 0.99× |
| 384 _(unquantized)_ | 192 | 1×1 | 16×16 | fp16 | 5 | 182.9 | 183.7 | 184.7 | **1.00×** | **0.99×** | 0.99× |
| 768 | 384 | 3×3 | 4×4 | fp16 | 5 | 162.1 | 79.9 | 41.3 | **2.03×** | **3.93×** | 1.94× |
| 768 | 768 | 3×3 | 2×2 | fp16 | 65 | 137.0 | 74.3 | 39.5 | **1.84×** | **3.47×** | 1.88× |
| 768 _(unquantized)_ | 1536 | 1×1 | 4×4 | **fp32** | 10 | 125.3 | 92.8 | 109.2 | **1.35×** | **1.15×** | 0.85× |
| 384 _(unquantized)_ | 768 | 1×1 | 8×8 | fp16 | 10 | 97.6 | 98.4 | 100.0 | **0.99×** | **0.98×** | 0.98× |
| 384 | 384 | 3×3 | 4×4 | fp16 | 10 | 96.8 | 41.2 | 26.5 | **2.35×** | **3.66×** | 1.56× |
| 768 _(unquantized)_ | 1152 | 1×1 | 4×4 | fp16 | 5 | 85.8 | 86.2 | 103.2 | **1.00×** | **0.83×** | 0.84× |
| 768 _(unquantized)_ | 384 | 1×1 | 4×4 | fp16 | 5 | 77.8 | 76.8 | 77.0 | **1.01×** | **1.01×** | 1.00× |
| 768 _(unquantized)_ | 1536 | 1×1 | 2×2 | fp16 | 15 | 75.0 | 74.9 | 90.1 | **1.00×** | **0.83×** | 0.83× |

33 layers matched in all three arms: 20 quantized, 13 unquantized controls.

| subset | n | int8 speedup | int4 speedup |
|---|--:|---|---|
| quantized, fp16 baseline also fp16-in — **the arithmetic-only number** | 14 | 1.32×–2.35× (median 1.98×) | 2.66×–4.06× (median 3.68×) |
| quantized, fp16 baseline fp32-in (includes an autocast cast) | 6 | 1.96×–2.33× (median 2.25×) | 3.92×–4.76× (median 4.55×) |
| all quantized | 20 | 1.32×–2.35× (median 2.00×) | 2.66×–4.76× (median 3.92×) |

**The controls work.** Of the 13 unquantized rows, the 7 whose activation dtype also matches across arms come out at 1.00×, 1.00×, 1.00×, 0.99×, 1.00×, 1.01×, 1.00× — the same kernel on the same input times the same in every arm, which is what shows the layout normalization is matching real layers and not coincidentally-shaped ones. The remaining 6 are the `fp32-in` rows above.

**13 of these rows are the control**: convs this pipeline does not quantize, so all three arms run the same `torch_conv2d_fp16` and the speedup must come out ≈1.00×. They do (2.38×, 1.00×, 1.86×, 1.00×, 1.00×, 1.00×, …), which is what shows the layout normalization is matching the right layers rather than coincidentally-shaped ones.

## 3. Per attention block

Matched on `T`, read from **K** (`[N,H,T,hd_pad]`), with the single-tensor packed-Q records assigned by their own `T`. N=128 and H=8 in every record, so they carry no information. Every record in the suite is assigned to exactly one row and the rows sum to the suite totals exactly — checked on every run. `ms/sample` is what the row costs; `µs/call` is **call-weighted** across the kernels in the row, because a row mixes dynamic and static variants at 10 and 5 calls.

| T | hd_pad fp16→int8/int4 | calls f/8/4 | fp16 ms | int8 ms | int4 ms | µs/call fp16→int8→int4 | int8 | int4 | noise |
|--:|---|---|--:|--:|--:|---|--:|--:|---|
| 1024 | 24→32/64 | 25/25/25 | 51.32 | 42.29 | 42.44 | 2052.9→1691.4→1697.7 | **1.21×** | **1.21×** | — |
| 256 | 48→64/64 | 25/25/25 | 8.73 | 6.42 | 5.22 | 349.0→257.0→208.9 | **1.36×** | **1.67×** | **int4 NOISY** |
| 64 | 48→64/64 | 25/25/25 | 2.28 | 1.08 | 0.92 | 91.3→43.2→36.6 | **2.11×** | **2.49×** | — |
| 16 | 96→96/96 | 25/25/25 | 1.22 | 1.40 | 1.40 | 48.9→55.9→56.1 | **0.87×** | **0.87×** | — |
| 4 | 96→96/96 | 5/5/5 | 0.24 | 0.19 | 0.22 | 48.7→37.5→43.7 | **1.30×** | **1.11×** | — |
| **total** | | | **63.79** | **51.38** | **50.20** | | **1.24×** | **1.27×** | |

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
| 131072 | 768 | 10 | 570.0 | 558.3 | **1.02×** | `gemm_w8a8_awq_qkv_i8_layouts` |
| 131072 | 640 | 15 | 440.2 | 379.4 | **1.16×** | `gemm_w8a8_awq_bias_res` |
| 131072 | 256 | 25 | 343.3 | 325.8 | **1.05×** | `gemm_w8a8_awq_bias_res` |
| 32768 | 1152 | 25 | 297.4 | 201.2 | **1.48×** | `gemm_w8a8_awq_bias_res` |
| 32768 | 384 | 25 | 189.6 | 171.3 | **1.11×** | `gemm_w8a8_awq_bias_res` |
| 8192 | 1152 | 25 | 86.0 | 60.9 | **1.41×** | `gemm_w8a8_awq_bias_res` |
| 8192 | 384 | 25 | 67.4 | 62.3 | **1.08×** | `gemm_w8a8_awq_bias_res` |
| 2048 | 2304 | 25 | 59.0 | 40.4 | **1.46×** | `gemm_w8a8_awq_bias_res` |
| 2048 | 768 | 25 | 46.3 | 37.0 | **1.25×** | `gemm_w8a8_awq_bias_res` |
| 512 | 768 | 5 | 29.0 | 30.0 | **0.97×** | `gemm_w8a8_awq_bias_res` |
| 512 | 2304 | 5 | 27.2 | 22.8 | **1.19×** | `gemm_w8a8_awq_bias_res` |

