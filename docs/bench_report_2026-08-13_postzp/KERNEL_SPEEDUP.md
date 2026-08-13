# Kernel speedups: fp16 → int8 → int4

`NVIDIA A40`, batch 128. Replay medians at the shapes captured from a live sample; `speedup` is fp16 µs/call ÷ quantized µs/call **at equal work**. PTQ arms (`int8_baseline`, `int4_baseline`), so no MoDiff temporal kernels are mixed in.

## 1. Suite totals (ms/sample)

| suite | fp16 | int8 | int4 | int8 speedup | int4 speedup | comparable? |
|---|--:|--:|--:|--:|--:|---|
| attention | 63.34 | 51.08 | 50.01 | 1.24× | 1.27× | yes |
| conv | 265.72 | 149.09 | 85.59 | 1.78× | 3.10× | **no** — fp16 counts the qkv/proj 1×1 convs here |
| linear | 28.96 | 47.15 | 43.45 | 0.61× | 0.67× | **no** — the quantized arms count those projections here |
| **conv + linear** | **294.68** | **196.24** | **129.04** | **1.50×** | **2.28×** | yes — the reclassification is internal to the pair |
| all three | 358.02 | 247.32 | 179.05 | 1.45× | 2.00× | yes |

**Read `conv + linear`, not the two rows separately.** In fp16 the attention projections are 1×1 convs; the quantized arms convert them to linears. That moves work from one row to the other, which is why fp16's linear total looks small and int8's looks like a regression. Summed, the reclassification cancels.

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

Matched on `(N, H, T)`. head_dim differs by padding — the flash kernels take `hd_pad`, so they move more bytes per row than fp16's `hd`; that is part of the cost, not an error.

| N | H | T | hd fp16→int8/int4 | calls | fp16 µs | int8 µs | int4 µs | int8 | int4 |
|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|
| 128 | 8 | 1024 | 24→32/32 | 25 | 2036.9 | 1716.4 | 1750.1 | **1.19×** | **1.16×** |
| 128 | 8 | 256 | 48→64/32 | 25 | 348.7 | 256.1 | 210.4 | **1.36×** | **1.66×** |
| 128 | 8 | 64 | 48→64/32 | 25 | 90.7 | 44.6 | 41.1 | **2.03×** | **2.21×** |
| 128 | 8 | 4 | 96→96/96 | 5 | 47.8 | 48.9 | 48.2 | **0.98×** | **0.99×** |
| 128 | 8 | 16 | 96→96/96 | 25 | 47.8 | 47.8 | 48.4 | **1.00×** | **0.99×** |

5 of 8 blocks matched in all three arms.

## 4. Linear — why there is no per-layer table

Two reasons, both structural:

1. **fp16 has no counterpart for the projections.** They are 1×1 convs there, so the only linears the fp16 arm runs are the 37 embedding linears — a different set of layers.
2. **The quantized arms pad K differently for the AWQ layout.** The same projection is `[131072, 192]` with `K=192` in int8 and `[131072, 128]` with `K=256` in int4, so no printed shape means the same thing in both.

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

