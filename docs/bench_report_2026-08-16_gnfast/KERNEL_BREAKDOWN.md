# Per-kernel breakdown: attention, conv, linear

`NVIDIA A40`, batch 128, replayed at the shapes captured from a live sample (8 rounds x 60 iters, median of round medians). Generated from `data/kernel_suites.json` -- see the script header for what `ms/sample` is and is not.

`ms/sample = us/call x calls/sample / 1000`. Both factors are shown because they point at different fixes: a fat kernel wants a better tile, a frequent one wants fusion.

## Reading the kernel names

Every kernel here is one of three things: an **unquantized fp16 fallback**, a **quantized compute kernel**, or the same compute kernel with a **different epilogue fused onto it**. The suffixes say which:

| suffix | what it means |
|---|---|
| `_fprop` | plain CUTLASS implicit-GEMM conv; the epilogue does dequant only, caller adds bias |
| `_evt_*` | an Epilogue Visitor Tree hand-assembled onto the conv Mma (CUTLASS 4.6.1 has no EVT-on-conv path), so bias/residual/o_hat fold into the conv's own store and no post-conv scratch tensor is ever written |
| `_vt` | V arrives PRE-TRANSPOSED as [N,H,hd_pad,T], straight from the qkv quantize, so the kernel skips a transpose |
| `_static` | Q and K each use ONE frozen calibrated scale, folded into the row scale. Removes both per-token scale tensors, their cp.async staging, and one fp32 multiply per score element from the hot loop |
| `_qout` | the epilogue writes the PROJECTION-QUANTIZED int8 output directly, fusing the next projection's input quantize. Mutually exclusive with MoDiff's fp16 o_hat state, so UNUSABLE under MoDiff -- all 21 blocks report qout_eligible == 0 |
| `_hd24` | exact specialization for the dominant T=1024 / hd=24 route: three PV/output fragments instead of the generic HD_PAD=32 kernel's, plus vectorized 24-byte compact-Q staging |
| `_small` | the staging variant that wins at small T (NNT = BC/8 halves) |
| `qi8` / `qpacked` | how Q is staged into the kernel -- plain int8 rows vs packed |
| `i4values_i8mma` | int4 V values fed through the int8 tensor-core MMA path |
| `_bias_res` | bias + residual epilogue. Under MoDiff an EMPTY residual returns o_hat itself; a given one also returns o_hat_t + residual as a SEPARATE tensor, because the ResBlock/attention skip must not be folded into the temporal state |
| `_o_hat` | MoDiff's temporal accumulate: o_hat[elem] += this step's contribution, in place, fp16 |
| `_out_i8` / `_codes` | emits int8/int4 CODES rather than dequantized values, so the next consumer (flash attention) reads them directly |
| `_layouts` | the fused qkv projection writes Q/K/V already in the attention kernel's per-head padded layouts, returning several tensors, so no separate reformat runs |

Descriptions below are taken from the kernels' own header comments in `csrc/baseline/conv/conv2d_evt.cu`, `csrc/baseline/attention/flash_attn_int8.cu`, `csrc/{baseline,modiff}/linear/gemm_wxax.cu` and `csrc/modiff_kernels_api.h`.

## attention

### fp16 — 63.79 ms/sample total  (REPORT.md: 63.79 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `torch_sdpa_fp16` | **63.79** | 100.0% | 105 | 607.6 | 5 | 1.55% |

- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.

<details><summary>signatures ≥ 4% of the suite (2 of 5)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 51.32 | 25 | 2052.9 | `[128,8,1024,24] x [128,8,1024,24]` | `torch_sdpa_fp16` |
| 8.73 | 25 | 349.0 | `[128,8,256,48] x [128,8,256,48]` | `torch_sdpa_fp16` |

</details>

### W8A8 PTQ — 51.38 ms/sample total  (REPORT.md: 51.38 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `flash_attn_int8_vt` | **21.57** | 42.0% | 30 | 719.0 | 3 | 2.59% |
| `flash_attn_int8_qi8_kv_static_qout_hd24` | **15.78** | 30.7% | 10 | 1578.4 | 1 | 1.51% |
| `flash_attn_int8_vt_static` | **9.46** | 18.4% | 15 | 630.8 | 3 | 2.09% |
| `flash_attn_int8_qi8_kv_static_qout` | **2.98** | 5.8% | 20 | 148.8 | 2 | 1.17% |
| `torch_sdpa_fp16` | **0.89** | 1.7% | 18 | 49.4 | 2 | 0.55% |
| `flash_attn_int8_qi8packed_small_qout` | **0.70** | 1.4% | 12 | 58.1 | 2 | 0.37% |

- **`flash_attn_int8_vt`** — fused int8 flash attention, V pre-transposed. Keeps the running softmax state in registers and never writes the T x T score matrix; QK^T via __dp4a int8x4->int32, AV accumulated in fp32 so P is never requantized. Softmax is always fp32.
- **`flash_attn_int8_qi8_kv_static_qout_hd24`** — the hd=24 exact specialization of the above. One signature, 10 calls, and ~31% of the attention suite -- the single most expensive call in it.
- **`flash_attn_int8_vt_static`** — the same with frozen Q/K scales -- the production steady state.
- **`flash_attn_int8_qi8_kv_static_qout`** — int8 flash with static K/V scales, emitting the projection-quantized int8 output.
- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.
- **`flash_attn_int8_qi8packed_small_qout`** — the small-T staging variant, packed Q.

<details><summary>signatures ≥ 4% of the suite (5 of 13)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.52 | 10 | 1851.6 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int8_vt` |
| 15.78 | 10 | 1578.4 | `[128,1024,8,32] x [128,8,1024,32]` | `flash_attn_int8_qi8_kv_static_qout_hd24` |
| 7.99 | 5 | 1597.1 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int8_vt_static` |
| 2.61 | 10 | 260.7 | `[128,8,256,64] x [128,8,256,64]` | `flash_attn_int8_vt` |
| 2.57 | 10 | 256.5 | `[128,256,8,48] x [128,8,256,64]` | `flash_attn_int8_qi8_kv_static_qout` |

</details>

### W8A8 MoDiff — 51.40 ms/sample total  (REPORT.md: 51.40 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `flash_attn_int8_vt` | **21.68** | 42.2% | 30 | 722.8 | 3 | 3.19% |
| `flash_attn_int8_qi8_kv_static_qout_hd24` | **15.71** | 30.6% | 10 | 1570.5 | 1 | 1.69% |
| `flash_attn_int8_vt_static` | **9.45** | 18.4% | 15 | 630.3 | 3 | 1.86% |
| `flash_attn_int8_qi8_kv_static_qout` | **2.98** | 5.8% | 20 | 149.0 | 2 | 0.94% |
| `torch_sdpa_fp16` | **0.88** | 1.7% | 18 | 48.8 | 2 | 1.80% |
| `flash_attn_int8_qi8packed_small_qout` | **0.70** | 1.4% | 12 | 58.1 | 2 | 0.45% |

- **`flash_attn_int8_vt`** — fused int8 flash attention, V pre-transposed. Keeps the running softmax state in registers and never writes the T x T score matrix; QK^T via __dp4a int8x4->int32, AV accumulated in fp32 so P is never requantized. Softmax is always fp32.
- **`flash_attn_int8_qi8_kv_static_qout_hd24`** — the hd=24 exact specialization of the above. One signature, 10 calls, and ~31% of the attention suite -- the single most expensive call in it.
- **`flash_attn_int8_vt_static`** — the same with frozen Q/K scales -- the production steady state.
- **`flash_attn_int8_qi8_kv_static_qout`** — int8 flash with static K/V scales, emitting the projection-quantized int8 output.
- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.
- **`flash_attn_int8_qi8packed_small_qout`** — the small-T staging variant, packed Q.

<details><summary>signatures ≥ 4% of the suite (5 of 13)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.63 | 10 | 1862.6 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int8_vt` |
| 15.71 | 10 | 1570.5 | `[128,1024,8,32] x [128,8,1024,32]` | `flash_attn_int8_qi8_kv_static_qout_hd24` |
| 7.99 | 5 | 1597.3 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int8_vt_static` |
| 2.61 | 10 | 261.0 | `[128,8,256,64] x [128,8,256,64]` | `flash_attn_int8_vt` |
| 2.57 | 10 | 257.0 | `[128,256,8,48] x [128,8,256,64]` | `flash_attn_int8_qi8_kv_static_qout` |

</details>

### W4A4 PTQ — 50.20 ms/sample total  (REPORT.md: 50.20 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `flash_attn_int4_vt` | **21.37** | 42.6% | 30 | 712.5 | 3 | 3.23% |
| `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24` | **15.38** | 30.6% | 10 | 1538.1 | 1 | 0.36% |
| `flash_attn_int4_vt_static` | **9.56** | 19.0% | 15 | 637.0 | 3 | 2.21% |
| `flash_attn_int4_vt_static_qout` | **2.27** | 4.5% | 20 | 113.5 | 2 | 3.06% |
| `torch_sdpa_fp16` | **0.92** | 1.8% | 18 | 51.2 | 2 | 1.41% |
| `flash_attn_i4values_small_qout` | **0.70** | 1.4% | 12 | 58.2 | 2 | 0.41% |

- **`flash_attn_int4_vt`** — int4 flash attention, V pre-transposed. W4A4's counterpart to flash_attn_int8_vt and, at ~42%, the largest single item in its suite.
- **`flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24`** — int4 V values through the int8 MMA path, hd=24 exact specialization, int8-code output -- W4A4's twin of the int8 hd24 kernel.
- **`flash_attn_int4_vt_static`** — int4 flash with frozen Q/K scales.
- **`flash_attn_int4_vt_static_qout`** — int4 flash, frozen scales, int8-code output.
- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.
- **`flash_attn_i4values_small_qout`** — int4 values, small-T variant, int8-code output.

<details><summary>signatures ≥ 4% of the suite (4 of 13)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.75 | 10 | 1875.2 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int4_vt` |
| 15.38 | 10 | 1538.1 | `[128,1024,8,32] x [128,8,1024,32]` | `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24` |
| 8.31 | 5 | 1661.8 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int4_vt_static` |
| 2.23 | 10 | 222.7 | `[128,8,256,32] x [128,8,256,32]` | `flash_attn_int4_vt` |

</details>

### W4A4 MoDiff — 50.25 ms/sample total  (REPORT.md: 50.25 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `flash_attn_int4_vt` | **21.28** | 42.3% | 30 | 709.3 | 3 | 3.27% |
| `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24` | **15.54** | 30.9% | 10 | 1554.1 | 1 | 0.81% |
| `flash_attn_int4_vt_static` | **9.56** | 19.0% | 15 | 637.4 | 3 | 1.46% |
| `flash_attn_int4_vt_static_qout` | **2.28** | 4.5% | 20 | 114.0 | 2 | 2.99% |
| `torch_sdpa_fp16` | **0.89** | 1.8% | 18 | 49.3 | 2 | 1.37% |
| `flash_attn_i4values_small_qout` | **0.70** | 1.4% | 12 | 58.3 | 2 | 0.96% |

- **`flash_attn_int4_vt`** — int4 flash attention, V pre-transposed. W4A4's counterpart to flash_attn_int8_vt and, at ~42%, the largest single item in its suite.
- **`flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24`** — int4 V values through the int8 MMA path, hd=24 exact specialization, int8-code output -- W4A4's twin of the int8 hd24 kernel.
- **`flash_attn_int4_vt_static`** — int4 flash with frozen Q/K scales.
- **`flash_attn_int4_vt_static_qout`** — int4 flash, frozen scales, int8-code output.
- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.
- **`flash_attn_i4values_small_qout`** — int4 values, small-T variant, int8-code output.

<details><summary>signatures ≥ 4% of the suite (4 of 13)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.65 | 10 | 1865.2 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int4_vt` |
| 15.54 | 10 | 1554.1 | `[128,1024,8,32] x [128,8,1024,32]` | `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24` |
| 8.31 | 5 | 1662.4 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int4_vt_static` |
| 2.24 | 10 | 223.5 | `[128,8,256,32] x [128,8,256,32]` | `flash_attn_int4_vt` |

</details>

## conv

### fp16 — 268.10 ms/sample total  (REPORT.md: 268.10 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `torch_conv2d_fp16` | **268.10** | 100.0% | 445 | 602.5 | 33 | 3.09% |

- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).

<details><summary>signatures ≥ 4% of the suite (8 of 33)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 37.97 | 10 | 3797.4 | `[128,384,32,32] x [384,384,3,3]` | `torch_conv2d_fp16` |
| 36.07 | 35 | 1030.6 | `[128,192,32,32] x [192,192,3,3]` | `torch_conv2d_fp16` |
| 35.27 | 40 | 881.8 | `[128,384,16,16] x [384,384,3,3]` | `torch_conv2d_fp16` |
| 18.91 | 10 | 1891.4 | `[128,768,16,16] x [384,768,3,3]` | `torch_conv2d_fp16` |
| 17.84 | 10 | 1783.6 | `[128,384,32,32] x [192,384,3,3]` | `torch_conv2d_fp16` |
| 16.40 | 5 | 3280.9 | `[128,576,32,32] x [192,576,3,3]` | `torch_conv2d_fp16` |
| 11.96 | 40 | 299.0 | `[128,768,4,4] x [768,768,3,3]` | `torch_conv2d_fp16` |
| 11.51 | 45 | 255.9 | `[128,384,8,8] x [384,384,3,3]` | `torch_conv2d_fp16` |

</details>

### W8A8 PTQ — 150.19 ms/sample total  (REPORT.md: 150.19 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `conv2d_int8_evt_bias_residual_fp16` | **126.97** | 84.5% | 350 | 362.8 | 29 | 2.96% |
| `torch_conv2d_fp16` | **23.22** | 15.5% | 95 | 244.4 | 13 | 1.55% |

- **`conv2d_int8_evt_bias_residual_fp16`** — D1 fusion: out = acc*alpha*weight_scale[k] + bias[k] + residual[elem] -> fp16, in the conv's own store. This is the PTQ arm's whole conv datapath.
- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).

<details><summary>signatures ≥ 4% of the suite (8 of 42)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 19.26 | 25 | 770.6 | `[128,192,32,32] x [192,3,3,192]` | `conv2d_int8_evt_bias_residual_fp16` |
| 13.60 | 30 | 453.4 | `[128,384,16,16] x [384,3,3,384]` | `conv2d_int8_evt_bias_residual_fp16` |
| 10.55 | 10 | 1055.3 | `[128,384,32,32] x [192,3,3,384]` | `conv2d_int8_evt_bias_residual_fp16` |
| 8.65 | 5 | 1729.1 | `[128,384,32,32] x [384,3,3,384]` | `conv2d_int8_evt_bias_residual_fp16` |
| 8.37 | 5 | 1674.8 | `[128,576,32,32] x [192,3,3,576]` | `conv2d_int8_evt_bias_residual_fp16` |
| 8.25 | 5 | 1649.4 | `[128,384,32,32] x [384,3,3,384]` | `conv2d_int8_evt_bias_residual_fp16` |
| 8.11 | 10 | 811.5 | `[128,768,16,16] x [384,3,3,768]` | `conv2d_int8_evt_bias_residual_fp16` |
| 7.37 | 10 | 737.2 | `[128,192,32,32] x [192,3,3,192]` | `conv2d_int8_evt_bias_residual_fp16` |

</details>

### W8A8 MoDiff — 269.13 ms/sample total  (REPORT.md: 269.13 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `conv2d_int8_fprop` | **131.23** | 48.8% | 350 | 374.9 | 20 | 2.31% |
| `conv2d_int8_evt_o_hat` | **58.49** | 21.7% | 140 | 417.8 | 20 | 2.92% |
| `conv2d_int8_evt_o_hat_residual` | **48.76** | 18.1% | 140 | 348.3 | 9 | 2.40% |
| `torch_conv2d_fp16` | **30.64** | 11.4% | 95 | 322.5 | 13 | 1.09% |

- **`conv2d_int8_fprop`** — int8 x int8 conv, plain output. On the MoDiff arm this is the t=T conv and the delta-step conv whose accumulate is done by a separate epilogue.
- **`conv2d_int8_evt_o_hat`** — D2 fusion without a skip: o_hat[elem] += acc*alpha*weight_scale[k], in place in fp16. MoDiff's temporal state advance (paper Eq 9).
- **`conv2d_int8_evt_o_hat_residual`** — D2 DUAL STORE: advances o_hat in place AND writes out = o_hat_new + residual[elem] -> fp16, one pass, two stores. Replaces an fp32 conv_out round-trip.
- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).

<details><summary>signatures ≥ 4% of the suite (6 of 62)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 27.89 | 35 | 796.9 | `[128,192,32,32] x [192,3,3,192]` | `conv2d_int8_fprop` |
| 18.47 | 40 | 461.7 | `[128,384,16,16] x [384,3,3,384]` | `conv2d_int8_fprop` |
| 17.48 | 10 | 1748.5 | `[128,384,32,32] x [384,3,3,384]` | `conv2d_int8_fprop` |
| 16.89 | 20 | 844.3 | `[128,192,32,32] x [192,3,3,192]` | `conv2d_int8_evt_o_hat_residual` |
| 11.76 | 24 | 490.2 | `[128,384,16,16] x [384,3,3,384]` | `conv2d_int8_evt_o_hat_residual` |
| 11.07 | 10 | 1106.7 | `[128,384,32,32] x [192,3,3,384]` | `conv2d_int8_fprop` |

</details>

### W4A4 PTQ — 86.69 ms/sample total  (REPORT.md: 86.69 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `conv2d_int4_evt_bias_residual_fp16` | **63.00** | 72.7% | 350 | 180.0 | 29 | 3.33% |
| `torch_conv2d_fp16` | **23.69** | 27.3% | 95 | 249.4 | 13 | 1.29% |

- **`conv2d_int4_evt_bias_residual_fp16`** — D1 fusion, int4. The PTQ arm's whole conv datapath.
- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).

<details><summary>signatures ≥ 4% of the suite (10 of 42)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 9.04 | 25 | 361.7 | `[128,32,32,96] x [192,3,3,96]` | `conv2d_int4_evt_bias_residual_fp16` |
| 6.88 | 30 | 229.2 | `[128,16,16,192] x [384,3,3,192]` | `conv2d_int4_evt_bias_residual_fp16` |
| 5.25 | 10 | 524.6 | `[128,32,32,192] x [192,3,3,192]` | `conv2d_int4_evt_bias_residual_fp16` |
| 4.93 | 10 | 493.0 | `[128,384,32,32] x [192,384,1,1]` | `torch_conv2d_fp16` |
| 4.28 | 5 | 855.2 | `[128,32,32,192] x [384,3,3,192]` | `conv2d_int4_evt_bias_residual_fp16` |
| 4.19 | 5 | 837.5 | `[128,32,32,288] x [192,3,3,288]` | `conv2d_int4_evt_bias_residual_fp16` |
| 4.15 | 5 | 829.9 | `[128,192,32,32] x [4,192,3,3]` | `torch_conv2d_fp16` |
| 4.07 | 5 | 814.0 | `[128,32,32,192] x [384,3,3,192]` | `conv2d_int4_evt_bias_residual_fp16` |
| 4.04 | 10 | 403.7 | `[128,16,16,384] x [384,3,3,384]` | `conv2d_int4_evt_bias_residual_fp16` |
| 3.47 | 10 | 347.3 | `[128,32,32,96] x [192,3,3,96]` | `conv2d_int4_evt_bias_residual_fp16` |

</details>

### W4A4 MoDiff — 156.28 ms/sample total  (REPORT.md: 156.28 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `conv2d_int4_fprop` | **68.44** | 43.8% | 350 | 195.5 | 20 | 2.60% |
| `torch_conv2d_fp16` | **30.61** | 19.6% | 95 | 322.3 | 13 | 1.20% |
| `conv2d_int4_evt_o_hat` | **28.94** | 18.5% | 140 | 206.7 | 20 | 3.53% |
| `conv2d_int4_evt_o_hat_residual` | **28.28** | 18.1% | 140 | 202.0 | 9 | 1.53% |

- **`conv2d_int4_fprop`** — int4 x int4 conv, plain output; same role as the int8 twin.
- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).
- **`conv2d_int4_evt_o_hat`** — D2 fusion without a skip, int4. MoDiff's temporal state advance.
- **`conv2d_int4_evt_o_hat_residual`** — D2 dual store, int4.

<details><summary>signatures ≥ 4% of the suite (6 of 62)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 13.62 | 35 | 389.1 | `[128,32,32,96] x [192,3,3,96]` | `conv2d_int4_fprop` |
| 10.26 | 40 | 256.4 | `[128,16,16,192] x [384,3,3,192]` | `conv2d_int4_fprop` |
| 9.52 | 20 | 476.0 | `[128,32,32,96] x [192,3,3,96]` | `conv2d_int4_evt_o_hat_residual` |
| 8.83 | 10 | 883.2 | `[128,32,32,192] x [384,3,3,192]` | `conv2d_int4_fprop` |
| 7.23 | 24 | 301.1 | `[128,16,16,192] x [384,3,3,192]` | `conv2d_int4_evt_o_hat_residual` |
| 6.91 | 5 | 1381.2 | `[128,576,32,32] x [192,576,1,1]` | `torch_conv2d_fp16` |

</details>

## linear

### fp16 — 61.85 ms/sample total  (REPORT.md: 61.85 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `fused_gn_qkv` | **32.00** | 51.7% | 50 | 640.0 | 2 | 0.70% |
| `torch_linear_fp16` | **29.85** | 48.3% | 345 | 86.5 | 12 | 1.76% |

- **`fused_gn_qkv`** — UNQUANTIZED, fp16 ONLY -- the qkv projection with the GroupNorm folded into its mainloop as a per-sample scale/bias, so the normalized activation is never written to HBM. Taken by the fp16 arm only, and only where T % 128 == 0 and c % 8 == 0, which is exactly the T=1024 and T=256 blocks; the smaller ones fall to plain GroupNorm + `torch_linear_fp16`. 52% of fp16's linear suite. It has NO counterpart in the quantized arms, which split the same work into a `norm_quantize` group_norm_silu_quantize_nhwc plus an AWQ GEMM here -- so the linear and norm_quantize suite totals do not compare across arms. docs/OPEN_ITEMS.md A1.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.

<details><summary>signatures ≥ 4% of the suite (7 of 14)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.73 | 25 | 749.3 | `[128,192,32,32] x [576,1,1,192]` | `fused_gn_qkv` |
| 13.27 | 25 | 530.8 | `[128,384,16,16] x [1152,1,1,384]` | `fused_gn_qkv` |
| 5.04 | 25 | 201.4 | `[128,1024,192] x [192,192]` | `torch_linear_fp16` |
| 4.89 | 75 | 65.2 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 4.67 | 75 | 62.3 | `[128,768] x [1536,768]` | `torch_linear_fp16` |
| 3.40 | 25 | 135.9 | `[128,256,384] x [384,384]` | `torch_linear_fp16` |
| 2.69 | 25 | 107.5 | `[128,16,768] x [2304,768]` | `torch_linear_fp16` |

</details>

### W8A8 PTQ — 47.46 ms/sample total  (REPORT.md: 47.47 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `gemm_w8a8_awq_bias_res` | **29.26** | 61.6% | 168 | 174.2 | 10 | 2.02% |
| `torch_linear_fp16` | **7.78** | 16.4% | 185 | 42.1 | 4 | 0.51% |
| `gemm_w8a8_awq_qkv_i8_layouts` | **5.70** | 12.0% | 10 | 570.0 | 1 | 1.21% |
| `gemm_w8a8_awq_qkv_i8_layouts_compact` | **4.13** | 8.7% | 20 | 206.3 | 2 | 2.03% |
| `gemm_w8a8_awq_out_i8_bias_nout` | **0.60** | 1.3% | 12 | 49.9 | 2 | 1.01% |

- **`gemm_w8a8_awq_bias_res`** — W8A8 AWQ-layout GEMM with the bias+residual epilogue. `a_scale` is a 1-ELEMENT DEVICE TENSOR, not a double, because MoDiff's delta scale is produced on device each call and taking it by value would force a host sync per linear per step.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.
- **`gemm_w8a8_awq_qkv_i8_layouts`** — fused qkv projection: one GEMM writing Q, K and V already in the attention kernel's per-head padded layouts as int8.
- **`gemm_w8a8_awq_qkv_i8_layouts_compact`** — the compact-staging variant of the above.
- **`gemm_w8a8_awq_out_i8_bias_nout`** — W8A8 GEMM emitting int8 codes of (out + bias) at a per-column scale, so a projection can feed flash attention without a separate quantize.

<details><summary>signatures ≥ 4% of the suite (8 of 19)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.58 | 25 | 343.3 | `[131072,192] x [256,192]` | `gemm_w8a8_awq_bias_res` |
| 6.60 | 15 | 440.2 | `[131072,192] x [640,192]` | `gemm_w8a8_awq_bias_res` |
| 5.70 | 10 | 570.0 | `[131072,192] x [768,192]` | `gemm_w8a8_awq_qkv_i8_layouts` |
| 4.74 | 25 | 189.6 | `[32768,384] x [384,384]` | `gemm_w8a8_awq_bias_res` |
| 4.11 | 15 | 273.9 | `[32768,384] x [1152,384]` | `gemm_w8a8_awq_bias_res` |
| 3.65 | 75 | 48.6 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 3.21 | 10 | 320.8 | `[32768,384] x [1152,384]` | `gemm_w8a8_awq_qkv_i8_layouts_compact` |
| 2.70 | 75 | 36.0 | `[128,768] x [1536,768]` | `torch_linear_fp16` |

</details>

### W8A8 MoDiff — 46.80 ms/sample total  (REPORT.md: 46.80 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `gemm_w8a8_awq_bias_res` | **29.21** | 62.4% | 168 | 173.9 | 10 | 2.05% |
| `torch_linear_fp16` | **7.19** | 15.4% | 185 | 38.8 | 4 | 0.86% |
| `gemm_w8a8_awq_qkv_i8_layouts` | **5.70** | 12.2% | 10 | 570.1 | 1 | 1.15% |
| `gemm_w8a8_awq_qkv_i8_layouts_compact` | **4.10** | 8.8% | 20 | 205.2 | 2 | 1.13% |
| `gemm_w8a8_awq_out_i8_bias_nout` | **0.60** | 1.3% | 12 | 49.9 | 2 | 1.02% |

- **`gemm_w8a8_awq_bias_res`** — W8A8 AWQ-layout GEMM with the bias+residual epilogue. `a_scale` is a 1-ELEMENT DEVICE TENSOR, not a double, because MoDiff's delta scale is produced on device each call and taking it by value would force a host sync per linear per step.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.
- **`gemm_w8a8_awq_qkv_i8_layouts`** — fused qkv projection: one GEMM writing Q, K and V already in the attention kernel's per-head padded layouts as int8.
- **`gemm_w8a8_awq_qkv_i8_layouts_compact`** — the compact-staging variant of the above.
- **`gemm_w8a8_awq_out_i8_bias_nout`** — W8A8 GEMM emitting int8 codes of (out + bias) at a per-column scale, so a projection can feed flash attention without a separate quantize.

<details><summary>signatures ≥ 4% of the suite (8 of 19)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.57 | 25 | 342.7 | `[131072,192] x [256,192]` | `gemm_w8a8_awq_bias_res` |
| 6.59 | 15 | 439.3 | `[131072,192] x [640,192]` | `gemm_w8a8_awq_bias_res` |
| 5.70 | 10 | 570.1 | `[131072,192] x [768,192]` | `gemm_w8a8_awq_qkv_i8_layouts` |
| 4.73 | 25 | 189.4 | `[32768,384] x [384,384]` | `gemm_w8a8_awq_bias_res` |
| 4.11 | 15 | 273.9 | `[32768,384] x [1152,384]` | `gemm_w8a8_awq_bias_res` |
| 3.19 | 10 | 318.9 | `[32768,384] x [1152,384]` | `gemm_w8a8_awq_qkv_i8_layouts_compact` |
| 3.05 | 75 | 40.7 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 2.73 | 75 | 36.4 | `[128,768] x [1536,768]` | `torch_linear_fp16` |

</details>

### W4A4 PTQ — 42.83 ms/sample total  (REPORT.md: 42.83 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `gemm_w4a4_awq_bias_res` | **25.41** | 59.3% | 168 | 151.3 | 10 | 0.69% |
| `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` | **9.82** | 22.9% | 30 | 327.2 | 3 | 5.08% |
| `torch_linear_fp16` | **7.19** | 16.8% | 185 | 38.9 | 4 | 1.17% |
| `gemm_w4a4_awq_qkv_codes` | **0.41** | 1.0% | 12 | 34.3 | 2 | 0.26% |

- **`gemm_w4a4_awq_bias_res`** — W4A4 AWQ-layout GEMM, bias+residual epilogue. The linear suite's largest item on both W4A4 arms.
- **`gemm_w4a4_awq_qkv_i4qk_i8v_layouts`** — fused qkv projection emitting int4 Q/K and int8 V in the attention layouts -- the asymmetry is deliberate: V's dot product accumulates in fp32, so it keeps 8 bits while Q/K drop to 4.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.
- **`gemm_w4a4_awq_qkv_codes`** — emits the qkv int4 codes plus their clamp limits rather than dequantized values.

<details><summary>signatures ≥ 4% of the suite (8 of 19)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.15 | 25 | 325.8 | `[131072,128] x [256,128]` | `gemm_w4a4_awq_bias_res` |
| 5.69 | 15 | 379.4 | `[131072,128] x [640,128]` | `gemm_w4a4_awq_bias_res` |
| 5.58 | 10 | 558.3 | `[131072,128] x [768,128]` | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` |
| 4.28 | 25 | 171.3 | `[32768,192] x [384,192]` | `gemm_w4a4_awq_bias_res` |
| 3.28 | 10 | 328.4 | `[32768,192] x [1536,192]` | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` |
| 3.06 | 75 | 40.8 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 3.02 | 15 | 201.2 | `[32768,192] x [1152,192]` | `gemm_w4a4_awq_bias_res` |
| 2.73 | 75 | 36.4 | `[128,768] x [1536,768]` | `torch_linear_fp16` |

</details>

### W4A4 MoDiff — 43.76 ms/sample total  (REPORT.md: 43.76 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `gemm_w4a4_awq_bias_res` | **25.41** | 58.1% | 168 | 151.3 | 10 | 1.53% |
| `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` | **9.78** | 22.4% | 30 | 326.1 | 3 | 4.02% |
| `torch_linear_fp16` | **8.15** | 18.6% | 185 | 44.1 | 4 | 1.39% |
| `gemm_w4a4_awq_qkv_codes` | **0.41** | 0.9% | 12 | 34.2 | 2 | 0.20% |

- **`gemm_w4a4_awq_bias_res`** — W4A4 AWQ-layout GEMM, bias+residual epilogue. The linear suite's largest item on both W4A4 arms.
- **`gemm_w4a4_awq_qkv_i4qk_i8v_layouts`** — fused qkv projection emitting int4 Q/K and int8 V in the attention layouts -- the asymmetry is deliberate: V's dot product accumulates in fp32, so it keeps 8 bits while Q/K drop to 4.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.
- **`gemm_w4a4_awq_qkv_codes`** — emits the qkv int4 codes plus their clamp limits rather than dequantized values.

<details><summary>signatures ≥ 4% of the suite (8 of 19)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.10 | 25 | 324.0 | `[131072,128] x [256,128]` | `gemm_w4a4_awq_bias_res` |
| 5.75 | 15 | 383.2 | `[131072,128] x [640,128]` | `gemm_w4a4_awq_bias_res` |
| 5.58 | 10 | 558.1 | `[131072,128] x [768,128]` | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` |
| 4.28 | 25 | 171.1 | `[32768,192] x [384,192]` | `gemm_w4a4_awq_bias_res` |
| 3.70 | 75 | 49.3 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 3.26 | 10 | 325.6 | `[32768,192] x [1536,192]` | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` |
| 3.04 | 15 | 202.4 | `[32768,192] x [1152,192]` | `gemm_w4a4_awq_bias_res` |
| 2.76 | 75 | 36.8 | `[128,768] x [1536,768]` | `torch_linear_fp16` |

</details>

