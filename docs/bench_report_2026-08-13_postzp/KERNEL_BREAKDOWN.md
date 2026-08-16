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

### fp16 — 63.34 ms/sample total  (REPORT.md: 63.34 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `torch_sdpa_fp16` | **63.34** | 100.0% | 105 | 603.2 | 5 | 34.99% |

- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.

<details><summary>signatures ≥ 4% of the suite (2 of 5)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 50.92 | 25 | 2036.9 | `[128,8,1024,24] x [128,8,1024,24]` | `torch_sdpa_fp16` |
| 8.72 | 25 | 348.7 | `[128,8,256,48] x [128,8,256,48]` | `torch_sdpa_fp16` |

</details>

### W8A8 PTQ — 51.08 ms/sample total  (REPORT.md: 51.09 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `flash_attn_int8_vt` | **21.55** | 42.2% | 30 | 718.2 | 3 | 2.84% |
| `flash_attn_int8_qi8_kv_static_qout_hd24` | **15.63** | 30.6% | 10 | 1562.8 | 1 | 1.98% |
| `flash_attn_int8_vt_static` | **9.40** | 18.4% | 15 | 626.5 | 3 | 2.88% |
| `flash_attn_int8_qi8_kv_static_qout` | **2.96** | 5.8% | 20 | 148.2 | 2 | 1.10% |
| `torch_sdpa_fp16` | **0.86** | 1.7% | 18 | 48.0 | 2 | 1.96% |
| `flash_attn_int8_qi8packed_small_qout` | **0.68** | 1.3% | 12 | 57.0 | 2 | 0.30% |

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
| 15.63 | 10 | 1562.8 | `[128,1024,8,32] x [128,8,1024,32]` | `flash_attn_int8_qi8_kv_static_qout_hd24` |
| 7.91 | 5 | 1581.2 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int8_vt_static` |
| 2.59 | 10 | 258.6 | `[128,8,256,64] x [128,8,256,64]` | `flash_attn_int8_vt` |
| 2.55 | 10 | 255.4 | `[128,256,8,48] x [128,8,256,64]` | `flash_attn_int8_qi8_kv_static_qout` |

</details>

### W8A8 MoDiff — 50.85 ms/sample total  (REPORT.md: 50.85 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `flash_attn_int8_vt` | **21.38** | 42.0% | 30 | 712.6 | 3 | 2.58% |
| `flash_attn_int8_qi8_kv_static_qout_hd24` | **15.62** | 30.7% | 10 | 1562.1 | 1 | 2.09% |
| `flash_attn_int8_vt_static` | **9.33** | 18.3% | 15 | 622.0 | 3 | 3.02% |
| `flash_attn_int8_qi8_kv_static_qout` | **2.97** | 5.8% | 20 | 148.4 | 2 | 0.85% |
| `torch_sdpa_fp16` | **0.87** | 1.7% | 18 | 48.2 | 2 | 0.84% |
| `flash_attn_int8_qi8packed_small_qout` | **0.69** | 1.4% | 12 | 57.2 | 2 | 0.66% |

- **`flash_attn_int8_vt`** — fused int8 flash attention, V pre-transposed. Keeps the running softmax state in registers and never writes the T x T score matrix; QK^T via __dp4a int8x4->int32, AV accumulated in fp32 so P is never requantized. Softmax is always fp32.
- **`flash_attn_int8_qi8_kv_static_qout_hd24`** — the hd=24 exact specialization of the above. One signature, 10 calls, and ~31% of the attention suite -- the single most expensive call in it.
- **`flash_attn_int8_vt_static`** — the same with frozen Q/K scales -- the production steady state.
- **`flash_attn_int8_qi8_kv_static_qout`** — int8 flash with static K/V scales, emitting the projection-quantized int8 output.
- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.
- **`flash_attn_int8_qi8packed_small_qout`** — the small-T staging variant, packed Q.

<details><summary>signatures ≥ 4% of the suite (5 of 13)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.35 | 10 | 1835.0 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int8_vt` |
| 15.62 | 10 | 1562.1 | `[128,1024,8,32] x [128,8,1024,32]` | `flash_attn_int8_qi8_kv_static_qout_hd24` |
| 7.86 | 5 | 1572.8 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int8_vt_static` |
| 2.58 | 10 | 258.2 | `[128,8,256,64] x [128,8,256,64]` | `flash_attn_int8_vt` |
| 2.56 | 10 | 255.8 | `[128,256,8,48] x [128,8,256,64]` | `flash_attn_int8_qi8_kv_static_qout` |

</details>

### W4A4 PTQ — 50.01 ms/sample total  (REPORT.md: 50.01 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `flash_attn_int4_vt` | **21.18** | 42.4% | 30 | 706.0 | 3 | 3.49% |
| `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24` | **15.38** | 30.8% | 10 | 1538.3 | 1 | 0.35% |
| `flash_attn_int4_vt_static` | **9.49** | 19.0% | 15 | 632.4 | 3 | 2.79% |
| `flash_attn_int4_vt_static_qout` | **2.40** | 4.8% | 20 | 119.8 | 2 | 3.12% |
| `torch_sdpa_fp16` | **0.87** | 1.7% | 18 | 48.3 | 2 | 1.30% |
| `flash_attn_i4values_small_qout` | **0.70** | 1.4% | 12 | 58.0 | 2 | 0.30% |

- **`flash_attn_int4_vt`** — int4 flash attention, V pre-transposed. W4A4's counterpart to flash_attn_int8_vt and, at ~42%, the largest single item in its suite.
- **`flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24`** — int4 V values through the int8 MMA path, hd=24 exact specialization, int8-code output -- W4A4's twin of the int8 hd24 kernel.
- **`flash_attn_int4_vt_static`** — int4 flash with frozen Q/K scales.
- **`flash_attn_int4_vt_static_qout`** — int4 flash, frozen scales, int8-code output.
- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.
- **`flash_attn_i4values_small_qout`** — int4 values, small-T variant, int8-code output.

<details><summary>signatures ≥ 4% of the suite (4 of 13)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.55 | 10 | 1855.2 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int4_vt` |
| 15.38 | 10 | 1538.3 | `[128,1024,8,32] x [128,8,1024,32]` | `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24` |
| 8.22 | 5 | 1645.0 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int4_vt_static` |
| 2.23 | 10 | 223.4 | `[128,8,256,32] x [128,8,256,32]` | `flash_attn_int4_vt` |

</details>

### W4A4 MoDiff — 49.78 ms/sample total  (REPORT.md: 49.78 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `flash_attn_int4_vt` | **21.06** | 42.3% | 30 | 701.9 | 3 | 3.33% |
| `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24` | **15.38** | 30.9% | 10 | 1538.0 | 1 | 0.77% |
| `flash_attn_int4_vt_static` | **9.52** | 19.1% | 15 | 634.8 | 3 | 0.91% |
| `flash_attn_int4_vt_static_qout` | **2.26** | 4.5% | 20 | 113.1 | 2 | 2.69% |
| `torch_sdpa_fp16` | **0.86** | 1.7% | 18 | 48.0 | 2 | 0.97% |
| `flash_attn_i4values_small_qout` | **0.70** | 1.4% | 12 | 58.0 | 2 | 11.48% |

- **`flash_attn_int4_vt`** — int4 flash attention, V pre-transposed. W4A4's counterpart to flash_attn_int8_vt and, at ~42%, the largest single item in its suite.
- **`flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24`** — int4 V values through the int8 MMA path, hd=24 exact specialization, int8-code output -- W4A4's twin of the int8 hd24 kernel.
- **`flash_attn_int4_vt_static`** — int4 flash with frozen Q/K scales.
- **`flash_attn_int4_vt_static_qout`** — int4 flash, frozen scales, int8-code output.
- **`torch_sdpa_fp16`** — UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is what the flash kernels exist to avoid.
- **`flash_attn_i4values_small_qout`** — int4 values, small-T variant, int8-code output.

<details><summary>signatures ≥ 4% of the suite (4 of 13)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.46 | 10 | 1845.5 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int4_vt` |
| 15.38 | 10 | 1538.0 | `[128,1024,8,32] x [128,8,1024,32]` | `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24` |
| 8.27 | 5 | 1653.7 | `[128,8,1024,32] x [128,8,1024,32]` | `flash_attn_int4_vt_static` |
| 2.21 | 10 | 220.9 | `[128,8,256,32] x [128,8,256,32]` | `flash_attn_int4_vt` |

</details>

## conv

### fp16 — 265.72 ms/sample total  (REPORT.md: 265.72 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `torch_conv2d_fp16` | **265.72** | 100.0% | 445 | 597.1 | 33 | 2.84% |

- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).

<details><summary>signatures ≥ 4% of the suite (8 of 33)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 37.42 | 10 | 3742.4 | `[128,384,32,32] x [384,384,3,3]` | `torch_conv2d_fp16` |
| 35.87 | 35 | 1025.0 | `[128,192,32,32] x [192,192,3,3]` | `torch_conv2d_fp16` |
| 34.68 | 40 | 866.9 | `[128,384,16,16] x [384,384,3,3]` | `torch_conv2d_fp16` |
| 18.77 | 10 | 1876.9 | `[128,768,16,16] x [384,768,3,3]` | `torch_conv2d_fp16` |
| 17.84 | 10 | 1784.2 | `[128,384,32,32] x [192,384,3,3]` | `torch_conv2d_fp16` |
| 16.25 | 5 | 3250.4 | `[128,576,32,32] x [192,576,3,3]` | `torch_conv2d_fp16` |
| 11.77 | 40 | 294.3 | `[128,768,4,4] x [768,768,3,3]` | `torch_conv2d_fp16` |
| 11.40 | 45 | 253.4 | `[128,384,8,8] x [384,384,3,3]` | `torch_conv2d_fp16` |

</details>

### W8A8 PTQ — 149.09 ms/sample total  (REPORT.md: 149.09 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `conv2d_int8_evt_bias_residual_fp16` | **126.13** | 84.6% | 350 | 360.4 | 29 | 3.33% |
| `torch_conv2d_fp16` | **22.97** | 15.4% | 95 | 241.7 | 13 | 1.24% |

- **`conv2d_int8_evt_bias_residual_fp16`** — D1 fusion: out = acc*alpha*weight_scale[k] + bias[k] + residual[elem] -> fp16, in the conv's own store. This is the PTQ arm's whole conv datapath.
- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).

<details><summary>signatures ≥ 4% of the suite (8 of 42)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 19.12 | 25 | 765.0 | `[128,192,32,32] x [192,3,3,192]` | `conv2d_int8_evt_bias_residual_fp16` |
| 13.50 | 30 | 450.0 | `[128,384,16,16] x [384,3,3,384]` | `conv2d_int8_evt_bias_residual_fp16` |
| 10.55 | 10 | 1055.2 | `[128,384,32,32] x [192,3,3,384]` | `conv2d_int8_evt_bias_residual_fp16` |
| 8.54 | 5 | 1708.5 | `[128,384,32,32] x [384,3,3,384]` | `conv2d_int8_evt_bias_residual_fp16` |
| 8.32 | 5 | 1663.6 | `[128,576,32,32] x [192,3,3,576]` | `conv2d_int8_evt_bias_residual_fp16` |
| 8.14 | 5 | 1628.7 | `[128,384,32,32] x [384,3,3,384]` | `conv2d_int8_evt_bias_residual_fp16` |
| 8.04 | 10 | 804.0 | `[128,768,16,16] x [384,3,3,768]` | `conv2d_int8_evt_bias_residual_fp16` |
| 7.33 | 10 | 733.2 | `[128,192,32,32] x [192,3,3,192]` | `conv2d_int8_evt_bias_residual_fp16` |

</details>

### W8A8 MoDiff — 266.65 ms/sample total  (REPORT.md: 266.65 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `conv2d_int8_fprop` | **129.79** | 48.7% | 350 | 370.8 | 20 | 2.82% |
| `conv2d_int8_evt_o_hat` | **58.05** | 21.8% | 140 | 414.6 | 20 | 3.02% |
| `conv2d_int8_evt_o_hat_residual` | **48.36** | 18.1% | 140 | 345.4 | 9 | 2.59% |
| `torch_conv2d_fp16` | **30.45** | 11.4% | 95 | 320.5 | 13 | 0.78% |

- **`conv2d_int8_fprop`** — int8 x int8 conv, plain output. On the MoDiff arm this is the t=T conv and the delta-step conv whose accumulate is done by a separate epilogue.
- **`conv2d_int8_evt_o_hat`** — D2 fusion without a skip: o_hat[elem] += acc*alpha*weight_scale[k], in place in fp16. MoDiff's temporal state advance (paper Eq 9).
- **`conv2d_int8_evt_o_hat_residual`** — D2 DUAL STORE: advances o_hat in place AND writes out = o_hat_new + residual[elem] -> fp16, one pass, two stores. Replaces an fp32 conv_out round-trip.
- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).

<details><summary>signatures ≥ 4% of the suite (6 of 62)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 27.61 | 35 | 788.8 | `[128,192,32,32] x [192,3,3,192]` | `conv2d_int8_fprop` |
| 18.22 | 40 | 455.6 | `[128,384,16,16] x [384,3,3,384]` | `conv2d_int8_fprop` |
| 17.10 | 10 | 1709.7 | `[128,384,32,32] x [384,3,3,384]` | `conv2d_int8_fprop` |
| 16.73 | 20 | 836.6 | `[128,192,32,32] x [192,3,3,192]` | `conv2d_int8_evt_o_hat_residual` |
| 11.67 | 24 | 486.2 | `[128,384,16,16] x [384,3,3,384]` | `conv2d_int8_evt_o_hat_residual` |
| 11.00 | 10 | 1099.7 | `[128,384,32,32] x [192,3,3,384]` | `conv2d_int8_fprop` |

</details>

### W4A4 PTQ — 85.59 ms/sample total  (REPORT.md: 85.59 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `conv2d_int4_evt_bias_residual_fp16` | **62.62** | 73.2% | 350 | 178.9 | 29 | 3.86% |
| `torch_conv2d_fp16` | **22.98** | 26.8% | 95 | 241.8 | 13 | 1.50% |

- **`conv2d_int4_evt_bias_residual_fp16`** — D1 fusion, int4. The PTQ arm's whole conv datapath.
- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).

<details><summary>signatures ≥ 4% of the suite (10 of 42)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.98 | 25 | 359.4 | `[128,32,32,96] x [192,3,3,96]` | `conv2d_int4_evt_bias_residual_fp16` |
| 6.87 | 30 | 229.1 | `[128,16,16,192] x [384,3,3,192]` | `conv2d_int4_evt_bias_residual_fp16` |
| 5.23 | 10 | 523.5 | `[128,32,32,192] x [192,3,3,192]` | `conv2d_int4_evt_bias_residual_fp16` |
| 4.94 | 10 | 493.6 | `[128,384,32,32] x [192,384,1,1]` | `torch_conv2d_fp16` |
| 4.23 | 5 | 845.3 | `[128,32,32,192] x [384,3,3,192]` | `conv2d_int4_evt_bias_residual_fp16` |
| 4.17 | 5 | 834.3 | `[128,32,32,288] x [192,3,3,288]` | `conv2d_int4_evt_bias_residual_fp16` |
| 4.14 | 5 | 827.2 | `[128,192,32,32] x [4,192,3,3]` | `torch_conv2d_fp16` |
| 4.03 | 5 | 806.5 | `[128,32,32,192] x [384,3,3,192]` | `conv2d_int4_evt_bias_residual_fp16` |
| 4.00 | 10 | 399.9 | `[128,16,16,384] x [384,3,3,384]` | `conv2d_int4_evt_bias_residual_fp16` |
| 3.46 | 10 | 345.7 | `[128,32,32,96] x [192,3,3,96]` | `conv2d_int4_evt_bias_residual_fp16` |

</details>

### W4A4 MoDiff — 155.30 ms/sample total  (REPORT.md: 155.30 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `conv2d_int4_fprop` | **67.91** | 43.7% | 350 | 194.0 | 20 | 2.87% |
| `torch_conv2d_fp16` | **30.45** | 19.6% | 95 | 320.5 | 13 | 1.05% |
| `conv2d_int4_evt_o_hat` | **28.73** | 18.5% | 140 | 205.2 | 20 | 13.17% |
| `conv2d_int4_evt_o_hat_residual` | **28.21** | 18.2% | 140 | 201.5 | 9 | 1.40% |

- **`conv2d_int4_fprop`** — int4 x int4 conv, plain output; same role as the int8 twin.
- **`torch_conv2d_fp16`** — UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline does not quantize (the stem/head convs and the 1x1 skips).
- **`conv2d_int4_evt_o_hat`** — D2 fusion without a skip, int4. MoDiff's temporal state advance.
- **`conv2d_int4_evt_o_hat_residual`** — D2 dual store, int4.

<details><summary>signatures ≥ 4% of the suite (6 of 62)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 13.49 | 35 | 385.6 | `[128,32,32,96] x [192,3,3,96]` | `conv2d_int4_fprop` |
| 10.13 | 40 | 253.4 | `[128,16,16,192] x [384,3,3,192]` | `conv2d_int4_fprop` |
| 9.44 | 20 | 472.2 | `[128,32,32,96] x [192,3,3,96]` | `conv2d_int4_evt_o_hat_residual` |
| 8.76 | 10 | 876.1 | `[128,32,32,192] x [384,3,3,192]` | `conv2d_int4_fprop` |
| 7.25 | 24 | 302.2 | `[128,16,16,192] x [384,3,3,192]` | `conv2d_int4_evt_o_hat_residual` |
| 6.90 | 5 | 1380.5 | `[128,576,32,32] x [192,576,1,1]` | `torch_conv2d_fp16` |

</details>

## linear

### fp16 — 60.92 ms/sample total  (REPORT.md: 60.92 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `fused_gn_qkv` | **31.96** | 52.5% | 50 | 639.2 | 2 | 1.05% |
| `torch_linear_fp16` | **28.96** | 47.5% | 345 | 83.9 | 12 | 1.91% |

- **`fused_gn_qkv`** — UNQUANTIZED, fp16 ONLY -- the qkv projection with the GroupNorm folded into its mainloop as a per-sample scale/bias, so the normalized activation is never written to HBM. Taken by the fp16 arm only, and only where T % 128 == 0 and c % 8 == 0, which is exactly the T=1024 and T=256 blocks; the smaller ones fall to plain GroupNorm + `torch_linear_fp16`. 52% of fp16's linear suite. It has NO counterpart in the quantized arms, which split the same work into a `norm_quantize` group_norm_silu_quantize_nhwc plus an AWQ GEMM here -- so the linear and norm_quantize suite totals do not compare across arms. docs/OPEN_ITEMS.md A1.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.

<details><summary>signatures ≥ 4% of the suite (7 of 14)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 18.71 | 25 | 748.2 | `[128,192,32,32] x [576,1,1,192]` | `fused_gn_qkv` |
| 13.26 | 25 | 530.2 | `[128,384,16,16] x [1152,1,1,384]` | `fused_gn_qkv` |
| 5.02 | 25 | 200.9 | `[128,1024,192] x [192,192]` | `torch_linear_fp16` |
| 4.65 | 75 | 62.0 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 4.37 | 75 | 58.3 | `[128,768] x [1536,768]` | `torch_linear_fp16` |
| 3.39 | 25 | 135.5 | `[128,256,384] x [384,384]` | `torch_linear_fp16` |
| 2.68 | 25 | 107.1 | `[128,16,768] x [2304,768]` | `torch_linear_fp16` |

</details>

### W8A8 PTQ — 47.15 ms/sample total  (REPORT.md: 47.15 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `gemm_w8a8_awq_bias_res` | **29.19** | 61.9% | 168 | 173.8 | 10 | 1.61% |
| `torch_linear_fp16` | **7.62** | 16.2% | 185 | 41.2 | 4 | 0.64% |
| `gemm_w8a8_awq_qkv_i8_layouts` | **5.65** | 12.0% | 10 | 565.2 | 1 | 0.87% |
| `gemm_w8a8_awq_qkv_i8_layouts_compact` | **4.09** | 8.7% | 20 | 204.3 | 2 | 5.71% |
| `gemm_w8a8_awq_out_i8_bias_nout` | **0.59** | 1.3% | 12 | 49.3 | 2 | 0.62% |

- **`gemm_w8a8_awq_bias_res`** — W8A8 AWQ-layout GEMM with the bias+residual epilogue. `a_scale` is a 1-ELEMENT DEVICE TENSOR, not a double, because MoDiff's delta scale is produced on device each call and taking it by value would force a host sync per linear per step.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.
- **`gemm_w8a8_awq_qkv_i8_layouts`** — fused qkv projection: one GEMM writing Q, K and V already in the attention kernel's per-head padded layouts as int8.
- **`gemm_w8a8_awq_qkv_i8_layouts_compact`** — the compact-staging variant of the above.
- **`gemm_w8a8_awq_out_i8_bias_nout`** — W8A8 GEMM emitting int8 codes of (out + bias) at a per-column scale, so a projection can feed flash attention without a separate quantize.

<details><summary>signatures ≥ 4% of the suite (8 of 19)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.59 | 25 | 343.6 | `[131072,192] x [256,192]` | `gemm_w8a8_awq_bias_res` |
| 6.59 | 15 | 439.2 | `[131072,192] x [640,192]` | `gemm_w8a8_awq_bias_res` |
| 5.65 | 10 | 565.2 | `[131072,192] x [768,192]` | `gemm_w8a8_awq_qkv_i8_layouts` |
| 4.74 | 25 | 189.7 | `[32768,384] x [384,384]` | `gemm_w8a8_awq_bias_res` |
| 4.09 | 15 | 272.6 | `[32768,384] x [1152,384]` | `gemm_w8a8_awq_bias_res` |
| 3.58 | 75 | 47.7 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 3.17 | 10 | 317.2 | `[32768,384] x [1152,384]` | `gemm_w8a8_awq_qkv_i8_layouts_compact` |
| 2.64 | 75 | 35.2 | `[128,768] x [1536,768]` | `torch_linear_fp16` |

</details>

### W8A8 MoDiff — 46.46 ms/sample total  (REPORT.md: 46.46 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `gemm_w8a8_awq_bias_res` | **29.13** | 62.7% | 168 | 173.4 | 10 | 1.24% |
| `torch_linear_fp16` | **6.98** | 15.0% | 185 | 37.7 | 4 | 32.29% |
| `gemm_w8a8_awq_qkv_i8_layouts` | **5.65** | 12.2% | 10 | 565.3 | 1 | 0.73% |
| `gemm_w8a8_awq_qkv_i8_layouts_compact` | **4.10** | 8.8% | 20 | 205.0 | 2 | 2.60% |
| `gemm_w8a8_awq_out_i8_bias_nout` | **0.60** | 1.3% | 12 | 49.6 | 2 | 1.13% |

- **`gemm_w8a8_awq_bias_res`** — W8A8 AWQ-layout GEMM with the bias+residual epilogue. `a_scale` is a 1-ELEMENT DEVICE TENSOR, not a double, because MoDiff's delta scale is produced on device each call and taking it by value would force a host sync per linear per step.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.
- **`gemm_w8a8_awq_qkv_i8_layouts`** — fused qkv projection: one GEMM writing Q, K and V already in the attention kernel's per-head padded layouts as int8.
- **`gemm_w8a8_awq_qkv_i8_layouts_compact`** — the compact-staging variant of the above.
- **`gemm_w8a8_awq_out_i8_bias_nout`** — W8A8 GEMM emitting int8 codes of (out + bias) at a per-column scale, so a projection can feed flash attention without a separate quantize.

<details><summary>signatures ≥ 4% of the suite (8 of 19)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.62 | 25 | 344.7 | `[131072,192] x [256,192]` | `gemm_w8a8_awq_bias_res` |
| 6.52 | 15 | 434.6 | `[131072,192] x [640,192]` | `gemm_w8a8_awq_bias_res` |
| 5.65 | 10 | 565.3 | `[131072,192] x [768,192]` | `gemm_w8a8_awq_qkv_i8_layouts` |
| 4.70 | 25 | 187.9 | `[32768,384] x [384,384]` | `gemm_w8a8_awq_bias_res` |
| 4.10 | 15 | 273.4 | `[32768,384] x [1152,384]` | `gemm_w8a8_awq_bias_res` |
| 3.19 | 10 | 318.6 | `[32768,384] x [1152,384]` | `gemm_w8a8_awq_qkv_i8_layouts_compact` |
| 2.98 | 75 | 39.7 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 2.65 | 75 | 35.3 | `[128,768] x [1536,768]` | `torch_linear_fp16` |

</details>

### W4A4 PTQ — 43.45 ms/sample total  (REPORT.md: 43.45 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `gemm_w4a4_awq_bias_res` | **25.36** | 58.4% | 168 | 151.0 | 10 | 1.02% |
| `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` | **9.78** | 22.5% | 30 | 326.1 | 3 | 5.75% |
| `torch_linear_fp16` | **7.90** | 18.2% | 185 | 42.7 | 4 | 1.33% |
| `gemm_w4a4_awq_qkv_codes` | **0.41** | 0.9% | 12 | 34.0 | 2 | 0.29% |

- **`gemm_w4a4_awq_bias_res`** — W4A4 AWQ-layout GEMM, bias+residual epilogue. The linear suite's largest item on both W4A4 arms.
- **`gemm_w4a4_awq_qkv_i4qk_i8v_layouts`** — fused qkv projection emitting int4 Q/K and int8 V in the attention layouts -- the asymmetry is deliberate: V's dot product accumulates in fp32, so it keeps 8 bits while Q/K drop to 4.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.
- **`gemm_w4a4_awq_qkv_codes`** — emits the qkv int4 codes plus their clamp limits rather than dequantized values.

<details><summary>signatures ≥ 4% of the suite (8 of 19)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.16 | 25 | 326.2 | `[131072,128] x [256,128]` | `gemm_w4a4_awq_bias_res` |
| 5.73 | 15 | 382.0 | `[131072,128] x [640,128]` | `gemm_w4a4_awq_bias_res` |
| 5.58 | 10 | 558.0 | `[131072,128] x [768,128]` | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` |
| 4.25 | 25 | 169.9 | `[32768,192] x [384,192]` | `gemm_w4a4_awq_bias_res` |
| 3.59 | 75 | 47.8 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 3.26 | 10 | 326.5 | `[32768,192] x [1536,192]` | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` |
| 2.99 | 15 | 199.2 | `[32768,192] x [1152,192]` | `gemm_w4a4_awq_bias_res` |
| 2.65 | 75 | 35.3 | `[128,768] x [1536,768]` | `torch_linear_fp16` |

</details>

### W4A4 MoDiff — 43.72 ms/sample total  (REPORT.md: 43.72 ✓)

| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |
|---|--:|--:|--:|--:|--:|--:|
| `gemm_w4a4_awq_bias_res` | **25.36** | 58.0% | 168 | 151.0 | 10 | 1.45% |
| `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` | **9.77** | 22.4% | 30 | 325.7 | 3 | 4.77% |
| `torch_linear_fp16` | **8.18** | 18.7% | 185 | 44.2 | 4 | 0.81% |
| `gemm_w4a4_awq_qkv_codes` | **0.41** | 0.9% | 12 | 34.0 | 2 | 0.28% |

- **`gemm_w4a4_awq_bias_res`** — W4A4 AWQ-layout GEMM, bias+residual epilogue. The linear suite's largest item on both W4A4 arms.
- **`gemm_w4a4_awq_qkv_i4qk_i8v_layouts`** — fused qkv projection emitting int4 Q/K and int8 V in the attention layouts -- the asymmetry is deliberate: V's dot product accumulates in fp32, so it keeps 8 bits while Q/K drop to 4.
- **`torch_linear_fp16`** — UNQUANTIZED fallback -- PyTorch fp16 linear.
- **`gemm_w4a4_awq_qkv_codes`** — emits the qkv int4 codes plus their clamp limits rather than dequantized values.

<details><summary>signatures ≥ 4% of the suite (8 of 19)</summary>

| ms/sample | calls | µs/call | shapes | kernel |
|--:|--:|--:|---|---|
| 8.14 | 25 | 325.7 | `[131072,128] x [256,128]` | `gemm_w4a4_awq_bias_res` |
| 5.70 | 15 | 380.3 | `[131072,128] x [640,128]` | `gemm_w4a4_awq_bias_res` |
| 5.58 | 10 | 558.4 | `[131072,128] x [768,128]` | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` |
| 4.25 | 25 | 170.1 | `[32768,192] x [384,192]` | `gemm_w4a4_awq_bias_res` |
| 3.78 | 75 | 50.4 | `[128,768] x [768,768]` | `torch_linear_fp16` |
| 3.25 | 10 | 324.9 | `[32768,192] x [1536,192]` | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` |
| 3.01 | 15 | 200.9 | `[32768,192] x [1152,192]` | `gemm_w4a4_awq_bias_res` |
| 2.66 | 75 | 35.4 | `[128,768] x [1536,768]` | `torch_linear_fp16` |

</details>

