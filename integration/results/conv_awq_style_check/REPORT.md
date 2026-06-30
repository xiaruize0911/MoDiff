# Conv2d AWQ-Style Optimization Check

Date: 2026-06-30

## Answer

We cannot apply the exact AWQ linear optimization to Conv2d as a drop-in.

AWQ in `/workspace/llm-awq` provides W8A8 GEMM kernels:

- `w8a8_gemm_forward_cuda`
- `w8a8_gemm_fuse_bias_forward_cuda`
- activation quantization helpers

It does not provide a W8A8 Conv2d kernel. Using AWQ GEMM for general Conv2d would require explicit im2col/unfold, which would create a large activation matrix and add much more IO than our current implicit-GEMM CUTLASS Conv2d path.

## Current Conv2d Path

The active LDM Conv2d baseline is:

- `integration/kernels/int8_optimized.py::OptimizedInt8Conv2d`
- CUTLASS implicit-GEMM Conv2d in `csrc/cuda_kernels.cu`

The standard calibrated path is:

1. `scale_quantize_int8_kernel`: activation -> INT8 activation tensor
2. CUTLASS INT8 implicit-GEMM Conv2d
3. one of:
   - `conv2d_int8_fprop_dequant_fp16_prealloc`: CUTLASS epilogue dequant to FP16
   - `conv2d_int8_fprop_no_ohat_prealloc_bias`: Conv2d to FP32 temp, then fused scale+bias+store
   - fallback `scale_store_half_kernel`

So the Conv2d path has already received the closest equivalent of the AWQ improvement: move dequant/scale/store into or next to the math kernel, and remove separate PyTorch postprocess kernels where possible.

## Why Not AWQ GEMM for Conv2d

For a 3x3 Conv2d, GEMM lowering expands input from:

`[B, C, H, W]`

to roughly:

`[B * Hout * Wout, C * 3 * 3]`

That is about 9x larger in the K dimension before GEMM, plus layout/materialization overhead. AWQ GEMM is fast, but explicit im2col would likely dominate runtime and memory bandwidth. Our current CUTLASS implicit-GEMM Conv2d avoids materializing that im2col matrix.

For 1x1 Conv2d only, AWQ GEMM could be feasible because there is no im2col expansion. It would be equivalent to flattening NHWC activations to `[B*H*W, C]` and running W8A8 GEMM. That is worth testing separately if many hot layers are 1x1, but the LDM ResBlock conv benchmark shapes here are 3x3.

## Existing Conv2d Benchmark Evidence

From `integration/results/ldm_int8_kernel_compare/ldm_int8_kernel_compare.json`:

| Shape | Median ms | TOPS |
|---|---:|---:|
| `res_128_32` | 0.2570 | 37.60 |
| `res_256_16` | 0.1540 | 62.74 |
| `mid_512_8` | 0.0990 | 97.61 |
| `up_128_64` | 0.8957 | 43.15 |

From the previous Conv2d fuse report:

| Optimization | Example speedup |
|---|---:|
| fused scale+bias store | 1.04x-1.13x |
| CUTLASS FP16 dequant epilogue, no-bias | 1.05x-1.34x |
| heuristic biased deep fuse | up to 1.14x |

## What Is Still Poor

The remaining Conv2d overhead is not AWQ-vs-ours GEMM anymore. It is mostly:

1. Activation INT8 materialization:
   - `scale_quantize_int8_kernel` writes a full INT8 activation tensor before Conv2d.

2. Remaining FP32 temporary paths:
   - Some biased/smaller outputs still use CUTLASS Conv2d -> FP32 temp -> fused scale/bias/store.

3. Boundary/layout traffic:
   - channels-last normalization can cause D2D copies if upstream tensors are not already channels-last.
   - quantized Conv islands still hand FP16/FP32 tensors back to surrounding GroupNorm/activation/residual code.

## Best Next Conv Optimization

The best Conv2d analog to the AWQ linear optimization is not AWQ GEMM. It is:

1. Extend the CUTLASS Conv2d epilogue to include both per-channel weight scale and bias for all biased layers.
2. Prefer direct FP16 output in the epilogue wherever numerically acceptable.
3. Reuse/preallocate INT8 activation buffers so `scale_quantize_int8_kernel` stops allocating/churning tensors.
4. For 1x1 Conv2d only, add an optional AWQ-GEMM path and benchmark it against CUTLASS implicit Conv2d.

## Bottom Line

Same idea: yes.

Same AWQ kernel: no, except maybe 1x1 Conv2d.

For the current 3x3 Conv2d-heavy path, the right direction is deeper CUTLASS Conv2d epilogue fusion and activation-buffer reuse, not lowering Conv2d into AWQ GEMM.

