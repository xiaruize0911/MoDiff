# Current Static INT8/AWQ Separated-Kernel Report

Experiment root: `integration/results/current_bs42_n168_s200_static`

This report intentionally does not present full-pipeline benchmark timing. The focus is separated kernel behavior for the current static INT8 conv path and the current linear backends, measured on an NVIDIA A40 with batch size 42. The expanded rerun used 30 warmup iterations, 200 timed iterations, and 5 full rounds per backend/shape.

## Implementation First

### Conv2d

`OptimizedInt8Conv2d` lives in `integration/kernels/int8_optimized.py`.

- FP16 comparison path: PyTorch `nn.Conv2d(...).half()` in channels-last format, dispatched by cuDNN/CUTLASS.
- INT8 weight path: per-output-channel symmetric INT8 weights, stored as NHWC filters for CUTLASS.
- Static activation path: `set_static_scale(...)` marks the layer calibrated and caches scale tensors.
- Benchmark path: `enable_modiff(False)` and `set_standard_output_fp16(True)`.
- Kernel sequence for 3x3 conv: `modiff_cutlass.scale_quantize_int8` then a CUTLASS implicit-GEMM conv wrapper such as `conv2d_int8_fprop_dequant_fp16_prealloc`.
- Kernel sequence for 1x1 conv: flatten NHWC activations to GEMM and call `awq_fused_quant_gemm_w8a8_prealloc`.

### Linear / GEMM

`OptimizedInt8Linear` lives in `integration/kernels/int8_linear.py`.

- FP16 backend: `F.linear` with FP16 activation/weight/bias.
- Current `int_gemm` backend: activation quantization plus `modiff_triton.kernels.gemm_w8a8.gemm_w8a8`; selected square projection shapes can route to AWQ fallback.
- AWQ backend: `awq_fused_quant_gemm_w8a8` or `awq_gemm_w8a8` from `modiff_triton/kernels/awq_w8a8.py`.
- Benchmark path: `enable_modiff(False)` and `set_standard_output_fp16(True)`.

## Conv Results

| Shape | FP16 ms | INT8 ms | FP16/INT8 |
|---|---:|---:|---:|
| `res_128_64` | 0.5946 | 0.8555 | 0.70x |
| `res_128_32` | 0.1708 | 0.2330 | 0.73x |
| `down_128_256_32` | 0.3192 | 0.3423 | 0.93x |
| `res_256_32` | 0.5136 | 0.5349 | 0.96x |
| `res_256_16` | 0.1354 | 0.1446 | 0.94x |
| `down_256_512_16` | 0.2514 | 0.2245 | 1.12x |
| `mid_512_8` | 0.1340 | 0.0988 | 1.36x |
| `up_512_256_16` | 0.2330 | 0.2424 | 0.96x |
| `up_256_128_32` | 0.2684 | 0.3812 | 0.70x |
| `up_128_64` | 0.6022 | 0.8576 | 0.70x |
| `pointwise_320_16` | 0.0747 | 0.1326 | 0.56x |

The current INT8 conv kernel is shape-dependent. It is faster on high-channel, small-spatial shapes and slower on large-spatial 128-channel shapes.

## Linear Results

| Shape | FP16 ms | int_gemm ms | AWQ ms | int_gemm/AWQ |
|---|---:|---:|---:|---:|
| `attn_proj_320_m2688` | 0.0403 | 0.2639 | 0.0989 | 2.67x |
| `attn_proj_320_m5376` | 0.0474 | 0.2552 | 0.1067 | 2.39x |
| `attn_proj_512` | 0.0525 | 0.1176 | 0.1142 | 1.03x |
| `attn_proj_640_m2688` | 0.0519 | 0.2550 | 0.1047 | 2.44x |
| `ffn_512_2048` | 0.1621 | 1.0941 | 0.1706 | 6.41x |
| `ffn_640_2560` | 0.1074 | 0.7151 | 0.1252 | 5.71x |
| `large_2048` | 0.1771 | 0.1782 | 0.1784 | 1.00x |
| `large_4096` | 0.3514 | 0.2678 | 0.2667 | 1.00x |
| `small_m_4096` | 0.0804 | 0.2287 | 0.4344 | 0.53x |

AWQ is the better INT8 linear backend for the attention/FFN projection shapes. FP16 is still fastest for smaller shapes where quantization and launch overhead dominate.

## Artifacts

- Expanded benchmark: `separated_kernel_benchmark_r5_i200/`
- Detailed separated-kernel report: `single_kernel_analysis/SINGLE_KERNEL_LINEAR_CONV_REPORT.md`
- Updated slides: `current_static_int8_awq_slides.tex`
- Compiled PDF: `current_static_int8_awq_slides.pdf`

