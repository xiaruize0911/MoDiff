# Separated Static INT8/AWQ Kernel Analysis

Experiment root: `integration/results/current_bs42_n168_s200_static`

This report excludes full-pipeline timing. It compares separated kernels using `integration/benchmarks/benchmark_ldm_int8_kernels.py` with batch size 42, static quantization, 30 warmup iterations, 200 timed CUDA-event iterations, and 5 full measurement rounds. Reported latency is the median of the per-round medians.

## Implementation Details Before Results

### Conv2d Path

- FP16 baseline: PyTorch `nn.Conv2d(...).half()` in channels-last layout, dispatched through cuDNN/CUTLASS FP16 tensor-core convolution kernels.
- INT8 module: `integration/kernels/int8_optimized.py::OptimizedInt8Conv2d`.
- Weight format: per-output-channel symmetric INT8 weights, stored as CUTLASS NHWC filter layout.
- Activation quantization: `set_static_scale(...)` marks the module calibrated and caches the static activation scale/inverse scale.
- Benchmark mode: `enable_modiff(False)` and `set_standard_output_fp16(True)`.
- 3x3 INT8 kernel sequence: `modiff_cutlass.scale_quantize_int8` followed by `modiff_cutlass.conv2d_int8_fprop_dequant_fp16_prealloc` or the bias/prealloc variant. These are CUTLASS implicit-GEMM convolution paths.
- 1x1 special case: `OptimizedInt8Conv2d._forward_awq_1x1` flattens NHWC activations to GEMM shape `M=BHW`, then calls `awq_fused_quant_gemm_w8a8_prealloc`.

### Linear / GEMM Path

- FP16 baseline: `F.linear(x.half(), weight_fp16, bias)`.
- Current linear: `integration/kernels/int8_linear.py::OptimizedInt8Linear(backend="int_gemm")`.
- Current GEMM kernel: activation quantization plus `modiff_triton.kernels.gemm_w8a8.gemm_w8a8`; selected square projection shapes may route to the AWQ fallback path.
- AWQ linear: `OptimizedInt8Linear(backend="awq")`.
- AWQ GEMM kernels: `awq_fused_quant_gemm_w8a8` when quantization is fused, or `awq_gemm_w8a8` when a static input scale is supplied.
- Benchmark mode: `enable_modiff(False)` and `set_standard_output_fp16(True)`.

## Benchmark Shapes

Conv shapes use `B=42`:

| Index | Shape | Cin | HxW | Cout | K |
|---:|---|---:|---:|---:|---:|
| 1 | `res_128_64` | 128 | 64x64 | 128 | 3 |
| 2 | `res_128_32` | 128 | 32x32 | 128 | 3 |
| 3 | `down_128_256_32` | 128 | 32x32 | 256 | 3 |
| 4 | `res_256_32` | 256 | 32x32 | 256 | 3 |
| 5 | `res_256_16` | 256 | 16x16 | 256 | 3 |
| 6 | `down_256_512_16` | 256 | 16x16 | 512 | 3 |
| 7 | `mid_512_8` | 512 | 8x8 | 512 | 3 |
| 8 | `up_512_256_16` | 512 | 16x16 | 256 | 3 |
| 9 | `up_256_128_32` | 256 | 32x32 | 128 | 3 |
| 10 | `up_128_64` | 128 | 64x64 | 128 | 3 |
| 11 | `pointwise_320_16` | 320 | 16x16 | 320 | 1 |

Linear shapes:

| Index | Shape | M | K | N |
|---:|---|---:|---:|---:|
| 1 | `attn_proj_320_m2688` | 2688 | 320 | 320 |
| 2 | `attn_proj_320_m5376` | 5376 | 320 | 320 |
| 3 | `attn_proj_512` | 5376 | 512 | 512 |
| 4 | `attn_proj_640_m2688` | 2688 | 640 | 640 |
| 5 | `ffn_512_2048` | 5376 | 512 | 2048 |
| 6 | `ffn_640_2560` | 2688 | 640 | 2560 |
| 7 | `large_2048` | 2048 | 2048 | 2048 |
| 8 | `large_4096` | 1024 | 4096 | 4096 |
| 9 | `small_m_4096` | 128 | 4096 | 4096 |

## Conv Results

| Shape | FP16 ms | INT8 ms | FP16/INT8 | INT8 TOPS | Round stdev ms |
|---|---:|---:|---:|---:|---:|
| `res_128_64` | 0.5946 | 0.8555 | 0.70x | 59.31 | 0.0013 |
| `res_128_32` | 0.1708 | 0.2330 | 0.73x | 54.44 | 0.0003 |
| `down_128_256_32` | 0.3192 | 0.3423 | 0.93x | 74.11 | 0.0013 |
| `res_256_32` | 0.5136 | 0.5349 | 0.96x | 94.84 | 0.0018 |
| `res_256_16` | 0.1354 | 0.1446 | 0.94x | 87.71 | 0.0020 |
| `down_256_512_16` | 0.2514 | 0.2245 | 1.12x | 112.99 | 0.0028 |
| `mid_512_8` | 0.1340 | 0.0988 | 1.36x | 128.44 | 0.0027 |
| `up_512_256_16` | 0.2330 | 0.2424 | 0.96x | 104.64 | 0.0037 |
| `up_256_128_32` | 0.2684 | 0.3812 | 0.70x | 66.55 | 0.0010 |
| `up_128_64` | 0.6022 | 0.8576 | 0.70x | 59.16 | 0.0017 |
| `pointwise_320_16` | 0.0747 | 0.1326 | 0.56x | 16.60 | 0.0179 |

INT8 conv wins only on the high-channel smaller-spatial cases `down_256_512_16` and `mid_512_8`. It loses on large-spatial 128-channel shapes and on the tested 1x1 pointwise shape.

## Linear Results

| Shape | FP16 ms | current `int_gemm` ms | AWQ ms | `int_gemm`/AWQ | FP16/AWQ |
|---|---:|---:|---:|---:|---:|
| `attn_proj_320_m2688` | 0.0403 | 0.2639 | 0.0989 | 2.67x | 0.41x |
| `attn_proj_320_m5376` | 0.0474 | 0.2552 | 0.1067 | 2.39x | 0.44x |
| `attn_proj_512` | 0.0525 | 0.1176 | 0.1142 | 1.03x | 0.46x |
| `attn_proj_640_m2688` | 0.0519 | 0.2550 | 0.1047 | 2.44x | 0.50x |
| `ffn_512_2048` | 0.1621 | 1.0941 | 0.1706 | 6.41x | 0.95x |
| `ffn_640_2560` | 0.1074 | 0.7151 | 0.1252 | 5.71x | 0.86x |
| `large_2048` | 0.1771 | 0.1782 | 0.1784 | 1.00x | 0.99x |
| `large_4096` | 0.3514 | 0.2678 | 0.2667 | 1.00x | 1.32x |
| `small_m_4096` | 0.0804 | 0.2287 | 0.4344 | 0.53x | 0.19x |

AWQ is much faster than the current `int_gemm` backend on the attention and FFN projection shapes, especially the FFN expansion shapes. FP16 remains best for small or moderate shapes where quantization overhead dominates. The large 4096 square shape is the case where INT8 GEMM beats FP16.

## Artifacts

- Expanded benchmark CSV: `integration/results/current_bs42_n168_s200_static/separated_kernel_benchmark_r5_i200/ldm_int8_kernel_compare.csv`
- Expanded benchmark JSON: `integration/results/current_bs42_n168_s200_static/separated_kernel_benchmark_r5_i200/ldm_int8_kernel_compare.json`
- Updated slides: `integration/results/current_bs42_n168_s200_static/current_static_int8_awq_slides.tex`
- Compiled deck: `integration/results/current_bs42_n168_s200_static/current_static_int8_awq_slides.pdf`

