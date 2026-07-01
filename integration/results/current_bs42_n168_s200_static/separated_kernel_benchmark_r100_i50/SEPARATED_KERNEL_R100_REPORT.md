# Separated Kernel Benchmark Report, 100 Rounds

## Measurement Facts

| Item | Value |
| --- | --- |
| GPU | NVIDIA A40 |
| Benchmark script | `integration/benchmarks/benchmark_ldm_int8_kernels.py` |
| Result directory | `integration/results/current_bs42_n168_s200_static/separated_kernel_benchmark_r100_i50` |
| Batch size for conv shapes | 42 |
| Warmup before each measured round | 50 launches |
| Timed iterations per round | 50 launches |
| Measurement rounds | 100 |
| Reported latency | median of 100 round medians |
| Timing API | CUDA events with `torch.cuda.synchronize()` |

## Conv2d Implementation Facts

| Component | Fact |
| --- | --- |
| FP16 conv path | `torch.nn.Conv2d(...).half()` in channels-last layout |
| INT8 conv module | `integration/kernels/int8_optimized.py::OptimizedInt8Conv2d` |
| INT8 weight format | per-output-channel symmetric INT8, stored as NHWC filter tensor |
| Static activation scale | `set_static_scale(32.0)` in this separated benchmark |
| Benchmark flags | `enable_modiff(False)`, `set_standard_output_fp16(True)` |
| 3x3 INT8 quant kernel | `modiff_cutlass.scale_quantize_int8` |
| 3x3 INT8 conv kernel | `modiff_cutlass.conv2d_int8_fprop_dequant_fp16_prealloc` or bias/prealloc variant |
| 1x1 INT8 path | `OptimizedInt8Conv2d._forward_awq_1x1` flattens NHWC to GEMM with M=BHW |
| 1x1 GEMM kernel | `awq_fused_quant_gemm_w8a8_prealloc` |

## Linear/GEMM Implementation Facts

| Backend | GEMM or kernel call |
| --- | --- |
| FP16 | `torch.nn.functional.linear(x.half(), weight_fp16, bias)` |
| Current `int_gemm` | `modiff_triton.kernels.gemm_w8a8.gemm_w8a8`; selected square shapes route through AWQ fallback in `OptimizedInt8Linear._int8_gemm_linear` |
| AWQ | `awq_fused_quant_gemm_w8a8` or `awq_gemm_w8a8` from `modiff_triton/kernels/awq_w8a8.py` |
| Benchmark flags | `enable_modiff(False)`, `set_standard_output_fp16(True)`, `int_gemm_min_m=1` |

## Conv Shapes

| Index | Shape | B | Cin | HxW | Cout | K | Stride | Pad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1] | `res_128_64` | 42 | 128 | 64x64 | 128 | 3 | 1 | 1 |
| [2] | `res_128_32` | 42 | 128 | 32x32 | 128 | 3 | 1 | 1 |
| [3] | `down_128_256_32` | 42 | 128 | 32x32 | 256 | 3 | 1 | 1 |
| [4] | `res_256_32` | 42 | 256 | 32x32 | 256 | 3 | 1 | 1 |
| [5] | `res_256_16` | 42 | 256 | 16x16 | 256 | 3 | 1 | 1 |
| [6] | `down_256_512_16` | 42 | 256 | 16x16 | 512 | 3 | 1 | 1 |
| [7] | `mid_512_8` | 42 | 512 | 8x8 | 512 | 3 | 1 | 1 |
| [8] | `up_512_256_16` | 42 | 512 | 16x16 | 256 | 3 | 1 | 1 |
| [9] | `up_256_128_32` | 42 | 256 | 32x32 | 128 | 3 | 1 | 1 |
| [10] | `up_128_64` | 42 | 128 | 64x64 | 128 | 3 | 1 | 1 |
| [11] | `pointwise_320_16` | 42 | 320 | 16x16 | 320 | 1 | 1 | 0 |

## Linear Shapes

| Index | Shape | M | K | N |
| --- | --- | --- | --- | --- |
| [1] | `attn_proj_320_m2688` | 2688 | 320 | 320 |
| [2] | `attn_proj_320_m5376` | 5376 | 320 | 320 |
| [3] | `attn_proj_512` | 5376 | 512 | 512 |
| [4] | `attn_proj_640_m2688` | 2688 | 640 | 640 |
| [5] | `ffn_512_2048` | 5376 | 512 | 2048 |
| [6] | `ffn_640_2560` | 2688 | 640 | 2560 |
| [7] | `large_2048` | 2048 | 2048 | 2048 |
| [8] | `large_4096` | 1024 | 4096 | 4096 |
| [9] | `small_m_4096` | 128 | 4096 | 4096 |

## Figures

| Figure | File |
|---|---|
| Conv latency | [conv_latency_ms.png](plots/conv_latency_ms.png) |
| Conv speed ratio FP16/INT8 | [conv_speed_ratio_fp16_over_int8.png](plots/conv_speed_ratio_fp16_over_int8.png) |
| Linear latency | [linear_latency_ms.png](plots/linear_latency_ms.png) |
| Linear speed ratio int_gemm/AWQ | [linear_speed_ratio_intgemm_over_awq.png](plots/linear_speed_ratio_intgemm_over_awq.png) |
| Linear speed ratio FP16/AWQ | [linear_speed_ratio_fp16_over_awq.png](plots/linear_speed_ratio_fp16_over_awq.png) |

## Conv Results

| Shape | FP16 ms | INT8 ms | FP16/INT8 | FP16 stdev | INT8 stdev | FP16 TOPS | INT8 TOPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1] `res_128_64` | 0.6001 | 0.8596 | 0.70x | 0.0103 | 0.0045 | 84.54 | 59.02 |
| [2] `res_128_32` | 0.1696 | 0.2316 | 0.73x | 0.0008 | 0.0015 | 74.81 | 54.75 |
| [3] `down_128_256_32` | 0.3307 | 0.3493 | 0.95x | 0.0008 | 0.0006 | 76.71 | 72.62 |
| [4] `res_256_32` | 0.4968 | 0.5364 | 0.93x | 0.0117 | 0.0005 | 102.12 | 94.58 |
| [5] `res_256_16` | 0.1334 | 0.1436 | 0.93x | 0.0004 | 0.0004 | 95.05 | 88.36 |
| [6] `down_256_512_16` | 0.2473 | 0.2243 | 1.10x | 0.0055 | 0.0009 | 102.57 | 113.12 |
| [7] `mid_512_8` | 0.1262 | 0.0972 | 1.30x | 0.0006 | 0.0007 | 100.47 | 130.47 |
| [8] `up_512_256_16` | 0.2210 | 0.2420 | 0.91x | 0.0024 | 0.0003 | 114.76 | 104.80 |
| [9] `up_256_128_32` | 0.2606 | 0.3810 | 0.68x | 0.0044 | 0.0003 | 97.34 | 66.58 |
| [10] `up_128_64` | 0.5785 | 0.8589 | 0.67x | 0.0178 | 0.0009 | 87.70 | 59.07 |
| [11] `pointwise_320_16` | 0.0639 | 0.0898 | 0.71x | 0.0004 | 0.0002 | 34.48 | 24.51 |

## Linear Results

| Shape | FP16 ms | int_gemm ms | AWQ ms | int_gemm/AWQ | FP16/AWQ | FP16 stdev | int_gemm stdev | AWQ stdev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1] `attn_proj_320_m2688` | 0.0132 | 0.1064 | 0.0305 | 3.49x | 0.43x | 0.0000 | 0.0010 | 0.0001 |
| [2] `attn_proj_320_m5376` | 0.0239 | 0.2305 | 0.0525 | 4.39x | 0.46x | 0.0001 | 0.0006 | 0.0003 |
| [3] `attn_proj_512` | 0.0354 | 0.0783 | 0.0787 | 1.00x | 0.45x | 0.0004 | 0.0023 | 0.0005 |
| [4] `attn_proj_640_m2688` | 0.0350 | 0.2551 | 0.0611 | 4.18x | 0.57x | 0.0006 | 0.0004 | 0.0002 |
| [5] `ffn_512_2048` | 0.1500 | 3.4431 | 0.1724 | 19.97x | 0.87x | 0.0012 | 0.7185 | 0.0003 |
| [6] `ffn_640_2560` | 0.1083 | 0.7205 | 0.1301 | 5.54x | 0.83x | 0.0003 | 0.0024 | 0.0014 |
| [7] `large_2048` | 0.1769 | 0.1836 | 0.1830 | 1.00x | 0.97x | 0.0007 | 0.0009 | 0.0006 |
| [8] `large_4096` | 0.3511 | 0.2601 | 0.2632 | 0.99x | 1.33x | 0.0102 | 0.0035 | 0.0006 |
| [9] `small_m_4096` | 0.0788 | 0.2262 | 0.4378 | 0.52x | 0.18x | 0.0004 | 0.0005 | 0.0023 |

## Output Files

| Artifact | Path |
| --- | --- |
| CSV | `integration/results/current_bs42_n168_s200_static/separated_kernel_benchmark_r100_i50/ldm_int8_kernel_compare.csv` |
| JSON | `integration/results/current_bs42_n168_s200_static/separated_kernel_benchmark_r100_i50/ldm_int8_kernel_compare.json` |
| Plots | `integration/results/current_bs42_n168_s200_static/separated_kernel_benchmark_r100_i50/plots` |
| Report | `integration/results/current_bs42_n168_s200_static/separated_kernel_benchmark_r100_i50/SEPARATED_KERNEL_R100_REPORT.md` |
