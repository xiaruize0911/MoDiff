# 1x1 Conv2d AWQ-GEMM Optimization Report

Date: 2026-06-30  
GPU: NVIDIA A40

## Change

Added a guarded 1x1 Conv2d fast path to:

- `integration/kernels/int8_optimized.py::OptimizedInt8Conv2d`

For eligible standard INT8 Conv2d calls, the layer now maps 1x1 Conv2d to AWQ W8A8 GEMM:

1. input is already NCHW channels-last
2. view as NHWC and flatten to `[B * H * W, C]`
3. use existing 1x1 quantized weight as AWQ `[K, C]`
4. call `awq_fused_quant_gemm_w8a8`
5. view output back to NCHW channels-last

Guard:

- `kernel_size == (1, 1)`
- `stride == (1, 1)`
- `padding == (0, 0)`
- `dilation == (1, 1)`
- `groups == 1`
- CUDA input
- channels-last input
- `enable_awq_1x1 == True`

The old CUTLASS implicit-Conv2d path remains available by setting:

```python
module.enable_awq_1x1 = False
```

## Benchmark

Median over 100 CUDA-event iterations.

| Shape | FP16 Torch ms | CUTLASS INT8 Conv1x1 ms | AWQ-GEMM 1x1 ms | Speedup vs CUTLASS | AWQ vs FP16 |
|---|---:|---:|---:|---:|---:|
| `B32 C128 H64 W64 -> K128` | 0.2555 | 0.5408 | 0.3135 | 1.73x | 1.23x slower |
| `B8 C320 H64 W64 -> K320` | 0.1848 | 0.4284 | 0.2281 | 1.88x | 1.23x slower |
| `B8 C640 H32 W32 -> K640` | 0.1097 | 0.2173 | 0.1297 | 1.68x | 1.18x slower |
| `B8 C1280 H16 W16 -> K1280` | 0.0715 | 0.1182 | 0.0991 | 1.19x | 1.39x slower |
| `B8 C320 H64 W64 -> K640` | 0.2994 | 0.6342 | 0.3107 | 2.04x | 1.04x slower |

Raw results:

- `integration/results/conv1x1_awq_gemm/results.json`

## Nsight Systems

Trace:

- `integration/results/conv1x1_awq_gemm/prof/awq_1x1_pwise_320.nsys-rep`
- `integration/results/conv1x1_awq_gemm/prof/awq_1x1_pwise_320_kernels.csv`

For `B8 C320 H64 W64 -> K320`, the optimized 1x1 path is dominated by:

| Kernel | Median ns | Role |
|---|---:|---|
| `vllm::quant_kernel` | 117,696 | AWQ per-token activation quantization |
| `dense_kernel0_fuse_bias` | 116,032 | AWQ W8A8 GEMM + dequant + bias |

This confirms the 1x1 Conv2d path is actually using the AWQ-style quantize + GEMM sequence instead of the CUTLASS implicit Conv2d kernel.

## Stability

Passed 300 repeated calls for:

- `B32 C128 H64 W64 -> K128`
- `B8 C320 H64 W64 -> K320`
- `B8 C640 H32 W32 -> K640`
- `B8 C1280 H16 W16 -> K1280`
- `B8 C320 H64 W64 -> K640`

All returned channels-last NCHW outputs.

## Interpretation

The optimization worked for INT8 1x1 Conv2d: AWQ-GEMM is 1.19x-2.04x faster than our existing CUTLASS implicit-Conv2d path.

It does not generally beat FP16 Torch Conv2d on these tested shapes. The closest case is the expansion shape `C320 -> K640`, where AWQ-GEMM is only 1.04x slower than FP16.

This suggests AWQ-GEMM is useful if we want a better INT8 pointwise Conv2d backend, but it is not yet an end-to-end reason to quantize pointwise Conv2d unless memory/consistency with the INT8 path matters.

