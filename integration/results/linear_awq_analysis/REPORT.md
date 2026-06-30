# Linear Layer AWQ Comparison and Kernel Findings

Date: 2026-06-30  
GPU: NVIDIA A40, PyTorch 2.4.1+cu124

## Scope

Benchmarked the active integration layer used by `benchmark_ldm.py`:

- `integration/kernels/int8_linear.py::OptimizedInt8Linear`
- `backend=fp16`
- `backend=int_gemm`
- `backend=awq` for verified square shapes

Nsight Systems was used for kernel composition. Nsight Compute was attempted but blocked by `ERR_NVGPUCTRPERM`, so hardware counter/roofline data was not available on this host.

## What Was Slow

Before the fix, `backend=int_gemm` performed activation quantization through separate PyTorch kernels before GEMM:

1. activation abs/reduction for scale
2. divide
3. round
4. clamp
5. cast/store `int8`
6. INT8 GEMM
7. int32-to-float dequantization/bias/output conversion

That extra global-memory traffic made active linear layers much slower than expected.

Before patch, active `OptimizedInt8Linear` median latency:

| Shape | M,N,K | FP16 ms | int_gemm ms | AWQ ms |
|---|---:|---:|---:|---:|
| ldm_attn_proj | 5376,512,512 | 0.053 | 0.435 | 0.258 |
| ldm_ffn | 5376,2048,512 | 0.145 | 1.092 | 0.169 |
| linear_2048 | 2048,2048,2048 | 0.163 | 0.810 | 0.176 |
| linear_4096 | 1024,4096,4096 | 0.326 | 1.115 | 0.256 |
| small_decode | 128,4096,4096 | 0.080 | 0.242 | 0.434 |

Raw data: `optimized_int8_linear_before.json`

## Fix Applied

Changed `OptimizedInt8Linear._int8_gemm_linear` to dispatch verified square CUDA projection shapes through AWQ's W8A8 path:

- `awq_fused_quant_gemm_w8a8` when no static input scale is supplied
- `awq_gemm_w8a8` when a static input scale is supplied
- fallback remains the existing local path

The guard is intentionally conservative:

- CUDA only
- `M > 128`
- even output features
- `out_features == in_features`

Reason: AWQ was stable for square attention-style projections, but repeated non-square FFN expansion/contraction dispatches triggered CUDA illegal memory accesses on this host. Those shapes remain on our local fallback.

Changed file:

- `integration/kernels/int8_linear.py`

## Final Benchmark

Final guarded implementation, median over 50 timed iterations:

| Shape | M,N,K | Backend | Median ms | vs FP16 | Max abs err vs FP16 |
|---|---:|---|---:|---:|---:|
| ldm_attn_proj | 5376,512,512 | fp16 | 0.0548 | 1.00x | 0 |
| ldm_attn_proj | 5376,512,512 | int_gemm | 0.1547 | 2.83x | 0.0320 |
| ldm_attn_proj | 5376,512,512 | awq | 0.1614 | 2.95x | 0.0320 |
| ldm_ffn_expand | 5376,2048,512 | fp16 | 0.1465 | 1.00x | 0 |
| ldm_ffn_expand | 5376,2048,512 | int_gemm | 1.1005 | 7.51x | 0.0388 |
| ldm_ffn_contract | 5376,512,2048 | fp16 | 0.0915 | 1.00x | 0 |
| ldm_ffn_contract | 5376,512,2048 | int_gemm | 0.8128 | 8.88x | 0.0366 |
| linear_2048 | 2048,2048,2048 | fp16 | 0.1629 | 1.00x | 0 |
| linear_2048 | 2048,2048,2048 | int_gemm | 0.1762 | 1.08x | 0.0303 |
| linear_2048 | 2048,2048,2048 | awq | 0.1785 | 1.10x | 0.0303 |
| linear_4096 | 1024,4096,4096 | fp16 | 0.3265 | 1.00x | 0 |
| linear_4096 | 1024,4096,4096 | int_gemm | 0.2621 | 0.80x | 0.0306 |
| linear_4096 | 1024,4096,4096 | awq | 0.2627 | 0.80x | 0.0306 |
| small_decode | 128,4096,4096 | fp16 | 0.0800 | 1.00x | 0 |
| small_decode | 128,4096,4096 | int_gemm | 0.2361 | 2.95x | 0.0332 |

Raw data: `optimized_int8_linear_final.json`

## Nsight Systems Findings

For square attention projection after the fix, `nsys` shows the expected AWQ composition:

| Kernel | Median ns | Role |
|---|---:|---|
| `vllm::quant_kernel` | 30,784 | per-token activation quantization |
| `dense_kernel0_fuse_bias` | 39,328 | AWQ INT8 GEMM + dequant + bias |
| PyTorch copy/cast kernels | ~2,200-2,900 | wrapper/output dtype handling |

Profiler files:

- `prof/optimized_int8_after_ldm_attn.nsys-rep`
- `prof/optimized_int8_after_ldm_attn_kernels.csv`
- `prof/ours_gemm_kernels.csv`
- `prof/awq_kernels.csv`

## Weak Spots Still Left

1. Non-square FFN linears are still poor.
   - `5376x512 -> 2048`: `int_gemm` is 7.5x slower than FP16.
   - `5376x2048 -> 512`: `int_gemm` is 8.9x slower than FP16.
   - AWQ is not safely usable for these on this host due repeated illegal memory accesses.

2. Small-M decode remains poor.
   - AWQ safe fallback is slower at `M=128`.
   - Current local path is still 3x slower than FP16.

3. Our local path still has excessive IO.
   - It materializes quantized activations to global memory.
   - It launches multiple elementwise kernels for quantization.
   - It returns through FP32/dequant paths in several cases.

## Verification

- `python -m py_compile integration/kernels/int8_linear.py`
- Active-layer CUDA benchmark wrote `optimized_int8_linear_final.json`
- Guarded `int_gemm` path passed 200 repeated calls for:
  - `(M,K,N)=(5376,512,512)`
  - `(5376,512,2048)`
  - `(5376,2048,512)`
  - `(2048,2048,2048)`
  - `(1024,4096,4096)`

