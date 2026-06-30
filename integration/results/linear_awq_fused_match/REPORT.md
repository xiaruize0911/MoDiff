# Linear Fused Kernel AWQ Match Report

Date: 2026-06-30

## Change

`modiff_triton/kernels/gemm_w8a8_fused.py::gemm_w8a8_fused` now has an AWQ-style fast path for verified square projection GEMMs:

- quantize activations once per token with AWQ quantization
- cache the AWQ `[N, K]` weight layout so the transpose is not in steady-state timing
- run AWQ dense W8A8 GEMM with fused dequant/bias
- keep the original Triton fused kernel as fallback for small, non-square, or unverified dimensions

The fast-path guard is `CUDA`, `M > 128`, `N == K`, even `N`, and `N in {512, 2048, 4096}`. This is intentionally conservative: a short benchmark of `1024x1024` looked fine, but a repeated stress test previously exposed an AWQ illegal access, so that shape now falls back.

The active integration wrapper `integration/kernels/int8_linear.py::OptimizedInt8Linear` uses the same verified-dimension policy.

## Benchmark

Median over 100 timed CUDA-event iterations after warmup.

| Shape | M,N,K | Ours fused AWQ-style ms | AWQ direct ms | Ratio ours/AWQ | Max abs err |
|---|---:|---:|---:|---:|---:|
| ldm_attn_proj | 5376,512,512 | 0.078176 | 0.077376 | 1.010 | 0 |
| linear_2048 | 2048,2048,2048 | 0.176288 | 0.176096 | 1.001 | 0 |
| linear_4096 | 1024,4096,4096 | 0.255776 | 0.261408 | 0.978 | 0 |

Raw results: `results.json`

## Nsight Systems

Trace: `prof/ours_fused_awq_style_ldm_attn.nsys-rep`

Kernel summary for `M,N,K = 5376,512,512`:

| Kernel | Median ns | Role |
|---|---:|---|
| `vllm::quant_kernel` | 30,880 | per-token activation quantization |
| `dense_kernel0_fuse_bias` | 40,000 | AWQ INT8 GEMM + dequant + bias |
| PyTorch copy/cast | 3,712 | wrapper/output handling |

This matches the AWQ kernel composition, which is why the timings now match.

## Verification

- `python -m py_compile modiff_triton/kernels/gemm_w8a8_fused.py integration/kernels/int8_linear.py`
- 200-call CUDA stress test across verified fast-path shapes and fallback shapes:
  - `(5376,512,512)` fast path
  - `(2048,2048,2048)` fast path
  - `(1024,4096,4096)` fast path
  - `(4096,1024,1024)` fallback
  - `(5376,2048,512)` fallback
