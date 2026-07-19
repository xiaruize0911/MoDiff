# Retired kernels backup

`gemm_wxax_own_kernels_2026-07-18.cu` is a verbatim copy of `csrc/kernels/gemm_wxax.cu`
as it stood on 2026-07-18, **before** we consolidated the int8/int4 Linear GEMM path onto
the AWQ-tiling ports.

## What it contains that the live file no longer does
- `gemm_w8a8_kernel<MT,OUT_I8,WideK>` / `gemm_w4a4_kernel<MT,OUT_I8>` — the hand-written
  templated tensor-core kernels (MT register-blocking, WideK 32/64 K-tile gating,
  `GW_STAGES` cp.async pipeline).
- Host entry points `gemm_w8a8`, `gemm_w4a4`, `gemm_w8a8_out_int8`, `gemm_w4a4_out_int8`.
- Helpers/macros `GW_WARPS/GW_BN/GW_STAGES/GW_LDS/GW_LDS8`, `gw_pick_mt`, `gw_pick_widek`,
  `gw_bias_ptr`.

## Why retired
The AWQ-tiling ports (`gemm_w8a8_awq` / `gemm_w4a4_awq`) beat these at every measured shape
(int8 vs fp16 6/6 and AWQ-ref 4/6; int4 vs fp16 up to 2.29×) and became the sole production
backend. See `docs/quant_speedup_vs_fp16_2026-07-16/` (SESSION_REPORT / NEXT_STEPS).

## Note
This directory is **not** listed in `setup.py` `sources`, so nothing here is compiled.
It is reference-only. The int8-output variants (`*_out_int8`) previously fed the fused
qkv→flash attention prototype in `integration/fused_ops/quantized_attention.py`, which was
refactored off them at the same time.
