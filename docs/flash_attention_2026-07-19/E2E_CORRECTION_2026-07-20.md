# ⚠️ Correction (2026-07-20): the e2e "int8 ≈ 2×" was a baseline artifact

**Any e2e "int8 ≈ 2.00× / int4 ≈ 2.03× / int8+flash ≈ 2.14× vs fp16" figure in these docs (and in
`bench5`) is inflated.** Corrected numbers below.

## What was wrong

The e2e measurement scripts (`bench5_confirm.py`, `e2e_int8_int4_vs_fp16.py`, `e2e_flash_attn.py`)
wrapped sampling in `torch.amp.autocast('cuda', enabled=quant, ...)` — autocast fp16 was enabled
**only for the int8/int4 path**. So the "fp16 baseline" ran with autocast OFF = **fp32 data on TF32
tensor cores** (verified by kernel names: `softmax_warp_forward<float>`, `tensorop_s1688gemm`,
`s1688fprop_optimized_tf32`, `group_norm_silu<float>`), at ~half the FP16 tensor-core rate on the A40.

(Note: `benchmark_ldm.py`'s own `benchmark()` path uses `autocast(enabled = mode != 'fp32')`, so its
fp16 does get autocast — the bug was in the docs re-implementations, now fixed to `enabled=True`.)

## Corrected e2e (b128, DDIM, autocast fp16 ON for all)

| config | ms/step | vs fp32/tf32 | vs **true** fp16 |
|---|---|---|---|
| "fp16" no-autocast (fp32/tf32) | 351.9 | 1.00× | — |
| **fp16 (true, autocast)** | **190.1** | 1.85× | 1.00× |
| int8_baseline | 175.2 | 2.00× | **1.08×** |
| int4_baseline | 173.9 | 2.03× | **1.09×** |
| int8_modiff / int4_modiff | ~199 | ~1.77× | **0.95× (slower)** |

**The "2×" = ~1.85× (fp32/tf32→fp16 precision) × ~1.08× (int8 quantization).** Real end-to-end
quantization benefit vs a true fp16 baseline is **~8%**, and the modiff temporal-cache variants are
*slower* than fp16.

## Consistent with the kernel-level analysis

Attention is ~half the step and stays fp16 (unquantized); quantized conv+linear are ~16% of the step
and gain only modestly after fusion; int8 doesn't help the memory-bound/attention parts. ~8% e2e is
exactly what that predicts.

## Not affected

The **kernel-level** ratios in the other reports used explicit fp16 tensors (`torch.float16`), so they
are true-fp16 comparisons and stand: linear W8A8 1.46× / W4A4 1.83× (fused-quant), GroupNorm 1.58–2.11×,
flash-attn int8/int4 2.73×/2.78× vs fp16 MATH, materialized int8/int4 attention 0.79–1.01×.

## Fair-comparison rule: no fp16-flash in the fp16↔int8 comparison

**Flash attention is an *algorithmic* optimization (keep scores in SRAM), not a *precision* one — both
fp16 and int8 can use it.** So it must NOT be used as the yardstick when comparing fp16 vs int8: giving
the fp16 baseline flash while int8 doesn't (or vice-versa) conflates algorithm with precision. The
honest fp16↔int8 comparison holds the attention algorithm fixed — **both on MATH** — which is exactly
what the corrected e2e above does (attention is fp16 MATH in *both* fp16 and int8 runs, ~82 ms each).
On that fair footing, quantizing attention (materialized int8) is a **regression** vs fp16 MATH (~0.79×
total / ~0.93× core), so attention stays fp16. The `flash_attn_int8/int4` kernels remain built but
out-of-pipeline and are not used to argue for or against int8.

## Authoritative scripts
`scripts/true_fp16_vs_int8.py` (precision-vs-quantization split), `scripts/e2e_true_fp16_table.py`
(corrected table), `scripts/kernel_name_diff.py` (the fp32/tf32-vs-fp16 kernel-name evidence).
