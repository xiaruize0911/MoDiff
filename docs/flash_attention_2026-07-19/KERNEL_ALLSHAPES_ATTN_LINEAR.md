> ⚠️ **2026-07-20:** any "e2e ~2× quantization" figure below is inflated by a fp32/tf32 baseline —
> real int8 e2e is **~1.08× vs true fp16**. See [E2E_CORRECTION_2026-07-20.md](E2E_CORRECTION_2026-07-20.md).
> The kernel-level ratios here (fp16-tensor inputs) are unaffected.

# Kernel benchmark — all attention + linear shapes (churches, b128, MATH attention)

Date: 2026-07-19 (post flash-removal). A40, torch 2.4.1+cu124. Median 5×100 iters, 30 warm,
GPU clock burn-in. "The real" = the shipped default (fp16 MATH attention; fp16 `F.linear`).

Scripts: `scripts/kernel_bench_all_shapes.py`, `scripts/linear_gemm_only.py`, `scripts/plot_kernel_all.py`
Data: `data/kernel_attn_b128.csv`, `data/kernel_linear_b128.csv`, `data/linear_gemm_only_b128.csv`, `data/kernel_policy_b128.csv`
Figure: `fig_kernel_all_b128.png`

## Shapes (per forward)

Attention: 5 shapes / 21 blocks (nh=8, BH=1024 @ b128). Linear = qkv (C→3C) + proj (C→C),
one pair per attention block → 42 GEMMs / forward. M = b·T.

## ATTENTION — fp16 MATH (real) vs materialized int8/int4 quant attention

| hd/T | count | fp16 MATH (real) µs | int8 mat | int4 mat | int8 relL2 | int4 relL2 |
|---|---|---|---|---|---|---|
| 24/1024 | 5 | **16211** | 16043 (1.01×) | 14679 (1.10×) | 0.029 | **0.43** |
| 48/256 | 5 | **1358** | 2432 (0.56×) | 2314 (0.59×) | 0.020 | **0.35** |
| 48/64 | 5 | **156** | 523 (0.30×) | 505 (0.31×) | 0.015 | **0.26** |
| 96/16 | 5 | **65** | N/A (kernel hd>… ) | N/A | — | — |
| 96/4 | 1 | **67** | N/A | N/A | — | — |

Weighted total / forward: fp16 MATH **89018 µs**; int8 **95379 (0.93×)**; int4 87879 (1.01×).
→ **Attention stays fp16 MATH.** int8 is a slight regression; int4 is only "fast" at the
dominant shape but its rel-L2 ≈ 0.26–0.43 is broken. hd=96 blocks have no quant kernel.

## LINEAR qkv/proj — fp16 vs int8/int4 AWQ GEMM

The key result: **the standalone activation quantize decides everything.** In production the
qkv quantize is fused into GroupNorm→qkv (`group_norm_silu_quantize`), so the realistic cost
is GEMM-only.

| K→N | M | count | fp16 µs | int8 GEMM-only | int4 GEMM-only | int8 +quant | int4 +quant |
|---|---|---|---|---|---|---|---|
| 192→576 | 131072 | 5 | 432 | 328 (1.32×) | 316 (1.37×) | 474 (0.91×) | 764 (0.57×) |
| 192→192 | 131072 | 5 | 193 | 144 (1.34×) | 127 (1.52×) | 285 (0.68×) | 574 (0.34×) |
| 384→1152 | 32768 | 5 | 422 | 242 (1.74×) | 159 (2.66×) | 299 | 216 |
| 384→384 | 32768 | 5 | 135 | 90 (1.50×) | 65 (2.09×) | 159 | 122 |
| 384→1152 | 8192 | 5 | 93 | 66 (1.40×) | 46 (2.04×) | 85 | 62 |
| 384→384 | 8192 | 5 | 58 | 31 (1.86×) | 23 (2.49×) | 52 | 40 |
| 768→2304 | 2048 | 5 | 82 | 60 (1.37×) | 37 (2.22×) | 69 | 45 |
| 768→768 | 2048 | 5 | 26 | 28 (0.96×) | 16 (1.66×) | 37 | 25 |
| 768→2304 | 512 | 1 | 22 | 19 (1.15×) | 12 (1.91×) | 22 | 15 |
| 768→768 | 512 | 1 | 21 | 19 (1.12×) | 11 (1.86×) | 22 | 14 |

Weighted total qkv/proj / forward @ b128:

| policy | µs/fwd | vs fp16 |
|---|---|---|
| fp16 (real) | 7249 | 1.00× |
| **int8 GEMM-only (fused quant)** | **4982** | **1.46×** |
| int8 full (+standalone quant) | 7346 | 0.99× |
| **int4 GEMM-only (fused quant)** | **3966** | **1.83×** |
| int4 full (+standalone quant) | 9273 | 0.78× |

## Takeaways

1. **Attention: keep fp16 MATH.** No quantized attention path wins at churches shapes, and
   int4 attention is inaccurate (rel-L2 ~0.4). Consistent with removing flash.
2. **Linear qkv/proj DOES win — but only with fused quantize** (as production runs it):
   int8 **1.46×**, int4 **1.83×** weighted. With a standalone quantize the win vanishes
   (int8 0.99×, int4 0.78×) — the quantize pass over the [M,K] activation is memory-bound
   and, at the M=131072 level-0 shapes, costs as much as the GEMM it feeds. The int GEMM
   itself wins biggest at K≥384 (int4 up to 2.66×); the small-K=192 level-0 shapes are the
   weakest (int4 GEMM only 1.37–1.52×).
3. **Scale in context:** at the kernel level MATH attention (89 ms/fwd) dwarfs qkv/proj
   linear (7.2 ms/fwd) by ~12× — so the linear win, while real, is small next to attention
   now that flash is gone. The e2e quantization win (~1.08× vs true fp16 — the old "~2× in bench5"
   was a fp32/tf32-baseline artifact, see [E2E_CORRECTION_2026-07-20.md](E2E_CORRECTION_2026-07-20.md))
   is NOT primarily from qkv/proj; among quantized layers it is dominated by the **ResBlock Conv2d**
   layers (`conv2d_int8/int4`), the bulk of UNet FLOPs, NOT covered here.
