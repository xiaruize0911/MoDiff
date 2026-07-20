> ⚠️ **OBSOLETE / DO-NOT-CITE for fp16-vs-int8 (scrubbed 2026-07-20).** This doc benchmarks
> int8/int4 flash **vs fp16 FlashAttention-2** ("the real"). That is **not** a fair fp16↔int8
> comparison: flash is an *algorithmic* optimization (scores in SRAM) that *both* precisions can
> use, so pitting int8 against fp16-flash conflates algorithm with precision. It also predates the
> decision to **remove flash from the pipeline** (attention is fp16 **MATH**). For the fair,
> current numbers use [E2E_CORRECTION_2026-07-20.md](E2E_CORRECTION_2026-07-20.md) (e2e int8 1.08×
> vs true fp16) and [KERNEL_ALLSHAPES_ATTN_LINEAR.md](KERNEL_ALLSHAPES_ATTN_LINEAR.md) /
> `scripts/attn_math_int_vs_fp16.py` (materialized int8 vs fp16 **MATH**, the same-algorithm
> comparison). Kept below only as a historical record of the flash-kernel experiment.

# Attention kernel — all-shapes analysis (churches UNet, b128) — HISTORICAL (fp16-flash yardstick)

Date: 2026-07-19. A40, torch 2.4.1+cu124. Median of 5×50 timed iters, 20 warm each,
after a GPU clock burn-in. "The real" here = fp16 FlashAttention-2 — **an unfair yardstick for
fp16-vs-int8 (see banner above); this is the flash-*algorithm* experiment, not a precision claim.**

Scripts: `scripts/capture_attn_shapes.py`, `scripts/attn_allshapes_bench.py`, `scripts/plot_attn_allshapes.py`
Data: `data/attn_shapes.csv`, `data/attn_allshapes_kernel_b128.csv`, `data/attn_policy_b128.csv`
Figure: `fig_attn_allshapes_b128.png`

## 1. Shapes the model actually runs (per forward)

Captured by hooking every `AttentionBlock` in the churches UNet (config-only, no ckpt).
All blocks use nh=8, so at b128 every block has BH = b·nh = 1024. Only T and hd differ.

| C | nh | hd | T | BH@b128 | blocks/fwd | flash-quant eligible? |
|---|----|----|----|--------|-----------|-----------------------|
| 192 | 8 | 24 | 1024 | 1024 | **5** | YES |
| 384 | 8 | 48 | 256 | 1024 | **5** | YES |
| 384 | 8 | 48 | 64 | 1024 | **5** | YES |
| 768 | 8 | 96 | 16 | 1024 | **5** | no (hd=96 > 48) |
| 768 | 8 | 96 | 4 | 1024 | **1** | no (hd=96 > 48, T%64≠0) |

**21 attention blocks / forward, 5 unique shapes.** Eligibility = `T%64==0 and hd<=48`
(the mma flash path). 15 of 21 blocks are eligible; the 6 hd=96 blocks always stay fp16 flash.

## 2. Per-shape kernel profiles (kernel-only µs, vs real)

| shape (hd/T) | fp16 flash (real) | fp16 MATH (old) | int8 flash (ours) | int4 flash (ours) | int8 mat. |
|---|---|---|---|---|---|
| hd24/T1024 | **1855** (1.00×) | 16216 (0.11×) | 7506 (0.25×) | 13148 (0.14×) | N/A* |
| hd48/T256 | **212** (1.00×) | 1356 (0.16×) | 2010 (0.11×) | 2352 (0.09×) | 2434 (0.09×) |
| hd48/T64 | **65** (1.00×) | 157 (0.42×) | 198 (0.33×) | 183 (0.36×) | 519 (0.13×) |
| hd96/T16 | **42** (1.00×) | 67 (0.63×) | — | — | — |
| hd96/T4 | **39** (1.00×) | 66 (0.59×) | — | — | — |

rel-L2 vs fp32: int8 flash 0.014–0.024, int4 flash 0.14–0.15 (all within the model's
e2e gates from PROGRESS.md). *int8 materialized errored at hd24 (kernel path); the flash
kernel covers that shape anyway.

**fp16 FlashAttention-2 is the fastest attention path at every single shape** — even
kernel-only, ignoring quantize. It keeps [BH,T,T] scores in SRAM; the quant paths pay
int MMA + a dynamic requant + (for the realistic total) a quantize prologue.

The quantize prologue is brutal: at hd24/T1024 it costs **5608 µs** — 3× the entire fp16
flash kernel — because it streams q/k/v (25M elts each) through abs/amax/round/clamp/pad/
contiguous in eager PyTorch.

## 3. Expected total-attention time per forward (weighted by the 21-block counts)

| policy | µs / forward | speedup vs real |
|---|---|---|
| fp16 MATH everywhere (old default) | 89029 | 0.12× |
| **fp16 flash everywhere (THE REAL, shipped)** | **10965** | **1.00×** |
| int8 flash where eligible, else fp16 flash | 94844 | 0.12× |
| int4 flash where eligible, else fp16 flash | 166988 | 0.07× |
| int8 flash *kernel-only* (quant fused away) | 48893 | 0.22× |
| int4 flash *kernel-only* (quant fused away) | 78662 | 0.14× |

## 4. Expected speedup vs the real — the answer

**Negative at every shape and in aggregate.** Switching the 15 eligible blocks from fp16
flash to our int8 flash makes total attention **0.12×** (≈8.6× *slower*); int4 is 0.07×.
Even in the best imaginable case — quantize fully fused into a prologue so it's free — the
int8 flash kernel-only ceiling is **0.22×** (4.5× slower) and int4 is 0.14×.

There is no shape at which the current quantized flash kernel beats FlashAttention-2.

## 5. Why, and what it means for the plan

- FA-2 at these tiny head dims (hd=24/48) is extremely well optimized and memory-light.
  Our int8 kernel is ~14× above its own exp-limited floor (533 µs, per PROGRESS.md), but
  even that floor (≈0.35× fp16 flash at hd24/T1024) would still lose.
- Attention is only **~9% of the step** after fp16 flash shipped. So even a *hypothetical*
  int8 flash that hit 2× on the eligible blocks would move e2e by only ~3–4%.
- The quantize prologue alone > fp16 flash kernel — any viable quant-attention MUST fuse
  quantize into the QKV-GEMM epilogue (produce int8 q/k/v directly), not do it in eager.

**Recommendation:** the "optimize int8 flash → beat fp16 flash" goal is not reachable at
churches shapes with this kernel — fp16 flash should remain the default. Keep int8/int4
flash opt-in (`MODIFF_FLASH_ATTN=8|4`) for larger-hd / longer-T models where FA-2's edge
shrinks, but stop spending optimization effort here; the e2e ceiling is ~3–4%. Redirect to
the higher-leverage GEMM/GroupNorm stages (attention is no longer the bottleneck).
