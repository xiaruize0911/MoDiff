# Addendum — the 82.2% E2E figure rests on a numerically wrong route

Three fixes were applied and, in verifying them, the recorded end-to-end state turned out to be
worse than the sweep suggested. This corrects it.

## Fix 1: blockwise conv could not complete a single MoDiff run

`OptimizedInt8Conv2d.forward` handles blockwise at line 1618; an ineligible layer "falls through
to the shipped path", but the path it fell into was `_forward_modulated_static_fused_silu` — a
FUSED entry point that `_sim_guard` refuses. At t=T every layer is uncalibrated, hence ineligible,
so the very first step raised. The five kill switches do not cover it: they gate the entry points
`fused_resblock` uses, and this one is reached from `forward()` itself, so the guard's own message
is incomplete.

Fixed by skipping that branch when `MODIFF_CONV_BLOCKK != 0`, landing on the unguarded
`_forward_modulated` / `_forward_first_step`. **The unfused blockwise-conv MoDiff arm now runs and
is numerically correct** — decoded-sample MSE 4.12e-04 (33.85 dB) against the same configuration
without blockwise, i.e. 0.24x the 1.705e-03 run-to-run floor.

## Fix 2: blockwise `a_hat` is now the default, and it degrades instead of raising

`MODIFF_AHAT_BLOCK` default 0 -> **32**. `_ahat_block()` became an instance method so it can return
0 for a layer whose channel count 32 does not divide, instead of `_pack_ahat_along_c` raising. As
an opt-in that only affected whoever set it; as a default it would break any model with a channel
count off the 32-grid. Verified on the real model: 70/70 conv layers get int8 `a_hat` with 4D
blockwise scales, 0 degraded; C=100 correctly returns 0.

Justification is measured (see `docs/ahat_conv_report_2026-09-02`): 1.016x faster end to end at
W8A8, peak -612 MB, cache 1403 -> 789 MB, samples indistinguishable, `eta_cum` 0.053 against a
0.30 threshold. Faster *and* smaller than the fp16 cache it replaces.

## Fix 3 (substituted): what I actually did instead

I had listed "relax the `blk32_vec4` gate" and "fp16 block scales". Both were re-scoped:

- **`blk32_vec4`'s `CPG % 4 == 0` is a correctness requirement, not a gate.** vec4 gives one thread
  four channels, and they must share a GN group or they would use different mean/inv_std. Making
  C=192 (CPG 6) and C=576 (CPG 18) use vec4 needs a kernel variant that loads two stats pairs and
  selects per channel — not a one-line relaxation. My earlier framing was wrong.
- **fp16 block scales** buy 789 -> 745 MB, 0.6% of peak, for a ~14-site change touching every
  a_hat read and write (`bind_ahat_cache`, `ahat_qparams`, `ahat_resolve`, `ahat_b32_*`, both
  commit kernels, `ahat_pack_block_nhwc`, two Python files). Poor value against the 16.4 ms below.

Instead: the composition rule that fix 2 exposed, made explicit. Both blockwise-conv routes keep
`a_hat` in fp16 (`_blockk_dequant` returns fp16; `d = x - a_hat` is an fp16 subtract) and both
decide `is_first_step` from `a_hat_cache.dtype != torch.float16`. With int8 `a_hat` that predicate
is always true, which would force the first-step branch on every step and silently disable the
temporal accumulation. Measured symptom: with `MODIFF_CONV_BLOCKK=64` the `AHAT_BLOCK=32` arm had
byte-identical peak allocation to the `AHAT_BLOCK=0` arm (7734 MB both) — the setting did nothing.
`_ahat_block()` now returns 0 when `_conv_blockk() != 0`, so the two features are explicitly
mutually exclusive rather than silently no-ops.

## The corrected E2E picture

W8A8, batch 128, 50 DDIM, one binary, a_hat fp16 unless stated:

| configuration | ms/step | vs shipped |
|---|---|---|
| shipped (fusions on, no blockwise conv) | 81.57 | 1.000x |
| all five fusions off, no blockwise conv | 97.97 | 0.833x |
| all five fusions off + blockwise conv B=64 | **157.77** | **0.517x** |

So the fusions are worth **16.4 ms/step** and the unfused blockwise-conv route costs a further
**59.8 ms/step**. That second number is the eager quantize/dequant the route performs per layer per
step (`_blockk_quant`, `_blockk_dequant`, `x - a_hat`, `a_hat + dq`, each a full-activation
elementwise pass) — exactly the "+21/+16 ms fusion loss the first wiring paid" that
`blockk_gn_fused`'s docstring says it exists to remove.

**Therefore the recorded 82.2% belongs to the FUSED route, and that route's MoDiff arm is the one
with the relL2 0.49 defect (open item 1). The 80% goal is not currently met by any route that is
both fast and correct.** The correct route is 51.7%; the fast route is wrong.

## What changed about the next step

Fix 1 supplies what open item 1 was missing: a **numerically correct blockwise-conv reference**
(33.85 dB) to diff `blockk_gn_fused` against, layer by layer and step by step, instead of chasing
a 0.49 relL2 with no ground truth. That is the one thing worth doing next; everything else on the
blockwise-conv axis is downstream of it.

## Errors of mine corrected here

1. **`MODIFF_CONV_BLOCKK` is a block size, not a flag.** `_blockk_eligible` rejects anything
   outside {32,64,128,256}, so the `"1"` my harness patch set disabled blockwise on every layer.
   The first decomposition I reported from it (109.09 ms, "74.8%, = 16.4 + 11.1 ms") was measuring
   the non-blockwise unfused path; it is withdrawn and replaced by the table above.
2. I hypothesised `blockk_gn_fused` discarded the conv's return value while `o_hat` was not
   updated in place. `conv2d_int8_blockk.cu:391` does `out = *o_hat_opt`, so the kernel aliases
   its output to `o_hat` and the pattern is correct. Wrong hypothesis.
3. See fix 3 for the `blk32_vec4` mis-scoping.

---

# Addendum 2 — fusion + blockwise quantize: it runs at 74.6%, and the defect is localised to 4 layers

## Speed

| configuration | ms/step | vs shipped |
|---|---|---|
| shipped (fusions on, no blockwise conv) | 81.47 | 1.000x |
| **fusions on + blockwise quantize B=64** | **109.30** | **0.746x** |
| fusions on + blockwise quantize B=32 | 118.49 | 0.688x |
| fusions off + blockwise conv B=64 (correct route) | 157.77 | 0.517x |

So the fused route is 2.1x better than the unfused one, and B=64 beats B=32 by 1.08x. It is still
short of 80%, and the recorded 82.2% is not reproduced here.

## It is numerically wrong, and the error enters at step 2

Decoded-sample MSE against the shipped path, same seed, by step count:

| steps | image MSE | PSNR |
|---|---|---|
| 1 | 3.14e-06 | **55.03 dB** |
| 2 | 1.81e-03 | 27.43 dB |
| 4 | 6.04e-03 | 22.19 dB |
| 50 | 2.57e-02 | 15.90 dB |

**The first step is essentially exact; the accumulation step is wrong and compounds.** B=32 is
equally wrong (2.55e-02), so it is not a block-size-specific kernel bug.

## Exactly 4 of 62 layers freeze

Instrumenting `blockk_gn_fused` per layer: `output_blocks.3.0.out_conv`,
`output_blocks.6.0.in_conv`, `output_blocks.9.0.in_conv`, `output_blocks.12.0.in_conv` have BOTH
`a_hat` and `o_hat` byte-identical on every non-first step — frozen at their t=T value for the
whole trajectory. Every other layer advances normally, and `is_first_step` fires exactly once per
layer everywhere.

## The mechanism, measured

For a frozen layer (`output_blocks.6.0.in_conv`, C=1152, CPG=36, eligible, B=64):

| quantity | frozen layer | healthy layer (C=1536) |
|---|---|---|
| `|sb|` max (block scale) | **7.874e-15** | 1.53e-02 |
| `|q|` max (codes) | 127 | 127 |
| a_hat change per step | **0** | 1.945 |

`7.874e-15 = 1e-12 / 127` — precisely the floor in
`g = fmaxf(ahat_group32_amax(dm), 1e-12f); sc = g/127`. So the kernel's per-block amax came out
**zero for every block in the layer**, i.e. its GN+SiLU output was ~0 everywhere. Then
`inv = 127/1e-12` saturates every code to ±127, and `a_hat += q*sc` is an increment of ~1e-12,
which underflows fp16 to nothing. Both caches freeze, and the frozen o_hat retains only the bias
(0.04291). Everything observed follows from that one quantity.

## It is NOT the kernel

Two checks separate the kernel from its inputs:

1. **The reference disagrees with the kernel on the same arguments.** Computing
   `F.silu(F.group_norm(x, ng, gn_w, gn_b, eps))` from the exact tensors the hook passes gives
   `|silu|max = 4.398`, with `x` in [-10.2, 10.9], GN output in [-4.36, 4.45], per-group variance
   in [0.90, 5.92], and healthy `|gn_w| = 0.986`, `|gn_b| = 0.272`. Nothing is degenerate.
2. **A standalone repro of the kernel is correct at every channel count**, including the frozen
   layer's: at B=64, num_groups=32, C ∈ {384, 576, 768, 1152, 1536} (CPG 12/18/24/36/48) the
   kernel's `|sb|` max is 0.035-0.040 against a reference `|silu|` max of 4.4-5.1. No C or CPG
   value reproduces the failure.

So the kernel, given those arguments, produces the right answer; the four layers must be receiving
something other than what the instrumentation printed. Candidates not yet eliminated: the tensor
`x` reaching the kernel differing from the `x` the wrapper observed (a layout or aliasing
difference that `x.contiguous(ChannelsLast)` resolves differently from `x.float()`), or the
`mean`/`inv_std` from `gn_launch_group_stats` for these specific inputs.

**Next probe, and it is decisive:** dump the four layers' exact `(x, gn_w, gn_b, ng, eps)` to disk
from inside `blockk_gn_fused`, then replay them through `gn_silu_blockk_quantize_b32` standalone.
If the standalone replay reproduces `|sb| = 7.9e-15` the fault is in the kernel on that specific
data; if it does not, the fault is between the hook and the kernel call. That separates the two
remaining hypotheses in one run and needs no model.

## Status against the goal

No route is both fast and correct: the fused route is 0.746x and wrong in 4 layers; the unfused
route is numerically correct (33.85 dB, 0.24x the reproducibility floor) at 0.517x. Fixing these
four layers would make the fused route correct at 0.746x, still short of 0.80x but within reach of
the tile work in FINDINGS.md (cfg8's 86.6% is conv-kernel-only and has not been ported).

---

# Addendum 3 — the tile was swapped, measured, and it does not pay at the BLK the fused route can supply

## What was built

`conv2d_blockk_cfg8` in `conv2d_blockk_tune.cu`: the swept-best tile
(M64 N128 TK=128B, warp 64x16, 2 stages) as a production entry point, with a new `ACCUM` template
parameter on `blockk_tune_kernel` whose epilogue aliases `Out` to `o_hat` and does a `__hadd2`
read-modify-write, matching `conv2d_int8_blockk`'s ACCUM. Parameterised on BLK ∈ {64,128,256};
only this tile is instantiated, to keep the 24-entry table's compile time from doubling.

The tile was adopted from the tune kernel rather than hand-ported into `conv2d_int8_blockk`,
because that kernel's swizzle is hardcoded 4-way (`^ ((row/2) & 3)`, valid only for CTA_K=64) and
CTA_K=128 needs 8-way plus matching cp.async loop bounds; the tune kernel already parameterises
both as `CPR = TK/TU_PACK` and `& (CPR-1)` and is validated at cfg8.

## Correctness

fp32 reference (dequantize per (pixel, C-block), then conv), against the fp16-store floor:

| precision | shape | mode | relL2 | floor |
|---|---|---|---|---|
| int8 | 2x768x16x16->128 | plain / accum | 2.98e-04 / 3.61e-04 | 2.07e-04 |
| int8 | 1x1536x8x8->256 | plain / accum | 3.04e-04 / 3.67e-04 | 2.10e-04 |
| int4 | 2x768x16x16->128 | plain / accum | 3.00e-04 / 3.57e-04 | 2.08e-04 |
| int4 | 1x1536x8x8->256 | plain / accum | 2.90e-04 / 3.53e-04 | 2.06e-04 |
| both | BLK=64, accum | | 3.58e-04 / 3.55e-04 | 2.08e-04 |

10/10 PASS. The ACCUM epilogue is correct at both precisions and both block sizes.

## Conv, per layer (batch 128, frequency-weighted over the 20 UNet conv shapes)

At **BLK=256** cfg8 beats the production tile on all 7 shapes it covers, decisively on the
small-spatial ones where the production tile is worst:

| C->K | HxW | shipped | cfg0 (prod) | cfg8 | cfg0 | cfg8 |
|---|---|---|---|---|---|---|
| 768->768 | 2x2 | 0.0866 | 0.1832 | **0.1137** | 2.114x | **1.312x** |
| 1536->768 | 2x2 | 0.1530 | 0.3606 | **0.2200** | 2.357x | **1.438x** |
| 768->384 | 8x8 | 0.2826 | 0.4641 | 0.4022 | 1.642x | 1.423x |
| 1536->768 | 4x4 | 0.3113 | 0.5214 | 0.4780 | 1.675x | 1.535x |
| 768->768 | 4x4 | 0.1814 | 0.2704 | 0.2583 | 1.491x | 1.424x |
| 768->768 | 8x8 | 0.4772 | 0.7544 | 0.7507 | 1.581x | 1.573x |
| 768->384 | 16x16 | 0.9389 | 1.4952 | 1.5003 | 1.593x | 1.598x |

At **BLK=64** — the only block size the fused kernel 1 can emit — it is WORSE overall:

| configuration | ms | % of shipped |
|---|---|---|
| shipped | 29.589 | 100% |
| all cfg0 / B=64 | 38.620 | **76.6%** |
| all cfg8 / B=64 | 39.802 | 74.5% |
| cfg8/B=256 on 47% + cfg0/B=64 on the rest | **37.135** | **79.7%** |

cfg8 wins the small-spatial shapes and loses the large-spatial ones at B=64 (768->768 8x8: 1.834x
vs 1.576x; 384->384 32x32: 1.549x vs 1.341x).

## E2E

| configuration | ms/step |
|---|---|
| shipped (no blockwise conv) | 81.47 |
| blockwise conv, production tile | 109.31 |
| blockwise conv, cfg8 tile at B=64 | **110.80** |

1.5 ms/step worse, exactly what the conv-kernel ratio predicts (74.5% vs 76.5% on a ~39 ms conv
bucket). Wired behind `MODIFF_CONV_BLOCKK_TILE=cfg8`, deliberately NOT the default.

## The real finding: cfg8's advantage is BLK=256, not the tile

`TK*EPB == BLK` gives exactly one scale flush per K tile and no int32 carry across tiles -- the
sweep's own comment says so. At TK=128B/BLK=256 that holds; at BLK=64 it becomes NB=2 slots per
tile and the advantage evaporates. So the swept ranking was a ranking of (tile, BLK) pairs, and
the tile alone carries none of it. My framing of "port cfg8's tile" was wrong on that point.

**The blocker is now specific and small.** `gn_silu_blockk_quantize_b32` caps at B=64 -- its own
TORCH_CHECK says larger B "needs >2 channels per thread and CPG % 4 == 0, which C=192/576
(CPG 6/18) do not satisfy". But the cfg8-eligible set is exactly C=768 (CPG 24) and C=1536
(CPG 48), both of which DO satisfy CPG % 4 == 0. So a 4-channels-per-thread variant of that kernel,
gated on CPG % 4 == 0, covers precisely the layers cfg8 wants and nothing else.

With it, the all-blockwise conv reaches the **79.7%** measured above, against the current 76.6%.
That is the whole remaining move on this axis, and it is one kernel variant, not a tile port.

---

# Addendum 4 — all code on this axis was REVERTED

Everything Addendum 3 built (`conv2d_blockk_cfg8`, the `ACCUM` template on `blockk_tune_kernel`,
its BLK parameterisation, the `MODIFF_CONV_BLOCKK_TILE` routing) and the `forward()`
fall-through fix from Addendum 1 were reverted at the user's request. This axis -- a blockwise
CONV-INPUT quantizer -- is not the configuration being pursued; that is blockwise `a_hat` STORAGE
with the conv input quantizer left per-tensor as shipped (`docs/ahat_conv_report_2026-09-02`).

What remains in the tree from this investigation is documentation only, plus the TK=32 entries
19-23 in `conv2d_blockk_tune.cu`'s table, which stay as executable evidence for the negative
result (TK=32 buys int4 100% shape coverage and loses 1.9x more than the fallback it removes).
`conv2d_blockk_tune.cu` is a sweep harness and is not on any production path.

The measurements stand and are worth keeping, because they bound this axis:

| | vs shipped int8 conv | vs fp16 conv |
|---|---|---|
| blockwise conv, production tile, B=64 | 1.729x | 1.248x |
| best all-blockwise mix (cfg8/B256 + cfg0/B64) | 1.657x | 1.196x |
| decomposition | 1.475x (our tile vs CUTLASS EVT) x 1.172x (blockwise tax) | 1.064x x 1.172x |

The gap is dominated by our hand-written mainloop losing to CUTLASS's EVT conv, not by the
blockwise dequant. Reaching 80% needs 1.25x and the tile alone costs 1.475x, so tile/BLK tuning
cannot get there -- it would take split-k or multiple tile candidates for the small-M shapes, or
abandoning the hand-written mainloop for CUTLASS EVT with a custom visitor doing the per-block
dequant. Recorded so that is not rediscovered.
