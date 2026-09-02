# Conv blockwise wired into the model: it works, it helps the BASELINE arm a lot, and MoDiff
# still dominates it on both axes

LSUN-churches LDM-KL-8, A40, batch 128, 50 DDIM, `MODIFF_LINEAR=0`, static delta,
`int8_calibration_realckpt.pt`, W8A8 static flash attention. Timing median of 2 after 1 warmup;
quality n=6, seed 20260805, latent relL2 vs the fp16 arm. fp16 = **101.84 ms/step**.
`scripts/blockk_e2e.sh` (one process per arm -- the fusion kill switches are import-time),
`data/arms.jsonl`.

## What was built to get here

`conv2d_int8_blockk` was previously baseline-only and unreachable from Python. Three additions:

- **`ACCUM` template parameter** on `conv2d_int8_blockk_kernel` -- `o_hat += A(Q(delta))` as an
  in-kernel fp16 read-modify-write, which is what the MoDiff arm needs. A template parameter, not
  a runtime flag, for the reason `docs/ahat_blockwise_2026-09-01` found the hard way. Verified
  against a fresh-output reference: max difference is exactly 1 fp16 ULP, and it is the *reference*
  that double-rounds (it rounds A to fp16 before adding; the kernel adds in fp32 and rounds once).
- **`conv_quantize_block_nhwc`** -- blockwise-along-C int8 quantize of a live conv input, B=32 and
  B=64, one warp per group. Scales bit-exact vs a torch reference; codes 99.98-100% bit-identical
  (residual is round-half tie-breaking).
- **`MODIFF_CONV_BLOCKK`** = 0 | 32 | 64 in `OptimizedInt8Conv2d`, plus
  `MODIFF_CONV_BLOCKK_CTRL=1` for a matched scalar-alpha control at the *same tile*, and a hard
  guard on all six fused entry points (same failure mode `_sim_guard` exists for).

**The `C % 64` constraint costs nothing on this model.** All 70 `OptimizedInt8Conv2d` layers
satisfy `C % 64 == 0` and `Kout % 2 == 0`; the C=4 first conv is not one of them. The fallback path
exists but never fires here. That closes the open item from `docs/act_blockwise_2026-09-01`.

## Result

### Baseline arm (PTQ, quantizes `a_t`)

| arm | ms/step | vs fp16 | vs shipped | relL2 |
|---|---:|---:|---:|---:|
| shipped (fused, CUTLASS EVT) | 72.45 | 1.406x | 1.000x | 0.3221 |
| unfused | 93.59 | 1.088x | 0.774x | 0.3219 |
| unfused + blockk scalar ctrl | 107.67 | 0.946x | 0.673x | 0.3224 |
| **unfused + blockwise B=64** | **119.66** | 0.851x | 0.605x | **0.1549** |
| unfused + blockwise B=32 | 125.83 | 0.809x | 0.576x | 0.1549 |

### MoDiff arm (quantizes the delta)

| arm | ms/step | vs fp16 | vs shipped | relL2 |
|---|---:|---:|---:|---:|
| shipped (fused, EVT o_hat) | 81.95 | 1.243x | 1.000x | 0.0897 |
| unfused | 98.41 | 1.035x | 0.833x | 0.0975 |
| unfused + blockk scalar ctrl | 142.39 | 0.715x | 0.576x | 0.0849 |
| **unfused + blockwise B=64** | **154.84** | 0.658x | 0.529x | **0.0880** |
| unfused + blockwise B=32 | 160.99 | 0.633x | 0.509x | 0.0641 |

### Cost attribution (ms/step)

| | baseline | MoDiff | what it is |
|---|---:|---:|---|
| fusion loss (shipped -> unfused) | +21.14 | +16.46 | no blockwise-emitting fused GN->quantize kernel exists |
| hand tile (unfused -> ctrl) | +14.08 | +43.98 | hand-written implicit-GEMM vs CUTLASS EVT |
| **blockwise B=64 (ctrl -> B=64)** | **+11.99** | **+12.45** | **the mainloop dequant + blockwise quantize** |
| blockwise B=32 | +18.17 | +18.60 | |

**The blockwise cost proper is +12 ms/step at B=64, and it is the same in both arms** (11.99 vs
12.45) -- which is the prediction from `docs/conv_kernel_sweep_2026-08-28` confirmed: blockwise is a
mainloop cost and the mainloop is arm-independent, so it lands as the same absolute add. On the
shipped MoDiff step that is +14.6%; my earlier projection of 21-37% (from scaling the conv bucket)
was pessimistic.

The other two rows are **not** blockwise costs and must not be charged to it. MoDiff's +43.98
hand-tile row is mostly my wiring, not the tile: the CTRL arm also loses the EVT `o_hat` fold and
pays an explicit dequant + `a_hat` add in eager torch, which the shipped kernel fuses.

## The two arms answer differently, and the budget predicted it

**MoDiff arm: blockwise buys nothing.** 0.0897 -> 0.0880 at B=64, 0.0641 at B=32, all inside this
tree's +-0.03 relL2 reproducibility. That is exactly what `docs/act_budget_2026-09-02` predicted --
in the MoDiff arm the conv-input quantizer is below the measurement floor and quantized attention
is the whole budget.

**Baseline arm: blockwise buys a LOT.** 0.3221 -> **0.1549, a 2.08x improvement**, far outside
noise. The baseline arm quantizes `a_t` on a static calibrated per-tensor scale that clips on 49 of
70 layers, and that clipping *is* its dominant error. The error budget was measured in the MoDiff
arm only, so it did not see this -- the per-layer table in `act_blockwise` had the two arms nearly
equal (0.1838 vs 0.1537), but end to end they are 3.6x apart (0.3221 vs 0.0897). **Per-layer
quantizer error did not predict which arm the granularity would matter in, either.**

## But it does not matter, because MoDiff dominates it

| | ms/step | relL2 |
|---|---:|---:|
| baseline + blockwise B=64 | 119.66 | 0.1549 |
| **MoDiff, shipped, no blockwise** | **81.95** | **0.0897** |

MoDiff is **1.46x faster AND 1.7x more accurate** than baseline-plus-blockwise. Blockwise is a
strictly worse way to buy the same quality than the temporal cache already shipping. Even granting
a perfectly fused blockwise implementation (baseline 72.45 + 11.99 = ~84.4 ms at relL2 0.1549),
MoDiff at 81.95 / 0.0897 still wins on both axes.

So the recommendation from `docs/act_budget_2026-09-02` stands, with its reasoning corrected:
**do not ship conv blockwise** -- not because it cannot improve quality (in the baseline arm it
clearly does), but because MoDiff already occupies the point it would move toward, more cheaply.

## Open

`relL2` is identical to four digits for baseline B=64 and B=32 (0.1549 / 0.1549) while their
timings clearly differ (119.66 vs 125.83), so the block size is definitely reaching the kernel.
B=32 vs B=64 is only 1.25x in per-layer quantizer error, and the baseline arm's residual error has
evidently saturated on something else (W8 weights + attention), so landing on the same 4-digit
value is plausible -- but it is a coincidence worth one confirming run at another seed.

The three cost rows above were measured with the quantize as a separate pass. A fused
blockwise-emitting GN->quantize kernel would recover most of the +21/+16 fusion row and part of the
+12 blockwise row; nobody should quote the 119.66 / 154.84 wall-clocks as the cost of blockwise.

---

# Addendum 2026-09-02: B=32 FUSED

`gn_silu_blockk_quantize_b32` -- one kernel doing GN(+mod)(+SiLU) -> blockwise-B=32 int8
quantize (-> delta + in-place `a_hat` update in the MoDiff arm), removing the separate quantize
pass. **Pair-major, 16 lanes x 2 channels = one B=32 group**, because a B=32 along-C group does
NOT nest inside a GN group here (num_groups=32 -> CPG 6/12/18/24), so a group-major CTA cannot
take the B=32 amax at all. Hooked into `_prequant_gn_conv`, ahead of the per-tensor folds.
Coverage: **310/350 conv calls fused**, the other 40 (8 resize `in_conv` layers) on the two-pass
path.

Kernel correctness: without SiLU, codes 99.99%+ bit-identical and scales exactly equal to a
two-pass reference; with SiLU 99.2% (the SiLU is evaluated slightly differently, ~5e-4 on the
amax, which flips a few codes). MoDiff recursion verified over 4 steps: `a_hat` tracks an
explicit reference to 2.2e-3 -> 3.8e-3 relL2.

| arm | ms/step | vs fp16 | vs shipped | relL2 |
|---|---:|---:|---:|---:|
| **BASELINE** shipped | 72.45 | 1.406x | 1.000x | 0.3221 |
| **BASELINE** B=32 unfused | 125.83 | 0.809x | 0.576x | 0.1549 |
| **BASELINE** B=32 **fused** | **108.23** | 0.941x | 0.669x | **0.1545** |
| MoDiff shipped | 81.95 | 1.243x | 1.000x | 0.0897 |
| MoDiff B=32 unfused | 160.99 | 0.633x | 0.509x | 0.0641 |
| MoDiff B=32 **fused** | 123.13 | 0.827x | 0.666x | **0.4894 -- WRONG** |

**Baseline arm: fusion recovered 17.6 ms and quality is unchanged (0.1549 -> 0.1545), so
108.23 is a valid number.** Blockwise B=32 fused costs +35.8 ms/step (+49.4%) over shipped and
buys 2.08x relL2. Still slower than fp16 (0.941x), and still dominated by MoDiff shipped
(81.95 ms at 0.0897 -- faster AND more accurate).

**MoDiff arm: the fused path is NOT correct yet and its 123.13 must not be quoted as a
like-for-like.** relL2 0.4894 vs 0.0641 unfused. Established so far: the fused kernel's MoDiff
math is right in isolation (recursion test above); path mixing is eliminated (a trace of every
conv call shows 0 layers on a non-blockwise path). The remaining defect is in the `o_hat` wiring
in `blockk_gn_fused`. Not found. The work performed is nearly identical to the baseline arm's,
so ~123 ms is indicative of the cost, but the arm is not correct.

Two real bugs found on the way, both in this session's own wiring:

1. `reset_state()` **zeroes** `a_hat`/`o_hat` in place and sets `is_first_step=True`; it does not
   free them. `blockk_gn_fused` had invented an allocation-shaped first-step check, which reports
   "not first" on the first step of every sample after the first, skipping the `o_hat` re-seed and
   running the whole sample with `bias=None`. **`is_first_step` is the authority.**
2. The updown resize fold (`_prequant_gn_resize_conv_modiff` -> `_conv_from_int8_o_hat`) emits
   PER-TENSOR codes and accumulates `o_hat` with a scalar alpha. On the 8 resize `in_conv` layers
   that mixed conventions with the blockwise step-1 write. It raised no error because
   `_conv_from_int8_o_hat` **had no `_sim_guard`** -- the one guarded-entry-point gap.
   Both fixed: the fold is gated off under `MODIFF_CONV_BLOCKK`, and the guard is added.

Neither fix moved the 0.489, so a third defect remains.

---

# Addendum 2: the B sweep, and why B=64 is a true optimum

Goal for this pass: get blockwise near the shipped layerwise int8 conv speed.

## A harness flaw invalidated Addendum 1's decomposition

`blockk_gn_fused` was injected at the TOP of `_prequant_gn_conv`, above the `HAS_GN_SILU_*`
kill-switch checks. So it fired even for arms asking for `fusions=off` or the scalar CTRL, which
(a) made "unfused" and "fused" the same measurement, and (b) made CTRL unmeasurable -- a profile
showed the CTRL arm running `conv2d_int8_blockk_kernel<32, true, ...>`, i.e. BLOCKWISE=true.
Now gated on the same switches and on `not _conv_blockk_ctrl()`.

Consequence: Addendum 1's "B=32 unfused 125.83 -> 109.08 from STAGES=2" was mostly the fusion,
not STAGES. STAGES=2's real effect is +5.7 ms on a genuinely two-pass B=64 arm and ~0 on fused.

## Where the conv time actually goes (batch 128, profiler, baseline arm)

| conv kernel | ms/step | vs CUTLASS |
|---|---:|---:|
| CUTLASS EVT (shipped, per-tensor) | 24.33 | 1.00x |
| blockk, **scalar alpha, same tile** | 25.77 | **1.06x** |
| blockk, blockwise **B=64** | 33.21 | 1.37x |
| blockk, blockwise **B=32** | 44.56 | 1.83x |

**The hand-written tile is at CUTLASS parity (1.06x).** The "1.1-1.5x tile deficit" in
`docs/act_blockwise_2026-09-01` was a batch-32 microbenchmark and does not hold at batch 128 on
real shapes. So essentially ALL of the blockwise cost is the mainloop dequant, and the flush rate
is the lever: 0 flushes/tile 25.77, 1 -> 33.21, 2 -> 44.56.

## Larger B does NOT help -- B is U-shaped

`BLK > BKC_CTA_K` was implemented (block spans TPB tiles, int32 carried across, flushed once per
TPB; `Kg % BLK == 0` follows from `C % BLK == 0` so no partial-block flush is needed). Correct at
B=128/256: relL2 3.5e-4 vs an exact fp32 per-block reference whose fp16 floor is 2.1e-4, and
equal-scale degeneration reproduces the scalar control 99.9%+ bitwise. **256 is the hard ceiling**
-- a block peaks at BLK*127*127 and `bkc_i2f` is exact only below 2^22, so B=256 has 1.5% margin
and B=512 would silently overflow.

Microbenchmark, C384 16x16 K384 3x3, batch 128, median of 50:

| | scalar ctrl | B=64 | B=128 |
|---|---:|---:|---:|
| ms | 0.681 | **0.850 (1.25x)** | 1.155 (1.70x) |

Carrying the int32 accumulator across tile boundaries lengthens its live range. Under
`__launch_bounds__(...,2)` that spills 160 B and costs **3.5x** (2.965 ms); dropping the register
cap for `BLK > CTA_K` gives 218 regs, no spill, 1.155 ms -- still worse than B=64, now because of
1 CTA/SM. Launch bounds are therefore conditional: min-2-blocks for `BLK <= CTA_K`, min-1 above.

So B=32 loses on flush count and B=128+ loses on register pressure. **B=64 is a genuine optimum,
not a compromise** -- which independently re-derives the B=64 recommendation the accuracy table
arrived at, from a completely different direction.

## Result: blockwise is now faster than fp16

The fused GN quantizer was B=32-only (16 lanes x 2 ch). B=64 needs a full-warp reduce instead of a
half-warp one -- `ahat_group32_amax`, 32 lanes x 2 ch -- so it is templated on BLK and still needs
only `CPG % 2 == 0` (a 4-channel-per-thread variant would need `CPG % 4 == 0`, which C=192 and 576
fail at CPG 6 and 18). Verified: codes 99.99%+ bit-identical to a two-pass reference, scales exact.

Baseline arm, batch 128, 50 DDIM:

| arm | ms/step | vs fp16 | vs shipped layerwise | relL2 |
|---|---:|---:|---:|---:|
| fp16 | 101.84 | 1.000x | — | — |
| shipped layerwise (per-tensor, fused) | 72.45 | 1.406x | 1.000x | 0.3221 |
| **blockwise B=64 fused** | **96.51** | **1.055x** | **0.751x** | **0.1543** |
| blockwise B=32 fused | 108.53 | 0.938x | 0.668x | 0.1547 |

**B=64 fused is faster than fp16 and 2.09x more accurate than the shipped per-tensor arm.**
Progress on the goal: the conv kernel went 1.83x -> **1.37x** of CUTLASS, and E2E 1.49x -> **1.33x**
of shipped.

## What still separates it from shipped (24.1 ms of E2E)

| | ms | recoverable? |
|---|---:|---|
| blockwise dequant in the mainloop (B=64, 1 flush/tile) | +7.4 | no -- intrinsic to a scale on the reduction axis |
| residual/skip-add no longer folded into the conv epilogue | ~+9 | **yes** -- same change as the `o_hat` ACCUM epilogue |
| separate GN stats pass (forced by pair-major) | +2.8 | partly -- needs a stats+apply split that keeps one pass |
| hand tile vs CUTLASS | +1.4 | marginal, already 1.06x |

The residual fold is the largest remaining item and is a known-shape change.

**int4 is not started.** `conv2d_int4_blockk` does not exist; the int4 half of the goal is
untouched. Note int4 packs two codes per byte, so the blockwise group and the pack boundary
interact -- it is not a parameter change on the int8 kernel.

## MoDiff arm still incorrect

Unchanged from Addendum 1: the MoDiff fused arm reports relL2 ~0.49 against 0.0641 for two-pass.
The kernel's MoDiff recursion is verified correct in isolation and path mixing is eliminated; the
defect is in the `o_hat` wiring in `blockk_gn_fused`. Only the BASELINE numbers above are valid.

---

# Addendum 3: the final W8A8 numbers, and two bugs in Addendum 2's own wiring

Addendum 2 stopped at B=64 fused = 96.51 ms/step. Two further fixes took it to **88.19**.

| arm | ms/step | vs fp16 | vs shipped layerwise | relL2 |
|---|---:|---:|---:|---:|
| fp16 | 101.84 | 1.000x | 0.711x | — |
| shipped layerwise (per-tensor, fused) | 72.45 | 1.406x | 1.000x | 0.3221 |
| B=32 fused | 108.53 | 0.938x | 0.668x | 0.1547 |
| B=64 fused | 96.51 | 1.055x | 0.751x | 0.1543 |
| B=64 fused + residual fold | 94.93 | 1.073x | 0.763x | 0.1539 |
| **B=64 fused + fold + allocation fix** | **88.19** | **1.155x** | **0.822x** | **0.1538** |

**The residual fold** (`RESID` template parameter on the conv epilogue, read-only so the caller's
skip tensor is not mutated) gained only **1.58 ms**, not the ~9 ms Addendum 2 projected from the
elementwise bucket. That projection was wrong: the residual add was the
`vectorized_elementwise_kernel<CUDAFunctor_add>` line (4.60 -> 2.24 ms), not the larger
`elementwise_kernel<128,4,nocast>` line.

**The allocation bug was the real 6.74 ms.** The host did

```cpp
out = torch::empty({N, Kout, P, Q}, ...)              // NCHW-contiguous
          .contiguous(at::MemoryFormat::ChannelsLast); // ...then COPIED it
```

`torch::empty` returns an NCHW-contiguous tensor and the trailing `.contiguous(ChannelsLast)`
copies a full output tensor of uninitialised data on **every conv call**. `aten::copy_` went
10.94 -> 4.27 ms/step, matching the shipped arm's 4.28 exactly. The memory format belongs in the
allocation. Same mistake is worth checking for in any other kernel host in this tree.

## Where the remaining 15.74 ms is

| | ms | removable? |
|---|---:|---|
| blockwise dequant in the mainloop (B=64) | +8.9 | no -- intrinsic to a scale on the reduction axis |
| GN stats pass, forced by the pair-major apply | +2.8 | partly |
| hand tile vs CUTLASS | +1.4 | see TUNE_FINDINGS in docs/conv_int4_blockk_2026-09-02 |
| rest (8 resize layers on the two-pass path, misc) | ~+2.6 | partly |

## Still open

**The MoDiff arm is still incorrect.** relL2 ~0.49 against 0.0641 for the two-pass path. The
fused kernel's MoDiff recursion is verified correct in isolation
(`scripts/check_modiff_recursion.py`: `a_hat` tracks an explicit reference to 2.2e-3 over 4
steps) and path mixing is eliminated (`scripts/trace_conv_paths.py` shows 0 layers on a
non-blockwise path). The defect is in the `o_hat` wiring in `blockk_gn_fused`. **Only the
BASELINE numbers in this document are valid.**

Two wiring bugs were found and fixed on the way, both worth remembering:

1. `reset_state()` **zeroes** `a_hat`/`o_hat` in place and sets `is_first_step=True`; it does not
   free them. An allocation-shaped first-step check therefore reports "not first" on the first
   step of every sample after the first. **`is_first_step` is the authority.**
2. The updown resize fold (`_prequant_gn_resize_conv_modiff` -> `_conv_from_int8_o_hat`) emits
   PER-TENSOR codes. On the 8 resize `in_conv` layers that mixed conventions with a blockwise
   step-1 write, and it raised no error because `_conv_from_int8_o_hat` had **no `_sim_guard`**.
   Guard added; the fold is now gated off under both `MODIFF_CONV_BLOCKK` and `MODIFF_ACT_BLOCK`.
