# The MoDiff delta clip ratio: what the knob actually does, and what clipping is worth

2026-08-06. Follows `docs/act_bits_2026-08-05/FINDINGS.md`, whose open item 2 was "sweep
`MODIFF_DELTA_CLIP` at A4 and A3 -- zero code, ~20 min, the cheapest quality win available".

Three results, in the order they landed:

1. That item cannot be run as written. **`MODIFF_DELTA_CLIP` stops being a clip below A8**, because
   the code ceiling is a literal in the kernels and only the scale moves. A sweep at A4 would have
   reported a large monotone "win" that is nothing but a higher effective bit width.
2. At A8, where the knob *is* a faithful clip, it is worth **nothing end to end** -- 1-2% at
   r=0.8-0.9, 1 of 3 paired seeds, and monotone losses below r=0.7 (+17.5% at r=0.4). The shipped
   default r=1.0 stands.
3. At A4/A3 a real clip looks worth **26-33% of the accumulated activation error**, at r=0.2-0.3, by
   two independent offline measurements that agree. That is the cheapest quality win still available,
   and collecting it needs the kernel change described at the bottom.

The reason 2 and 3 point opposite ways is not noise, and it corrects a piece of the 2026-08-04
reasoning: over a trajectory this quantizer is solving a *tracking* problem, not a reconstruction
problem, so grid resolution matters more than the ability to represent the single largest delta.

## `MODIFF_DELTA_CLIP` only moves the scale; the code ceiling is a literal

`MODIFF_DELTA_CLIP=r` is implemented as `Q_level = act_q / r` (`int8_optimized.py:472,482,515,1482`),
and `Q_level` enters the kernels in exactly one place: the scale, `scale = Q_level/max(absmax, eps)`.
The code ceiling is a hardcoded constant -- `fmaxf(-127.0f, fminf(127.0f, roundf(...)))` at 7 sites in
`csrc/kernels/quantize/modiff_delta_quantize.cu` and 7 in `csrc/kernels/norm/group_norm_silu.cu`, with
`7.0f` at the int4 ones. So the value the quantizer saturates at is

    clamp_at = (127 / Q_level) * absmax = (r * 127 / act_q) * absmax

At **A8** `act_q` is 127, the two constants cancel, and `clamp_at = r*absmax` -- a faithful clip
ratio. At **A4** `act_q` is 7 and `clamp_at = 18.1*r*absmax`: for every `r` above ~0.055 nothing
saturates, and the knob only makes the grid finer. Sweeping it at A4 measures effective activation
*precision* at a non-integer bit width. The size of that trap is on record in the `shipped` column of
`data/clip_probe.json`: at A4/r=0.1 it reads 0.090 against 0.569 at r=1, a "6x improvement" that is
an A7-resolution grid wearing an A4 label.

## What a real clip is worth per step (`scripts/clip_probe.py`)

Deltas are the input to an eligible conv at DDIM step t minus the same conv's input at step t+1,
captured from an fp16 sampling run (batch 8, DDIM 50) at three points on the trajectory. In this
datapath the conv's input is already post-GroupNorm-SiLU, i.e. the tensor the fused kernel quantizes.
All 70 eligible convs. `clipped` clamps codes at ±Q (what a b-bit quantizer with a clip ratio *is*);
`shipped` clamps at ±127 (what the knob does today). Error is relative to ‖delta‖.

| | best r (aggregate) | relL2 at r=1 → at best r | gain | modal per-layer best r |
|---|---|---|---|---|
| A8, t2→t3 | 0.80 | 0.0433 → 0.0389 | 10.3% | 0.7-0.9 |
| A8, t24→t25 | 0.60 | 0.0799 → 0.0554 | 30.7% | 0.7-0.9 |
| A8, t46→t47 | 0.60 | 0.0507 → 0.0374 | 26.2% | 0.7-0.9 (18/16/17 layers) |
| A4, t2→t3 | 0.30 | 0.5061 → 0.2791 | 44.9% | 0.25-0.4 |
| A4, t24→t25 | 0.20 | 0.6863 → 0.3032 | 55.8% | 0.2-0.4 |
| A4, t46→t47 | 0.25 | 0.5690 → 0.2565 | 54.9% | 0.2-0.4 (16/16/10/12) |
| A3, t24→t25 | 0.10 | 0.8927 → 0.4408 | 50.6% | 0.15-0.25 |
| A3, t46→t47 | 0.15 | 0.8356 → 0.3949 | 52.7% | 0.15-0.25 (15/13/18) |
| A2, all three | 0.10 | 0.97-0.99 → 0.60-0.66 | 33-40% | 0.10-0.15 |

**Read the last column, not the second.** The aggregate is `sqrt(sum_l ‖e_l‖² / sum_l ‖d_l‖²)`, which
is the right thing if you want total error energy but is dominated by the few layers with the largest
‖delta‖. Those layers want a harder clip than the rest, which drags the A8 aggregate optimum to 0.60
while 51 of 70 layers individually prefer 0.7-0.9. The end-to-end sweep below follows the modal
column, and the difference is exactly the size of the disagreement -- so quote the per-layer figure.

The optimum is stable along the trajectory (A4 wants 0.2-0.3 at t2, t24 and t46 alike), so a single
constant captures most of this; no schedule is needed.

## End to end at A8, where the knob is faithful (`scripts/clip_e2e.py`, `clip_e2e_paired.py`)

int8+MoDiff dynamic, `MODIFF_ACT_Q=127`, latent relL2 vs a per-seed fp16 reference, paired over 3
seeds, one warm-up sampling run per arm discarded. Means alone cannot carry this: the r=1.0 row is
0.0596 ± 0.0316, a CV above 50%, several times any predicted effect. So the table reports paired
per-seed wins and the worst single-seed regression.

| K | clip r | 1234 | 20260805 | 777 | mean | vs r=1 | wins | worst seed |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.00 | 0.0407 | 0.0961 | 0.0419 | 0.0596 | 1.000x | — | — |
| 1 | 0.90 | 0.0296 | 0.0988 | 0.0484 | 0.0589 | 0.989x | 1/3 | +15.5% |
| 1 | 0.80 | 0.0295 | 0.0965 | 0.0484 | 0.0582 | 0.976x | 1/3 | +15.8% |
| 1 | 0.70 | 0.0450 | 0.0978 | 0.0468 | 0.0632 | 1.061x | 0/3 | +11.7% |
| 1 | 0.60 | 0.0400 | 0.1010 | 0.0462 | 0.0624 | 1.048x | 1/3 | +10.5% |
| 1 | 0.50 | 0.0428 | 0.1059 | 0.0446 | 0.0644 | 1.082x | 0/3 | +10.2% |
| 1 | 0.40 | 0.0490 | 0.1091 | 0.0411 | 0.0664 | 1.114x | 1/3 | +20.2% |
| 4 | 1.00 | 0.0376 | 0.0954 | 0.0505 | 0.0611 | 1.000x | — | — |
| 4 | 0.90 | 0.0404 | 0.0941 | 0.0491 | 0.0612 | 1.001x | 2/3 | +7.4% |
| 4 | 0.80 | 0.0456 | 0.0986 | 0.0468 | 0.0637 | 1.041x | 1/3 | +21.4% |
| 4 | 0.70 | 0.0468 | 0.0979 | 0.0451 | 0.0633 | 1.035x | 1/3 | +24.6% |
| 4 | 0.60 | 0.0502 | 0.1048 | 0.0460 | 0.0670 | 1.096x | 1/3 | +33.6% |
| 4 | 0.50 | 0.0479 | 0.1064 | 0.0482 | 0.0675 | 1.104x | 1/3 | +27.6% |
| 4 | 0.40 | 0.0463 | 0.1141 | 0.0551 | 0.0718 | 1.175x | 0/3 | +23.3% |

**No change to the default.** The best row is r=0.80 at K=1, a 2.4% mean improvement on 1 of 3 seeds
with a 15.8% regression on another -- indistinguishable from the seed spread. Everything at or below
0.70 loses, monotonically in the mean, and the loss is larger at the shipped K=4 than at K=1, which
is the opposite of what the doc comment at `int8_optimized.py:399` predicts ("pair it with
`MODIFF_DELTA_CLIP` < 1, which ... leaves headroom" for a stale reused scale). A stale scale does
clip on its own; adding a deliberate clip on top compounds rather than compensates.

ms/step is flat across every row (25.2-29.1, no trend) -- the knob changes one float, as designed.

The prediction that failed was the *aggregate* one-step number (26% better at r=0.6). The per-layer
modal optimum (0.7-0.9) and the trajectory measurement below both say "small gain near 0.9, losses
below 0.7", which is what happened. One-step aggregate MSE over-weighted a handful of layers.

## Why per-step MSE is the wrong objective: the accumulator is tracking (`scripts/accum_probe.py`)

MoDiff does not use Q(delta) and discard it -- it accumulates,
`a_hat_t = a_hat_{t-1} + Q(a_t - a_hat_{t-1})`, and the conv consumes a_hat. This iterates that
recursion in fp32 on a captured 50-step fp16 activation trajectory (10 eligible convs spread across
the UNet, warm-up 5 rounds as shipped), and reports error relative to ‖a‖ -- the quantity the conv
actually sees -- rather than relative to ‖delta‖.

| A8 | step 1 | traj mean | final | bias | | A4 | step 1 | traj mean | final | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| r=1.00 | 0.0034 | 0.0059 | 0.0141 | +0.001 | | r=1.00 | 0.0327 | 0.0885 | 0.1572 | +0.001 |
| r=0.90 | 0.0031 | **0.0057** | 0.0131 | +0.000 | | r=0.60 | **0.0271** | 0.0735 | 0.1219 | +0.002 |
| r=0.80 | **0.0030** | 0.0060 | 0.0133 | +0.000 | | r=0.40 | 0.0357 | 0.0673 | 0.1033 | +0.002 |
| r=0.60 | 0.0051 | 0.0081 | 0.0181 | -0.000 | | r=0.30 | 0.0522 | 0.0654 | 0.0988 | +0.004 |
| r=0.40 | 0.0091 | 0.0142 | 0.0341 | -0.001 | | r=0.25 | 0.0607 | **0.0651** | 0.1006 | +0.003 |
| r=0.20 | 0.0216 | 0.0328 | 0.0784 | -0.002 | | r=0.20 | 0.0697 | 0.0664 | 0.1068 | +0.003 |

A3 behaves like A4, more so: best trajectory r=0.20 at 0.668x of r=1.0 (0.1779 → 0.1188).

**The hypothesis this was built to test was wrong.** The guess was that clipping error is biased
inward, and that an accumulator integrates a systematic term instead of cancelling it. The bias
column refutes it: |mean signed error / rms error| never exceeds 0.008 at any ratio or precision,
and it does not trend with r.

**What the data says instead.** At A4 the trajectory optimum (r=0.25) is *harder* than the one-step
optimum (r=0.60), and it wins while being 2.2x worse on step 1 -- 0.0651 against 0.0735 on the
trajectory mean, 0.1006 against 0.1219 at the end. Clipping is not unrecoverable here: what saturates
this step stays in `a_t - a_hat_t` and gets quantized again next step, so a clip defers error rather
than destroying it. What does *not* get another chance is the grid step, `r*absmax/Q`, which is the
floor on tracking error at every step. Trading representable range for resolution is therefore the
right trade once the loop is tracking, and the r=1.0 row shows the cost of not making it: it has the
best step-1 error at A4 and the worst trajectory error, growing 4.8x from step 1 to step 50 where
r=0.25 grows 1.7x.

This is the piece of the 2026-08-04 reasoning that needs amending. That measurement found the static
grid clipping on 49 of 70 layers and concluded "clipping is unrecoverable error, and MoDiff's feedback
term then carries it forward"; the direction was right -- the static grid is far too coarse *and* it
clips -- but the mechanism was not. MoDiff's feedback term *recovers* clipping error. The reason a
per-call absmax beat the static grid is resolution, not clip avoidance, which is also why going
further in the same direction (r < 1, finer still) keeps helping at A4/A3 and not at A8: at Q=127 the
grid is already fine enough that the slew-rate cost of a hard clip dominates any resolution gain.

Approximations that bound both offline sections: deltas come from an fp16 trajectory, and the
subtrahend is the true previous activation rather than the dequantized one for `clip_probe`
(`accum_probe` uses the real recursion, so only the trajectory is fp16). Both perturb the delta by
about one quantization error. Fine for ranking clip ratios; not a substitute for the end-to-end
number, which is why A8 was measured both ways -- and the two agree.

## Code changes

None. All three scripts are measurement only; no kernel, harness or default was touched.

## Open, in the order I would take them

1. **A real code ceiling in the kernels**, which is what collecting the A4/A3 gain requires.
   `Q_level` is already a threaded `float`, so this is a sibling `float code_ceiling` (default -1 →
   keep the literal, so every existing call stays bit-identical) clamped instead of the constant, at
   7+7 int8 sites and the int4 ones, plus `csrc/modiff_kernels_api.h`, `csrc/pybind.cpp`, and the
   Python callers passing `act_q`. Rebuild required (~minutes with ninja). Then the A4/A3 sweep
   becomes a real measurement, with a specific prediction to check it against: r≈0.25 at A4 should
   cut accumulated conv activation error ~26%, and at A8 the same machinery should reproduce the
   ~2% null already measured through the existing knob.
   Temper the expected end-to-end size by the A8 calibration: a 2.6% predicted trajectory gain there
   showed up as 1-2% of latent relL2, so the conv delta path carries a fraction of the total. A 26%
   trajectory gain at A4 is worth having but should not be assumed to move W8A4's 0.155 by 26%.
2. Once the ceiling exists, `MODIFF_ACT_Q` at A4 stops being a pure quality instrument and the
   `act_bit_sweep.py` A8..A2 table should be re-run: every row below A8 in
   `docs/act_bits_2026-08-05` was measured with a quantizer that could not saturate, which is
   generous to both arms but not equally.
3. The FID items inherited from `docs/act_bits_2026-08-05` -- **note the sample stores were lost**
   when this container was reset: `fid/fp16` holds 1965 of its 10k samples and `fid/int8_baseline`,
   `fid/int8_modiff`, `fid/int4_*` and `fid/real` are empty, so those items now cost an fp16
   regeneration too. `/workspace/lsun_dl` is still empty, so FID-vs-real still needs the LSUN LMDB.
4. Per-row / per-token dynamic activation scales, unchanged from the previous list: the foldable
   analogue of the paper's channel-wise, and the only granularity improvement with a real datapath
   on this hardware.
