# The MoDiff delta clip ratio: what the knob actually does, and what clipping is worth

2026-08-06. Follows `docs/act_bits_2026-08-05/FINDINGS.md`, whose open item 2 was "sweep
`MODIFF_DELTA_CLIP` at A4 and A3 -- zero code, ~20 min, the cheapest quality win available".

Four results, in the order they landed:

1. That item cannot be run as written. **`MODIFF_DELTA_CLIP` stops being a clip below A8**, because
   the code ceiling is a literal in the kernels and only the scale moves. A sweep at A4 would have
   reported a large monotone "win" that is nothing but a higher effective bit width.
2. At A8, where the knob *is* a faithful clip, it is worth **nothing end to end** -- 1-2% at
   r=0.8-0.9, 1 of 3 paired seeds, and monotone losses below r=0.7 (+17.5% at r=0.4). The shipped
   default r=1.0 stands.
3. At A4/A3 a real clip looks worth **26-33% of the accumulated activation error**, at r=0.2-0.3, by
   two independent offline measurements that agree. Collecting it needs a code ceiling in the kernels.
4. With that ceiling landed, the measured end-to-end win is **larger than predicted: A4 halves
   (0.1758 → 0.0861 at r=0.40) and A3 drops 2.6x (0.3934 → 0.1548 at r=0.25)**, 3 of 3 seeds on every
   clip row, while A8 stays flat-to-worse exactly as in (2). A real clip buys back roughly the bit
   that was lost. The ceiling is also a defect fix: on scale-reuse steps the old literal let an "A4"
   layer emit codes far outside ±7, so the previously published A4/A3 rows were flattered.

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

## The code ceiling, and what the clip is worth at A4/A3 (`scripts/clip_e2e_bits.py`)

With `code_ceiling` threaded through the delta-quantize kernels (see Code changes), a b-bit delta
quantizer saturates at Q_b and the A4/A3 rows become measurable. Same protocol: batch 8, DDIM 50,
paired over 3 seeds against a per-seed fp16 reference, one warm-up run per arm discarded.

**Mean relL2, and the ratio to that row's own r=1.0.** Every A4 and A3 clip row below wins on 3 of 3
seeds; every A8 clip row loses in the mean. Per-seed detail is in `data/clip_e2e_bits.json`.

| K | bits | r=1.0 | r=0.60 | r=0.40 | r=0.30 | r=0.25 | r=0.20 |
|---|---|---|---|---|---|---|---|
| 1 | A8 | 0.0619 | 0.0631 (1.02x) | 0.0667 (1.08x) | 0.0739 (1.19x) | 0.0855 (1.38x) | 0.1094 (1.77x) |
| 1 | A4 | 0.1516 | 0.0896 (0.59x) | **0.0765 (0.50x)** | 0.0796 (0.53x) | 0.0906 (0.60x) | 0.1129 (0.74x) |
| 1 | A3 | 0.4233 | 0.2372 (0.56x) | 0.1665 (0.39x) | 0.1475 (0.35x) | **0.1452 (0.34x)** | 0.1450 (0.34x) |
| 4 | A8 | 0.0615 | 0.0641 (1.04x) | 0.0734 (1.19x) | 0.0794 (1.29x) | 0.0951 (1.55x) | 0.1237 (2.01x) |
| 4 | A4 | 0.1758 | 0.1013 (0.58x) | **0.0861 (0.49x)** | 0.0908 (0.52x) | 0.1002 (0.57x) | 0.1272 (0.72x) |
| 4 | A3 | 0.3934 | 0.2753 (0.70x) | 0.1867 (0.47x) | 0.1585 (0.40x) | **0.1548 (0.39x)** | 0.1587 (0.40x) |

The offline probes predicted this and understated it. `accum_probe.py` put the trajectory optimum at
r≈0.25/A4 and r≈0.20/A3, worth 26%/33% of accumulated conv activation error; the end-to-end optima are
r≈0.40 at A4 and r≈0.20-0.25 at A3, worth **50% and 66% of latent relL2**. Note the direction of the
miss: at A8 a predicted 2.6% showed up as 1-2% (attenuated, as expected from the conv delta path being
one contributor among several), while at A4/A3 a predicted 26-33% shows up as 50-66%. The clip is
doing more than reducing that one error term -- plausibly because a_hat feeds the next step's delta, so
a better-tracked a_hat also shrinks what the next quantizer has to represent, but that is a hypothesis
this data does not test.

The per-seed spread is no longer a problem the way it was at A8: at A4/K=1 the worst single seed still
improves 36.2% at r=0.40, and at A3/K=1 the worst improves 60.4% at r=0.30. These are effects several
times the spread, not inside it.

**Roughly one bit, for free.** Old A4 without a clip was 0.1553 (K=4); new A3 *with* a clip is 0.1548.
Old A5 was 0.0768; new A4 with a clip is 0.0861. So a real clip buys back approximately the bit that
was lost -- which is what the paper's "up to 3 bits" claim needs, and this implementation could not
previously deliver because its quantizer could not saturate.

**No default change.** The optimum is precision-dependent (1.0 at A8, 0.40 at A4, 0.25 at A3) and
`MODIFF_DELTA_CLIP` is one global knob, so a single new default would regress the shipped W8A8 mode --
where r=1.0 is the best row at both K, and 0.20 costs 1.8-2.0x. A precision-dependent default (keyed
off `act_q`) is the obvious follow-up, and it wants its own measurement: more seeds, and the W8A4 FID
question, before a default that changes shipping behaviour.

### The ceiling is also a defect fix, and here is the control that shows it

On a `MODIFF_DELTA_REFRESH=K>1` reuse step the scale is up to K-1 steps old, so the delta can outgrow
it. Those codes are supposed to saturate at Q_b; clamped at 127 instead, an "A4" layer could emit a
code of 100. `scripts/verify_ceiling.py` demonstrates it directly at the kernel level: at A4 with
r=0.25 the old literal produces `max|code|` of 27 (step1) and 31 (GN-fused), and with the ceiling both
saturate at exactly 7, with 31% and 19% of codes at the ceiling.

So the previously published A8..A2 table was measured with a quantizer that exceeded its own bit width
on 3 of every 4 steps. Per-seed against the committed pre-ceiling JSONs, at r=1.0:

| | seed 1234 | 20260805 | 777 | |
|---|---|---|---|---|
| A8, K=1 | 0.0416 → 0.0437 | 0.0974 → 0.0954 | 0.0451 → 0.0467 | mixed sign, within the floor below |
| A4, K=1 | 0.1560 → 0.1549 | 0.1874 → 0.1833 | 0.1153 → 0.1166 | unchanged (≤0.4%) |
| A3, K=1 | 0.4621 → 0.4599 | 0.4176 → 0.4141 | 0.4109 → 0.3959 | unchanged (≤3.6%) |
| A8, K=4 | 0.0387 → 0.0385 | 0.0958 → 0.0944 | 0.0441 → 0.0515 | 2 flat, 1 unexplained (+17%) |
| A4, K=4 | 0.1572 → **0.1956** | 0.1691 → **0.1870** | 0.1397 → **0.1448** | worse 3/3 (+3.7..+24%) |
| A3, K=4 | 0.3923 → **0.4105** | 0.3623 → **0.3992** | 0.3239 → **0.3705** | worse 3/3 (+4.6..+14%) |

The **A4/A3 rows are the predicted pattern**, and they are the reason to believe the change does what
it claims rather than something else: unchanged at K=1, where every step refreshes its own absmax so
no code could exceed Q_b and the ceiling is a no-op; worse on every seed at K=4, where reuse steps now
saturate as a b-bit quantizer must. A K=1 row moving would have meant the parameter was reaching calls
it should not.

The corrected rows are the honest baseline the clip is measured against, and the clip more than pays
for the correction: A4/K=4 goes 0.1553 (flattered) → 0.1758 (honest) → 0.0861 (clipped).

**The A8 rows carry a caveat I could not resolve.** At A8 `act_q` is 127, identical to the old
literal, and `verify_ceiling.py` proves the kernels emit byte-identical codes there -- so A8 cannot
move for any reason internal to this change. Most seeds agree. But A8/K=4/seed 777 reads 0.0441 before
and 0.0515 after, and a second independent post-ceiling process reproduces 0.0522, so it is not
one-off scatter either. What it is *not* is the ceiling. The remaining difference between the two
measurements is the script: `act_bit_sweep.py` runs a baseline arm between the fp16 reference and the
MoDiff arm, in one process, so its MoDiff arm sees different accumulated GPU state (and its own fp16
references) than `clip_e2e_bits.py`'s does. That is the likeliest cause and it is untested here;
attributing it would need the two harnesses run against each other at A8, which nothing above depends
on. It does mean cross-script relL2 comparisons in this project should not be trusted below ~20%.

**Same-script cross-process floor** (`data/a8_control_repeat.json`, an independent rerun of the A8
r=1.0 rows): per-seed differences of −9.5%, −0.3%, −4.0% at K=1 and −0.4%, −1.8%, +1.4% at K=4; means
0.0619 vs 0.0598 and 0.0615 vs 0.0611. So ~10% per seed, ~3% in the mean, with everything held fixed.
That is the floor the A4/A3 clip effects (50-66%, 3/3 seeds, worst seed 15-61%) clear by a wide
margin, and it is why nothing smaller than a 3/3-seed effect is called real anywhere in this document.

## Re-running A8..A2 with the ceiling: what the flattered rows were worth (`scripts/act_bits_ceiling_diff.py`)

`docs/act_bits_2026-08-05/scripts/act_bit_sweep.py`, unmodified, re-run at both refresh settings
(`data/act_bit_sweep_ceiling_k{4,1}.json`) and diffed per seed against its own committed pre-ceiling
output. `MODIFF_DELTA_CLIP` is 1.0 throughout, so this isolates the defect fix from the clip.

**MoDiff arm at K=4 (the shipped default, and the configuration the published table used):**

| bits | was | now | | 3 seeds, old → new |
|---|---|---|---|---|
| A8 | 0.0595 | 0.0607 | 1.019x, 1/3 worse | 0.0387→0.0377  0.0958→0.0927  0.0441→0.0517 |
| A7 | 0.0588 | 0.0597 | 1.015x, 2/3 | 0.0368→0.0372  0.0972→0.0957  0.0425→0.0461 |
| A6 | 0.0590 | 0.0596 | 1.009x, 2/3 | 0.0276→0.0279  0.0991→0.1007  0.0504→0.0501 |
| A5 | 0.0768 | 0.0804 | 1.047x, 1/3 | 0.0597→0.0572  0.1036→0.1233  0.0670→0.0605 |
| A4 | 0.1553 | **0.1825** | 1.175x, **3/3** | 0.1572→0.1912  0.1691→0.2156  0.1397→0.1406 |
| A3 | 0.3595 | **0.3926** | 1.092x, **3/3** | 0.3923→0.4127  0.3623→0.3923  0.3239→0.3729 |
| A2 | 0.6058 | **0.6539** | 1.079x, **3/3** | 0.6516→0.6916  0.5800→0.6302  0.5857→0.6399 |

**A4, A3 and A2 were flattered by 8-18%; A5 and above were not.** Only those three move on all three
seeds; A8-A5 are mixed-sign scatter inside the ~10% per-seed floor. That the effect appears exactly
where the delta path dominates the total error, and only below A5, is worth noting but is not something
this data explains — the fraction of codes that outgrow a stale scale does not obviously depend on Q_b.

The reported gain shrinks with it: A4 5.19x → 4.42x, A3 2.59x → 2.37x, A2 1.54x → 1.43x. Every
qualitative conclusion in that report survives — MoDiff still flat A8→A5, A4 still beats the W8A8
baseline (0.183 vs 0.256), still helps at A2 — with A4's margin narrower than first quoted.

**Both controls pass, which is what makes the above readable as the fix rather than as drift.**

* *The baseline arm does not move, at any precision or either K*: ratios 0.999-1.011x with per-seed
  differences in the fourth decimal and no consistent sign. It goes through `scale_quantize_int8` /
  `dynamic_quantize_int8_fprop`, which this change never touched, and `_delta_code_ceiling` is -1 in
  static mode. A baseline row moving would have meant the parameter was reaching calls it should not.
* *The MoDiff arm does not move at K=1*: 0.955-1.005x, 0/3 to 2/3 worse, mixed sign, with A4 reading
  4.5% *better* (0/3 worse) — noise in the favourable direction. At K=1 every step measures its own
  absmax, so no code can exceed Q_b and the ceiling is provably a no-op. This is the row that
  distinguishes "the ceiling bit on reuse steps" from "something else changed".

`docs/act_bits_2026-08-05/FINDINGS.md` has been amended in place with the corrected table, since that
is the table a reader will use; its original values are kept in a labelled column.

One cross-script note, consistent with the floor discussion above: this sweep puts A4/K=4/r=1.0 at
0.1825 where `clip_e2e_bits.py` put it at 0.1758, a 3.8% disagreement between two harnesses measuring
the same configuration with the same code. Well inside the ~20% cross-script caveat, and a reminder to
compare within one script.

## Code changes

Measurement scripts (`clip_probe.py`, `clip_e2e.py`, `clip_e2e_paired.py`, `accum_probe.py`,
`act_bits_ceiling_diff.py`) touch nothing. The ceiling is a kernel change:

| file | change |
|---|---|
| `csrc/common.cuh` | new `clamp_code(v, ceiling, native)`: clamps at `ceiling` when it is > 0, else at the literal the call site used before the parameter existed. One place, so the fallback rule cannot drift between kernels |
| `csrc/kernels/quantize/modiff_delta_quantize.cu` | `float code_ceiling` on the five int8 static delta kernels (fp32 cache, fp16 cache, fp16-cache vec2, and the two SiLU-fused variants) and their 9 launch sites; `step1_static_quantize_fprop` and `..._silu` take it as a trailing argument |
| `csrc/kernels/norm/group_norm_silu.cu` | same for `gn_apply_delta_quantize_flat_kernel` and `..._vec2_kernel` (4 launch sites) and `group_norm_silu_delta_quantize_nhwc`. Note it is NOT `Q_level`, which that kernel already has for a different purpose: `Q_level/absmax` is the scale it publishes for a later step, `code_ceiling` is where this step's codes saturate |
| `csrc/pybind.cpp` | a second overload per entry point rather than `py::arg` defaults. pybind11 does not inherit C++ default arguments, this file annotates no argument names anywhere, and ~20 call sites across `integration/`, `analysis_*/` and 8 archived `docs/*/scripts` pass the short form. Short form registered first, so those are untouched |
| `integration/kernels/int8_optimized.py` | new `_delta_code_ceiling`: `act_q` in dynamic mode, `-1` in static mode. Static keeps the 127 ceiling deliberately -- there the scale is the calibrated Q_b/range rather than this call's absmax, and the 127 ceiling is what lets a delta above the calibrated range keep resolution instead of saturating. That behaviour is load-bearing for the published baseline comparison (it is the asymmetry in `MODIFF_ACT_Q`'s comment, and it favours the baseline arm), so changing it belongs to its own measurement |
| `integration/kernels/int8_optimized.py` | `_delta_gn_dynamic_args` returns the ceiling as an 8th element; the three MoDiff `step1_static_quantize_fprop[_silu]` call sites pass it. The 4th call site (`forward_from_int8`, a zeroed a_hat and the static grid) is the baseline quantize path and deliberately keeps the literal |
| `docs/delta_clip_2026-08-06/scripts/verify_ceiling.py` | the kernel-level test: short overload == `-1`, `127` == the literal at the A8 scale, and `Q_b` saturates where the literal does not. Both entry points, 6 checks each |

Not migrated, and deliberately so: the int4 kernels (`7.0f` literals) and the updown resize path. At
W4A4 the ceiling is 7 and `act_q` is 7, so the literal is already correct at r=1.0; a clip ratio there
needs the same treatment, and W4A4 is a speed configuration per `docs/act_bits_2026-08-05`, so it can
wait for a reason to spend the rebuild.

## Open, in the order I would take them

1. ~~Re-run the A8..A2 sweep in `docs/act_bits_2026-08-05`.~~ **Done** — see the section above. A4,
   A3 and A2 were flattered by 8-18% (0.1553 → 0.1825, 0.3595 → 0.3926, 0.6058 → 0.6539); A5 and
   above were inside the noise floor; both controls passed. That report's table is amended.
2. **A precision-dependent `MODIFF_DELTA_CLIP` default**, which is how the A4/A3 win actually gets
   collected rather than left on a knob. Measured optima: 1.0 at A8, 0.40 at A4, 0.25 at A3. Wants
   more than 3 seeds and a look at whether the optimum is batch- or step-count-dependent before it
   changes shipping behaviour, since one global default would regress W8A8 (0.20 costs 2.0x there).
3. **FID at W8A4+MoDiff with r=0.40**, now that relL2 there (0.086) is in the same range as the A5/A6
   rows that were considered free. This is the row where a quality claim would actually land, and
   relL2 is a poor proxy for perceptual quality at these levels. Note the sample stores were lost
   when this container was reset: `fid/fp16` holds 1965 of its 10k samples and `fid/int8_baseline`,
   `fid/int8_modiff`, `fid/int4_*` and `fid/real` are empty, so this costs an fp16 regeneration too.
   `/workspace/lsun_dl` is still empty, so FID-vs-real still needs the LSUN LMDB re-downloaded.
4. **Why the end-to-end gain exceeds the offline prediction** (50-66% measured against 26-33%
   predicted), when at A8 it was attenuated instead. The hypothesis in that section -- a
   better-tracked a_hat shrinks the next step's delta, so the clip compounds along the trajectory --
   is untested, and `accum_probe.py` is teacher-forced on an fp16 trajectory and so cannot see it.
   A quantized-trajectory capture would.
5. **The int4 kernels and the updown resize path**, if a clip at W4A4 is ever wanted: same change,
   same 4 sites plus the resize twin. Not done here because at W4A4 the ceiling and `act_q` are both
   7, so r=1.0 is already correct, and W4A4 is a speed configuration.
6. Per-row / per-token dynamic activation scales, unchanged from the previous list: the foldable
   analogue of the paper's channel-wise, and the only granularity improvement with a real datapath
   on this hardware.
