# Activation precision at fixed W8: A8 → A2, MoDiff on and off, against the paper

**2026-08-05 · A40 · LSUN-Churches LDM-8, real checkpoint · DDIM 50 · batch 8 · 3 seeds, paired ·
latent relL2 vs fp16**

The first sweep of activation precision in this project, and the first measurement of the
configuration the paper actually claims. Every shipped mode pairs the two precisions (W8A8, W4A4), so
every earlier quality number confounds weight error with activation error — and MoDiff only addresses
the second. W4A4+MoDiff at FID 200 (`docs/fid_2026-08-05`) is the visible consequence: int4 *weights*
dominate there, so that row cannot test the paper's claim either way.

Data: `data/act_bit_sweep.json` (post-fix, the headline table), `data/act_bit_sweep_prewarmupfix.json`
and `data/act_bit_sweep_paper_anchor_prewarmupfix.json` (kept for the before/after),
`data/verify_vs_old_w8a4.json`.

## The t=T warm-up was broken, and that was the whole story below A6

MoDiff quantizes a difference against â, and â is seeded by the t=T warm-up. The paper's warm-up
(Appendix D.5) is *"repeatedly inputting a_T. This process converges to the full-precision activation
due to the contraction of the quantization error ... approximately 4 to 5 steps are sufficient to
reduce the quantization error to a negligible level on CIFAR-10 using 4-bit precision."*

The contraction needs each round to quantize the shrunken residual on a grid matched to it. This tree
ran `warmup_steps = 3` but passed the **static activation grid** to every round on the calibrated path
that ships. After round 1 the residual is under half an LSB of that grid, so round 2 rounds it to zero:
**the loop was a no-op.** Measured on real activations, |â − x| / |x| per round
(`scripts/probe_warmup.py`):

| precision | scheme | r1 | r2 | r3 | r4 | r5 |
|---|---|---:|---:|---:|---:|---:|
| A8 | static (was shipped) | 0.0197 | 0.0197 | 0.0197 | 0.0197 | 0.0197 |
| A8 | dynamic (the paper's) | 0.0197 | 0.00008 | 0.00000 | 0.00000 | 0.00000 |
| **A4** | **static (was shipped)** | **0.4006** | **0.4006** | **0.4006** | **0.4006** | **0.4006** |
| **A4** | dynamic (the paper's) | 0.4006 | 0.0263 | 0.0018 | 0.00013 | **0.00001** |

So at A4 the anchor every later step is measured against carried 40% relative error, where the paper's
carries 1e-5. Fixed in both `int8_optimized.py` and `int4_optimized.py`: the calibrated path now uses a
per-round dynamic scale, and `warmup_steps` defaults to 5 (`MODIFF_WARMUP_STEPS`). Cost is two extra
quantize+conv per layer on one step in fifty.

`MODIFF_DELTA_REFRESH` was also changed 4 → 1 on fidelity grounds — the paper's dynamic quantizer
recomputes the scale every step — and then **measured, and reverted**. See the section below; the
sweep numbers here are all at K=4, so every improvement in them is the warm-up fix alone.

## Result

Q is the symmetric code ceiling 2^(b−1)−1, applied to **every** conv activation site including the t=T
warm-up — the paper's protocol, now that the warm-up converges. `MODIFF_DELTA_REFRESH=4`
throughout. Baseline is MoDiff off with the
calibrated per-tensor grid rescaled to b bits; it is unaffected by either fix and reproduces its
pre-fix values to ≤0.0015, which is the cross-run control.

| A bits | levels | baseline (PTQ) | **MoDiff** | gain | MoDiff before the fix |
|---|---:|---:|---:|---:|---:|
| A8 | 255 | 0.2563 ± 0.017 | **0.0595 ± 0.032** | 4.30× | 0.0627 |
| A7 | 127 | 0.2574 ± 0.049 | **0.0588 ± 0.033** | 4.38× | 0.0669 |
| A6 | 63 | 0.3170 ± 0.042 | **0.0590 ± 0.037** | 5.37× | 0.0795 |
| A5 | 31 | 0.7361 ± 0.020 | **0.0768 ± 0.024** | 9.59× | 0.1563 |
| A4 | 15 | 0.8066 ± 0.038 | **0.1553 ± 0.015** | 5.19× | 0.3367 |
| A3 | 7 | 0.9317 ± 0.033 | **0.3595 ± 0.034** | 2.59× | 0.8206 |
| A2 | 3 | 0.9308 ± 0.057 | **0.6058 ± 0.040** | 1.54× | 1.3308 |

Read against the relL2 → FID anchors from `docs/fid_2026-08-05` (0.039 → FID 7.80 = parity with fp16;
0.238 → 16.4; 0.456 → 200; 0.784 → 278):

* **MoDiff is flat from A8 to A6** — 0.0595, 0.0588, 0.0590 — and still nearly flat at A5 (0.0768).
  Three to four bits of activation precision removed for nothing measurable.
* **A4 costs something** (0.155, between the FID-7.8 and FID-16.4 anchors) and **A3 is where it breaks**
  (0.360, approaching the 0.456 → FID 200 anchor).
* **The baseline collapses two to three bits earlier**: FID ≈ 16 already at A8 (0.256), degrading at A6
  and gone by A5 (0.736, next to the 0.784 → FID 278 anchor).
* **MoDiff at A4 (0.155) beats the W8A8 baseline (0.256)** — the paper's claim in substance, 4-bit
  activations with modulation beating 8-bit PTQ without it, reproduced at per-tensor granularity.
* **MoDiff helps at every precision, including A2** (1.54×). An earlier revision of this document
  reported MoDiff being *worse* than the baseline at A2 and explained it via Theorem 4.4's error bound
  failing at three levels. That was entirely the broken warm-up — a 2-bit anchor with a no-op warm-up
  is noise, and the feedback term propagated it. The explanation is withdrawn.

ms/step is flat across every row (baseline 12.6–14.8, MoDiff 17.6–20.3, fp16 16.7) — the control
confirming this is a quality instrument only. A low A_b costs nothing and saves nothing here:
activations keep their int8 container and the GEMM stays W8A8. A real 4-bit activation datapath needs
int4 tensor cores, which take both operands at 4 bits — no mainstream ISA has a mixed s8×s4 MMA, so
W8A4 is not a speed configuration on any hardware, only a quality one.

## Fidelity to the paper's quantizer lost to the measurement

`MODIFF_DELTA_REFRESH=K` reuses the dynamic delta scale for K steps. The paper has no such
approximation, so on 2026-08-06 the default went 4 → 1, and then the sweep was re-run at K=1 to check.
Paired over the same 3 seeds, MoDiff arm, warm-up fix in place (`data/act_bit_sweep_refresh1.json`):

| A bits | K=4 | K=1 | seeds where K=1 wins |
|---|---:|---:|---:|
| A8 | **0.0595** | 0.0613 | 0/3 |
| A7 | **0.0588** | 0.0601 | 1/3 |
| A6 | **0.0590** | 0.0630 | 1/3 |
| A5 | 0.0768 | 0.0688 | 2/3 |
| A4 | 0.1553 | 0.1529 | 2/3 |
| A3 | **0.3595** | 0.4302 | 0/3, by 0.055–0.087 |
| A2 | **0.6058** | 0.7063 | 0/3, by 0.079–0.121 |

K=1 never wins, is a wash at A5/A4, and loses badly at 3 and 2 bits — on top of recomputing the absmax
every step instead of every fourth. The mechanism is that a per-step absmax sets the grid from *this*
step's single worst outlier, while holding it for K steps smooths that estimate; with 3–7 levels one
outlier eats most of the grid. **Default reverted to 4**, with the paired data recorded at the knob so
the next fidelity argument has to beat it.

This is the second parameter today changed on a plausible argument and then refuted by measuring it
(the first was AdaRound's learning rate, where "a gate only needs ~2 units of travel" picked 1e-3 and
1e-2 measured almost twice as good).

## Speed, at the final configuration

`e2e_three_mode_bench.py`, 3 warm-up samples + 5 timed repeats, DDIM 200, warm-up fix in place and
`MODIFF_DELTA_REFRESH=4` (`data/e2e_postfix_b128.json`, `data/e2e_postfix_b8.json`):

| batch | fp16 | INT8 baseline | INT8 + MoDiff | MoDiff vs fp16 | MoDiff vs baseline |
|---|---:|---:|---:|---:|---:|
| **128** (CV ≤ 0.33%) | 106.20 | 74.05 | **78.06** | **1.360×** | **0.949×** |
| 8 (CV 3.6–6.9%, NOISY) | 18.09 | 16.76 | 19.82 | 0.912× | 0.845× |

At batch 128 this reproduces the pre-fix report's ratios exactly (1.359× / 0.946× on 2026-08-04), so
the warm-up fix costs nothing measurable — it adds two quantize+conv per layer on 1 step in 200. All
three modes are ~3.4% slower in absolute terms than that report, fp16 included, and fp16 is untouched
by every change since; that is cross-session drift, which this project has documented before, and it
is why the ratios rather than the absolutes are the thing to quote.

**At batch 8 the whole stack barely pays.** INT8 is only 1.079× fp16, and MoDiff is *slower than fp16*
(0.912×). The three extra launches per layer per step are pure latency at that size. Treat the batch-8
row as directional: at CV ~7% it does not support fine comparisons, including against the 0.691× the
batch sweep reported on 2026-08-04.

## What this means for the earlier numbers, and for docs/fid_2026-08-05

Three readings of "W8A4+MoDiff" have now been measured, and they differ only in the anchor:

| configuration | relL2 |
|---|---:|
| broken warm-up, A4 everywhere | 0.337–0.358 |
| broken warm-up, t=T left on the A8 grid (the old FID report's row) | 0.153–0.163 |
| **fixed warm-up, A4 everywhere — the paper's protocol** | **0.1553 ± 0.015** |

The old FID report's 0.127 row (single seed, bottom of a 0.130–0.196 spread) landed close to the right
answer for the wrong reason: leaving t=T on the A8 grid happened to approximate what a converged
warm-up would have given. Its conclusion — **W8A4+MoDiff beats the W8A8 baseline** — holds under the
correct protocol. The "paper anchor vs strict" framing that an earlier revision of this document
introduced was a workaround for the bug and is gone; there is one protocol, the paper's.

## Against the paper

arXiv 2506.22463, Table 2 — our exact model: LSUN-Church, LDM-8, W8, 200 steps, FID over 50k images.
`Q-Diff` is Q-Diffusion (static, MSE-calibrated); `LCQ` is dynamic **channel-wise** min-max (BRECQ).

| A bits | Q-Diff | Q-Diff+MoDiff | LCQ | LCQ+MoDiff |
|---|---:|---:|---:|---:|
| 32 (fp act) | 4.03 | — | — | — |
| 8 | 4.24 | 3.85 | 4.02 | 3.99 |
| 6 | 55.13 | 5.43 | 4.50 | 3.89 |
| 4 | 355.85 | 3.97 | 198.37 | 34.02 |
| 3 | 367.51 | 5.40 | 341.62 | 12.05 |

Their per-**tensor** variant — the fair analogue of our kernels — is `LTQ`, reported on CIFAR-10 only
(Table 14, W8): A8 4.21, A6 4.00, A4 28.19, A2 186.04, with the explanation that *"the minimum
activation bit-width achievable with LTQ is higher than that of LCQ"*.

**What matches.** The shape is theirs: MoDiff flat while the baseline collapses, the baseline dying
between A6 and A5, MoDiff surviving several bits further, and per-tensor A4 being degraded-but-usable
rather than clean (their LTQ+MoDiff 28.19 against a 4.24 reference; our 0.153 lands between the
FID-7.8 and FID-16.4 anchors). Our A4 result is in line with their per-tensor result, not with their
per-channel one.

**What does not, and why.** Their best A4 row is Q-Diff+MoDiff at FID 3.97 — essentially free — and
we do not have that. Two differences, in the order I would attack them:

1. **The delta quantizer's scale rule.** Ours is dynamic absmax, `Q/max|Δ|` — the coarsest
   non-clipping choice. Theirs is *calibrated*: Appendix B says they reconstruct the calibration set
   for Q-Diff+MoDiff, "storing the inputs and outputs of MoDiff rather than the raw activations", and
   fit the step size by MSE. At 255 levels not clipping is optimal (this project's clip sweep is
   monotone at W8A8), but at 15 levels trading a little clipping for a finer grid is the classic
   bias–variance win, and the old W4A4 clip sweep already showed the optimum moving to ratio ≈ 0.35.
   **This is testable with zero code** via `MODIFF_ACT_Q=7` + a `MODIFF_DELTA_CLIP` sweep.
2. **Granularity.** Their headline A4/A3-on-Churches numbers rest on per-channel scales. That axis is
   the GEMM's reduction axis, so a per-channel activation scale does not fold into the epilogue —
   it would need per-step weight requantization or a scale multiply inside the K-loop, which is why
   the paper calls tensor-wise "more hardware-friendly" and why Remark 5.1 reports BOPs rather than
   time. The foldable analogue is **per-row / per-token** (per pixel, along M), and there is
   precedent for it in this tree already (`modiff_kernels_api.h:511`, the per-token attention path).

Caveat on the comparison throughout: they report FID over 50k images at 200 steps; this is latent
relL2 at 50 steps, and relL2 is badly nonlinear against FID. It supports "which is better, and
roughly where the cliff is", not "our A4 equals their FID 28".

## The weight floor: how much of W4A4 is reachable from the activation side at all

`scripts/weight_ceiling.py` fake-quantizes weights with the shipped scale rules and leaves
activations in fp16, so each row is the error **no** activation-side work — MoDiff, clip tuning,
per-token scales — can get below. Same protocol: batch 8, DDIM 50, 3 seeds, paired.

| weights quantized | layers | relL2 vs fp16 |
|---|---|---:|
| int8 convs (control) | 70 | 0.0229 ± 0.0019 |
| **int4 convs, MSE clip (shipped)** | 70 | **0.2443 ± 0.0064** |
| int4 convs, absmax (pre-08-05 rule) | 70 | 0.2184 ± 0.0070 |
| int4 convs MSE + linears absmax (shipped linear rule) | 70 + 79 | 0.2603 ± 0.0055 |
| int4 convs MSE + linears MSE | 70 + 79 | 0.2888 ± 0.0099 |

Caveat on the linear rows: the predicate used here (`wxax_linear._eligible(bits=4)`) matches 79
`nn.Linear` modules while the runtime gate quantizes 42, so those two rows are an upper bound on the
linear contribution, not the shipped set. The conv rows match the shipped set exactly.

**"W4A4 is weight-limited" was too strong.** Against W4A4+MoDiff's measured 0.44–0.47, a 0.244 weight
floor is about half the error in RMS terms and roughly a quarter of it in squared terms — the
activation side still contributes more. What *is* true, and is the useful statement:

> Even with a perfect activation datapath, int4 weights alone cost relL2 0.244, which sits on the
> 0.238 → FID 16.4 anchor. **W4A4 cannot reach fp16 parity by any amount of activation-side work.**
> Better weight quantization is a necessary condition, not a sufficient one.

For contrast, at int8 the weight floor is 0.0229 against a W8A8+MoDiff total of 0.063 — there the
activation side is clearly dominant, which is why MoDiff reaches FID parity at W8A8 and cannot at W4A4.

**Two things this kills.** First, the MSE clip search is a *regression* on weight-only error (0.2443
vs absmax's 0.2184, a ~4σ gap on these seeds). Its shipped justification stands — the paired
end-to-end A/B at W4A4 measured 0.5067 → 0.4689 — but that means its win comes from an *interaction*
with activation quantization (clipping weight outliers costs less than the activation error it avoids),
exactly as `_int4_weight_scale`'s own comment suspected. Second, extending that MSE rule to the
Linear layers, which was on my list, makes the floor worse (0.2603 → 0.2888). Both are reminders that
neither `‖W − Q(W)‖` nor weight-only relL2 predicts end-to-end error; each rule has to be measured in
the mode it ships in.

**This gives learned rounding (AdaRound) a cheap evaluation loop.** A new weight quantizer can be
scored by re-running this script — one model build, no MoDiff, no activation confounder — and only
promoted to a full W4A4 run if the 0.244 floor actually moves.

### Learned rounding moves the floor 25%, and reverses which scale rule is best

`scripts/adaround_int4.py` implements AdaRound over the 70 int4 convs: initialise each weight's
rounding gate at the nearest-rounding decision, then minimise the *layer output* error against real
captured activations (6 timesteps × batch 2), annealing the gates to hard 0/1. The product is int4
codes plus one fp16 per-channel scale — the shipped layout, so **inference and kernels are
untouched**. 15 minutes of A40 time for both arms.

| scale rule | layer-output err ↓ | codes flipped | nearest floor | **AdaRound floor** |
|---|---:|---:|---:|---:|
| MSE clip (shipped) | 64.6% | 8.9% | 0.2442 ± 0.0063 | **0.1833 ± 0.0223** |
| absmax | 28.8% | 4.0% | 0.2186 ± 0.0071 | 0.1946 ± 0.0068 |

* **The floor drops 25%**, 0.2442 → 0.1833. By the FID anchors that is roughly FID 16 → 13 — real,
  but not enough on its own to bring W4A4 near fp16's 7.80.
* **The MSE clip rule and learned rounding compose, and the ranking flips.** Under nearest rounding
  MSE is *worse* than absmax (0.2442 vs 0.2186); under AdaRound it is *better* (0.1833 vs 0.1946).
  Clipping outliers buys a finer grid at the cost of rounding accuracy, and AdaRound is exactly what
  buys the rounding accuracy back. It also has more to work with — 64.6% layer-error reduction and
  8.9% of codes flipped, against 28.8% / 4.0% on the unclipped grid. So the shipped default is more
  right than the weight-only rows above suggested, *provided* rounding is learned.
* This run is deliberately the **cheap corner** of the design space: 12 calibration samples, 2000
  iterations, layer-wise with fp16 inputs. Q-Diffusion uses 256 samples per timestep over 20
  timesteps, ~10× the iterations, and feeds each layer the *already-quantized* network's activations.
  None of those three levers has been pulled here.
* Caveat: the winning arm's seed spread is 3.5× the loser's (±0.0223 vs ±0.0068). Three seeds
  separates 0.183 from 0.244 comfortably; it does not confidently separate 0.183 from 0.195.

**What this does not yet do:** the learned codes are evaluated through the fp16 conv path, not the
int4 CUTLASS kernel. Using them for real means writing codes — not just scales — into
`OptimizedInt4Conv2d`'s packed AWQ layout, which the existing export/apply mechanism does not cover.
That work is only worth doing once the floor has been pushed as far as it goes, since by itself it
buys nothing.

### Full-strength AdaRound: the floor stops at ~0.15, and sequential buys nothing

`scripts/adaround_int4_full.py` and `scripts/adaround_sequential.py` pull the three levers
Q-Diffusion uses and the cheap run left alone: 136 calibration samples (17 trajectory positions ×
batch 8) instead of 12, 10k iterations instead of 2k, and layer inputs taken from the
already-quantized network instead of the FP one.

| weight quantizer | floor relL2 | per seed |
|---|---:|---|
| nearest rounding | 0.2442 ± 0.0064 | 0.244, 0.251, 0.238 |
| AdaRound, cheap (12 samples, 2k iters) | 0.1833 ± 0.0223 | 0.158, 0.201, 0.191 |
| **AdaRound, 136 samples, 10k iters** | **0.1531 ± 0.0236** | 0.156, 0.175, 0.128 |
| **AdaRound, + sequential (in-order)** | **0.1467 ± 0.0258** | 0.129, 0.176, 0.135 |
| AdaRound, sequential done wrong | 0.4893 ± 0.0650 | 0.563, 0.439, 0.467 |

* **Data and iterations pay**: 0.1833 → 0.1531, and 3/3 seeds improve paired, though unevenly
  (+0.002, +0.026, +0.063) so the magnitude is not to be quoted precisely.
* **Sequential reconstruction, done correctly, is a tie**: 0.1467 vs 0.1531 is 1/3 seeds paired
  (+0.028, −0.001, −0.007). The lever that should matter most on paper buys nothing measurable here.
* **The floor stops around 0.147–0.153**, a 40% reduction from nearest rounding. Against the
  acceptance test set before the run (≈0.12 continue, ≈0.17 abandon) this lands in between, so the
  decision has to be made on what it implies end to end rather than on the floor alone.

**Two implementation faults, both caught by the floor and neither by the layer-error metric.**

*Bias.* Targets captured with a forward hook include the conv's bias while the prediction used
`bias=None`, so the optimiser was asked to cancel a constant offset with rounding decisions. Floor
came back 0.2879 — worse than nearest rounding — while its own layer-error metric read a healthy
"61% improvement", because both sides of that ratio carried the same offset.

*Simultaneous compensation.* The first `sequential` arm captured every layer's input from the fully
nearest-rounded model up front and then optimised all 70 layers against those inputs at once. Each
layer learned to compensate an upstream error that the finished model does not have, since its
upstream is AdaRound'd. 70 layers of over-correction compounded to floor 0.4893, twice as bad as
doing nothing, with 27.8% of codes flipped — moving hard in a direction calibrated to the wrong
input distribution. Correct sequential reconstruction processes layers in **forward execution order**
(taken from hook firing order, not `named_modules()` — a UNet with skip connections does not execute
in definition order) and writes each learned weight back before capturing the next layer's input.

*Learning rate.* 1e-3 with 10k iterations reasons correctly about how far a gate must travel and
still loses to 1e-2 under mini-batch noise: probed on 6 layers, layer output error 0.438× vs 0.251×
of nearest. Probed on the direct objective, not on the floor, so the acceptance test stayed clean.

**What it means for W4A4.** The weight floor moved 0.244 → 0.147, but W4A4+MoDiff's total is
0.44–0.47 and the activation side contributes most of it. Composing in quadrature, a floor this much
lower predicts roughly a 10% end-to-end improvement — from a mode whose FID is 200. So AdaRound is
real and it works, and it does not rescue W4A4: reaching fp16 parity there needs a much better
activation quantizer *as well*, which is a research programme rather than a fix. The recommendation
is to treat W4A4 as a speed configuration, stop spending on its quality, and not build the
codes-into-AWQ-layout injection path, which by itself buys nothing.

## Scope of MODIFF_ACT_Q, which bounds every number above

It reaches the quantized **conv** path (70 calibrated layers of 89 converted). It does **not** reach
the 21 quantized attention blocks (`quantized_std_attention.py` hardcodes `lvl = 127.0` / `7.0`), the
42 quantized Linear layers (`wxax_linear.py`), or `int4_optimized.py`'s plain modulated path (still a
literal `7.0`; irrelevant at W8, noted so nobody assumes symmetry with int8). Both arms therefore keep
A8 attention and Linears — the *comparison* is symmetric, but a row labelled A4 is "A4 in the conv
path", not a whole-network A4, and a whole-network sweep would be strictly worse.

One asymmetry favours the baseline: its static scale is Q_b/calibrated_range while the quantize
kernels clamp codes at ±127 rather than ±Q_b, so a baseline activation above its calibrated range
keeps resolution where a true b-bit quantizer saturates. The MoDiff arm's dynamic delta quantizer
cannot clip by construction. Both effects understate MoDiff, so the gains above are lower bounds.

## Code changes

| file | change |
|---|---|
| `integration/kernels/int8_optimized.py` | new `MODIFF_ACT_Q` (default 127 = shipped A8, bit-identical). Applied at the delta quantizer's `Q_level` and, as a Q_b/127 rescale of the calibrated grid, in `set_static_scale` / `end_calibration` — so the baseline path and MoDiff's t=T warm-up move together. Load-time only |
| `integration/kernels/int8_optimized.py` | `_forward_modulated`'s `step1_quantize_fprop` passed a literal `127.0` while the GN-fused paths honoured the knobs, so `MODIFF_DELTA_CLIP` was silently partial — 8 of 70 layers kept an 8-bit delta grid at any setting |
| `int8_optimized.py`, `int4_optimized.py` | **the warm-up fix**: the calibrated t=T path passed the static activation grid to every residual round, making the loop a no-op. Now a per-round dynamic scale, and `warmup_steps` 3 → 5 (`MODIFF_WARMUP_STEPS`), per the paper's Appendix D.5 |
| `int8_optimized.py`, `int4_optimized.py` | `MODIFF_DELTA_REFRESH` default 4 → 1: the paper's dynamic quantizer recomputes the scale every step, and K>1 was only ever validated at A8 |
| `scripts/probe_warmup.py` | the per-round contraction measurement that found the no-op |
| `scripts/act_bit_sweep.py` | **removed an env pin**: `main()` set `MODIFF_DELTA_REFRESH=4` unconditionally, overriding the caller, so every sweep here silently ran at K=4 — including runs launched to vary it. The value used is now inherited and recorded in the output JSON |
| `scripts/act_bit_sweep.py` | the sweep. `SWEEP_ANCHOR=strict|paper`, `SWEEP_BITS`, `SWEEP_SEEDS` |
| `scripts/verify_vs_old_w8a4.py` | the anchor attribution and forward-path instrumentation (`forward_gn_fused_modiff` 62 layers, `_forward_modulated` 8, `_forward_first_step` 70) |

Controls: A8 reproduces the shipped numbers (baseline 0.2564 against the documented 0.2378, MoDiff
0.0628 against 0.0393–0.068, both within seed spread), `MODIFF_ACT_Q=127` leaves the scale
bit-identical, and the baseline arm reproduced across two independent processes to ≤0.0002.

## Open, in the order I would take them

1. **Re-measure FID for W8A8+MoDiff.** The 7.802 in `docs/fid_2026-08-05` was taken with the broken
   warm-up. At A8 that fix is worth ~5% of relL2 so the FID is probably unchanged, but it should be
   confirmed. Only `fid/int8_modiff` needs regenerating — fp16 and int8_baseline are untouched by any
   change since. FID-vs-fp16 works today; FID-vs-real needs the LSUN LMDB re-downloaded first.
2. **Sweep `MODIFF_DELTA_CLIP` at A4 and A3.** Zero code, ~20 min, and it is what separates our delta
   quantizer from the paper's best A4 row. If an MSE-ish clip ratio moves A4 from 0.155 toward the
   A5/A6 plateau, that is the cheapest quality win available.
3. **FID for W8A5+MoDiff** (relL2 0.077) — the lowest activation precision that looks
   free here, and the row whose FID is not guessable from the anchors. `fid/fp16` still holds its 10k
   samples so FID-vs-fp16 is one generation; FID-vs-real also needs the LSUN LMDB re-downloaded
   (`/workspace/lsun_dl` is empty) and `fid/real` re-exported.
4. **Per-row / per-token dynamic activation scales** — the foldable analogue of the paper's
   channel-wise, and the only granularity improvement with a real datapath on this hardware.
5. Thread `MODIFF_ACT_Q` through attention and Linear if a whole-network claim is ever needed.
