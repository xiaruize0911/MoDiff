# Activation precision at fixed W8: A8 → A2, MoDiff on and off, against the paper

**2026-08-05 · A40 · LSUN-Churches LDM-8, real checkpoint · DDIM 50 · batch 8 · 3 seeds, paired ·
latent relL2 vs fp16**

The first sweep of activation precision in this project, and the first measurement of the
configuration the paper actually claims. Every shipped mode pairs the two precisions (W8A8, W4A4), so
every earlier quality number confounds weight error with activation error — and MoDiff only addresses
the second. W4A4+MoDiff at FID 200 (`docs/fid_2026-08-05`) is the visible consequence: int4 *weights*
dominate there, so that row cannot test the paper's claim either way.

Data: `data/act_bit_sweep.json`, `data/act_bit_sweep_paper_anchor.json`, `data/verify_vs_old_w8a4.json`.

## The t=T anchor is a protocol choice, and it dominates everything below A6

MoDiff quantizes a difference against â, and â is seeded by the t=T warm-up. So "what precision does
t=T run at" is a free parameter, and it turns out to be the single largest one at low activation
bits. Both readings are measured here:

| | t=T warm-up | what it answers |
|---|---|---|
| **paper** | left on the A8 grid | the published method. Appendix B: *"Warm-up: We apply warm-up at the first step, where we use full activation for computation."* Their anchor is full precision; the A8 static grid is this implementation's nearest equivalent |
| **strict** | at A_b like everything else | what a pure b-bit activation datapath would cost, with no high-precision step anywhere |

## Result

Q is the symmetric code ceiling 2^(b−1)−1. The baseline column is MoDiff off with the calibrated
per-tensor grid rescaled to b bits — identical configuration in both runs, and it reproduced to
≤0.0002 across them, which is the pairing's cross-run control.

| A bits | levels | baseline (PTQ) | **MoDiff, paper anchor** | MoDiff, strict | gain (paper) |
|---|---:|---:|---:|---:|---:|
| A8 | 255 | 0.2564 ± 0.017 | **0.0628 ± 0.029** | 0.0627 ± 0.028 | 4.09× |
| A7 | 127 | 0.2578 ± 0.049 | **0.0596 ± 0.030** | 0.0669 ± 0.028 | 4.33× |
| A6 | 63 | 0.3185 ± 0.043 | **0.0617 ± 0.035** | 0.0795 ± 0.025 | 5.16× |
| A5 | 31 | 0.7356 ± 0.020 | **0.0730 ± 0.034** | 0.1563 ± 0.018 | 10.07× |
| A4 | 15 | 0.8071 ± 0.038 | **0.1529 ± 0.022** | 0.3367 ± 0.066 | 5.28× |
| A3 | 7 | 0.9313 ± 0.034 | 0.3791 ± 0.026 | 0.8206 ± 0.021 | 2.46× |
| A2 | 3 | 0.9366 ± 0.061 | 0.6319 ± 0.031 | 1.3308 ± 0.077 | 1.48× |

Read against the relL2 → FID anchors from `docs/fid_2026-08-05` (0.039 → FID 7.80 = parity with fp16;
0.238 → 16.4; 0.456 → 200; 0.784 → 278):

* **Under the paper's protocol MoDiff is flat from A8 to A5** — 0.063, 0.060, 0.062, 0.073, i.e. four
  bits of activation precision removed for nothing measurable, all inside the seed spread.
* **A4 is where it starts to cost** (0.153, between the FID-7.8 and FID-16.4 anchors) and **A3 is
  where it breaks** (0.379, approaching the 0.456 → FID 200 anchor).
* **The baseline collapses two to three bits earlier**: already at FID ≈ 16 at A8 (0.256), degrading
  at A6 (0.318) and gone by A5 (0.736, next to the 0.784 → FID 278 anchor).
* **MoDiff at A4 (0.153) beats the W8A8 baseline (0.256).** This is the paper's claim in substance —
  4-bit activations with modulation beating 8-bit PTQ without it — and it reproduces here at
  per-tensor granularity.
* **The anchor is worth 2.2× at A4 and 2.1× at A2**, and it is what decides whether MoDiff helps at
  all in the extreme: with a 2-bit anchor MoDiff is *worse* than the baseline (1.33 vs 0.94), with an
  8-bit anchor it is still 1.48× better. A 2-bit â is not a reference the feedback term can correct
  against; it is noise that the feedback then propagates.

Keeping t=T high-precision is cheap and is not a thumb on the scale: it is 1 step in 50, it is the
grid the W8A8 model already carries, and the paper specifies it. Anyone quoting an A_b number from
this project should say which anchor it used.

ms/step is flat across every row (baseline 11.8–16.8, MoDiff 17.2–22.8, fp16 18.1–21.1) — the control
confirming this is a quality instrument only. A low A_b costs nothing and saves nothing here:
activations keep their int8 container and the GEMM stays W8A8. A real 4-bit activation datapath needs
int4 tensor cores, which take both operands at 4 bits — no mainstream ISA has a mixed s8×s4 MMA, so
W8A4 is not a speed configuration on any hardware, only a quality one.

## What this corrects in docs/fid_2026-08-05/FINDINGS.md

That report's W8A4+MoDiff row (relL2 0.127) was **mislabelled but not wrong in its conclusion**. It
was produced by abusing `MODIFF_DELTA_CLIP` (Q_level = 127/ratio), which moves only the delta
quantizer and leaves the static grid at A8 — i.e. it measured the *paper-anchor* configuration, not
"A4 everywhere". Reproduced in `data/verify_vs_old_w8a4.json`:

| configuration | relL2 |
|---|---:|
| old: `DELTA_CLIP=127/7`, anchor at A8 | 0.1626 ± 0.033 (0.127 was its bottom seed; single-seed) |
| this sweep, paper anchor, A4 | 0.1529 ± 0.022 |
| strict: `ACT_Q=7`, every conv activation site at A4 | 0.3367–0.3581 |

The two arms differ in exactly one thing — the t=T grid — so the 2.2× is attributable with no
confounder. And since the paper's own protocol keeps the warm-up at full precision, **the old row's
claim that W8A4+MoDiff beats the W8A8 baseline stands**; it just needed the anchor stated. An earlier
version of this document withdrew that claim, which was an over-correction, and the note in the FID
report has been revised accordingly.

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
| `scripts/act_bit_sweep.py` | the sweep. `SWEEP_ANCHOR=strict|paper`, `SWEEP_BITS`, `SWEEP_SEEDS` |
| `scripts/verify_vs_old_w8a4.py` | the anchor attribution and forward-path instrumentation (`forward_gn_fused_modiff` 62 layers, `_forward_modulated` 8, `_forward_first_step` 70) |

Controls: A8 reproduces the shipped numbers (baseline 0.2564 against the documented 0.2378, MoDiff
0.0628 against 0.0393–0.068, both within seed spread), `MODIFF_ACT_Q=127` leaves the scale
bit-identical, and the baseline arm reproduced across two independent processes to ≤0.0002.

## Open, in the order I would take them

1. **Sweep `MODIFF_DELTA_CLIP` at A4 and A3.** Zero code, ~20 min, and it is what separates our delta
   quantizer from the paper's best A4 row. If an MSE-ish clip ratio moves A4 from 0.153 toward the
   A5/A6 plateau, that is the cheapest quality win available.
2. **Make the anchor precision its own knob.** Today `MODIFF_ACT_Q` moves the delta grid and the t=T
   grid together; the measurements above show they should be settable independently, and that the
   shipped policy should keep t=T at A8.
3. **FID for W8A5+MoDiff** (paper anchor, relL2 0.073) — the lowest activation precision that looks
   free here, and the row whose FID is not guessable from the anchors. `fid/fp16` still holds its 10k
   samples so FID-vs-fp16 is one generation; FID-vs-real also needs the LSUN LMDB re-downloaded
   (`/workspace/lsun_dl` is empty) and `fid/real` re-exported.
4. **Per-row / per-token dynamic activation scales** — the foldable analogue of the paper's
   channel-wise, and the only granularity improvement with a real datapath on this hardware.
5. Thread `MODIFF_ACT_Q` through attention and Linear if a whole-network claim is ever needed.
