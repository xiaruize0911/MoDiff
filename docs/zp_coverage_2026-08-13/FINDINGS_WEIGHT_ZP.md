# P2 / fix #4 REVERSES: AdaRound wins 1.58× end to end. It was deprioritised on the wrong metric

**Fix #4 should be reprioritised.** It was deprioritised on weight reconstruction error, and
`docs/paper_repro_2026-08-12/FINDINGS.md` section 5 said so plainly at the time: *"AdaRound optimises
block OUTPUT error rather than ‖W−Q(W)‖, so this does not prove ours is better end to end."* That
caveat was the whole of the remaining doubt, and closing it flips the conclusion.

| metric | ours (RTN + MSE scale) | AdaRound + weight zero point | winner |
|---|--:|--:|---|
| ‖W − Q(W)‖, median over 70 convs | **0.1293** | 0.1506 | ours, 1.16× |
| conv **output** error on real activations, median | 0.0680 | **0.0504** | AdaRound, **1.35×** (54/70 convs) |
| **end-to-end** relL2, weight-only W4 | 0.2476 | **0.1564** | AdaRound, **1.58×** |

Both instruments reproduce independently committed numbers before being used:
‖W−Q(W)‖ gives 0.1293/0.1506 against the recorded 0.1296/0.1506, and our weight-only end-to-end 0.2476
against the recorded weight-axis figure of 0.2728.

This is textbook AdaRound behaviour — it spends weight MSE to buy output fidelity — so *"ours wins on
‖W−Q(W)‖"* was never evidence that ours wins where it matters. It measured the one metric AdaRound is
willing to lose.

## Why this could be decided today, with no kernel

**Weight-only quantization emulates nothing.** Dequantize the 4-bit weights back to fp16 and the model
multiplies exactly the values a deployed W4 kernel would. The missing kernel is needed to make it
*fast*, not to make it *measurable*. That is the opposite of the activation-grid situation, where the
fake-quant harness had to emulate a grid and got it wrong three times
([FINDINGS_HARNESS.md](FINDINGS_HARNESS.md)).

**Two checkpoints, two baselines.** The AdaRound weights come from Q-Diffusion's
`church_w4a8_ckpt.pth`, whose fp32 weights differ from ours by a median relative 0.086 per layer and
**0.2490 relL2 on the sampled latent**. Scoring both quantizations against our fp16 reference would
have charged AdaRound for that checkpoint difference and called it quantization error — it is larger
than either arm's quantization error. So each arm is scored against a reference built from its own fp32
weights, in the same process, and the comparison is of the two ratios.

## A bug caught by its own magnitude, worth recording

The first version of the layer-level script treated the checkpoint's `weight` field as integer codes and
computed `(weight − z) · delta`. It produced a median weight-reconstruction error of **3.27** and an
output error of **10.33**, and the script printed *"FIX #4 IS CLOSED ON EVIDENCE, AdaRound loses by
152×"*.

**3.27 is impossible**: zeroing the weights entirely gives 1.0. Q-Diffusion stores the *original fp32*
weight plus a quantizer (`alpha`, `delta`, `zero_point`), so the quantizer has to be **applied**, not
inverted:

```
x_floor = floor(w / delta)
x_int   = x_floor + (alpha >= 0)                  # AdaRound's learned rounding, hard target
x_quant = clamp(x_int + zero_point, 0, 15)
w_q     = (x_quant - zero_point) * delta
```

The script now **refuses a verdict** if any layer's weight-reconstruction error exceeds 1.0, so this
class of bug cannot be reported as a finding again. This is the fourth time in this area that a wrong
result announced itself by magnitude rather than by being caught in review, which is an argument for
putting the bound in the code every time there is one.

## What fix #4 now costs, and why it is one kernel for two levers

```
sum_i (w_q[k,i] - z_w[k]) * a[i] = sum_i w_q[k,i]*a[i] - z_w[k] * sum_i a[i]
```

`sum_i a[i]` runs over the conv **window**, so it is per-output-**pixel** and cannot fold into a
per-output-channel bias. The paper's zero point spans **1..14** across channels, so it cannot be
dismissed as centred. It is cheap in principle — it does not depend on the output channel, so one
reduction serves all K — but the kernel does not exist.

**That is the same capability fix #2's padding defect needs.** A zero-padded tap in an asymmetric
activation grid is wrong by `-z·Σ_{missing} w_q[k]·ws[k]/s`, also per output pixel
([FINDINGS.md](FINDINGS.md)). One windowed-reduction epilogue would unlock both. Their prizes are very
different, though, and that is what should drive the decision:

| lever | prize | status |
|---|--:|---|
| fix #2, activation zero point | **1.06×** ceiling (under the 1.15× bar) | closed, negative |
| fix #4, weight zero point + AdaRound | **1.58×** end-to-end, weight-only | **reopened** |

## Recommendation, and what this file does not claim

**Reprioritise fix #4**; the 1.58× is 26× the W4A4 cross-process floor of 0.13% and does not depend on
any untrusted instrument.

Two honest limits, stated rather than buried:

1. **This is the weight axis in isolation (W4A16).** In W4A4 the activation grid contributes 0.47–0.50
   relL2 against the weight axis's 0.25, so a 1.58× weight-side improvement will not move the combined
   W4A4 number by 1.58×. Measuring the combined effect needs the kernel — or an W4A4 arm with AdaRound
   weights and symmetric-per-channel *dequantized* weights, which is the cheap next step.
2. **The kernel is not built here.** This session establishes the prize and the cost; it does not spend
   the cost.
