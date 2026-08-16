# The t=T warm-up: five rounds is worse than one at W4A4, and the metric that chose five cannot see it

**A40, LSUN-churches LDM, real checkpoint, batch 8, DDIM 50, latent relL2 against a per-seed fp16
reference, 8 seeds, paired.** `MODIFF_WARMUP_STEPS` defaults to **5** for both precisions.

| rounds | W8A8 relL2 | vs 5 | W4A4 relL2 | vs 5 |
|--:|--:|---|--:|---|
| **1** | 0.0318 | −0.30% ± 8.67 (not resolved) | **0.4009** | **−26.52% ± 1.27, 8/8 seeds, RESOLVED** |
| 2 | 0.0349 | +1.41% ± 3.62 | 0.5496 | +0.61% ± 0.70 |
| 3 | 0.0342 | −1.37% ± 2.23 | 0.5449 | −0.29% ± 0.49 |
| 5 *(shipped)* | 0.0351 | — | 0.5466 | — |

**At W4A4, doing no residual refinement at all beats doing four rounds of it by 26.5%, on every seed.**
Rounds 2, 3 and 5 are indistinguishable from each other; the whole effect is between 1 and 2. At W8A8
nothing resolves in either direction.

## 1. This contradicts the reason five was chosen

`warmup_steps` went 3 → 5 on 2026-08-05, on the strength of the **per-round activation reconstruction**
(`docs/act_bits_2026-08-05`, `scripts/probe_warmup.py`) — |â − x| / |x| after each round:

| precision | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| A8 | 0.0197 | 0.00008 | 0.00000 | 0.00000 | 0.00000 |
| **A4** | **0.4006** | 0.0263 | 0.0018 | 0.00013 | **0.00001** |

By that metric five rounds is unambiguously right at A4: the anchor every later step is measured against
goes from 40% relative error to 1e-5. The latent says the opposite, and **the size of the disagreement
tracks the size of the improvement in the metric** — A4, where the anchor improves by 4 orders of
magnitude, is where the extra rounds cost 26.5%; A8, where it improves by 2 and then stops, is flat.

**A hypothesis, offered as one.** MoDiff's recursion is `o_hat_t = conv(Q(a_t − â_{t+1})) + o_hat_{t+1}`,
which is only exact if `o_hat_{t+1}` is `conv(â_{t+1})` — the two caches have to *agree*, not merely be
individually accurate. One round produces a matched pair by construction: `â = Q(a_T)` and
`o_hat = conv(â) + bias`, one quantize and one conv. Five rounds refine `â` toward `a_T` while
accumulating four more conv outputs into an **fp16** `o_hat` at successively finer dynamic scales — and
fp16 cannot hold increments that shrink by 4 orders of magnitude. So the anchor gets better and the pair
gets less consistent, and the recursion cares about the pair. This is untested; it predicts that an fp32
`o_hat` cache would restore the monotonicity, which is the experiment that would settle it.

## 2. This measurement reversed at 4 seeds and that is worth recording

The 4-seed run said something different, and I wrote it up as a recommendation before extending it:

| rounds | 4 seeds | 8 seeds |
|--:|---|---|
| 1 | −9.31% ± 2.55 | **−26.52% ± 1.27** |
| 2 | **−9.52% ± 1.25** | **+0.61% ± 0.70** |
| 3 | −3.62% ± 0.65 | −0.29% ± 0.49 |

At 4 seeds the picture was "monotone in rounds, 1 and 2 both ≈9.5% better", and the natural
recommendation was 5 → 2. At 8 seeds the 2-round effect is **gone** (+0.61%, 1/8 seeds) and the 1-round
effect has nearly tripled. Both readings are internally consistent; the 4-seed one is simply wrong.

`docs/act_bits_2026-08-05` is the report that established this exact failure mode — a 3-seed mean there
reversed sign at 8 — and it was cited two commits before this sweep was run. Citing a lesson is not the
same as applying it.

## 3. What the rounds cost

Per **cold sample**, from `bench_report_2026-08-13_postzp` §4: **+663 ms (W8A8) / +615 ms (W4A4)**, being
5 convs where a steady step runs 1. At 1 round that is ~1/5 of the work, so ~530 and ~490 ms come back per
cold sample — 4–5% of a 200-step sample and **17–20% of a 50-step one**. Every quality harness pays it
70 times, because a stale `a_hat` cache produces NaN latents and it must reset.

## 4. Recommended, not applied

`MODIFF_WARMUP_STEPS=1` is faster on both arms and resolvedly better on one. It has not been made the
default, for two reasons that are about scope rather than confidence:

1. **W8A8 is the shipping configuration and its result is unresolved.** There is no quality argument for
   moving it, only a speed one — and the warm-up does not appear in a steady-state ms/step at all, so the
   speed argument only pays for cold samples.
2. **The committed FID numbers were measured at 5.** Changing the default would make the next FID run
   non-comparable with `docs/fid_2026-08-05`, which is the table W8A8+MoDiff's FID 7.802 lives in. The
   W4A4 improvement is real but W4A4 is not usable at either setting (FID 200 vs PTQ's 278; the dominant
   error is in the weights).

So the actionable form is narrower than the headline: **set it to 1 for the W4A4 relL2 work and for
harness turnaround, and re-measure FID at 1 before changing any default.** Five is not defensible on the
evidence that chose it, but replacing it needs the FID row, not the relL2 row.

## 5. Reproduce

```bash
python integration/tests/sweep_warmup_steps.py --bits 4 --seeds 8
```
```bash
python integration/tests/sweep_warmup_steps.py --bits 8 --seeds 8
```

Each takes ~12 min on an idle A40. `MODIFF_WARMUP_STEPS` is read in the conv wrapper's `__init__`, so the
sweep rebuilds the model per setting — flipping it after construction would leave every layer on whatever
value was live at build time and report one number four times.
