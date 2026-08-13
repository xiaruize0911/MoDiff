# P5: the floors hold. The thing that breaks comparability is arm order, by 28%

**The recorded noise floors are correct, and they are not the binding constraint.** Measured with one
arm per process, N processes, on an idle GPU with a contention check before every launch
([`noise_floor.py`](scripts/noise_floor.py), [`data/noise_floor.json`](data/noise_floor.json)):

| arm | mean of means | cross-process spread | recorded floor | |
|---|--:|--:|---|---|
| W8A8 PTQ (`int8_baseline`) | 0.1125 | **1.97%** | 1.3–5.1% | holds |
| W8A8 MoDiff (`int8`) | 0.0452 | **0.91%** | 1.3–5.1% | holds, tighter |
| W4A4 PTQ (`int4_baseline`) | 0.5024 | **0.09%** | 0.05–0.6% | holds |
| W4A4 MoDiff (`int4`) | 0.3958 | **0.13%** | 0.05–0.6% | holds |

So `docs/paper_repro_2026-08-12/FINDINGS.md` section 7 stands: anything under ~5% on W8A8 is not
resolvable, W4A4 is safe to a few tenths of a percent. The 7% wobble that prompted this
(0.5267 → 0.4901 → 0.5022 on one arm in one afternoon) was **not** the floor.

**With one qualification, added after more samples: the W4A4 MoDiff arm measured FIRST in a process is
bimodal.** Across six such measurements it read 0.3954, 0.3954, 0.3959, 0.3959, 0.3960 — and once
**0.3560**, a 10% excursion. Measured *second* it reads 0.3090 / 0.3095, i.e. the committed value, every
time. So the 0.13% above is the *modal* spread and under-samples the tail; the reliable protocol is not
"one arm per process" but **"measure after a warm-up arm"**, which is what the committed harnesses
happen to do. A discrete excursion rather than a drift is what a kernel- or algorithm-selection flip
looks like, which is the same suspect as the order effect below.

## What actually breaks comparability

**The W4A4 MoDiff arm depends on which arm ran before it, by 28%** —
[`arm_order_reproducer.py`](scripts/arm_order_reproducer.py),
[`data/arm_order.json`](data/arm_order.json):

| | mean | per seed |
|---|--:|---|
| `int4` measured FIRST in a fresh process | 0.3954 | [0.4415, 0.3693, 0.3754] |
| `int4` measured AFTER `int4_baseline`, same process | **0.3095** | [0.3266, 0.2883, 0.3134] |

0.3095 is the committed 0.3090. **+27.8%, against a measured cross-process floor of 0.13% — 200×.**

This is the discrepancy the tree has been chasing.
[`arm_position_effect.py`](../attn_modiff_2026-08-13/scripts/arm_position_effect.py) exists because
`linear_modiff_w4a4_ab.py` read this arm at 0.3303 against a committed 0.3090, with every axis it
could check identical. It tested *position* — int4 as arm 1 vs arm 4, two fillers between — got
bit-identical 0.30940, and concluded `position_irrelevant`, leaving the gap open.

**Position was the wrong variable; the identity of the preceding arm is the right one.** That harness's
arm 1 was already preceded by work with the same effect, so its A1 and A4 agreed with each other and
with the committed value — the comparison it drew could not see the thing it was looking for. Its
verdict is retracted, and the 6.9% it could not explain is the same phenomenon as this 27.8%.

### Ruled out

* **Process warming in general.** Building an fp16 model and running a latent first — exactly what
  `export_and_measure_zp.collect_ranges()` does to its process — leaves the int4 arm **bit-identical**
  at 0.3954. It is not "the process is cold"; it is something a quantized W4A4 arm leaves behind that
  a MoDiff W4A4 arm then reads.
* **GPU contention.** Excluded by construction: `noise_floor.py` refuses to start, and aborts
  mid-sweep, if any other process holds a GPU compute context.
* **My own first instrument.** The first version of `noise_floor.py`'s child open-coded the same
  sequence as `measure()` instead of calling it, and reported a **12.69%** spread for `int4` — which
  would have been a dramatic and wrong retraction of the W4A4 floor. Calling the real `measure()`
  brought it to 0.13%. A floor measured with a harness that does not reproduce the arm is a
  measurement of the harness; that file is kept as
  [`noise_floor_openoded_child.json`](data/noise_floor_openoded_child.json) so the mistake is on the
  record rather than deleted.

### Why MoDiff and not PTQ

`int4_baseline` reads 0.5022–0.5026 whether it runs first or after others. The asymmetry points at
temporal state: MoDiff seeds `a_hat`/`o_hat` at t=T and then accumulates `o_hat` across all 50 steps,
so a perturbation of the first step is integrated rather than averaged away. Same reason the padding
defect cost MoDiff +204% and PTQ +82% ([FINDINGS.md](FINDINGS.md)).

### Candidate mechanism, not proven

Every build runs a short sampling pass to self-calibrate 42 attention linear scales. A preceding W4A4
arm can leave global algorithm-selection state warm (cuDNN benchmark cache, CUTLASS
`can_implement`/autotune results), changing that pass's reduction orders and therefore the **scales** —
a calibration difference, which is the right order of magnitude for 28% where a pure rounding
difference is not. Proving it means capturing the 42 scales in both orders and diffing them. That is
the next step and is not claimed here.

## The rule this establishes

1. **Compare only arms measured in the same process in the same order.** Every A/B in this tree that
   does so is fine — including the fix #2 measurement, whose symmetric and asymmetric arms are in one
   run of one script.
2. **A single-arm-per-process harness is not interchangeable with a multi-arm one.** No floor argument
   licenses the comparison: 28% is 200× the floor.
3. **A harness that cannot reproduce a committed number should suspect arm order before drift.**

## Both arms are order-sensitive, and that accounts for every committed number

`int4_baseline` reads 0.5022–0.5026 measured first, against a committed 0.4695 — which looked like a
+7% regression in the tree. It is not. It is the same effect, smaller:

| arm | measured FIRST | measured SECOND | committed |
|---|--:|--:|--:|
| W4A4 PTQ (`int4_baseline`) | 0.5022–0.5023 | **0.4620 / 0.4692** (after `int4`) | 0.4695 |
| W4A4 MoDiff (`int4`) | 0.3954–0.3960 | **0.3090 / 0.3095** (after `int4_baseline`) | 0.3090 |

**Both committed values are second-arm values, and the MoDiff one reproduces exactly** (0.3090) —
0.4692 also matches `arm_position_effect.json`'s own position-2 reading of 0.46919. The PTQ second-arm
value is the looser of the two (0.4620 and 0.4692 in two runs, straddling the committed 0.4695), which
is consistent with the same bimodality noted above rather than with a second effect. So:

* **There is no drift and nothing to bisect.** Every committed W4A4 number in this tree is reproducible
  today, provided the arm order matches the harness that produced it.
* **The effect is not MoDiff-only, it is just much larger there**: 7.1% on PTQ, 27.8% on MoDiff. The
  asymmetry is consistent with MoDiff integrating a first-step perturbation into `o_hat` across all 50
  steps while PTQ re-derives each step independently.
* **The direction is consistent**: the second arm is always the *better* number, which is what a
  settling explanation predicts (whatever the first arm leaves warm, the second benefits from).

Nothing in [FINDINGS.md](FINDINGS.md) depends on any of this: fix #2's conclusion compares against the
symmetric baseline **measured in the same run**, not against a committed number, and its margins are
+82%/+204%.
