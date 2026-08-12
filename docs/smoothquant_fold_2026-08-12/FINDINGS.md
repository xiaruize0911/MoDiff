# SmoothQuant and clipping are antagonistic, and the landed W4A4 default won on the clipping

**The recorded candidate is refuted.** `int4_optimized.py::set_static_calibration` carried an
un-instrumented guess: folding SmoothQuant's `s` into the weights widens each output channel's weight
range, and at 15 levels that costs more than the activation clipping it prevents. The first half is
true and now measured — folding costs **1.215× weight error on 65 of 70 convs**. The conclusion is
not: at matched scaling, folding is **17% better**, so its weight cost is outpaid by what smoothing
buys.

**What actually won is clipping, and the landed default gets it by accident.** The shipped scale
applied to unsmoothed input is a median **5.13× too large**, which clips the peak of 43% of input
channels — and that lands near 4-bit clip-optimal. Scale it "correctly" and W4A4 PTQ goes from 0.4887
to **0.8622**, a 76% regression.

**The two do not compose.** Fold + clipping loses to clipping alone at every `k` tested. SmoothQuant
*equalises* the per-channel maxima — that is its purpose — and clipping is only cheap when they are
spread, because then only the outlier tail clips. At the same over-scale, folding roughly doubles the
share of channels that clip.

| arm | fold | over-scale k | W4A4 PTQ | W4A4 MoDiff |
|---|---|---:|---:|---:|
| `shipped` | yes | 1 (no clipping) | 0.7120 | 0.4176 |
| `recal` | no | 1 (no clipping) | 0.8622 | 0.3964 |
| **`nosmooth` — landed default** | no | ~5.13 (accidental) | **0.4887** | 0.3974 |
| `fold + clip` | yes | 2.5 | 0.5102 | 0.4289 |
| `fold + clip` | yes | 5 | 0.6119 | — |
| `fold + clip` | yes | 10 | 0.8514 | — |
| `fold + clip` | yes | 20 | 1.5218 | — |

Real LSUN-churches checkpoint, DDIM S=50, batch 8, seeds {1234, 20260805, 777}, latent relL2 vs a
per-seed fp16 reference, first run per arm discarded, all arms of a table in one process. The
`nosmooth` control reproduced to 4 decimal places across two independent runs (0.4887 / 0.4887).

**No new default.** The landed `nosmooth` file is still the best configuration measured. §5 says what
would plausibly beat it and has not been run.

---

## 1. Where the disagreement came from

`d77c516` measured that restoring `smooth_scale` on checkpoint apply is worth ~2× — rel ~0.20 against
~0.40 — and `test_int4_export_apply` still asserted it as `apply_acc < legacy_acc - 0.05`.
`docs/qdiff_bridge_2026-08-12/FINDINGS.md` §5a/§5e measured the opposite over 50 steps on the real
checkpoint, and the int4 defaults now ship bare floats because of it. Both numbers are real. The
question was which generalises, and why.

## 2. The instrument: the calibration file already contains the activation range

No forward-pass hook was needed, which matters because §5d records that both hooking techniques in
this repo return 0/70 on W4A4 (the quantize is fused into `_prequant_gn_conv`).

SmoothQuant defines `s_c = sqrt(act_max_c / w_max_c)`, and the weights are still on the module —
`_orig_weight` is an alias the file-load path never releases, because it only releases in
`end_calibration`. So

    act_max_c = s_c^2 * w_max_c

recovers the per-input-channel activation range the shipped calibration actually observed.

**This is gated, not assumed.** `static_scale` is *defined* as `7 / max_c(act_max_c / s_c)`, so if the
recovery is right, pushing the recovered ranges back through it must return exactly 7. It does:
**7.0000 on 70/70 layers**, 0 channels excluded for identity or clamped `s`. The probe refuses to
report the clipping half if more than 10% of layers miss by 5%.

## 3. The two halves, measured

**Weight half — folding costs 1.215×.** Relative Frobenius error of the reconstructed 4-bit weight,
under the per-output-channel MSE scale rule the tree ships:

| | median | worst |
|---|---:|---:|
| unfolded `W` | 0.1293 | 0.2608 |
| folded `W·s` | 0.1605 | 0.3158 |

Folding is worse on **65/70** layers. The candidate's first half is confirmed.

**Clipping half — not folding clips 43% of channels.** With the shipped scale applied to unsmoothed
input, a median **43.1%** of input channels have their observed peak pushed past ±7; of those, the
median sits 1.59× past the ceiling and the worst layer 10.27×. Under the fold, 0.00% clip — by
construction, since the scale was derived from exactly that range.

So the trade is real in both directions. What the candidate got wrong is which side wins.

## 4. The arms that separate them

**`recal` — no fold, no clipping.** Using `7/max_c(act_max_c)`, the scale a correct unsmoothed
calibration would have produced (a median 0.195× the shipped scale, so a 5× coarser grid):

* against `shipped` (fold, also no clipping): **0.8622 vs 0.7120 — folding is 17% better.** The
  recorded candidate is refuted here. Its weight-error cost is real and is outpaid.
* against `nosmooth` (no fold, clipping): **0.8622 vs 0.4887 — the clipping is worth 43%.** This is
  where the landed win actually comes from, and it is an accident of arithmetic: nothing chose that
  scale for its clipping behaviour, it is simply the smoothed-range scale used on unsmoothed input.

The implied assumed range makes the accident legible. `7/static_scale`, median over the 70 convs:

| | implied range |
|---|---:|
| shipped / nosmooth | 0.816 |
| recal — the true observed absmax | 3.937 |

A grid sized for the true absmax spends almost all of 15 levels on a range the bulk of the
distribution never reaches. A grid 4.8× tighter resolves the bulk and sacrifices the tail.

> This also corrects a reading in §5c of the qdiff report, which took qdiff's ~3.7 to be an
> *under*-estimate with ~14.8 "near the truth". The shipped calibration's own observation, recovered
> independently here, is **3.937** — so qdiff's 3.769 was very nearly the true absmax. The qdiff arm's
> saturated samples are not explained by an under-measured range, and §5c's remaining conclusion —
> that the failure mode is clipping — is not what these numbers support. The open question in §9.1
> there should be re-read in this light.

**`fold + clip` — both factors at once.** Swept, because one point cannot distinguish "clipping helps"
from "this clipping helps", and because over-clipping must eventually turn back up or the reasoning is
wrong. It does turn: k=2.5 → 0.5102, k=5 → 0.6119, k=10 → 0.8514, k=20 → **1.5218**. A clean U with
its minimum near k=2.5.

But the best folded arm (0.5102) still loses to `nosmooth` (0.4887), and at *matched* clipping the gap
is wide: at k≈5, folded 0.6119 against unfolded 0.4887. **The fold's sign flips with the clipping
regime** — better at k=1, worse at k≥5.

## 5. Why they do not compose

Over-scale by `k` and a channel clips iff its peak exceeds `1/k` of the layer's global peak. So the
cost of clipping is set entirely by how *spread* the channel maxima are — a long tail means only the
tail clips. SmoothQuant equalises them by construction: `act_max_c / s_c = sqrt(act_max_c · w_max_c)`,
a geometric mean, which is the whole point of migrating range into the weights.

Median share of input channels whose peak clips:

| over-scale k | folded (smoothed) | unfolded |
|---:|---:|---:|
| 1 | 0.0% | 0.0% |
| 2.5 | **32.6%** | 13.2% |
| 5 | **90.5%** | 51.9% |
| 10 | **100.0%** | 78.7% |

At every level, folding roughly doubles the damage the same nominal clipping does. SmoothQuant
destroys the property that makes clipping cheap. The two techniques are individually good and
antagonistic in combination, which is why the tree's best W4A4 activation configuration uses exactly
one of them.

**Not measured, and the obvious next thing.** Only the *folded* arm was swept over k. `nosmooth`'s
effective k is whatever the arithmetic produced — median 5.13 but ranging 2.15 to 10.27 across
layers — and the folded sweep's optimum sat at 2.5, not 5. An explicit no-fold clip search
(`scale = k · 7/act_max`, k swept per layer or globally) has never been run and could beat the landed
0.4887. That is the experiment this report ends on rather than one it did.

## 6. What this says the gates should assert

**`test_int4_conv` — the golden was stale, not a regression.** Attributed to `82af5bc` (2026-08-05),
which switched `_int4_weight_scale` from per-channel absmax to a per-channel MSE clip search. Proof
without a bisect: the tree ships `MODIFF_INT4_WSCALE=absmax` as a revert switch, and under it the
golden matches **bit-exactly (rel 0.00e+00)** while the default reads 8.97e-02. That commit's own
message predicted this — *"integration/tests/golden/ was captured under absmax, so
test_kernel_correctness reports int4_conv FAIL at defaults"* — so the gate has been knowingly red for
a week, which also means it could not have caught a *new* int4 conv regression in that time. Golden
refreshed against the shipped MSE rule; the absmax original is preserved at
`data/int4_conv_golden_absmax_2026-07-27.pt` (md5 `767a197d…`, new `aa3d09f4…`).

**`test_int4_export_apply` / `test_int8_export_apply` — the fixture cannot express the claim.** The
clause asserted a 0.05 accuracy separation on `nn.Conv2d(256,256,3)` with Gaussian weights and a
Gaussian input. SmoothQuant migrates per-*channel* outliers, and Gaussian data has none:

| | fixture | real checkpoint |
|---|---:|---:|
| activation absmax per input channel, max/median | 1.262 | 4.834 |
| `s`, max/median | 1.123 | 2.299 |
| weight error, folded/unfolded | 1.021× | 1.215× |

A near-uniform `s` cancels against the scale — the tree's own `_apply_smoothquant` says so. The
fixture can produce ~0.002 of separation, in the *opposite* direction, against a threshold of 0.05.
This is not a threshold to retune; the fixture is inert.

Replaced with what the method actually guarantees and the fixture *can* express: the restored
`smooth_scale`, repacked weights and per-tensor scale must equal live calibration's **bit-for-bit**.
That is strictly stronger than the ±0.02 accuracy clause it sits beside. The original clause existed
to prove the gate bites, and that intent is kept explicitly — the test now also asserts the legacy
float-only load does **not** reproduce live's state, so the discriminator is checked rather than
assumed (`discriminates=1` in the detail line).

## 7. A weakness this exposed, not fixed

`check_golden` **creates** a missing golden and returns a non-FAIL string, so on any fresh clone every
golden test passes vacuously on its first run — and `integration/tests/golden/*.pt` is gitignored
(`.gitignore:6 *.pt`), so a fresh clone is exactly the normal case. The suite that caught the
`e2e_*_vacuous` goldens has the same hole one level up. Two options, neither taken here because both
change behaviour for every developer's first run: commit the goldens behind a `.gitignore` negation
(~1.5 MB, the precedent being `!docs/**/plots/*.png`), or make a missing golden FAIL with a "seed it
with UPDATE_GOLDEN=1" message.

## Reproducing

```bash
python docs/smoothquant_fold_2026-08-12/scripts/smoothquant_fold_probe.py   # §2,3,5 + fixture, ~2 min
python docs/smoothquant_fold_2026-08-12/scripts/nosmooth_recal_ab.py        # §4 recal arm, ~10 min
python docs/smoothquant_fold_2026-08-12/scripts/fold_plus_clip_ab.py        # §4 sweep, ~16 min
python integration/tests/test_kernel_correctness.py                         # §6
MODIFF_INT4_WSCALE=absmax python integration/tests/test_kernel_correctness.py   # §6, golden attribution
```

`.pt` artifacts are gitignored and regenerable; scripts, `data/*.json` and this file are committed.
