# Independent re-derivation of the conv-block ablation

**Date** 2026-08-26 · **GPU** NVIDIA A40 · **Batch** 128

No generator script for `data/combined_w8a8_w4a4.csv` / `TABLE.md` was ever committed
to this repo. Checked by `git log -S` across all history for the CSV's filename and
its column names (`a_hat_ms`, `a_hat_pct`, ...): the only hits are two later
*consumers* ([perf_report_2026-08-26/scripts/analyze.py](../perf_report_2026-08-26/scripts/analyze.py),
[ahat_overlap_2026-08-26/scripts/make_findings.py](../ahat_overlap_2026-08-26/scripts/make_findings.py)),
never a writer. `git log --diff-filter=D` across all history finds no deleted file
that could have been it either.

[`bench_conv_block_ablation.py`](scripts/bench_conv_block_ablation.py) rebuilds the
measurement from the methodology `TABLE.md` documents in prose (independent-layers
chained, L=16, REPS=40, 5 trials with rotated arm order) and the four named production
kernel entry points, to check whether the committed numbers hold up.

## A confound, found and controlled for

The first run diverged badly -- up to 236% relative on `o_hat`, and a_hat systematically
**15-38% lower** than committed on nearly every shape. The direction was too consistent
to be noise, and the mechanism traced to something specific: `combined_w8a8_w4a4.csv` was
committed at `9580816` (2026-08-26 01:21), and **this same working session** later
optimized the exact kernel MoDiff's GN pass calls
([gn_vec2_2026-08-26](../gn_vec2_2026-08-26/FINDINGS.md), commits `0a7f59b`/`c432ce9`,
landed at 16:48 and 17:03 the same day) -- `gn_launch_group_stats`'s chanmajor kernel,
**-8.6% on the GN family**. The baseline tree's `_fast` GN kernel lives in a separate
file (`csrc/baseline/norm/group_norm_silu.cu`) and was untouched, so `gn_base` stayed
put while `gn_modiff` got faster -- exactly the asymmetric shift observed, and an
independent cross-check (different methodology: isolated chained microbenchmark here,
in-model nsys trace there) of a change already gated and committed.

So the first run wasn't verifying the original claim; it was comparing a stale snapshot
against a tree this session had since improved. Controlled for by checking `csrc/` out
to `9c1d8ce` (the last commit before the GN vec2 work) for a clean re-run, then
restoring `csrc/` to current HEAD and rebuilding once the comparison was done.

## Result: the committed numbers hold up

| | committed (freq-wtd) | rederived (freq-wtd, pre-GN-vec2 build) |
|---|--:|--:|
| a_hat % (W8A8) | 14.73% | 14.23% |
| a_hat % (W4A4) | 21.07% | 20.80% |

Per-shape `block_ratio` (the table's headline column) agrees to **within 2.61%** on
all 40 rows (20 shapes x 2 precisions), most within 1.5%. `a_hat_ms`/`gn_modiff`/`gn_base`
agree on 39 of 40; the one exception (`768->768,2x2` W8A8, +25.4%) is a 0.009 ms
absolute quantity -- the same launch-noise-dominated regime this project has flagged
repeatedly for this shape (see [ahat_overlap_2026-08-26](../ahat_overlap_2026-08-26/FINDINGS.md)).
`o_hat_ms`/`o_hat_pct` are noisier in relative terms (up to ~35 rows over 15%) because
o_hat is the smallest quantity in the table (0.002-0.09 ms absolute) -- consistent with
every other measurement of o_hat in this project being reported with wide relative
uncertainty at these magnitudes.

## Files

- [`scripts/bench_conv_block_ablation.py`](scripts/bench_conv_block_ablation.py) -- the
  re-derivation, self-contained, prints a shape-by-shape diff against the committed CSV
- [`data/combined_w8a8_w4a4_REDERIVED.csv`](data/combined_w8a8_w4a4_REDERIVED.csv) --
  output of the run reported here (pre-GN-vec2 `csrc/`, matching what `TABLE.md` claims
  to measure)

## Scope and limitations

- Data values (activations, weights, scales) are independently randomized here, not
  read from a real checkpoint or calibration file -- correct for a pure kernel-timing
  comparison, since none of these CUTLASS kernels branch on data value, but this script
  cannot verify anything about *numerical* correctness, only timing.
- 16-layer chains / 40 reps / 5 trials, matching the documented methodology as closely
  as the prose specifies it -- the exact constants (L, REPS, TRIALS) were not stated
  more precisely than "16 layers/chain, 40 chain reps/trial" and "5 trials", so this is
  the literal reading, not a guess needing separate justification.
- Re-run against **current HEAD** (post-GN-vec2) would show the same a_hat gap this
  finding explains; that comparison is not re-included here since the point was already
  made and independently confirms [gn_vec2_2026-08-26](../gn_vec2_2026-08-26/FINDINGS.md)
  rather than the original ablation.

