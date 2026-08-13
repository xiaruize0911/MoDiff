# P1–P5, 2026-08-13: five open problems, five answers

| # | problem | answer | file |
|---|---|---|---|
| **P1** | fix #2 activation zero point: coverage incomplete, both arms' end-to-end numbers artifacts | **NO.** Coverage completed and gated; the obstacle is zero-padding (per-output-**pixel**, unfoldable into a per-channel bias) and the ceiling is **1.06×** against a 1.15× bar. Closed negative. | [FINDINGS.md](FINDINGS.md) |
| **P2** | fix #4 weight zero point / AdaRound: deprioritised on evidence | **REVERSED.** It was deprioritised on ‖W−Q(W)‖, the one metric AdaRound is willing to lose. On conv output error it wins 1.35×; **end to end, weight-only, 1.58×**. Reprioritise. | [FINDINGS_WEIGHT_ZP.md](FINDINGS_WEIGHT_ZP.md) |
| **P3** | 70 orphaned int4 wrappers: memory cost unmeasured, ~113 MiB estimated | **114 MiB**, two independent methods agreeing to 0.7%. The docstring's "1014.6 MiB leak" overstated by 8.9× — 901 MiB of it aliased the shared fp16 weight. Docstring corrected. | [FINDINGS_ORPHAN_MEMORY.md](FINDINGS_ORPHAN_MEMORY.md) |
| **P4** | fake-quant harness untrusted after two failed self-checks | **RETIRED.** Scored against the now-known answer it failed a **third** time, and would have said "implement" where the truth is negative. Replacement validated, with its scope stated: magnitude questions, not ratio selection. | [FINDINGS_HARNESS.md](FINDINGS_HARNESS.md) |
| **P5** | W8A8 noise floor 1.3–5.1% makes <5% unresolvable on that axis | **Floors hold** (W4A4 0.09%/0.13% modal, W8A8 1.97%/0.91%). But they are not the binding constraint: **arm order moves W4A4 MoDiff by 28% and PTQ by 7–9%**, and both committed values are second-arm values. Explains the 6.9% `arm_position_effect.py` could not, and retracts its verdict. | [FINDINGS_NOISE_FLOOR.md](FINDINGS_NOISE_FLOOR.md) |

## The thread running through P1 and P2

Both remaining quality levers need the **same missing capability**: a per-output-pixel windowed
reduction. Fix #2 needs it because a zero-padded tap in an asymmetric activation grid is wrong by
`-z·Σ_{missing} w_q·ws/s`; fix #4 needs it because `z_w·Σ_i a[i]` sums over the conv window. One
epilogue would unlock both — and the measurements say build it for fix #4 (1.58×), not for fix #2
(1.06×).

## What each conclusion rests on

Every number here is reproducible from a script in [`scripts/`](scripts) with data in
[`data/`](data). Instruments are validated against independently committed numbers before use:

| instrument | reproduces |
|---|---|
| `zp_activation_error.py` | asymmetry ratio 19.89× vs the recorded 19.91× |
| `weight_zp_output_error.py` | ‖W−Q(W)‖ 0.1293 / 0.1506 vs the recorded 0.1296 / 0.1506 |
| `weight_zp_end_to_end.py` | weight-axis relL2 0.2476 vs the recorded 0.2728 |
| `noise_floor.py` | the committed W4A4 MoDiff 0.3090, when arm order matches |

## Four bugs that announced themselves by magnitude

Kept on the record because the pattern is the actionable part — in each case the *size* of the number,
not review, is what caught it:

1. **relL2 7.3057** on the MoDiff arm — the t=T quantize had no guard at all, so the census that
   listed 70 "contaminated" pairs never saw the one site that was broken (P1).
2. **350 zp calls where 70 was right** — the zero point applied to the four warm-up *residuals* too, a
   bias-free path. A `> 0` check passed it; the 4× ratio invariant caught it (P1).
3. **12.69% noise floor** — measured with a harness that open-coded `measure()` instead of calling it.
   Calling the real one gave 0.13%. That would have been a dramatic and wrong retraction (P5).
4. **weight reconstruction 3.27** — impossible, since zeroing the weights gives 1.0. The AdaRound
   quantizer was inverted instead of applied, and the script printed "FIX #4 IS CLOSED, AdaRound loses
   by 152×" before the bound was added (P2).

Three of the four produced a confident, quotable, wrong conclusion first. Each now has an explicit
bound in the code rather than a reader expected to notice.
