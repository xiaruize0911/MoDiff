# int8/fixed-point a_hat cache: the "free" framing was wrong; a basic simulation shows why

**Status: corrected and re-scoped, not yet built.** No CUDA code was written for this idea. This
records a correction to a claim made earlier in the same working session, plus a first
quantitative check using real calibration data already on disk.

## The claim that needed checking

The idea: replace a_hat's fp16 storage (4 B/elem read+write) with an int8 fixed-point
representation (2 B/elem), since the ceiling for removing a_hat's WRITE was already measured at
2.024/1.742 ms/step ([ahat_overlap_2026-08-26](../ahat_overlap_2026-08-26/FINDINGS.md)). The
update `a_hat += q/scale` is always an exact integer step in units of `1/scale`, so it was claimed
this could be **exact — zero additional rounding** — if a_hat is stored as an integer at that same
grid spacing.

That claim is only true if `scale` is **held constant for the whole 200-step schedule**. It is
not: `_delta_scale_args` in
[`int8_optimized.py`](../../integration/kernels/int8_optimized.py:571) reads a **per-DDIM-step**
table, `self.static_delta_scale[i]` where `i` is the step index. A fixed a_hat storage grid cannot
simultaneously match `1/scale` at every step if `scale` itself moves — and real calibration data on
this exact checkpoint already shows it moves a lot:
[`docs/modiff_correctness_2026-08-03/data/delta_calibration.json`](../modiff_correctness_2026-08-03/data/delta_calibration.json),
70 real layers, `step_gain_tail = delta_scale_tail / delta_scale_step0`:

| | min | median | max |
|---|--:|--:|--:|
| step_gain_tail | 0.25x | **12.45x** | 124.51x |

A typical layer's required delta resolution changes by **an order of magnitude** across the
schedule (finer at the tail, where deltas shrink as the trajectory converges). Re-expressing a_hat
onto a fixed int8 grid after every update therefore needs an *extra* rounding step the code's own
`q/scale` increment does not need — and by linearity of convolution that extra term does **not**
cancel: `conv(a_hat) - o_hat` accumulates it as a running sum, `sum_t conv(eps_t)`, since `o_hat`
only ever adds `conv(q_t) * dequant_t` and never sees a_hat's own storage error.

## Basic validation: simulate the drift, no CUDA needed

[`scripts/simulate_drift.py`](scripts/simulate_drift.py) — a 1D proxy for the accumulation
recursion (conv is linear, so a scalar running sum transfers the argument), using the **real**
measured `delta_scale_step0`/`delta_scale_tail` for two real layers (log-interpolated across 200
steps — the calibration file has only the two endpoints, not all 200 points; this is an
approximation, flagged in the script), against a synthetic but plausible slowly-drifting AR(1)
"true activation" trajectory.

| layer | storage | max \|a_hat − o_hat\| | drift ÷ step0 quantum | drift ÷ tail quantum |
|---|---|--:|--:|--:|
| median gain (12.45x) | fp16 (shipped) | 3.6e-3 | — | — |
| median gain (12.45x) | **int8** | **5.1e-2** | 2.4x | **14.5x** |
| max gain (124.5x) | fp16 (shipped) | 2.8e-3 | — | — |
| max gain (124.5x) | **int8** | **1.5e-1** | 3.4x | **7137x** |

"Quantum" = `1/scale` at that point in the schedule — the resolution the delta code itself already
operates at, i.e. the size of error the scheme is designed to tolerate every step by construction
(error feedback). The right way to read this table: at **step 0**, the int8 storage error is a
modest 2–3x the code's own quantum — probably tolerable, in the same spirit as ordinary
quantization noise. At the **tail**, where the per-step table has moved to a much finer quantum
because real deltas have gotten smaller, the *same, fixed* int8 storage error is now **14.5x to
7137x** larger than what the scheme is trying to resolve there. For the higher-gain layer the
drift **never stops growing** (max = final value) — it is not a bounded, self-correcting wobble,
it is a monotonic divergence for as long as the schedule keeps demanding finer resolution than a
flat 8-bit grid can supply.

## What this means for the idea

**A single flat int8 grid across the whole schedule is very likely not viable**, for any layer
whose `step_gain_tail` is much above 1x — which the calibration data says is *most* layers
(median 12.45x). This is a materially different conclusion from the "free, exact" framing floated
earlier in this session, and the correction matters: shipping the naive version would plausibly
show up as degraded fine-detail quality specifically in the late denoising steps, which is exactly
where diffusion models do their most quality-sensitive work.

**What would make it viable, at added cost:**
1. **A refreshed/rescaled int8 grid** — re-derive a_hat's own LSB periodically (analogous to
   `MODIFF_DELTA_REFRESH`), paying a re-quantization event every K steps instead of never. This
   reopens the exactness argument locally (between refreshes, if the LSB is chosen to track that
   window's scale) at the cost of K "resync" operations across the run — cuts into, but likely does
   not eliminate, the 2.024/1.742 ms/step ceiling.
2. **A wider fixed format** (e.g. 12 bits instead of 8) buys back headroom at the cost of some of
   the bandwidth saving — halves the win rather than fully claiming it.
3. **Per-layer gating** on `step_gain_tail`: layers near 1x (a handful exist, min observed 0.25x)
   might tolerate a flat grid fine; high-gain layers would not, without (1) or (2).

None of these is "free" the way the original framing implied, and any of them needs the same thing
this idea always needed for its *quality* dimension regardless: **FID, not a proxy metric** — this
simulation only checks the `o_hat = conv(a_hat)` bookkeeping invariant, not whether the resulting
drift actually hurts sample quality. It could plausibly be quality-neutral even at 14x-7000x the
local quantum if error feedback absorbs it faster than it visibly matters — that is an empirical
question this script does not answer.

## Files

- [`scripts/simulate_drift.py`](scripts/simulate_drift.py) — self-contained, no CUDA, reads the
  committed calibration JSON

## Scope and limitations

- **Only 2 of 70 layers**, chosen as the median- and max-`step_gain_tail` cases from the committed
  calibration file — a spot check, not a sweep. The full distribution (min 0.25x) suggests some
  layers may not have this problem at all.
- **The 200-step scale trajectory is log-interpolated from 2 measured points**, not the true
  per-step curve — the calibration file does not record intermediate points. If the true curve is
  not smooth (e.g. jumps sharply near a specific timestep), the interpolation could over- or
  under-state the drift at any given step, though the two ENDPOINTS (where drift is measured
  worst) are real, not interpolated.
- **The "true activation" trajectory is synthetic** (an AR(1) walk), not drawn from a real model
  run — chosen only to be "slowly varying," which is what the calibration data's scale curve
  implies, not calibrated to match any specific layer's real statistics.
- **This checks the bookkeeping invariant only, not FID.** A drift that looks alarming here could
  still be quality-neutral, or a drift that looks small could still matter — this script cannot
  distinguish those; it can only tell you the flat-8-bit idea, AS STATED, does not preserve the
  invariant the way a constant-scale schedule would have, and by how much.

---

## Follow-up 2026-08-26: fp8, a periodic-refresh attempt, and a data-grounded bit budget

**Prompted by:** the project owner relaxing the constraint from "strictly no numerics change" to
"small error is acceptable." This reopens the a_hat storage-format question, but the answer below
is still negative for a plain int8 format — the reopened constraint does not change how many bits
the problem needs.

### fp8 is worse than int8-fixed, not better

Floating-exponent fp8 (`torch.float8_e4m3fn` / `float8_e5m2`, both natively supported in this
torch build) was the natural next guess — its floating exponent tracks magnitude the way fp16
already does, seemingly avoiding int8-fixed's "flat grid can't track a moving target" problem.
[`scripts/simulate_drift_fp8.py`](scripts/simulate_drift_fp8.py) tested this against the same real
step0/tail scale trajectories as the original simulation, and found the opposite:

| layer | storage | drift ÷ step0 quantum | drift ÷ tail quantum |
|---|---|--:|--:|
| median gain (12.45×) | int8-fixed | 2.38x | 14.5x |
| median gain (12.45×) | **e4m3** | 38.83x | **236.9x** |
| median gain (12.45×) | **e5m2** | 55.24x | **337.0x** |
| max gain (124.5×) | int8-fixed | 3.44x | 7137x |
| max gain (124.5×) | **e4m3** | 14.19x | **29410x** |
| max gain (124.5×) | **e5m2** | 29.71x | **61577x** |

**Mechanism, confirmed directly:** at magnitude ≈1.0 (a_hat's typical range), fp8's LSB is **0.125**
(e4m3) or **0.25** (e5m2) — 13× to 27× coarser than a dedicated int8 fixed-point grid's **0.0094**
over the same ±1.2 range. fp8 spends 4–5 of its 8 bits on an exponent a_hat's already-bounded
magnitude does not need (GroupNorm keeps it in a narrow range throughout the schedule), leaving
fewer effective mantissa bits than a purpose-built fixed format that spends all 8 bits on
precision. **The real problem was never a_hat's own dynamic range; it is the delta code's shrinking
quantum**, which fp8's exponent — driven by a_hat's value, not by the schedule's scale table — does
not track at all.

### The "periodic refresh" idea had a bug, and fixing it exposes why refresh cannot work either

The first version of a K-step-refreshed grid ([`scripts/simulate_drift_refresh.py`](scripts/simulate_drift_refresh.py))
showed drift of 50–7900× the tail quantum — far worse than the unrefreshed case. That run had a
bug: the refresh derived a_hat's grid from `headroom / scale`, which shrank the *represented range*
at every refresh rather than the *resolution*, so a_hat's own value (≈1.0) was catastrophically
clipped against a grid sized for the much smaller delta.

Fixing the bug exposes the real constraint directly, and it has nothing to do with cadence:

```
levels needed = 2 * a_hat_range / quantum_at_tail
```

For the median-gain layer: `2 × 1.2 / 0.0035 ≈ 683 levels ≈ 9.4 bits`. For the max-gain layer:
`2 × 1.2 / 0.000021 ≈ 116,000 levels ≈ 16.8 bits`. **No refresh cadence creates more quantization
levels** — 8 bits is 256 levels, full stop, regardless of how often the grid is re-derived.
Refreshing only changes *where* the grid sits, never *how many* distinct values it can represent.

### Data-grounded: how many of the 70 real layers could ever fit in 8 bits

[`data/bits_needed_70layers.json`](data/bits_needed_70layers.json), computed directly from
`activation_scale` (a proxy for a_hat's own range) and `delta_scale_tail` for all 70 calibrated
layers in the committed calibration file:

| | bits needed to span a_hat's range at the tail's resolution |
|---|--:|
| min | 6.0 |
| p25 | 11.3 |
| **median** | **11.6** |
| p75 | 12.3 |
| max | 14.9 |

**Only 2 of 70 layers (2.9%) need ≤ 8 bits** — both `step_gain_tail ≈ 0.25×` (the delta actually
gets *coarser*, not finer, toward the tail for these two). The median real layer needs **11.6
bits**; int8 offers 8. This is not a marginal shortfall an acceptable-error budget can obviously
absorb — it is roughly an order of magnitude short, matching the earlier drift simulation's
10×–7000× figures.

### What this means for "small error is now acceptable"

The relaxed constraint does not rescue a flat 8-bit format for a_hat — the shortfall is a bit-count
problem, not a rounding-mode or refresh-cadence problem, and no variant tested here (fixed, fp8,
refreshed-fixed) changes that. **The only way to make further progress on this specific idea is an
actual FID measurement**: truncate a real layer's a_hat to int8 (median-layer shortfall ≈ 3.6 bits,
i.e. roughly 12× coarser resolution than what an 11.6-bit-adequate format would give) and see
whether that specific, now-quantified magnitude of extra error is visible in samples. Proxy-metric
analysis has reached the limit of what it can decide here.

**Two structurally different directions this does NOT rule out, both unexplored:**
- **Per-layer gating on `step_gain_tail`**: a small number of real layers (the ~2.9% needing ≤8
  bits, and plausibly a few more that would tolerate a *little* extra error) could adopt int8 a_hat
  near-losslessly. The win is capped by how few such layers exist, likely too small to be worth the
  engineering on its own.
- **Non-uniform (companding) 8-bit quantization** — concentrate levels where a_hat's value actually
  spends its time rather than spacing them uniformly across its full range. Not evaluated
  quantitatively here; would need real a_hat value *histograms*, not just range, to size properly.

### Files

- [`scripts/simulate_drift_fp8.py`](scripts/simulate_drift_fp8.py)
- [`scripts/simulate_drift_refresh.py`](scripts/simulate_drift_refresh.py)
- [`data/bits_needed_70layers.json`](data/bits_needed_70layers.json)
