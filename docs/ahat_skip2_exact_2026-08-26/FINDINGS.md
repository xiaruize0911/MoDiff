# K=2 deferred-write a_hat: mathematically exact, numerically verified, not yet built

**Status: verified in Python (bit-exact), no CUDA code written.** This is a different design from
the one refuted in [int8_ahat_cache_2026-08-26 follow-up (2)](../int8_ahat_cache_2026-08-26/FINDINGS.md)
("skip-K a_hat writes"), prompted by the project owner's intuition that K=2 specifically should be
trustworthy where general K was not.

## Why the earlier skip-K design failed, and what is different here

The earlier design froze a_hat at a checkpoint `A` for the whole window and computed every step's
code against that same frozen `A`: `c1 = Q(v1 - A)`, `c2 = Q(v2 - A)`. Both codes are independent
"distance from anchor" measurements, not increments relative to each other, so `o_hat`'s `+=`
accumulation double-counts the shared displacement — a structural bug, proven by a concrete trace
in that document (4 steps overshooting the true trajectory 3-4x).

**The fix: don't freeze the reference — defer only the DRAM *write*, and reconstruct the correct
intermediate reference on demand.** At the skipped step, keep the code `c1` (already being written
to the code buffer anyway, for the conv/o_hat pipeline to consume) in a small pending-code register.
At the next (catch-up) step, recompute `a_hat_1 = A + D(c1)` — the *exact same formula, exact same
inputs* the standard scheme would use if it had written `a_hat_1` to its own buffer — and quantize
`c2` against *that*, not against the frozen `A`. Because `a_hat_1` is bit-identical to what the
standard scheme would compute, every downstream quantity (`c2`, the new checkpoint `A'`, and
`o_hat` at every step) is forced to be bit-identical too, by induction.

```
step 1 (skip):     Δ1 = v1 - A                c1 = Q(Δ1; s1)
                   write c1 to the code buffer (needed anyway); do NOT write a_hat
                   keep c1 in a small pending-code register until step 2

step 2 (catch-up): â1 = A + D(c1; s1)          [reconstructed, not read from a_hat's own buffer]
                   Δ2 = v2 - â1                c2 = Q(Δ2; s2)
                   A' = â1 + D(c2; s2)          [THIS is written -- new checkpoint]
```

`o_hat_t = o_hat_{t-1} + α_t · conv(c_t)` is untouched by any of this — it only ever depends on the
codes, which are unchanged.

## Numerical verification

[`scripts/verify_skip2_exact.py`](scripts/verify_skip2_exact.py) implements both schemes in
`torch.float16` (matching `a_hat_cache`'s real dtype) and checks three things at every step, using
a real per-step delta-scale trajectory (the same `step_gain_tail` calibration data used throughout
this investigation, so scale genuinely varies step to step rather than being held constant) and a
random per-step scalar standing in for conv (valid by linearity — if the identity holds for an
arbitrary linear operator's scalar case, it holds for any linear operator, including the real
conv):

1. Every code `c_t` bit-identical to the standard scheme.
2. `o_hat` bit-identical to the standard scheme at every step.
3. `a_hat`'s DRAM buffer, at every catch-up (write) step, bit-identical to the standard scheme's
   `a_hat` at the same step index.

**Result: all three hold, with zero exceptions, across 70 real calibrated layers × 5 random seeds
× 200 steps = 350 independent trajectories.** (An earlier version of the verification script's own
*comparison* logic had an off-by-one bug — comparing the reconstructed reference at step `t` against
the standard scheme's `a_hat` at `t` instead of `t-1` — caught immediately by inspecting the raw
traces before trusting the aggregate pass/fail output; fixed before the sweep above.)

## What this buys, and what it costs (byte accounting, per element, per 2-step window)

| | standard (K=1) | K=2 deferred-write |
|---|--:|--:|
| a_hat reads | 2 (4 B) | 2 (4 B) — both steps re-read the same checkpoint `A` |
| a_hat writes | 2 (4 B) | **1 (2 B)** |
| extra code read | 0 | **1 (~1 B, int8)** — `c1` must be read back at the catch-up step |
| **total, a_hat-related traffic** | 8 B / 2 steps | **7 B / 2 steps** |

A **12.5% reduction in a_hat's own DRAM traffic** by byte count — smaller than the naive
"eliminate one write out of two" estimate (25%) because of the extra code read the reconstruction
needs. Prompted by the project owner questioning whether this nets out to "just saving one read"
— a fair reading of the raw byte accounting, since the net difference (1 B saved per 2-step
window) is numerically the size of one code read.

**That reading understates the real saving**, because it assumes reads and writes cost the same
per byte. They do not: [ahat_overlap_2026-08-26](../ahat_overlap_2026-08-26/FINDINGS.md) §1 found
removing a_hat's write saves *more* than a byte-ratio model predicts (31-32% measured vs 28.6%
predicted) because a store drags in write-allocate traffic a plain read does not. Recomputing with
the real freq-weighted kernel costs recovered from that document's per-shape table
(`w1c1` ≈ 6.494 ms/step, `w0c1` ≈ 4.471 ms/step, five dominant shapes) and pricing the *extra*
pending-code read at `w0c1`'s own implied per-byte rate (0.894 ms per B/elem — a rate uncontaminated
by write-allocate, since `w0c1` has no a_hat write at all):

| | ms/step |
|---|--:|
| skip step (= today's `w0c1`) | 4.471 |
| catch-up step (= today's `w1c1` + one extra pending-code read) | 7.388 |
| **average over the 2-step window** | **5.930** |
| baseline (`w1c1` every step) | 6.494 |
| **net saving** | **0.564 ms/step** (0.73% of the 77.00 ms step; **28%** of the full write-elision ceiling, 2.024 ms) |

So the real expected saving is better than the naive "one read's worth" framing (which would put it
near zero once measured in time rather than bytes), but it is a genuinely small number — roughly
half of the already-small `gn_vec2_2026-08-26` win (-1.29 ms/step) and about a quarter of the full
write-elision ceiling. This is the honest scale of the benefit to weigh against the implementation
cost below, not the 25-50% a "K=1 skipped, K=1 written" framing might suggest at first glance.

## Implementation requirements, not yet built

- A small persistent **pending-code buffer** (one value per element, sized like the code buffer
  itself) that survives from the skip step to the catch-up step. The existing code buffer is a
  transient scratch tensor already consumed and potentially reused/overwritten by the conv kernel
  within the same step; this needs either a dedicated small buffer or a guarantee about lifetime
  ordering so the catch-up step's read happens before anything could clobber it.
- The reconstruction (`â1 = A + D(c1)`) needs to happen inside the GN apply kernel at the catch-up
  step, which already reads `a_hat` and would now also read the pending-code buffer — this keeps
  the traffic on the kernel side that already gets 73-81% of peak bandwidth for a_hat's own access
  ([ahat_overlap](../ahat_overlap_2026-08-26/FINDINGS.md) §1), not the conv side (measured worse,
  57-69% of peak, in the same document).
- `_delta_should_refresh`/the static per-step delta-scale table interact with this only through
  `scale_traj` in the verification above (a fixed, externally-supplied per-step value) — the
  design does not depend on which mode (static/dynamic) supplies that scale, since it only changes
  *when a_hat is written*, not how the scale is obtained.

## Real CUDA-level build: bit-exact, and the model's estimate held up almost exactly

Built at the project owner's request ("实现一下试试看" — implement it and try), rather than stopping
at the Python-level model. [`scripts/probe.cu`](scripts/probe.cu) adds two real kernels alongside a
`probe_standard` reference (all three copied from the shipped
`gn_apply_delta_quantize_flat_vec2_kernel`, same launch geometry: block 256,
`grid=ceil(numel/2/256)`):

- `probe_skip` — identical to ahat_overlap's already-measured `w0c1` arm (write elided).
- `probe_catchup` — the new kernel: reads the checkpoint `A` and the still-resident skip-step code
  `c1` (both **before** either is overwritten), reconstructs `â1`, quantizes this step's delta
  against it, writes the new checkpoint, and overwrites the code buffer with `c2`.

**First correctness run failed** — `max|diff|≈0.0156` on every shape, not zero. The bug: the
standard kernel implicitly *rounds a_hat to fp16 every time it writes it* (`gn_store2` stores fp16;
the next read via `gn_load2` reads back the ROUNDED value, not the exact float32 sum). The
reconstruction `â1 = A + D(c1)` in the catch-up kernel was computed in pure float32, carrying MORE
precision than the standard scheme's `a_hat_1` actually has. The Python-level check earlier in this
document *had* modelled this correctly (`.to(torch.float16)` after the sum) — the CUDA port simply
missed porting that detail. Fix: round-trip the reconstruction through `__half` before using it
(`__half2float(__float2half_rn(...))`) — an ALU-only operation, so it costs no extra memory traffic
and does not change the byte-accounting model above.

After the fix, [`scripts/verify_and_bench.py`](scripts/verify_and_bench.py) confirms bit-exactness
against `probe_standard` (two independent, differently-seeded steps with different scales, so the
window boundary and the scale change are both exercised) across all 7 measured shapes — `a_hat` and
the code buffer both `torch.equal`, **zero exceptions**:

| shape | a_hat exact | Yq exact |
|---|---|---|
| all 7 shapes (192-768 ch, 2×2-32×32) | **True** | **True** |

**Benchmark, same 5-trial rotated-order methodology as ahat_overlap:**

| shape | freq | baseline ms/step | skip2 ms/step | saved | % |
|---|--:|--:|--:|--:|--:|
| 192,32×32 | 7 | 0.4031 | 0.3743 | +0.0288 | +7.2% |
| 384,16×16 | 7 | 0.2043 | 0.1896 | +0.0148 | +7.2% |
| 384,32×32 | 2 | 0.8036 | 0.7452 | +0.0583 | +7.3% |
| 576,32×32 | 1 | 1.2012 | 1.1129 | +0.0883 | +7.4% |
| 768,16×16 | 2 | 0.4020 | 0.3735 | +0.0285 | +7.1% |
| 768,2×2 | 12 | 0.0072 | 0.0072 | −0.0001 | −0.8% (launch-noise dominated, as in ahat_overlap) |
| 384,8×8 | 8 | 0.0537 | 0.0505 | +0.0033 | +6.1% |

**Freq-weighted over the 5 dominant shapes: 0.5671 ms/step saved — the model predicted 0.564
ms/step. Within 0.5% of the model**, despite the model being built from a different document's
byte-rate data, not fit to this measurement. This is an unusually strong confirmation that the
economics (write-allocate-inflated write savings, ordinary-rate extra read cost) were understood
correctly, not just lucky.

## Verdict

**Confirmed on real hardware: bit-exact, and worth roughly 0.567 ms/step (~0.74% of the 77 ms
step) at the kernel level.** Unlike every quantization/sparsity/temporal-gating idea earlier in
this investigation, this one is not a quality-for-speed trade — it is a free, zero-error
restructuring, and it is now measured rather than modelled. The one bug the real build surfaced
(missing fp16 round-trip in the reconstruction) was a porting slip, not a flaw in the underlying
math — the Python-level model had it right, and the fix cost nothing in the byte/time economics.

## Production integration: built, validated end-to-end, and the honest timing result

Built at the project owner's request ("推进" — proceed). Everything below is a **Python-level
monkeypatch** of `OptimizedInt8Conv2d.forward_gn_fused_modiff` (the entry point
`fused_ops/fused_resblock.py` actually dispatches to for the real int8 generation path), not a
change to `csrc/` — kept reversible and low-risk per this session's established practice, at the
cost of an extra Python/pybind round-trip whose price shows up below.

### A real bug the production comparison caught: `--use_fast_math`

Before trusting anything, [`scripts/verify_vs_production.py`](scripts/verify_vs_production.py)
compared my `stats_launch` + `probe_standard` pipeline against the REAL
`modiff_cutlass.group_norm_silu_delta_quantize_nhwc`, with real modulation (`mod_scale`/`mod_shift`)
and `smooth_inv` active — not probe-vs-probe. **First run: `max|diff|≈0.0156` on 4 of 6 shapes.**
Root cause: my `build_probe.py` compiled with `--use_fast_math`; the project's own `setup.py`
does not. Fast-math changes transcendental/FMA rounding enough to occasionally flip a quantization
decision. Removing the flag made every shape bit-exact — a reminder that "bit-exact" claims are
only as good as the compiler flags backing them, and this would have silently invalidated every
number in this document's benchmark section too (fixed by rebuilding without it; the 0.567 ms/step
result above already reflects the corrected build).

After the fix, both parts of `verify_vs_production.py` pass on all 6 shapes tested, WITH real
modulation: my stats+apply pipeline matches production bit-for-bit, and the full skip/catchup pair
matches two real production calls bit-for-bit.

### End-to-end correctness

[`scripts/patch_skip2.py`](scripts/patch_skip2.py) wires the validated kernels into the real
forward path, replacing only the "Step1" computation (`silu(gn(x))` → delta-quantize → `a_hat`
update); the conv/o_hat/residual code after it is untouched, since it only ever consumes the
returned code tensor. Scope: static delta mode, fp16, `C<=1024` and even (the K=1 chanmajor stats
path — excludes the two `C=1152/1536` decoder concat blocks, which correctly fall back to the
original kernel).

[`scripts/validate_e2e.py`](scripts/validate_e2e.py) runs a real generation twice with the same
seed, with and without the patch, and diffs the raw output tensor. First finding:
**`torch.backends.cudnn.benchmark = True` (set by `benchmark_ldm.py`) makes the UNPATCHED baseline
non-deterministic run-to-run** (60% of pixels differed between two unpatched runs at N=8/50 steps)
— a pre-existing property of this codebase's benchmark harness, unrelated to this patch. With
`cudnn.benchmark` forced off:

| check | result |
|---|---|
| unpatched run 1 == unpatched run 2 (N=4, 12 steps) | **bit-exact** |
| unpatched vs patched (N=4, 12 steps, 1320 patched calls, 168 in-scope-excluded fallbacks) | **bit-exact** |

At the full 50-step schedule the same `cudnn.benchmark`-independent noise reappeared (N=8, 50
steps: unpatched-vs-unpatched differed on 16.2% of pixels, max abs diff 34/255) even with
`cudnn.benchmark=False` — a SEPARATE, deeper nondeterminism source in the broader pipeline (likely
attention or another kernel elsewhere; not chased down, out of scope for this idea) that this
gate cannot fully control for at longer schedules. Patched-vs-unpatched showed the same order of
magnitude of difference (16.6%, max 34) as unpatched-vs-unpatched, and patched landed measurably
*closer* to the second unpatched run (8.3% differing, max 20) than the two unpatched runs were to
each other — consistent with (not a substitute for) the isolated kernel-level bit-exactness proof
above, which does not depend on this confounded signal.

**Quality check at real generation scale** (the cleanest signal given the pipeline's own noise
floor): `FID(baseline, skip2)` computed directly, both arms same seed:

| N, steps | FID(fp16_ref, baseline) | FID(fp16_ref, skip2) | FID(baseline, skip2) direct |
|---|--:|--:|--:|
| 16, 20 | 18.375 | 18.375 | **-0.000** |
| 200, 50 | 8.211 | 8.208 | **0.154** |

Both are statistically indistinguishable from zero at this scale — the patch produces generation
quality identical to the shipped kernel, exactly as the zero-error design predicts. Sample images
sent alongside this update.

### Timing: the kernel-level win did not survive the current Python integration

| N, steps | baseline ms/sample-step | skip2 ms/sample-step | delta |
|---|--:|--:|--:|
| 16, 20 (naive, fresh tensors every call) | 48.58 | 50.48 | **-1.90** |
| 16, 20 (after caching mean/inv_std/pending-code buffers on `self`) | 40.72 | 41.68 | **-0.96** |
| 200, 50 (same caching, full schedule) | 5.058 | 5.145 | **-0.088** |

The regression shrinks sharply as N grows (fixed per-call overhead amortizes better), but never
crosses into a net win. The reason: production's `group_norm_silu_delta_quantize_nhwc` computes
stats AND applies the quantize in ONE C++ host-function call; this patch calls `stats_launch` and
`probe_skip_launch`/`probe_catchup_launch` as TWO separate Python→pybind→CUDA round-trips per
layer per step. That extra Python/launch overhead, paid on every call, is larger than the ~0.567
ms/step the kernel-level economics alone would return over a 2-step window. The kernel-level
result in this document is not wrong — it is measured with zero Python between the two kernel
launches, a condition this monkeypatch cannot reproduce without becoming a real `csrc/` change.

### Verdict

**Correct and quality-neutral, confirmed at real generation scale — but not a net speed win as
integrated.** The math, the kernel, and the end-to-end quality are all validated to the same
standard this project holds every other shipped optimization to. What is missing is exactly what
the original verdict above anticipated: turning the measured 0.567 ms/step kernel-level result
into a real step-time win needs this built as a genuine fused C++ host function inside `csrc/`
(one call doing stats+skip-or-catchup, like `group_norm_silu_delta_quantize_nhwc` itself does),
not a Python-level dispatch across two separate extensions. That is strictly more engineering than
this session has invested so far, for a ceiling that was already known to be small (~0.7% of the
step). Recommend: hold here unless the project owner wants to fund the `csrc/` port specifically.

## Sweeping K: is 2 actually optimal?

Generalized the scheme from K=2 to arbitrary K. Within a window, position `p` (0-indexed)
reconstructs its reference from the checkpoint `A` plus `p` pending codes via `p` **sequential**
`__float2half_rn` round-trips — one per prior step, not a single round after summing all of them,
because fp16 rounding is not linear: the standard scheme rounds after every individual step, and
reproducing that exactly requires replaying the same chain, not a shortcut. This means the
reconstruction cost (extra code reads + sequential ALU rounds) grows with window position, and the
*last* step of a K-window pays for all K-1 of them — a real cost that grows faster than the
write savings as K increases. [`scripts/probe.cu`](scripts/probe.cu)'s `probe_window_step` kernel
implements this (K=1 degenerates exactly to the standard write-every-step kernel, giving a clean,
apples-to-apples K=1 baseline using the identical code path).

**Byte model** (per-step, W=2B fp16 a_hat, C=1B int8 code): extra reads over a K-window sum to
`K(K-1)/2 * C`, against `(K-1)` writes saved, giving a per-step estimate of
`(K-1)/K * (W - K*C/2)`. This is a downward parabola in K, predicting a small-K optimum and
eventually negative returns.

**Correctness first** ([`scripts/sweep_k.py`](scripts/sweep_k.py)): K ∈ {2,3,5,8} checked against
K real, chained calls to the real production kernel, 3 shapes each — **bit-exact on every
combination**, extending the K=2 proof to general K.

**Real measured sweep**, K ∈ {1,2,3,4,5,6,8,10,12,16}, same 5 dominant shapes, freq-weighted,
stats+apply cost together (a more complete per-step cost than the K=2 section's apply-kernel-only
number above, so the two aren't directly comparable in absolute terms — the K-to-K comparison
within this sweep is what matters):

| K | ms/step | vs K=1 |
|--:|--:|--:|
| 1 (baseline) | 0.6047 | — |
| 2 | 0.5260 | +13.0% |
| **3** | **0.5170** | **+14.5%** |
| 4 | 0.5254 | +13.1% |
| 5 | 0.5343 | +11.6% |
| 6 | 0.5466 | +9.6% |
| 8 | 0.5799 | +4.1% |
| 10 | 0.6158 | -1.8% |
| 12 | 0.6552 | -8.4% |
| 16 | 0.7382 | -22.1% |

Plot sent alongside this update. **Empirically optimal K=3**, matching the byte model's predicted
shape (peak at small K) closely, though the model alone would not have pinned down 2 vs 3 vs 4 —
they're within 1.5 percentage points of each other, effectively tied, while the model's write-vs-
read tradeoff explains WHY the curve has this shape and why it turns negative (K>=10 here, vs the
model's naive crossing near K=4 — the gap is the same write-allocate-favors-savings effect noted in
the K=2 section, which the linear byte model doesn't capture). The one shape that disagrees is the
tiny, launch-noise-dominated `768,2×2` reference (monotonically WORSE for all K>1) — consistent
with its behavior everywhere else in this investigation.

**This does not change the production-integration verdict above.** The sweep is measured the same
way as the K=2 kernel-level number: real CUDA kernels, no Python between the stats and apply calls.
The same Python/pybind-round-trip cost that erased K=2's 0.567 ms/step end-to-end would apply
here too, and K=3's added reconstruction complexity (a variable-length loop instead of a single
fixed lookup) makes an eventual `csrc/` port slightly more involved, not less. If this line is
ever pursued into production, K=3 (or K=2, given how close they are and K=2's simpler code) is the
window size to target, not larger.

### Files (this section)

- [`scripts/sweep_k.py`](scripts/sweep_k.py) — general-K correctness gate + timing sweep + plot
- [`data/k_sweep.json`](data/k_sweep.json), [`data/k_sweep.png`](data/k_sweep.png)

## Files

- [`scripts/verify_skip2_exact.py`](scripts/verify_skip2_exact.py) — Python-level exactness check
- [`scripts/probe.cu`](scripts/probe.cu), [`scripts/build_probe.py`](scripts/build_probe.py),
  [`scripts/verify_and_bench.py`](scripts/verify_and_bench.py) — real CUDA kernels, bit-exactness
  gate against the shipped kernel's logic, and the isolated-kernel timing benchmark
- [`scripts/verify_vs_production.py`](scripts/verify_vs_production.py) — bit-exactness against the
  REAL `modiff_cutlass` kernel, with modulation active; caught the `--use_fast_math` bug
- [`scripts/patch_skip2.py`](scripts/patch_skip2.py) — the production monkeypatch
- [`scripts/validate_e2e.py`](scripts/validate_e2e.py), `validate_e2e_full50.py` — end-to-end
  correctness gates (12-step bit-exact; 50-step confounded by pre-existing pipeline nondeterminism)
- [`scripts/generate_samples.py`](scripts/generate_samples.py) — real FID + timing measurement,
  N=16/20 and N=200/50, samples in `fid_run/{fp16_ref,int8_baseline,int8_skip2}/`

## Correction 2026-08-27: the production patch was wrong on real runs, and its gate could not see it

**Two defects, both in the Python integration, both now fixed and gated.** The kernel math and the
isolated-kernel benchmarks in this document stand; what follows invalidates the *end-to-end*
correctness claim above ("bit-exact", "FID(baseline, skip2) direct = -0.000") as stated, because
that claim was measured in a configuration that happened to be correct on the FIRST generation of a
process and diverged afterwards.

### Defect 1: the delta-REFRESH cadence was ignored

`OptimizedInt8Conv2d.forward_gn_fused_modiff` calls `_delta_gn_dynamic_args(x.device)` on **every**
step and passes its 8 trailing arguments to `group_norm_silu_delta_quantize_nhwc`. On steps where
`_delta_should_refresh()` is true -- `step_count` = 1, 1+R, 1+2R, ... for
`R = MODIFF_DELTA_REFRESH`, **default 4** -- that returns REAL absmax reduction buffers
(`_absmax_buf`, `_scale_buf`, `_inv_scale_buf`, `_retire_count`) and sets `_delta_seeded`; the
kernel then runs an extra reduction/publish pass. On the other steps it returns empty buffers and
the kernel skips that pass.

`probe_skip` / `probe_catchup` / `probe_window_step` take **no such arguments at all**, so they
implement only the non-refresh branch. **`sweep_k.py` and `verify_and_bench.py` could not detect
this**: both call production with all-empty buffers (`empty32, empty16, ..., 127.0, False, 1.0`),
i.e. they only ever compared against the non-refresh branch. That is why an isolated-kernel gate
reported bit-exact on 7 shapes and 4 values of K while a real generation diverges.

Measured, batch 4 / 12 steps / N=4 real samples, patched vs unpatched:

| `MODIFF_DELTA_REFRESH` | patched run1 vs run2 | run1 vs baseline | run2 vs baseline |
|---|--:|--:|--:|
| **4 (shipped default)** | **78/255 on 11.8% px** | 0/255 | **78/255 on 11.8% px** |
| 1000 (only step 1 refreshes) | 0/255 | 0/255 | 0/255 |

The unpatched pipeline is deterministic in this configuration -- 4 runs, all 6 pairs bit-exact -- so
these are real divergences, not harness noise.

**Fix:** refresh steps delegate to the original method, and the deferred-write window covers only
the run of non-refresh steps between two refresh steps, closing (writing a_hat) before the next
refresh step reads it. Consequence for the economics: with the default R=4 the longest usable
window is **3**, and `patch_skip2`'s K=2 pair now saves one a_hat write per 4 steps rather than one
per 2 -- so the ceiling measured in the sections above is **not** reachable at the shipped refresh
cadence without also implementing the refresh branch in the kernel.

### Defect 2: the gate never ran the patched path twice

`validate_e2e.py` runs the BASELINE twice (to establish pipeline determinism) but the PATCHED path
only ONCE. Defect 1 is invisible to that design: the first generation in a process matched
bit-exactly and the second did not (91/255 on 59.9% of pixels for the committed K=2 patch). Any
gate for a stateful patch has to re-run the patched arm, not just the reference arm.

**Fix:** [`gate_skipk_full.py`](scripts/gate_skipk_full.py) runs each arm twice and compares
run1-vs-run2, run1-vs-baseline and run2-vs-baseline, and fails if the patched call count is zero
(an earlier version of this gate reported BIT-EXACT on **0 patched calls**, because it drove the
model through `_setup_model` instead of `run_mode` and never reached the patched entry point --
the same vacuous-gate failure `gn_fast_reduce_2026-08-16` section 3 records).

### Post-fix gate

[`patch_skipk.py`](scripts/patch_skipk.py) (generalized K, one code path via `probe_window_step`)
and the fixed [`patch_skip2.py`](scripts/patch_skip2.py), at the shipped `MODIFF_DELTA_REFRESH=4`:

| K | run1 vs run2 | run1 vs baseline | run2 vs baseline | verdict |
|--:|--:|--:|--:|---|
| 2, 3, 4, 5, 6, 8 | 0/255 | 0/255 | 0/255 | **bit-exact and deterministic** |

660 patched calls and 330 refresh-delegated calls per run at every K. K=4, 5, 6 and 8 produce
**identical** results to K=3 because `K_eff = min(K, R-1) = 3` at the default refresh cadence --
so K>3 is unreachable without changing `MODIFF_DELTA_REFRESH`, independent of what the
isolated-kernel sweep's K curve says.

### Hypotheses ruled out by measurement, not argument

- **Another a_hat reader.** `OptimizedInt8Conv2d.forward` runs 140 times per generation, but its
  body contains no `a_hat_cache` access at all.
- **The table's alpha vs 1/scale.** Substituting `d_scale.reciprocal()` for `static_delta_alpha[i]`
  in the reconstruction left the divergence unchanged.
- **`act_q` varying with the cadence.** Instrumented: in static mode production passes
  `act_q=127.0, report=False, safety=1.0, a4=False` on both branches -- constant, and matching what
  the probe kernel assumes.
- **Windows not dividing the step count.** An earlier fix attempt tied the window to an independent
  counter; the set of failing K then *inverted* rather than shrinking, which is what ruled this out.

### Post-fix end-to-end timing, 5 repeats per arm

Every arm re-measured after the fix, in independent processes, **batch 128 / 200 DDIM steps /
128 samples** on the A40. `benchmark_ldm.py` warms up with a full schedule per mode before timing.
Median of 5; `spread` is (max-min)/median. Data: [`rebench_2026-08-27_b128.csv`](data/rebench_2026-08-27_b128.csv).

| arm | n | median ms/step | min | max | spread | vs fp32 | vs fp16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| fp32 | 5 | 2.350 | 2.35 | 2.35 | 0.0% | 1.00× | 0.35× |
| fp16 | 5 | 0.820 | 0.82 | 0.82 | 0.0% | 2.87× | 1.00× |
| **int8_baseline** | 5 | **0.580** | 0.58 | 0.58 | 0.0% | **4.05×** | **1.41×** |
| int8 + MoDiff | 5 | 0.660 | 0.66 | 0.67 | 1.5% | 3.56× | 1.24× |
| int8 + MoDiff + skipK, K=2 | 5 | 0.660 | 0.66 | 0.67 | 1.5% | 3.56× | 1.24× |
| int8 + MoDiff + skipK, K=3 | 5 | 0.670 | 0.67 | 0.67 | 0.0% | 3.51× | 1.22× |
| **int4_baseline** | 5 | **0.540** | 0.53 | 0.56 | 5.6% | **4.35×** | **1.52×** |
| int4 + MoDiff | 5 | 0.640 | 0.63 | 0.67 | 6.3% | 3.67× | 1.28× |

**The deferred-write patch is now exactly neutral, not a win**: K=2 lands on the same 0.660 ms/step
as unpatched MoDiff, and K=3 is 0.010 ms/step *worse*. That is the predicted consequence of the
fix -- with the shipped `MODIFF_DELTA_REFRESH=4` only one a_hat write in four can be deferred, so
the ~0.567 ms/step isolated-kernel saving has roughly a quarter of its former headroom, and what
remains is inside the Python round-trip cost the earlier sections already measured. The verdict of
this document is unchanged and now rests on correct arithmetic rather than a cadence coincidence:
**hold unless someone funds both the fused `csrc/` host function AND the refresh branch inside it.**

Two incidental notes for whoever reads the speed numbers:
- **int4_baseline overtakes int8_baseline at this batch** (1.52× vs 1.41×) but carries a 5.6-6.3%
  spread against int8's 0-1.5%, so the two int4 rows need more repeats before their gap to each
  other is claimed to better than ~0.05 ms/step.
- `int4_delta_calibration.pt` did not exist on this tree, so every earlier int4+MoDiff number ran
  the uncalibrated full-activation fallback. It was generated with the int8 protocol
  (`live absmax, STEPS=20/BATCH=4/ROUNDS=6`) and the row above uses it. Its acceptance statistic is
  worth recording on its own: **median tail step-gain 0.7×, range 0.1-2.9×** -- i.e. for most layers
  the int4 temporal delta needs a COARSER grid than the activation grid, not a finer one, where the
  int8 table's median gain is 12.45×. That is independent corroboration of A9/A23's finding that
  W4A4's dominant error is in the weights, not the activations.

### Correction to the table above: that benchmark cannot resolve this effect at all

The row "the deferred-write patch is now exactly neutral" **overstated what the data supports**, in
two ways that are worth recording because both are unit/resolution errors of the class this repo
keeps catching.

**Unit error.** `benchmark_ldm.py` reports `total_time / (num_gen * steps)`, i.e. ms per **sample**
per step. Every ms/step figure elsewhere in this document (the 0.5671 kernel saving, the 77.00 step,
ahat_overlap's 2.024 ceiling) is ms per **batch** step. At batch 128 those differ by 128×:

| | ms/batch-step | ms/sample-step |
|---|--:|--:|
| isolated-kernel saving, pre-fix | 0.5671 | 0.004430 |
| same, after the refresh fix (~half the headroom) | ~0.28 | ~0.002215 |
| printed precision of `Per-step:` | — | **0.01** |

So the pre-fix effect is **0.44 of one least-significant printed digit**, and the post-fix effect
**0.22 of one**. The rounded printout could not have shown it under any circumstances.

**Resolution error, which survives full precision.** Recomputing from the saved
`results.json` (full float, not the 2-decimal printout) gives the unpatched int8 arm a run-to-run
spread of **0.61% = 0.004060 ms/sample-step over 5 repeats** — **1.8× larger than the entire
post-fix effect (0.002215)**. The signal-to-noise ratio is below 1. More repeats shrink the
standard error of the median but do not change what is being measured: this instrument measures
sample throughput across processes, and the effect lives two orders of magnitude below its noise.

**Therefore the correct verdict on the timing is "uninformative", not "neutral".** The K=2/K=3
deltas measured here (+0.0009 and −0.0051 ms/sample-step) are inside ±0.004 of noise and should not
be read as either a win or a loss. (They were also computed on n=1 for the skipK arms: the sweep
script wrote all five repeats of each K to the same `--output_dir`, so `results.json` retained only
the last. Fixed and re-run with per-repeat directories.)

The same caveat applies to two other rows: `int4_baseline` and `int4 + MoDiff` carry 6.06% and 5.69%
spread, so **the "int4_baseline overtakes int8_baseline" ordering is not resolved by this
instrument** either — that claim needs the paired in-process instrument below, or many more repeats.

**The instrument that can resolve it** is the one this repo already uses for sub-1% effects: a
paired, in-process A/B timing the UNet step with CUDA events and asserting call counts per arm
(`integration/tests/ab_gn_fast_reduce.py` is the pattern; `gn_fast_reduce_2026-08-16` §4 explains why
the counters are what make it trustworthy). Cross-process sample-throughput timing is the wrong tool
for a 0.3% change, independent of repeat count.

#### The skipK arms re-run properly (n=5, distinct output dirs, full precision)

| arm | n | median ms/sample-step | min | max | spread | vs unpatched int8 |
|---|--:|--:|--:|--:|--:|--:|
| int8 + MoDiff (unpatched) | 5 | 0.664823 | 0.661521 | 0.665581 | 0.61% | — |
| + skipK K=2 | 5 | 0.663529 | 0.661292 | 0.665677 | 0.66% | +0.001294 (faster) |
| + skipK K=3 | 5 | 0.667628 | 0.666851 | 0.668705 | 0.28% | −0.002805 (slower) |

Worst observed run-to-run range among these three arms: **0.004386 ms/sample-step**, against a
post-fix effect of **~0.002215** → **SNR 0.51**. Both deltas are smaller than the noise of the arm
they are measured against, and they point in OPPOSITE directions, which is what a null looks like on
an instrument that cannot resolve the quantity. Data:
[`rebench_skipk_2026-08-27_n5.csv`](data/rebench_skipk_2026-08-27_n5.csv).

**Nothing about the patch's speed can be concluded from this benchmark** — not a win, not a loss,
not neutrality. The correctness result (bit-exact and deterministic at every K, at the shipped
refresh cadence) is what this session establishes; the speed question needs the paired in-process
CUDA-event instrument and is left open.

## 2026-08-27, kernel level at the REAL cadence: the scheme is a net LOSS, and a csrc port would not rescue it

The e2e benchmark cannot resolve this effect (previous section), so the question was re-asked with
the right instrument: [`cadence_kernel_bench.py`](scripts/cadence_kernel_bench.py), pure CUDA
kernels, no sampler, batch 128, 7 trials x 30 reps with rotated arm order, gated bit-exact on the
full cadence before any timing is reported.

**What changed versus `sweep_k.py`, and why it matters.** That sweep compared K windows of the probe
kernel against production called with **all-empty reduction buffers** -- production's non-refresh
branch -- and let *every* step be deferrable. Neither holds in production: one step in
R = `MODIFF_DELTA_REFRESH` (default 4) takes the refresh branch, and the 2026-08-27 fix must delegate
it and close the window before it. So the honest unit is **one R-step cadence, patched vs
unpatched**, which is what this measures. Three arms, all bit-exact against production over the full
cadence:

| arm | launches/windowed step | writes saved per 4 steps | freq-weighted ms/batch-step | vs unpatched |
|---|--:|--:|--:|--:|
| unpatched (production, fused) | 1 | 0 | 0.5225 | +0.00% |
| **K=1** (probe path, writes every step) | 2 | 0 | 0.5256 | **−0.59%** |
| K=2 | 2 | 1 | 0.5326 | **−1.93%** |
| K=3 | 2 | 2 | 0.5511 | **−5.48%** |

Consistent on every dominant shape (K=2 −1.8 to −2.3%, K=3 −5.4 to −6.4%); `768x2x2` is the usual
launch-bound outlier (−25%, −42%) and is excluded from the weighting as everywhere else in this
document.

**The K=1 arm is the load-bearing control.** It runs the probe path with the same two launches per
step but writes a_hat every step, so it isolates implementation overhead from the scheme:

* extra launch / non-fused stats pass: **−0.59%** — this is the part a real fused `csrc/` host
  function would recover.
* deferring 1 write per 4 steps (K=2): a **further −1.34 points**, with launch counts matched.
* deferring 2 writes per 4 steps (K=3): a **further −4.89 points**, matched.

So deferring writes is monotonically *worse*, and the cost is not launch overhead. Reconstructing
the reference costs more than the write it elides: at window position p the kernel reads p pending
code arrays and replays p sequential fp16 round-trips, and at the real cadence only 1-2 writes in 4
can be elided (versus every other step in the sweep's idealised window), so the byte model's
"2 B write saved vs 1 B code read added" no longer nets positive.

### Consequence: the recommendation changes from "hold" to "closed"

Earlier sections closed with *"hold unless the project owner wants to fund the `csrc/` port"*.
**That port would not rescue this.** Adding back the full 0.59% the non-fused stats pass costs still
leaves **K=2 at −1.34% and K=3 at −4.89%** against production, because those deltas are already
measured with launch counts matched. The +13.0%/+14.5% in the sweep above is real for the cadence it
measured -- a cadence with no refresh steps, which production does not run.

To make the scheme profitable, the refresh branch itself would have to be implemented inside the
windowed kernel so refresh steps stop being delegated and every step becomes deferrable. That is
strictly more work than the `csrc/` port that was already judged disproportionate, for a scheme
whose ceiling at the shipped cadence is now measured negative. **Recommend closing this line.**

The correctness work stands and is reusable: bit-exact by induction, gated at K in {2,3,4,5,6,8}
end to end and over the full cadence at kernel level, and the two gate defects it exposed
(cadence-blind probes, a gate that never re-ran the patched arm) are fixed.

Data: [`cadence_kernel_bench.json`](data/cadence_kernel_bench.json).

## 2026-08-27 SELF-CORRECTION: the "refresh cadence" mechanism in a397406 is WRONG

`_delta_gn_dynamic_args`'s **first statement** is a static-mode early return:

```python
if not self.delta_dynamic:
    e = self._empty_f32_arg(device)
    return e, e, e, e, 127.0, False, 1.0, False
```

So in the shipped **static** delta mode production ALWAYS passes empty reduction buffers, the
refresh branch is unreachable, and the probe kernels' argument pattern **does** match production on
every step. Confirmed in a real static-mode run: `_scale_buf` holds the same uninitialised value
(-0.0014323) at every step, i.e. the kernel never writes it. The mechanism story committed in
a397406 -- "production takes the refresh branch every R'th step and runs an extra reduction the
probes don't implement" -- is therefore **false for the configuration that was failing**.

**What that story was built on, and why it was not evidence.** One observation each of
`MODIFF_DELTA_REFRESH=4` (diverged) and `=1000` (bit-exact). The patched path was *later* measured to
be run-to-run unstable at that point (78/255 in one invocation, 0/255 in another), so a single draw
of an unstable quantity was used to identify a mechanism. That is the same n=1 error this repo keeps
recording, committed by me while documenting an instance of it.

**Consequences, stated rather than smoothed over:**

1. **The root cause of the original divergence is UNKNOWN again.** The fix in a397406 (delegate
   phase 0 and phase R-1) does make the gate pass -- bit-exact and deterministic at K in
   {2,3,4,5,6,8}, three comparisons each -- but it works by writing a_hat *more often* (3 of every 4
   steps at K=2, versus 2 of 4 before), so it may be **masking** the defect rather than removing it.
   "Fixed" should be read as **"no longer reproduces under this gate"**, not as a root-cause fix.
2. **The kernel-level cadence benchmark's `unpatched` arm is not representative**: its phase 0 issues
   a refresh-args call that static mode never makes, worth ~11.2% of that one step in four. Both
   arms share it, so the **K=1 vs K=2 vs K=3 comparison is unaffected** and its conclusion stands
   (deferring writes is monotonically worse with launch counts matched: -1.34 and -4.89 points).
   The absolute "vs unpatched" column should be re-run with plain args on all four steps.
3. What is unaffected: the gate improvements (re-run the patched arm; assert non-zero patched calls),
   the e2e resolution analysis (instrument noise 1.8x the effect), and the retraction in 546612a.

**The one thing worth measuring next, since it is now cheap and interesting on its own:** the
refresh-args call path measured **+11.23% on the GN-fused kernel** (0.0586 ms/call freq-weighted, so
~1.11 ms/batch-step across the 19 dominant calls, ~1.3% of the step) and it is **not** free -- with
real buffers the kernel *ignores the passed scale and quantizes with its own measured absmax*
(verified bit-exact: passing the measured 17.7388 with empty buffers reproduces the refresh arm
exactly, against the 50.0 that was passed). That is a genuine property of the DYNAMIC path worth
knowing: on refresh steps the quantize scale is measured, and whether the conv's dequant alpha
follows it is a question the dynamic path's own bookkeeping has to answer.
