"""Gate for step 1 of the L1 fusion plan: the static per-step delta table for the wxax projections.

Four things, and the third is the one worth the file.

  1. CALIBRATION FILLS IT. begin -> modulated steps -> end produces a per-step table, and export/apply
     round-trips it exactly.
  2. THE REDUCTION STOPS. With a table loaded, `delta_absmax_fp16` must not be called at all -- that call
     is both the cost and the thing that blocks fusing GN+delta-quantize into one kernel, since a global
     absmax cannot be fused with its consumer.
  3. IT IS DETERMINISTIC, and the arm without it is NOT. docs/OPEN_ITEMS.md A18 records L1 as
     run-to-run nondeterministic at 4.5-6.2/255 while every L0 arm is bit-exact, cause unknown, with the
     per-call reduction as the prime suspect (its `_retire` argument is the signature of a
     last-block-retires grid reduction, which is order-dependent). This tests that hypothesis at unit
     scale instead of by 8-image sampling: dynamic must differ across repeats, static must not.
  4. THE NUMBERS ARE THE PRODUCTION ONES. The table is scored against the absmax the DYNAMIC path
     actually wrote, not against a second estimate of it -- so a table that is self-consistent but
     describes the wrong quantity fails.

Run: python integration/tests/test_wxax_delta_table.py
"""
import os
import sys

import torch
import torch.nn as nn

import modiff_cutlass as mc
from integration.kernels import wxax_linear as W

DEV = "cuda"
torch.manual_seed(20260817)
M, K, N, STEPS = 1024, 192, 768, 6


def build():
    lin = nn.Linear(K, N).to(DEV).half()
    q = W.QuantLinearWxAx(lin, 4, modiff=True).to(DEV)
    q.set_a_scale(12.34)
    return q


def drive(q, xs):
    """Seed on the first tensor, then run the rest as modulated steps. Returns the outputs."""
    q.reset_modiff()
    outs = []
    for x in xs:
        outs.append(q(x).float().clone())
    return outs


xs = [torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.5 for _ in range(STEPS)]
fails = []

# ---- 1. calibration fills the table, and export/apply round-trips ---------------------------------
q = build()
model = nn.Sequential(q)
n_armed = W.begin_wxax_delta_calibration(model)
drive(q, xs)
n_set = W.end_wxax_delta_calibration(model)
table = W.export_wxax_delta_scales(model)
layers = [k for k in table if not k.startswith("__")]
nz = int((q.static_delta_scale != 0).sum())
print(f"  1. armed {n_armed}, calibrated {n_set}, table entries non-zero: {nz}/{W.MODIFF_MAX_STEPS}, "
      f"exported layers {len(layers)}, clip ratio in artifact: "
      f"{float(table['__clip_ratio__']) if '__clip_ratio__' in table else None}")
if not (n_armed == 1 and n_set == 1 and nz == W.MODIFF_MAX_STEPS and len(layers) == 1):
    fails.append(f"calibration did not produce a full table (armed {n_armed}, set {n_set}, nz {nz}, "
                 f"layers {len(layers)})")
if "__clip_ratio__" not in table:
    fails.append("the artifact does not record the clip ratio its values were baked at, so a later "
                 "change to the default silently reinterprets it")

# Round-trip the WHOLE artifact, which is how it is used -- and the ratio travelling with it is what
# makes this identity hold across a change to LINEAR_DELTA_CLIP_RATIO.
q2 = build()
m2 = nn.Sequential(q2)
n_load = W.apply_wxax_delta_scales(m2, table)
same_table = bool(torch.equal(q2.static_delta_scale.cpu(), q.static_delta_scale.cpu()))
print(f"  1b. apply loaded {n_load}, table byte-identical after round-trip: {same_table}")
if not (n_load == 1 and same_table):
    fails.append(f"export/apply did not round-trip (loaded {n_load}, identical {same_table})")

# And the ratio is honoured rather than ignored: asking for 2x the baked ratio must double the scales.
q5 = build()
os.environ["MODIFF_LINEAR_DELTA_TABLE_RATIO"] = str(2.0 * float(table["__clip_ratio__"]))
W.apply_wxax_delta_scales(nn.Sequential(q5), table)
del os.environ["MODIFF_LINEAR_DELTA_TABLE_RATIO"]
got = float(q5.static_delta_scale[0]) / max(float(q.static_delta_scale[0]), 1e-30)
print(f"  1c. requesting 2x the baked ratio scaled the table by {got:.4f} (want 2.0)")
if abs(got - 2.0) > 1e-3:
    fails.append(f"the requested ratio was not applied (scaled by {got:.4f}, want 2.0)")

# ---- 2. the per-call reduction stops being called -------------------------------------------------
calls = {"n": 0}
orig = mc.delta_absmax_fp16


def counted(*a, **k):
    calls["n"] += 1
    return orig(*a, **k)


mc.delta_absmax_fp16 = counted
W.mc = getattr(W, "mc", None)      # wxax_linear imports it as _mc; patch that binding too
W._mc.delta_absmax_fp16 = counted
try:
    calls["n"] = 0
    drive(q2, xs)                                   # table loaded
    with_table = calls["n"]
    q3 = build()                                    # no table
    calls["n"] = 0
    drive(q3, xs)
    without_table = calls["n"]
finally:
    mc.delta_absmax_fp16 = orig
    W._mc.delta_absmax_fp16 = orig
print(f"  2. delta_absmax_fp16 calls: {with_table} with the table, {without_table} without")
if with_table != 0:
    fails.append(f"the table is loaded but the reduction still ran {with_table} times -- fusion stays "
                 f"blocked and the cost stays")
if without_table == 0:
    fails.append("the reduction never ran even WITHOUT a table, so check 2 proves nothing")

# ---- 3. A18: static must be deterministic, dynamic must not be ------------------------------------
def spread(q_, reps=3):
    runs = [drive(q_, xs) for _ in range(reps)]
    worst = 0.0
    for i in range(1, reps):
        for a, b in zip(runs[0], runs[i]):
            worst = max(worst, float((a - b).abs().max()))
    return worst


s_static = spread(q2)
s_dynamic = spread(build())
print(f"  3. max |Δ| over 3 repeats: static table {s_static:.3e}, per-call reduction {s_dynamic:.3e}")
if s_static != 0.0:
    fails.append(f"WITH the table the output still varies ({s_static:.3e}) -- A18's cause is not (only) "
                 f"the per-call reduction, and this step does not fix it")
if s_dynamic == 0.0:
    print("     NOTE: the dynamic arm was also deterministic here, so this unit test does not reproduce")
    print("     A18 at this size. Check 3 is then only evidence that the table did not INTRODUCE noise.")

# ---- 4. the table describes the quantity the dynamic path used ------------------------------------
q4 = build()
m4 = nn.Sequential(q4)
W.begin_wxax_delta_calibration(m4)
drive(q4, xs)
obs = q4._delta_absmax_obs.detach().cpu().clone() if q4._delta_absmax_obs is not None else None
W.end_wxax_delta_calibration(m4)
if obs is None or float(obs.max()) <= 0:
    fails.append("nothing was observed during calibration -- the table cannot describe production")
else:
    # scale = Q / (absmax * safety / ratio), inverted on the steps that were actually seen
    i = int(torch.nonzero(obs > 0)[0])
    want = float(q4.Q) / (float(obs[i]) * 1.02 / W.LINEAR_DELTA_CLIP_RATIO)
    got = float(q4.static_delta_scale[i])
    rel = abs(got - want) / max(want, 1e-30)
    print(f"  4. step {i}: table {got:.4f} vs formula on the observed absmax {want:.4f} (rel {rel:.2e})")
    if rel > 0.35:      # 3-wide smoothing can raise a step's absmax to a neighbour's
        fails.append(f"table entry {got:.4f} does not follow from the observed absmax {want:.4f}")

print()
if fails:
    print("GATE FAILED:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("GATE PASSED: the projections have a static per-step delta table, the per-call reduction is gone, "
      "and\nthe table follows from the absmax the dynamic path measured.")
