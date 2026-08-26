"""Now that small error is acceptable: does a K-step-refreshed int8 a_hat grid keep drift bounded?

C15 found a FLAT int8 grid drifts 14.5x-7137x the code's own quantum by the tail, because the
grid's LSB never changes while delta_scale swings 12-125x across the schedule. The fp8 alternative
(simulate_drift_fp8.py) is WORSE, not better -- it spends bits on an exponent a_hat's bounded
magnitude does not need, leaving fewer effective mantissa bits (LSB ~0.125 at magnitude 1.0 for
e4m3, vs ~0.0094 for a dedicated 8-bit fixed range).

This tests the mechanism that actually targets the real problem: re-derive a_hat's OWN int8 LSB
every K steps to track delta_scale's CURRENT value (K = MODIFF_DELTA_REFRESH's existing cadence,
default 4, already a real knob in this codebase for a different scale). Between refreshes, the
grid is flat (same drift mechanism as C15); AT each refresh, a_hat is re-quantized onto a NEW grid
matching that window's scale, bounding how far the drift can grow before it's reset.
"""
import json

import numpy as np

np.random.seed(0)
STEPS = 200
QLEVEL = 127.0

with open("/workspace/MoDiff/docs/modiff_correctness_2026-08-03/data/delta_calibration.json") as f:
    CAL = json.load(f)["report"]

LAYERS = {
    "median gain (12.45x)": next(r for r in CAL if r["layer"] == "input_blocks.5.0.out_conv"),
    "max gain (124.5x)": next(r for r in CAL if r["layer"] == "input_blocks.3.0.in_conv"),
}


def scale_trajectory(step0, tail, steps):
    t = np.linspace(0, 1, steps)
    return step0 * (tail / step0) ** t


def make_true_trajectory(n, level=1.0, corr=0.98, step_scale=0.05):
    x = np.zeros(n)
    x[0] = level
    for t in range(1, n):
        x[t] = level + corr * (x[t - 1] - level) + np.random.randn() * step_scale
    return x


def simulate_refreshed(scale_traj, true_traj, K, headroom=8.0):
    """headroom: the grid at each refresh spans +-headroom/scale_at_refresh, i.e. the grid can
    represent deltas up to `headroom` code-widths beyond a single step's quantum before clipping
    -- a stand-in for "a_hat's own value range relative to one step's delta," not a free parameter
    tuned to make the result look good (see the note on this in the module's Scope section)."""
    a_hat_stored = 0.0
    o_hat = 0.0
    drift = []
    lsb = None
    for t in range(len(true_traj)):
        if t % K == 0:
            lsb = headroom / scale_traj[t] / 128.0   # re-derive the grid to track THIS window's scale
            a_hat_stored = float(np.clip(np.round(a_hat_stored / lsb), -128, 127) * lsb)
        s = scale_traj[t]
        delta = true_traj[t] - a_hat_stored
        q = np.clip(round(delta * s), -QLEVEL, QLEVEL)
        a_hat_true = a_hat_stored + q / s
        o_hat += q / s
        a_hat_stored = float(np.clip(np.round(a_hat_true / lsb), -128, 127) * lsb)
        drift.append(a_hat_stored - o_hat)
    return np.array(drift)


print(f"{'layer':>24} {'K':>4}  {'max |a_hat-o_hat|':>18} {'final |drift|':>14} "
      f"{'drift/step0-q':>14} {'drift/tail-q':>13}")
for name, row in LAYERS.items():
    scale_traj = scale_trajectory(row["delta_scale_step0"], row["delta_scale_tail"], STEPS)
    q0, q1 = 1.0 / scale_traj[0], 1.0 / scale_traj[-1]
    for K in (1, 4, 8, 20):
        true_traj = make_true_trajectory(STEPS)
        d = simulate_refreshed(scale_traj, true_traj, K)
        print(f"{name:>24} {K:>4}  {np.abs(d).max():>18.3e} {abs(d[-1]):>14.3e} "
              f"{np.abs(d).max()/q0:>13.2f}x {np.abs(d).max()/q1:>12.1f}x")
    print()

print("K=1 refreshes every step (the ideal, upper bound on quality -- but pays a resync every")
print("step, which is most of the point of doing this at all). Compare K=4/8/20's drift/tail-q")
print("against C15's UNREFRESHED result (14.5x-7137x) to see how much a periodic resync recovers,")
print("and against fp16's own baseline drift (~1x, from simulate_drift_fp8.py) to judge whether")
print("the residual is now in the same regime the scheme already tolerates.")
