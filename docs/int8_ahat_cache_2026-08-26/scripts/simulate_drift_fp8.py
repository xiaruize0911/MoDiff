"""fp8 a_hat: does a FLOATING exponent (unlike C15's flat int8 grid) survive the same real
per-step delta-scale swing that killed the fixed-point idea?

C15 found a flat int8 grid drifts 14.5x-7137x the code's own quantum by the tail of the schedule,
because a fixed LSB cannot track delta_scale's real 12-125x swing across the 200-step schedule
(docs/modiff_correctness_2026-08-03/data/delta_calibration.json, 70 real layers). fp8 (torch's
native float8_e4m3fn / float8_e5m2) has a FLOATING exponent, same mechanism fp16 already uses to
adapt to a_hat's current magnitude automatically -- the question this asks is whether that
adaptation is enough to avoid the catastrophic tail behaviour, at the cost of coarser mantissa
(3 bits for e4m3, 2 for e5m2, vs fp16's 10).

Same harness as simulate_drift.py: real step0/tail delta-scale endpoints, log-interpolated across
200 steps, against a synthetic AR(1) "true activation" trajectory. Storage variants compared:
fp16 (shipped), int8-fixed (C15, for reference), float8_e4m3fn, float8_e5m2 -- via torch's native
dtypes, so the round-trip precision is exactly what real fp8 hardware/software would produce, not
a hand-rolled approximation.
"""
import json

import numpy as np
import torch

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


def store(value, kind, lsb=None):
    if kind == "fp16":
        return float(np.float16(value))
    if kind == "int8":
        code = np.clip(np.round(value / lsb), -128, 127)
        return float(code * lsb)
    if kind in ("e4m3", "e5m2"):
        dt = torch.float8_e4m3fn if kind == "e4m3" else torch.float8_e5m2
        return float(torch.tensor(value, dtype=torch.float32).to(dt).float())
    raise ValueError(kind)


def simulate(scale_traj, true_traj, kind, lsb=None):
    a_hat_stored = 0.0
    o_hat = 0.0
    drift = []
    for t in range(len(true_traj)):
        s = scale_traj[t]
        delta = true_traj[t] - a_hat_stored
        q = np.clip(round(delta * s), -QLEVEL, QLEVEL)
        a_hat_true = a_hat_stored + q / s
        o_hat += q / s
        a_hat_stored = store(a_hat_true, kind, lsb)
        drift.append(a_hat_stored - o_hat)
    return np.array(drift)


def make_true_trajectory(n, level=1.0, corr=0.98, step_scale=0.05):
    x = np.zeros(n)
    x[0] = level
    for t in range(1, n):
        x[t] = level + corr * (x[t - 1] - level) + np.random.randn() * step_scale
    return x


print(f"{'layer':>24} {'storage':>8}  {'max |a_hat-o_hat|':>18} {'final |drift|':>14} "
      f"{'drift/step0-q':>14} {'drift/tail-q':>13}")
for name, row in LAYERS.items():
    scale_traj = scale_trajectory(row["delta_scale_step0"], row["delta_scale_tail"], STEPS)
    true_traj = make_true_trajectory(STEPS)
    q0, q1 = 1.0 / scale_traj[0], 1.0 / scale_traj[-1]

    d_fp16 = simulate(scale_traj, true_traj, "fp16")
    print(f"{name:>24} {'fp16':>8}  {np.abs(d_fp16).max():>18.3e} {abs(d_fp16[-1]):>14.3e} "
          f"{np.abs(d_fp16).max()/q0:>13.2f}x {np.abs(d_fp16).max()/q1:>12.1f}x")

    a_range = np.abs(true_traj).max() * 1.2
    lsb8 = (2 * a_range) / 255
    d_int8 = simulate(scale_traj, true_traj, "int8", lsb=lsb8)
    print(f"{name:>24} {'int8':>8}  {np.abs(d_int8).max():>18.3e} {abs(d_int8[-1]):>14.3e} "
          f"{np.abs(d_int8).max()/q0:>13.2f}x {np.abs(d_int8).max()/q1:>12.1f}x")

    for kind in ("e4m3", "e5m2"):
        d_fp8 = simulate(scale_traj, true_traj, kind)
        print(f"{name:>24} {kind:>8}  {np.abs(d_fp8).max():>18.3e} {abs(d_fp8[-1]):>14.3e} "
              f"{np.abs(d_fp8).max()/q0:>13.2f}x {np.abs(d_fp8).max()/q1:>12.1f}x")
    print()

print("'drift/step0-q' and 'drift/tail-q' are the storage drift as a multiple of the code's own")
print("quantum (1/scale) at that point in the schedule -- the resolution the scheme already")
print("tolerates every step by design (error feedback). fp16 and fp8 should both stay near a few x")
print("throughout, because their floating exponent tracks a_hat's actual magnitude the same way")
print("at step 0 and at the tail; int8-fixed should reproduce C15's blow-up at the tail for the")
print("high-gain layer specifically, since nothing here has changed about WHY that happens.")
