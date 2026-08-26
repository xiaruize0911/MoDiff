"""Does an int8 fixed-point a_hat cache preserve the o_hat = conv(a_hat) invariant?

CORRECTED framing. A previous verbal proposal in this session claimed a_hat could be stored as an
integer at grid spacing 1/scale with ZERO extra rounding, because the update is always
`a_hat += q/scale` (an exact integer step). That is only true if `scale` is held CONSTANT for the
whole 200-step schedule. It is not: `int8_optimized.py:_delta_scale_args` reads a PER-STEP table
(`self.static_delta_scale[i]`, i = the DDIM step index), and real calibration data on this exact
checkpoint already shows why that matters --
docs/modiff_correctness_2026-08-03/data/delta_calibration.json, 70 real layers:

    step_gain_tail = delta_scale_tail / delta_scale_step0
    min 0.25x, median 12.45x, max 124.51x

The scale swings by an ORDER OF MAGNITUDE across the schedule for a typical layer. A FIXED int8
grid for a_hat cannot simultaneously match "1/scale" at both ends, so re-expressing a_hat onto that
grid after every update requires an EXTRA rounding step beyond the code's own q/scale increment --
and by linearity of convolution, that extra term does NOT cancel: it accumulates in
`conv(a_hat) - o_hat` as a running sum of `conv(eps_t)`, where eps_t is step t's storage-rounding
residual. Whether this is a quality problem (FID) is separate from whether it is an INVARIANT
problem (does o_hat still equal conv(a_hat)) -- this script measures the invariant question only,
with a 1D linear proxy for conv (conv is linear, so this transfers): `o_hat` tracks
`sum_t(q_t/scale_t)` exactly by construction; the question is how far a STORED (rounded) a_hat
trajectory drifts from that same running sum.

METHOD: no CUDA. Simulate the a_hat update recursion in pure Python/NumPy, for two storage
schemes, against a SLOWLY-VARYING synthetic "true" trajectory (an AR(1) walk, because a_hat's whole
premise is that consecutive diffusion steps are correlated -- iid noise would make every delta as
large as the signal, which is not what the calibration data's smoothly-changing scale implies) and
the REAL measured scale endpoints, log-interpolated across steps (the calibration file has only
step0/tail, not all 200 points -- log-interpolation is the natural curve for a quantity that is
itself a ratio of exponentially-varying magnitudes, but this IS an approximation, flagged below).

    (A) fp16 storage  -- the shipped scheme (baseline for comparison, not the subject of this test)
    (B) int8 fixed-point storage -- the proposal, LSB chosen from the a_hat value range actually
        produced by the simulated trajectory (not from the delta scale, which is a different
        quantity -- a_hat tracks the ACTIVATION's absolute level, not its per-step increment)
"""
import json
import math

import numpy as np

np.random.seed(0)
STEPS = 200
QLEVEL = 127.0   # int8 delta code datapath

with open("/workspace/MoDiff/docs/modiff_correctness_2026-08-03/data/delta_calibration.json") as f:
    CAL = json.load(f)["report"]

LAYERS = {
    "median gain (12.45x)": next(r for r in CAL if r["layer"] == "input_blocks.5.0.out_conv"),
    "max gain (124.5x)": next(r for r in CAL if r["layer"] == "input_blocks.3.0.in_conv"),
}


def scale_trajectory(step0, tail, steps):
    """Log-interpolate between the two measured endpoints. APPROXIMATION -- flagged in the
    module docstring; the calibration file does not record the intermediate 198 points."""
    t = np.linspace(0, 1, steps)
    return step0 * (tail / step0) ** t


def simulate(scale_traj, true_traj, storage="fp16", ahat_bits=8, ahat_range=None):
    """true_traj: the sequence of 'true' continuous activation values (what out_t would be).
    Returns (a_hat_stored trajectory, o_hat trajectory = exact running sum, drift = their gap)."""
    n = len(true_traj)
    a_hat_true = 0.0          # the exact a_hat + q/scale recursion, no storage rounding at all
    a_hat_stored = 0.0        # what is actually read back and used next step
    o_hat = 0.0               # exact running sum of dequantized codes -- what conv(a_hat) SHOULD equal
    a_hat_series, o_hat_series, drift = [], [], []

    if storage == "int8":
        lsb = (2 * ahat_range) / (2 ** ahat_bits - 1)

    for t in range(n):
        s = scale_traj[t]
        delta = true_traj[t] - a_hat_stored
        q = np.clip(np.round(delta * s), -QLEVEL, QLEVEL)
        a_hat_true = a_hat_stored + q / s
        o_hat += q / s                      # exact, matches the real conv2d_..._o_hat kernel

        if storage == "fp16":
            # np.float16 round-trip -- the ACTUAL storage precision the shipped kernel uses.
            a_hat_stored = float(np.float16(a_hat_true))
        elif storage == "int8":
            code = np.clip(np.round(a_hat_true / lsb), -128, 127)
            a_hat_stored = float(code * lsb)
        else:
            raise ValueError(storage)

        a_hat_series.append(a_hat_stored)
        o_hat_series.append(o_hat)
        drift.append(a_hat_stored - o_hat)
    return np.array(a_hat_series), np.array(o_hat_series), np.array(drift)


def make_true_trajectory(n, level=1.0, corr=0.98, step_scale=0.05):
    """AR(1) walk around `level`, mimicking a slowly-drifting silu(gn(x)) value across DDIM steps.
    corr close to 1 => small per-step deltas relative to the signal, consistent with the
    calibration data's tail scale being LARGER than step0's (finer resolution needed as deltas
    shrink)."""
    x = np.zeros(n)
    x[0] = level
    for t in range(1, n):
        x[t] = level + corr * (x[t - 1] - level) + np.random.randn() * step_scale
    return x


print(f"{'layer':>24} {'storage':>8} {'a_hat LSB':>10}  {'max |a_hat-o_hat|':>18} "
      f"{'final |drift|':>14} {'max |true delta|':>16}")
for name, row in LAYERS.items():
    scale_traj = scale_trajectory(row["delta_scale_step0"], row["delta_scale_tail"], STEPS)
    true_traj = make_true_trajectory(STEPS)

    _, _, drift_fp16 = simulate(scale_traj, true_traj, storage="fp16")
    print(f"{name:>24} {'fp16':>8} {'--':>10}  {np.abs(drift_fp16).max():>18.3e} "
          f"{abs(drift_fp16[-1]):>14.3e} {np.abs(np.diff(true_traj)).max():>16.4f}")

    a_range = np.abs(true_traj).max() * 1.2   # 20% headroom over the observed range
    for bits in (8,):
        _, _, drift_int8 = simulate(scale_traj, true_traj, storage="int8", ahat_bits=bits,
                                    ahat_range=a_range)
        lsb = (2 * a_range) / (2 ** bits - 1)
        print(f"{name:>24} {'int'+str(bits):>8} {lsb:>10.4f}  {np.abs(drift_int8).max():>18.3e} "
              f"{abs(drift_int8[-1]):>14.3e}")
        q_step0, q_tail = 1.0 / scale_traj[0], 1.0 / scale_traj[-1]
        print(f"{'':>24} {'':>8} {'':>10}   drift/step0-quantum: {np.abs(drift_int8).max()/q_step0:6.2f}x"
              f"    drift/tail-quantum: {np.abs(drift_int8).max()/q_tail:8.1f}x   "
              f"(quanta: {q_step0:.4f} -> {q_tail:.6f})")
    print()

print("Interpretation: fp16's drift is the EXISTING baseline (already present in the shipped")
print("kernel, from fp16 rounding on every store) -- not zero, but tiny relative to a_hat's own")
print("range. int8's drift is the NEW quantity this proposal would introduce. If int8's max |drift|")
print("is comparable to or smaller than a single delta code's worth of value (1/scale at that")
print("step), the invariant break is within the same order of magnitude as noise the scheme")
print("already tolerates by design (error feedback); if it is much larger, the fixed grid is too")
print("coarse for this layer's dynamic range and the idea needs a wider LSB (fewer effective bits")
print("of headroom) or per-layer/per-phase recalibration, not a flat 8 bits everywhere.")
