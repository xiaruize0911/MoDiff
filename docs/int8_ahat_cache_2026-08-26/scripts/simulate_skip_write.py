"""Does skipping a_hat's WRITE for K-1 out of every K steps (freezing a_hat, letting delta widen
against a stale reference, and correcting fully at the refresh step) blow up code clipping the way
C7's warmup-rounds finding suggests it might?

Mechanism, precisely: this reuses the ALREADY-BUILT and ALREADY-MEASURED a_hat write-elision
kernel variant (ahat_overlap_2026-08-26's w0c1 probe) for K-1 out of every K calls -- read x, read
a_hat, quantize a fresh code every step (so o_hat / conv still updates every step, unlike a full
layer-skip scheme), but skip the a_hat WRITE. On the Kth call, do a full write (today's behaviour).
Ceiling: (K-1)/K x the measured write-elision saving (2.024 ms/step W8A8, 1.742 W4A4).

The numerics risk: a_hat's reference goes stale for up to K-1 steps, so the delta being quantized
against it widens over the window -- exactly the mechanism C7 already measured to hurt FID by
+13.08% in a related (but not identical) context (fewer WARM-UP reconstruction rounds at t=T,
where deltas start large anyway; this tests PERIODIC skipping mid-schedule, where deltas are
normally small and steady). This script checks the cheap, informative proxy -- how much does the
delta magnitude / code clipping rate grow within a skip window, using the REAL measured
delta-scale trajectory -- before recommending (or not) an actual FID run.
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


def make_true_trajectory(n, scale_traj, level=1.0, target_code_mag=20.0):
    """FIXED 2026-08-26: the first version used a CONSTANT step_scale=0.05 regardless of the
    calibrated delta_scale trajectory. Real calibration data shows K=1 clips 0.0% of the time
    (obs_clipped_frac=0.0 for every real layer) -- i.e. the true per-step delta shrinks roughly in
    step with the growing scale, by construction (that IS what the scale table is calibrated
    against). A constant-variance random walk does NOT shrink, so it clipped 32.5% of the time at
    K=1 even in the earlier (buggy) run -- an artifact of an unrealistic proxy trajectory, not a
    real property of the deployed system. This version scales each step's innovation so that
    delta*scale stays near `target_code_mag` (well under the 127 ceiling) throughout, matching the
    real system's near-zero K=1 clip rate."""
    x = np.zeros(n)
    x[0] = level
    for t in range(1, n):
        step_scale = target_code_mag / scale_traj[t]
        x[t] = x[t - 1] + np.random.randn() * step_scale
    return x


def simulate_skip(scale_traj, true_traj, K):
    """a_hat is written only every K steps; between writes it is FROZEN, and each step's code is
    quantized against that frozen reference (so the effective delta widens across the window)."""
    a_hat_written = 0.0     # what's actually in DRAM -- only changes on refresh steps
    codes, clipped, deltas = [], [], []
    for t, s in enumerate(scale_traj):
        delta = true_traj[t] - a_hat_written     # against the STALE (or fresh, if t%K==0) reference
        q = np.clip(round(delta * s), -QLEVEL, QLEVEL)
        codes.append(q)
        clipped.append(abs(delta * s) > QLEVEL)
        deltas.append(abs(delta))
        if (t + 1) % K == 0:
            # refresh: write the exact accumulated value back to a_hat -- a_hat_written plus every
            # code since the last refresh, dequantized and summed. No special case for the first
            # window: a_hat_written legitimately starts at 0.0 (a_hat_cache's real initial value),
            # and the codes computed against that 0.0 reference in steps 0..K-1 are EXACTLY what
            # o_hat received, so they must be the same codes a_hat's reconstruction sums -- a
            # special case that instead reset a_hat_written to true_traj[0] here discarded that
            # history, desynchronising a_hat from o_hat from step 0 onward. That was the bug.
            window_start = t + 1 - K
            acc = 0.0
            for j in range(window_start, t + 1):
                acc += codes[j] / scale_traj[j]
            a_hat_written = a_hat_written + acc
    return np.array(codes), np.array(clipped), np.array(deltas)


print(f"{'layer':>24} {'K':>4}  {'clip rate':>10} {'clip rate K=1':>14} {'max |delta|':>12} "
      f"{'max delta K=1':>14} {'ceiling ms/step (W8A8)':>22}")
for name, row in LAYERS.items():
    scale_traj = scale_trajectory(row["delta_scale_step0"], row["delta_scale_tail"], STEPS)
    true_traj = make_true_trajectory(STEPS, scale_traj)
    _, clip1, delta1 = simulate_skip(scale_traj, true_traj, 1)   # K=1 baseline: today's behaviour
    for K in (1, 2, 4, 8, 16):
        _, clipK, deltaK = simulate_skip(scale_traj, true_traj, K)
        ceiling = (K - 1) / K * 2.024
        print(f"{name:>24} {K:>4}  {100*clipK.mean():>9.3f}% {100*clip1.mean():>13.3f}% "
              f"{deltaK.max():>12.4f} {delta1.max():>14.4f} {ceiling:>21.3f} ms/step")
    print()

print("clip rate = fraction of the 200 steps whose code saturates at +-127 (a HARD floor on")
print("quantization error for that step, beyond what the scale table was sized to avoid). If K>1's")
print("clip rate is close to K=1's, the widening within a window is not enough to matter for THIS")
print("synthetic trajectory; if it grows sharply with K, that is the C7-style risk showing up")
print("quantitatively, and argues for a small K (e.g. 2-4) or an FID check before going further.")
