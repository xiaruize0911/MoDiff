"""Numerically verify: is the K=2 "deferred write, exact reconstruction" a_hat scheme bit-identical
to the standard (write-every-step) scheme?

Design being checked (see docs/ahat_skip2_exact_2026-08-26/FINDINGS.md for the derivation):
  step 1 (odd, "skip"):   c1 = Q(v1 - A)              ; a_hat's DRAM buffer NOT written
  step 2 (even, "catchup"): a1_hat = A + D(c1)          ; RECONSTRUCTED, not read from a_hat's own
                                                           buffer -- computed from the checkpoint A
                                                           (still in DRAM, unwritten since step 1)
                                                           plus c1 (read back from the code buffer)
                            c2 = Q(v2 - a1_hat)
                            A' = a1_hat + D(c2)         ; this IS written -- new checkpoint

Claim: because a1_hat is computed with the EXACT SAME formula and EXACT SAME inputs (A, c1) that
the standard scheme would use when it writes a_hat after step 1, every downstream quantity (c2,
A', and o_hat at every step) must be bit-identical to the standard scheme. This script checks that
directly, in torch.float16 (matching a_hat_cache's real dtype), over many windows and using a
semi-realistic per-step delta-scale trajectory (reusing the real step_gain_tail growth already
measured in the C15 calibration, so the scale actually varies step to step the way it does in
deployment -- a constant scale would not exercise the interesting case).

conv is linear, so for checking a_hat's own bit-exactness a conv is not needed at all; o_hat's
recursion (o_hat_t = o_hat_{t-1} + alpha_t * conv(c_t)) is checked with a random per-step SCALAR
"conv" (valid because linearity lets a scalar stand in for any linear operator when checking an
algebraic identity like this one -- if it holds for an arbitrary scalar per step, it holds for the
real conv, which is just that scalar operator applied per output element).

Run: python docs/ahat_skip2_exact_2026-08-26/scripts/verify_skip2_exact.py
"""
import json

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

QLEVEL = 127.0
STEPS = 200  # even, so windows tile exactly; real schedule is 50/200 depending on config

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
    """Same construction as simulate_skip_write.py's (already-fixed) version: innovation scaled
    inversely with the calibrated scale so the per-step delta stays near target_code_mag, matching
    the real system's near-zero K=1 clip rate rather than an unrealistic constant-variance walk."""
    x = np.zeros(n)
    x[0] = level
    for t in range(1, n):
        step_scale = target_code_mag / scale_traj[t]
        x[t] = x[t - 1] + np.random.randn() * step_scale
    return x


def quantize(delta, scale):
    q = torch.clamp(torch.round(delta * scale), -QLEVEL, QLEVEL)
    return q


def dequantize(code, scale):
    return code / scale


def run_standard(v, scales, conv_coef):
    """Baseline: a_hat written every step. All state in float16, matching a_hat_cache's real dtype."""
    n = len(v)
    a_hat = torch.zeros(1, dtype=torch.float16)
    o_hat = torch.zeros(1, dtype=torch.float32)  # o_hat is fp32-accumulated in the real kernel
    a_hat_trace = torch.zeros(n, dtype=torch.float16)
    o_hat_trace = torch.zeros(n, dtype=torch.float32)
    code_trace = torch.zeros(n, dtype=torch.float32)
    for t in range(n):
        vt = torch.tensor([v[t]], dtype=torch.float32)
        s = float(scales[t])
        delta = vt.to(torch.float32) - a_hat.to(torch.float32)
        code = quantize(delta, s)
        a_hat = (a_hat.to(torch.float32) + dequantize(code, s)).to(torch.float16)
        o_hat = o_hat + conv_coef[t] * code  # conv is linear; a scalar stands in for it here
        a_hat_trace[t] = a_hat.item()
        o_hat_trace[t] = o_hat.item()
        code_trace[t] = code.item()
    return a_hat_trace, o_hat_trace, code_trace


def run_skip2(v, scales, conv_coef):
    """K=2 deferred-write scheme. a_hat's DRAM buffer (`A`) is only written on even steps (t odd
    in 0-indexed terms, i.e. the 2nd step of each window). On odd-indexed (0-indexed even) steps
    the code is computed and a pending-code register holds it for the next step's reconstruction;
    a_hat's own buffer is untouched."""
    n = len(v)
    assert n % 2 == 0
    A = torch.zeros(1, dtype=torch.float16)   # a_hat's actual DRAM buffer -- only touched on catch-up steps
    o_hat = torch.zeros(1, dtype=torch.float32)
    pending_code = None
    a_hat_dram_trace = torch.full((n,), float("nan"), dtype=torch.float16)  # NaN when not written this step
    a_hat_reconstructed_trace = torch.zeros(n, dtype=torch.float16)  # the value actually USED as reference
    o_hat_trace = torch.zeros(n, dtype=torch.float32)
    code_trace = torch.zeros(n, dtype=torch.float32)

    for t in range(n):
        vt = torch.tensor([v[t]], dtype=torch.float32)
        s = float(scales[t])
        window_pos = t % 2  # 0 = skip step, 1 = catchup step
        if window_pos == 0:
            ref = A.to(torch.float32)  # checkpoint, exactly what standard scheme's a_hat_{t-1} is
            a_hat_reconstructed_trace[t] = A.item()
        else:
            # reconstruct a_hat_{t-1} from the checkpoint plus the pending code -- NOT read from
            # A's own buffer directly (A still holds the value from 2 steps ago at this point)
            recon = (A.to(torch.float32) + dequantize(pending_code, prev_scale)).to(torch.float16)
            ref = recon.to(torch.float32)
            a_hat_reconstructed_trace[t] = recon.item()

        delta = vt - ref
        code = quantize(delta, s)
        o_hat = o_hat + conv_coef[t] * code
        o_hat_trace[t] = o_hat.item()
        code_trace[t] = code.item()

        if window_pos == 0:
            pending_code = code
            prev_scale = s
            # A is NOT written this step
        else:
            A = (ref + dequantize(code, s)).to(torch.float16)
            a_hat_dram_trace[t] = A.item()
            pending_code = None

    return a_hat_dram_trace, a_hat_reconstructed_trace, o_hat_trace, code_trace


print(f"{'layer':>24} {'max |o_hat diff|':>18} {'max |code diff|':>16} "
      f"{'a_hat @checkpoints bit-exact?':>30}")
all_ok = True
for name, row in LAYERS.items():
    scale_traj = scale_trajectory(row["delta_scale_step0"], row["delta_scale_tail"], STEPS)
    true_traj = make_true_trajectory(STEPS, scale_traj)
    conv_coef = np.random.randn(STEPS) * 0.7 + 1.0  # arbitrary per-step linear "conv" stand-in

    a_std, o_std, c_std = run_standard(true_traj, scale_traj, conv_coef)
    a_dram_s2, a_recon_s2, o_s2, c_s2 = run_skip2(true_traj, scale_traj, conv_coef)

    # codes must match bit-for-bit at every single step
    code_diff = (c_std - c_s2).abs().max().item()
    # o_hat must match bit-for-bit at every single step
    o_diff = (o_std - o_s2).abs().max().item()
    # a_hat's DRAM buffer only has a defined value on catchup steps (odd t, 0-indexed) -- compare
    # those against the standard scheme's a_hat at the SAME step index (both represent a_hat AFTER
    # step t's update)
    catchup_idx = torch.tensor([t for t in range(STEPS) if t % 2 == 1])
    a_hat_exact = torch.equal(a_std[catchup_idx], a_dram_s2[catchup_idx])
    # a_recon_s2[t] is the REFERENCE used to compute step t's delta, i.e. a_hat_{t-1} -- compare
    # against a_std shifted by one (a_std[t-1]), with a_recon_s2[0] checked against the shared
    # zero initial state instead
    recon_exact = torch.equal(a_std[:-1], a_recon_s2[1:]) and a_recon_s2[0].item() == 0.0

    ok = (code_diff == 0.0) and (o_diff == 0.0) and a_hat_exact and recon_exact
    all_ok = all_ok and ok
    print(f"{name:>24} {o_diff:>18.6g} {code_diff:>16.6g} "
          f"{'YES' if (a_hat_exact and recon_exact) else 'NO':>30}")

print(f"\nCodes bit-identical to the standard scheme at every step: {'YES' if all_ok else 'NO'}")
print("o_hat bit-identical to the standard scheme at every step:", "YES" if all_ok else "NO")
print("a_hat's DRAM buffer at every checkpoint (catch-up) step matches the standard scheme's a_hat"
      " at the same step:", "YES" if all_ok else "NO")
print("\nInterpretation: if all YES, the K=2 deferred-write design introduces ZERO additional"
      " numerical error relative to the standard per-step scheme -- the only change is WHEN a_hat's"
      " own DRAM buffer is written, not WHAT gets computed.")
