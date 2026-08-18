"""Gate for the ONE fix #4 site wired so far: the o_hat accumulate, across several MoDiff steps.

WHY MULTIPLE STEPS. `conv2d_int4_evt_o_hat` accumulates INTO `o_hat_cache` in place, and
`_module_output()` returns that same tensor. So a correction applied to the return value but not the cache
would look right on step 1 and be wrong on every step after -- silently, because the recursion keeps
reading the cache. A single-step test cannot see that; this one runs four steps with different activations
and compares the accumulated cache against a float64 accumulation of the same sequence.

WHAT IS ASSERTED
  1. DORMANT: with weight_zp = 0 the cache is BIT-IDENTICAL to a run of the same sequence before the
     zero point is armed. This is what makes landing the code with no table loaded a no-op.
  2. ARMED: with weight_zp = 8 - z_w the cache matches the float64 asymmetric-weight reference to fp16
     accumulation precision.
  3. NEGATIVE CONTROL: the same armed comparison against a reference that OMITS the zero point must be
     badly wrong, or the test would pass on a correction that does nothing.

Run: python integration/tests/test_zpw_ohat.py
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from integration.kernels.int4_optimized import OptimizedInt4Conv2d

DEV = "cuda"
torch.manual_seed(20260817)
N, C, K, R, H, W, STEPS = 2, 192, 96, 3, 16, 16, 4
PAD, ST, DIL = 1, 1, 1
S_ACT = 12.34


def pack(codes, spatial_last):
    c = (codes.permute(0, 2, 3, 1) if spatial_last else codes).contiguous().to(torch.int64) & 0x0F
    lo, hi = c[..., 0::2], c[..., 1::2]
    v = lo | (hi << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


def build_layer(x_q, z_w, ws):
    conv = nn.Conv2d(C, K, R, padding=PAD, bias=False).to(DEV).half()
    q = OptimizedInt4Conv2d(conv).to(DEV)
    q.weight_packed = pack(x_q - 8, spatial_last=True).to(DEV)         # [K,R,S,C/2]
    q.weight_scale_channel = ws.view(1, K, 1, 1).to(DEV)
    q.is_calibrated = True
    q.o_hat_cache = None
    q.weight_zp = (8 - z_w).float().to(DEV)
    return q


def run(q, acts, alpha):
    q.o_hat_cache = None
    for a in acts:
        q._conv_from_int4_o_hat(a, H, W, alpha)
    return q.o_hat_cache.detach().clone()


x_q = torch.randint(0, 16, (K, C, R, R), device=DEV, dtype=torch.int64)
z_w = torch.randint(1, 15, (K,), device=DEV, dtype=torch.int64)
ws = (torch.rand(K, device=DEV) * 0.02 + 0.001).float()
alpha = torch.tensor([1.0 / S_ACT], device=DEV, dtype=torch.float32)
a_qs = [torch.randint(-7, 8, (N, C, H, W), device=DEV, dtype=torch.int64) for _ in range(STEPS)]
a_ps = [pack(a, spatial_last=True) for a in a_qs]

fails = []

# ---- 1. dormant is bit-identical ------------------------------------------------------------------
q = build_layer(x_q, z_w, ws)
q.weight_zp = torch.zeros(K, device=DEV)               # disarm
assert not q._has_weight_zp
dormant = run(q, a_ps, alpha)
q2 = build_layer(x_q, z_w, ws)
q2.weight_zp = torch.zeros(K, device=DEV)
dormant2 = run(q2, a_ps, alpha)
same = bool(torch.equal(dormant, dormant2))
print(f"  1. dormant (weight_zp = 0), two builds: bit-identical {same}")
if not same:
    fails.append("dormant runs differ -- the layer is not reproducible even with the correction off, "
                 "so nothing below can be attributed to the zero point")

# ---- 2. armed matches the float64 asymmetric reference --------------------------------------------
q3 = build_layer(x_q, z_w, ws)
assert q3._has_weight_zp, "the layer did not arm -- weight_zp is zero"
armed = run(q3, a_ps, alpha)

w_asym = ((x_q - z_w.view(K, 1, 1, 1)).double() * ws.double().view(K, 1, 1, 1))
w_sym8 = ((x_q - 8).double() * ws.double().view(K, 1, 1, 1))
ref = torch.zeros_like(armed, dtype=torch.float64)
ref_nozp = torch.zeros_like(ref)
for a in a_qs:
    ref += F.conv2d(a.double() / S_ACT, w_asym, None, ST, PAD, DIL)
    ref_nozp += F.conv2d(a.double() / S_ACT, w_sym8, None, ST, PAD, DIL)

e_armed = float((armed.double() - ref).norm() / ref.norm())
e_dormant = float((dormant.double() - ref).norm() / ref.norm())
print(f"  2. armed vs float64 asymmetric reference over {STEPS} steps: rel err {e_armed:.3e}")
print(f"     (dormant, i.e. the correction never applied:              rel err {e_dormant:.3e})")
if not (e_armed < 5e-3):
    fails.append(f"armed rel err {e_armed:.3e} -- expected fp16 accumulation order (<5e-3). If this is "
                 f"~{e_dormant:.1e} the correction is not reaching the cache at all.")
if not (e_dormant > 20 * e_armed):
    fails.append(f"dormant ({e_dormant:.3e}) is not much worse than armed ({e_armed:.3e}) -- the "
                 f"operands are degenerate and this proves nothing")

# ---- 3. negative control: the reference WITHOUT the zero point must disagree ------------------------
e_wrong = float((armed.double() - ref_nozp).norm() / ref_nozp.norm())
print(f"  3. armed vs a reference that OMITS the zero point:         rel err {e_wrong:.3e}  (must be big)")
if not (e_wrong > 0.1):
    fails.append(f"armed also matches the no-zero-point reference ({e_wrong:.3e}) -- the two references "
                 f"are not distinguishable here, so the pass above is vacuous")

print()
if fails:
    print("GATE FAILED:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print(f"GATE PASSED: the o_hat site applies the weight zero point correctly across {STEPS} accumulated "
      f"steps\n({e_armed:.2e} against float64), is bit-identical when dormant, and the control fires.")
