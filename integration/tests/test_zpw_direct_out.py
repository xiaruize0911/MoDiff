"""Gate for the DIRECT-OUTPUT fix #4 sites, which were wired without one and then measured FID 309.7.

test_zpw_ohat covers `_conv_from_int4_o_hat` -- the accumulate-into-cache pattern. The four sites wired
on 2026-08-18 (`_int4_conv`, `_int4_conv_dynamic_fused`, `_conv_from_int4:{evt_bias_residual,fprop}`)
return the conv result directly and go through `_zpw_add_to_out` instead. Nothing tested them, the first
end-to-end run read FID 309.689 against a 52.584 baseline, and a 6x regression is bug magnitude rather
than a suboptimal grid -- this project's own rule ("anything past 3x is bug magnitudes, not a result").

Same three-part shape as the o_hat gate, because it is the shape that catches sign and unit errors:
  1. DORMANT: weight_zp = 0 must be bit-identical to before the code existed.
  2. ARMED: must match a float64 ASYMMETRIC-weight reference to fp16 epilogue precision.
  3. NEGATIVE CONTROL: the same comparison against a SYMMETRIC reference must be badly wrong, or the
     test would pass on a correction that does nothing.

Run: python integration/tests/test_zpw_direct_out.py
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from integration.kernels.int4_optimized import OptimizedInt4Conv2d

DEV = "cuda"
torch.manual_seed(20260818)
N, C, K, R, H, W = 2, 192, 96, 3, 16, 16
PAD, S_ACT = 1, 12.34


def pack_w(codes):                       # [K,C,R,S] signed -> [K,R,S,C/2]
    c = codes.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    v = c[..., 0::2] | (c[..., 1::2] << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


def build(x_q, z_w, ws, arm_zp):
    conv = nn.Conv2d(C, K, R, padding=PAD, bias=False).to(DEV).half()
    q = OptimizedInt4Conv2d(conv).to(DEV)
    q.weight_packed = pack_w(x_q - 8).to(DEV)
    q.weight_scale_channel = ws.view(1, K, 1, 1).to(DEV)
    q.weight_scale_channel_half = ws.half().contiguous().to(DEV)
    q.is_calibrated = True
    q.weight_zp = ((8 - z_w).float() if arm_zp else torch.zeros(K)).to(DEV)
    return q


x_q = torch.randint(0, 16, (K, C, R, R), device=DEV, dtype=torch.int64)
z_w = torch.randint(1, 15, (K,), device=DEV, dtype=torch.int64)
ws = (torch.rand(K, device=DEV) * 0.01 + 0.002).float()
a_codes = torch.randint(-7, 8, (N, C, H, W), device=DEV, dtype=torch.int64)

# fp64 references from the integer codes, so no fp16 and no epilogue rounding enter the target.
a64 = (a_codes.double() / S_ACT)
W_asym = ((x_q - z_w.view(K, 1, 1, 1)).double() * ws.double().view(K, 1, 1, 1))
W_sym = ((x_q - 8).double() * ws.double().view(K, 1, 1, 1))
ref_asym = F.conv2d(a64, W_asym, None, 1, PAD)
ref_sym = F.conv2d(a64, W_sym, None, 1, PAD)


def pack_act(codes):                     # [N,C,H,W] signed -> NHWC packed
    c = codes.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    v = c[..., 0::2] | (c[..., 1::2] << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


x_packed = pack_act(a_codes)
alpha = torch.tensor([1.0 / S_ACT], device=DEV, dtype=torch.float32)
fails = []


def run_direct(q):
    """The `_conv_from_int4` direct-output path, which `_zpw_add_to_out` sits in."""
    q._cached_alpha_tensor = alpha
    q._empty_bias = torch.empty(0, device=DEV)
    return q._conv_from_int4(x_packed, H, W).double()


# 1. dormant
d0 = run_direct(build(x_q, z_w, ws, arm_zp=False))
d0b = run_direct(build(x_q, z_w, ws, arm_zp=False))
same = bool(torch.equal(d0, d0b))
print(f"  1. dormant (weight_zp = 0), two builds bit-identical: {same}")
if not same:
    fails.append("the dormant path is not reproducible, so nothing below can be attributed")

rel_sym = float((d0 - ref_sym).norm() / ref_sym.norm())
print(f"     dormant vs the SYMMETRIC fp64 reference: rel {rel_sym:.3e}  (should be ~1e-3, fp16 epilogue)")
if rel_sym > 2e-2:
    fails.append(f"the dormant path does not even match the symmetric reference ({rel_sym:.3e}) -- the "
                 f"harness, not the zero point, is wrong")

# 2. armed vs the asymmetric truth
d1 = run_direct(build(x_q, z_w, ws, arm_zp=True))
rel_asym = float((d1 - ref_asym).norm() / ref_asym.norm())
print(f"  2. armed vs the ASYMMETRIC fp64 reference: rel {rel_asym:.3e}")
if rel_asym > 2e-2:
    fails.append(f"ARMED output is {rel_asym:.3e} from the asymmetric truth -- the direct-output "
                 f"correction is wrong in sign, units, or placement. This is what FID 309.689 was.")

# 3. negative control
rel_ctl = float((d1 - ref_sym).norm() / ref_sym.norm())
print(f"  3. armed vs the SYMMETRIC reference (must be BIG): rel {rel_ctl:.3e}")
if rel_ctl < 0.05:
    fails.append("armed and symmetric agree, so the correction is doing nothing and check 2 is vacuous")

print()
if fails:
    print("GATE FAILED:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("GATE PASSED: the direct-output sites apply the weight zero point correctly.")
