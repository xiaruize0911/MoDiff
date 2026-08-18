"""Gate for the DUAL-STORE fix #4 sites -- the last of the three patterns without one.

`conv2d_int4_evt_o_hat_residual` writes BOTH `o_hat += conv` and `out = o_hat_new + residual` in one
pass, so `_zpw_add_to_ohat_and_out` has to correct two tensors from one computation. Getting it half
right -- cache only -- returns a value short by the correction while leaving the next step's accumulator
correct, which looks like a small quality regression rather than a bug.

Wired 2026-08-18 with no test. The end-to-end run then read FID 309.689 against a 52.584 baseline and
latent relL2 2.52 -- and zeroing the weights entirely gives 1.0, so that is divergence, not a grid. The
o_hat and direct-output patterns are both gated and both pass, which leaves this one.

  1. DORMANT: weight_zp = 0 bit-identical on BOTH outputs.
  2. ARMED, CACHE: the accumulated o_hat matches a float64 asymmetric reference.
  3. ARMED, RETURN: `out` matches o_hat + residual in float64 -- the half that a cache-only fix misses.
  4. NEGATIVE CONTROL: both compared against a symmetric reference must be badly wrong.

Run: python integration/tests/test_zpw_dual_store.py
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from integration.kernels.int4_optimized import OptimizedInt4Conv2d

DEV = "cuda"
torch.manual_seed(20260818)
N, C, K, R, H, W, STEPS = 2, 192, 192, 3, 16, 16, 3
PAD, S_ACT = 1, 12.34


def pack(codes):
    c = codes.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    v = c[..., 0::2] | (c[..., 1::2] << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


x_q = torch.randint(0, 16, (K, C, R, R), device=DEV, dtype=torch.int64)
z_w = torch.randint(1, 15, (K,), device=DEV, dtype=torch.int64)
ws = (torch.rand(K, device=DEV) * 0.01 + 0.002).float()
acts = [torch.randint(-7, 8, (N, C, H, W), device=DEV, dtype=torch.int64) for _ in range(STEPS)]
resid = (torch.randn(N, K, H, W, device=DEV) * 0.1).half().contiguous(
    memory_format=torch.channels_last)
alpha = torch.tensor([1.0 / S_ACT], device=DEV, dtype=torch.float32)


def build(arm):
    conv = nn.Conv2d(C, K, R, padding=PAD, bias=False).to(DEV).half()
    q = OptimizedInt4Conv2d(conv).to(DEV)
    q.weight_packed = pack(x_q - 8).to(DEV)
    q.weight_scale_channel = ws.view(1, K, 1, 1).to(DEV)
    q.weight_scale_channel_half = ws.half().contiguous().to(DEV)
    q.is_calibrated = True
    q.o_hat_cache = None
    q.weight_zp = ((8 - z_w).float() if arm else torch.zeros(K)).to(DEV)
    return q


def run(q):
    """Drive the dual-store helper directly on the entry the two wired sites call."""
    import modiff_cutlass as mc
    q.o_hat_cache = torch.zeros(N, K, H, W, device=DEV, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    last = None
    for a in acts:
        xp = pack(a)
        out = torch.empty_like(q.o_hat_cache)
        mc.conv2d_int4_evt_o_hat_residual(
            xp, q.weight_packed, alpha, q.weight_scale_channel.view(-1),
            q.o_hat_cache, resid, out, 1, 1, PAD, PAD, 1, 1)
        q._zpw_add_to_ohat_and_out(xp, alpha, out)
        last = out
    return q.o_hat_cache.double().clone(), last.double().clone()


def ref(W_kernel):
    acc = torch.zeros(N, K, H, W, device=DEV, dtype=torch.float64)
    for a in acts:
        acc = acc + F.conv2d(a.double() / S_ACT, W_kernel, None, 1, PAD)
    return acc, acc + resid.double()


W_asym = ((x_q - z_w.view(K, 1, 1, 1)).double() * ws.double().view(K, 1, 1, 1))
W_sym = ((x_q - 8).double() * ws.double().view(K, 1, 1, 1))
oh_a, out_a = ref(W_asym)
oh_s, out_s = ref(W_sym)
fails = []

c0, o0 = run(build(False))
c0b, o0b = run(build(False))
same = bool(torch.equal(c0, c0b) and torch.equal(o0, o0b))
print(f"  1. dormant, two builds bit-identical (cache and out): {same}")
if not same:
    fails.append("the dormant dual-store path is not reproducible")

c1, o1 = run(build(True))
r_c = float((c1 - oh_a).norm() / oh_a.norm())
r_o = float((o1 - out_a).norm() / out_a.norm())
print(f"  2. armed CACHE  vs asymmetric fp64: rel {r_c:.3e}")
print(f"  3. armed RETURN vs asymmetric fp64: rel {r_o:.3e}")
if r_c > 3e-2:
    fails.append(f"the accumulated cache is {r_c:.3e} from the asymmetric truth")
if r_o > 3e-2:
    fails.append(f"the RETURNED tensor is {r_o:.3e} from the asymmetric truth -- this is the half a "
                 f"cache-only correction misses, and it is what the recursion hands downstream")

r_ctl = float((c1 - oh_s).norm() / oh_s.norm())
print(f"  4. armed cache vs SYMMETRIC reference (must be BIG): rel {r_ctl:.3e}")
if r_ctl < 0.05:
    fails.append("armed and symmetric agree -- the correction does nothing and 2/3 are vacuous")

print()
if fails:
    print("GATE FAILED:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("GATE PASSED: the dual-store sites correct both the cache and the returned tensor.")
