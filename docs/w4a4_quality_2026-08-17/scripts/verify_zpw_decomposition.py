"""Fix #4, step 1: is the `ACC - z_w*S[p]` decomposition exact? Verified before any CUDA is written.

THE CLAIM the kernel work would rest on. With asymmetric int4 weights and a SYMMETRIC activation grid
(z_a = 0, which is what ships and what makes this padding-clean):

    out[k,p] = sum_i (w_q[k,i] - z_w[k]) * ws[k] * a_q[i]/s
             = (ws[k]/s) * ( sum_i w_q[k,i]*a_q[i]  -  z_w[k] * sum_i a_q[i] )
             = (ws[k]/s) * ( ACC[k,p]               -  z_w[k] * S[p]          )

so the only new quantity is S[p] = sum over the conv window of a_q -- ONE scalar per output pixel,
independent of k. Two properties this script checks rather than asserts:

  1. EXACTNESS. The decomposition must reproduce a direct dequantized conv to floating-point round-off,
     on real shapes including stride, padding and dilation. If it does not, no kernel can.

  2. PADDING-CLEANNESS -- the property that separates fix #4 from fix #2. A padded tap is a_q = 0, and
     with z_a = 0 that code means the value 0.0 exactly, so a box filter over valid taps needs NO border
     correction. Checked by comparing the padded case against an explicitly zero-padded reference.
     zp_coverage's fix #2 had `-z_a*sum(missing w_q)` per output pixel here and it cost that item the
     cheap route; this asserts the analogue does not exist.

S[p] itself is `F.conv2d(sum_c a_q, ones(R,S))` -- so the "windowed per-output-pixel reduction" the
record prices as a missing capability is a box filter, and in CUDA it is a channel sum plus 9 adds.

Run: python docs/w4a4_quality_2026-08-17/scripts/verify_zpw_decomposition.py
"""
import itertools

import torch
import torch.nn.functional as F

torch.manual_seed(20260817)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
Q_LO, Q_HI = 0, 15          # AdaRound's unsigned 4-bit grid; z_w brings it on-centre


def window_sum(a_q, R, S, stride, padding, dilation):
    """S[p] = sum over the conv window of a_q, summed over ALL input channels. [N,1,Ho,Wo].

    Channel sum first (it is independent of the window), then a box filter. That ordering is what makes
    this O(N*C*H*W) rather than O(N*C*H*W*R*S), and it is available because S[p] does not depend on k.
    """
    t = a_q.sum(dim=1, keepdim=True)                                    # [N,1,H,W]
    ones = torch.ones(1, 1, R, S, device=a_q.device, dtype=t.dtype)
    return F.conv2d(t, ones, stride=stride, padding=padding, dilation=dilation)


def check(K, C, R, S, H, W, N, stride, padding, dilation, tag):
    # real-ish operands: int4 activation codes on a symmetric grid, asymmetric int4 weight codes
    a_q = torch.randint(-7, 8, (N, C, H, W), device=DEV, dtype=torch.float64)
    w_q = torch.randint(Q_LO, Q_HI + 1, (K, C, R, S), device=DEV, dtype=torch.float64)
    z_w = torch.randint(1, 15, (K,), device=DEV, dtype=torch.float64)     # the measured 1..14 span
    ws = (torch.rand(K, device=DEV, dtype=torch.float64) * 0.02 + 0.001)
    s_act = 12.34

    # (A) the direct answer: dequantize the weights, convolve
    w_deq = (w_q - z_w.view(K, 1, 1, 1)) * ws.view(K, 1, 1, 1)
    direct = F.conv2d(a_q / s_act, w_deq, None, stride, padding, dilation)

    # (B) the decomposition the epilogue would compute
    acc = F.conv2d(a_q, w_q, None, stride, padding, dilation)             # int32 accumulator
    Sp = window_sum(a_q, R, S, stride, padding, dilation)                 # [N,1,Ho,Wo]
    decomp = (acc - z_w.view(1, K, 1, 1) * Sp) * (ws / s_act).view(1, K, 1, 1)

    rel = float((decomp - direct).norm() / direct.norm().clamp_min(1e-30))
    # (C) padding-cleanness: an explicitly zero-padded activation with padding=0 must agree, i.e. the
    #     padded taps needed no correction term.
    ok_pad = True
    if padding > 0:
        ap = F.pad(a_q, (padding,) * 4)
        acc_p = F.conv2d(ap, w_q, None, stride, 0, dilation)
        Sp_p = window_sum(ap, R, S, stride, 0, dilation)
        dec_p = (acc_p - z_w.view(1, K, 1, 1) * Sp_p) * (ws / s_act).view(1, K, 1, 1)
        ok_pad = float((dec_p - decomp).norm()) == 0.0

    print(f"  {tag:44} rel err {rel:.3e}   padding-clean {ok_pad}")
    assert rel < 1e-12, f"{tag}: decomposition is NOT exact (rel {rel:.3e})"
    assert ok_pad, f"{tag}: zero-padded taps needed a correction -- fix #2's defect DOES have an analogue"
    return rel


print(f"device {DEV}; float64 so the check is about the algebra, not about precision\n")
print("shapes taken from KERNEL_SPEEDUP's per-layer table (the real quantized convs):")
worst = 0.0
for K, C, R, H, W, st, pad, dil in [
        (384, 384, 3, 32, 32, 1, 1, 1),
        (192, 576, 3, 32, 32, 1, 1, 1),
        (384, 768, 3, 16, 16, 1, 1, 1),
        (768, 768, 3, 8, 8, 1, 1, 1),
        (768, 1536, 3, 4, 4, 1, 1, 1),
        (192, 192, 1, 32, 32, 1, 0, 1),      # 1x1, no padding
        (384, 384, 3, 16, 16, 2, 1, 1),      # strided downsample
        (192, 192, 3, 32, 32, 1, 2, 2)]:     # dilated, padding != 1
    worst = max(worst, check(K, min(C, 96), R, R, H, W, 2, st, pad, dil,
                             f"K{K} C{min(C,96)} {R}x{R} {H}x{W} s{st} p{pad} d{dil}"))

# The one case that must FAIL, so the checks above are known to be able to fail at all: drop the
# z_w term and the same comparison must blow up. A gate that cannot fail is not a gate.
a_q = torch.randint(-7, 8, (2, 64, 16, 16), device=DEV, dtype=torch.float64)
w_q = torch.randint(Q_LO, Q_HI + 1, (32, 64, 3, 3), device=DEV, dtype=torch.float64)
z_w = torch.randint(1, 15, (32,), device=DEV, dtype=torch.float64)
ws = torch.rand(32, device=DEV, dtype=torch.float64) * 0.02 + 0.001
direct = F.conv2d(a_q / 12.34, (w_q - z_w.view(-1, 1, 1, 1)) * ws.view(-1, 1, 1, 1), None, 1, 1, 1)
no_zp = F.conv2d(a_q, w_q, None, 1, 1, 1) * (ws / 12.34).view(1, -1, 1, 1)
bad = float((no_zp - direct).norm() / direct.norm())
print(f"\n  NEGATIVE CONTROL, z_w term omitted:            rel err {bad:.3e}  (must be large)")
assert bad > 0.1, "omitting z_w changed almost nothing -- the operands are degenerate, not the check"

print(f"\nDECOMPOSITION EXACT on all 8 shapes (worst rel err {worst:.3e}) and PADDING-CLEAN.")
print("So fix #4 needs: a channel sum + an R x S box filter, and one Mul+Add in the EVT.")
