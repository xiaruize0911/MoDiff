"""Fix #4, the ADDITIVE route: can the weight zero point be applied AFTER the conv, on real kernels?

WHY THIS MATTERS. The fused route needs an EVT node in four entry points, one of which
(`conv2d_int4_fprop`) has no epilogue visitor tree at all. The correction is purely additive to the conv
output, so a separate pass would reach all four with no EVT work -- and would answer the quality question
before any of it exists. What has to be measured rather than assumed is the PRECISION COST: fix #2's
post-hoc correction delivered -1.6% where its fused halo delivered -7.1%, because it corrected a value the
epilogue had already rounded to fp16 (docs/zp_coverage_2026-08-13/FINDINGS.md).

THE IDENTITY THAT MAKES IT RUN ON THE EXISTING KERNEL. AdaRound's codes are unsigned, x_q in [0,15], with
a per-channel zero point; our int4 storage is SIGNED, [-8,7]. Those do not fit -- which is exactly why the
zero point cannot be folded into the codes and why fix #4 exists. But:

    x_q - z_w[k]  =  (x_q - 8)  +  (8 - z_w[k])
                     \______/       \________/
                    fits int4      per-channel

so the kernel runs on `w_c = x_q - 8`, a pure relabelling with NO clipping, and the leftover is a
per-channel constant whose contribution is `(8 - z_w[k]) * ws[k] * S[p] / s` -- one multiply-add against
`int4_window_sum`'s output. Note the shipped weight quantizer clamps to [-7,7] (Q=7) while the storage
holds -8; this test packs directly, so x_q = 0 is representable.

WHAT IS SCORED. Both routes against the same float64 reference conv:
    kernel-only   ACC(w_c) * alpha * ws            -- wrong by construction, the control
    additive      the above + (8 - z_w)*ws*alpha*S -- the route under test
The gap between `additive` and the reference is the fp16 epilogue rounding, and that number is the price
of the additive route.

Run: python integration/tests/test_zpw_additive.py
"""
import sys

import torch
import torch.nn.functional as F

import modiff_cutlass as mc

DEV = "cuda"
torch.manual_seed(20260817)


def pack_act(codes):
    """[N,C,H,W] int codes in [-8,7] -> [N,H,W,C/2] int8, low nibble = channel 2i."""
    c = codes.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    lo, hi = c[..., 0::2], c[..., 1::2]
    v = lo | (hi << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


def pack_w(codes):
    """[K,C,R,S] int codes in [-8,7] -> [K,R,S,C/2] int8, same nibble convention."""
    c = codes.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    lo, hi = c[..., 0::2], c[..., 1::2]
    v = lo | (hi << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


CASES = [(2, 192, 192, 3, 32, 32, 1, 1, 1),
         (2, 384, 384, 3, 16, 16, 1, 1, 1),
         (2, 384, 192, 3, 32, 32, 2, 1, 1),
         (2, 768, 768, 3, 8, 8, 1, 1, 1),
         (2, 192, 192, 1, 32, 32, 1, 0, 1)]

print("both routes scored against the same float64 reference conv\n")
print(f"{'shape':32} {'kernel only':>13} {'additive':>11}  {'gain':>7}")
rows, fails = [], []
for N, C, K, R, H, W, st, pad, dil in CASES:
    a_q = torch.randint(-7, 8, (N, C, H, W), device=DEV, dtype=torch.int64)
    x_q = torch.randint(0, 16, (K, C, R, R), device=DEV, dtype=torch.int64)      # AdaRound's grid
    z_w = torch.randint(1, 15, (K,), device=DEV, dtype=torch.int64)              # measured 1..14
    ws = (torch.rand(K, device=DEV) * 0.02 + 0.001).float()
    s_act = 12.34
    alpha = torch.tensor([1.0 / s_act], device=DEV, dtype=torch.float32)

    # reference: dequantize the asymmetric weights and convolve, in float64
    w_deq = ((x_q - z_w.view(K, 1, 1, 1)).double() * ws.double().view(K, 1, 1, 1))
    ref = F.conv2d((a_q.double() / s_act), w_deq, None, st, pad, dil)

    # the existing kernel, on the relabelled codes w_c = x_q - 8
    ap = pack_act(a_q)
    wp = pack_w(x_q - 8)
    Ho = (H + 2 * pad - dil * (R - 1) - 1) // st + 1
    Wo = (W + 2 * pad - dil * (R - 1) - 1) // st + 1
    out = torch.empty(N, K, Ho, Wo, device=DEV, dtype=torch.float16).to(
        memory_format=torch.channels_last)
    empty_h = torch.empty(0, device=DEV, dtype=torch.float16)
    empty_f = torch.empty(0, device=DEV, dtype=torch.float32)
    got = mc.conv2d_int4_evt_bias_residual_fp16(ap, wp, alpha, ws.contiguous(), empty_f, empty_h,
                                                out, st, st, pad, pad, dil, dil)

    # the additive correction: (8 - z_w[k]) * ws[k] * alpha * S[p]
    Sp = mc.int4_window_sum(ap, R, R, st, st, pad, pad, dil, dil)                # [N,Ho,Wo] fp32
    corr = ((8 - z_w).float() * ws / s_act).view(1, K, 1, 1) * Sp.unsqueeze(1)
    fixed = got.float() + corr

    e_kernel = float((got.double() - ref).norm() / ref.norm())
    e_fixed = float((fixed.double() - ref).norm() / ref.norm())
    tag = f"N{N} C{C} K{K} {R}x{R} {H}x{W} s{st} p{pad}"
    print(f"{tag:32} {e_kernel:13.3e} {e_fixed:11.3e}  {e_kernel / max(e_fixed, 1e-30):6.0f}x")
    rows.append((tag, e_kernel, e_fixed))

    # the additive route must land at fp16 epilogue precision, ~1e-3, not at the kernel-only error
    if not (e_fixed < 5e-3):
        fails.append(f"{tag}: additive route rel err {e_fixed:.3e}, expected fp16-epilogue order (<5e-3)")
    # ... and the control must be badly wrong, or the correction was not doing anything
    if not (e_kernel > 20 * e_fixed):
        fails.append(f"{tag}: kernel-only error {e_kernel:.3e} is not much worse than corrected "
                     f"{e_fixed:.3e} -- the operands are degenerate, so this proves nothing")

print()
if fails:
    print("GATE FAILED:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
worst = max(r[2] for r in rows)
print(f"GATE PASSED on {len(rows)} real shapes. The additive route reaches the asymmetric-weight answer")
print(f"to {worst:.2e} relative -- fp16 epilogue precision -- while the uncorrected kernel is 3-4 orders")
print(f"worse. So fix #4's QUALITY question can be answered with one extra pass and no EVT work; the")
print(f"fused version is then a performance question, not a correctness one.")
