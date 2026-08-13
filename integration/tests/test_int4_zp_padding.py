"""Does an asymmetric activation grid survive ZERO PADDING? Measured per output pixel.

THE QUESTION. With a zero point, a stored code c means the activation (c - z)/s. A padded tap is not
a stored code -- the conv's implicit GEMM reads literal 0 there -- so it dequantizes to (0 - z)/s =
-z/s, which is NOT zero. Every output pixel whose receptive field crosses the border therefore sees a
fabricated activation of -z/s on the missing taps, weighted by those taps' w_q.

The bias fold cannot absorb it. It subtracts z * sum_i w_q[k,i] over the WHOLE kernel, one constant per
output channel; the padded pixels need the sum over only the taps that are actually missing, which
depends on WHERE the pixel is. Per-output-pixel, not per-channel -- structurally the same obstacle that
deprioritised the weight zero point (fix #4, docs/paper_repro_2026-08-12/FINDINGS.md section 5).

THE PREDICTION, written before the run:

  * padding = 1 (3x3 convs, i.e. every ResBlock conv here): error concentrated on the border ring,
    interior essentially exact, and the border error scaling with |z|/s.
  * padding = 0: no border, so no error beyond the grid's own -- z is harmless there.
  * error rising as s falls, because the fabricated activation is -z/s.

If that is what comes out, then fix #2 is not "implemented and unhelpful"; it is INCOMPATIBLE with
zero-padded convolution as built, and the measured +70%/+170% end-to-end are that incompatibility
rather than a statement about asymmetric grids.

THE REFERENCE is what the module is supposed to compute: quantize x asymmetrically, DEQUANTIZE it in
fp32, and convolve with the dequantized weights and the ORIGINAL bias. Padding then happens on real
zeros, which is the correct semantics. Comparing the kernel against that isolates exactly the padding
question and nothing else -- both sides use the same codes, the same weights and the same bias.

Run: python integration/tests/test_int4_zp_padding.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

#: This file EXISTS to measure the padded-conv error, so it must opt in to the refusal that
#: _refold_zp_bias now raises for exactly this configuration. Set before the import that reads it.
os.environ["MODIFF_ZP_ALLOW_PADDED"] = "1"

import torch                                                              # noqa: E402
import torch.nn as nn                                                     # noqa: E402
import torch.nn.functional as F                                           # noqa: E402
from integration.kernels.int4_optimized import (                          # noqa: E402
    OptimizedInt4Conv2d, _int4_weight_scale)

Q = 7.0


def build(cin, cout, k, pad, seed=20260813):
    torch.manual_seed(seed)
    c = nn.Conv2d(cin, cout, k, padding=pad, bias=True).cuda().half()
    return OptimizedInt4Conv2d(c, layer_name=f"k{k}p{pad}").cuda()


def reference(m, x, s, z):
    """What the module is supposed to compute: dequantize the asymmetric codes, then convolve.

    Weights are taken from the module's OWN quantized state (weight_scale_channel + the same rounding
    rule), so any disagreement is about the activation path, not about weight quantization.
    """
    xq = torch.clamp(torch.round(x.float() * s) + z, -Q, Q)
    xdq = (xq - z) / s
    w = m._orig_weight if m._orig_weight is not None else m.weight
    K = w.shape[0]
    wf = w.reshape(K, -1).float()
    sc = _int4_weight_scale(wf)
    wdq = ((wf / sc[:, None]).round().clamp(-7, 7) * sc[:, None]).reshape_as(w).float()
    # _orig_bias is stored [1, K, 1, 1] (broadcast-shaped for the eager `out + bias`), F.conv2d wants
    # [K]. Flatten rather than reshape-guess.
    b = m._orig_bias
    return F.conv2d(xdq, wdq, None if b is None else b.float().reshape(-1),
                    stride=m.stride, padding=m.padding, dilation=m.dilation)


def split_border(err, pad):
    """(interior, border) views. With padding p and a 2p+1 kernel the affected ring is p deep."""
    if pad == 0:
        return err, None
    inner = err[..., pad:-pad, pad:-pad]
    mask = torch.ones_like(err, dtype=torch.bool)
    mask[..., pad:-pad, pad:-pad] = False
    return inner, err[mask]


def run(cin, cout, k, pad, s, z, n=2, hw=16):
    m = build(cin, cout, k, pad)
    m.set_static_calibration(s, None, z)
    x = (torch.randn(n, cin, hw, hw, device="cuda") * 0.7).abs()      # silu-like, one-sided
    # FP32 ON PURPOSE. _forward_standard's fp16 branch reaches
    # step1_static_quantize_pack_int4_noahat_fprop, which is deliberately guarded rather than taught
    # (the 2026-08-13 census found it unreachable in the shipped configuration), so an fp16 input here
    # RAISES -- as it should. The fp32 branch routes through scale_quantize_and_pack_zp, which honours
    # z. Both branches quantize the activation on the activation grid into the same biased conv, so
    # this substitution does not affect what the padding question is asking.
    xf = x.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        got = m(xf).float()
    want = reference(m, x, s, z)
    err = (got - want).abs()
    scale = want.abs().mean().clamp(min=1e-6)
    inner, border = split_border(err, pad)
    return {
        "rel": float(err.norm() / want.norm()),
        "interior": float(inner.mean() / scale),
        "border": (float(border.mean() / scale) if border is not None else None),
        "border_frac": (float(border.numel() / err.numel()) if border is not None else 0.0),
    }


def corner_prediction(m, s, z):
    """The exact error predicted at output pixel (0,0) of a 3x3 pad=1 stride=1 conv.

        o_kernel - o_ref = -z * sum_{i in MISSING taps} w_q[k,i] * ws[k] / s

    because the kernel's padded taps contribute code 0 while the bias fold subtracted z times the
    sum over ALL taps. At the top-left corner the missing taps are exactly those with kh == 0 or
    kw == 0 (the -1 input row/column). This is the gate that turns "the pattern looks like padding"
    into "the error IS the predicted term".
    """
    w = m._orig_weight if m._orig_weight is not None else m.weight
    K = w.shape[0]
    wf = w.reshape(K, -1).float()
    sc = _int4_weight_scale(wf)
    wq = (wf / sc[:, None]).round().clamp(-7, 7).reshape_as(w)      # [K, C, 3, 3]
    missing = wq[:, :, 0, :].sum(dim=(1, 2)) + wq[:, :, 1:, 0].sum(dim=(1, 2))
    ws = m.weight_scale_channel.float().reshape(-1)
    return -z * missing * ws / s


def main():
    fails = []
    print("A. padding=1 (3x3, every ResBlock conv) vs padding=0 (1x1), z = -5, s = 8\n")
    print(f"{'conv':22s}{'relL2':>9}{'interior':>11}{'border':>10}{'border %':>10}")
    p1 = run(64, 32, 3, 1, s=8.0, z=-5.0)
    p0 = run(64, 32, 1, 0, s=8.0, z=-5.0)
    # THE FLOOR OF THIS COMPARISON, measured rather than assumed. `reference` re-derives the weight
    # codes and accumulates in a different order from the CUTLASS kernel, so even a perfectly correct
    # symmetric path disagrees with it by a little. The first version of this file gated on an
    # absolute 5e-3 I picked out of the air and "failed" at 0.0066 -- which is that floor, not a
    # defect. Every gate below is stated relative to it.
    floor = run(64, 32, 3, 1, s=8.0, z=0.0)["rel"]
    print(f"{'1x1 pad=0':22s}{p0['rel']:9.4f}{p0['interior']:11.5f}{'-':>10}{'-':>10}")
    print(f"\nharness floor (z=0, pad=1, same comparison): {floor:.5f}")

    # GATE 1: with no padding the asymmetric path must sit at that floor. If it does not, the problem
    # is the arithmetic, not the padding, and every conclusion below is void.
    ok = p0["rel"] <= 2.0 * floor
    print(f"1. pad=0 asymmetric is AT the floor (<= 2x)            : {'ok' if ok else 'FAIL'} "
          f"({p0['rel']:.5f} vs {floor:.5f})")
    fails += [] if ok else ["pad=0 arithmetic"]

    # GATE 1b: and with padding it must be far ABOVE it.
    ok = p1["rel"] > 10.0 * floor
    print(f"1b. pad=1 asymmetric is far above the floor (>10x)     : {'ok' if ok else 'FAIL'} "
          f"({p1['rel']:.5f} = {p1['rel'] / floor:.0f}x floor)")
    fails += [] if ok else ["pad=1 above floor"]

    # GATE 2: with padding the error must be concentrated on the border. This is the claim.
    ratio = p1["border"] / max(p1["interior"], 1e-12)
    ok = ratio > 10.0
    print(f"2. pad=1 error is on the BORDER, not the interior     : {'ok' if ok else 'FAIL'} "
          f"(border/interior = {ratio:.0f}x)")
    fails += [] if ok else ["border concentration"]

    # GATE 3: the border error must scale with |z|/s -- the fabricated activation's magnitude.
    print("\nB. the border error is the fabricated -z/s activation\n")
    print(f"{'z':>5}{'s':>7}{'|z|/s':>9}{'border err':>13}{'relL2':>9}")
    prev = None
    mono = True
    for z, s in ((0.0, 8.0), (-2.0, 8.0), (-5.0, 8.0), (-7.0, 8.0), (-5.0, 2.0), (-5.0, 32.0)):
        r = run(64, 32, 3, 1, s=s, z=z)
        print(f"{z:5.0f}{s:7.1f}{abs(z) / s:9.4f}{r['border']:13.5f}{r['rel']:9.4f}")
        if s == 8.0:
            if prev is not None and r["border"] < prev:
                mono = False
            prev = r["border"]
    ok = mono
    print(f"\n3. border error grows monotonically with |z| at fixed s: {'ok' if ok else 'FAIL'}")
    fails += [] if ok else ["monotonic in z"]

    ok = p1["rel"] / floor > 10.0
    print(f"4. z=0 with padding is the control, and z!=0 is {p1['rel'] / floor:.0f}x it  : "
          f"{'ok' if ok else 'FAIL'}")
    fails += [] if ok else ["z=0 control"]

    # GATE 5: THE QUANTITATIVE ONE. Predict the corner pixel's error from -z*sum(missing w_q)*ws/s
    # and require agreement. A scaling law can be coincidence; this cannot.
    print("\nC. the corner error equals -z * sum(missing w_q) * ws / s\n")
    print(f"{'z':>5}{'s':>7}{'measured':>12}{'predicted':>12}{'rel diff':>11}")
    worst = 0.0
    for z, s in ((-2.0, 8.0), (-5.0, 8.0), (-7.0, 8.0), (-5.0, 2.0)):
        m = build(64, 32, 3, 1)
        m.set_static_calibration(s, None, z)
        x = (torch.randn(2, 64, 16, 16, device="cuda") * 0.7).abs()
        xf = x.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            got = m(xf).float()
        want = reference(m, x, s, z)
        meas = (got - want)[0, :, 0, 0]
        pred = corner_prediction(m, s, z)
        rel = float((meas - pred).norm() / pred.norm().clamp(min=1e-9))
        worst = max(worst, rel)
        print(f"{z:5.0f}{s:7.1f}{float(meas.norm()):12.4f}{float(pred.norm()):12.4f}{rel:11.4f}")
    ok = worst < 0.05
    print(f"\n5. corner error matches the prediction (<5% rel)       : {'ok' if ok else 'FAIL'} "
          f"(worst {worst:.4f})")
    fails += [] if ok else ["corner prediction"]

    print()
    if fails:
        print(f"GATES FAILED ({len(fails)}): {', '.join(fails)}")
        print("The padding explanation is NOT established. Do not cite it.")
        return 1
    print("ESTABLISHED: the asymmetric activation grid is arithmetically correct where there is no\n"
          "padding, and wrong on the padded border, by an amount that grows with the fabricated\n"
          "-z/s activation. Zero-padded convolution and a per-channel bias fold cannot both hold:\n"
          "the missing-tap sum is per OUTPUT PIXEL. This is fix #2's real obstacle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
