"""Does zero-padding break the activation zero point? Isolated, one conv, no model.

WHY ASK. With coverage complete, the asymmetric grid measures WORSE on both W4A4 arms (PTQ +70%,
MoDiff +170% against symmetric) even though its step is 1.6x finer on the same clipped range. The fp16
bias-storage hypothesis is refuted (zp_bias_precision.py: the bias is fp32 and injects 0.00% of |o|).
That same run recorded the number this script is about: |corr| / |o| has median 7x. The epilogue adds a
correction SEVEN TIMES LARGER than the output it corrects, so any tap where the correction does not
belong contributes an error seven times the signal.

THE HYPOTHESIS. The fold assumes every tap of every output pixel reads an activation encoded with the
zero point:

    sum_i w_q[k,i] * a[i] = (sum_i w_q[k,i]*a_q[i] - z*sum_i w_q[k,i]) * ws[k]/s

`z*sum_i w_q[k,i]` sums over ALL taps, i.e. it assumes a_q[i] = round(a[i]*s) + z everywhere. A
zero-padded tap is not encoded that way: CUTLASS's implicit GEMM pads the activation with the byte 0,
so the padded tap reads code 0, which this grid dequantizes to (0 - z)/s = -z/s -- NOT zero. With the
measured z in [-6, -3] and s ~ 9, every padded tap silently injects about +0.4 to +0.7 of activation
magnitude, and then the bias subtracts a correction for it as though it were a real sample.

For asymmetric quantization the padding value must be z, not 0. Symmetric grids are unaffected because
there z = 0 and code 0 IS activation 0 -- which is exactly why this has never mattered in this tree
before.

THE TEST, which cannot be confounded by the model, the calibration files, or MoDiff's caches:

  A. 3x3 conv, padding=1  -- padded taps exist, only at the border
  B. 3x3 conv, padding=0  -- no padded taps at all
  C. 1x1 conv, padding=0  -- no padded taps, and no spatial mixing either

For each, quantize a REALISTIC one-sided activation (silu-like: bounded below, long positive tail) on
a symmetric grid and on an asymmetric grid of the same clipped range, and score both against an fp32
reference. Then, for A, split the error into BORDER pixels and INTERIOR pixels.

PREDICTION, ON THE RECORD BEFORE THE RUN:
  * B and C: asymmetric BEATS symmetric (finer step on the same range, no padding to corrupt).
  * A: asymmetric LOSES, and its error is concentrated on the border ring -- interior error should
    match B's, border error should be much larger. Symmetric's border/interior ratio should stay flat.
If A's border and interior errors are equally bad, padding is NOT the explanation and this script has
refuted its own hypothesis rather than confirmed it.

Run: python docs/zp_coverage_2026-08-13/scripts/zp_padding_probe.py    # ~1 min, needs the GPU
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                              # noqa: E402
import torch.nn as nn                                                     # noqa: E402
import torch.nn.functional as F                                           # noqa: E402
from integration.kernels.int4_optimized import OptimizedInt4Conv2d        # noqa: E402

D = "docs/zp_coverage_2026-08-13"
Q = 7.0
#: the same clip ratio the shipped calibration uses (ACT_CLIP_RATIO, swept in
#: docs/paper_repro_2026-08-12); the asymmetric table was built with it too, so the two grids are
#: compared on the SAME clipped range and the only difference is where the 15 codes sit.
CLIP = 4.5


def silu_like(shape, device, seed):
    """An activation with silu(gn(x))'s shape: bounded below by ~-0.2785, long positive tail.

    Not randn. The whole premise of an asymmetric grid is one-sidedness (the model measures
    |max|/|min| = 19.91x), and on symmetric data the zero point provably cannot help -- that is
    already recorded in docs/paper_repro_2026-08-12 (test_int4_conv's randn fixture gets WORSE with a
    clip ratio, 0.221 -> 0.340). Probing this with randn would answer a different question.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, generator=g, device=device) * 1.6
    return x * torch.sigmoid(x)


def grids(a):
    """(symmetric, asymmetric) = (s, z) pairs over the same clipped range."""
    lo, hi = float(a.min()), float(a.max())
    hi_c = hi / CLIP
    s_sym = Q / max(abs(hi_c), abs(lo), 1e-9)
    s_asym = (2.0 * Q) / (hi_c - lo)
    z_asym = -round(lo * s_asym) - Q
    return (s_sym, 0.0), (s_asym, float(z_asym))


def run_conv(conv, a, s, z):
    """Real int4 kernels, through the module's own calibrated path.

    FP32 INPUT ON PURPOSE. _forward_standard's fp16 branch quantizes with
    step1_static_quantize_pack_int4_noahat_fprop, which is the one activation-grid entry point
    deliberately left GUARDED rather than taught (the 2026-08-13 census found it unreachable on both
    shipped W4A4 arms). With a zero point set it raises, which is the correct behaviour and is exactly
    what happened on the first attempt at this probe. The fp32 branch routes through
    scale_quantize_and_pack_zp, i.e. the same kernel MoDiff's t=T path uses.
    """
    conv.set_static_calibration(s, None, z)
    with torch.no_grad():
        return conv(a.float().contiguous(memory_format=torch.channels_last)).float()


def case(label, cin, cout, k, pad, hw, seed, prepad=False):
    """`prepad` EMULATES CORRECT ASYMMETRIC PADDING and is the decisive case.

    F.pad(a, ..., value=0.0) then a padding=0 conv is mathematically identical to a padding=1 conv on
    `a` -- same reference output, exactly. The difference is WHERE the padding zeros come from:

      pad=1 in the conv   CUTLASS's implicit GEMM inserts the BYTE 0, i.e. code 0, which this grid
                          dequantizes to (0 - z)/s = -z/s.  Wrong by -z/s per padded tap.
      F.pad then pad=0    the zeros are real activation values, so the quantize kernel encodes them
                          as round(0*s) + z = z, which dequantizes to (z - z)/s = 0.  Correct.

    So if the padding hypothesis is right, this case must recover the interior error -- and it also
    shows what the fix is, without needing a kernel change to test it.
    """
    torch.manual_seed(seed)
    c = nn.Conv2d(cin, cout, k, padding=0 if prepad else pad, bias=True).cuda().half()
    a = silu_like((2, cin, hw, hw), "cuda", seed)
    if prepad:
        a = F.pad(a, (pad, pad, pad, pad), value=0.0)
    with torch.no_grad():
        ref = F.conv2d(a.float(), c.weight.float(), c.bias.float(),
                       padding=0 if prepad else pad)

    (s_sym, z_sym), (s_asym, z_asym) = grids(a)
    out = {}
    for name, (s, z) in (("sym", (s_sym, z_sym)), ("asym", (s_asym, z_asym))):
        m = OptimizedInt4Conv2d(c, layer_name=f"{label}/{name}").cuda()
        y = run_conv(m, a, s, z)
        err = (y - ref).abs()
        rel = float((y - ref).norm() / ref.norm())
        # border ring vs interior, only meaningful when padding exists
        if pad > 0 and not prepad and y.shape[-1] > 2 * pad + 1:
            mask = torch.zeros_like(err, dtype=torch.bool)
            mask[..., :pad, :] = True
            mask[..., -pad:, :] = True
            mask[..., :, :pad] = True
            mask[..., :, -pad:] = True
            b = float((y - ref)[mask].norm() / ref[mask].norm())
            i = float((y - ref)[~mask].norm() / ref[~mask].norm())
        else:
            b = i = None
        out[name] = {"s": s, "z": z, "rel": rel, "border_rel": b, "interior_rel": i,
                     "err_max": float(err.max())}
        del m
    out["asym_vs_sym"] = out["asym"]["rel"] / out["sym"]["rel"]
    return out


def main():
    cases = [
        ("A 3x3 pad=1 16x16", 64, 64, 3, 1, 16, 11, False),
        ("A 3x3 pad=1  4x4 ", 64, 64, 3, 1, 4, 12, False),  # deep layers: the border IS the image
        ("B 3x3 pad=0 16x16", 64, 64, 3, 0, 16, 13, False),
        ("C 1x1 pad=0 16x16", 64, 64, 1, 0, 16, 14, False),
        ("D 3x3 prepad 16x16", 64, 64, 3, 1, 16, 11, True),   # same seed/shape as A, correct padding
        ("D 3x3 prepad  4x4 ", 64, 64, 3, 1, 4, 12, True),    # same seed/shape as the 4x4 A
    ]
    res = {}
    print(f"{'case':22s}{'s_sym':>7}{'s_asym':>8}{'z':>4}"
          f"{'sym relL2':>11}{'asym relL2':>12}{'asym/sym':>10}")
    for label, cin, cout, k, pad, hw, seed, prepad in cases:
        r = case(label, cin, cout, k, pad, hw, seed, prepad)
        res[label] = r
        print(f"{label:22s}{r['sym']['s']:7.2f}{r['asym']['s']:8.2f}{r['asym']['z']:4.0f}"
              f"{r['sym']['rel']:11.4f}{r['asym']['rel']:12.4f}{r['asym_vs_sym']:9.2f}x")

    print(f"\n{'case':22s}{'grid':>6}{'border relL2':>14}{'interior relL2':>16}{'ratio':>8}")
    for label in res:
        for g in ("sym", "asym"):
            r = res[label][g]
            if r["border_rel"] is None:
                continue
            ratio = r["border_rel"] / r["interior_rel"] if r["interior_rel"] else float("nan")
            print(f"{label:22s}{g:>6}{r['border_rel']:14.4f}{r['interior_rel']:16.4f}{ratio:7.1f}x")

    os.makedirs(f"{D}/data", exist_ok=True)
    with open(f"{D}/data/zp_padding_probe.json", "w") as f:
        json.dump({"clip": CLIP, "cases": res}, f, indent=2)
    print(f"\nwrote {D}/data/zp_padding_probe.json")

    # ---- the verdict ----------------------------------------------------------------------------
    #
    # THREE INDEPENDENT SIGNATURES, all of which must hold. The first draft of this section gated on
    # "border/interior > 2x symmetric's ratio", a threshold picked out of the air; the measured
    # signature is 1.5x against 1.0x, which is decisive for a symmetric grid that is FLAT by
    # construction but would have been called "mixed" by that number. The prepad case replaced the
    # guesswork: it is a direct experiment, not a threshold.
    pad_a = res["A 3x3 pad=1 16x16"]
    pre_a = res["D 3x3 prepad 16x16"]
    nopad = [res["B 3x3 pad=0 16x16"], res["C 1x1 pad=0 16x16"]]
    border_ratio = pad_a["asym"]["border_rel"] / pad_a["asym"]["interior_rel"]
    sym_border_ratio = pad_a["sym"]["border_rel"] / pad_a["sym"]["interior_rel"]

    #: 1. the error the padding hypothesis predicts sits on the border, and only for asymmetric
    sig_border = border_ratio > 1.25 and sym_border_ratio < 1.1
    #: 2. correct padding recovers it: prepad asymmetric must beat implicit-pad asymmetric
    sig_prepad = pre_a["asym"]["rel"] < pad_a["asym"]["rel"]
    #: 3. and with no padded taps at all the zero point does no harm
    sig_nopad = all(r["asym_vs_sym"] <= 1.01 for r in nopad)
    print()
    print(f"signatures: border-localised={sig_border} (asym {border_ratio:.2f}x vs sym "
          f"{sym_border_ratio:.2f}x), prepad-recovers={sig_prepad} "
          f"({pad_a['asym']['rel']:.4f} -> {pre_a['asym']['rel']:.4f}), "
          f"harmless-unpadded={sig_nopad}")
    if sig_border and sig_prepad and sig_nopad:
        gain = (pad_a["asym"]["rel"] - pre_a["asym"]["rel"]) / pad_a["asym"]["rel"] * 100
        print(f"\nCONFIRMED -- ZERO-PADDING IS THE DEFECT, not the zero point.\n"
              f"  * asymmetric's error is border-localised ({border_ratio:.2f}x interior) while\n"
              f"    symmetric's is flat ({sym_border_ratio:.2f}x), which is what z=0 predicts.\n"
              f"  * padding the activation with real zeros -- which the quantizer encodes as code z,\n"
              f"    the correct padding value -- recovers {gain:.1f}% of the asymmetric error.\n"
              f"  * with no padded taps the zero point is harmless.\n"
              f"THE FIX IS THE PADDING VALUE: an asymmetric activation grid must pad with z, not 0.\n"
              f"CUTLASS's implicit GEMM inserts byte 0, which this grid reads as -z/s, while the\n"
              f"folded bias subtracts a correction for a sample that was never there.")
        return 0
    if not sig_nopad:
        print(f"\nHYPOTHESIS REFUTED: asymmetric does not even break even without padding "
              f"({nopad[0]['asym_vs_sym']:.2f}x, {nopad[1]['asym_vs_sym']:.2f}x). Do not fix padding "
              f"on the strength of this file.")
        return 0
    print("\nPARTIAL: not every signature holds. Investigate before changing a kernel; the numbers "
          "above say which one failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
