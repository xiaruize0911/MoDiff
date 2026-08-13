"""Gate for MODIFF_ZP_PREPAD: padding a packed int4 activation with the code z, on the real conv path.

WHY THIS EXISTS. With coverage complete, an asymmetric activation grid measures WORSE end to end
(+82% PTQ, +204% MoDiff) and the cause is zero-padding: CUTLASS's implicit GEMM zero-fills padded taps,
so a padded tap reads code 0, which the grid dequantizes to -z/s rather than 0, while the folded bias
subtracts a per-output-CHANNEL correction for a sample that was never taken
(docs/zp_coverage_2026-08-13/FINDINGS.md).

The correct padding value is code z, and MODIFF_ZP_PREPAD=1 supplies it without a new kernel: pad the
PACKED tensor with the byte whose two nibbles are both z, then run the conv with padding=0 over the
enlarged input. This file gates that mechanism so the end-to-end measurement it enables cannot be a
measurement of a broken emulation.

THREE GATES:

  1. z = 0 IS BIT-IDENTICAL to the normal path, with the flag ON. At z = 0 the pad byte is 0x00, which
     is exactly what CUTLASS inserts, so the two paths must agree to the bit (`torch.equal`). This is
     the gate that catches a wrong pad byte, a wrong slice offset, or a padding/stride mismatch,
     because all three would show up here as a difference on a case whose answer is known.

  2. THE PAD BYTE ENCODES z IN BOTH NIBBLES, two's complement in 4 bits. Checked directly on the padded
     tensor rather than inferred from an output: unpacking the border must give exactly z everywhere,
     for negative z too (z = -5 -> nibble 0xB -> byte 0xBB -> int8 -69).

  3. IT ACTUALLY FIXES THE DEFECT. On a padded conv with a real asymmetric grid, prepad must beat the
     zero-filled path against an fp32 reference. If it does not, the emulation is not doing what
     zp_padding_probe.py showed in isolation and no end-to-end number from it means anything.

Run: python integration/tests/test_int4_zp_prepad.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

#: the padded+asymmetric configuration is the SUBJECT here, so the deployability refusal is overridden
os.environ.setdefault("MODIFF_ZP_ALLOW_PADDED", "1")

import torch                                                              # noqa: E402
import torch.nn as nn                                                     # noqa: E402
import torch.nn.functional as F                                           # noqa: E402
from integration.kernels.int4_optimized import OptimizedInt4Conv2d        # noqa: E402

Q = 7.0
CLIP = 4.5


def silu_like(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, generator=g, device="cuda") * 1.6
    return x * torch.sigmoid(x)


def grids(a):
    lo, hi = float(a.min()), float(a.max())
    hi_c = hi / CLIP
    s_sym = Q / max(abs(hi_c), abs(lo), 1e-9)
    s_asym = (2.0 * Q) / (hi_c - lo)
    return (s_sym, 0.0), (s_asym, float(-round(lo * s_asym) - Q))


def unpack_int4(y, C):
    lo = (y & 0x0F).to(torch.int16)
    hi = ((y >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*y.shape[:-1], C)


def run(conv, a, s, z, mode):
    """`mode` names the padding treatment explicitly: none | halo | border. The default is no longer
    'defective', so a test that only flipped MODIFF_ZP_PREPAD would compare two CORRECT arms."""
    os.environ.pop("MODIFF_ZP_PREPAD", None)
    os.environ["MODIFF_ZP_PAD_MODE"] = mode
    conv.set_static_calibration(s, None, z)
    with torch.no_grad():
        return conv(a.float().contiguous(memory_format=torch.channels_last)).float()


def main():
    fails = []

    # ---- 1. z = 0 is bit-identical with the flag on -------------------------------------------
    print("1. z=0 with MODIFF_ZP_PREPAD=1 is bit-identical to the normal path")
    for (cin, k, pad, hw) in ((64, 3, 1, 16), (128, 3, 1, 8), (64, 3, 1, 4)):
        torch.manual_seed(5)
        c = nn.Conv2d(cin, 64, k, padding=pad, bias=True).cuda().half()
        a = silu_like((2, cin, hw, hw), 5)
        m = OptimizedInt4Conv2d(c, layer_name="probe").cuda()
        (s_sym, _), _ = grids(a)
        y_off = run(m, a, s_sym, 0.0, "none")
        y_on = run(m, a, s_sym, 0.0, "halo")
        ok = torch.equal(y_off, y_on)
        print(f"   C={cin:4d} {hw}x{hw} pad={pad}   {'ok' if ok else 'FAIL'}")
        if not ok:
            d = (y_off - y_on).abs().max()
            print(f"      max |diff| = {float(d):.6f}")
            fails.append(f"z=0 identity C={cin}")

    # ---- 2. the pad byte encodes z in both nibbles --------------------------------------------
    print("\n2. the pad byte decodes to z on the border, for negative z too")
    torch.manual_seed(6)
    c = nn.Conv2d(64, 64, 3, padding=1, bias=True).cuda().half()
    m = OptimizedInt4Conv2d(c, layer_name="probe").cuda()
    for z in (-7.0, -5.0, -1.0, 0.0, 3.0, 7.0):
        m.static_input_zp.fill_(z)
        m._zp_float = z
        packed = torch.zeros(1, 4, 4, 32, dtype=torch.int8, device="cuda")
        out, h, w = m._prepad_packed_with_zp(packed, 4, 4)
        codes = unpack_int4(out, 64)
        border = torch.cat([codes[:, 0, :, :].reshape(-1), codes[:, -1, :, :].reshape(-1),
                            codes[:, :, 0, :].reshape(-1), codes[:, :, -1, :].reshape(-1)])
        interior = codes[:, 1:-1, 1:-1, :]
        ok = bool((border == int(z)).all()) and bool((interior == 0).all()) and (h, w) == (6, 6)
        print(f"   z={z:+5.1f} border all == z, interior untouched, shape {h}x{w}   "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            print(f"      border uniques {border.unique().tolist()}, "
                  f"interior uniques {interior.unique().tolist()}")
            fails.append(f"pad byte z={z}")

    # ---- 2b. the CUDA pad kernel is bit-identical to the eager reference ------------------------
    print("\n2b. pad_packed_int4_code (CUDA, one pass) == the eager torch.full+copy reference")
    import modiff_cutlass as mc
    if not hasattr(mc, "pad_packed_int4_code"):
        print("   MISSING pad_packed_int4_code -- rebuild the extension                FAIL")
        fails.append("pad kernel missing")
    else:
        torch.manual_seed(9)
        m2 = OptimizedInt4Conv2d(nn.Conv2d(64, 64, 3, padding=1, bias=True).cuda().half(),
                                 layer_name="probe").cuda()
        for z in (-7.0, -5.0, -1.0, 0.0, 3.0, 7.0):
            for (H, W, Cb) in ((4, 4, 32), (16, 16, 96), (2, 3, 8)):
                m2._zp_float = z
                xp = torch.randint(-128, 127, (2, H, W, Cb), dtype=torch.int8, device="cuda")
                a_ref, h1, w1 = m2._prepad_packed_with_zp_eager(xp, H, W)
                a_cuda = mc.pad_packed_int4_code(xp, 1, 1, z)
                ok = torch.equal(a_ref, a_cuda)
                if not ok:
                    print(f"   z={z:+5.1f} {H}x{W}x{Cb}   FAIL "
                          f"({int((a_ref != a_cuda).sum())} bytes differ)")
                    fails.append(f"pad kernel z={z} {H}x{W}")
        if not any("pad kernel" in f for f in fails):
            print("   all z in {-7,-5,-1,0,3,7} x 3 shapes, bit-identical            ok")
        #: and it must refuse a code that is not a signed int4 value
        try:
            mc.pad_packed_int4_code(torch.zeros(1, 2, 2, 4, dtype=torch.int8, device="cuda"),
                                    1, 1, 9.0)
            print("   |z| > 7 REFUSES                                              FAIL (returned)")
            fails.append("pad kernel range check")
        except RuntimeError:
            print("   |z| > 7 REFUSES                                              ok")

    # ---- 3. it actually fixes the defect -------------------------------------------------------
    print("\n3. on a padded conv with a real asymmetric grid, prepad beats zero-fill")
    for (cin, hw, seed) in ((64, 16, 11), (64, 4, 12), (128, 8, 13)):
        torch.manual_seed(seed)
        c = nn.Conv2d(cin, 64, 3, padding=1, bias=True).cuda().half()
        a = silu_like((2, cin, hw, hw), seed)
        with torch.no_grad():
            ref = F.conv2d(a.float(), c.weight.float(), c.bias.float(), padding=1)
        (s_sym, _), (s_asym, z_asym) = grids(a)
        m = OptimizedInt4Conv2d(c, layer_name="probe").cuda()
        e_sym = float((run(m, a, s_sym, 0.0, "none") - ref).norm() / ref.norm())
        e_zf = float((run(m, a, s_asym, z_asym, "none") - ref).norm() / ref.norm())
        e_pp = float((run(m, a, s_asym, z_asym, "halo") - ref).norm() / ref.norm())
        e_bc = float((run(m, a, s_asym, z_asym, "border") - ref).norm() / ref.norm())
        #: both corrections must beat the defect, and they must AGREE -- they are two independent
        #: derivations of the same quantity, so a disagreement means one of them is wrong.
        ok = e_pp < e_zf and e_bc < e_zf and abs(e_bc / e_pp - 1.0) < 0.02
        print(f"   C={cin:4d} {hw}x{hw} z={z_asym:+.0f}   sym {e_sym:.4f}   zero-fill {e_zf:.4f}   "
              f"halo {e_pp:.4f}   border {e_bc:.4f}   {'ok' if ok else 'FAIL'}  "
              f"(border/halo {e_bc / e_pp:.3f}x, best vs sym {min(e_pp, e_bc) / e_sym:.2f}x)")
        if not ok:
            fails.append(f"correction C={cin} {hw}")

    os.environ["MODIFF_ZP_PAD_MODE"] = "border"
    print()
    if fails:
        print(f"FAILED ({len(fails)}): {', '.join(fails)}")
        return 1
    print("ALL PASS: the prepad path is bit-identical at z=0, encodes z in both nibbles of the pad\n"
          "byte for negative z included, and reduces the padded conv's error at z != 0. An end-to-end\n"
          "measurement through this flag is therefore measuring the zero point, not a broken emulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
