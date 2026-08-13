"""Gates for the three quantize kernels that learned the activation zero point on 2026-08-13.

test_int4_zero_point.py gates the BIAS FOLD (host side). This file gates the KERNELS, and it is the
gate that decides whether fix #2's coverage is real:

    scale_quantize_and_pack_zp                  MoDiff's t=T entry point
    group_norm_silu_quantize_resize_nhwc_zp     PTQ updown ResBlocks (8 layers)
    upsample2x_quantize_pack_noahat_fprop_zp    PTQ standalone Upsample

THREE GATES PER KERNEL, and the third is the one that catches a parameter that is plumbed but unused:

  1. z = 0 IS BIT-IDENTICAL to the non-zp entry point (torch.equal, not allclose). The two arities
     must be indistinguishable at z = 0 or every committed symmetric number moves silently.

  2. z != 0 DIFFERS. Trivial, and it is the gate that would have caught a wrapper that accepted
     `zero_point` and dropped it -- which is a real failure mode here, because the value travels
     through a variadic launch macro (group_norm_silu_quantize_resize_nhwc's MODIFF_GNQR_LAUNCH) where
     a missing argument is a silent default rather than a compile error.

  3. THE SHIFT IS EXACTLY +z ON THE UNCLAMPED CODES. Both arities compute round(a*s) identically and
     then differ only by `+ z` before the same clamp, so for every element whose symmetric code is not
     itself clamped (|c0| < 7) and whose shifted code still fits (|c0 + z| <= 7):

         c_zp == c0 + z          exactly

     This is checked by unpacking the nibbles, and it needs no reimplementation of GroupNorm, SiLU,
     the 2x2 average or the nearest upsample -- which is the point. A reference reimplementation
     would be a second thing that can be wrong; this identity is a property of the kernel pair.
     Elements at the clamp are EXCLUDED rather than predicted, because round(a*s) is unrecoverable
     from a saturated code. The comparable count is printed AND ENFORCED: if a kernel's shift gate
     found nothing comparable in any case it asserted nothing, and main() fails the run rather than
     letting "0 of 0 matched" read as green.

PLUS TWO REFUSAL GATES. The zero point belongs to the activation grid. Two of these kernels can also
be asked to quantize a DELTA (x - a_hat), where z is undefined -- it cancels in a difference -- and
where applying it would additionally corrupt the a_hat update, which dequantizes as q/s. Both must
raise rather than choose:

  4. upsample2x_quantize_pack_noahat_fprop_zp with a NON-EMPTY a_hat cache and z != 0 raises.
  5. group_norm_silu_quantize_resize_nhwc_zp with pack=false (int8 output) and z != 0 raises -- the
     int8 path's bias carries no -z*sum(w_q) correction, so honouring z there IS the mismatch.

Shapes are the model's, not synthetic: the eight updown ResBlock (C, H, W, direction) tuples, and
C in {192, 768, 1536} for the elementwise kernel.

Run: python integration/tests/test_int4_zero_point_kernels.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                              # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402

#: the eight updown ResBlocks of lsun_churches256, same list as test_gn_resize_fusion.py
UPDOWN = [(192, 32, 32, -1), (384, 16, 16, -1), (384, 8, 8, -1), (768, 4, 4, -1),
          (768, 2, 2, +1), (768, 4, 4, +1), (384, 8, 8, +1), (384, 16, 16, +1)]
GROUPS, EPS = 32, 1e-5
ZS = (-4.0, -1.0, 3.0, 7.0)


def unpack_int4(y, C):
    """[..., C/2] packed bytes -> [..., C] signed nibbles as int16."""
    lo = (y & 0x0F).to(torch.int16)
    hi = ((y >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*y.shape[:-1], C)


def shift_gate(c0, cz, z, label, fails, seen):
    """Gate 3: on the codes the shift cannot saturate, c_zp == c0 + z exactly.

    A SKIP IS RECORDED, NOT SWALLOWED. `seen` collects the comparable-element count of every call so
    main() can fail the run if a whole kernel's worth of comparisons was skipped. The docstring at the
    top of this file claims "0 of 0 matched cannot pass as a green gate"; printing the count is not
    that claim, enforcing it is, and the first version of this file only printed."""
    want = c0 + int(z)
    comparable = (c0.abs() < 7) & (want.abs() <= 7)
    n = int(comparable.sum())
    seen.append(n)
    if n == 0:
        print(f"   {label:52s} SKIP (no unclamped comparable codes)")
        return
    ok = bool(torch.equal(cz[comparable], want[comparable]))
    frac = 100.0 * n / c0.numel()
    print(f"   {label:52s} {'ok' if ok else 'FAIL'}  "
          f"({n} of {c0.numel()} codes comparable, {frac:.0f}%)")
    if not ok:
        bad = (cz[comparable] != want[comparable]).sum()
        print(f"      {bad} of {n} shifted codes wrong")
        fails.append(label)


def test_scale_quantize_and_pack(fails, seen):
    print("1. scale_quantize_and_pack_zp  (MoDiff t=T)")
    torch.manual_seed(20260813)
    for C in (192, 768, 1536):
        x = (torch.randn(2, C, 8, 8, device="cuda") * 2.0).contiguous(
            memory_format=torch.channels_last)
        s = torch.tensor([3.3], device="cuda")
        base = mc.scale_quantize_and_pack(x, s)
        z0 = mc.scale_quantize_and_pack_zp(x, s, 0.0)
        ok = torch.equal(base, z0)
        print(f"   C={C:5d} z=0 bit-identical                          "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"sqp z=0 C={C}")
        c0 = unpack_int4(base, C)
        for z in ZS:
            zz = mc.scale_quantize_and_pack_zp(x, s, z)
            if torch.equal(base, zz):
                print(f"   C={C:5d} z={z:+5.1f} DIFFERS from symmetric              FAIL")
                fails.append(f"sqp z={z} inert C={C}")
                continue
            shift_gate(c0, unpack_int4(zz, C), z, f"C={C} z={z:+5.1f} shift is exactly +z",
                       fails, seen)
    # This kernel is 4D channels_last ONLY -- CHECK_CONTIGUOUS enforces it, so a 2D activation
    # raises rather than being silently reinterpreted. Gate the refusal so a future "helpful"
    # relaxation has to be deliberate.
    x2 = (torch.randn(64, 256, device="cuda") * 2.0).contiguous()
    s = torch.tensor([2.0], device="cuda")
    try:
        mc.scale_quantize_and_pack_zp(x2, s, 0.0)
        print("   2D input REFUSES (channels_last only)             FAIL (it returned)")
        fails.append("sqp 2D refusal")
    except RuntimeError:
        print("   2D input REFUSES (channels_last only)             ok")


def test_gn_resize(fails, seen):
    print("\n2. group_norm_silu_quantize_resize_nhwc_zp  (PTQ updown)")
    torch.manual_seed(4)
    dev = "cuda"
    for (C, H, W, d) in UPDOWN:
        x = torch.randn(2, C, H, W, device=dev, dtype=torch.float16).contiguous(
            memory_format=torch.channels_last)
        g = torch.randn(C, device=dev, dtype=torch.float16)
        b = torch.randn(C, device=dev, dtype=torch.float16)
        s = torch.tensor([4.0], device=dev)
        e = torch.empty(0, device=dev, dtype=torch.float32)
        em = x.new_empty(0)
        args = (x, g, b, GROUPS, EPS, True, s, e, em, em, 0, d, True)
        base = mc.group_norm_silu_quantize_resize_nhwc(*args)
        z0 = mc.group_norm_silu_quantize_resize_nhwc_zp(*args, 0.0)
        ok = torch.equal(base, z0)
        tag = "up  " if d > 0 else "down"
        print(f"   C={C:5d} {H}x{W} {tag} z=0 bit-identical             {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"gnr z=0 C={C} d={d}")
        c0 = unpack_int4(base, C)
        for z in (-4.0, 3.0):
            zz = mc.group_norm_silu_quantize_resize_nhwc_zp(*args, z)
            if torch.equal(base, zz):
                print(f"   C={C:5d} {H}x{W} {tag} z={z:+5.1f} DIFFERS                  FAIL")
                fails.append(f"gnr z={z} inert C={C}")
                continue
            shift_gate(c0, unpack_int4(zz, C), z,
                       f"C={C} {H}x{W} {tag} z={z:+5.1f} shift is exactly +z", fails, seen)
    # Gate 5: int8 output has no bias correction for z -> must refuse.
    x = torch.randn(2, 192, 8, 8, device=dev, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    g = torch.randn(192, device=dev, dtype=torch.float16)
    b = torch.randn(192, device=dev, dtype=torch.float16)
    s = torch.tensor([4.0], device=dev)
    e = torch.empty(0, device=dev, dtype=torch.float32)
    em = x.new_empty(0)
    try:
        mc.group_norm_silu_quantize_resize_nhwc_zp(x, g, b, GROUPS, EPS, True, s, e, em, em,
                                                   0, -1, False, 3.0)
        print("   pack=false + z!=0 REFUSES                          FAIL (it returned)")
        fails.append("gnr int8 refusal")
    except RuntimeError:
        print("   pack=false + z!=0 REFUSES                          ok")
    # ... and pack=false with z == 0 must still work, i.e. the check is on z, not on pack.
    try:
        mc.group_norm_silu_quantize_resize_nhwc_zp(x, g, b, GROUPS, EPS, True, s, e, em, em,
                                                   0, -1, False, 0.0)
        print("   pack=false + z==0 still works                      ok")
    except RuntimeError as ex:
        print(f"   pack=false + z==0 still works                      FAIL ({ex})")
        fails.append("gnr int8 z=0")


def test_upsample(fails, seen):
    print("\n3. upsample2x_quantize_pack_noahat_fprop_zp  (PTQ Upsample)")
    torch.manual_seed(7)
    dev = "cuda"
    for C, H, W in ((192, 16, 16), (768, 4, 4), (1536, 2, 2)):
        x = torch.randn(2, C, H, W, device=dev, dtype=torch.float16).contiguous(
            memory_format=torch.channels_last)
        s = torch.tensor([3.0], device=dev)
        e = torch.empty(0, device=dev, dtype=torch.float32)
        noah = torch.empty(0, device=dev, dtype=torch.float16)
        base = mc.upsample2x_quantize_pack_noahat_fprop(x, s, e, noah)
        z0 = mc.upsample2x_quantize_pack_noahat_fprop_zp(x, s, e, noah, 0.0)
        ok = torch.equal(base, z0)
        print(f"   C={C:5d} {H}x{W} z=0 bit-identical                  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"ups z=0 C={C}")
        c0 = unpack_int4(base, C)
        for z in (-4.0, 3.0):
            zz = mc.upsample2x_quantize_pack_noahat_fprop_zp(x, s, e, noah, z)
            if torch.equal(base, zz):
                print(f"   C={C:5d} z={z:+5.1f} DIFFERS                            FAIL")
                fails.append(f"ups z={z} inert C={C}")
                continue
            shift_gate(c0, unpack_int4(zz, C), z, f"C={C} z={z:+5.1f} shift is exactly +z",
                       fails, seen)
    # Gate 4: with an a_hat cache this kernel quantizes a DELTA -> refuse a non-zero z.
    C, H, W = 192, 8, 8
    x = torch.randn(2, C, H, W, device=dev, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    s = torch.tensor([3.0], device=dev)
    e = torch.empty(0, device=dev, dtype=torch.float32)
    ah = torch.zeros(2, C, H * 2, W * 2, device=dev, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    try:
        mc.upsample2x_quantize_pack_noahat_fprop_zp(x, s, e, ah, 3.0)
        print("   a_hat present + z!=0 REFUSES                       FAIL (it returned)")
        fails.append("ups delta refusal")
    except RuntimeError:
        print("   a_hat present + z!=0 REFUSES                       ok")
    ah2 = ah.clone()
    try:
        r0 = mc.upsample2x_quantize_pack_noahat_fprop(x, s, e, ah)
        r1 = mc.upsample2x_quantize_pack_noahat_fprop_zp(x, s, e, ah2, 0.0)
        ok = torch.equal(r0, r1) and torch.equal(ah, ah2)
        print(f"   a_hat present + z==0 identical incl. cache update  {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append("ups delta z=0")
    except RuntimeError as ex:
        print(f"   a_hat present + z==0 identical                     FAIL ({ex})")
        fails.append("ups delta z=0")


def main():
    for n in ("scale_quantize_and_pack_zp", "group_norm_silu_quantize_resize_nhwc_zp",
              "upsample2x_quantize_pack_noahat_fprop_zp"):
        if not hasattr(mc, n):
            print(f"MISSING ENTRY POINT {n} -- rebuild the extension")
            return 1
    fails = []
    #: comparable-element counts per kernel, so an all-skipped kernel fails instead of reading green
    seen = {"scale_quantize_and_pack": [], "gn_resize": [], "upsample": []}
    test_scale_quantize_and_pack(fails, seen["scale_quantize_and_pack"])
    test_gn_resize(fails, seen["gn_resize"])
    test_upsample(fails, seen["upsample"])
    for k, v in seen.items():
        if not v or all(n == 0 for n in v):
            print(f"\nVACUOUS: {k}'s shift gate had no comparable codes in any case, so it asserted "
                  f"nothing. Treating that as a failure rather than a pass.")
            fails.append(f"{k}: shift gate vacuous")
    print()
    if fails:
        print(f"FAILED ({len(fails)}): {', '.join(fails)}")
        return 1
    print("ALL PASS: all three kernels are bit-identical at z=0, responsive to z!=0, shift the\n"
          "unclamped codes by exactly +z, and refuse a zero point on a delta quantize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
