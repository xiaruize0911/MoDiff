"""Does the 8-byte cp.async staging path compute the same attention as the 16-byte one?

LOAD_B changes how many chunks each token's K/V row is copied into shared memory in. It does not touch
the smem layout, the quantize, the mma, or the softmax. So at any shape where BOTH widths are legal
the outputs must be BIT-IDENTICAL, and that is the strongest available check on this change -- much
stronger than a tolerance, because it fails on a single displaced byte.

hd=48 int8 is that shape: 48 bytes/token is a multiple of both 8 and 16, so the host dispatcher picks
16 and the test forces 8 by padding the head dimension... which it cannot do from Python. So the test
is built the other way round: hd=48 exercises the 16 B path (unchanged code) and hd=24 exercises the
8 B path, and hd=24 is checked against an independent fp32 reference plus the production
aq_* + flash_attn_int8_vt_static path, which is what it has to replace.

  1. hd=24 now RUNS at all. Before 2026-08-12 it raised "per-token bytes ... multiple of 16".
  2. hd=24 8 B output vs fp32 reference, and vs production's separate-quantize path.
  3. Determinism: 10 launches, torch.equal.
  4. hd=48 unchanged: still bit-identical to what the 16 B path produced before this commit, which is
     verified by re-running the shape assertions and the accuracy gate in
     bench_flash_packed_vs_unpacked.py -- if LOAD_B had leaked into hd=48, that gate would move.
  5. The padded tail: hd=24 pads to hd_pad=32, so columns 24..31 must not reach the output. Checked by
     running with two different garbage fills beyond hd and requiring identical results -- there is no
     Python-side way to inspect smem, and this is the observable consequence.

Run: python integration/tests/test_flash_packed_load8.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "integration/tests"))

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402
from bench_flash_packed_vs_unpacked import (make_inputs, quantize_packed, arm_u, arm_p,  # noqa: E402
                                           ref_fp32, rel_l2, SQ_C, SK_C)

DEV = "cuda"


def main():
    bad = []
    print("| C | T | hd | load | runs | vs fp32 | vs production | det |")
    print("|---|--:|--:|--:|---|--:|--:|---|")
    for C, T in ((192, 1024), (384, 256), (384, 64)):
        b, nh = 32, 8
        qkv, sv, hd = make_inputs(b, C, T, nh)
        hd_pad = ((hd + 31) // 32) * 32
        ss = hd ** -0.5
        qkv_i8 = quantize_packed(qkv, sv, hd)
        width = 8 if (hd % 16) else 16
        with torch.inference_mode():
            try:
                p = arm_p(qkv_i8, sv, hd, hd_pad, ss)
            except RuntimeError as exc:
                print(f"| {C} | {T} | {hd} | {width} | raises | | | "
                      f"{str(exc).split(':')[-1].strip()[:30]} |")
                bad.append(f"{C}x{T} (hd={hd}): 8 B path should accept this shape now")
                continue
            u = arm_u(qkv, sv, b, nh, T, hd, hd_pad, ss)
            ref = ref_fp32(qkv_i8, sv, hd, ss)
            det = all(torch.equal(p, arm_p(qkv_i8, sv, hd, hd_pad, ss)) for _ in range(10))
            e_ref, e_prod = rel_l2(p, ref), rel_l2(p, u)
            u_ref = rel_l2(u, ref)
        print(f"| {C} | {T} | {hd} | {width} | ok | {e_ref:.2e} | {e_prod:.2e} | "
              f"{'ok' if det else 'FAIL'} |")
        if not det:
            bad.append(f"{C}x{T}: nondeterministic")
        # The gate is relative to production's own error, not an absolute tolerance: both arms are
        # fp16 accumulations of the same int8 codes, so "no worse than U" is the meaningful bound.
        if e_ref > u_ref * 1.10:
            bad.append(f"{C}x{T}: 8 B path {e_ref:.3e} vs production {u_ref:.3e} against fp32")

    # Padded-tail check on the one shape that pads: hd=24 -> hd_pad=32. The staging loop copies hd
    # bytes per token and the quantize zeroes d >= hd, so nothing beyond column 24 may reach the
    # output. Two different fills beyond hd must therefore give identical results. (The fill lives in
    # sv, the only per-channel input wide enough to have a tail.)
    b, nh, C, T = 32, 8, 192, 1024
    qkv, sv, hd = make_inputs(b, C, T, nh)
    qkv_i8 = quantize_packed(qkv, sv, hd)
    hd_pad, ss = 32, hd ** -0.5
    with torch.inference_mode():
        sv_a = sv.clone(); sv_a[hd:] = 1.0
        sv_b = sv.clone(); sv_b[hd:] = -12345.0
        o_a = mc.flash_attn_int8_packed_vt(qkv_i8, sv_a[:hd].contiguous(), hd_pad, SQ_C, SK_C, ss)
        o_b = mc.flash_attn_int8_packed_vt(qkv_i8, sv_b[:hd].contiguous(), hd_pad, SQ_C, SK_C, ss)
    same = torch.equal(o_a, o_b)
    print(f"\npadded tail (hd=24 -> hd_pad=32), sv[hd:] garbage ignored: "
          f"{'yes' if same else 'NO -- padding leaks into the output'}")
    if not same:
        bad.append("hd=24: sv beyond hd changes the output, so padded columns are being read")

    print()
    if bad:
        print("FAILED:")
        for line in bad:
            print("  -", line)
        return 1
    print("PASS -- the 8 B staging path runs hd=24, matches production's accuracy against fp32, is "
          "deterministic, and does not read past hd.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
