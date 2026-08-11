"""Which shapes does `flash_attn_int8_packed_vt` accept on its int8 GATHER path? As assertions.

This file exists because the answer was recorded WRONG. `_qkv_i8_ok`'s comment said "hd=24 fails the
cp.async alignment and hd=48 fails the mma eligibility, so route (b) does not currently run on any
block of this model", and flagged the second constraint as one it "had not enumerated". Enumerated
from check_packed (flash_attn_int8.cu), the int8 constraints are exactly:

    hd_pad <= FA_MMA_MAXHD (64)      hd_pad = ceil(hd/32)*32
    T % (FA_MMA_WARPS * FA_MMA_BR) == 0    i.e. T % 64 == 0
    hd % 8 == 0
    (FA_MMA_BC * hd_pad) % 16 == 0
    (hd * sizeof(TIn)) % 16 == 0     i.e. hd % 16 == 0 for int8, always true for fp16

hd=48 satisfies every one of them. The shape that actually raises "mma-eligible shapes only" is
hd=96/T=16 (hd_pad=128 > 64, and T%64 != 0) -- and those 6 blocks never used the custom flash at all,
since _resolve_flash requires hd<=48 and T%64==0. The gate admitted them because it checked only
`head_dim % 16 == 0`, which 96 passes.

Keeping this as a test rather than a comment: a comment cannot fail when the kernel's constraints
change, and this one was believed for a full session.

Run: python integration/tests/test_flash_packed_int8_shapes.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402

DEV = "cuda"

#: (C, T, expect_int8_ok, why) at nh=8, the churches attention shapes. b is small: this checks
#: acceptance, not speed.
CASES = [
    # hd=24 became LEGAL on 2026-08-12 (8-byte cp.async staging) and is still not USED: measured at
    # 2.11x the mma kernel against a 1.44x break-even, i.e. -0.907 ms per block. Legality and
    # profitability are separate questions and this test only answers the first; the second lives in
    # _qkv_i8_ok's hd%16 condition and in docs/aq_fusion_2026-08-12.
    (192, 1024, True, "hd=24 -> 24 B/token: legal via 8 B cp.async, but SLOWER, so not used"),
    (384, 256, True, "hd=48, hd_pad=64: every constraint satisfied"),
    (384, 64, True, "hd=48 at the smallest eligible T"),
    (768, 16, False, "hd=96 -> hd_pad=128 > FA_MMA_MAXHD, and T%64 != 0"),
]


def call(C, T, dtype, b=8, nh=8):
    hd = C // nh
    hd_pad = ((hd + 31) // 32) * 32
    torch.manual_seed(3)
    if dtype == torch.int8:
        qkv = torch.randint(-127, 127, (b, T, nh, 3, hd), device=DEV, dtype=torch.int8)
    else:
        qkv = (torch.randn(b, T, nh, 3, hd, device=DEV, dtype=torch.float16) * 0.5)
    sv = torch.ones(hd, device=DEV, dtype=torch.float32) * 0.02
    return mc.flash_attn_int8_packed_vt(qkv.contiguous(), sv, hd_pad, 0.031, 0.027, hd ** -0.5)


def main():
    print("| C | T | hd | int8 gather | fp16 (route (a)'s path) | note |")
    print("|---|--:|--:|---|---|---|")
    bad = []
    for C, T, expect_ok, why in CASES:
        hd = C // 8
        try:
            out = call(C, T, torch.int8)
            got_i8, msg = True, f"[{out.shape[0]},{out.shape[1]},{out.shape[2]},{out.shape[3]}]"
        except RuntimeError as exc:
            got_i8, msg = False, str(exc).split("\n")[0].split(":")[-1].strip()[:34]
        # The fp16 side is asserted too, because the asymmetry is the whole reason route (a) ran
        # everywhere and route (b) cannot: at 2 bytes/element the 16 B rule is free.
        try:
            call(C, T, torch.float16)
            got_f16 = True
        except RuntimeError:
            got_f16 = False
        print(f"| {C} | {T} | {hd} | {'ok' if got_i8 else 'raises'} | "
              f"{'ok' if got_f16 else 'raises'} | {why} |")
        if got_i8 != expect_ok:
            bad.append(f"{C}x{T} (hd={hd}): int8 gather expected "
                       f"{'accept' if expect_ok else 'reject'}, got "
                       f"{'accept' if got_i8 else f'reject ({msg})'}")
        # hd=48 is where both dtypes must work; T%64 gates fp16 too, so hd96/T16 fails both.
        if expect_ok and not got_f16:
            bad.append(f"{C}x{T}: fp16 rejected a shape the int8 path accepts")

    print()
    if bad:
        print("FAILED:")
        for line in bad:
            print("  -", line)
        return 1
    print("PASS -- 3 of the 4 attention shapes are LEGAL on the int8 gather path; the 10 hd=48 blocks "
          "are the ones where it is also FASTER, and hd=24 is legal-but-slower by 0.907 ms/block.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
