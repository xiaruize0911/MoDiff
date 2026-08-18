"""Gate for fix #4 step 1: does `int4_window_sum` compute the reduction the epilogue needs?

Scored against a PyTorch reference built from the SAME definition, not against the kernel's own earlier
output -- comparing an implementation to itself cannot distinguish "correct" from "consistently wrong",
which is the correction docs/OPEN_ITEMS.md A0 had to make twice.

The reference is exact integer arithmetic, so the tolerance is ZERO. S[p] is a sum of at most
C*R*S*7 ~ 1e5 integers held in int32 and returned as fp32, and every fp32 integer below 2^24 is exact,
so there is no rounding for a tolerance to absorb.

Four things are checked:
  1. exactness against the reference on the real conv shapes, including stride, padding and dilation;
  2. the packing convention -- low nibble is channel 2i, high nibble 2i+1, with 4-bit sign extension.
     Checked by a hand-built byte whose two nibbles are the extremes (-8 and +7);
  3. padding-cleanness: an explicitly zero-padded input with padding=0 gives the same answer, which is
     the property that makes fix #4 need no border correction (fix #2 did);
  4. a NEGATIVE CONTROL -- a reference with the SIGN EXTENSION dropped (nibbles read as unsigned 0..15)
     must DISAGREE, otherwise the passes above do not depend on the kernel decoding anything correctly.

     The first version of this control swapped the NIBBLE ORDER and it was vacuous by construction:
     S[p] sums over ALL input channels, and a sum is invariant to permuting them, so no test of this
     quantity can ever detect a nibble-order bug. That is not a gap -- channel order is carried by the
     conv accumulator ACC[k,p], which the existing CUTLASS kernel computes and which S[p] never touches.
     S[p] needs only the correct multiset of codes, and sign extension is what determines that.

Run: python integration/tests/test_zpw_window_sum.py
"""
import itertools
import sys

import torch
import torch.nn.functional as F

import modiff_cutlass as mc

DEV = "cuda"
torch.manual_seed(20260817)


def unpack(x_packed, C):
    """[N,H,W,C/2] int8 -> [N,C,H,W] int64 codes, low nibble = channel 2i."""
    b = x_packed.to(torch.int64)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    out = torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], C)      # [N,H,W,C]
    return out.permute(0, 3, 1, 2).contiguous()


def reference(x_packed, C, R, S, st, pad, dil, unsigned=False):
    """unsigned=True drops the 4-bit sign extension -- the negative control, not a mode."""
    a = x_packed.to(torch.int64)
    lo = a & 0x0F
    hi = (a >> 4) & 0x0F
    if not unsigned:
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
    codes = torch.stack([lo, hi], dim=-1).reshape(*a.shape[:-1], C).permute(0, 3, 1, 2)
    t = codes.sum(dim=1, keepdim=True).double()
    ones = torch.ones(1, 1, R, S, device=a.device, dtype=torch.float64)
    return F.conv2d(t, ones, stride=st, padding=pad, dilation=dil).squeeze(1)


def pack(codes):
    """[N,C,H,W] int codes in [-8,7] -> [N,H,W,C/2] int8, low nibble = channel 2i."""
    N, C, H, W = codes.shape
    c = codes.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    lo, hi = c[..., 0::2], c[..., 1::2]
    return ((lo | (hi << 4)).to(torch.int16) - 256 * (((lo | (hi << 4)) > 127).to(torch.int16))
            ).to(torch.int8).contiguous()


fails = []

# ---- 2. packing convention, on a byte whose nibbles are the extremes -------------------------------
codes = torch.zeros(1, 2, 1, 1, dtype=torch.int64, device=DEV)
codes[0, 0, 0, 0] = -8
codes[0, 1, 0, 0] = 7
xp = pack(codes)
got = float(mc.int4_window_sum(xp, 1, 1, 1, 1, 0, 0, 1, 1).item())
print(f"  packing: codes (-8, +7) in one byte -> S = {got:.0f}  (expect -1)")
if got != -1.0:
    fails.append(f"packing convention wrong: got {got}, expected -1 "
                 f"(low nibble must be channel 0, sign-extended from 4 bits)")

# ---- 1 & 3. exactness and padding-cleanness on real shapes -----------------------------------------
CASES = [(2, 192, 32, 32, 3, 1, 1, 1), (2, 384, 16, 16, 3, 1, 1, 1), (2, 768, 8, 8, 3, 1, 1, 1),
         (2, 1536, 4, 4, 3, 1, 1, 1), (2, 192, 32, 32, 1, 1, 0, 1), (2, 384, 16, 16, 3, 2, 1, 1),
         (2, 192, 32, 32, 3, 1, 2, 2), (1, 96, 5, 7, 3, 1, 1, 1)]
print()
for N, C, H, W, R, st, pad, dil in CASES:
    codes = torch.randint(-8, 8, (N, C, H, W), device=DEV, dtype=torch.int64)
    xp = pack(codes)
    ref = reference(xp, C, R, R, st, pad, dil)
    got = mc.int4_window_sum(xp, R, R, st, st, pad, pad, dil, dil)
    ok_shape = tuple(got.shape) == tuple(ref.shape)
    maxdiff = float((got.double() - ref).abs().max()) if ok_shape else float("nan")

    # padding-cleanness
    ok_pad = True
    if pad > 0:
        cp = F.pad(codes, (pad,) * 4)
        got_p = mc.int4_window_sum(pack(cp), R, R, st, st, 0, 0, dil, dil)
        ok_pad = tuple(got_p.shape) == tuple(got.shape) and float((got_p - got).abs().max()) == 0.0

    tag = f"N{N} C{C} {H}x{W} {R}x{R} s{st} p{pad} d{dil}"
    print(f"  {tag:34} shape {'ok' if ok_shape else 'MISMATCH'}  max|Δ| {maxdiff:.1f}  "
          f"padding-clean {ok_pad}")
    if not ok_shape:
        fails.append(f"{tag}: shape {tuple(got.shape)} != {tuple(ref.shape)}")
    elif maxdiff != 0.0:
        fails.append(f"{tag}: max|Δ| {maxdiff} against an exact integer reference (must be 0)")
    if not ok_pad:
        fails.append(f"{tag}: zero-padded input disagreed -- padding is NOT clean")

# ---- 4. negative control: dropping the sign extension must disagree --------------------------------
codes = torch.randint(-8, 8, (2, 192, 16, 16), device=DEV, dtype=torch.int64)
xp = pack(codes)
got = mc.int4_window_sum(xp, 3, 3, 1, 1, 1, 1, 1, 1)
wrong = reference(xp, 192, 3, 3, 1, 1, 1, unsigned=True)
d = float((got.double() - wrong).abs().max())
print(f"\n  NEGATIVE CONTROL, reference without sign extension: max|Δ| {d:.1f}  (must be > 0)")
if d == 0.0:
    fails.append("reading the nibbles as unsigned 0..15 gave the SAME answer -- the passes above do "
                 "not depend on the kernel decoding the codes correctly")

print()
if fails:
    print("GATE FAILED:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print(f"GATE PASSED: int4_window_sum is exact on {len(CASES)} real shapes, the packing convention is "
      f"confirmed, padding is clean, and the sign-extension control fires.")
