"""Gate for the vec2 a_hat access in group_norm_silu_delta_quantize_resize_nhwc_kernel.

WHAT CHANGED. Both the UP and DOWN paths used to read a_hat as two scalar __half loads at `ci` and
`ci + 1` and write it back as two scalar __float2half_rn stores. They now use gn_load2/gn_store2 --
one naturally-aligned 4-byte access each. Same values, same rounding (__float22half2_rn rounds each
component to nearest even, exactly as the two __float2half_rn calls did), so the output must be
BIT-IDENTICAL. Measured 3.184 -> 2.719 ms/step in-model (nsys, batch 128, W8A8 MoDiff).

WHY THIS IS A TWO-BUILD TEST AND NOT AN ENV FLAG. The access sits in the innermost loop of a
heavily templated kernel; a runtime branch there would cost more than the change saves, and a
template bool would double an already large instantiation set. So the reference is captured from a
build of the previous commit:

    git stash push csrc/                 # or: git checkout <prev> -- csrc/
    python setup.py build_ext --inplace
    python integration/tests/test_gn_resize_ahat_vec2.py --capture /tmp/resize_ref.pt
    git stash pop                        # restore the vec2 version
    python setup.py build_ext --inplace
    python integration/tests/test_gn_resize_ahat_vec2.py --compare /tmp/resize_ref.pt

WHY NOT AN END-TO-END IMAGE COMPARISON. Tried, and it cannot decide anything: two runs of the SAME
build produce different images. Measured here on 2026-08-26 -- HEAD against itself gave sha256
4f15fc19... and 44dbe9b2... over 64 images. That is OPEN_ITEMS A18/A19 (the cross-process floor is
8.7/255), and it would have read as "the change broke correctness" if the control had been skipped.
Kernel-level capture with a fixed seed is the instrument that resolves.

COVERAGE: 4 shapes x {upsample, downsample} x {int8, packed int4} x {mod on, mod off} = 32 cases,
because resize direction, PACK and mod each select a different branch through compute_pair.
"""
import argparse
import sys

sys.path.insert(0, "/workspace/MoDiff/build/lib.linux-x86_64-cpython-311")
import torch
import modiff_cutlass as mc

N, G = 32, 32
SHAPES = [(384, 16, 16), (192, 32, 32), (768, 8, 8), (576, 16, 16)]
CASES = [(C, H, W, r, p, m) for (C, H, W) in SHAPES for r in (1, -1)
         for p in (False, True) for m in (False, True)]


def run_all():
    res = {}
    for i, (C, H, W, resize, pack, mod) in enumerate(CASES):
        torch.manual_seed(1000 + i)          # identical input bits in both builds
        Ho, Wo = (H * 2, W * 2) if resize > 0 else (H // 2, W // 2)
        cl = torch.channels_last
        x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=cl)
        a = (0.1 * torch.randn(N, C, Ho, Wo, device="cuda", dtype=torch.float16)).to(memory_format=cl)
        g = torch.randn(C, device="cuda", dtype=torch.float16)
        b = torch.randn(C, device="cuda", dtype=torch.float16)
        sc = torch.tensor([64.0], device="cuda", dtype=torch.float32)
        e16 = torch.empty(0, device="cuda", dtype=torch.float16)
        e32 = torch.empty(0, device="cuda", dtype=torch.float32)
        ei = torch.empty(0, device="cuda", dtype=torch.int32)
        ms = torch.randn(N, C, device="cuda", dtype=torch.float16) * 0.1 if mod else e16
        sh = torch.randn(N, C, device="cuda", dtype=torch.float16) * 0.1 if mod else e16
        y = mc.group_norm_silu_delta_quantize_resize_nhwc(
            x, g, b, G, 1e-5, True, sc, e32, ms, sh, 0, resize, pack, a,
            e32, e32, e32, ei, 7.0 if pack else 127.0, False, 1.0, pack)
        res[str((C, H, W, resize, pack, mod))] = (y.cpu().clone(), a.cpu().clone())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", metavar="PATH")
    ap.add_argument("--compare", metavar="PATH")
    args = ap.parse_args()
    if not (args.capture or args.compare):
        ap.error("pass --capture PATH (on the reference build) or --compare PATH")
    res = run_all()
    if args.capture:
        torch.save(res, args.capture)
        print(f"captured {len(res)} cases -> {args.capture}")
        return
    ref = torch.load(args.compare)
    bad, nz_tot = [], 0
    for k in ref:
        (ya, aa), (yb, ab) = ref[k], res[k]
        oky, oka = torch.equal(ya, yb), torch.equal(aa, ab)
        nz = int((yb != 0).sum())            # non-vacuity: all-zero codes would pass trivially
        nz_tot += nz
        if not (oky and oka and nz > 0):
            bad.append((k, oky, oka, nz))
    print(f"cases {len(ref)}   bit-identical {len(ref) - len(bad)}   nonzero codes {nz_tot:,}")
    if bad:
        for x in bad:
            print("  FAIL", x)
        sys.exit(1)
    print("PASS -- every code and every a_hat entry bit-identical to the reference build")


if __name__ == "__main__":
    main()
