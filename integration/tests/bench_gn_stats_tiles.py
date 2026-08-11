"""Second attempt at the GN-stats-in-epilogue reduction: does a warp tree beat shared atomics?

The 2026-08-11 prototype accumulated per-(n, group) sum/sumsq with two shared `atomicAdd` per element
and failed both gates -- 6.5x too slow, and `det=False` on every shape -- because 23-56 slots with 256
contending threads serialize, and float atomicAdd is order-dependent. The kernel now does a SEGMENTED
WARP REDUCTION instead (`__match_any_sync` groups the lanes sharing a slot, a masked butterfly sums
them, one leader per group writes to its own warp's private slots) and sums the warps in a fixed order.
No atomics anywhere.

THREE GATES, in the order docs/gn_stats_in_epilogue_2026-08-11 sets them, because each can kill it:

  1. Correctness -- partials, summed over blocks, against a torch fp32 reference per (n, group).
     Compared against the reference rather than against the shipped kernel because
     `gn_stats_partials_chanmajor_kernel` has no pybind entry; the reference is stricter anyway.
  2. Determinism -- 10 launches, `torch.equal` on both outputs. This is the gate the last attempt
     failed, and the one the rewrite is FOR.
  3. Speed -- must beat the pass it replaces. The shipped per-shape numbers are quoted from that
     report (it measured them; they are not re-measurable here without a pybind entry), and the
     weighted total is computed the SAME way for both columns so the ratio is apples to apples.

Run: python integration/tests/bench_gn_stats_tiles.py [--batch 128]
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402

DEV = "cuda"
GROUPS = 32                                       # GroupNorm(32, C) throughout this UNet
MT, NT = 128, 128                                 # the verified conv epilogue tile

#: (C, H, W, count, shipped_us, atomics_prototype_us) -- the last two from
#: docs/gn_stats_in_epilogue_2026-08-11's own table, so the three columns line up row by row.
SHAPES = [
    (192, 32, 32, 14, 476.2, 542.7),
    (384, 32, 32, 4, 771.5, 1439.0),
    (768, 16, 16, 4, 416.6, 1277.3),
    (768, 4, 4, 10, 52.5, 237.0),
]


def reference(x, G):
    """sum and sumsq per (n, group), fp32, from torch. The arithmetic ground truth.

    Works off the LOGICAL tensor, so it is layout-agnostic -- which the kernel is not. See make_x.
    """
    N, C = x.shape[0], x.shape[1]
    xf = x.float().reshape(N, G, C // G * x.shape[2] * x.shape[3])
    return xf.sum(-1), (xf * xf).sum(-1)


def make_x(B, C, H, W):
    """CHANNELS_LAST, and this is load-bearing.

    The kernel indexes `X[m * C + c]` with m over N*H*W, i.e. it reads a [N,H,W,C] buffer -- the
    layout the whole MoDiff conv path uses. Handing it a contiguous NCHW tensor does not crash and
    does not lose any element: it reads each one exactly once, into the WRONG (n, group) bucket. The
    first version of this test did exactly that and reported "max rel err 1.045e-01" while total mass
    matched to 4 digits (3010.3 vs 3010.3), which is the signature of a permutation rather than a
    reduction fault -- and it was nearly attributed to the reduction.
    """
    return (torch.randn(B, C, H, W, device=DEV, dtype=torch.float16) * 0.7
            ).contiguous(memory_format=torch.channels_last)


def time_us(fn, iters=20, warmups=3):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print(f"{torch.cuda.get_device_name(0)}, batch {args.batch}, tile {MT}x{NT}, G={GROUPS}\n")
    print("| C | HxW | n | tree us | shipped us | atomics us | tree/shipped | atomics/shipped "
          "| max rel err | det |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|---|")
    bad, rows = [], []
    w_tree = w_ship = w_atom = 0.0
    for C, H, W, count, ship_us, atom_us in SHAPES:
        x = make_x(args.batch, C, H, W)
        with torch.inference_mode():
            ps, pq = mc.gn_stats_from_tiles(x, GROUPS, MT, NT)
            nblk = ps.numel() // (args.batch * GROUPS)
            got_s = ps.reshape(args.batch, GROUPS, nblk).sum(-1)
            got_q = pq.reshape(args.batch, GROUPS, nblk).sum(-1)
            ref_s, ref_q = reference(x, GROUPS)
            # sumsq is strictly positive -> plain relative error. sum is signed and centred near
            # zero, so dividing by it inflates a correct result without bound (the first version of
            # this test read 1e3 on a kernel accurate to 2.5e-7); normalise by sqrt(sumsq), the scale
            # a sum of that many terms actually has.
            err = max(((got_s - ref_s).abs() / ref_q.sqrt()).max().item(),
                      ((got_q - ref_q).abs() / ref_q).max().item())
            det = True
            for _ in range(10):
                ps2, pq2 = mc.gn_stats_from_tiles(x, GROUPS, MT, NT)
                if not (torch.equal(ps, ps2) and torch.equal(pq, pq2)):
                    det = False
                    break
            us = time_us(lambda: mc.gn_stats_from_tiles(x, GROUPS, MT, NT))
        w_tree += count * us
        w_ship += count * ship_us
        w_atom += count * atom_us
        print(f"| {C} | {H}x{W} | {count} | {us:.1f} | {ship_us:.1f} | {atom_us:.1f} | "
              f"{us / ship_us:.2f}x | {atom_us / ship_us:.2f}x | {err:.2e} | "
              f"{'ok' if det else 'FAIL'} |")
        rows.append(dict(C=C, H=H, W=W, count=count, tree_us=us, shipped_us=ship_us,
                         atomics_us=atom_us, max_rel_err=err, deterministic=det))
        if not det:
            bad.append(f"{C}x{H}x{W}: NONDETERMINISTIC")
        # 1e-5: both paths accumulate fp16 inputs in fp32, so this bounds reduction-order effects
        # rather than precision. Measured 2.5e-7 for the tree reduction.
        if err > 1e-5:
            bad.append(f"{C}x{H}x{W}: max rel err {err:.2e}")

    print(f"\ncount-weighted (sum of count x per-shape us, all three columns the same way):")
    print(f"  tree     {w_tree / 1e3:7.2f} ms   {w_tree / w_ship:.2f}x shipped")
    print(f"  shipped  {w_ship / 1e3:7.2f} ms")
    print(f"  atomics  {w_atom / 1e3:7.2f} ms   {w_atom / w_ship:.2f}x shipped  (2026-08-11)")
    verdict = ("BEATS the shipped pass" if w_tree < w_ship
               else f"still LOSES by {w_tree / w_ship:.2f}x")
    print(f"\nSpeed gate: {verdict}. Improvement over the atomics prototype: "
          f"{w_atom / w_tree:.2f}x")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(gpu=torch.cuda.get_device_name(0), batch=args.batch, groups=GROUPS,
                           tile=[MT, NT], rows=rows,
                           weighted_ms=dict(tree=w_tree / 1e3, shipped=w_ship / 1e3,
                                            atomics=w_atom / 1e3),
                           tree_over_shipped=w_tree / w_ship,
                           atomics_over_shipped=w_atom / w_ship,
                           beats_shipped=bool(w_tree < w_ship)), f, indent=1)
        print(f"wrote {args.json}")

    if bad:
        print("\nGATE NOT MET:")
        for line in bad:
            print("  -", line)
        return 1
    print("\nCorrectness and determinism gates met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
