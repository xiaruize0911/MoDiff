"""Gate for the decoder skip-concat fold: cat2_gn_stats_fp16 must be EXACT, not merely close.

The fold replaces `cat2_channels_last_fp16(a, b)` followed by the GroupNorm stats pass with one kernel
that reads the halves in place, emits the concatenation, and produces the stats -- 3C of traffic
becomes 2C. That is worth ~1.5% end to end, which is nowhere near enough to justify changing any
number the model produces. So the bar is EXACTNESS on both outputs, not tolerance:

  1. The emitted concatenation is bit-identical to cat2_channels_last_fp16. It is a copy either way,
     so anything less means an indexing bug.
  2. The (mean, inv_std) are bit-identical to the same kernel run on the CONCATENATED tensor. Only the
     ADDRESS a value is loaded from changes -- each channel's spatial sum still accumulates in
     ascending hw order in one thread's register, and the group combine still reads shared memory in
     ascending channel index -- so the fp32 summation order is unchanged and equality is the right
     assertion. `torch.equal`, not allclose.
  3. Determinism: 10 launches, all three outputs equal. This is the gate the GN-stats-in-epilogue
     prototype failed with shared atomics, and the reason this design uses none.
  4. Against an fp64 reference, so a bug that happens to be self-consistent between (1) and (2) still
     gets caught. Bit-exactness against a sibling kernel proves agreement, not correctness.

WHY THERE IS NO PERFORMANCE ASSERTION HERE. This file is the correctness gate; the speed question is
integration/tests/bench_cat2_fold.py's. A gate that also checks speed either becomes flaky on a busy
GPU or gets its threshold loosened until it cannot fail -- which happened twice in this session's
measurement scripts (a divide-by-zero that printed "CONSISTENT", and a fraction compared against 1.0
instead of 0.01).

SHAPES ARE THE MODEL'S OWN, probed from a live sampling run rather than invented: 5 of the 9 are
group-aligned (C1 % CPG == 0) and 4 have one GroupNorm group straddling the two buffers, which is
exactly the case an index bug would survive on aligned shapes and fail on. Both are covered.

Run: python integration/tests/test_cat2_gn_fold.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402

GROUPS = 32
#: (C1, C2, H, W) -- every decoder skip-concat this UNet runs, from bench_cat2_fold.py's probe.
#: The last four have a group straddling the C1 boundary.
SHAPES = [(768, 768, 2, 2), (768, 768, 4, 4), (384, 384, 8, 8), (384, 384, 16, 16),
          (192, 192, 32, 32), (768, 384, 4, 4), (768, 384, 8, 8),
          (384, 192, 16, 16), (384, 192, 32, 32)]
EPS = 1e-5
BATCH = 4


def ref_stats_fp64(cat, G, eps):
    """GroupNorm statistics in fp64 from the concatenated tensor -- an INDEPENDENT reference.

    Bit-exactness against the same kernel fed a concatenated tensor proves the split indexing agrees
    with itself; it cannot catch a bug that is symmetric between the two. This can.
    """
    N, C = cat.shape[0], cat.shape[1]
    v = cat.double().reshape(N, G, C // G, -1)
    mean = v.mean(dim=(2, 3))
    var = v.var(dim=(2, 3), unbiased=False)
    return mean.reshape(-1), (var.reshape(-1) + eps).rsqrt()


def main():
    if not hasattr(mc, "cat2_gn_stats_fp16"):
        print("FAIL: modiff_cutlass has no cat2_gn_stats_fp16 -- the extension needs rebuilding")
        return 1
    torch.manual_seed(20260813)
    bad = 0
    print(f"{'C1':>5}{'C2':>5}{'HxW':>9}{'straddle':>10}"
          f"{'cat exact':>11}{'stats exact':>13}{'det':>6}{'vs fp64':>11}")
    for C1, C2, H, W in SHAPES:
        C = C1 + C2
        cpg = C // GROUPS
        straddle = (C1 % cpg) != 0
        a = torch.randn(BATCH, C1, H, W, device="cuda", dtype=torch.float16
                        ).to(memory_format=torch.channels_last)
        b = torch.randn(BATCH, C2, H, W, device="cuda", dtype=torch.float16
                        ).to(memory_format=torch.channels_last)

        cat_f, mean_f, inv_f = mc.cat2_gn_stats_fp16(a, b, GROUPS, EPS)
        cat_ref = mc.cat2_channels_last_fp16(a, b)
        #: THE CONTIGUOUS PATH, through gn_stats_fp16 -- the same kernel with X2 == nullptr, reading a
        #: materialized concatenation. That is the comparison that means something.
        #:
        #: The first draft of this gate instead re-split cat_ref at C1 and called cat2_gn_stats_fp16 on
        #: the halves. But cat-then-split is the identity, so that is the SAME function with the SAME
        #: inputs: it asserts determinism and reports it as split-vs-contiguous equivalence. Caught
        #: before it ran, and it is why gn_stats_fp16 was added to pybind in the same change.
        mean_r, inv_r = mc.gn_stats_fp16(cat_ref, GROUPS, EPS)

        cat_ok = torch.equal(cat_f, cat_ref)
        stats_ok = torch.equal(mean_f, mean_r) and torch.equal(inv_f, inv_r)

        det = True
        for _ in range(9):
            c2_, m2_, i2_ = mc.cat2_gn_stats_fp16(a, b, GROUPS, EPS)
            if not (torch.equal(c2_, cat_f) and torch.equal(m2_, mean_f)
                    and torch.equal(i2_, inv_f)):
                det = False
                break

        m64, i64 = ref_stats_fp64(cat_ref, GROUPS, EPS)
        err = max(float((mean_f.double() - m64).abs().max()),
                  float((inv_f.double() - i64).abs().max() / i64.abs().max()))
        ref_ok = err < 2e-3

        ok = cat_ok and stats_ok and det and ref_ok
        bad += 0 if ok else 1
        print(f"{C1:>5}{C2:>5}{f'{H}x{W}':>9}{'YES' if straddle else 'no':>10}"
              f"{'ok' if cat_ok else 'FAIL':>11}{'ok' if stats_ok else 'FAIL':>13}"
              f"{'ok' if det else 'FAIL':>6}{err:>11.2e}{'' if ref_ok else ' FAIL'}")
        del a, b, cat_f, mean_f, inv_f, cat_ref

    print()
    if bad:
        print(f"{bad}/{len(SHAPES)} shapes FAILED -- do not wire this into the model path")
        return 1
    print(f"ALL {len(SHAPES)} SHAPES PASS: concat bit-identical to cat2, stats bit-identical to the "
          f"contiguous path, deterministic over 10 launches, and within 2e-3 of an fp64 reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
