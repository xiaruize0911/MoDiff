"""Does the implemented fold actually beat cat2 + the stats pass?

The fold is built and gated (test_cat2_gn_fold.py: concat bit-identical to cat2_channels_last_fp16,
stats bit-identical to the contiguous path through gn_stats_fp16, deterministic, within 2e-7 of an fp64
reference, on all 9 real shapes including the 4 where a GroupNorm group straddles the two buffers). This
is the speed question, kept in a separate file from the gate on purpose -- a gate that also checks
speed gets its threshold loosened until it cannot fail, which happened twice in this session's
measurement scripts.

    baseline   cat2_channels_last_fp16(a, b)  then  gn_stats_fp16(cat)      2 kernels, 3C of traffic
    fold       cat2_gn_stats_fp16(a, b)                                     1 kernel,  2C of traffic

BOTH HALVES ARE NOW DIRECTLY TIMEABLE, which they were not this morning. gn_stats_fp16 was added in the
same change as the fold precisely because its absence is what made an inherited (and 3.8x wrong) number
the basis of a claim that had to be retracted. This comparison needs no inherited column.

WHAT THE END-TO-END PROJECTION ASSUMES, stated because it is the soft part. The profiled W4A4 window has
cat2 at 392 ms of 12106. The fold removes one pass over the concatenated tensor, so the saving is
expressed as a FRACTION OF CAT2 measured here and then applied to that 392 ms. Using a ratio rather
than absolute microbenchmark time means a systematic offset between CUDA-event timing here and profiler
self-time there cannot inflate the answer. It does assume the decoder's stats passes cost in the real
run what they cost here at the same shapes, which is the same assumption every other projection in this
report makes.

Run: python integration/tests/bench_cat2_gn_fold.py    # ~2 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402

OUT = "docs/cat2_fold_2026-08-13/data/cat2_gn_fold_measured.json"
GROUPS = 32
EPS = 1e-5
BATCH = 128
PEAK_GBS = 696.0
#: (C1, C2, H, W, count) -- every decoder skip-concat, counts from one sampling pass
#: (bench_cat2_fold.py's probe). Counts weight the shapes against each other, nothing more.
SHAPES = [(768, 768, 2, 2, 21), (768, 768, 4, 4, 14), (384, 384, 8, 8, 14),
          (384, 384, 16, 16, 14), (192, 192, 32, 32, 14), (768, 384, 4, 4, 7),
          (768, 384, 8, 8, 7), (384, 192, 16, 16, 7), (384, 192, 32, 32, 7)]
E2E_CAT2_MS, E2E_WINDOW_MS = 392.0, 12106.0


def time_us(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def main():
    print(f"{torch.cuda.get_device_name(0)}, batch {BATCH}, G={GROUPS}\n")
    print("| C1 | C2 | HxW | cat2 | stats | baseline | FOLD | saved | fold GB/s | % peak |")
    print("|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|")
    rows = []
    cat_tot = base_tot = fold_tot = 0.0
    for C1, C2, H, W, n in SHAPES:
        C = C1 + C2
        a = torch.randn(BATCH, C1, H, W, device="cuda", dtype=torch.float16
                        ).to(memory_format=torch.channels_last)
        b = torch.randn(BATCH, C2, H, W, device="cuda", dtype=torch.float16
                        ).to(memory_format=torch.channels_last)
        cat = mc.cat2_channels_last_fp16(a, b)

        t_cat = time_us(lambda: mc.cat2_channels_last_fp16(a, b))
        t_stats = time_us(lambda: mc.gn_stats_fp16(cat, GROUPS, EPS))

        def baseline():
            c = mc.cat2_channels_last_fp16(a, b)
            mc.gn_stats_fp16(c, GROUPS, EPS)
        t_base = time_us(baseline)
        t_fold = time_us(lambda: mc.cat2_gn_stats_fp16(a, b, GROUPS, EPS))

        #: The fold moves 2C: one read of the halves, one write of the concatenation.
        byts = 2 * BATCH * C * H * W * 2
        gbs = byts / (t_fold * 1e-6) / 1e9
        saved = t_base - t_fold
        cat_tot += n * t_cat
        base_tot += n * t_base
        fold_tot += n * t_fold
        rows.append(dict(C1=C1, C2=C2, C=C, H=H, W=W, count=n, cat2_us=t_cat, stats_us=t_stats,
                         baseline_us=t_base, fold_us=t_fold, saved_us=saved, fold_gbs=gbs))
        print(f"| {C1} | {C2} | {H}x{W} | {t_cat:.1f} | {t_stats:.1f} | {t_base:.1f} | "
              f"{t_fold:.1f} | {'+' if saved > 0 else ''}{saved:.1f} | {gbs:.0f} | "
              f"{gbs/PEAK_GBS*100:.0f}% |")
        del a, b, cat

    saved_tot = base_tot - fold_tot
    frac = saved_tot / cat_tot if cat_tot else 0.0
    print(f"\nweighted cat2 alone        {cat_tot/1e3:7.2f} ms")
    print(f"weighted baseline          {base_tot/1e3:7.2f} ms   (cat2 + stats)")
    print(f"weighted FOLD              {fold_tot/1e3:7.2f} ms")
    print(f"weighted saved             {saved_tot/1e3:7.2f} ms   "
          f"({saved_tot/base_tot*100:.0f}% of the pair, {frac*100:.0f}% of cat2)")
    e2e = E2E_CAT2_MS * frac / E2E_WINDOW_MS * 100
    print(f"\nprojected end to end       {e2e:.2f}%   "
          f"(W4A4 1.749x -> ~{1.749 / (1 - e2e / 100):.3f}x)")
    print()
    if saved_tot <= 0:
        print("THE FOLD IS SLOWER than cat2 + stats. It is correct (the gate passes) and it is not "
              "worth wiring -- stop here and record the number.")
    elif e2e < 0.5:
        print(f"REAL BUT UNDER 0.5% END TO END. Wiring it means deferring the concat through the "
              f"ResBlock boundary, which touches the model's hottest prologue. That is a poor trade "
              f"for {e2e:.2f}% and should not be done on this evidence.")
    else:
        print(f"WORTH WIRING at {e2e:.2f}%. Next step is deferring the concat into the fused ResBlock "
              f"so the halves reach this kernel, keeping cat2 as the fallback for every shape this "
              f"kernel rejects (C1 % 32 != 0, non-channels_last, fp32).")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(gpu=torch.cuda.get_device_name(0), batch=BATCH, groups=GROUPS, rows=rows,
                   weighted_ms=dict(cat2=cat_tot / 1e3, baseline=base_tot / 1e3,
                                    fold=fold_tot / 1e3, saved=saved_tot / 1e3),
                   saved_fraction_of_pair=saved_tot / base_tot if base_tot else 0.0,
                   saved_fraction_of_cat2=frac, projected_e2e_pct=e2e),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
