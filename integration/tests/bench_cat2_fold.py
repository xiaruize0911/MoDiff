"""Can the decoder skip-concat be folded into the conv prologue, and is it worth it?

`cat2_channels_last_fp16_kernel` is 392 ms of the 12106 ms W4A4 window (3.2%) over 3000 launches. It is
pure data movement -- reads a and b, writes their channel concatenation -- so unlike the GroupNorm
stats pass (which turned out to run at 57-70% of peak and therefore had no headroom) a fold here would
remove the traffic ENTIRELY rather than merely do it faster. That makes the ceiling the whole 3.2%.

THE OBSTACLE, read out of the code before measuring anything: the concatenated tensor has TWO consumers
inside the fused ResBlock, not one --

    fused_resblock.py:844   _prequant_gn_conv(x, fused_in_norm_silu, in_conv)   the GN+SiLU+quantize
                                                                               prologue
    fused_resblock.py:863   self.skip_connection(x)                             the 1x1 skip conv
                                                                               (or Identity, whose
                                                                               output is x itself and
                                                                               is still read as the
                                                                               out-conv's residual)

So folding into the prologue alone does NOT remove the concat: the skip path would still need the
materialized tensor. Both consumers have to accept a split input. That is exact for both --
GroupNorm over a channel range does not care which buffer a channel lives in, and a 1x1 conv splits as
W*[a;b] = W1*a + W2*b -- but the second one means either two conv launches plus an accumulate, or a new
two-input conv kernel. THAT is the risk that decides this, so it is what gets priced.

FOUR MEASUREMENTS, in the order that can kill the idea:

  1. The real shapes. Monkeypatch `_skip_concat` and record every (C1, C2, H, W) the decoder actually
     runs, with counts -- rather than inferring them from the config.
  2. Group alignment. GroupNorm at G=32 has CPG = C/32 channels per group. If C1 % CPG != 0 a group
     STRADDLES the two buffers. Still implementable (in the chan-major layout a thread owns one channel
     for the whole kernel, so "which pointer" is decided once per thread and is nearly free), but it
     decides whether the fold is a clean case or a fiddly one.
  3. cat2's own cost and achieved bandwidth at those shapes -- to confirm the 392 ms is real and to see
     whether it is already at peak, i.e. whether the traffic really is all there is to save.
  4. THE RISK: one 1x1 conv over C channels vs two 1x1 convs over C1 and C2 plus an add. If the split
     costs more than cat2 saves, the whole idea is dead and no kernel work should start.

Run: python integration/tests/bench_cat2_fold.py    # ~3 min, needs the GPU
"""
import json
import os
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import modiff_cutlass as mc                                                 # noqa: E402

OUT = "docs/cat2_fold_2026-08-13/data/cat2_fold.json"
GROUPS = 32
PEAK_GBS = 696.0
#: profiled in the shipped W4A4 run (docs/state_report_2026-08-12/data/e2e.json)
E2E_CAT2_MS, E2E_WINDOW_MS, E2E_CALLS = 392.0, 12106.0, 3000
#: The skip conv's OUT channels per concat width, read off the real model rather than assumed
#: (output_blocks.*.skip_connection.weight). Several widths appear with two different Cout, so both
#: are measured and the concat count is split evenly between them. The first version of this file
#: assumed Cout == C2, which is right for the symmetric shapes and WRONG for the asymmetric ones
#: (C=1152 also maps to 768, C=576 also maps to 384) -- and a larger Cout makes the split MORE
#: expensive, so that assumption was optimistic.
REAL_COUT = {1536: [768], 1152: [768, 384], 768: [384], 576: [384, 192], 384: [192]}


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


def collect_shapes(batch):
    """Record every (C1, C2, H, W) the decoder concatenates, by wrapping _skip_concat itself.

    Read off the running model rather than derived from the config: the point of this whole
    investigation is that a plausible derivation is not a measurement.
    """
    import dynamic_delta_ab as H
    import ldm.modules.diffusionmodules.openaimodel as om
    import integration.benchmarks.benchmark_ldm as B
    H.STEPS, H.BATCH = 2, batch
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"
    seen = Counter()
    orig = om._skip_concat

    def spy(h, skip):
        seen[(int(h.shape[1]), int(skip.shape[1]), int(h.shape[2]), int(h.shape[3]))] += 1
        return orig(h, skip)

    om._skip_concat = spy
    try:
        r, m, s = H.build("int4", B._default_calibration_path("int4"), "static")
        H.SEED = 1234
        H.latent(r, m, s)
        skips = {}
        for name, mod in m.model.diffusion_model.output_blocks.named_modules():
            if hasattr(mod, "skip_connection"):
                skips[name] = type(mod.skip_connection).__name__
        del r, m, s
        torch.cuda.empty_cache()
    finally:
        om._skip_concat = orig
    return seen, skips


def main():
    batch = 128
    print(f"{torch.cuda.get_device_name(0)}, batch {batch}, G={GROUPS}\n")
    print("collecting the real decoder concat shapes ...", flush=True)
    seen, skips = collect_shapes(8)          # shapes are batch-independent; build small
    print(f"\nskip_connection types across output_blocks: {dict(Counter(skips.values()))}\n")

    print("| C1 | C2 | C | HxW | calls | CPG | C1 % CPG | group straddles? |")
    print("|--:|--:|--:|---|--:|--:|--:|---|")
    rows = []
    for (c1, c2, h, w), n in sorted(seen.items(), key=lambda kv: -kv[1]):
        C = c1 + c2
        cpg = C // GROUPS
        straddle = (c1 % cpg) != 0
        rows.append(dict(C1=c1, C2=c2, C=C, H=h, W=w, count=n, cpg=cpg, straddles=straddle))
        print(f"| {c1} | {c2} | {C} | {h}x{w} | {n} | {cpg} | {c1 % cpg} | "
              f"{'YES' if straddle else 'no'} |")

    print("\n--- cat2 cost at those shapes (batch 128) ---")
    print("| C1 | C2 | HxW | cat2 us | traffic MiB | GB/s | % peak |")
    print("|--:|--:|---|--:|--:|--:|--:|")
    cat_tot = 0.0
    for r in rows:
        a = torch.randn(batch, r["C1"], r["H"], r["W"], device="cuda", dtype=torch.float16
                        ).to(memory_format=torch.channels_last)
        b = torch.randn(batch, r["C2"], r["H"], r["W"], device="cuda", dtype=torch.float16
                        ).to(memory_format=torch.channels_last)
        us = time_us(lambda: mc.cat2_channels_last_fp16(a, b))
        #: reads a+b once, writes the concatenation once -> 2x the output bytes
        byts = 2 * batch * r["C"] * r["H"] * r["W"] * 2
        gbs = byts / (us * 1e-6) / 1e9
        r["cat2_us"] = us
        r["cat2_gbs"] = gbs
        cat_tot += r["count"] * us
        print(f"| {r['C1']} | {r['C2']} | {r['H']}x{r['W']} | {us:.1f} | {byts/2**20:.0f} | "
              f"{gbs:.0f} | {gbs/PEAK_GBS*100:.0f}% |")
        del a, b

    print("\n--- THE RISK: does splitting the 1x1 skip conv cost more than cat2 saves? ---")
    #: TWO CORRECTIONS over the first version of this measurement, both of which changed the verdict:
    #:
    #:  ACCUMULATE, not add. The first version timed the split as `f(a,wa) + f(b,wb)`, which
    #:  materializes TWO full outputs and then reads both back to add them. A real implementation has
    #:  the second half accumulate in place (beta=1) -- which this codebase already does for o_hat --
    #:  so the fair comparison is one output buffer, written once and accumulated into. A 1x1 conv IS
    #:  a GEMM, so both are timed as GEMMs where beta=1 is expressible (addmm_). The naive version put
    #:  the net saving at 12% of cat2; the accumulate version puts it at ~73%, a 6x difference. The
    #:  naive number would have killed the idea on a measurement artifact.
    #:
    #:  REAL Cout. The first version assumed the skip conv maps C -> C2. Probed off the model, that is
    #:  right for the symmetric shapes and WRONG for the asymmetric ones: C=1152 maps to 768 (block
    #:  5.0) as well as 384 (6.0), and C=576 maps to 384 (11.0) as well as 192 (12.0). A larger Cout
    #:  makes the split MORE expensive, so the assumption was optimistic.
    print("| C1 | C2 | Cout | HxW | 1 gemm us | 2 gemm acc us | split cost | cat2 us | net |")
    print("|--:|--:|--:|---|--:|--:|--:|--:|---|")
    net_tot = 0.0
    for r in rows:
        C, H, W = r["C"], r["H"], r["W"]
        for Cout in REAL_COUT[C]:
            M = batch * H * W
            x = torch.randn(M, C, device="cuda", dtype=torch.float16)
            a, b = x[:, :r["C1"]].contiguous(), x[:, r["C1"]:].contiguous()
            w = torch.randn(C, Cout, device="cuda", dtype=torch.float16)
            wa, wb = w[:r["C1"]].contiguous(), w[r["C1"]:].contiguous()
            out = torch.empty(M, Cout, device="cuda", dtype=torch.float16)
            one = time_us(lambda: torch.mm(x, w, out=out))

            def two_acc():
                torch.mm(a, wa, out=out)
                out.addmm_(b, wb)
            two = time_us(two_acc)
            d = two - one
            net = r["cat2_us"] - d
            #: count is split evenly across the Cout variants this C is seen with, since the shape
            #: probe cannot tell which block a given concat call belonged to.
            wgt = r["count"] / len(REAL_COUT[C])
            net_tot += wgt * net
            suspect = one > two          # one GEMM slower than two is not a real property
            r.setdefault("split", []).append(
                dict(Cout=Cout, gemm_one_us=one, gemm_two_acc_us=two, split_cost_us=d,
                     net_us=net, weight=wgt, suspect=suspect))
            print(f"| {r['C1']} | {r['C2']} | {Cout} | {H}x{W} | {one:.1f} | {two:.1f} | {d:+.1f} | "
                  f"{r['cat2_us']:.1f} | {'+' if net > 0 else ''}{net:.1f}"
                  f"{'  <- SUSPECT' if suspect else ''} |")
            del x, a, b, w, wa, wb, out

    #: Scale to the profiled window via a RATIO, so a systematic offset between CUDA-event timing here
    #: and profiler self-time there cannot inflate the answer.
    #: Report the total BOTH ways -- with and without the rows where a single GEMM timed slower than
    #: two, which is a cuBLAS heuristic artifact rather than a property of the split. In the first run
    #: one such row carried 42% of the entire claimed saving, which is far too much weight for a
    #: number that cannot be true.
    sus = sum(v["weight"] * v["net_us"] for r in rows for v in r.get("split", []) if v["suspect"])
    frac = net_tot / cat_tot if cat_tot else 0.0
    frac_clean = (net_tot - sus) / cat_tot if cat_tot else 0.0
    if sus:
        print(f"\nNOTE: suspect rows (one GEMM timed slower than two) carry "
              f"{sus/net_tot*100:.0f}% of the saving; both totals are reported.")
    print(f"\nweighted cat2 (microbench)      {cat_tot/1e3:8.2f} ms")
    print(f"weighted NET saving if folded   {net_tot/1e3:8.2f} ms   ({frac*100:.0f}% of cat2)")
    print(f"\nscaled onto the profiled W4A4 window ({E2E_CAT2_MS:.0f} ms cat2 of {E2E_WINDOW_MS:.0f}):")
    print(f"  end-to-end ceiling  {E2E_CAT2_MS/E2E_WINDOW_MS*100:.2f}%   (cat2 removed entirely)")
    print(f"  realistic           {E2E_CAT2_MS*frac/E2E_WINDOW_MS*100:.2f}%   (after the split's cost)")
    print(f"  realistic, suspect rows dropped  "
          f"{E2E_CAT2_MS*frac_clean/E2E_WINDOW_MS*100:.2f}%")
    print()
    if frac <= 0:
        print("DEAD: splitting the skip conv costs MORE than cat2 saves. No kernel work is justified -- "
              "the concat is cheaper than the two convs that would replace it.")
    #: 0.01, not 1.0. The first version compared this FRACTION against 1.0, so the "below 1% end to
    #: end" branch could never be false -- it printed that verdict for the naive-add run too, where it
    #: happened to agree with the number for the wrong reason. Same class of defect as the
    #: divide-by-zero "CONSISTENT" in bench_gn_stats_roofline: a threshold that cannot fail.
    elif E2E_CAT2_MS * frac_clean / E2E_WINDOW_MS < 0.01:
        print("BELOW 1% END TO END: the fold is real but small, and it needs a two-input GN prologue "
              "AND a split skip conv -- two kernel changes. Record the number; do not build it unless "
              "something else makes those kernels necessary anyway.")
    else:
        print("WORTH SCOPING: the net saving survives the split. Start with the two-input GN prologue, "
              "the half with no arithmetic risk (in the chan-major layout a channel's buffer is "
              "decided once per thread).")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(gpu=torch.cuda.get_device_name(0), batch=batch, groups=GROUPS,
                   skip_types=dict(Counter(skips.values())), rows=rows,
                   weighted_cat2_ms=cat_tot / 1e3, weighted_net_ms=net_tot / 1e3,
                   net_fraction_of_cat2=frac, net_fraction_clean=frac_clean,
                   suspect_share_of_saving=(sus / net_tot if net_tot else 0.0),
                   e2e=dict(cat2_ms=E2E_CAT2_MS, window_ms=E2E_WINDOW_MS,
                            ceiling_pct=E2E_CAT2_MS / E2E_WINDOW_MS * 100,
                            realistic_pct=E2E_CAT2_MS * frac / E2E_WINDOW_MS * 100)),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
