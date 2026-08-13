"""Why does the GroupNorm stats pass run at 15-19% of the A40's bandwidth?

docs/gn_stats_in_epilogue_2026-08-11/FINDINGS.md established that fusing the stats into the conv
epilogue is worth ~0.9% end to end, while getting this kernel nearer its own roofline is worth ~3.8%.
That made the deficit the thing to measure, and this is the driver for measuring it.

TWO JOBS.

  1. Time the SHIPPED kernel directly. Every previous number for
     `gn_stats_partials_chanmajor_kernel` was inherited from the 2026-08-11 report, because the kernel
     has no pybind entry of its own -- which was the stated caveat on the whole roofline argument. It
     is reachable through `group_norm_silu_nhwc`, which launches stats + combine + apply; under
     Nsight Compute with a kernel-name filter the stats launch is isolated and timed on its own, so
     the inherited column can be replaced with a measurement.

  2. Give ncu something to profile at the four real shapes, so the deficit can be attributed:
     achieved DRAM throughput, achieved vs theoretical occupancy, and the launch config. `C/K` threads
     per block is only 6 warps at C=192, which is the leading hypothesis and one ncu can confirm or
     kill outright.

Shapes and counts are the model's own, from the same table the prototype harness uses, so the
weighting matches every other number in that report.

Run bare (wall-clock, includes combine+apply, so an upper bound):
    python integration/tests/bench_gn_stats_roofline.py

Run under Nsight Compute (isolates the stats kernel; this is the point):
    ncu --kernel-name regex:gn_stats_partials --launch-skip 4 --launch-count 4 \
        --section SpeedOfLight --section Occupancy --section LaunchStats \
        python integration/tests/bench_gn_stats_roofline.py --once
"""
import argparse
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402

DEV = "cuda"
GROUPS = 32
#: (C, H, W, count) -- the GroupNorm shapes this UNet actually runs, and how many layers hit each.
#: Same table as bench_gn_stats_tiles.py so the weighting is comparable.
SHAPES = [(192, 32, 32, 14), (384, 32, 32, 4), (768, 16, 16, 4), (768, 4, 4, 10)]
PEAK_GBS = 696.0        # A40 spec DRAM bandwidth


STATS_TOT = [0.0]
FULL_TOT = [0.0]


def profile_kernels(fn, iters=20):
    """Per-CUDA-kernel self time in us per call, via CUPTI activity tracing."""
    from torch.profiler import profile, ProfilerActivity
    from torch.profiler import DeviceType
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = {}
    for e in prof.key_averages():
        if e.device_type != DeviceType.CUDA:
            continue
        us = float(getattr(e, "self_device_time_total", 0) or 0)
        if us > 0:
            out[e.key] = us / iters
    return out


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--once", action="store_true",
                    help="one call per shape, no timing loop -- for running under ncu")
    ap.add_argument("--profile", action="store_true",
                    help="isolate the stats kernel from apply with torch.profiler (CUPTI activity "
                         "tracing, which works where ncu's perf counters are blocked by "
                         "ERR_NVGPUCTRPERM)")
    a = ap.parse_args()
    B = a.batch
    print(f"{torch.cuda.get_device_name(0)}, batch {B}, G={GROUPS}\n")
    if not a.once:
        print("| C | HxW | n | full GN us | X MiB | GB/s (>=1 read+1 write) | % of peak |")
        print("|---|---|--:|--:|--:|--:|--:|")

    if a.profile:
        print("| C | HxW | n | stats us | combine us | apply us | other us | stats GB/s | % peak |")
        print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    tot = 0.0
    for C, H, W, n in SHAPES:
        x = torch.randn(B, C, H, W, device=DEV, dtype=torch.float16).to(
            memory_format=torch.channels_last)
        # weight/bias dtype must MATCH the input dtype -- the kernel rejects the fp32 affine params
        # that torch's own GroupNorm would hand it.
        wt = torch.ones(C, device=DEV, dtype=torch.float16)
        bs = torch.zeros(C, device=DEV, dtype=torch.float16)
        empty = torch.empty(0, device=DEV, dtype=torch.float16)
        empty32 = torch.empty(0, device=DEV, dtype=torch.float32)
        # THE TWO-PASS ENTRY POINT, not group_norm_silu_nhwc. The first version of this script drove
        # group_norm_silu_nhwc, which is a SINGLE fused kernel and never launches
        # gn_stats_partials_chanmajor at all -- so it measured 0.0 us of "stats" and the bucket that
        # was supposed to catch that instead swallowed the whole fused kernel. The two-kernel split
        # lives in the MoDiff delta path: stats -> reduce_partials -> gn_apply_delta_quantize_pack,
        # which the real run shows at 11000/12400/12400 calls.
        a_hat = torch.zeros_like(x)
        scale = torch.tensor([8.0], device=DEV, dtype=torch.float32)
        call = lambda: mc.group_norm_silu_delta_quantize_pack_nhwc(
            x, wt, bs, a_hat, GROUPS, 1e-5, True, scale, empty32, empty, empty,
            empty32, empty32, empty32, empty32, 7.0, False, 1.0)
        if a.once:
            call()
            torch.cuda.synchronize()
            print(f"  ran C={C} {H}x{W}")
            continue
        if a.profile:
            per = profile_kernels(call)
            # Exact families. "group_norm_silu" as a substring matched the single fused kernel too,
            # which is how the first version reported stats=0 and apply=everything.
            stats_us = sum(v for k, v in per.items() if "gn_stats_partials" in k)
            comb_us = sum(v for k, v in per.items() if "gn_stats_reduce_partials" in k)
            apply_us = sum(v for k, v in per.items() if "gn_apply_delta_quantize" in k)
            other_us = sum(per.values()) - stats_us - comb_us - apply_us
            byts = B * C * H * W * 2
            #: The stats pass reads X exactly once and writes a tiny [N,G,nblocks] partials buffer,
            #: so 1 read is the whole traffic and % of peak here is honest rather than an upper bound.
            gbs = byts / (stats_us * 1e-6) / 1e9 if stats_us else 0.0
            print(f"| {C} | {H}x{W} | {n} | {stats_us:.1f} | {comb_us:.1f} | {apply_us:.1f} | "
                  f"{other_us:.1f} | {gbs:.0f} | {gbs/PEAK_GBS*100:.0f}% |")
            STATS_TOT[0] += n * stats_us
            FULL_TOT[0] += n * sum(per.values())
            continue
        us = time_us(call)
        tot += n * us
        byts = B * C * H * W * 2
        #: The FULL op reads X once for stats, reads it again in apply, and writes Y -- so >=3 passes.
        #: Quoted as (1 read + 1 write) so the number is a floor on what the hardware is asked to move
        #: and the % of peak is an UPPER bound on efficiency, never a flattering one.
        gbs = 2 * byts / (us * 1e-6) / 1e9
        print(f"| {C} | {H}x{W} | {n} | {us:.1f} | {byts/2**20:.0f} | {gbs:.0f} | "
              f"{gbs/PEAK_GBS*100:.0f}% |")

    if a.profile:
        print(f"\ncount-weighted STATS pass only : {STATS_TOT[0]/1e3:.2f} ms")
        print(f"count-weighted FULL GN op      : {FULL_TOT[0]/1e3:.2f} ms")
        print(f"stats is {STATS_TOT[0]/FULL_TOT[0]*100:.0f}% of the full op")
        #: THE NUMBER THIS EXISTS TO CHECK. bench_gn_stats_tiles.py compares its prototype against a
        #: "shipped us" column of 11.94 ms weighted, inherited from the 2026-08-11 report because the
        #: stats kernel has no pybind entry. If the stats pass is actually a few ms, that column is not
        #: the stats pass, the prototype's speed gate compares against the wrong thing, and the
        #: roofline argument built on it in docs/gn_stats_in_epilogue_2026-08-11/FINDINGS.md is wrong.
        print(f"\ninherited 'shipped' column used by bench_gn_stats_tiles.py: 11.94 ms")
        if STATS_TOT[0] <= 0:
            # The first version divided by zero here and printed "CONSISTENT" -- an instrument
            # reporting agreement when it has NO DATA. Refuse instead.
            print("  NO STATS KERNEL WAS OBSERVED: this driver is not exercising the two-pass path, "
                  "so nothing here can be compared. Fix the driver, do not read the table above.")
        else:
            r = 11.94e3 / STATS_TOT[0]
            print(f"  that is {r:.1f}x the stats pass measured here -- "
                  f"{'CONSISTENT' if r < 1.3 else 'NOT the stats pass, the inherited column is wrong'}")
        return 0
    if not a.once:
        print(f"\ncount-weighted FULL group_norm_silu_nhwc: {tot/1e3:.2f} ms")
        print("This is stats + combine + apply together, i.e. an UPPER bound on the stats pass.\n"
              "Run under ncu (see the docstring) to isolate gn_stats_partials_chanmajor and get\n"
              "occupancy + achieved bandwidth per launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
