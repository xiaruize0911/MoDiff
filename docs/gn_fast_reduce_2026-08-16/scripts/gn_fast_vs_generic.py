"""Is `group_norm_silu_quantize_nhwc_fast` a free win on the ResBlock's GN shapes?

WHY THIS EXISTS. docs/OPEN_ITEMS.md C1 said the GroupNorm+SiLU family is 32.2% of the W4A4 run at
1.13x with "no design landed". Both halves were wrong. The family is NOT at the memory roofline -- it
runs at 10-65% of the A40's 696 GB/s -- and a faster design is already IN THE TREE. `..._fast` differs
from the plain entry point in exactly one way: it passes fast_reduce=true, which swaps the block-size
heuristic (128-512 threads, pair-major pass 1) for the generic one (up to 1024). The kernel's own
comment records the generic path as "1.27-4.3x slower after warp reductions".

The attention paths call it (via getattr(_mc, "..._fast", <plain>)). fused_resblock.py does not -- it
names the plain entry point directly at lines 481 and 499 -- and the ResBlock is where the bulk of the
GN time is. So the question this file answers is not "is the fast kernel faster" (known) but "is it
faster ON THE RESBLOCK'S SHAPES, and by how much end to end".

WHAT IS MEASURED. Every GN signature the 2026-08-13 capture recorded in the norm_quantize suite, at its
real shape and call count, both entry points, plus a numeric comparison. Speedup is per kernel; the
weighted total uses the captured calls/step, so it is directly comparable to a ms/step figure.

NUMERICS ARE NOT ASSUMED. fast_reduce changes the fp32 reduction ORDER, so the mean/inv_std can differ
in the last bits and a value sitting exactly on a code boundary can land either side. That is a
legitimate difference, not a bug, but its SIZE has to be measured before the swap can be called free --
so this reports max |code difference| and the fraction of elements that move, per shape.

Run: python docs/gn_fast_reduce_2026-08-16/scripts/gn_fast_vs_generic.py    # wants an idle GPU, ~2 min
Writes docs/gn_fast_reduce_2026-08-16/data/gn_fast.json
"""
import collections
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
#: the built extension is modiff_cutlass*.so in ROOT. chdir does not put it on sys.path, and sys.path[0]
#: here is this script's own directory, so the import needs ROOT added explicitly.
sys.path.insert(0, ROOT)
import modiff_cutlass as mc                                                       # noqa: E402

D = "docs/gn_fast_reduce_2026-08-16"
SRC = "docs/bench_report_2026-08-13_postzp/data/kernel_suites.json"
CAPTURE_STEPS = 5          # the source capture's window; calls_per_sample counts calls over it

#: (plain entry, fast entry) pairs. The int4 sibling packs 2 codes per byte; same dispatch, same args.
PAIRS = [("group_norm_silu_quantize_nhwc", "group_norm_silu_quantize_nhwc_fast", "int8_baseline"),
         ("group_norm_silu_quantize_pack_nhwc", "group_norm_silu_quantize_pack_nhwc_fast",
          "int4_baseline")]

WARM, ITERS, ROUNDS = 10, 60, 8

#: production passes scale-shift modulation at both fused_resblock call sites; run that branch.
MOD = os.environ.get("GN_FAST_MOD", "1") == "1"


def timed(fn, warm=WARM, iters=ITERS, rounds=ROUNDS):
    """Median of per-round medians, the same estimator kernel_suites_bench uses."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(rounds):
        ts = []
        for _ in range(iters):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000.0)                                 # us
        ts.sort()
        meds.append(ts[len(ts) // 2])
    meds.sort()
    return meds[len(meds) // 2]


def build_args(shape, num_groups=32, mod=True):
    """The captured call's arguments, rebuilt.

    Real values, not zeros: the quantize rounds, so a constant input would put every element on the same
    code and hide any reduction-order difference.

    `mod=True` is the PRODUCTION configuration and the one that matters. fused_resblock.py passes
    `ms2d`/`sh2d` -- the ResBlock's [N, C] scale-shift modulation -- at both call sites, and a
    measurement with empty mod tensors exercises a different branch of the kernel's dispatch (has_mod
    picks up two more strided reads per element and a different inner loop). mod=False is kept only to
    show the two agree.
    """
    n, c, h, w = shape
    g = torch.Generator(device="cuda").manual_seed(1234)
    #: NHWC, as the kernel requires and as production passes it -- the whole point of the _nhwc family.
    #: A plain contiguous tensor is rejected outright ("x must be channels_last contiguous").
    x = torch.randn(shape, generator=g, device="cuda",
                    dtype=torch.float16).to(memory_format=torch.channels_last)
    weight = torch.randn(c, generator=g, device="cuda", dtype=torch.float16)
    bias = torch.randn(c, generator=g, device="cuda", dtype=torch.float16)
    scale = torch.full((1,), 8.0, device="cuda", dtype=torch.float32)              # 127/absmax-ish
    empty_f = torch.empty(0, device="cuda", dtype=torch.float32)
    empty_h = torch.empty(0, device="cuda", dtype=torch.float16)
    if not mod:
        return (x, weight, bias, num_groups, 1e-5, True, scale, empty_f, empty_h, empty_h)
    #: [N, C] and fp16, matching x's dtype -- the impl TORCH_CHECKs both.
    ms = (torch.randn((n, c), generator=g, device="cuda", dtype=torch.float16) * 0.1 + 1.0)
    sh = torch.randn((n, c), generator=g, device="cuda", dtype=torch.float16) * 0.1
    return (x, weight, bias, num_groups, 1e-5, True, scale, empty_f, ms, sh)


def main():
    ks = json.load(open(SRC))
    out = {"gpu": torch.cuda.get_device_name(0), "capture_steps": CAPTURE_STEPS, "mod": MOD,
           "warm": WARM, "iters_per_round": ITERS, "rounds": ROUNDS, "pairs": []}

    for plain, fast, arm in PAIRS:
        if not (hasattr(mc, plain) and hasattr(mc, fast)):
            print(f"skip {plain}: not exported")
            continue
        recs = [r for r in ks["modes"][arm]["norm_quantize"] if r["entry"] == plain]
        #: one entry per distinct shape; calls summed, because the same shape at two call sites is the
        #: same kernel launch and the swap would apply to both.
        byshape = collections.OrderedDict()
        for r in recs:
            k = tuple(r["arg_shapes"][0])
            e = byshape.setdefault(k, {"calls": 0, "captured_us": []})
            e["calls"] += r["calls_per_sample"]
            e["captured_us"].append(r["stats"]["median"])

        res = {"plain": plain, "fast": fast, "arm": arm, "shapes": []}
        print(f"\n=== {plain}  vs  {fast}   ({len(byshape)} shapes)")
        print(f"{'shape':22s}{'calls/step':>11}{'plain us':>10}{'fast us':>9}{'speedup':>9}"
              f"{'max|dcode|':>11}{'% moved':>9}")
        for shape, meta in byshape.items():
            args = build_args(list(shape), mod=MOD)
            fp, ff = getattr(mc, plain), getattr(mc, fast)
            a = fp(*args)
            b = ff(*args)
            #: int8 codes compared as int16 so the difference cannot wrap
            diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
            mx = int(diff.max().item())
            moved = float((diff != 0).float().mean().item()) * 100.0
            t_plain = timed(lambda: fp(*args))
            t_fast = timed(lambda: ff(*args))
            calls_per_step = meta["calls"] / CAPTURE_STEPS
            row = {"shape": list(shape), "calls_per_step": calls_per_step,
                   "plain_us": t_plain, "fast_us": t_fast, "speedup": t_plain / t_fast,
                   "max_code_diff": mx, "pct_codes_moved": moved,
                   "captured_us": meta["captured_us"]}
            res["shapes"].append(row)
            print(f"{str(list(shape)):22s}{calls_per_step:11.1f}{t_plain:10.1f}{t_fast:9.1f}"
                  f"{t_plain / t_fast:8.2f}x{mx:11d}{moved:8.2f}%")
            del args, a, b, diff
            torch.cuda.empty_cache()

        ms_plain = sum(r["plain_us"] * r["calls_per_step"] for r in res["shapes"]) / 1000.0
        ms_fast = sum(r["fast_us"] * r["calls_per_step"] for r in res["shapes"]) / 1000.0
        res["ms_per_step_plain"] = ms_plain
        res["ms_per_step_fast"] = ms_fast
        res["ms_per_step_saved"] = ms_plain - ms_fast
        res["weighted_speedup"] = ms_plain / ms_fast if ms_fast else 0.0
        print(f"  weighted over captured calls/step: {ms_plain:.3f} -> {ms_fast:.3f} ms/step "
              f"= {ms_plain / ms_fast:.2f}x, SAVES {ms_plain - ms_fast:.3f} ms/step")
        out["pairs"].append(res)

    os.makedirs(f"{D}/data", exist_ok=True)
    tag = "mod" if MOD else "nomod"
    json.dump(out, open(f"{D}/data/gn_fast_{tag}.json", "w"), indent=1)
    print(f"\nwrote {D}/data/gn_fast_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
