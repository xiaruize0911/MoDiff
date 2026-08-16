"""Does the GN fast-reduce swap change what comes out? relL2 vs a per-seed fp16 reference, both arms.

The swap is NOT bit-identical and was never claimed to be: fast_reduce changes the fp32 reduction
order, so a mean/inv_std can move in the last bits and a value sitting exactly on a quantize code
boundary can land either side. The kernel microbenchmark measured that at <=1 code on <=1.6e-5% of
elements (docs/gn_fast_reduce_2026-08-16). This file asks whether it survives 100 sampler steps, where
each step's output is the next step's input.

MEASURED AGAINST fp16, NOT AGAINST EACH OTHER. An arm-to-arm latent relL2 answers the wrong question.
Both arms are approximations of the fp16 model, and a swap that merely rounds differently -- landing the
same distance from the truth in a different direction -- is free even though the two latents differ. A
first version of this file reported the arm-to-arm number (7.5e-3) and could not distinguish those two
cases. So each arm is scored against the SAME per-seed fp16 latent and those two scores are compared,
which is the instrument quality_route_b_paired.py already established.

RUN 1 IS DISCARDED, and that is not a formality. The first sample in a process is not steady state --
the calibration/scale-freeze window falls inside it -- and it shows: an OFF-vs-OFF control differed by
relL2 0.134 on the first seed and by exactly 0.0 on every seed after it. Without the discard, the
harness reports the warm-up as if it were the effect.

PAIRED per docs/act_bits_2026-08-05: relL2 at batch 8 varies 10-30% run to run, so both arms see the
same seeds and the same per-seed fp16 reference, and the statistic is the per-seed difference.

Run: python integration/tests/quality_gn_fast_paired.py [--seeds 4] [--bits 8]
Writes docs/gn_fast_reduce_2026-08-16/data/quality_gn_fast.json
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))
sys.path.insert(0, os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"))

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402
import kernel_suites_bench as ks                                         # noqa: E402

D = "docs/gn_fast_reduce_2026-08-16"

#: THE PTQ ARMS, not the MoDiff ones, and this is load-bearing. In MoDiff mode the ResBlock takes the
#: GN->delta-quantize fusion (group_norm_silu_delta_quantize_nhwc), which this swap does not touch, so
#: the flag is INERT there -- a first version of this file ran mode "int8", reported BIT-IDENTICAL, and
#: was measuring nothing at all. The counters below exist so that cannot recur silently.
MODES = {8: ("int8_baseline", "group_norm_silu_quantize_nhwc"),
         4: ("int4_baseline", "group_norm_silu_quantize_pack_nhwc")}


def fp16_refs(seeds):
    """Per-seed fp16 latent: the denominator of every relL2 below."""
    r, m, s = H.build("fp16", None, "dynamic")
    H.latent(r, m, s)                                    # warm-up, discarded
    refs = {}
    for seed in seeds:
        H.SEED = seed
        lat, _ = H.latent(r, m, s)
        refs[seed] = lat
    del r, m, s
    torch.cuda.empty_cache()
    return refs


def arm(gn_fast, mode, calib, seeds, refs, plain):
    """relL2 per seed for one configuration, from one freshly built model.

    Returns (rel, calls) where `calls` counts the PLAIN entry point over the timed seeds. A
    quality verdict is only meaningful if the two arms actually ran different code, so the caller
    asserts on it: plain must be 0 with the swap ON and non-zero with it OFF.
    """
    os.environ["MODIFF_GN_FAST"] = "1" if gn_fast else "0"
    r, m, s = H.build(mode, calib, "dynamic")
    H.latent(r, m, s)                                    # run 1 is not steady state -- discard
    hits = [0]
    orig = getattr(mc, plain)

    def counting(*a, **kw):
        hits[0] += 1
        return orig(*a, **kw)
    setattr(mc, plain, counting)
    rel = {}
    for seed in seeds:
        H.SEED = seed
        lat, _ = H.latent(r, m, s)
        ref = refs[seed]
        rel[seed] = float((lat - ref).norm() / ref.norm())
    setattr(mc, plain, orig)
    del r, m, s
    torch.cuda.empty_cache()
    return rel, hits[0]


def report(name, off, on, seeds):
    po = [off[s] for s in seeds]
    pn = [on[s] for s in seeds]
    mo, mn = statistics.mean(po), statistics.mean(pn)
    ident = all(abs(x - y) < 1e-12 for x, y in zip(po, pn))
    per = [(y - x) / x * 100.0 for x, y in zip(po, pn)]
    dm = statistics.mean(per)
    sd = statistics.stdev(per) if len(per) > 1 else 0.0
    sem = sd / len(per) ** 0.5 if len(per) > 1 else 0.0
    wins = sum(1 for x, y in zip(pn, po) if x < y)
    print(f"\n=== {name} : latent relL2 vs fp16, lower is better ===")
    print(f"{'arm':>16} | " + " ".join(f"{s:>9}" for s in seeds) + f" | {'mean':>8}")
    print("-" * (20 + 10 * len(seeds) + 11))
    print(f"{'OFF (generic)':>16} | " + " ".join(f"{v:>9.4f}" for v in po) + f" | {mo:>8.4f}")
    print(f"{'ON (fast_reduce)':>16} | " + " ".join(f"{v:>9.4f}" for v in pn) + f" | {mn:>8.4f}")
    print(f"ON/OFF mean: {mn / mo:>6.4f}x   ON better on {wins}/{len(seeds)} seeds")
    print(f"paired per-seed diff: {dm:+.3f}% +- {sem:.3f}% (SEM), stdev {sd:.3f}%")
    if ident:
        print("  -> BIT-IDENTICAL across the two arms")
    elif abs(dm) <= 2 * sem:
        print(f"  -> NOT RESOLVED: |{dm:+.2f}%| is inside 2*SEM ({2 * sem:.2f}%). The swap does move "
              f"the latent, but not measurably toward or away from fp16 at this seed count.")
    else:
        print(f"  -> RESOLVED at {len(seeds)} seeds: {dm:+.2f}%")
    return dict(per_seed_off=off, per_seed_on=on, mean_off=mo, mean_on=mn, ratio=mn / mo,
                on_wins=wins, identical=ident, paired_diff_pct_mean=dm,
                paired_diff_pct_stdev=sd, paired_diff_pct_sem=sem,
                resolved=bool(abs(dm) > 2 * sem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    args = ap.parse_args()

    seeds = [1234 + i for i in range(args.seeds)]
    mode, plain = MODES[args.bits]
    calib = ks.CALIB.get(mode)

    print(f"fp16 references for {len(seeds)} seeds ...")
    refs = fp16_refs(seeds)
    off, off_calls = arm(False, mode, calib, seeds, refs, plain)
    on, on_calls = arm(True, mode, calib, seeds, refs, plain)
    res = report(f"W{args.bits}A{args.bits} PTQ", off, on, seeds)

    print(f"\nnon-vacuity check on `{plain}` over {len(seeds)} timed seeds:")
    print(f"  OFF arm called it {off_calls} times, ON arm {on_calls} times")
    vacuous = (on_calls != 0) or (off_calls == 0)
    if vacuous:
        print("  FAIL: the two arms did not run different code -- this verdict means NOTHING.")
    else:
        print("  PASS: the swap fired, so the relL2 above is comparing the two implementations.")
    res["off_plain_calls"] = off_calls
    res["on_plain_calls"] = on_calls
    res["vacuous"] = bool(vacuous)

    os.makedirs(f"{D}/data", exist_ok=True)
    path = f"{D}/data/quality_gn_fast_a{args.bits}.json"
    json.dump({"bits": args.bits, "seeds": seeds, "mode": mode, **res}, open(path, "w"), indent=1)
    print(f"wrote {path}")
    return 1 if res["vacuous"] else 0


if __name__ == "__main__":
    sys.exit(main())
