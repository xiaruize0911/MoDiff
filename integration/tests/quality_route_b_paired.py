"""Paired-seed relL2 for route (b): does feeding flash int8 from the GEMM epilogue change the image?

Route (b) is not a pure-plumbing change. It removes the three `aq_*` quantize passes AND swaps the
score kernel, from the mma kernel that reads pre-transposed qi/ki/vt to the packed kernel that
gathers per-token bytes. So it has to be measured on quality, not just asserted to be equivalent.

  arm OFF : MODIFF_FUSE_QKV_I8=0 -- production
  arm ON  : MODIFF_FUSE_QKV_I8=1 -- the 10 hd=48 blocks take the gather path

EXPECTED DIRECTION, written before running so the result cannot be read to taste. The int8 codes are
provably the same: the per-column out scale is built from the same frozen _fq_sqc/_fq_skc/_fq_svv the
aq_* kernels use, and at the kernel level the two arms' codes are BIT-IDENTICAL (0% differing,
integration/tests/bench_flash_packed_vs_unpacked.py). What differs is fp16 accumulation order inside
flash. Measured there against an fp32 reference: 4.17e-3 vs 4.17e-3 at T=256 and 3.73e-3 vs 3.87e-3
at T=64 -- i.e. the arms are equally close to the truth, and their disagreement with each other
(2.6e-3 at T=64) is smaller than their common distance from it. So the PAIRED PER-SEED DIFFERENCE
should be small relative to its own spread -- not necessarily zero, since fp16 order matters.

NOT COMPARABLE, and worth stating because it is an easy mistake: route (a)'s recorded 0.01710 is an
ARM-TO-ARM latent relL2, whereas this script measures each arm against fp16 and compares those two
numbers. They are different quantities and must not share an axis. An arm-to-arm figure would need
its own run differencing the two arms' latents at a fixed seed.

A large, RESOLVED difference here would mean the "the three scales already line up" claim is wrong
somewhere, and the fusion should not ship on a +0.79 ms/step gain.

CONTROLS, because a null is only informative if the instrument can show a non-null:
  * the same arm twice -- must be bit-identical, or the harness itself is noisy at this level
  * hd=24 blocks are ineligible, so a change concentrated there would indicate the gate leaked

PAIRED per docs/act_bits_2026-08-05: relL2 at batch 8 varies 10-30% run to run, so both arms see the
same seeds and the same per-seed fp16 reference, and the statistic is the per-seed difference.

Run: python integration/tests/quality_route_b_paired.py [--seeds 1234,5678,9012]
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
import dynamic_delta_ab as H                                             # noqa: E402

CALIB = H.CALIB["int8"]


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


def arm(route_b, seeds, refs, refresh=4, proj_refresh=4):
    """relL2 per seed for one configuration, from one freshly built model.

    MODIFF_FUSE_QKV_I8 is read per call by the gate, but it is set before the build anyway so the arm
    is self-describing rather than depending on call order -- the same reason quality_updown_a4_paired
    sets MODIFF_UPDOWN_A4 up front.
    """
    os.environ["MODIFF_DELTA_REFRESH"] = str(refresh)
    os.environ["MODIFF_LINEAR_DELTA_REFRESH"] = str(proj_refresh)
    os.environ["MODIFF_FUSE_QKV_I8"] = "1" if route_b else "0"
    r, m, s = H.build("int8", CALIB, "dynamic")
    H.latent(r, m, s)                                    # run 1 is not steady state -- discard
    rel = {}
    for seed in seeds:
        H.SEED = seed
        lat, _ = H.latent(r, m, s)
        ref = refs[seed]
        rel[seed] = float((lat - ref).norm() / ref.norm())
    del r, m, s
    torch.cuda.empty_cache()
    return rel


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
    print(f"\n=== {name} ===")
    print(f"{'arm':>14} | " + " ".join(f"{s:>9}" for s in seeds) + f" | {'mean':>8}")
    print("-" * (18 + 10 * len(seeds) + 11))
    print(f"{'OFF (today)':>14} | " + " ".join(f"{v:>9.4f}" for v in po) + f" | {mo:>8.4f}")
    print(f"{'ON  (route b)':>14} | " + " ".join(f"{v:>9.4f}" for v in pn) + f" | {mn:>8.4f}")
    print(f"ON/OFF mean: {mn / mo:>6.4f}x   ON better on {wins}/{len(seeds)} seeds")
    print(f"paired per-seed diff: {dm:+.3f}% +- {sem:.3f}% (SEM), stdev {sd:.3f}%")
    if ident:
        print("  -> BIT-IDENTICAL across the two arms")
    elif abs(dm) <= 2 * sem:
        print(f"  -> NOT RESOLVED: |{dm:+.2f}%| is inside 2*SEM ({2 * sem:.2f}%)")
    else:
        print(f"  -> RESOLVED at {len(seeds)} seeds: {dm:+.2f}%")
    return dict(per_seed_off=off, per_seed_on=on, mean_off=mo, mean_on=mn, ratio=mn / mo,
                on_wins=wins, identical=ident, paired_diff_pct_mean=dm,
                paired_diff_pct_stdev=sd, paired_diff_pct_sem=sem,
                resolved=bool(abs(dm) > 2 * sem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1234,5678,9012")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {seeds}, {torch.cuda.get_device_name(0)}")
    refs = fp16_refs(seeds)
    off = arm(False, seeds, refs)
    on = arm(True, seeds, refs)
    res = report("route (b): int8 qkv -> flash gather, 10 hd=48 blocks", off, on, seeds)

    # Control: the OFF arm rebuilt and re-run. Anything but bit-identical means the instrument's own
    # noise floor is above the effect and the comparison above cannot be read.
    off2 = arm(False, seeds, refs)
    ctl_ident = all(abs(off[s] - off2[s]) < 1e-12 for s in seeds)
    print(f"\ncontrol (OFF twice): {'bit-identical' if ctl_ident else 'DIFFERS'}"
          + ("" if ctl_ident else "  <- the arms above are not separable from run-to-run noise"))
    res["control_off_twice_identical"] = bool(ctl_ident)
    res["per_seed_off_rerun"] = off2
    # Deliberately NOT recording route (a)'s 0.01710 here: it is an arm-to-arm relL2 and these are
    # relL2-vs-fp16, so shipping it in the same record invites a figure that puts them on one axis.

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(batch=H.BATCH, steps=H.STEPS, seeds=seeds,
                           gpu=torch.cuda.get_device_name(0), result=res), f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
