"""Paired-seed relL2 for the `a4` change on the eight updown ResBlocks.

The 2026-08-10 commit gave those eight layers the activation bit-width they had been ignoring: the
resize fusion kernel took no code ceiling, so it clamped at the int8 store's literal 127 while the
other 62 convs clamped at act_q. That is a NUMERICS change on the K>1 reuse path, and no timing run
can say what it is worth. This is the measurement that can.

WHAT IS ISOLATED, and why it needs a flag. At K>1 the pre-commit code had TWO differences at once:
the fusion declined on refresh steps (so 6/8 fused at K=4) *and* the fused ones ignored the ceiling.
Comparing against the old code would measure both together. So the fusion is held ON in both arms and
only `MODIFF_UPDOWN_A4` moves:

    arm 127 : MODIFF_UPDOWN_A4=0 -- the eight layers clamp at 127 (the pre-commit defect), 8/8 fused
    arm  Q_b: default            -- they clamp at act_q, consistent with the other 62,  8/8 fused

EXPECTED DIRECTION, stated before running so the result cannot be read to taste. Clamping at 127 lets
a 4-bit layer keep resolution a true 4-bit quantizer would have thrown away, so the DEFECTIVE arm
should look BETTER on relL2 -- the same way commit 3be1986 found the old A4/A3 rows had been
flattered for the other 62 layers. A correction that makes the number worse is the correct outcome
here; what matters is the size, and whether A8 is untouched.

CONTROLS, because a null result is only informative if the instrument can show a null:
  * W8A8 -- act_q is 127, so a4 is False in both arms and the two must come out IDENTICAL. If they
    do not, the flag is reaching something it should not.
  * K=1 at A4 -- the scale is Q_b/absmax, so no code can exceed Q_b and the ceiling is a near-no-op.
    A large difference here would mean the effect is not the ceiling.

PAIRED, per docs/act_bits_2026-08-05: relL2 at batch 8 varies 10-30% run to run, which is larger than
several of the gaps involved, so every arm sees the same seeds and the same per-seed fp16 reference,
and the reported statistic is "how often, and by how much, on the SAME seed".

Run: python integration/tests/quality_updown_a4_paired.py [--seeds 1234,5678,9012]
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

import torch                                                          # noqa: E402
import dynamic_delta_ab as H                                          # noqa: E402

# The real-checkpoint calibration, not the stub's: H.CALIB names it.
CALIB = H.CALIB["int8"]


def fp16_refs(seeds):
    """Per-seed fp16 latent, the denominator of every relL2 below."""
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


def arm(act_bits, refresh, updown_a4, seeds, refs):
    """relL2 per seed for one configuration, from one freshly built model.

    MODIFF_ACT_BITS and MODIFF_DELTA_REFRESH are read in the conv wrappers' __init__, so they have
    to be set before the build. MODIFF_UPDOWN_A4 is read per call, but it is set here too so the
    arm is self-describing rather than depending on call order.
    """
    os.environ["MODIFF_ACT_BITS"] = str(act_bits)
    os.environ["MODIFF_DELTA_REFRESH"] = str(refresh)
    os.environ["MODIFF_UPDOWN_A4"] = "1" if updown_a4 else "0"
    os.environ["MODIFF_UPDOWN_FUSE_REFRESH"] = "1"       # hold the fusion ON in BOTH arms
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


def report(name, a127, aqb, seeds):
    """a127 = the defective arm, aqb = the corrected one. Paired, per seed."""
    pa = [a127[s] for s in seeds]
    pb = [aqb[s] for s in seeds]
    ma, mb = statistics.mean(pa), statistics.mean(pb)
    # "wins" counts seeds where the CORRECTED arm has lower relL2. Expected to be low: the defect
    # was extra resolution.
    wins = sum(1 for x, y in zip(pb, pa) if x < y)
    worst = max((y - x) / x for x, y in zip(pa, pb)) if all(pa) else float("nan")
    ident = all(abs(x - y) < 1e-12 for x, y in zip(pa, pb))
    print(f"\n=== {name} ===")
    print(f"{'arm':>12} | " + " ".join(f"{s:>9}" for s in seeds) + f" | {'mean':>8}")
    print("-" * (16 + 10 * len(seeds) + 11))
    print(f"{'clamp 127':>12} | " + " ".join(f"{v:>9.4f}" for v in pa) + f" | {ma:>8.4f}")
    print(f"{'clamp Q_b':>12} | " + " ".join(f"{v:>9.4f}" for v in pb) + f" | {mb:>8.4f}")
    print(f"corrected/defective mean: {mb / ma:>6.3f}x   corrected wins {wins}/{len(seeds)} seeds"
          f"   worst per-seed move {worst:>+.1%}")
    # The paired per-seed difference and its SEM. A ratio alone cannot say whether an effect is
    # there: at 3 seeds this comparison read 1.061x / 1 win, and at 8 seeds the SIGN FLIPPED to
    # 0.967x / 5 wins. The per-seed spread is what decides, so report it.
    per = [(y - x) / x * 100.0 for x, y in zip(pa, pb)]
    dm = statistics.mean(per)
    sd = statistics.stdev(per) if len(per) > 1 else 0.0
    sem = sd / len(per) ** 0.5 if len(per) > 1 else 0.0
    print(f"paired per-seed diff: {dm:+.2f}% +- {sem:.2f}% (SEM), stdev {sd:.2f}%")
    resolved = abs(dm) > 2 * sem
    if ident:
        print("  -> BIT-IDENTICAL across the two arms")
    elif not resolved:
        print(f"  -> NOT RESOLVED: |{dm:+.1f}%| is inside 2*SEM ({2 * sem:.1f}%). No effect this "
              f"protocol can see at {len(seeds)} seeds.")
    return {"per_seed_127": a127, "per_seed_qb": aqb, "mean_127": ma, "mean_qb": mb,
            "ratio": mb / ma, "corrected_wins": wins, "worst_pct": worst, "identical": ident,
            "paired_diff_pct_mean": dm, "paired_diff_pct_stdev": sd, "paired_diff_pct_sem": sem,
            "resolved": bool(resolved)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1234,5678,9012")
    ap.add_argument("--only", default="",
                    help="substring filter on the case label; both controls came out exactly "
                         "identical at 3 seeds, so widening the seed list only needs the effect")
    ap.add_argument("--out",
                    default="docs/updown_refresh_fusion_2026-08-10/data/quality_a4_paired.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {seeds}, real checkpoint, "
          f"{torch.cuda.get_device_name(0)}")
    print("relL2 vs the SAME-seed fp16 latent; run 1 discarded per arm (attention self-calibrates)")

    refs = fp16_refs(seeds)

    out = {"batch": H.BATCH, "steps": H.STEPS, "seeds": seeds, "cases": {}}
    # (label, act_bits, refresh) -- the effect, then the two controls.
    cases = [("W8A4, K=4  (the effect)", 4, 4),
             ("W8A4, K=1  (control: ceiling is a near-no-op)", 4, 1),
             ("W8A8, K=4  (control: must be identical)", 8, 4)]
    if args.only:
        cases = [c for c in cases if args.only in c[0]]
    for label, bits, k in cases:
        a127 = arm(bits, k, False, seeds, refs)
        aqb = arm(bits, k, True, seeds, refs)
        out["cases"][label] = report(label, a127, aqb, seeds)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
