"""What does a one-step-stale delta scale cost? OPEN_ITEMS B7's first gate, measured.

B7 reads: "Part 3, the a_hat-aware flash qout epilogue: the first gate is a numerics decision, not code.
The delta scale must either come from a previous step's `report_next` or be accepted one step stale, and
the relL2 cost of stale is unmeasured."

IT IS MEASURABLE WITH AN ENV FLAG, and has been all along. `MODIFF_DELTA_REPORT` (default 0) selects
exactly the scheme B7 describes. From `int8_optimized._delta_gn_dynamic_args`:

    report on : "the quantize kernel quantizes with the pair currently in force while publishing the
                 next pair into the OTHER buffers. Flip afterwards so later steps use the new one."
    report off: a separate reduction-only pass measures this call's own delta range, so the scale is
                fresh and used immediately -- at the cost of that extra pass.

So "accepted one step stale" is `MODIFF_DELTA_REPORT=1`, and the pass it saves is what makes the qout
epilogue viable. `MODIFF_DELTA_SAFETY` (default 1.15) is the headroom multiplier applied to a published
scale, which exists precisely because it will be applied to a delta it did not measure.

NON-VACUITY IS CHECKED, and it needs a different mechanism than usual. This flag does not switch kernels
-- it changes which branch of `_delta_gn_dynamic_args` runs and hence the `report_next` argument the same
kernel receives. A kernel-name counter cannot see it. So this wraps the accessor and records the
`report_next` element of every returned tuple: it must be True in the ON arm and False in the OFF arm,
and the counts must be non-zero in both. Four gates in this tree have already reported a confident
nothing because they asserted no such thing (see docs/OPEN_ITEMS.md, archive).

MEASURED AGAINST fp16 at matched seeds, not arm-to-arm, for the reason quality_gn_fast_paired.py records:
both arms approximate fp16, and a scheme that merely rounds differently is free even though its latents
differ.

Run: python integration/tests/quality_delta_report_paired.py [--seeds 8] [--bits 8]
Writes docs/gn_fast_reduce_2026-08-16/data/delta_report_a<bits>.json
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

D = "docs/gn_fast_reduce_2026-08-16"
#: index of report_next in the tuple _delta_gn_dynamic_args returns
REPORT_NEXT_IDX = 5


def fp16_refs(seeds):
    r, m, s = H.build("fp16", None, "dynamic")
    H.latent(r, m, s)
    out = {}
    for seed in seeds:
        H.SEED = seed
        out[seed], _ = H.latent(r, m, s)
    del r, m, s
    torch.cuda.empty_cache()
    return out


def arm(report, mode, calib, seeds, refs):
    """relL2 per seed, plus a tally of the report_next values the kernels actually received.

    MODIFF_DELTA_REPORT is read in the conv wrapper's __init__, so it must be set before build().
    """
    os.environ["MODIFF_DELTA_REPORT"] = "1" if report else "0"
    r, m, s = H.build(mode, calib, "dynamic")

    # Wrap the accessor on the CLASSES, after build, so every conv instance is covered and the
    # original is restored even if a seed raises.
    tally = {True: 0, False: 0, "other": 0}
    patched = []
    for modname, cls, meth in (("int8_optimized", "OptimizedInt8Conv2d", "_delta_gn_dynamic_args"),
                               ("int4_optimized", "OptimizedInt4Conv2d", "_delta_gn_dynamic_args_i4")):
        try:
            mod = __import__(f"integration.kernels.{modname}", fromlist=[cls])
            klass = getattr(mod, cls)
            orig = getattr(klass, meth)
        except (ImportError, AttributeError):
            continue

        def wrapped(self, device, _o=orig):
            t = _o(self, device)
            v = t[REPORT_NEXT_IDX] if len(t) > REPORT_NEXT_IDX else "other"
            tally[v if isinstance(v, bool) else "other"] += 1
            return t
        setattr(klass, meth, wrapped)
        patched.append((klass, meth, orig))

    try:
        H.latent(r, m, s)                                # discard: not steady state
        tally[True] = tally[False] = tally["other"] = 0   # count only the timed seeds
        rel = {}
        for seed in seeds:
            H.SEED = seed
            lat, _ = H.latent(r, m, s)
            rel[seed] = float((lat - refs[seed]).norm() / refs[seed].norm())
    finally:
        for klass, meth, orig in patched:
            setattr(klass, meth, orig)
    del r, m, s
    torch.cuda.empty_cache()
    return rel, dict(tally)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    args = ap.parse_args()
    seeds = [1234 + i for i in range(args.seeds)]
    mode = "int8" if args.bits == 8 else "int4"
    calib = H.CALIB[mode]

    print(f"fp16 references, {len(seeds)} seeds ...")
    refs = fp16_refs(seeds)
    off, off_t = arm(False, mode, calib, seeds, refs)
    on, on_t = arm(True, mode, calib, seeds, refs)

    po = [off[s] for s in seeds]
    pn = [on[s] for s in seeds]
    mo, mn = statistics.mean(po), statistics.mean(pn)
    per = [(y - x) / x * 100.0 for x, y in zip(po, pn)]
    dm = statistics.mean(per)
    sd = statistics.stdev(per) if len(per) > 1 else 0.0
    sem = sd / len(per) ** 0.5 if len(per) > 1 else 0.0

    print(f"\n=== W{args.bits}A{args.bits}: one-step-stale delta scale, relL2 vs fp16 (lower better) ===")
    print(f"{'arm':>22} | " + " ".join(f"{s:>8}" for s in seeds) + f" | {'mean':>8}")
    print(f"{'OFF (fresh, extra pass)':>22} | " + " ".join(f"{v:>8.4f}" for v in po) + f" | {mo:>8.4f}")
    print(f"{'ON (stale + safety)':>22} | " + " ".join(f"{v:>8.4f}" for v in pn) + f" | {mn:>8.4f}")
    print(f"paired per-seed diff: {dm:+.3f}% +- {sem:.3f}% (SEM), stdev {sd:.3f}%")
    resolved = abs(dm) > 2 * sem
    print(f"  -> {'RESOLVED' if resolved else 'NOT RESOLVED'}: "
          f"{dm:+.2f}% against 2*SEM {2 * sem:.2f}%")

    print(f"\nnon-vacuity — report_next values the kernels received over {len(seeds)} timed seeds:")
    print(f"  OFF arm: {off_t}")
    print(f"  ON  arm: {on_t}")
    vacuous = not (off_t.get(False, 0) > 0 and on_t.get(True, 0) > 0
                   and off_t.get(True, 0) == 0)
    print("  FAIL: the arms did not differ in report_next — this verdict means NOTHING."
          if vacuous else "  PASS: OFF passed report_next=False, ON passed True.")

    os.makedirs(f"{D}/data", exist_ok=True)
    path = f"{D}/data/delta_report_a{args.bits}.json"
    json.dump({"bits": args.bits, "seeds": seeds, "per_seed_off": off, "per_seed_on": on,
               "mean_off": mo, "mean_on": mn, "paired_diff_pct_mean": dm,
               "paired_diff_pct_sem": sem, "resolved": bool(resolved),
               "tally_off": off_t, "tally_on": on_t, "vacuous": bool(vacuous),
               "safety": os.environ.get("MODIFF_DELTA_SAFETY", "1.15")}, open(path, "w"), indent=1)
    print(f"wrote {path}")
    return 1 if vacuous else 0


if __name__ == "__main__":
    sys.exit(main())
