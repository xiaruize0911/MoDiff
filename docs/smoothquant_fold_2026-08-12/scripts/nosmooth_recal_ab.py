"""The arm that separates "folding s is bad" from "the oversized scale happens to help".

The shipped W4A4 file changes TWO things at once relative to the landed nosmooth default, and §5e
could not tell them apart:

  shipped   fold s into the weights (weight error 1.215x, probe §2)  AND  a scale derived from the
            smoothed range, which is correct for smoothed input
  nosmooth  no fold (weight error back to 0.1293)                    AND  that same scale applied to
            UNSMOOTHED input, where it is a median 5.13x too large -- 43% of channels clip (probe §3)

nosmooth wins by 32%. But it wins while clipping 43% of channels, so either the weight-error term is
big enough to pay for that, or the clipping is itself doing something useful (clipping outliers is
what a clip-optimal calibration does on purpose -- §5b measured an 80-candidate clip search).

THIS ARM REMOVES BOTH. `recal` keeps the weights unfolded and uses the scale a correct unsmoothed
calibration would have produced, 7/max_c(act_max_c), recovered and gated in smoothquant_fold_probe.py
(the gate: static_scale is 7/max_c(act_max_c/s_c) by construction, and the recovery reproduces that
to 7.0000 on 70/70 layers).

  PREDICTION, stated before the run. If the recorded mechanism is right -- folding widens each output
  channel's weight range and at 15 levels that costs more than the clipping it prevents -- then recal,
  which has neither the widened range nor the clipping, must beat BOTH. If recal instead loses to
  nosmooth, the 32% was never really about the fold and the clipping is load-bearing.

Protocol identical to w4a4_ab.py / w4a4_defaults_verify.py: DDIM S=50, batch 8, seeds
{1234, 20260805, 777}, latent relL2 against a per-seed fp16 reference, first run per arm discarded.
All three arms in ONE process so the fp16 reference is shared.

Run: python docs/smoothquant_fold_2026-08-12/scripts/nosmooth_recal_ab.py    # ~15 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

PROBE = "docs/smoothquant_fold_2026-08-12/data/fold_probe.json"
RECAL = "docs/smoothquant_fold_2026-08-12/data/int4_nosmooth_recal.pt"
OUT = "docs/smoothquant_fold_2026-08-12/data/nosmooth_recal_ab.json"
ARMS = {"shipped": "integration/calibration/int4_calibration_realckpt.pt",
        "nosmooth": "integration/calibration/int4_calibration_nosmooth.pt",
        "recal": RECAL}
SEEDS = [1234, 20260805, 777]
#: w4a4_defaults.json for the PTQ rows (same container, same protocol) and w4a4_ab.json §5a for
#: MoDiff/nosmooth. CORRECTED 2026-08-12: this table first carried 0.3540 for MoDiff/nosmooth, which
#: is the QDIFF file's number -- MoDiff's landed default is qdiff, not nosmooth, so the two are
#: different arms. The right reference is §5a's shipped_nosmooth row, 0.3963, and the run reproduced
#: it at 0.3974. The first log therefore printed a 12% "discrepancy" that was a mislabel, not a
#: measurement disagreement.
PRIOR = {("int4_baseline", "shipped"): 0.7121, ("int4_baseline", "nosmooth"): 0.4823,
         ("int4", "shipped"): 0.4220, ("int4", "nosmooth"): 0.3963}


def build_recal():
    """7/max_c(act_max_c) per layer, as bare floats so the loader takes the no-fold path."""
    if not os.path.exists(PROBE):
        print(f"FAIL: missing {PROBE} -- run smoothquant_fold_probe.py first")
        return None
    probe = json.load(open(PROBE))
    scales = {r["layer"]: float(r["static_scale_unsmoothed"]) for r in probe["layers"]}
    shipped = torch.load(ARMS["shipped"], map_location="cpu", weights_only=True)
    if set(scales) != set(shipped):
        print(f"FAIL: recovered {len(scales)} layers, shipped has {len(shipped)}")
        return None
    torch.save(scales, RECAL)
    r = [scales[k] / (shipped[k]["static_scale"] if isinstance(shipped[k], dict) else shipped[k])
         for k in shipped]
    print(f"recal: {len(scales)} bare floats, {statistics.median(r):.3f}x the shipped scale "
          f"(median; {min(r):.3f}-{max(r):.3f}) -- smaller, so a coarser grid that cannot clip")
    return scales


def main():
    if build_recal() is None:
        return 1
    H.STEPS, H.BATCH = 50, 8
    os.environ["MODIFF_LINEAR"] = "0"

    print("\nfp16 references ...", flush=True)
    rf, mf, sf = H.build("fp16", None, "static")
    refs = {}
    for s in SEEDS:
        H.SEED = s
        H.latent(rf, mf, sf)
        refs[s] = H.latent(rf, mf, sf)[0].float()
    del rf, mf, sf
    torch.cuda.empty_cache()

    out = {}
    for mode, kind in (("int4_baseline", "PTQ"), ("int4", "MoDiff")):
        for arm, cal in ARMS.items():
            r, m, s = H.build(mode, cal, "static" if "baseline" in mode else "dynamic")
            H.SEED = SEEDS[0]
            H.latent(r, m, s)                               # discard: attention self-calibration
            rels = []
            for sd in SEEDS:
                H.SEED = sd
                H.latent(r, m, s)
                lat, _ = H.latent(r, m, s)
                rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
            out[f"{mode}/{arm}"] = {"path": cal, "relL2": rels, "mean": statistics.mean(rels)}
            p = PRIOR.get((mode, arm))
            tag = f"   (prior {p:.4f})" if p else ""
            print(f"  {kind:6s} {arm:9s} {statistics.mean(rels):.4f}  "
                  f"{[round(x, 4) for x in rels]}{tag}", flush=True)
            del r, m, s
            torch.cuda.empty_cache()

    print(f"\n{'arm':16s} {'shipped':>9} {'nosmooth':>9} {'recal':>9}   best")
    verdict = {}
    for mode, kind in (("int4_baseline", "PTQ"), ("int4", "MoDiff")):
        v = {a: out[f"{mode}/{a}"]["mean"] for a in ARMS}
        best = min(v, key=v.get)
        verdict[mode] = {**v, "best": best,
                         "recal_over_nosmooth": v["recal"] / v["nosmooth"]}
        print(f"W4A4 {kind:11s} {v['shipped']:9.4f} {v['nosmooth']:9.4f} {v['recal']:9.4f}   {best}")

    beats = [m for m in verdict if verdict[m]["best"] == "recal"]
    print()
    if len(beats) == 2:
        print("PREDICTION HELD: recal wins on both axes -- the fold is the problem, and the "
              "clipping nosmooth pays for is a cost it absorbs, not a benefit.")
    elif not beats:
        print("PREDICTION REFUTED: recal loses on both axes despite having neither the widened "
              "weight range nor the clipping. The oversized scale is load-bearing.")
    else:
        print(f"PREDICTION SPLIT: recal wins only on {beats[0]}. The two axes do not share a story.")

    json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH, "arms": ARMS,
               "prior": {f"{m}/{a}": v for (m, a), v in PRIOR.items()},
               "results": out, "verdict": verdict}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
