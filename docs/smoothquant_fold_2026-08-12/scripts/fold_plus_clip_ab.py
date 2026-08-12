"""If the clipping is what wins, then SmoothQuant + clipping -- never measured -- should win more.

WHERE THIS COMES FROM. nosmooth_recal_ab.py split the landed nosmooth win into its two factors and
the answer was not the recorded one:

  shipped   fold s, scale matched to the smoothed range (no clipping)      0.7120
  recal     no fold, scale matched to the true range    (no clipping)      0.8622
  nosmooth  no fold, the smoothed-range scale on unsmoothed input          0.4887   <- landed

At MATCHED no-clipping, folding is 17% BETTER (0.7120 against 0.8622), so the candidate mechanism --
"folding widens the weight range and that costs more than the clipping it prevents" -- is refuted:
folding's 1.215x weight-error cost is real but is outpaid by what smoothing buys the activation.
What actually wins is the CLIPPING nosmooth gets by accident, because its scale is a median 5.13x too
large for unsmoothed input. Removing that clipping costs 76% (0.4887 -> 0.8622).

So the two effects are independent and both helpful, and no arm has ever had both.

THE ARM. Keep the fold, then deliberately over-scale by k, so the smoothed activation's max lands at
7k and clips the same way nosmooth's does. k is exactly "times past the +-7 ceiling": nosmooth's
effective k is that 5.13. Swept, because a single point cannot tell "clipping helps" from "this
particular clipping helps", and because over-clipping has to turn back up eventually -- if it does
not, something is wrong with the reasoning, not with the model.

Protocol identical to the other W4A4 A/Bs: DDIM S=50, batch 8, seeds {1234, 20260805, 777}, latent
relL2 against a per-seed fp16 reference, first run per arm discarded, all arms in one process.

Run: python docs/smoothquant_fold_2026-08-12/scripts/fold_plus_clip_ab.py    # ~16 min, needs the GPU
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

SHIPPED = "integration/calibration/int4_calibration_realckpt.pt"
NOSMOOTH = "integration/calibration/int4_calibration_nosmooth.pt"
TMP = "docs/smoothquant_fold_2026-08-12/data/int4_foldclip_k{}.pt"
OUT = "docs/smoothquant_fold_2026-08-12/data/fold_plus_clip_ab.json"
SEEDS = [1234, 20260805, 777]
KS = [2.5, 5.0, 10.0, 20.0]
#: measured this session, same container and protocol (nosmooth_recal_ab.json)
PRIOR = {"PTQ": {"shipped": 0.7120, "nosmooth": 0.4887, "recal": 0.8622},
         "MoDiff": {"shipped": 0.4176, "nosmooth": 0.3974, "recal": 0.3964}}


def foldclip_file(k):
    """The shipped dict with static_scale * k: fold preserved, smoothed max pushed to 7k."""
    shipped = torch.load(SHIPPED, map_location="cpu", weights_only=True)
    out = {}
    for name, e in shipped.items():
        if not isinstance(e, dict):
            out[name] = float(e) * k                 # identity-smoothed layer, nothing to preserve
        else:
            out[name] = {"static_scale": float(e["static_scale"]) * k,
                         "smooth_scale": e["smooth_scale"]}
    path = TMP.format(k)
    torch.save(out, path)
    return path


def measure(mode, cal, refs):
    delta = "static" if "baseline" in mode else "dynamic"
    r, m, s = H.build(mode, cal, delta)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                                # discard: attention self-calibration
    rels = []
    for sd in SEEDS:
        H.SEED = sd
        H.latent(r, m, s)
        lat, _ = H.latent(r, m, s)
        rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
    del r, m, s
    torch.cuda.empty_cache()
    return rels


def main():
    H.STEPS, H.BATCH = 50, 8
    os.environ["MODIFF_LINEAR"] = "0"

    print("fp16 references ...", flush=True)
    rf, mf, sf = H.build("fp16", None, "static")
    refs = {}
    for s in SEEDS:
        H.SEED = s
        H.latent(rf, mf, sf)
        refs[s] = H.latent(rf, mf, sf)[0].float()
    del rf, mf, sf
    torch.cuda.empty_cache()

    out = {}
    print("\nPTQ sweep -- fold kept, smoothed max pushed to 7k", flush=True)
    rels = measure("int4_baseline", NOSMOOTH, refs)
    out["PTQ/nosmooth"] = {"mean": statistics.mean(rels), "relL2": rels}
    print(f"  nosmooth (control, effective k~5.13)  {statistics.mean(rels):.4f}  "
          f"(prior {PRIOR['PTQ']['nosmooth']:.4f})", flush=True)
    for k in KS:
        rels = measure("int4_baseline", foldclip_file(k), refs)
        out[f"PTQ/foldclip_k{k}"] = {"mean": statistics.mean(rels), "relL2": rels, "k": k}
        print(f"  fold + clip k={k:<5}                    {statistics.mean(rels):.4f}  "
              f"{[round(x, 4) for x in rels]}", flush=True)

    ptq = {k: out[f"PTQ/foldclip_k{k}"]["mean"] for k in KS}
    best_k = min(ptq, key=ptq.get)
    print(f"\n  best k on PTQ = {best_k} at {ptq[best_k]:.4f}; nosmooth control "
          f"{out['PTQ/nosmooth']['mean']:.4f}, shipped (k=1) {PRIOR['PTQ']['shipped']:.4f}")

    print(f"\nMoDiff at k={best_k}", flush=True)
    rels = measure("int4", NOSMOOTH, refs)
    out["MoDiff/nosmooth"] = {"mean": statistics.mean(rels), "relL2": rels}
    print(f"  nosmooth (control)                    {statistics.mean(rels):.4f}", flush=True)
    rels = measure("int4", foldclip_file(best_k), refs)
    out[f"MoDiff/foldclip_k{best_k}"] = {"mean": statistics.mean(rels), "relL2": rels, "k": best_k}
    print(f"  fold + clip k={best_k:<5}                    {statistics.mean(rels):.4f}", flush=True)

    print(f"\n{'arm':28s} {'PTQ':>9} {'MoDiff':>9}")
    print(f"{'shipped (fold, k=1)':28s} {PRIOR['PTQ']['shipped']:9.4f} {PRIOR['MoDiff']['shipped']:9.4f}")
    print(f"{'recal (no fold, no clip)':28s} {PRIOR['PTQ']['recal']:9.4f} {PRIOR['MoDiff']['recal']:9.4f}")
    print(f"{'nosmooth (LANDED)':28s} {out['PTQ/nosmooth']['mean']:9.4f} "
          f"{out['MoDiff/nosmooth']['mean']:9.4f}")
    print(f"{'fold + clip k=' + str(best_k):28s} {ptq[best_k]:9.4f} "
          f"{out[f'MoDiff/foldclip_k{best_k}']['mean']:9.4f}")

    win_ptq = ptq[best_k] < out["PTQ/nosmooth"]["mean"]
    win_mod = out[f"MoDiff/foldclip_k{best_k}"]["mean"] < out["MoDiff/nosmooth"]["mean"]
    turns_up = ptq[KS[-1]] > ptq[best_k]
    print()
    print(f"clipping is a CURVE, not monotone: k={KS[-1]} reads {ptq[KS[-1]]:.4f} against "
          f"{ptq[best_k]:.4f} at k={best_k}" if turns_up else
          f"WARNING: error still falling at k={KS[-1]} -- the sweep did not bracket the optimum")
    if win_ptq and win_mod:
        print(f"BOTH EFFECTS COMPOSE: fold + clip beats the landed nosmooth on both axes. The two "
              f"factors are independent and the landed default has only one of them.")
    elif win_ptq or win_mod:
        print(f"PARTIAL: fold + clip wins on {'PTQ' if win_ptq else 'MoDiff'} only.")
    else:
        print(f"THEY DO NOT COMPOSE: fold + clip loses to nosmooth despite having both factors.")

    json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH, "ks": KS, "best_k": best_k,
               "prior": PRIOR, "results": out,
               "verdict": {"foldclip_beats_nosmooth_ptq": win_ptq,
                           "foldclip_beats_nosmooth_modiff": win_mod,
                           "sweep_brackets_optimum": turns_up}},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
