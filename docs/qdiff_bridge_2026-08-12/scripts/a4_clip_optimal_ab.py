"""At 4-bit activations the NON-CLIPPING (absmax) scale loses. Test the clipping-OPTIMAL one.

Measured in data/w8a4_ab.json, and it refuted the prediction that motivated it:

    config          shipped     qdiff    gain
    W8A8 PTQ         0.2565    0.1138   2.25x
    W8A4 PTQ         0.8073    0.8892   0.91x   <- the fix makes A4 WORSE

MECHANISM. set_static_scale rescales by act_q/127 (int8_optimized.py:1822), so at A4 both files get
the same x7/127 and their RATIO is preserved -- qdiff's scale stays ~2.9x smaller. qdiff is
non-clipping by construction, so at A4 it spends all 15 levels covering the full range including rare
outliers, while the inflated shipped scale clips the tail and concentrates those 15 levels on the bulk
of the distribution. At 15 levels resolution beats coverage. docs/modiff_correctness_2026-08-03
recorded the same principle for the delta table: "with 255 levels that headroom is affordable... With
15 levels it is not."

So absmax is the wrong OBJECTIVE at low activation bit width, whatever produced it -- and qdiff's
other export is built for exactly this. With --a_min_max off, UniformAffineQuantizer runs an
80-candidate clip search minimising lp_loss(p=2.4), i.e. it picks a clipping-optimal range rather than
the maximum. It lost narrowly at A8 (0.1135 vs 0.1082) where clipping is affordable. This asks whether
it wins at A4, where it should.

Three scale files x {PTQ, MoDiff} at MODIFF_ACT_BITS=4, 3 paired seeds against one fp16 reference.

Run: python docs/qdiff_bridge_2026-08-12/scripts/a4_clip_optimal_ab.py
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

FILES = {"shipped": "integration/calibration/int8_calibration_realckpt.pt",
         "qdiff_absmax": "integration/calibration/int8_calibration_qdiff.pt",
         "qdiff_mse": "docs/qdiff_bridge_2026-08-12/data/qdiff_act_mse.pt"}
OUT = "docs/qdiff_bridge_2026-08-12/data/a4_clip_optimal_ab.json"
SEEDS = [1234, 20260805, 777]
#: from data/w8a4_ab.json, so the new arm is judged against the same protocol
KNOWN = {"W8A4 PTQ / shipped": 0.8073, "W8A4 PTQ / qdiff_absmax": 0.8892,
         "W8A4 MoDiff / shipped": 0.1424, "W8A4 MoDiff / qdiff_absmax": 0.1464}


def main():
    for n, p in FILES.items():
        if not os.path.exists(p):
            print(f"FAIL: {n} missing at {p}")
            return 1
    H.STEPS, H.BATCH = 50, 8

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
    for mode, kind in (("int8_baseline", "PTQ"), ("int8", "MoDiff")):
        for cname, cal in FILES.items():
            os.environ["MODIFF_ACT_BITS"] = "4"
            os.environ["MODIFF_LINEAR"] = "0"
            r, m, s = H.build(mode, cal, "static" if "baseline" in mode else "dynamic")
            H.SEED = SEEDS[0]
            H.latent(r, m, s)                              # discard: attention self-calibration
            rels = []
            for sd in SEEDS:
                H.SEED = sd
                H.latent(r, m, s)
                lat, _ = H.latent(r, m, s)
                rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
            k = f"W8A4 {kind} / {cname}"
            out[k] = {"relL2": rels, "mean": statistics.mean(rels)}
            prev = KNOWN.get(k)
            tag = f"   (was {prev:.4f})" if prev else "   <- NEW"
            print(f"  {k:32s} {statistics.mean(rels):.4f}  "
                  f"{[round(x, 4) for x in rels]}{tag}", flush=True)
            del r, m, s
            torch.cuda.empty_cache()
    os.environ.pop("MODIFF_ACT_BITS", None)
    os.environ.pop("MODIFF_LINEAR", None)

    print(f"\n{'arm':16s} {'shipped':>9} {'qdiff absmax':>13} {'qdiff MSE':>11}   best")
    for kind in ("PTQ", "MoDiff"):
        v = {c: out[f"W8A4 {kind} / {c}"]["mean"] for c in FILES}
        best = min(v, key=v.get)
        print(f"W8A4 {kind:11s} {v['shipped']:9.4f} {v['qdiff_absmax']:13.4f} "
              f"{v['qdiff_mse']:11.4f}   {best}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH,
               "act_bits": 4, "results": out}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
