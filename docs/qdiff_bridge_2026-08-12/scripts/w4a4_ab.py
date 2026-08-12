"""W4A4 with Q-Diffusion scales, decomposed so SmoothQuant is not confounded with calibration.

W4A4 is where quantization currently fails outright: FID 277.96 (PTQ) / 200.14 (MoDiff) against
fp16's 7.80 -- literal fog and ghost outlines respectively. It was never recalibrated, because the
Q-Diffusion export was int8-only.

TWO THINGS DIFFER between the shipped int4 file and a qdiff one, and they must not be reported as
one. int4's shipped file is {name: {"static_scale", "smooth_scale"}} with SmoothQuant LIVE
(per-input-channel, 2.96-5.39); qdiff has no SmoothQuant, so its export is bare floats and loading it
turns smoothing OFF. The exported scales are 3.8x (sym) / 5.4x (mse) LARGER than shipped -- the
opposite direction from int8, and about the size of the smooth factor, so the confound is not
hypothetical.

FOUR SCALE FILES, and the third is the control that separates them:

  shipped            dict, SmoothQuant ON              the current default
  shipped_nosmooth   the same static_scale, floats     isolates what SmoothQuant alone is worth
  qdiff_sym          bare floats, absmax at 4 bits     calibration change + no smoothing
  qdiff_mse          bare floats, clip-search at 4 bits

qdiff-vs-shipped is a two-factor bundle; qdiff-vs-shipped_nosmooth is the calibration change alone.

Calibrated AT --weight_bit 4 --act_bit 4. Both matter: W4A4 quantizes weights too, so the activations
the quantizer observes are the 4-bit-weight ones; and an 8-bit-optimal clip rescaled to 15 levels is
not clip-optimal, which was measured directly (data/a4_clip_optimal_ab.json).

Run: python docs/qdiff_bridge_2026-08-12/scripts/w4a4_ab.py
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

D = "docs/qdiff_bridge_2026-08-12/data"
FILES = {"shipped": "integration/calibration/int4_calibration_realckpt.pt",
         "shipped_nosmooth": f"{D}/int4_shipped_nosmooth.pt",
         "qdiff_sym": f"{D}/qdiff_w4a4_sym.pt",
         "qdiff_mse": f"{D}/qdiff_w4a4_mse.pt"}
OUT = f"{D}/w4a4_ab.json"
SEEDS = [1234, 20260805, 777]
#: committed reference, docs/modiff_correctness_2026-08-03 (same steps/batch/seed protocol)
REFERENCE = {"W4A4 PTQ / shipped": 0.7837, "W4A4 MoDiff / shipped": 0.4176}


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
    for mode, kind in (("int4_baseline", "PTQ"), ("int4", "MoDiff")):
        for cname, cal in FILES.items():
            os.environ["MODIFF_LINEAR"] = "0"
            H.CALIB["int4"] = cal
            r, m, s = H.build(mode, cal, "static" if "baseline" in mode else "dynamic")
            H.SEED = SEEDS[0]
            H.latent(r, m, s)                               # discard: attention self-calibration
            rels = []
            for sd in SEEDS:
                H.SEED = sd
                H.latent(r, m, s)
                lat, _ = H.latent(r, m, s)
                rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
            k = f"W4A4 {kind} / {cname}"
            out[k] = {"relL2": rels, "mean": statistics.mean(rels)}
            ref = REFERENCE.get(k)
            tag = f"  (committed {ref:.4f})" if ref else ""
            print(f"  {k:34s} {statistics.mean(rels):.4f}  "
                  f"{[round(x, 4) for x in rels]}{tag}", flush=True)
            del r, m, s
            torch.cuda.empty_cache()
    os.environ.pop("MODIFF_LINEAR", None)

    print(f"\n{'arm':14s} {'shipped':>9} {'no-smooth':>10} {'qdiff sym':>10} {'qdiff mse':>10}   best")
    for kind in ("PTQ", "MoDiff"):
        v = {c: out[f"W4A4 {kind} / {c}"]["mean"] for c in FILES}
        best = min(v, key=v.get)
        print(f"W4A4 {kind:9s} {v['shipped']:9.4f} {v['shipped_nosmooth']:10.4f} "
              f"{v['qdiff_sym']:10.4f} {v['qdiff_mse']:10.4f}   {best}")
        print(f"     SmoothQuant alone : {v['shipped']:.4f} -> {v['shipped_nosmooth']:.4f} "
              f"({v['shipped_nosmooth'] / v['shipped']:.2f}x)")
        print(f"     calibration alone : {v['shipped_nosmooth']:.4f} -> "
              f"{min(v['qdiff_sym'], v['qdiff_mse']):.4f} "
              f"({min(v['qdiff_sym'], v['qdiff_mse']) / v['shipped_nosmooth']:.2f}x)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH,
               "reference": REFERENCE, "results": out}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
