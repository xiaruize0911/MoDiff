"""A9: the 2x2 on the REAL CUDA kernels -- {shipped, qdiff} x {baseline, MoDiff}.

It has to be a 2x2, not a one-dimensional swap, because the activation file is read by BOTH arms:
_forward_first_step (int8_optimized.py:1428) uses static_input_scale at t=T even under MoDiff.

A7's fake-quant harness predicted the ordering with no CUDA at all:
    baseline  0.2558 -> 0.1082   (qdiff wins 3/3 seeds)
    MoDiff    0.0069 -> 0.0088   (a wash; MoDiff reads the static scale only at t=T)
This run is what turns that into a quotable absolute, and a disagreement in ORDERING would mean the
harness's idealisation (fp32 o_hat, fp16 weights) hides something -- chase that before believing
either number.

Warm-up discipline is dynamic_delta_ab's and is not optional: the quantized attention blocks
self-calibrate over their first MODIFF_ATTN_CALIB_STEPS forwards, so run 1 after model construction
is several x worse than run 2 (0.2107 vs 0.0399). Every arm discards a run.

Run: python docs/qdiff_bridge_2026-08-12/scripts/qdiff_ab.py [--steps 50] [--seeds 3]
"""
import argparse
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

SHIPPED = "integration/calibration/int8_calibration_realckpt.pt"
QDIFF = "integration/calibration/int8_calibration_qdiff.pt"
OUT = "docs/qdiff_bridge_2026-08-12/data/qdiff_ab.json"

#: (label, mode, calibration file). "int8" is MoDiff on, "int8_baseline" is MoDiff off.
ARMS = [("baseline / shipped absmax", "int8_baseline", SHIPPED),
        ("baseline / qdiff", "int8_baseline", QDIFF),
        ("MoDiff / shipped absmax", "int8", SHIPPED),
        ("MoDiff / qdiff", "int8", QDIFF)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    H.STEPS, H.BATCH = a.steps, 8
    seeds = [1234, 20260805, 777][:a.seeds]

    for p in (SHIPPED, QDIFF):
        if not os.path.exists(p):
            print(f"FAIL: {p} missing")
            return 1

    results = {}
    for label, mode, calib in ARMS:
        H.CALIB["int8"] = calib
        # delta_mode: MoDiff keeps its DYNAMIC delta path. A7 measured every static delta table
        # losing to it by 3.5-4.6x, so this A/B varies ONLY the activation file.
        delta_mode = "dynamic" if mode == "int8" else "static"
        r, m, s = H.build(mode, calib, delta_mode)
        per_seed = []
        for seed in seeds:
            H.SEED = seed
            H.latent(r, m, s)                                  # warm-up, discarded
            lat, ms = H.latent(r, m, s)
            # fp16 reference for this seed, built once per seed below
            per_seed.append((seed, lat.float(), ms))
        results[label] = {"mode": mode, "calib": calib, "lat": per_seed}
        del r, m, s
        torch.cuda.empty_cache()
        print(f"  ran {label}", flush=True)

    print("  building fp16 reference ...", flush=True)
    rf, mf, sf = H.build("fp16", None, "static")
    refs = {}
    for seed in seeds:
        H.SEED = seed
        H.latent(rf, mf, sf)
        lat, _ = H.latent(rf, mf, sf)
        refs[seed] = lat.float()
    del rf, mf, sf
    torch.cuda.empty_cache()

    summary = {}
    print(f"\n{'arm':30s} {'relL2 mean':>11} {'per seed':>34} {'ms/step':>9}")
    for label, r in results.items():
        rels = [float((lat - refs[seed]).norm() / refs[seed].norm()) for seed, lat, _ in r["lat"]]
        ms = statistics.median(m for _, _, m in r["lat"])
        summary[label] = {"mode": r["mode"], "calib": r["calib"], "relL2": rels,
                          "relL2_mean": statistics.mean(rels), "ms_per_step": ms}
        print(f"{label:30s} {statistics.mean(rels):11.4f} "
              f"{str([round(x, 4) for x in rels]):>34} {ms:9.2f}")

    b_sh = summary["baseline / shipped absmax"]["relL2_mean"]
    b_qd = summary["baseline / qdiff"]["relL2_mean"]
    m_sh = summary["MoDiff / shipped absmax"]["relL2_mean"]
    m_qd = summary["MoDiff / qdiff"]["relL2_mean"]
    print(f"\n  baseline: {b_sh:.4f} -> {b_qd:.4f}   ({b_sh / max(b_qd, 1e-9):.2f}x)")
    print(f"  MoDiff  : {m_sh:.4f} -> {m_qd:.4f}   ({m_sh / max(m_qd, 1e-9):.2f}x)")
    print("\n  A7 fake-quant predicted: baseline 2.36x better, MoDiff a wash.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"steps": a.steps, "batch": H.BATCH, "seeds": seeds, "summary": summary,
               "a7_prediction": {"baseline_ratio": 2.36, "modiff": "wash"}},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
