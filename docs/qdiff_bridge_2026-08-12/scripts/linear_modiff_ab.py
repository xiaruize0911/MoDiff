"""Stage D: re-make the MODIFF_LINEAR quality comparison on the CORRECTED model.

The speed half is settled and was measured elsewhere with the right instrument (profiler-free
differential timing, 200 steps x 5 repeats, docs/current_state_2026-08-12): conv-only reaches
1.371x fp16, conv+proj only 1.059x. Stage B added that the int8 projection GEMM path is a net LOSS
at these shapes anyway (0.86x vs cuBLAS fp16) before MoDiff's delta traffic is counted.

What was NOT settled is quality, and specifically whether the old quality argument survives the
recalibration. The prior evidence (docs/modiff_correctness_2026-08-03/data/linear_modiff_ab.json)
was measured against the absmax scales that Q-Diffusion has now replaced:

    int8  MODIFF_LINEAR=0  0.0413   ->  =1  0.0396   (4% better, 18.8 -> 23.8 ms/step)
    int4  MODIFF_LINEAR=0  0.4804   ->  =1  0.4513   (6% better, 15.6 -> 18.0 ms/step)

If part of what MoDiff-on-Linear was buying was compensation for a bad activation scale, that benefit
should shrink now that the scale is fixed. This measures it.

ms/step here is INDICATIVE ONLY -- arms are built sequentially in one process, so the first pays
cuDNN autotuning the second inherits, and this A40 idles at 210 MHz so short runs bounce between
clock states. Quote the differential-timing numbers for speed, not these.

Run: python docs/qdiff_bridge_2026-08-12/scripts/linear_modiff_ab.py [--steps 50] [--seeds 3]
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

QDIFF = "integration/calibration/int8_calibration_qdiff.pt"
SHIPPED = "integration/calibration/int8_calibration_realckpt.pt"
OUT = "docs/qdiff_bridge_2026-08-12/data/linear_modiff_ab.json"

#: (label, mode, MODIFF_LINEAR, calibration file)
ARMS = [("W8A8 qdiff, MODIFF_LINEAR=0", "int8", "0", QDIFF),
        ("W8A8 qdiff, MODIFF_LINEAR=1", "int8", "1", QDIFF),
        ("W8A8 shipped, MODIFF_LINEAR=0", "int8", "0", SHIPPED),
        ("W8A8 shipped, MODIFF_LINEAR=1", "int8", "1", SHIPPED),
        ("W4A4, MODIFF_LINEAR=0", "int4", "0", None),
        ("W4A4, MODIFF_LINEAR=1", "int4", "1", None)]


def qout_eligible(model):
    """How many attention blocks can use the fused int8-output epilogue.

    benchmark_ldm.py:752-753 records that MoDiff on the Linears sets _out_i8 False and thereby
    disables this on ALL 21 blocks. If that is right, MODIFF_LINEAR=0 should report 21 and =1 should
    report 0 -- i.e. turning the flag off does not merely save delta traffic, it unlocks a fusion.
    """
    n = 0
    for b in model.model.diffusion_model.modules():
        if type(b).__name__ != "QuantizedStandardAttentionBlock":
            continue
        fn = getattr(b, "_qout_eligible", None)
        try:
            if callable(fn) and fn():
                n += 1
        except Exception:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    H.STEPS, H.BATCH = a.steps, 8
    seeds = [1234, 20260805, 777][:a.seeds]

    print("building fp16 references ...", flush=True)
    rf, mf, sf = H.build("fp16", None, "static")
    refs = {}
    for seed in seeds:
        H.SEED = seed
        H.latent(rf, mf, sf)
        refs[seed], _ = H.latent(rf, mf, sf)
        refs[seed] = refs[seed].float()
    del rf, mf, sf
    torch.cuda.empty_cache()

    results = {}
    for label, mode, lin, calib in ARMS:
        os.environ["MODIFF_LINEAR"] = lin
        cal = calib or H.CALIB["int4" if "int4" in mode else "int8"]
        H.CALIB["int8"] = calib if calib else H.CALIB["int8"]
        r, m, s = H.build(mode, cal, "dynamic")
        nq = qout_eligible(m)
        H.SEED = seeds[0]
        H.latent(r, m, s)                                   # discard: attention self-calibration
        nq_after = qout_eligible(m)
        rels, mss = [], []
        for seed in seeds:
            H.SEED = seed
            H.latent(r, m, s)
            lat, ms = H.latent(r, m, s)
            rels.append(float((lat.float() - refs[seed]).norm() / refs[seed].norm()))
            mss.append(ms)
        results[label] = {"mode": mode, "modiff_linear": lin, "calib": cal,
                          "relL2": rels, "relL2_mean": statistics.mean(rels),
                          "ms_per_step_indicative": statistics.median(mss),
                          "qout_eligible_blocks": nq_after}
        print(f"  {label:34s} relL2 {statistics.mean(rels):.4f}  "
              f"qout-eligible {nq_after}/21  ms {statistics.median(mss):.2f}", flush=True)
        del r, m, s
        torch.cuda.empty_cache()
    os.environ.pop("MODIFF_LINEAR", None)

    print(f"\n{'pair':28s} {'OFF':>9} {'ON':>9} {'ON/OFF':>8}  verdict")
    verdicts = {}
    for tag, off, on in (("W8A8 qdiff", "W8A8 qdiff, MODIFF_LINEAR=0", "W8A8 qdiff, MODIFF_LINEAR=1"),
                         ("W8A8 shipped", "W8A8 shipped, MODIFF_LINEAR=0",
                          "W8A8 shipped, MODIFF_LINEAR=1"),
                         ("W4A4", "W4A4, MODIFF_LINEAR=0", "W4A4, MODIFF_LINEAR=1")):
        o, n = results[off]["relL2_mean"], results[on]["relL2_mean"]
        wins = sum(1 for x, y in zip(results[off]["relL2"], results[on]["relL2"]) if y < x)
        verdicts[tag] = {"off": o, "on": n, "ratio": n / o, "on_wins_seeds": wins}
        print(f"{tag:28s} {o:9.4f} {n:9.4f} {n / o:8.3f}  "
              f"ON better on {wins}/{len(seeds)} seeds")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"steps": a.steps, "batch": H.BATCH, "seeds": seeds,
               "results": results, "verdicts": verdicts,
               "note_ms": "ms_per_step_indicative is contaminated by build order and clock ramp; "
                          "use docs/current_state_2026-08-12 differential timing for speed"},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
