"""Is W4A4 static bad because it is STATIC, or because qdiff's table is a single CONSTANT?

static_vs_dynamic_ab.py measured W4A4 MoDiff static at 1.0469 against dynamic's 0.3577 -- and worse
than the W4A4 PTQ baseline's 0.8642, i.e. MoDiff actively hurting. Before writing that up as "static
Q-Diffusion costs 2.93x at 4 bits", it is worth separating two things that both changed:

  STATIC vs DYNAMIC   the step size is fixed at calibration instead of computed per call
  CONSTANT vs PER-STEP  qdiff reports ONE act_quantizer.delta per layer -- export_qdiff_scales
                      refuses a per-channel delta and fills all 256 table slots with that single
                      scalar. The tree's own end_delta_calibration instead observes the delta absmax
                      at each step index and builds a genuinely per-step table. The MoDiff residual
                      shrinks as t decreases, so a constant sized for the largest step leaves the
                      later steps on a grid far too coarse -- with 15 levels that is fatal, with 255
                      it may not be.

Third arm: static mode with the tree's NATIVE per-step table (int4_delta_calibration.pt), which is
static but not constant. If it lands near dynamic, the cost is the constancy and a per-step qdiff
export would fix it. If it lands near qdiff's 1.0469, the cost is staticness itself and there is
nothing to fix.

Protocol identical to static_vs_dynamic_ab.py, same container, one process, shared fp16 reference.

Run: python docs/static_qdiff_2026-08-12/scripts/constant_vs_perstep_ab.py   # ~8 min, needs the GPU
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
import integration.benchmarks.benchmark_ldm as B                           # noqa: E402

OUT = "docs/static_qdiff_2026-08-12/data/constant_vs_perstep_ab.json"
SEEDS = [1234, 20260805, 777]
#: static_vs_dynamic_ab.json, same container and protocol
PRIOR = {"int4": {"qdiff_constant": 1.0469, "dynamic": 0.3577, "ptq": 0.8642},
         "int8": {"qdiff_constant": 0.0520, "dynamic": 0.0611, "ptq": 0.1138}}
ARMS = [("int4", "integration/calibration/int4_delta_qdiff.pt", "qdiff constant"),
        ("int4", "integration/calibration/int4_delta_calibration.pt", "native per-step"),
        ("int8", "integration/calibration/int8_delta_qdiff.pt", "qdiff constant"),
        ("int8", "integration/calibration/int8_delta_calibration.pt", "native per-step")]


def spread(table):
    """max/min of a layer's table across steps -- 1.0 means constant, which is the whole question."""
    d = torch.load(table, map_location="cpu", weights_only=True)
    r = [float(v.max() / v.min().clamp_min(1e-12)) for v in d.values()]
    return statistics.median(r)


def main():
    H.STEPS, H.BATCH = 50, 8
    H.AUTO_DELTA_TABLE = False          # this script passes the table explicitly, per arm
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"

    for _, t, label in ARMS:
        print(f"  {label:16s} {os.path.basename(t):34s} per-step spread (max/min) {spread(t):.3f}")

    print("\nfp16 references ...", flush=True)
    os.environ["MODIFF_DELTA_MODE"] = "dynamic"
    rf, mf, sf = H.build("fp16", None, "static")
    refs = {}
    for s in SEEDS:
        H.SEED = s
        H.latent(rf, mf, sf)
        refs[s] = H.latent(rf, mf, sf)[0].float()
    del rf, mf, sf
    torch.cuda.empty_cache()
    os.environ["MODIFF_DELTA_MODE"] = "static"

    out = {}
    print()
    for mode, table, label in ARMS:
        cal = B._default_calibration_path(mode)
        r, m, s = H.build(mode, cal, "static")
        from integration.kernels.int4_optimized import apply_int4_delta_scales
        from integration.kernels.int8_optimized import apply_int8_delta_scales
        apply = apply_int4_delta_scales if mode == "int4" else apply_int8_delta_scales
        n = apply(m.model.diffusion_model, torch.load(table, map_location="cpu", weights_only=True))
        if n == 0:
            print(f"FAIL: {table} matched 0 layers in {mode}")
            return 1
        H.SEED = SEEDS[0]
        H.latent(r, m, s)                                # discard: attention self-calibration
        rels = []
        for sd in SEEDS:
            H.SEED = sd
            H.latent(r, m, s)
            lat, _ = H.latent(r, m, s)
            rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
        key = f"{mode}/{label.replace(' ', '_')}"
        out[key] = {"mean": statistics.mean(rels), "relL2": rels, "table": table, "layers": n}
        bits = "W4A4" if mode == "int4" else "W8A8"
        print(f"  {bits} MoDiff static, {label:16s} {statistics.mean(rels):.4f}  "
              f"{[round(x, 4) for x in rels]}   (table on {n} layers)", flush=True)
        del r, m, s
        torch.cuda.empty_cache()

    print(f"\n{'axis':6s} {'qdiff constant':>15} {'native per-step':>16} {'dynamic':>9} {'PTQ':>9}")
    verdict = {}
    for mode, bits in (("int8", "W8A8"), ("int4", "W4A4")):
        c = out[f"{mode}/qdiff_constant"]["mean"]
        p = out[f"{mode}/native_per-step"]["mean"]
        dy, ptq = PRIOR[mode]["dynamic"], PRIOR[mode]["ptq"]
        verdict[mode] = {"qdiff_constant": c, "native_per_step": p, "dynamic": dy, "ptq": ptq,
                         "perstep_over_constant": p / c, "perstep_over_dynamic": p / dy}
        print(f"{bits:6s} {c:15.4f} {p:16.4f} {dy:9.4f} {ptq:9.4f}")

    v = verdict["int4"]
    print()
    if v["native_per_step"] < v["qdiff_constant"] * 0.7:
        print(f"THE CONSTANCY IS THE COST at W4A4: a per-step table takes "
              f"{v['qdiff_constant']:.4f} -> {v['native_per_step']:.4f}. qdiff's one-scalar-per-layer "
              f"delta is what 15 levels cannot carry, not staticness -- a per-step qdiff export "
              f"would be worth having.")
    elif v["native_per_step"] > v["qdiff_constant"] * 2:
        # Do NOT read this as "staticness is the cost". The native tables were calibrated against a
        # DIFFERENT activation configuration: int4_delta_table.py builds int4_delta_calibration.pt
        # with dynamic_delta_ab's CALIB, i.e. the shipped absmax file with SmoothQuant ON, and this
        # script applies it on top of the qdiff file, which has smoothing OFF. MoDiff's a_hat cache
        # holds the SMOOTHED activation, so the delta distribution the table was fitted to is not
        # the one it now sees. The arm is confounded and answers nothing.
        print(f"ARM CONFOUNDED at W4A4, not a result: the native per-step table reads "
              f"{v['native_per_step']:.4f}, {v['native_per_step'] / v['qdiff_constant']:.1f}x WORSE "
              f"than the constant. It was calibrated with SmoothQuant on (via CALIB = the shipped "
              f"absmax file) and applied here with it off, so the delta domain does not match.\n"
              f"  To settle constancy-vs-staticness, rebuild the per-step table IN THIS "
              f"configuration:\n"
              f"    AB_CALIB4=integration/calibration/int4_calibration_qdiff.pt \\\n"
              f"      python docs/modiff_correctness_2026-08-03/scripts/int4_delta_table.py\n"
              f"  then re-run this script. Until then W4A4's 2.93x is measured but undecomposed.")
    else:
        print(f"NO SEPARATION at W4A4: per-step {v['native_per_step']:.4f} against constant "
              f"{v['qdiff_constant']:.4f}, both far from dynamic's {v['dynamic']:.4f}. Staticness "
              f"itself is the cost and there is no better static table to export.")

    json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH, "prior": PRIOR,
               "results": out, "verdict": verdict}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
