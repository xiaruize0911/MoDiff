"""Build the W4A4 delta table PER STEP in the shipped configuration, and measure whether it fixes it.

WHAT THE DEBUG FOUND. debug_w4a4_delta_scale.py refuted the obvious suspicion -- the shipped table is
not under-sized, it over-sizes the delta by ~2x and clips on 0/70 layers. What it is, is CONSTANT:
qdiff reports one `act_quantizer.delta` per layer and export_qdiff_scales fills all 256 step slots
with it. Measured against the delta the deployed model actually produces:

    delta absmax across steps, max/median   1.77x   (per layer, median over 70 layers)
    the median step therefore uses            52%   of the assumed range
    wasted resolution                       ~0.95   bit

At W8A8 that is 1.15 bit out of eight and does not matter -- static already beats dynamic there. At
W4A4 it is ~1 bit out of FOUR, and MoDiff's feedback term accumulates the resulting error over 50
modulated steps, which is a mechanism for ending up worse than not modulating at all. W4A4 MoDiff
measures 1.0469 against its own PTQ baseline's 0.8642, and its samples are accumulating structured
scribble rather than the PTQ arm's fog.

THE TEST. The tree can observe the exact per-step delta absmax (`begin_delta_calibration_int4` forces
dynamic for one pass and records what the kernel computes) and turn it into a per-step table
(`end_delta_calibration_int4`). Doing that HERE, in the shipped configuration -- qdiff activation
file, SmoothQuant off -- also settles the question static_qdiff FINDINGS §4 had to leave open,
where the only per-step table available had been calibrated with SmoothQuant on and was confounded.

Three arms, one process, shared fp16 reference:

    qdiff constant   the shipped table                        expect ~1.05
    per-step         built here, same configuration           the question
    dynamic          MODIFF_DELTA_MODE=dynamic                 expect ~0.36, the bar

Run: python docs/state_report_2026-08-12/scripts/fix_w4a4_delta_perstep.py   # ~10 min, needs the GPU
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

TABLE_OUT = "docs/state_report_2026-08-12/data/int4_delta_perstep_qdiffcfg.pt"
OUT = "docs/state_report_2026-08-12/data/fix_w4a4_delta_perstep.json"
SEEDS = [1234, 20260805, 777]
SHIPPED_TABLE = "integration/calibration/int4_delta_qdiff.pt"


def build_perstep_table(steps=50, batch=8):
    """Observe the exact per-step delta absmax in the shipped config, then emit the table."""
    from integration.kernels.int4_optimized import (begin_delta_calibration_int4,
                                                    end_delta_calibration_int4,
                                                    export_int4_delta_scales)
    H.STEPS, H.BATCH, H.SEED = steps, batch, SEEDS[0]
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_DELTA_MODE"] = "static"
    r, m, s = H.build("int4", B._default_calibration_path("int4"), "static")
    H.latent(r, m, s)                       # warm-up: let quantized attention self-calibrate
    n = begin_delta_calibration_int4(m.model.diffusion_model, reset=True)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                       # observation pass (forced dynamic by the arming)
    got = end_delta_calibration_int4(m.model.diffusion_model)
    table = export_int4_delta_scales(m.model.diffusion_model)
    print(f"  armed {n}, calibrated {got}, exported {len(table)} layers", flush=True)
    if len(table) < 70:
        print(f"  FAIL: only {len(table)} layers exported, expected 70")
        return None
    torch.save(table, TABLE_OUT)
    sp = statistics.median(float(v.max() / v.min().clamp_min(1e-12)) for v in table.values())
    print(f"  wrote {TABLE_OUT}   per-step spread (max/min) {sp:.2f}x  "
          f"against the shipped table's 1.00x", flush=True)
    del r, m, s
    torch.cuda.empty_cache()
    return table


def measure(delta_mode, table, refs, label):
    os.environ["MODIFF_DELTA_MODE"] = delta_mode
    H.AUTO_DELTA_TABLE = (table is None and delta_mode == "static")
    r, m, s = H.build("int4", B._default_calibration_path("int4"), delta_mode)
    if table is not None:
        from integration.kernels.int4_optimized import apply_int4_delta_scales
        n = apply_int4_delta_scales(m.model.diffusion_model,
                                    torch.load(table, map_location="cpu", weights_only=True))
        if n == 0:
            print(f"  FAIL: {table} matched 0 layers")
            return None
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                       # discard: attention self-calibration
    rels = []
    for sd in SEEDS:
        H.SEED = sd
        H.latent(r, m, s)
        lat, _ = H.latent(r, m, s)
        rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
    del r, m, s
    torch.cuda.empty_cache()
    print(f"  {label:34s} {statistics.mean(rels):.4f}  {[round(x, 4) for x in rels]}", flush=True)
    return rels


def main():
    os.environ["MODIFF_LINEAR"] = "0"
    print("building the per-step table in the shipped configuration ...", flush=True)
    if build_perstep_table() is None:
        return 1

    H.STEPS, H.BATCH = 50, 8
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

    print()
    out = {}
    for key, dm, tbl, label in (
            ("qdiff_constant", "static", SHIPPED_TABLE, "static, qdiff constant (shipped)"),
            ("perstep", "static", TABLE_OUT, "static, per-step (built here)"),
            ("dynamic", "dynamic", None, "dynamic (the bar)")):
        rels = measure(dm, tbl, refs, label)
        if rels is None:
            return 1
        out[key] = {"mean": statistics.mean(rels), "relL2": rels, "delta_mode": dm, "table": tbl}

    c, p, d = out["qdiff_constant"]["mean"], out["perstep"]["mean"], out["dynamic"]["mean"]
    ptq = 0.8642      # static_qdiff/data/static_vs_dynamic_ab.json, same protocol
    print(f"\n{'arm':34s}{'relL2':>9}")
    print(f"{'W4A4 PTQ baseline (reference)':34s}{ptq:9.4f}")
    for k, lab in (("qdiff_constant", "static, qdiff constant (shipped)"),
                   ("perstep", "static, per-step (built here)"), ("dynamic", "dynamic")):
        print(f"{lab:34s}{out[k]['mean']:9.4f}")
    print()
    if p < c * 0.8:
        print(f"THE CONSTANCY WAS THE PROBLEM: a per-step table takes W4A4 MoDiff "
              f"{c:.4f} -> {p:.4f}" + (f", and it now beats its PTQ baseline ({ptq:.4f})."
                                       if p < ptq else f", though still above PTQ's {ptq:.4f}.") +
              f" Dynamic remains ahead at {d:.4f}. export_qdiff_scales fills all 256 slots with "
              f"qdiff's single per-layer delta, so the paper's static path cannot express this.")
    else:
        print(f"NOT THE CONSTANCY: per-step {p:.4f} against the constant's {c:.4f}. Both are far "
              f"from dynamic's {d:.4f}, so what a 4-bit static delta cannot do is track the "
              f"trajectory at all, and no better table exists to export.")

    json.dump({"seeds": SEEDS, "ptq_reference": ptq, "results": out}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
