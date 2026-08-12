"""Does HEADROOM rescue the per-step W4A4 delta table? If so, the int4 calibration path has a bug.

THE CHAIN SO FAR.
  debug_w4a4_delta_scale.py   the shipped constant table is not under-sized -- it OVER-sizes the
                              observed delta by ~2x and clips on 0/70 layers.
  fix_w4a4_delta_perstep.py   a per-step table built from that same observation reads 18.7152,
                              against the constant's 1.0502. Not "per-step does not help" -- broken.

WHY IT IS BROKEN. The delta is a FIXED POINT: `a_t - a_hat_{t+1}` depends on the scale that built
`a_hat`. The two paths observe different trajectories:

  int8_optimized.begin_delta_calibration   seeds static_delta_scale = act_scale/4 -- provably
                                           non-clipping since |a_t - a_hat| <= 2*act_absmax -- and
                                           observes while running STATIC. Same trajectory it will
                                           be deployed on, so one pass is exact.
  int4_optimized.begin_delta_calibration   sets `delta_dynamic = True` and observes the DYNAMIC
                                           trajectory, where a_hat is built per call and the deltas
                                           are small (measured: 0.52x the constant table's range).

Deploy the second table in static mode and step 1's a_hat is worse than the one observed, so step
2's delta is larger than the table allows, so it clips, so step 3's is larger still. 18.7 is that
runaway. The constant table escapes it only because its accidental 2x oversizing IS headroom.

THE TEST. Rescale the per-step table by a headroom factor h -- scale/h, i.e. assume h times the
observed range. No new observation needed, so this is nearly free. If the error falls monotonically
with h and passes the constant, the fixed-point diagnosis holds and the fix is on the int4
calibration path (observe statically like int8, or iterate a second round), not on the export.

Run: python docs/state_report_2026-08-12/scripts/w4a4_delta_headroom_ab.py   # ~14 min, needs GPU
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

PERSTEP = "docs/state_report_2026-08-12/data/int4_delta_perstep_qdiffcfg.pt"
SHIPPED = "integration/calibration/int4_delta_qdiff.pt"
TMP = "docs/state_report_2026-08-12/data/int4_delta_perstep_h{}.pt"
OUT = "docs/state_report_2026-08-12/data/w4a4_delta_headroom_ab.json"
SEEDS = [1234, 20260805, 777]
HEADROOM = [2.0, 4.0, 8.0]
PTQ = 0.8642        # static_qdiff/data/static_vs_dynamic_ab.json, same protocol


def headroom_table(h):
    """scale/h == assume h times the observed absmax. Pure arithmetic on the saved table."""
    d = torch.load(PERSTEP, map_location="cpu", weights_only=True)
    out = {k: (v / h) for k, v in d.items()}
    p = TMP.format(h)
    torch.save(out, p)
    return p


def measure(table, delta_mode, refs, label):
    os.environ["MODIFF_DELTA_MODE"] = delta_mode
    H.AUTO_DELTA_TABLE = False
    r, m, s = H.build("int4", B._default_calibration_path("int4"), delta_mode)
    if table:
        from integration.kernels.int4_optimized import apply_int4_delta_scales
        n = apply_int4_delta_scales(m.model.diffusion_model,
                                    torch.load(table, map_location="cpu", weights_only=True))
        if n != 70:
            print(f"  FAIL: {table} matched {n} layers, expected 70")
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
    print(f"  {label:36s} {statistics.mean(rels):8.4f}  {[round(x, 3) for x in rels]}", flush=True)
    return rels


def main():
    if not os.path.exists(PERSTEP):
        print(f"FAIL: missing {PERSTEP} -- run fix_w4a4_delta_perstep.py first")
        return 1
    os.environ["MODIFF_LINEAR"] = "0"
    H.STEPS, H.BATCH = 50, 8

    print("fp16 references ...", flush=True)
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
    rels = measure(SHIPPED, "static", refs, "static, qdiff constant (shipped)")
    if rels is None:
        return 1
    out["constant"] = {"mean": statistics.mean(rels), "relL2": rels}
    for h in HEADROOM:
        rels = measure(headroom_table(h), "static", refs, f"static, per-step x{h:g} headroom")
        if rels is None:
            return 1
        out[f"perstep_h{h:g}"] = {"mean": statistics.mean(rels), "relL2": rels, "headroom": h}
    rels = measure(None, "dynamic", refs, "dynamic (the bar)")
    out["dynamic"] = {"mean": statistics.mean(rels), "relL2": rels}

    print(f"\n{'arm':36s}{'relL2':>9}")
    print(f"{'W4A4 PTQ baseline':36s}{PTQ:9.4f}")
    print(f"{'static, qdiff constant (shipped)':36s}{out['constant']['mean']:9.4f}")
    print(f"{'static, per-step x1 (no headroom)':36s}{18.7152:9.4f}   (fix_w4a4_delta_perstep.py)")
    for h in HEADROOM:
        print(f"{'static, per-step x' + f'{h:g}' + ' headroom':36s}{out[f'perstep_h{h:g}']['mean']:9.4f}")
    print(f"{'dynamic':36s}{out['dynamic']['mean']:9.4f}")

    best = min((out[f"perstep_h{h:g}"]["mean"], h) for h in HEADROOM)
    print()
    if best[0] < out["constant"]["mean"]:
        print(f"FIXED-POINT DIAGNOSIS HOLDS. Headroom alone takes the per-step table from 18.72 to "
              f"{best[0]:.4f} at x{best[1]:g}, past the shipped constant's "
              f"{out['constant']['mean']:.4f}" +
              (f" and past the PTQ baseline's {PTQ:.4f}" if best[0] < PTQ else "") +
              f". The defect is that int4's begin_delta_calibration observes the DYNAMIC trajectory "
              f"and deploys the table on the STATIC one; int8 observes statically at act_scale/4 and "
              f"does not have this problem.")
    else:
        print(f"HEADROOM IS NOT ENOUGH: best per-step {best[0]:.4f} at x{best[1]:g}, still above the "
              f"constant's {out['constant']['mean']:.4f}. The per-step table's SHAPE, not just its "
              f"scale, is wrong for the static trajectory.")

    json.dump({"seeds": SEEDS, "ptq": PTQ, "perstep_h1": 18.7152, "results": out},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
