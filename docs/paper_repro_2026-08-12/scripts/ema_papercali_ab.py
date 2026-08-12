"""Does aligning EMA + the paper's calibration set close any of the remaining W4A4 gap?

The last two alignment items from the plan's fix #6, both landed as opt-in flags and neither measured
until now:

  MODIFF_USE_EMA=1   the reference calibrates AND samples on the EMA network; this loader never did,
                     which is why every calibration run in this tree passed --no_ema. The two
                     networks differ: 0/70 conv weights match, worst 13.8% relative L2.
  CALI_PAPER=1       cali_data/church.pt, the set the README points at, instead of the locally
                     generated residual file (which is an fp16 trajectory because --generate residual
                     exits before `if opt.ptq:`).

EACH ARM NEEDS ITS OWN fp16 REFERENCE. relL2 is measured against fp16, so scoring an EMA-deployed
model against a NON-EMA reference folds the 13.8% weight difference into the number and reports it as
quantization error. That would make EMA look bad for a reason that has nothing to do with
quantization. Two reference sets, one per network, and each arm is graded against its own.

Both arms use the same clip ratios (DELTA_CLIP_RATIO 8, ACT_CLIP_RATIO 4.5), so the only variables
are the network and the calibration set.

Run: python docs/paper_repro_2026-08-12/scripts/ema_papercali_ab.py    # ~20 min, needs the GPU
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

D = "docs/paper_repro_2026-08-12/data"
OUT = f"{D}/ema_papercali_ab.json"
SEEDS = [1234, 20260805, 777]
#: shipped, measured on the same protocol (docs/static_qdiff_2026-08-12/data/static_vs_dynamic_ab.json)
SHIPPED = {"int4_baseline": 0.4695, "int4": 0.3090}


def refs_for(ema):
    os.environ["MODIFF_USE_EMA"] = "1" if ema else "0"
    os.environ["MODIFF_DELTA_MODE"] = "dynamic"
    r, m, s = H.build("fp16", None, "static")
    out = {}
    for sd in SEEDS:
        H.SEED = sd
        H.latent(r, m, s)
        out[sd] = H.latent(r, m, s)[0].float()
    del r, m, s
    torch.cuda.empty_cache()
    os.environ["MODIFF_DELTA_MODE"] = "static"
    return out


def measure(mode, act, delta, refs, ema, label):
    os.environ["MODIFF_USE_EMA"] = "1" if ema else "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"
    H.AUTO_DELTA_TABLE = delta is None
    r, m, s = H.build(mode, act, "static")
    if delta is not None:
        from integration.kernels.int4_optimized import apply_int4_delta_scales
        n = apply_int4_delta_scales(m.model.diffusion_model,
                                   torch.load(delta, map_location="cpu", weights_only=True))
        if n != 70:
            print(f"  FAIL: delta table matched {n}/70")
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
    print(f"  {label:44s} {statistics.mean(rels):.4f}   {[round(x, 3) for x in rels]}", flush=True)
    return statistics.mean(rels), rels


def main():
    H.STEPS, H.BATCH = 50, 8
    os.environ["MODIFF_LINEAR"] = "0"
    for p in (f"{D}/ema_act.pt", f"{D}/ema_delta.pt"):
        if not os.path.exists(p):
            print(f"FAIL: missing {p}")
            return 1

    out = {}
    print("fp16 reference, NON-EMA network ...", flush=True)
    ref_plain = refs_for(False)
    print("fp16 reference, EMA network ...", flush=True)
    ref_ema = refs_for(True)

    print("\nshipped: non-EMA network, non-EMA calibration, local cali data", flush=True)
    for mode, lab in (("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")):
        res = measure(mode, B._default_calibration_path(mode), None, ref_plain, False,
                      f"{lab}  shipped")
        if res is None:
            return 1
        out[f"{mode}/shipped"] = {"mean": res[0], "relL2": res[1]}

    print("\naligned: EMA network, EMA calibration, the paper's cali set", flush=True)
    for mode, lab in (("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")):
        res = measure(mode, f"{D}/ema_act.pt", f"{D}/ema_delta.pt", ref_ema, True,
                      f"{lab}  EMA + paper cali")
        if res is None:
            return 1
        out[f"{mode}/ema_paper"] = {"mean": res[0], "relL2": res[1]}
    os.environ.pop("MODIFF_USE_EMA", None)

    print(f"\n{'axis':16s}{'shipped':>10}{'EMA+paper':>12}{'change':>10}")
    verdict = {}
    for mode, lab in (("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")):
        a, b = out[f"{mode}/shipped"]["mean"], out[f"{mode}/ema_paper"]["mean"]
        verdict[mode] = {"shipped": a, "ema_paper": b, "ratio": b / a}
        print(f"{lab:16s}{a:10.4f}{b:12.4f}{(b / a - 1) * 100:9.1f}%")
    print()
    better = [m for m in verdict if verdict[m]["ratio"] < 0.97]
    worse = [m for m in verdict if verdict[m]["ratio"] > 1.03]
    if better and not worse:
        print("ALIGNING HELPS on both axes -- promote MODIFF_USE_EMA/CALI_PAPER from opt-in to "
              "default, with the calibration files regenerated on the EMA network.")
    elif worse and not better:
        print("ALIGNING HURTS. The remaining gap to the paper is NOT EMA or the calibration set, so "
              "it sits in the activation zero point and the AdaRound weights. Keep both flags opt-in.")
    else:
        print("MIXED or WITHIN NOISE (W4A4 floor is 0.6%): not a lever worth promoting. The remaining "
              "gap sits in the activation zero point and the AdaRound weights.")

    json.dump({"seeds": SEEDS, "shipped_reference": SHIPPED, "results": out, "verdict": verdict},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
