"""Sweep the activation clip ratio ON THE REAL KERNELS, both W4A4 axes.

Why not fake quant, which is faster. The harness runs fp16 weights, and once the activation grid is
fine the 4-bit weight error (0.2728 on its own) stops being negligible -- so it reads systematically
optimistic in exactly the regime this sweep explores. Measured on fix #1: the harness predicted
0.1147 and the kernels delivered 0.3099. A 2.7x miss is fine for finding a direction and useless for
choosing a constant.

The ratio is applied by scaling the shipped activation file, which is exactly what
OptimizedInt4Conv2d.ACT_CLIP_RATIO does in end_calibration and what export_qdiff_scales.py does on
the way out -- so the sweep measures the thing the constant will ship.

BOTH AXES, because they use the grid differently. int4_baseline quantizes on it every step; int4
(MoDiff) touches it only at t=T and then refines with 5 warm-up rounds, so it should be far less
sensitive. If MoDiff turns out insensitive, the constant can be chosen for PTQ without costing the
MoDiff arm anything.

Run: python docs/paper_repro_2026-08-12/scripts/act_clip_sweep_real.py    # ~35 min, needs the GPU
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

ACT = "integration/calibration/int4_calibration_qdiff.pt"
TMP = "docs/paper_repro_2026-08-12/data/int4_act_clip_r{}.pt"
OUT = "docs/paper_repro_2026-08-12/data/act_clip_sweep_real.json"
SEEDS = [1234, 20260805, 777]
RATIOS = [1.0, 2.0, 3.0, 4.5, 6.7, 10.0]
#: current shipped numbers at ACT_CLIP_RATIO 1.0, after fix #1 landed
BASE = {"int4_baseline": 0.8643, "int4": 0.3099}


def scaled(r):
    d = torch.load(ACT, map_location="cpu", weights_only=True)
    p = TMP.format(f"{r:g}")
    torch.save({k: float(v) * r for k, v in d.items()}, p)
    return p


def main():
    H.STEPS, H.BATCH = 50, 8
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"

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
    os.environ["MODIFF_DELTA_MODE"] = "static"

    out = {}
    for mode, label in (("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")):
        print(f"\n{label}   (ratio 1.0 shipped = {BASE[mode]:.4f})", flush=True)
        for r in RATIOS:
            cal = scaled(r)
            rr, m, s = H.build(mode, cal, "static" if "baseline" in mode else "static")
            H.SEED = SEEDS[0]
            H.latent(rr, m, s)                      # discard: attention self-calibration
            rels = []
            for sd in SEEDS:
                H.SEED = sd
                H.latent(rr, m, s)
                lat, _ = H.latent(rr, m, s)
                rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
            out[f"{mode}/r{r:g}"] = {"mean": statistics.mean(rels), "relL2": rels, "ratio": r}
            print(f"  ratio {r:<5g} {statistics.mean(rels):.4f}   {[round(x, 3) for x in rels]}",
                  flush=True)
            del rr, m, s
            torch.cuda.empty_cache()

    print(f"\n{'ratio':>7}{'W4A4 PTQ':>12}{'W4A4 MoDiff':>14}")
    for r in RATIOS:
        print(f"{r:>7g}{out[f'int4_baseline/r{r:g}']['mean']:>12.4f}"
              f"{out[f'int4/r{r:g}']['mean']:>14.4f}")
    bp = min((out[f"int4_baseline/r{r:g}"]["mean"], r) for r in RATIOS)
    bm = min((out[f"int4/r{r:g}"]["mean"], r) for r in RATIOS)
    print(f"\nPTQ best   ratio {bp[1]:g} at {bp[0]:.4f}  ({bp[0] / BASE['int4_baseline']:.2f}x shipped)")
    print(f"MoDiff best ratio {bm[1]:g} at {bm[0]:.4f}  ({bm[0] / BASE['int4']:.2f}x shipped)")
    spread = max(out[f"int4/r{r:g}"]["mean"] for r in RATIOS) / bm[0]
    print(f"MoDiff spread across the sweep: {spread:.2f}x"
          + ("  -- insensitive, so pick the ratio for PTQ" if spread < 1.3 else
             "  -- sensitive, the two axes want different ratios"))
    json.dump({"seeds": SEEDS, "ratios": RATIOS, "base": BASE, "results": out},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
