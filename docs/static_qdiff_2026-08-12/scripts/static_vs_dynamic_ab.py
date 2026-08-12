"""What switching to static Q-Diffusion costs, measured on every axis it touches.

The tree now follows README:96 -- `--modulate --quant_mode qdiff --cali_min_max` -- which makes both
the activation scale and the per-step delta table static and Q-Diffusion-derived. That is a paper
fidelity decision, not a quality one, and this is the bill.

THREE THINGS CHANGED AT ONCE, so all three are measured separately rather than as one number:

  1. MODIFF_DELTA_MODE default dynamic -> static, for the modulated steps t<T. Only the MoDiff
     modes have modulated steps, so this cannot touch the PTQ baselines.
  2. The static delta table is now actually LOADED. apply_int*_delta_scales had zero call sites, so
     every previous "static" measurement in this tree ran on an uncalibrated grid -- at int4 it did
     not even have a table to miss, it quantized the delta on the full activation grid. So the
     honest comparison is not against the old static numbers; it has to be re-run.
  3. int4's activation default moved to the qdiff file. FINDINGS §5a graded that file at 1.1945 on
     the PTQ axis against the shipped 0.7119, so the W4A4 PTQ arm is expected to REGRESS hard. It
     is included because the paper's qdiff path calibrates the baseline the same way, and a cost
     that is only visible when you go looking is not a cost anyone will find.

Protocol as every other W4A4 A/B in this tree: real LSUN-churches checkpoint, DDIM S=50, batch 8,
seeds {1234, 20260805, 777}, latent relL2 against a per-seed fp16 reference, first run per arm
discarded (the quantized attention self-calibrates on it), all arms in one process.

Run: python docs/static_qdiff_2026-08-12/scripts/static_vs_dynamic_ab.py    # ~15 min, needs the GPU
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

OUT = "docs/static_qdiff_2026-08-12/data/static_vs_dynamic_ab.json"
SEEDS = [1234, 20260805, 777]
#: committed references, all same protocol. The MoDiff/dynamic rows are what the tree shipped
#: before this change; the PTQ rows are from docs/qdiff_bridge_2026-08-12/data/w4a4_ab.json.
PRIOR = {"int8/dynamic": 0.0393, "int4/dynamic": 0.4199,
         "int8_baseline/-": 0.2564, "int4_baseline/-": 1.1945}


def measure(mode, delta_mode, refs):
    """One arm. The calibration path is left to resolve from CALIBRATION_PREFERENCE, and the delta
    table from DELTA_CALIBRATION_PREFERENCE inside _setup_model, so what is graded is the DEFAULT."""
    os.environ["MODIFF_DELTA_MODE"] = delta_mode
    cal = B._default_calibration_path(mode)
    r, m, s = H.build(mode, cal, delta_mode)
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
    return rels, cal


def main():
    for bits in ("int8", "int4"):
        p = B._default_delta_path(bits)
        if not p or "qdiff" not in p:
            print(f"FAIL: {bits} delta default resolves to {p}, not a qdiff table.\n"
                  f"      run docs/static_qdiff_2026-08-12/scripts/install_qdiff_defaults.py")
            return 1
        print(f"gate  {bits} delta table -> {p}")

    H.STEPS, H.BATCH = 50, 8
    #: dynamic_delta_ab defaults this OFF so its own "static, table off" arm keeps meaning that.
    #: Here the shipped default IS the thing under test, so it goes back on.
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"

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

    out = {}
    #: (mode, delta_mode, label). The PTQ baselines have no modulated steps, so they get one arm.
    ARMS = [("int8_baseline", "static", "W8A8 PTQ"), ("int8", "static", "W8A8 MoDiff static"),
            ("int8", "dynamic", "W8A8 MoDiff dynamic"),
            ("int4_baseline", "static", "W4A4 PTQ"), ("int4", "static", "W4A4 MoDiff static"),
            ("int4", "dynamic", "W4A4 MoDiff dynamic")]
    print()
    for mode, dm, label in ARMS:
        rels, cal = measure(mode, dm, refs)
        key = f"{mode}/{dm if 'baseline' not in mode else '-'}"
        out[key] = {"label": label, "mean": statistics.mean(rels), "relL2": rels,
                    "calibration": cal, "delta_mode": dm}
        p = PRIOR.get(key)
        print(f"  {label:22s} {statistics.mean(rels):.4f}  {[round(x, 4) for x in rels]}"
              f"{f'   (prior {p:.4f})' if p else ''}", flush=True)

    print(f"\n{'axis':16s} {'static (shipped)':>17} {'dynamic':>10} {'cost':>9}")
    verdict = {}
    for bits, name in (("int8", "W8A8"), ("int4", "W4A4")):
        st, dy = out[f"{bits}/static"]["mean"], out[f"{bits}/dynamic"]["mean"]
        verdict[bits] = {"static": st, "dynamic": dy, "ratio": st / dy}
        print(f"{name + ' MoDiff':16s} {st:17.4f} {dy:10.4f} {st / dy:8.2f}x")
    print("\nPTQ baselines are unaffected by the delta mode (no modulated steps); their number moves "
          "only\nwith the activation file, which at int4 is now the qdiff one:")
    for bits, name in (("int8", "W8A8"), ("int4", "W4A4")):
        k = f"{bits}_baseline/-"
        print(f"  {name} PTQ  {out[k]['mean']:.4f}   {os.path.basename(out[k]['calibration'])}")

    json.dump({"seeds": SEEDS, "steps": H.STEPS, "batch": H.BATCH, "prior": PRIOR,
               "results": out, "verdict": verdict}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
