"""A/A: two measure() implementations, one process, one reference set, the same arm.

arm_position_effect.py established that int4/static is DETERMINISTIC -- measured at position 1 and
position 4 of one process it returned [0.3266, 0.2882, 0.3134] both times, and that matches the
committed 0.3090 to +0.1%. linear_modiff_w4a4_ab.py measured the same arm at 0.3303
([0.360, 0.308, 0.323]). Position is excluded, drift is excluded, the delta table loaded in both, the
calibration file is the same, and the reference construction is line-for-line identical.

So the difference is in the measure() path, and this isolates it by running BOTH implementations
back to back against ONE reference set. Whichever way it comes out, one arm here is wrong and the
report should not quote either number until it is known which.

  A  "clean"  -- exactly arm_position_effect.measure: set delta mode, build, discard 1, 3 seeds
  B  "ab"     -- exactly linear_modiff_w4a4_ab.measure: the same PLUS
                   (1) os.environ["MODIFF_LINEAR"] set explicitly per arm
                   (2) H.AUTO_DELTA_TABLE assigned inside the function rather than in main
                   (3) a count_modulated() walk over .modules() between build and first sample
  C  "ab minus the walk" -- B without (3), to attribute it if B != A

Nothing here is read-only by assumption: (3) is the only step that touches the model between
construction and sampling, and "it only does getattr" is precisely the kind of claim this session has
had to retract, so it gets measured rather than reasoned about.

Run: python docs/attn_modiff_2026-08-13/scripts/measure_path_aa.py    # ~15 min, needs the GPU
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

OUT = "docs/attn_modiff_2026-08-13/data/measure_path_aa.json"
SEEDS = [1234, 20260805, 777]
CLEAN = 0.3094          # arm_position_effect, positions 1 and 4, identical
AB = 0.3303             # linear_modiff_w4a4_ab, first arm


def count_modulated(model):
    """Verbatim from linear_modiff_w4a4_ab.py, so arm B is that script's path and not a paraphrase."""
    wxax = emb = conv = 0
    for m in model.model.diffusion_model.modules():
        t = type(m).__name__
        if "QuantLinear" in t and bool(getattr(m, "modiff", False)):
            wxax += 1
        elif t == "OptimizedInt4Linear" and bool(getattr(m, "modiff_enabled", False)):
            emb += 1
        elif t == "OptimizedInt4Conv2d" and bool(getattr(m, "modiff_enabled", False)):
            conv += 1
    return {"attn_proj_wxax": wxax, "emb_linear": emb, "conv": conv}


def run(variant, refs):
    if variant in ("ab", "ab_no_walk"):
        os.environ["MODIFF_LINEAR"] = "0"
        H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_DELTA_MODE"] = "static"
    cal = B._default_calibration_path("int4")
    r, m, s = H.build("int4", cal, "static")
    if variant == "ab":
        print(f"    walk: {count_modulated(m)}", flush=True)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)
    rels = []
    for sd in SEEDS:
        H.SEED = sd
        H.latent(r, m, s)
        lat, _ = H.latent(r, m, s)
        rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
    del r, m, s
    torch.cuda.empty_cache()
    mean = statistics.mean(rels)
    print(f"  {variant:12s} {mean:.4f}  {[round(x, 4) for x in rels]}", flush=True)
    return {"variant": variant, "mean": mean, "relL2": rels}


def main():
    H.STEPS, H.BATCH = 50, 8
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"

    print("fp16 references ...", flush=True)
    os.environ["MODIFF_DELTA_MODE"] = "dynamic"
    rf, mf, sf = H.build("fp16", None, "static")
    refs = {}
    for sd in SEEDS:
        H.SEED = sd
        H.latent(rf, mf, sf)
        refs[sd] = H.latent(rf, mf, sf)[0].float()
    del rf, mf, sf
    torch.cuda.empty_cache()

    print("\nvariants (same arm, same refs, one process):", flush=True)
    out = [run("clean", refs), run("ab", refs), run("ab_no_walk", refs)]
    by = {o["variant"]: o["mean"] for o in out}

    print(f"\n{'variant':14}{'relL2':>9}{'vs clean':>10}")
    for o in out:
        print(f"{o['variant']:14}{o['mean']:9.4f}{(o['mean']/by['clean']-1)*100:9.1f}%")

    print()
    same = abs(by["ab"] / by["clean"] - 1) <= 0.006
    if same:
        print(f"BOTH PATHS AGREE here ({by['clean']:.4f} vs {by['ab']:.4f}), and both match the "
              f"committed {CLEAN:.4f}. So the {AB:.4f} that linear_modiff_w4a4_ab.py reported is NOT "
              f"reproducible from its code path -- it came from that PROCESS, not that script. The "
              f"linmodiff comparison was internally consistent (one process, one reference set), but "
              f"its absolute numbers cannot be quoted against committed values and the arm needs "
              f"re-running.")
    elif abs(by["ab_no_walk"] / by["clean"] - 1) <= 0.006:
        print(f"THE .modules() WALK IS THE CAUSE: ab={by['ab']:.4f} but ab_no_walk="
              f"{by['ab_no_walk']:.4f} == clean. A walk that only does getattr is changing the "
              f"measurement, which means something in it is not read-only -- find it before "
              f"trusting any gate built this way.")
    else:
        print(f"REPRODUCED the discrepancy WITHOUT the walk (ab_no_walk={by['ab_no_walk']:.4f}), so "
              f"the cause is the MODIFF_LINEAR/AUTO_DELTA_TABLE assignment placement. Both are "
              f"supposed to be idempotent; one is not.")

    json.dump({"seeds": SEEDS, "clean_reference": CLEAN, "ab_reported": AB, "results": out},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
