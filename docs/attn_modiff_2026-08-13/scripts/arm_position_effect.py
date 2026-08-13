"""Does an arm's POSITION in the process change its measured relL2?

WHY THIS EXISTS. linear_modiff_w4a4_ab.py measured the shipped W4A4 MoDiff arm at 0.3303 where the
committed value is 0.3090 -- 6.9%, against a 0.6% noise floor. The tree did not drift: re-running
static_vs_dynamic_ab.py unmodified reproduced all six of its arms to four decimals, per-seed. And the
configurations match on every axis checked -- same calibration file, same delta table (both logs show
it loading), same seeds, same batch, same steps, same discard-run-1 protocol, same reference
construction.

The one difference left is WHERE the arm sits in the process. In the committed harness the W4A4 MoDiff
arm is the FIFTH model built; in mine it was the FIRST. And my third arm (the int4_baseline control)
reproduced its committed number to 0.5%.

THERE IS A KNOWN MECHANISM. dynamic_delta_ab.measure's own docstring: "One sampling run is NOT steady
state ... for int8 dynamic, the first run after model construction gives relL2 0.2107 and the second
gives 0.0399 -- a 5.3x difference. The quantized attention blocks self-calibrate their static scales
over the first MODIFF_ATTN_CALIB_STEPS forwards." Every harness here therefore discards ONE run. This
asks whether one is enough for the first model in a fresh process, or whether something else about a
cold process (cuDNN autotune picking different kernels, hence different reduction orders) also has to
settle.

WHAT IS AT STAKE. If position matters, it is not a curiosity about one script: it means a
single-arm measurement in a fresh process reads differently from the same arm measured after others,
and several conclusions in this session came from harnesses that measure a handful of arms in an
order chosen for convenience. It would also mean 0.3090 is the warm value and 0.3303 the cold one,
rather than either being wrong.

DESIGN. The SAME arm (int4, static, shipped calibration and delta table) is measured twice in one
process -- as arm 1 and as arm 4 -- with two unrelated arms in between to advance the process state.
Identical code path both times, so any difference is position and nothing else.

  A1  int4/static      position 1  (cold process)
  A2  int4_baseline    position 2  (filler, and a second read on a committed number)
  A3  int8/static      position 3  (filler, different bit width)
  A4  int4/static      position 4  (warm process) -- must match A1 if position is irrelevant

PREDICTION, STATED BEFORE THE RUN: if position is the variable, A1 ~ 0.330 and A4 ~ 0.309. If A1 == A4
then position is NOT the explanation and the discrepancy is something my script does that this one
also does, which would then need a line-by-line diff against static_vs_dynamic_ab.

Run: python docs/attn_modiff_2026-08-13/scripts/arm_position_effect.py    # ~20 min, needs the GPU
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

OUT = "docs/attn_modiff_2026-08-13/data/arm_position_effect.json"
SEEDS = [1234, 20260805, 777]
COMMITTED = {"int4": 0.3090, "int4_baseline": 0.4695, "int8": 0.0605}


def measure(mode, refs, pos):
    """Byte-for-byte the committed harness's arm: env, build, discard one run, then 3 seeds."""
    os.environ["MODIFF_DELTA_MODE"] = "static"
    cal = B._default_calibration_path(mode)
    r, m, s = H.build(mode, cal, "static")
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
    mean = statistics.mean(rels)
    ref = COMMITTED.get(mode)
    print(f"  pos {pos}  {mode:14s} {mean:.4f}  {[round(x, 4) for x in rels]}"
          f"{f'   committed {ref:.4f} -> {(mean/ref-1)*100:+.1f}%' if ref else ''}", flush=True)
    return {"mode": mode, "position": pos, "mean": mean, "relL2": rels}


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

    print("\narms (the SAME int4/static arm at position 1 and position 4):", flush=True)
    out = [measure("int4", refs, 1),
           measure("int4_baseline", refs, 2),
           measure("int8", refs, 3),
           measure("int4", refs, 4)]

    a1, a4 = out[0]["mean"], out[3]["mean"]
    print(f"\nsame arm, position 1: {a1:.4f}")
    print(f"same arm, position 4: {a4:.4f}")
    print(f"position effect:      {(a4/a1-1)*100:+.1f}%")
    #: W4A4 run-to-run floor on this protocol (FINDINGS.md section 7)
    FLOOR = 0.006
    verdict = "position_matters" if abs(a4 / a1 - 1) > FLOOR else "position_irrelevant"
    print()
    if verdict == "position_matters":
        print(f"POSITION MATTERS, beyond the {FLOOR*100:.1f}% floor. A single-arm measurement in a "
              f"fresh process does NOT equal the same arm measured after others, so discarding one "
              f"run is not sufficient warm-up. Every harness in this tree that measures a short arm "
              f"list needs to be read with its ORDER in mind, and any two numbers compared across "
              f"harnesses need the same position to be comparable.")
        if abs(a4 / COMMITTED["int4"] - 1) <= FLOOR:
            print(f"  And position 4 lands on the committed {COMMITTED['int4']:.4f}, so 0.3090 is the "
                  f"warm value and 0.3303 the cold one -- neither is a defect, but they are not "
                  f"interchangeable.")
    else:
        print("POSITION IS NOT THE EXPLANATION. The same arm gives the same answer first or fourth, "
              "so the 0.3303-vs-0.3090 gap comes from something linear_modiff_w4a4_ab.py does that "
              "this script does not. Diff the two measure() paths line by line before trusting "
              "either number.")

    json.dump({"seeds": SEEDS, "committed": COMMITTED, "floor": FLOOR,
               "arms": out, "position_effect": a4 / a1, "verdict": verdict},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
