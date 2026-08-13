"""The W4A4 MoDiff arm depends on WHICH ARM RAN BEFORE IT, by 28%. Reproducer.

THE TREE HAS BEEN CHASING THIS FOR A WHILE. docs/attn_modiff_2026-08-13/scripts/arm_position_effect.py
was written because linear_modiff_w4a4_ab.py read the shipped W4A4 MoDiff arm at 0.3303 where the
committed value is 0.3090 -- 6.9% against a 0.6% floor -- with every axis it could check (calibration
file, delta table, seeds, batch, steps, discard protocol, reference construction) identical. It tested
POSITION by measuring int4 as arm 1 and again as arm 4 with two fillers between, got bit-identical
0.30940 both times, and concluded "position_irrelevant" -- leaving the discrepancy open.

Position is not the variable. THE IDENTITY OF THE PRECEDING ARM IS:

    int4 measured FIRST in a fresh process              mean 0.3954  [0.4415, 0.3693, 0.3754]
    int4 measured AFTER int4_baseline, same process     mean 0.3095  [0.3266, 0.2883, 0.3134]
                                                                     ^^^ the committed 0.3090

28%, reproducible, one variable. arm_position_effect.py missed it because its arm 1 was ALREADY
preceded by work that has the same effect, so its A1 and A4 agreed with each other and with the
committed value -- the comparison it drew could not see the thing it was looking for.

NOT PROCESS WARMING IN GENERAL. Building an fp16 model and running a latent before the int4 arm --
which is exactly what export_and_measure_zp.collect_ranges() does to its process -- leaves the int4 arm
BIT-IDENTICAL at 0.3954. So it is not "the process is cold"; it is something a quantized W4A4 arm
leaves behind that a MoDiff W4A4 arm then reads.

WHY MoDIFF AND NOT PTQ. int4_baseline is stable at 0.5022-0.5026 whether it runs first or after
others. The asymmetry points at temporal state: MoDiff seeds a_hat/o_hat at t=T and then accumulates
o_hat across all 50 steps, so anything that perturbs the first step is integrated rather than averaged
away -- the same reason the padding defect cost MoDiff +204% and PTQ +82%
(docs/zp_coverage_2026-08-13/FINDINGS.md).

CANDIDATE MECHANISM, NOT YET PROVEN. Each build runs a short sampling pass to self-calibrate 42
attention linear scales ("Calibrated 42 W4A4 linear activation scales"). A preceding W4A4 arm can leave
global algorithm-selection state warm (cuDNN benchmark cache, CUTLASS can_implement/autotune results),
which changes that pass's reduction orders, which changes the SCALES -- a calibration difference, which
is the right order of magnitude for 28% where a pure rounding difference is not. Proving it means
capturing the 42 scales in both orders and diffing them; that is the next step and this file does not
claim it.

WHAT TO DO ABOUT IT MEANWHILE -- the rule this establishes:

  * Compare only arms measured IN THE SAME PROCESS IN THE SAME ORDER. Every A/B in this tree that does
    this is fine, including the fix #2 measurement (symmetric and asymmetric arms in one run).
  * A single-arm-per-process harness is NOT interchangeable with a multi-arm one. Numbers do not
    transfer between the two, and 28% is 200x the measured cross-process floor of 0.13%, so no floor
    argument licenses the comparison.
  * A harness that cannot reproduce a committed number should suspect ARM ORDER before drift.

Run: python docs/zp_coverage_2026-08-13/scripts/arm_order_reproducer.py    # ~8 min, needs an idle GPU
"""
import io
import contextlib
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = "docs/zp_coverage_2026-08-13"
SEEDS = [1234, 20260805, 777]

CHILD = r'''
import io, contextlib, json, os, re, sys
ROOT = %(root)r
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/attn_modiff_2026-08-13/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.join(ROOT, "docs/zero_point_2026-08-13/scripts")]
import torch, export_and_measure_zp as M, fp16_refs
M.H.STEPS, M.H.BATCH = 50, 8
SEEDS = %(seeds)r
M.SEEDS = SEEDS
refs = fp16_refs.get(50, 8, SEEDS)
out = []
for mode in sys.argv[1:]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mean = M.measure(mode, None, refs, mode)
    t = buf.getvalue()
    s = re.search(r"\[([0-9.,\s]+)\]", t)
    out.append({"mode": mode, "mean": mean,
                "rels": [float(x) for x in s.group(1).split(",")] if s else []})
print("RESULT_JSON " + json.dumps(out))
'''


def run(order, child):
    p = subprocess.run([sys.executable, child] + order, capture_output=True, text=True,
                       cwd=ROOT, timeout=5400)
    for line in reversed(p.stdout.split("\n")):
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
    raise RuntimeError(f"no result for {order}")


def main():
    child = os.path.join(ROOT, D, "scripts", "_arm_order_child.py")
    os.makedirs(os.path.dirname(child), exist_ok=True)
    with open(child, "w") as f:
        f.write(CHILD % {"root": ROOT, "seeds": SEEDS})

    #: The two orders. "alone" is the control that isolates the preceding arm as the only difference.
    orders = {"int4_alone": ["int4"],
              "int4_after_baseline": ["int4_baseline", "int4"]}
    res = {}
    for label, order in orders.items():
        rows = run(order, child)
        res[label] = rows
        for r in rows:
            print(f"  {label:22s} {r['mode']:16s} mean {r['mean']:.4f}  "
                  f"{[round(x, 4) for x in r['rels']]}", flush=True)

    alone = [r for r in res["int4_alone"] if r["mode"] == "int4"][0]["mean"]
    after = [r for r in res["int4_after_baseline"] if r["mode"] == "int4"][0]["mean"]
    delta = (alone - after) / after * 100
    committed = 0.3090
    out = {"seeds": SEEDS, "results": res, "int4_alone": alone, "int4_after_baseline": after,
           "delta_pct": delta, "committed": committed,
           "cross_process_floor_pct": 0.13}
    json.dump(out, open(f"{ROOT}/{D}/data/arm_order.json", "w"), indent=1)

    print(f"\nint4 alone              {alone:.4f}")
    print(f"int4 after int4_baseline{after:9.4f}   (committed {committed:.4f})")
    print(f"difference              {delta:+.1f}%   against a measured cross-process floor of 0.13%")
    if abs(delta) > 1.0:
        print("\nCONFIRMED: the preceding arm changes the W4A4 MoDiff arm far outside the floor.\n"
              "Compare only arms measured in the same process in the same order. This retracts\n"
              "arm_position_effect.py's \"position_irrelevant\" verdict -- not because position is the\n"
              "variable, but because the preceding ARM is, and that harness's arm 1 was already\n"
              "preceded by work with the same effect.")
    else:
        print("\nNOT REPRODUCED on this run. Do not cite the 28% without re-establishing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
