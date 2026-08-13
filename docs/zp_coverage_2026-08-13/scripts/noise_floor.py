"""P5: what IS the relL2 noise floor, per arm, across processes? Measured, with nothing changed.

WHAT IS ON RECORD AND WHY IT NEEDS REDOING. docs/paper_repro_2026-08-12/FINDINGS.md section 7 reports a
zero-change repeat: "W8A8 arms 1.3-5.1%, W4A4 arms 0.05-0.6%", and every W4A4 conclusion since has
leaned on that 0.6%. Today the W4A4 PTQ symmetric arm -- identical inputs, identical cached fp16
references, identical protocol -- read 0.5267, then 0.4901, then 0.5022 in three separate runs, having
earlier reproduced 0.5266851782798767 to sixteen significant digits. A 7% spread against a 0.6% floor.

TWO CANDIDATE EXPLANATIONS, and this script is built to separate them:

  A. CROSS-PROCESS NONDETERMINISM in the arm itself. Each arm rebuilds the model, and the W4A4 build
     runs a short sampling pass to self-calibrate 42 attention linear scales. If that pass is
     nondeterministic (fp16 attention kernel selection varies with GPU state), the SCALES differ, and
     an arm graded against a pinned reference moves without anything having changed.
  B. GPU CONTENTION. run_all.sh already documents a second CUDA process turning CV 0.23% into 38%, and
     at least one of today's three runs overlapped another session's GPU work.

The design that distinguishes them: N repeats of each arm, each in its OWN PROCESS, run STRICTLY
SEQUENTIALLY on an otherwise-idle GPU, with the GPU checked for other compute processes before every
launch and the run ABORTED if one appears. Under that protocol B is excluded by construction, so
whatever spread remains is A.

WITHIN-PROCESS REPEATS WOULD NOT ANSWER THIS. measure() already discards its first sampling run and
averages three seeds inside one process; that is what produced the 16-digit agreement. The variable
under test is the PROCESS, so the repeat has to be a process.

Reports, per arm: the three seed values from each repeat, the per-seed spread across repeats, and the
spread of the 3-seed mean -- because the mean is what every A/B in this tree actually quotes, and its
spread is the number that should be called the floor.

Run: python docs/zp_coverage_2026-08-13/scripts/noise_floor.py [--repeats 3] [--arms ...]
     ~2 min per (arm, repeat); the default 4 arms x 3 repeats is ~25 min. Needs an idle GPU.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = "docs/zp_coverage_2026-08-13"
SEEDS = [1234, 20260805, 777]

#: One arm, one process. Printed as JSON on the last line so the parent does not have to parse prose.
#:
#: IT CALLS export_and_measure_zp.measure() RATHER THAN REIMPLEMENTING IT, and that is the whole point
#: of this block. The first version of this child open-coded the same sequence -- build, one discard
#: latent, then two latents per seed against the pinned refs -- and read 0.3797 for the int4 arm where
#: measure() reads 0.3090, a 23% disagreement between two harnesses that were supposed to be the same
#: measurement. A floor measured with a harness that does not reproduce the arm is a measurement of the
#: harness. So the child imports the real one; whatever spread survives belongs to the arm.
#:
#: measure() returns the mean and prints the per-seed list, so the child re-derives the seeds from its
#: stdout rather than duplicating the loop.
CHILD = r'''
import json, os, re, sys, io, contextlib
ROOT = %(root)r
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/attn_modiff_2026-08-13/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.join(ROOT, "docs/zero_point_2026-08-13/scripts")]
import torch
import export_and_measure_zp as M
import fp16_refs
mode = sys.argv[1]
M.H.STEPS, M.H.BATCH = %(steps)d, %(batch)d
SEEDS = %(seeds)r
M.SEEDS = SEEDS
refs = fp16_refs.get(M.H.STEPS, M.H.BATCH, SEEDS)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mean = M.measure(mode, None, refs, mode)
txt = buf.getvalue()
sys.stderr.write(txt)
m = re.search(r"\[([0-9.,\s]+)\]", txt)
rels = [float(x) for x in m.group(1).split(",")] if m else []
print("RESULT_JSON " + json.dumps({"mode": mode, "rels": rels, "mean": mean}))
'''


def gpu_busy_pids():
    """Other processes holding GPU compute contexts. Empty is the precondition for a valid repeat."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    return [p.strip() for p in out.split("\n") if p.strip()]


def one(mode, steps, batch, child_path):
    p = subprocess.run([sys.executable, child_path, mode], capture_output=True, text=True,
                       cwd=ROOT, timeout=3600)
    for line in reversed(p.stdout.split("\n")):
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])["rels"]
    sys.stderr.write(p.stdout[-3000:] + "\n" + p.stderr[-3000:] + "\n")
    raise RuntimeError(f"{mode}: child produced no result")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", default="int8_baseline,int8,int4_baseline,int4")
    a = ap.parse_args()
    arms = a.arms.split(",")

    os.makedirs(f"{D}/data", exist_ok=True)
    child_path = os.path.join(ROOT, D, "scripts", "_noise_floor_child.py")
    with open(child_path, "w") as f:
        f.write(CHILD % {"root": ROOT, "steps": a.steps, "batch": a.batch, "seeds": SEEDS})

    pre = gpu_busy_pids()
    if pre:
        print(f"REFUSING TO START: other GPU compute processes present ({pre}). This measurement is "
              f"about cross-process reproducibility and contention would be indistinguishable from it.")
        return 1

    out = {"steps": a.steps, "batch": a.batch, "seeds": SEEDS, "repeats": a.repeats, "arms": {}}
    for mode in arms:
        runs = []
        for i in range(a.repeats):
            busy = [p for p in gpu_busy_pids()]
            if busy:
                print(f"  ABORTING at {mode} repeat {i + 1}: another GPU process appeared ({busy}).")
                out["aborted_at"] = f"{mode}/{i + 1}"
                break
            rels = one(mode, a.steps, a.batch, child_path)
            runs.append(rels)
            print(f"  {mode:16s} repeat {i + 1}  {[round(x, 4) for x in rels]}  "
                  f"mean {statistics.mean(rels):.4f}", flush=True)
        if not runs:
            continue
        means = [statistics.mean(r) for r in runs]
        per_seed_spread = []
        for j in range(len(SEEDS)):
            vals = [r[j] for r in runs]
            per_seed_spread.append((max(vals) - min(vals)) / statistics.mean(vals) * 100)
        mean_spread = (max(means) - min(means)) / statistics.mean(means) * 100
        out["arms"][mode] = {
            "runs": runs, "means": means,
            "mean_spread_pct": mean_spread,
            "per_seed_spread_pct": per_seed_spread,
            "bit_identical": all(r == runs[0] for r in runs),
        }
        print(f"  {mode:16s} -> mean spread {mean_spread:.2f}%  "
              f"per-seed {[round(x, 2) for x in per_seed_spread]}  "
              f"bit-identical across processes: {out['arms'][mode]['bit_identical']}\n", flush=True)

    json.dump(out, open(f"{D}/data/noise_floor.json", "w"), indent=1)
    print(f"wrote {D}/data/noise_floor.json")

    print(f"\n{'arm':18}{'mean of means':>15}{'mean spread':>13}{'recorded floor':>16}")
    RECORDED = {"int8_baseline": "1.3-5.1%", "int8": "1.3-5.1%",
                "int4_baseline": "0.05-0.6%", "int4": "0.05-0.6%"}
    verdict = []
    for mode, v in out["arms"].items():
        mm = statistics.mean(v["means"])
        print(f"{mode:18}{mm:15.4f}{v['mean_spread_pct']:12.2f}%{RECORDED.get(mode, '?'):>16}")
        if mode.startswith("int4") and v["mean_spread_pct"] > 0.6:
            verdict.append(f"{mode} spreads {v['mean_spread_pct']:.2f}% against a recorded 0.6%")
    print()
    if verdict:
        print("THE RECORDED W4A4 FLOOR DOES NOT HOLD ACROSS PROCESSES:")
        for v in verdict:
            print(f"  - {v}")
        print("Measured on an idle GPU with the contention check above, so this is the arm's own\n"
              "cross-process nondeterminism, not another process. Any conclusion in this tree resting\n"
              "on a few percent of W4A4 relL2 needs re-checking against THIS number.")
    elif out["arms"]:
        print("The recorded floors hold on this protocol. Today's 7% spread on W4A4 PTQ was therefore\n"
              "contention or a genuine change, not the arm's own cross-process variance -- and the\n"
              "contention hypothesis is the one with independent support (run_all.sh).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
