"""The clip sweep at A8/A4/A3 end to end, now that the kernels have a real code ceiling.

`clip_e2e.py` could only measure A8, because before `code_ceiling` existed `MODIFF_DELTA_CLIP` was a
clip only where `act_q` happened to equal the kernels' hardcoded 127. With the ceiling threaded
through the GN-fused and modulated delta paths, a b-bit delta quantizer saturates at Q_b as it should,
and the offline predictions from `clip_probe.py` / `accum_probe.py` become testable:

    A8   trajectory-optimal r ~= 0.90, worth ~2.6% of accumulated conv activation error
         (measured end to end pre-ceiling: 1-2%, inside the seed spread -> no change to the default)
    A4   trajectory-optimal r ~= 0.25, worth ~26%
    A3   trajectory-optimal r ~= 0.20, worth ~33%

Two distinct effects are in play and the table separates them by row:

  * r < 1 is the clip itself, and it only does anything now.
  * r = 1 at MODIFF_DELTA_REFRESH > 1 is a DEFECT FIX, not a knob. On a reuse step the scale is up to
    K-1 steps old, so the delta can outgrow it; those codes are supposed to saturate at Q_b, and
    clamped at 127 instead an "A4" layer could emit a code of 100 on 3 of every 4 steps. So the
    A4/A3 r=1.0 rows are expected to MOVE at K=4 and to be unchanged at K=1, and the A8 rows are
    expected to be unchanged at both (act_q == 127 == the old literal). That pattern is the control:
    if an A8 row moves, the change is not doing what it claims.

Same protocol as every sweep in this directory: one warm-up sampling run per arm discarded, paired
over seeds against a per-seed fp16 reference, real-checkpoint calibration.
"""

import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                    # noqa: E402
import dynamic_delta_ab as H                                                    # noqa: E402

BITS = [(int(b), 2.0 ** (int(b) - 1) - 1) for b in os.environ.get("CB_BITS", "8,4,3").split(",")]
RATIOS = [float(r) for r in os.environ.get("CB_RATIOS", "1.0,0.6,0.4,0.3,0.25,0.2").split(",")]
REFRESH = [int(k) for k in os.environ.get("CB_REFRESH", "1").split(",")]
SEEDS = [int(s) for s in os.environ.get("CB_SEEDS", "1234,20260805,777").split(",")]
OUT = os.environ.get("CB_OUT", "docs/delta_clip_2026-08-06/data/clip_e2e_bits.json")
CALIB = H.CALIB["int8"]


def runs(mode, delta_mode, refs):
    r, m, s = H.build(mode, None if mode == "fp16" else CALIB, delta_mode)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                                   # warm-up, discarded
    rel, ms, lat = {}, [], {}
    for seed in SEEDS:
        H.SEED = seed
        cur, t = H.latent(r, m, s)
        ms.append(t)
        if refs is None:
            lat[seed] = cur
        else:
            rel[seed] = float((cur - refs[seed]).norm() / refs[seed].norm())
    del m, s, r
    torch.cuda.empty_cache()
    return (lat if refs is None else rel), sum(ms) / len(ms)


def stat(d):
    v = list(d.values())
    return (statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), min(v), max(v))


def main():
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_DELTA_CLIP"] = "1.0"
    os.environ["MODIFF_ACT_Q"] = "127"

    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}, bits {[b for b, _ in BITS]}, "
          f"ratios {RATIOS}, refresh {REFRESH}\n", flush=True)
    refs, fp16_ms = runs("fp16", "static", None)
    print(f"fp16 reference: {fp16_ms:6.2f} ms/step\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "seeds": SEEDS, "ratios": RATIOS,
           "fp16_ms_per_step": fp16_ms, "rows": []}
    print(f"{'K':>3} {'bits':>5} {'clip r':>7} | {'MoDiff relL2':>34} | {'vs r=1':>7} | "
          f"{'ms/step':>8}", flush=True)
    print("-" * 74, flush=True)

    for k in REFRESH:
        os.environ["MODIFF_DELTA_REFRESH"] = str(k)
        for bits, q in BITS:
            os.environ["MODIFF_ACT_Q"] = str(q)
            anchor = None
            for r in RATIOS:
                os.environ["MODIFF_DELTA_CLIP"] = str(r)
                rel, ms = runs("int8", "dynamic", refs)
                st = stat(rel)
                anchor = st[0] if anchor is None else anchor
                out["rows"].append({"delta_refresh": k, "act_bits": bits, "q_level": q,
                                    "clip_ratio": r, "mean": st[0], "stdev": st[1],
                                    "per_seed": rel, "ms_per_step": ms,
                                    "ratio_to_r1": st[0] / anchor})
                print(f"{k:>3} {('A%d' % bits):>5} {r:>7.2f} | {st[0]:>8.4f} +- {st[1]:<6.4f} "
                      f"[{st[2]:.4f},{st[3]:.4f}] | {st[0] / anchor:>6.3f}x | {ms:>7.1f}",
                      flush=True)
                with open(OUT, "w") as f:
                    json.dump(out, f, indent=2)
            print(flush=True)

    os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = "127", "1.0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
