"""Does the clip ratio the offline probe likes actually improve the latent? A8, end to end.

clip_probe.py measured, on real deltas, that the MSE-optimal clip ratio is well below the shipped
r=1.0: about 0.6-0.8 at A8 (10-31% less delta quantization error) and 0.2-0.3 at A4 (45-56%). Only
the A8 number is testable without touching CUDA, and for a specific reason: `MODIFF_DELTA_CLIP=r` is
`Q_level = act_q/r` and the kernels clamp codes at a hardcoded +-127, so the clamp is at
`(127/Q_level)*absmax = (r*127/act_q)*absmax`. At A8, act_q == 127 and that is exactly `r*absmax` --
a faithful clip. At A4 (act_q=7) it is `18.1*r*absmax`, which for every r in the useful range does
not clip at all; the knob just refines the grid, and a sweep there would report a monotone "win"
that is only a higher effective bit width. So A4/A3 waits on a real code ceiling in the kernels.

Protocol is act_bit_sweep.py's, for the same reasons: one warm-up sampling run per arm DISCARDED
(the quantized attention blocks self-calibrate over their first forwards), PAIRED over seeds against
a per-seed fp16 reference, real-checkpoint calibration.

Swept at both refresh settings because they interact. At K=1 the scale is remeasured every step,
which is what the offline probe assumed. At K=4 (the shipped default) a scale measured up to 3 steps
ago is reused, and the delta's range grows in between -- which clips even at r=1.0. If a clip ratio
helps anywhere it should help more at K=4, and the two columns separate "clipping is good for the
quantizer" from "clipping compensates for a stale scale".
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

RATIOS = [float(r) for r in os.environ.get("CLIP_RATIOS", "1.0,0.9,0.8,0.7,0.6,0.5,0.4").split(",")]
REFRESH = [int(k) for k in os.environ.get("CLIP_REFRESH", "1,4").split(",")]
SEEDS = [int(s) for s in os.environ.get("CLIP_SEEDS", "1234,20260805,777").split(",")]
OUT = os.environ.get("CLIP_E2E_OUT", "docs/delta_clip_2026-08-06/data/clip_e2e_a8.json")
CALIB = H.CALIB["int8"]


def runs(mode, delta_mode, refs):
    """relL2 vs the same-seed fp16 latent, per seed, from one freshly built model."""
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
    os.environ["MODIFF_ACT_Q"] = "127"          # A8: the one precision where the knob is a real clip
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_DELTA_CLIP"] = "1.0"

    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}, A8 (MODIFF_ACT_Q=127), "
          f"ratios {RATIOS}, refresh {REFRESH}\n", flush=True)
    refs, fp16_ms = runs("fp16", "static", None)
    print(f"fp16 reference: {fp16_ms:6.2f} ms/step\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "seeds": SEEDS, "act_q": 127,
           "fp16_ms_per_step": fp16_ms, "rows": []}
    print(f"{'K':>3} {'clip r':>7} | {'MoDiff relL2':>34} | {'vs r=1.0':>9} | {'ms/step':>8}",
          flush=True)
    print("-" * 70, flush=True)

    for k in REFRESH:
        os.environ["MODIFF_DELTA_REFRESH"] = str(k)
        anchor = None
        for r in RATIOS:
            os.environ["MODIFF_DELTA_CLIP"] = str(r)
            rel, ms = runs("int8", "dynamic", refs)
            st = stat(rel)
            anchor = st[0] if anchor is None else anchor
            out["rows"].append({"delta_refresh": k, "clip_ratio": r, "mean": st[0],
                                "stdev": st[1], "per_seed": rel, "ms_per_step": ms,
                                "ratio_to_r1": st[0] / anchor})
            print(f"{k:>3} {r:>7.2f} | {st[0]:>8.4f} +- {st[1]:<6.4f} "
                  f"[{st[2]:.4f},{st[3]:.4f}] | {st[0] / anchor:>8.3f}x | {ms:>7.1f}", flush=True)
            with open(OUT, "w") as f:                    # per row: the sweep is resumable
                json.dump(out, f, indent=2)
        print(flush=True)

    os.environ["MODIFF_DELTA_CLIP"] = "1.0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
