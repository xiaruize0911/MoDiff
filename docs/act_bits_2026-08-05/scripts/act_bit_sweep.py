"""Sweep activation precision at fixed W8, MoDiff on and off, A8 down to A2.

This is the configuration the MoDiff paper actually claims -- W8 weights with the activation
precision pushed down, "up to 3 bits" per the repo README -- and it is the one configuration this
project had never measured as a sweep. The shipped modes pair the two precisions (W8A8, W4A4), so
every earlier quality number confounds weight error with activation error. W4A4+MoDiff at FID 200
(docs/fid_2026-08-05) is the visible consequence: int4 WEIGHTS dominate, and MoDiff is an activation
method, so that row cannot test the paper's claim either way.

Both arms move together, which is the point:

    baseline   MoDiff off, static per-tensor activation grid rescaled to Q_b/range   (= PTQ at A_b)
    MoDiff     Q(a_t - a_hat_{t+1}) on a dynamic Q_b/max|delta| grid, t=T warm-up also at A_b

so each row answers "at b activation bits, what does modulation buy", and the paper's claim is that
the MoDiff column stays usable several bits below where the baseline column collapses.

Protocol, inherited from dynamic_delta_ab.py because its warnings were paid for:
  * ONE warm-up sampling run per arm, DISCARDED. The quantized attention blocks self-calibrate over
    their first forwards; run 1 differs from run 2 by up to 5x, and it flatters the arms unequally.
  * PAIRED over seeds. Run-to-run relL2 at batch 8 varies 10-30%, which is larger than several of
    the gaps here. Every arm sees the same seed list and the fp16 reference is regenerated per seed,
    so each relL2 is a within-seed comparison and the spread across seeds is reported.
  * real-checkpoint calibration (int8_calibration_realckpt.pt). The un-suffixed artifact was fitted
    against the old stub checkpoint and gives relL2 0.88 at W8A8.
  * ms/step is recorded per arm as a control, NOT as a result: a low A_b costs nothing and saves
    nothing here (int8 containers, W8A8 GEMM), so the timing column should be flat. If it is not,
    something other than the quantizer changed.

Q is the symmetric code ceiling 2^(b-1)-1, so A8=127 ... A2=1. Q=127 reproduces the shipped
configuration bit-for-bit, which is the plumbing control: expect ~0.24 baseline / ~0.04 MoDiff at
batch 8 / DDIM 50 (dynamic_delta_ab.py, 2026-08-04).

Scope, stated because it bounds what the numbers mean: MODIFF_ACT_Q reaches the quantized CONV path
(87 layers). The attention blocks and the Linear layers stay at A8 in BOTH arms -- so this is
"A_b in the conv path", not a whole-network A_b, and the baseline arm keeps int8 headroom above its
calibrated range where a true b-bit quantizer would saturate (see MODIFF_ACT_Q's comment in
integration/kernels/int8_optimized.py). Both effects understate rather than overstate MoDiff.
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
import dynamic_delta_ab as H          # one definition of build()/latent()       # noqa: E402

#: Q = 2^(b-1)-1. 127 is the shipped A8 and doubles as the control row.
Q_LEVELS = [(8, 127.0), (7, 63.0), (6, 31.0), (5, 15.0), (4, 7.0), (3, 3.0), (2, 1.0)]
if os.environ.get("SWEEP_BITS"):                        # subset, for smoke tests and re-runs
    _want = {int(b) for b in os.environ["SWEEP_BITS"].split(",")}
    Q_LEVELS = [(b, q) for b, q in Q_LEVELS if b in _want]
SEEDS = [int(s) for s in os.environ.get("SWEEP_SEEDS", "1234,20260805,777").split(",")]
CALIB = H.CALIB["int8"]
OUT = "docs/act_bits_2026-08-05/data/act_bit_sweep.json"


def runs(mode, delta_mode, refs):
    """relL2 vs the same-seed fp16 latent, for each seed, from one freshly built model.

    H.latent() seeds from the module constant H.SEED, so the seed is set there rather than passed;
    the alternative is a second copy of the harness, which is how the earlier sweeps drifted apart.
    """
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
    return (statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0),
            min(v), max(v))


def main():
    os.environ["MODIFF_DELTA_REFRESH"] = "4"
    os.environ["MODIFF_DELTA_CLIP"] = "1.0"    # the bits knob does the clipping now, not this one
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_ACT_Q"] = "127"

    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}\n", flush=True)
    refs, fp16_ms = runs("fp16", "static", None)
    print(f"fp16 reference: {fp16_ms:6.2f} ms/step, "
          f"|x|max {max(float(v.abs().max()) for v in refs.values()):.4f}\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "seeds": SEEDS,
           "fp16_ms_per_step": fp16_ms, "rows": []}
    print(f"{'bits':>5} {'Q':>4} {'levels':>7} | {'baseline relL2':>22} | "
          f"{'MoDiff relL2':>22} | {'gain':>6} | {'ms/step b/M':>13}", flush=True)
    print("-" * 96, flush=True)

    for bits, q in Q_LEVELS:
        os.environ["MODIFF_ACT_Q"] = str(q)
        base_rel, base_ms = runs("int8_baseline", "static", refs)
        mod_rel, mod_ms = runs("int8", "dynamic", refs)
        b, m = stat(base_rel), stat(mod_rel)
        out["rows"].append({
            "act_bits": bits, "q_level": q, "levels": int(2 * q + 1),
            "baseline": {"mean": b[0], "stdev": b[1], "per_seed": base_rel,
                         "ms_per_step": base_ms},
            "modiff": {"mean": m[0], "stdev": m[1], "per_seed": mod_rel,
                       "ms_per_step": mod_ms},
            "gain_baseline_over_modiff": (b[0] / m[0]) if m[0] > 0 else None,
        })
        print(f"A{bits:<4d} {int(q):>4d} {int(2 * q + 1):>7d} | "
              f"{b[0]:>8.4f} +- {b[1]:<6.4f} [{b[2]:.3f},{b[3]:.3f}] | "
              f"{m[0]:>8.4f} +- {m[1]:<6.4f} [{m[2]:.3f},{m[3]:.3f}] | "
              f"{b[0] / m[0]:>5.2f}x | {base_ms:5.1f} {mod_ms:5.1f}", flush=True)
        with open(OUT, "w") as f:                        # written per row: the sweep is resumable
            json.dump(out, f, indent=2)

    os.environ["MODIFF_ACT_Q"] = "127"
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
