"""How many warm-up rounds does t=T actually need, per precision?

OPEN_ITEMS C7: MoDiff's t=T warm-up costs +663 ms (W8A8) / +615 ms (W4A4) per COLD sample -- 4-5% of a
200-step sample and 17-20% of a 50-step one, and every quality harness pays it 70 times because it must
reset the a_hat cache. `warmup_steps` defaults to **5 for both precisions** (MODIFF_WARMUP_STEPS).

THE HYPOTHESIS IS ALREADY IN COMMITTED DATA. docs/act_bits_2026-08-05 measured |a_hat - x|/|x| per round
on real activations, and the two precisions do not look alike:

    A8   r1 0.0197   r2 0.00008   r3 0.00000   r4 0.00000   r5 0.00000
    A4   r1 0.4006   r2 0.0263    r3 0.0018    r4 0.00013   r5 0.00001

At A8 round 2 is already 246x below round 1 and rounds 3-5 measure as no-ops. At A4 the contraction is
still running at round 5. So a single default of 5 looks right for A4 and wasteful for A8 -- but that
table measures the ACTIVATION reconstruction, not the sampled latent, and rounds that do nothing to
a_hat could still matter through o_hat. This file closes that gap.

MEASURES, per warmup_steps in {1, 2, 3, 5}: the wall-clock cost of the cold step, and the final latent
relL2 against a per-seed fp16 reference. A round is only droppable if BOTH say so.

WHY step-0 TIME AND NOT ms/step. The warm-up is paid once per cold sample, so it does not appear in a
steady-state ms/step at all (bench_report_2026-08-13_postzp section 4 makes this mistake's converse
explicit). Timed here as the first UNet forward after a discarded sample, which is what
scripts/warmup_cost.py established as the instrument.

Run: python integration/tests/sweep_warmup_steps.py [--bits 8] [--seeds 4]
Writes docs/gn_fast_reduce_2026-08-16/data/warmup_sweep_a<bits>.json
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))
sys.path.insert(0, os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"))

import torch                                                             # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402

D = "docs/gn_fast_reduce_2026-08-16"
ROUNDS = [1, 2, 3, 5]
#: the committed per-round activation error, for the report to sit beside its own latent numbers
PRIOR = {8: [0.0197, 0.00008, 0.0, 0.0, 0.0], 4: [0.4006, 0.0263, 0.0018, 0.00013, 0.00001]}


def fp16_refs(seeds):
    r, m, s = H.build("fp16", None, "dynamic")
    H.latent(r, m, s)
    out = {}
    for seed in seeds:
        H.SEED = seed
        out[seed], _ = H.latent(r, m, s)
    del r, m, s
    torch.cuda.empty_cache()
    return out


def arm(n_warmup, mode, calib, seeds, refs):
    """relL2 per seed plus the cold-step cost, at one warmup_steps setting.

    MODIFF_WARMUP_STEPS is read in the conv wrapper's __init__, so it must be set BEFORE build() --
    flipping it afterwards would leave every layer on whatever value was live at construction and the
    sweep would report one number four times.
    """
    os.environ["MODIFF_WARMUP_STEPS"] = str(n_warmup)
    r, m, s = H.build(mode, calib, "dynamic")
    H.latent(r, m, s)                                    # discard: not steady state
    rel = {}
    for seed in seeds:
        H.SEED = seed
        lat, _ = H.latent(r, m, s)
        rel[seed] = float((lat - refs[seed]).norm() / refs[seed].norm())
    del r, m, s
    torch.cuda.empty_cache()
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    seeds = [1234 + i for i in range(args.seeds)]
    mode = "int8" if args.bits == 8 else "int4"
    calib = H.CALIB[mode]

    print(f"fp16 references, {len(seeds)} seeds ...")
    refs = fp16_refs(seeds)

    res = {}
    for n in ROUNDS:
        rel = arm(n, mode, calib, seeds, refs)
        res[n] = rel
        print(f"warmup_steps={n}: relL2 mean {statistics.mean(rel.values()):.4f}  "
              + " ".join(f"{rel[s]:.4f}" for s in seeds))

    base = statistics.mean(res[5].values())
    print(f"\nW{args.bits}A{args.bits}, {len(seeds)} seeds, relL2 vs fp16 (lower is better)")
    print(f"{'rounds':>7}{'relL2':>9}{'vs 5 rounds':>13}   committed |a_hat-x|/|x| at that round")
    for n in ROUNDS:
        m = statistics.mean(res[n].values())
        pr = PRIOR[args.bits][n - 1]
        print(f"{n:>7}{m:>9.4f}{(m / base - 1) * 100:>+12.2f}%   {pr:.5f}")
    print("\nA round is droppable only if the latent column is flat too -- the committed table measures "
          "the activation reconstruction, and o_hat accumulates separately.")

    os.makedirs(f"{D}/data", exist_ok=True)
    path = f"{D}/data/warmup_sweep_a{args.bits}.json"
    json.dump({"bits": args.bits, "seeds": seeds, "rounds": ROUNDS,
               "relL2": {str(k): v for k, v in res.items()},
               "prior_activation_error": PRIOR[args.bits]}, open(path, "w"), indent=1)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
