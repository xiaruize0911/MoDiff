"""The 16-image screen's noise floor, per arm, measured the only way that isolates it: repeat draws.

Same process, same model, same seed, MoDiff state reset between draws. Anything that differs is the
datapath's own run-to-run nondeterminism amplified over 50 iterated sampler steps -- not seeding, not
arm order, not process state.

This exists because docs/w4a4_quality_2026-08-17 ranked eight arms on gaps as small as 0.25/255, and a
first probe put the W4A4 spread at 4.1-5.4/255. Every ordering in that report has to be re-read against
this number, and the ones inside it are not results.

W8A8 is measured too, because the mechanism predicts the floor is bit-width dependent: at 255 levels a
1-ulp perturbation stays inside a code, at 15 levels it crosses one and the delta recursion carries it
forward.
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/fid_2026-08-05/scripts")]
os.environ["MODIFF_WARMUP_STEPS"] = "5"
sys.argv = ["x", "--n", "0", "--modes", "int4_l1", "--out", "/workspace/fid_det/tmp"]

N_DRAW = int(os.environ.get("N_DRAW", "3"))
BATCH = int(os.environ.get("NF_BATCH", "8"))
SEED = 4242

ARMS = [("int4_l1", "1", "W4A4 MoDiff  L1 + static"),
        ("int4_l0", "0", "W4A4 MoDiff  L0 + dynamic"),
        ("int8_l0", "0", "W8A8 MoDiff  L0 + dynamic")]

out = {}
for mode, static, label in ARMS:
    os.environ["DELTA_STATIC"] = static
    for m in [m for m in list(sys.modules) if m == "generate_fid_samples"]:
        del sys.modules[m]
    import generate_fid_samples as G                                    # noqa: E402
    spec = G.SPEC[mode]
    runner, model, sampler = G.build(mode, spec[1], spec[2], spec[3])
    G.reset(model)
    G.sample_batch(runner, model, sampler, BATCH, SEED - 1)              # settle, discarded
    draws = []
    for _ in range(N_DRAW):
        G.reset(model)
        draws.append(G.sample_batch(runner, model, sampler, BATCH, SEED).astype(np.int16))
    ds = [float(np.abs(a - b).mean()) for a, b in itertools.combinations(draws, 2)]
    mx = [int(np.abs(a - b).max()) for a, b in itertools.combinations(draws, 2)]
    out[label] = {"n_draws": N_DRAW, "batch": BATCH, "pairwise_mean_abs_delta_255": ds,
                  "pairwise_max_pixel_diff": mx,
                  "floor_mean_255": sum(ds) / len(ds), "floor_max_255": max(ds)}
    print(f"\n{label}: {N_DRAW} draws of {BATCH} images, same seed, state reset between")
    for (i, j), d, m_ in zip(itertools.combinations(range(N_DRAW), 2), ds, mx):
        print(f"  draw {i} vs {j}:  mean |Δ| {d:7.4f}/255   max pixel diff {m_:3d}")
    print(f"  FLOOR: mean {out[label]['floor_mean_255']:.4f}/255, worst pair "
          f"{out[label]['floor_max_255']:.4f}/255")
    del model, sampler, runner
    import torch
    torch.cuda.empty_cache()

p = "docs/w4a4_quality_2026-08-17/data/noise_floor.json"
json.dump(out, open(p, "w"), indent=1)
print(f"\nwrote {p}")
print("\nREAD THE REPORT'S TABLES AGAINST THESE. A gap smaller than the arm's floor is not a result.")
