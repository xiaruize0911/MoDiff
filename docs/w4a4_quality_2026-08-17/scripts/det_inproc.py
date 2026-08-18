"""Is the 4.1/255 run-to-run spread CROSS-process state, or nondeterministic kernels?

Decisive because it removes the process from the experiment: same process, same model, same seed, two
draws. If they differ, no amount of seeding can fix it and the spread is a property of the kernels
(atomics in a reduction, or fp16 non-associativity) amplified by 50 iterated sampler steps.
"""
import os, sys, numpy as np, torch
ROOT = "/workspace/MoDiff"
os.chdir(ROOT); sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                                os.path.join(ROOT, "integration/benchmarks/report"),
                                os.path.join(ROOT, "docs/fid_2026-08-05/scripts")]
os.environ["DELTA_STATIC"] = "1"; os.environ["MODIFF_WARMUP_STEPS"] = "5"
sys.argv = ["x", "--n", "0", "--modes", "int4_l1", "--out", "/workspace/fid_det/tmp"]
import generate_fid_samples as G

runner, model, sampler = G.build("int4_l1", "dynamic", 8, 1)
G.reset(model); G.sample_batch(runner, model, sampler, 4, 12345)      # settle
outs = []
for i in range(2):
    G.reset(model)
    outs.append(G.sample_batch(runner, model, sampler, 4, 999).astype(np.int16))
d = np.abs(outs[0] - outs[1])
print(f"\nSAME PROCESS, same model, same seed, two draws:")
print(f"  mean |delta|   {d.mean():.4f}/255")
print(f"  max pixel diff {int(d.max())}")
print(f"  identical      {bool(d.max() == 0)}")
