"""Is cudnn autotuning the cross-process source? Same arm, two processes, benchmark=False.

benchmark_ldm sets torch.backends.cudnn.benchmark = True at import (line 59) AND again inside
_setup_model (line 546), so it has to be turned off after BOTH.
"""
import os, sys, torch
OUT = sys.argv[1]                          # read BEFORE argv is overwritten
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers"),
              os.path.join(ROOT,"integration/benchmarks/report"),
              os.path.join(ROOT,"docs/fid_2026-08-05/scripts")]
sys.argv=["x","--n","0","--modes","int4_l0","--out","/workspace/fid_det/tmp"]
import generate_fid_samples as G
torch.backends.cudnn.benchmark = False
import numpy as np
spec=G.SPEC["int4_l0"]
runner, model, sampler = G.build("int4_l0", spec[1], spec[2], spec[3])
torch.backends.cudnn.benchmark = False     # _setup_model re-enables it
G.reset(model); G.sample_batch(runner, model, sampler, 8, 20260804)
G.reset(model)
np.save(OUT + ".npy", G.sample_batch(runner, model, sampler, 8, 20260805))
print("saved", OUT + ".npy", "cudnn.benchmark =", torch.backends.cudnn.benchmark)
