"""Does the static linear delta table shrink L1's latency penalty? Timed with and without it.

Step 1 of the L1 fusion plan removes `delta_absmax_fp16` from the 42 projections' hot path. The record
attributes +876 ms of the profiled window to that call (docs/OPEN_ITEMS.md on MODIFF_LINEAR), i.e. ~4.4
ms/step of L1's ~29.8 ms/step penalty. This measures it rather than trusting the attribution, because
that attribution was taken on `int4_linmodiff` = L1 + DYNAMIC delta and this arm is L1 + static.

L0 is timed too, as the reference the +49% was quoted against.

PROTOCOL. One process per arm, 2 discarded warm-up samples then 3 timed, CUDA events around a full
sample. 100 steps rather than 20: the e2e harness records that at 20 steps this family reported 132.0
ms/step against a true 99.73, because the 5 MoDiff warm-up rounds amortise over too few.

Run: python docs/w4a4_quality_2026-08-17/scripts/time_linear_delta.py <arm>
     arm in {l1_table, l1_notable, l0}
"""
import json
import os
import statistics
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

ARM = sys.argv[1]
BATCH, STEPS, WARM, REPS = 128, 100, 2, 3

import torch                                                                # noqa: E402
import kernel_suites_bench as ks                                            # noqa: E402
import integration.benchmarks.benchmark_ldm as B                            # noqa: E402

ks.set_env("int4")
os.environ["MODIFF_DELTA_MODE"] = "static"
os.environ["MODIFF_DELTA_REFRESH"] = "4"
os.environ["MODIFF_ACT_BITS"] = "8"
os.environ["MODIFF_WARMUP_STEPS"] = "5"
os.environ["MODIFF_LINEAR"] = "0" if ARM == "l0" else "1"
if ARM == "l1_notable":
    # point it at a path that does not exist -> the projections keep the per-call reduction
    os.environ["MODIFF_LINEAR_DELTA_TABLE"] = "/nonexistent/no_linear_delta.pt"

runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="/workspace/fid_det/tmp", batch_size=BATCH, steps=STEPS,
    shape=(4, 32, 32), calibration_path="integration/calibration/int4_calibration_realckpt.pt")
model, sampler = runner._setup_model("int4")

from ldm.models.diffusion.ddim import DDIMSampler                           # noqa: E402
from integration.kernels.wxax_linear import QuantLinearWxAx                 # noqa: E402

# NON-VACUITY: assert the arm is what its name says, from the MODULE not the environment.
lins = [m for m in model.model.diffusion_model.modules() if isinstance(m, QuantLinearWxAx)]
n_mod = sum(1 for m in lins if m.modiff)
n_tab = sum(1 for m in lins if getattr(m, "_delta_cal", False))
print(f"arm={ARM}: {len(lins)} wxax Linears, {n_mod} modulated, {n_tab} carrying a static delta table")
if ARM == "l1_table":
    assert n_mod > 0 and n_tab == n_mod, f"expected every modulated Linear to have a table, got {n_tab}/{n_mod}"
elif ARM == "l1_notable":
    assert n_mod > 0 and n_tab == 0, f"expected NO tables, got {n_tab}/{n_mod}"
else:
    assert n_mod == 0, f"L0 must have no modulated Linears, got {n_mod}"


def sample():
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False,
                       **runner._cond_kwargs(model, BATCH))


for _ in range(WARM):
    sample()
torch.cuda.synchronize()
times = []
for _ in range(REPS):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); sample(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
med = statistics.median(times)
out = {"arm": ARM, "batch": BATCH, "steps": STEPS, "ms_per_batch": med,
       "ms_per_step": med / STEPS, "all": times,
       "cv_pct": (statistics.stdev(times) / statistics.mean(times) * 100) if len(times) > 1 else 0.0}
print(f"\n{ARM}: {out['ms_per_step']:.2f} ms/step   (CV {out['cv_pct']:.2f}%, {REPS} repeats)")
p = f"/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad/time_{ARM}.json"
json.dump(out, open(p, "w"), indent=1)
print(f"wrote {p}")
