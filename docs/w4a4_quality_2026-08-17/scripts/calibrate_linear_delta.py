"""Produce the static per-step delta table for the 42 attention projections (L1 fusion plan, step 1).

WHY THIS ARTIFACT UNBLOCKS THE FUSION. Every modulated projection call currently runs
`delta_absmax_fp16`, a global reduction over `x - a_hat`. A global reduction cannot be fused with the
kernel that consumes its result -- the whole tensor must be reduced before the first element can be
quantized. That is exactly why the CONV path can fuse GroupNorm+SiLU+delta-quantize into one kernel
(`group_norm_silu_delta_quantize_pack_nhwc`) and this path cannot: the conv path reads a static table.
So the un-fused quantize that costs L1 +2890 ms of the profiled window is a CONSEQUENCE of the missing
table, not an independent problem, and this script is the prerequisite for step 2.

It is also the prime suspect for docs/OPEN_ITEMS.md A18 (L1 is run-to-run nondeterministic at
4.5-6.2/255 where every L0 arm is bit-exact): `delta_absmax_fp16`'s `_retire` argument is the signature
of a last-block-retires grid reduction, which is order-dependent. NOTE the unit gate
(integration/tests/test_wxax_delta_table.py) could NOT reproduce A18 at [1024,192], so that hypothesis
is untested at model scale -- generating this table and re-running the noise floor is how it gets tested.

WHAT IS NOT CALIBRATED HERE. `LINEAR_DELTA_CLIP_RATIO` is seeded at the conv path's 8, which was swept
ON CONVS. The projections' delta is a different distribution and the conv sweep found a steep U
(relL2 .4945 at ratio 1, .1147 at 8, .3117 at 32), so being on the wrong side is expensive. Sweep it
with `MODIFF_LINEAR_DELTA_TABLE_RATIO` (which rescales a loaded table, so no re-export is needed) and
with FID -- OPEN_ITEMS A19 shows the 16-image pixel screen cannot resolve effects this size.

Run: python docs/w4a4_quality_2026-08-17/scripts/calibrate_linear_delta.py [--bits 4] [--steps 50]
"""
import argparse
import os
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch                                                                # noqa: E402
from integration.utils.preflight import preflight, MODEL                    # noqa: E402
preflight(*MODEL, what="calibrate_linear_delta.py")

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=4, choices=(4, 8))
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--out", default=None)
a = ap.parse_args()
OUT = a.out or f"integration/calibration/int{a.bits}_linear_delta.pt"

import kernel_suites_bench as ks                                            # noqa: E402
import integration.benchmarks.benchmark_ldm as B                            # noqa: E402
from integration.kernels import wxax_linear as W                            # noqa: E402

mode = "int4" if a.bits == 4 else "int8"
ks.set_env(mode)
os.environ["MODIFF_DELTA_MODE"] = "static"
os.environ["MODIFF_DELTA_REFRESH"] = "4"
os.environ["MODIFF_LINEAR"] = "1"                 # L1 -- the arm the table is for
os.environ["MODIFF_ACT_BITS"] = "8"
os.environ["MODIFF_WARMUP_STEPS"] = "5"
# The table must not already be on disk, or _setup_model loads it and the observation pass would
# measure a model that is already reading the thing being produced.
if os.path.exists(OUT):
    os.rename(OUT, OUT + ".prev")
    print(f"moved existing {OUT} -> {OUT}.prev so this run observes the dynamic path")

runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="/workspace/fid_det/tmp", batch_size=a.batch, steps=a.steps,
    shape=(4, 32, 32), calibration_path=f"integration/calibration/int{a.bits}_calibration_realckpt.pt")
model, sampler = runner._setup_model(mode)
unet = model.model.diffusion_model

n_armed = W.begin_wxax_delta_calibration(unet)
assert n_armed > 0, ("0 modulated wxax Linears found. Either MODIFF_LINEAR did not reach the conversion "
                     "or this mode does not install them -- either way there is nothing to calibrate.")
print(f"armed {n_armed} modulated Linears; observing {a.steps} steps at batch {a.batch}")

from ldm.models.diffusion.ddim import DDIMSampler                           # noqa: E402
with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
    DDIMSampler(model).sample(S=a.steps, batch_size=a.batch, shape=runner.shape, eta=0.0,
                              verbose=False, **runner._cond_kwargs(model, a.batch))

n_set = W.end_wxax_delta_calibration(unet)
table = W.export_wxax_delta_scales(unet)
print(f"calibrated {n_set}/{n_armed}, exported {len(table)} tables")
assert n_set == n_armed, f"only {n_set}/{n_armed} layers produced a table -- a partial table would load "\
                         f"cleanly and leave the rest on the per-call reduction"
assert len(table) == n_armed, f"export dropped layers ({len(table)} of {n_armed})"

# Non-vacuity: a table of constants would load fine and mean nothing. The delta range varies over the
# DDIM trajectory, so a real table must too.
import statistics                                                            # noqa: E402
spreads = []
for k, v in table.items():
    seen = v[:a.steps]
    spreads.append(float(seen.max() / seen.min().clamp_min(1e-30)))
print(f"per-layer max/min scale over the {a.steps} observed steps: "
      f"median {statistics.median(spreads):.3f}, range {min(spreads):.3f}..{max(spreads):.3f}")
assert statistics.median(spreads) > 1.05, (
    f"the tables are nearly constant across steps (median max/min {statistics.median(spreads):.3f}). A "
    f"per-step table that does not vary per step is a scalar with extra steps -- either the observation "
    f"did not see distinct steps, or the delta range really is flat and a scalar should be used instead.")

torch.save(table, OUT)
print(f"\nWROTE {OUT}  ({len(table)} layers x {W.MODIFF_MAX_STEPS} steps)")
print(f"LINEAR_DELTA_CLIP_RATIO = {W.LINEAR_DELTA_CLIP_RATIO} (UNCALIBRATED for projections -- sweep it)")
