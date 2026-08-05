"""Does the per-step delta scale stop MoDiff from diverging? A/B on activation growth.

This is a better acceptance test than the step-size gain, and it came out of chasing the
"out_conv scale discrepancy". The discrepancy is not a calibration bug and not a SiLU mismatch (the
fused kernel applies silu(x) then smooth_inv then subtracts the cache, exactly matching what
calibration observes). It is DIVERGENCE: with the delta quantized on the full-activation grid --
i.e. the pre-2026-08-03 behaviour, where the step size is unchanged so MoDiff buys only error
feedback -- a_hat cannot track a, the residual compounds across timesteps, and the activations grow
far past the range everything was calibrated for.

Metric: for each conv, max over 20 steps of |input|, divided by the absmax calibration measured for
that same layer. 1.0 means the layer is running inside its calibrated range. Large means the
pipeline has walked out of it, so every downstream quantizer is clipping.

A/B: identical seeded network and identical activation calibration; the only difference is whether
the per-step delta-scale table is applied.
"""

import collections
import json
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks
from integration.kernels.int8_optimized import (
    OptimizedInt8Conv2d, apply_int8_delta_scales, reset_modiff_state)
from integration.utils import attention_identity_guard as guard

STEPS = int(os.environ.get("AB_STEPS", "20"))
BATCH = int(os.environ.get("AB_BATCH", "4"))
TABLE = "docs/modiff_correctness_2026-08-03/data/int8_delta_calibration.pt"

OBS = collections.defaultdict(list)


def install():
    names = [n for n in dir(OptimizedInt8Conv2d)
             if n == "forward" or n.startswith("forward_") or n.startswith("_forward")]
    for nm in names:
        fn = getattr(OptimizedInt8Conv2d, nm)
        if not callable(fn):
            continue

        def mk(fn):
            def w(self, x, *a, **k):
                # fp inputs only: forward_from_int8 and friends take int8 CODES, and recording
                # those yields a spurious absmax of 127 that reads as a huge activation.
                if torch.is_tensor(x) and x.is_floating_point():
                    OBS[self.layer_name].append(float(x.detach().abs().max()))
                return fn(self, x, *a, **k)
            return w
        setattr(OptimizedInt8Conv2d, nm, mk(fn))


def run(apply_table):
    ks.set_env("int8")
    guard.seed_model_construction()
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=None)
    model, sampler = runner._setup_model("int8")
    runner._calibrate_int8(model, sampler, num_runs=2)

    n_tab = 0
    if apply_table:
        n_tab = apply_int8_delta_scales(model, torch.load(TABLE, weights_only=True))

    convs = {c.layer_name: c for c in model.modules() if isinstance(c, OptimizedInt8Conv2d)}
    cal = {n: 127.0 / float(c.static_input_scale) for n, c in convs.items()}

    OBS.clear()
    reset_modiff_state(model)
    cond = runner._cond_kwargs(model, BATCH)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                             verbose=False, **cond)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    lat = lat.detach().float().cpu()

    res = {}
    for fam in ("in_conv", "out_conv"):
        sel = [n for n in OBS if n.endswith(fam) and OBS[n]]
        if sel:
            res[fam] = {
                "n": len(sel),
                "ratio_median": st.median(max(OBS[n]) / max(cal[n], 1e-12) for n in sel),
                "ratio_max": max(max(OBS[n]) / max(cal[n], 1e-12) for n in sel),
                "runtime_median": st.median(max(OBS[n]) for n in sel),
            }
    res["table_layers"] = n_tab
    res["latent_absmax"] = float(lat.abs().max())
    res["latent_finite"] = bool(torch.isfinite(lat).all())
    del model, sampler, runner
    torch.cuda.empty_cache()
    return res


def main():
    install()
    out = {}
    for tag, use in (("delta_scale_OFF (full-activation grid)", False),
                     ("delta_scale_ON  (per-step delta grid)", True)):
        r = run(use)
        out[tag] = r
        print(f"\n### {tag}")
        if r.get("table_layers"):
            print(f"   delta table applied to {r['table_layers']} layers")
        for fam in ("in_conv", "out_conv"):
            if fam in r:
                d = r[fam]
                print(f"   {fam:<9} n={d['n']:2d}  runtime |x| median {d['runtime_median']:.4g}  "
                      f"vs-calibrated-range median {d['ratio_median']:.1f}x  max {d['ratio_max']:.0f}x")
        print(f"   latent absmax {r['latent_absmax']:.4g}  finite={r['latent_finite']}")

    a, b = out["delta_scale_OFF (full-activation grid)"], out["delta_scale_ON  (per-step delta grid)"]
    print(f"\n{'=' * 70}\nEFFECT OF THE PER-STEP DELTA SCALE\n{'=' * 70}")
    for fam in ("in_conv", "out_conv"):
        if fam in a and fam in b:
            print(f"   {fam:<9} out-of-range factor {a[fam]['ratio_median']:.1f}x -> "
                  f"{b[fam]['ratio_median']:.1f}x   "
                  f"({a[fam]['ratio_median'] / max(b[fam]['ratio_median'], 1e-12):.1f}x better)")
    with open("docs/modiff_correctness_2026-08-03/data/divergence_ab.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
