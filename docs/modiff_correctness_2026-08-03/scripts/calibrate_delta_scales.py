"""Two-round self-consistent MoDiff delta-scale calibration, plus the Stage-1 acceptance test.

Why two rounds: the delta a_t - a_hat_{t+1} depends on the scale used to build a_hat, so the
observation is a fixed point. Round 0 observes with the full-activation scale (which cannot clip,
since |delta| <= activation range -- that is what makes this rebuild-free). Round 1 re-observes on
round 0's grid, where the codes land near Q and the range estimate is ~1% accurate.

Acceptance test (paper Theorem 4.3, ||x - Q(x)||^2 <= s^2 d): report the step-size gain
delta_scale/activation_scale per layer. A gain of g means the squared quantization error on the
delta falls by g^2 relative to what the tree did before. Before this change the gain was exactly
1.0 by construction, i.e. MoDiff bought no error reduction at all.

Run: python docs/modiff_correctness_2026-08-03/scripts/calibrate_delta_scales.py
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks
from integration.utils import attention_identity_guard as guard
from integration.kernels.int8_optimized import (
    begin_delta_calibration_int8, delta_calibration_report_int8, end_delta_calibration_int8,
    export_int8_delta_scales, reset_modiff_state)

MODE = os.environ.get("DC_MODE", "int8")          # internal name: MoDiff ON
STEPS = int(os.environ.get("DC_STEPS", "20"))     # must divide 1000
BATCH = int(os.environ.get("DC_BATCH", "4"))
ROUNDS = int(os.environ.get("DC_ROUNDS", "6"))   # >2: clipped layers need a geometric backoff
OUT = f"docs/modiff_correctness_2026-08-03/data/{'int4' if 'int4' in MODE else 'int8'}_delta_calibration.pt"


def main():
    ks.set_env(MODE)
    # Two things are load-bearing here and both come from the stub checkpoint:
    #
    # 1. SEED the construction. The 856-byte stub has an empty state_dict, so every weight comes
    #    from default-init off torch's global RNG -- which torch seeds nondeterministically per
    #    process. Unseeded, this is a DIFFERENT random network every run.
    # 2. Do NOT load integration/calibration/*.pt. Because of (1) those scales were calibrated
    #    against some other random network, so they do not describe this one. Measured 2026-08-03
    #    with the file loaded: the emitted int8 codes saturate at +-127 on ~50% of calls for the
    #    plain activation quantize and ~97% for the delta quantize, which makes the delta's range
    #    unobservable (everything reads as "as large as the activation") and makes the baseline
    #    itself run clipped. Passing calibration_path=None runs _calibrate_int8 in-process against
    #    the network actually being used.
    guard.seed_model_construction()
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
        calibration_path=None)
    model, sampler = runner._setup_model(MODE)
    # _setup_model only *loads* a conv calibration file; the calibration pass itself lives in
    # run_mode (benchmark_ldm.py:1002), which no harness that drives _setup_model directly ever
    # calls. So with calibration_path=None the convs stay is_calibrated=False, keep fp32 a_hat/o_hat
    # caches, and the fp32 output then trips the attention proj's `residual fp16 [M,n_out]` check.
    # Run the conv calibration explicitly against THIS network.
    # refine_rounds=1: round 0 necessarily observes the uncalibrated path, whose activations differ
    # from the calibrated path production runs. Measured effect on in_conv: utilisation 170 -> 124.
    runner._calibrate_int8(model, sampler, num_runs=int(os.environ.get("DC_CALIB_RUNS", "2")),
                           refine_rounds=1)
    cond = runner._cond_kwargs(model, BATCH)

    def sample():
        # reset_modiff_state zeroes a_hat/o_hat and step_count, so every round starts at t=T with
        # a_hat seeded by Q(a_T) -- the same state production runs start from.
        reset_modiff_state(model)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                           verbose=False, **cond)
        torch.cuda.synchronize()

    for r in range(ROUNDS):
        n = begin_delta_calibration_int8(model, reset=(r == 0))
        sample()
        m = end_delta_calibration_int8(model)
        rep = delta_calibration_report_int8(model)
        gains = sorted(x["step_gain_tail"] for x in rep if x.get("step_gain_tail"))
        clip = [x["obs_clipped_frac"] for x in rep if x.get("obs_clipped_frac") is not None]
        print(f"round {r}: observed {n}, table set on {m}; gain min {gains[0]:.1f}x "
              f"median {gains[len(gains)//2]:.1f}x max {gains[-1]:.1f}x; "
              f"layers still clipping {sum(1 for c in clip if c > 0)}/{len(clip)}")

    rep = delta_calibration_report_int8(model)
    table = export_int8_delta_scales(model)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(table, OUT)

    gains = [x["step_gain_tail"] for x in rep if x.get("step_gain_tail")]
    print(f"\n{'=' * 74}\nSTAGE-1 ACCEPTANCE: delta step-size gain over the activation grid\n{'=' * 74}")
    print(f"  layers calibrated        : {len(table)}")
    print(f"  gain (tail)  median      : {sorted(gains)[len(gains)//2]:.1f}x")
    print(f"  gain (tail)  min / max   : {min(gains):.1f}x / {max(gains):.1f}x")
    print(f"  => squared-error factor  : {sorted(gains)[len(gains)//2]**2:.0f}x lower (error ~ s^2)")
    print("\n  per-step scale shape, 3 example layers (step0 = Q(a_T) is NOT from this table):")
    for x in rep[:3]:
        if x.get("calibrated"):
            print(f"    {x['layer'][:52]:<52} act {x['activation_scale']:.4g}  "
                  f"step1 {x['delta_scale_step1']:.4g}  tail {x['delta_scale_tail']:.4g}")
    print(f"\nWROTE {OUT}")

    with open("docs/modiff_correctness_2026-08-03/data/delta_calibration.json", "w") as f:
        json.dump({"mode": MODE, "steps": STEPS, "batch": BATCH, "rounds": ROUNDS,
                   "report": rep}, f, indent=2)


if __name__ == "__main__":
    main()
