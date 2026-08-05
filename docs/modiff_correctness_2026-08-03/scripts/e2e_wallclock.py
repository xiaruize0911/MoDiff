"""D8: end-to-end wall-clock for all five modes, in one process, reported whatever it says.

The goal commits to reporting MoDiff's speed against a named baseline even when MoDiff loses, because
the â/ô cache traffic is intrinsic and the honest framing is quality-at-low-bits, not speed parity.

Method: one process so every mode shares a thermal state; `reset_modiff_state` before each timed
sample so MoDiff starts from t=T as a real run does; median of `REPEATS` profiler-free samples.

Configuration caveat, stated because it changes what the numbers mean: this loads the shipped
`integration/calibration/*.pt`, which is the configuration the published report used. Those scales do
not describe this tree's model (the stub checkpoint is re-randomised per process), so ~50% of even the
BASELINE activation quantizations clip. That is a property of the shipped setup, not of this
measurement -- but it means these are "the shipped configuration's speed", not "the speed of a
correctly calibrated model".
"""

import json
import os
import statistics as st
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks

BATCH = int(os.environ.get("E2E_BATCH", "128"))
STEPS = int(os.environ.get("E2E_STEPS", "200"))
REPEATS = int(os.environ.get("E2E_REPEATS", "3"))
WARM = 1
# (label, benchmark mode, MODIFF_DELTA_MODE). The delta mode is a separate axis from the
# benchmark mode: it only affects the two MoDiff rows, but it affects them a lot -- dynamic adds a
# reduction pass per modulated conv, which is exactly the cost this benchmark exists to price.
# (label, benchmark mode, MODIFF_DELTA_MODE, MODIFF_DELTA_REFRESH). refresh=1 forces the exact
# per-call scale; None uses the shipped default (4), which the staleness sweep showed is free.
MODES = [("fp16", "fp16", "static", None),
         ("int8_baseline", "int8_baseline", "static", None),
         ("int8 modiff static", "int8", "static", None),
         ("int8 dynamic K=1", "int8", "dynamic", 1),
         ("int8 dynamic K=4", "int8", "dynamic", 4),
         ("int4_baseline", "int4_baseline", "static", None),
         ("int4 modiff static", "int4", "static", None),
         ("int4 dynamic K=1", "int4", "dynamic", 1),
         ("int4 dynamic K=4", "int4", "dynamic", 4)]
# The REAL-checkpoint calibration. The un-suffixed files were produced against the 856-byte stub's
# random weights and give latent relL2 0.88/3.02 with real weights (see FINDINGS 2026-08-04) --
# they would not change the timing much, but a benchmark that loads invalid scales invites the
# reader to treat its quality numbers as valid too.
CALIB = {"int4_baseline": "integration/calibration/int4_calibration_realckpt.pt",
         "int4": "integration/calibration/int4_calibration_realckpt.pt"}
DEFAULT_CALIB = "integration/calibration/int8_calibration_realckpt.pt"


def run(mode, delta_mode="static", refresh=None):
    ks.set_env(mode)
    os.environ["MODIFF_DELTA_MODE"] = delta_mode   # read in the conv wrappers' __init__
    if refresh is None:
        os.environ.pop("MODIFF_DELTA_REFRESH", None)
    else:
        os.environ["MODIFF_DELTA_REFRESH"] = str(refresh)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
        calibration_path=(None if mode == "fp16" else CALIB.get(mode, DEFAULT_CALIB)))
    model, sampler = runner._setup_model(mode)
    cond = runner._cond_kwargs(model, BATCH)
    from integration.kernels.int8_optimized import reset_modiff_state as r8
    from integration.kernels.int4_optimized import reset_modiff_state as r4

    def one():
        for r in (r8, r4):
            try:
                r(model.model.diffusion_model)
            except Exception:
                pass
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                           verbose=False, **cond)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(WARM):
        one()
    samples = [one() for _ in range(REPEATS)]
    del model, sampler, runner
    torch.cuda.empty_cache()
    return samples


def main():
    bn = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(8):
        bn = bn @ bn
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    out = {}
    for label, mode, dm, rf in MODES:
        s = run(mode, dm, rf)
        med = st.median(s)
        spread = (max(s) - min(s)) / med * 100.0
        out[label] = {"samples_ms": [round(x, 1) for x in s], "median_ms_batch": round(med, 1),
                      "ms_per_step": round(med / STEPS, 3), "spread_pct": round(spread, 1),
                      "mode": mode, "delta_mode": dm, "refresh": rf}
        print(f"{label:22s} median {med:9.1f} ms/batch   {med/STEPS:7.3f} ms/step   "
              f"spread {spread:4.1f}%   samples {[round(x) for x in s]}", flush=True)

    fp16 = out["fp16"]["median_ms_batch"]
    print(f"\n{'=' * 78}\nSpeedup vs FP16, and MoDiff's cost against its own baseline\n{'=' * 78}")
    for label, _, _, _ in MODES:
        print(f"  {label:22s} {fp16 / out[label]['median_ms_batch']:6.3f}x vs fp16")
    print()
    for mo, base in (("int8 modiff static", "int8_baseline"),
                     ("int8 dynamic K=1", "int8_baseline"),
                     ("int8 dynamic K=4", "int8_baseline"),
                     ("int4 modiff static", "int4_baseline"),
                     ("int4 dynamic K=1", "int4_baseline"),
                     ("int4 dynamic K=4", "int4_baseline")):
        a, b = out[mo]["median_ms_batch"], out[base]["median_ms_batch"]
        print(f"  {mo:22s} vs {base:16s} {b / a:.3f}x  ({a - b:+.0f} ms/batch, "
              f"{(a - b) / STEPS:+.3f} ms/step)")
    for bits in ("int8", "int4"):
        st_ = out[f"{bits} modiff static"]["median_ms_batch"]
        k1 = out[f"{bits} dynamic K=1"]["median_ms_batch"]
        k4 = out[f"{bits} dynamic K=4"]["median_ms_batch"]
        print(f"  {bits}: the absmax reduction pass costs {(k1 - st_) / STEPS:+.3f} ms/step at K=1 "
              f"and {(k4 - st_) / STEPS:+.3f} at K=4 "
              f"(K=4 recovers {(k1 - k4) / STEPS:.3f} ms/step)")
    print(f"\n  Reported spread is (max-min)/median over {REPEATS} repeats after "
          f"{WARM} warm-up run(s); treat differences smaller than it as noise.")
    out["_config"] = {"batch": BATCH, "steps": STEPS, "repeats": REPEATS,
                      "gpu": torch.cuda.get_device_name(0)}
    with open("docs/modiff_correctness_2026-08-03/data/e2e_wallclock.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
