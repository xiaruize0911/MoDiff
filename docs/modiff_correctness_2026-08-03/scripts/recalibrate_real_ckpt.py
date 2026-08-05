"""Regenerate the static calibration against the REAL checkpoint, then measure latent fidelity.

Why this is mandatory before any FID: the shipped `integration/calibration/*.pt` were produced when
`models/ldm/lsun_churches256/model.ckpt` was an 856-byte stub, i.e. against a randomly-initialised
network. With the real weights installed they describe nothing. Measured with them loaded, latent
relative L2 vs fp16 was 0.88 (int8) and 3.02 (int4) -- catastrophic, not quantization error.

Writes new files next to the old ones rather than overwriting, so the originals stay inspectable.
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

STEPS = int(os.environ.get("RC_STEPS", "50"))
BATCH = int(os.environ.get("RC_BATCH", "8"))
RUNS = int(os.environ.get("RC_RUNS", "3"))
SEED = 1234
OUT = {"int8": "integration/calibration/int8_calibration_realckpt.pt",
       "int4": "integration/calibration/int4_calibration_realckpt.pt"}


def build(mode, calib_path):
    ks.set_env(mode)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib_path)
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def latent(runner, model, sampler):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cond = runner._cond_kwargs(model, BATCH)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                            verbose=False, **cond)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    # 1. Calibrate int8 and int4 against the real weights, writing to fresh files.
    for base, path in (("int8", OUT["int8"]), ("int4", OUT["int4"])):
        mode = base  # internal "int8"/"int4" == MoDiff ON; scales are shared with the baselines
        runner, model, sampler = build(mode, path)
        runner.calibration_path = path
        if base == "int8":
            runner._calibrate_int8(model, sampler, num_runs=RUNS)
        else:
            runner._calibrate_int4(model, sampler, num_runs=RUNS)
        del model, sampler, runner
        torch.cuda.empty_cache()
        print(f"  wrote {path}  ({os.path.getsize(path)/1e3:.0f} kB)", flush=True)

    # 2. fp16 reference, then each quantized mode on the FRESH scales.
    print(f"\n{'=' * 76}\nLatent fidelity vs fp16, real checkpoint, fresh calibration\n{'=' * 76}")
    r, m, s = build("fp16", None)
    ref = latent(r, m, s)
    del m, s, r; torch.cuda.empty_cache()
    print(f"  fp16 reference: |x|max {float(ref.abs().max()):.4f}")

    res = {}
    for mode in ("int8_baseline", "int8", "int4_baseline", "int4"):
        path = OUT["int4"] if "int4" in mode else OUT["int8"]
        r, m, s = build(mode, path)
        lat = latent(r, m, s)
        rel = float((lat - ref).norm() / ref.norm())
        res[mode] = {"rel_l2_vs_fp16": rel, "latent_absmax": float(lat.abs().max())}
        print(f"  {mode:16s} relL2 vs fp16 = {rel:.4f}   |x|max {float(lat.abs().max()):.4f}")
        del m, s, r; torch.cuda.empty_cache()

    print("\n  For reference, with the STALE (stub-derived) calibration these were:")
    print("    int8_baseline 0.8820   int4_baseline 3.0230")
    with open("docs/modiff_correctness_2026-08-03/data/recalibrated_fidelity.json", "w") as f:
        json.dump({"steps": STEPS, "batch": BATCH, "runs": RUNS, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
