"""Live absmax calibration on the real LSUN-churches checkpoint.

Writes the four files the kernel path actually loads when Q-Diffusion tables
are absent:

  integration/calibration/int8_calibration_realckpt.pt
  integration/calibration/int4_calibration_realckpt.pt
  integration/calibration/int8_delta_calibration.pt
  integration/calibration/int4_delta_calibration.pt

Activation = two-round live absmax (conv + linear). Delta = one static
observation pass at act_scale/4 (cannot clip). Not Q-Diffusion.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks
from integration.kernels.int8_optimized import (
    begin_delta_calibration_int8, delta_calibration_report_int8,
    end_delta_calibration_int8, export_int8_delta_scales, reset_modiff_state)
from integration.kernels.int4_optimized import (
    begin_delta_calibration_int4, delta_calibration_report_int4,
    end_delta_calibration_int4, export_int4_delta_scales)

STEPS = int(os.environ.get("CALIB_STEPS", "50"))
BATCH = int(os.environ.get("CALIB_BATCH", "8"))
ACT_RUNS = int(os.environ.get("CALIB_ACT_RUNS", "5"))
SEED = 1234

ACT = {"int8": "integration/calibration/int8_calibration_realckpt.pt",
       "int4": "integration/calibration/int4_calibration_realckpt.pt"}
DELTA = {"int8": "integration/calibration/int8_delta_calibration.pt",
         "int4": "integration/calibration/int4_delta_calibration.pt"}


def build(mode, calib_path):
    ks.set_env(mode)
    os.environ["MODIFF_DELTA_MODE"] = "static"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_fake_quant_2026-08-27/tmp_calib",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
        calibration_path=calib_path, auto_delta_table=False)
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def sample(runner, model, sampler):
    reset_modiff_state(model.model.diffusion_model)
    try:
        from integration.kernels.int4_optimized import reset_modiff_state as reset4
        reset4(model.model.diffusion_model)
    except Exception:
        pass
    B._reset_wxax_modiff_safe(model)
    cond = runner._cond_kwargs(model, BATCH)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                       verbose=False, **cond)
    torch.cuda.synchronize()


def calibrate_activation(bits: str):
    path = ACT[bits]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"\n===== activation calib {bits} -> {path} =====", flush=True)
    runner, model, sampler = build(bits, path)
    runner.calibration_path = path
    if bits == "int8":
        runner._calibrate_int8(model, sampler, num_runs=ACT_RUNS, refine_rounds=1)
    else:
        runner._calibrate_int4(model, sampler, num_runs=ACT_RUNS)
    del model, sampler, runner
    torch.cuda.empty_cache()
    print(f"  wrote {path} ({os.path.getsize(path) / 1e3:.1f} kB)", flush=True)


def calibrate_delta(bits: str):
    path = DELTA[bits]
    print(f"\n===== delta calib {bits} -> {path} =====", flush=True)
    runner, model, sampler = build(bits, ACT[bits])
    unet = model.model.diffusion_model
    if bits == "int8":
        n = begin_delta_calibration_int8(unet, reset=True)
        sample(runner, model, sampler)
        m = end_delta_calibration_int8(unet)
        rep = delta_calibration_report_int8(unet)
        table = export_int8_delta_scales(unet)
    else:
        n = begin_delta_calibration_int4(unet, reset=True)
        sample(runner, model, sampler)
        m = end_delta_calibration_int4(unet)
        rep = delta_calibration_report_int4(unet)
        table = export_int4_delta_scales(unet)
    gains = sorted(x["step_gain_tail"] for x in rep if x.get("step_gain_tail"))
    clip = [x["obs_clipped_frac"] for x in rep if x.get("obs_clipped_frac") is not None]
    torch.save(table, path)
    print(f"  armed {n}, tables set {m}/{len(table)}; "
          f"gain median {gains[len(gains)//2]:.1f}x min {gains[0]:.1f}x max {gains[-1]:.1f}x; "
          f"clipping {sum(1 for c in clip if c > 0)}/{len(clip)}", flush=True)
    print(f"  wrote {path} ({os.path.getsize(path) / 1e3:.1f} kB)", flush=True)
    del model, sampler, runner
    torch.cuda.empty_cache()


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"steps={STEPS} batch={BATCH} act_runs={ACT_RUNS}")
    calibrate_activation("int8")
    calibrate_activation("int4")
    calibrate_delta("int8")
    calibrate_delta("int4")
    print("\n===== all four calibration files =====")
    for p in list(ACT.values()) + list(DELTA.values()):
        print(f"  {p}: {os.path.getsize(p)/1e3:.1f} kB")


if __name__ == "__main__":
    main()
