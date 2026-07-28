"""Measure the REAL GPU-busy fraction per mode: how much of a step the GPU spends actually
executing kernels, vs sitting idle in kernel-launch gaps.

Why this needs care. The obvious computation -- sum(kernel self-times) / wall-clock of the
PROFILED window -- is wrong, because torch.profiler adds per-launch overhead, so it inflates
the denominator. That artifact is easy to mistake for real idle: it scales with kernel COUNT,
so it makes int4 (81 kernels) look far more idle than fp16 (78) purely from instrumentation.

The kernel self-times themselves are fine (CUPTI measures them on-device). So the correct
denominator is the UNPROFILED wall clock, which profile_tree.py already measured (its
`ms_step`, from 5x150 un-instrumented steps). This script only re-does the cheap profiling
pass and divides by that, reusing profile_tree.json's ms_step.

Writes data/gpu_busy_fraction.json.
"""
import os, sys, json
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

HERE = "docs/final_report_2026-07-28"
BATCH, WARMUP, PROF_STEPS = 128, 20, 20
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]


def run(mode, unprofiled_ms_step):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    for k in ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND"):
        os.environ.pop(k, None)
    calib = ("integration/calibration/int8_calibration.pt" if "int8" in mode else
             "integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=PROF_STEPS, shape=(4, 32, 32),
                          calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode)
    cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    total_us = 0.0
    n_launch = 0
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        if evt.self_device_time_total > 0:
            total_us += evt.self_device_time_total
            n_launch += evt.count
    gpu_us_per_step = total_us / PROF_STEPS
    launches_per_step = n_launch / PROF_STEPS
    busy = gpu_us_per_step / (unprofiled_ms_step * 1000.0)
    del model, sampler, prof
    torch.cuda.empty_cache()
    return dict(gpu_us_per_step=round(gpu_us_per_step, 1),
                unprofiled_ms_step=unprofiled_ms_step,
                gpu_busy_frac=round(busy, 4),
                idle_ms_per_step=round(unprofiled_ms_step - gpu_us_per_step / 1000.0, 3),
                kernel_launches_per_step=round(launches_per_step, 1))


def main():
    tree = json.load(open(f"{HERE}/data/profile_tree.json"))
    bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    out = {}
    print(f"{'mode':16} {'GPU us/step':>12} {'wall ms/step':>13} {'busy':>7} "
          f"{'idle ms':>8} {'launches/step':>14}")
    for label, mode in VERS:
        res = run(mode, tree[label]["ms_step"])
        out[label] = res
        print(f"{label:16} {res['gpu_us_per_step']:12.0f} {res['unprofiled_ms_step']:13.2f} "
              f"{res['gpu_busy_frac']*100:6.1f}% {res['idle_ms_per_step']:8.2f} "
              f"{res['kernel_launches_per_step']:14.0f}")
    with open(f"{HERE}/data/gpu_busy_fraction.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {HERE}/data/gpu_busy_fraction.json")


if __name__ == "__main__":
    main()
