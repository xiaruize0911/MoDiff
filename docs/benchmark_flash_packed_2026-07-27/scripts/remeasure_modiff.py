"""Re-measure int8_modiff / int4_modiff only, after the GPU has cooled from the prior
sweep's thermal throttling (confirmed via nvidia-smi: 78C/1650MHz under load -> 71C/1740MHz
idle immediately after; round-by-round timing in run_profile2.log showed a monotonic
130->288ms climb across the 5 rounds for int8_modiff, the signature of thermal throttling,
not a real perf regression). Prints per-round ms and GPU temp/clock before and after each
mode so the numbers can be trusted this time.
"""
import os, sys, csv, json, time, statistics, subprocess
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

sys.path.insert(0, "docs/benchmark_flash_packed_2026-07-27/scripts")
from mode_speedup_and_profile import categorize, BATCH, WARMUP, TIMED, RUNS, PROF_STEPS, HERE

VERS = [("int8_modiff", "int8"), ("int4_modiff", "int4")]

def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return out

def run(mode):
    os.environ["MODIFF_QUANT_LINEAR"] = "1"; os.environ["MODIFF_QUANT_ATTN"] = "1"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ.pop("MODIFF_FLASH_PACKED", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else "integration/calibration/int4_calibration.pt"
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend="int_gemm")
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    print(f"    gpu before timed rounds: {gpu_state()}")
    ms = []
    for i in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        rt = (time.time() - t0) / TIMED * 1000
        ms.append(rt)
        print(f"    round {i}: {rt:.2f} ms/step   gpu: {gpu_state()}")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()

    cat_time, kernel_time = {}, {}
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total
        if t <= 0:
            continue
        cat = categorize(evt.key)
        cat_time[cat] = cat_time.get(cat, 0.0) + t
        kernel_time[evt.key] = kernel_time.get(evt.key, 0.0) + t
    total = sum(cat_time.values()) or 1.0
    cat_pct = {k: round(v / total * 100, 2) for k, v in cat_time.items()}
    top_kernels = sorted(kernel_time.items(), key=lambda x: -x[1])[:15]
    top_kernels = [dict(name=k, us_total=round(v, 1)) for k, v in top_kernels]

    del model, sampler, prof; torch.cuda.empty_cache()
    return statistics.mean(ms), statistics.median(ms), min(ms), cat_pct, top_kernels

print("cooldown check:", gpu_state())
results = {}
for label, mode in VERS:
    print(f"-- {label} --")
    mean, med, mn, cat_pct, top_k = run(mode)
    print(f"{label}: mean={mean:.2f} median={med:.2f} min={mn:.2f}")
    results[label] = dict(mean_ms=round(mean, 2), median_ms=round(med, 2), min_ms=round(mn, 2),
                          category_pct=cat_pct, top_kernels=top_k)
    time.sleep(20)  # brief cooldown between modes

with open(f"{HERE}/data/modiff_remeasure.json", "w") as f:
    json.dump(results, f, indent=2)
print("WROTE", f"{HERE}/data/modiff_remeasure.json")
