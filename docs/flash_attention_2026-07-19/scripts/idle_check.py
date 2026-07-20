"""Is the fp16 GPU-idle real, or a cross-harness artifact? Measure wall-clock AND GPU-busy
in the SAME process/model for fp16 and int8, over the SAME sampler.sample call. b128."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
def setup(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if quant else None
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/idle",
                          batch_size=BATCH, steps=20, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    return r, model, sampler, cond, quant

for mode in ["fp16", "int8_baseline"]:
    r, model, sampler, cond, quant = setup(mode)
    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=quant, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(20); torch.cuda.synchronize()                       # warmup
    # wall (CUDA events, no profiler), 3x40 steps
    ms = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time(); smp(40); torch.cuda.synchronize(); ms.append((time.time() - t0) / 40 * 1000)
    wall = statistics.mean(ms)
    # GPU-busy (profiler), 20 steps
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        smp(20)
    torch.cuda.synchronize()
    gpu = sum(e.self_device_time_total for e in p.key_averages() if e.self_device_time_total > 0) / 20 / 1000
    print(f"{mode:14}: wall {wall:7.2f} ms/step | GPU-busy {gpu:7.2f} ms/step | idle {wall-gpu:6.2f} ms ({(1-gpu/wall)*100:4.1f}%)")
    del model, sampler; torch.cuda.empty_cache()
