"""Correct per-category GPU-time breakdown for fp16 vs int8, normalized so category sums
match the true per-step total. Profiles exactly N sampler steps in one window, divides by N.
Verifies total vs a separate wall measurement."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = 128; NP = 20

def cat(name):
    l = name.lower()
    if "softmax" in l or "scaled_dot" in l: return "attention softmax"
    if "wmma_tensorop_f16" in l or "bmm" in l: return "attention QK/AV bmm (fp16)"
    if "implicit" in l or "cudnn" in l or "scudnn" in l or ("conv" in l and "int" not in l): return "conv"
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l: return "qkv/proj int GEMM"
    if "s1688gemm" in l or "ampere_fp16" in l or "cutlass" in l and "conv" not in l: return "other fp16 GEMM"
    if "group_norm" in l or "gn_" in l: return "GroupNorm"
    if "quant" in l or "absmax" in l or "requant" in l: return "quantize/dequant"
    if "upsample" in l or "interpolate" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "store" in l: return "elementwise/copy"
    return "other"

def setup(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if quant else None
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/catprof",
                          batch_size=BATCH, steps=NP, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    return r, model, sampler, cond, quant

for mode in ["fp16", "int8_baseline"]:
    r, model, sampler, cond, quant = setup(mode)
    def smp(S):  # autocast fp16 ON for ALL modes (fixed 2026-07-20: was enabled=quant -> fp16 ran fp32/tf32)
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(20); torch.cuda.synchronize()
    ms = []
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.time(); smp(40); torch.cuda.synchronize(); ms.append((time.time() - t0) / 40 * 1000)
    wall = statistics.mean(ms)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        smp(NP)
    torch.cuda.synchronize()
    agg = {}
    for e in p.key_averages():
        if e.self_device_time_total > 0:
            agg[cat(e.key)] = agg.get(cat(e.key), 0.0) + e.self_device_time_total / NP / 1000
    tot = sum(agg.values())
    print(f"\n===== {mode}  wall {wall:.1f} ms/step | GPU-busy(sum cats) {tot:.1f} ms/step =====")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"   {v:7.1f} ms  {100*v/tot:5.1f}%  {k}")
    del model, sampler; torch.cuda.empty_cache()
