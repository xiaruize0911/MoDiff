"""Benchmark + profile the W8A8/W4A4 Linear-quantization feature (MODIFF_QUANT_LINEAR).
Four modes x {linear fp16, linear quant}, batch 32, heavy-warmup median wall + peak mem,
plus a per-operation GPU profile for one mode to show where the linear-quant time goes.
Emits data/linear_quant_speed.csv and prints a profile breakdown."""
import os, sys, time, importlib.util, statistics, gc
import torch
from torch.profiler import profile, ProfilerActivity
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
OUT = "/workspace/MoDiff/docs/comprehensive_benchmark_2026-07-16/data"

class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 32; args.steps = 12
args.linear_backend = "fp16"; args.calibration = None
MODES = ["int8_baseline", "int4_baseline", "int8", "int4"]

def run(model, sampler, runner, mode):
    cond = runner._cond_kwargs(model, args.batch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(mode != "fp32")):
        sampler.sample(S=args.steps, batch_size=args.batch_size, shape=runner.shape, eta=0.0, verbose=False, **cond)

def lbucket(name):
    l = name.lower()
    if "gemm_w8a8" in l or "gemm_w4a4" in l: return "linear int GEMM (ours)"
    if "quant_act" in l: return "linear act-quantize"
    if "cutlass" in l or "gemm" in l or "cublas" in l or "bmm" in l or "ampere" in l: return "other GEMM (conv/attn/fp16 linear)"
    if "conv" in l or "implicit" in l or "fprop" in l or "cudnn" in l or "scudnn" in l or "wgrad" in l: return "conv"
    if "flash" in l or "softmax" in l or "attention" in l or "scaled_dot" in l: return "attention"
    if "group_norm" in l or "groupnorm" in l or "norm" in l: return "groupnorm"
    if "quant" in l or "dequant" in l or "absmax" in l or "step1" in l or "o_hat" in l or "pack" in l: return "conv quant/modiff"
    return "elementwise/other"

def measure(mode, ql):
    os.environ["MODIFF_QUANT_LINEAR"] = ql
    torch.cuda.empty_cache(); gc.collect()
    runner, model, sampler = abb.build(mode, args)
    tw = time.time()
    while time.time() - tw < 6.0: run(model, sampler, runner, mode)   # heavy warmup (clock ramp)
    torch.cuda.synchronize()
    walls = []
    for _ in range(12):
        torch.cuda.synchronize(); t0 = time.time(); run(model, sampler, runner, mode)
        torch.cuda.synchronize(); walls.append((time.time() - t0) / args.steps * 1000)
    torch.cuda.reset_peak_memory_stats(); run(model, sampler, runner, mode); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    prof = None
    if ql == "1":   # profile the quant path once, to show the linear-quant buckets
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(3): run(model, sampler, runner, mode)
            torch.cuda.synchronize()
        b = {}
        for e in pr.key_averages():
            dt = e.self_device_time_total
            if dt > 0: b[lbucket(e.key)] = b.get(lbucket(e.key), 0.0) + dt
        prof = {k: v / (3 * args.steps) / 1e3 for k, v in b.items()}
    del runner, model, sampler; torch.cuda.empty_cache(); gc.collect()
    return statistics.median(walls), peak, prof

rows = []
for m in MODES:
    w0, p0, _ = measure(m, "0")
    w1, p1, prof = measure(m, "1")
    rows.append((m, w0, w1, w0 / w1, p0, p1))
    print(f"{m:14s} fp16-lin {w0:6.2f}  quant-lin {w1:6.2f} ms/step  {w0/w1:.3f}x  peak {p0:.0f}->{p1:.0f} MiB", flush=True)
    if prof:
        print("   profile (quant-lin, ms/step):", flush=True)
        for k, v in sorted(prof.items(), key=lambda x: -x[1]):
            print(f"      {k:32s} {v:6.2f}", flush=True)

import csv
with open(f"{OUT}/linear_quant_speed.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["mode", "fp16lin_ms", "quantlin_ms", "speedup", "fp16lin_peakMiB", "quantlin_peakMiB"])
    wr.writerows(rows)
print("\nWROTE linear_quant_speed.csv", flush=True)
