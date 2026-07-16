"""F4 speed/memory: fused qkv->flash vs fp16 attention vs §6 per-token flash, int8_baseline,
batch 32. Noise-controlled: >=6s warmup, 12 back-to-back timed runs (median/min/stdev),
GPU-busy (throttle-robust), peak mem. Flags any config with wall stdev > 1 ms."""
import os, sys, time, importlib.util, statistics, gc
import torch
from torch.profiler import profile, ProfilerActivity
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)

class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 32; args.steps = 12
args.linear_backend = "fp16"; args.calibration = None
MODE = "int8_baseline"

def bucket(nm):
    l = nm.lower()
    if "transpose_qkv" in l: return "qkv transpose (int8)"
    if "flash" in l or "fmha" in l or "scaled_dot" in l or "softmax" in l: return "attention (softmax/flash)"
    if "gemm_w8a8" in l or "gemm_w4a4" in l: return "qkv int-out GEMM (ours)"
    if "quant_act" in l or "quantize_qkv" in l or "faq_" in l: return "qkv quantize"
    if "cudnn" in l or "implicit" in l or "fprop" in l or "conv2d" in l or "scudnn" in l: return "conv"
    if "gemm" in l or "cutlass" in l or "cublas" in l or "bmm" in l: return "GEMM (qkv/proj+QK·AV)"
    return "other"

def run(model, sampler, runner):
    cond = runner._cond_kwargs(model, args.batch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        sampler.sample(S=args.steps, batch_size=args.batch_size, shape=runner.shape, eta=0.0, verbose=False, **cond)

def measure(name, env):
    for k in ("MODIFF_QUANT_ATTN", "MODIFF_QKV_FLASH_FUSED", "MODIFF_FLASH_MIN_T"):
        os.environ.pop(k, None)
    os.environ.update(env)
    torch.cuda.empty_cache(); gc.collect()
    runner, model, sampler = abb.build(MODE, args)
    tw = time.time()
    while time.time() - tw < 7.0: run(model, sampler, runner)   # warm + finish calibration
    torch.cuda.synchronize()
    walls = []
    for _ in range(12):
        torch.cuda.synchronize(); t0 = time.time(); run(model, sampler, runner)
        torch.cuda.synchronize(); walls.append((time.time() - t0) / args.steps * 1000)
    wall, wmin, wsd = statistics.median(walls), min(walls), statistics.pstdev(walls)
    torch.cuda.reset_peak_memory_stats(); run(model, sampler, runner); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    NP = 3; buckets = {}
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(NP): run(model, sampler, runner)
        torch.cuda.synchronize()
    for e in pr.key_averages():
        if e.self_device_time_total > 0:
            buckets[bucket(e.key)] = buckets.get(bucket(e.key), 0.0) + e.self_device_time_total
    gpu = sum(buckets.values()) / (NP * args.steps) / 1e3
    att = buckets.get("attention (softmax/flash)", 0)/(NP*args.steps)/1e3
    qg = buckets.get("qkv int-out GEMM (ours)", 0)/(NP*args.steps)/1e3
    tr = buckets.get("qkv transpose (int8)", 0)/(NP*args.steps)/1e3
    flag = "  <-- NOISY" if wsd > 1.0 else ""
    print(f"{name:26s} wall={wall:6.2f} (min {wmin:6.2f} sd {wsd:.2f}) GPU={gpu:6.2f} peak={peak:5.0f} | "
          f"attn={att:5.2f} qkvGEMM={qg:4.2f} transpose={tr:4.2f}{flag}", flush=True)
    del runner, model, sampler; torch.cuda.empty_cache(); gc.collect()
    return dict(name=name, wall=wall, wmin=wmin, wsd=wsd, gpu=gpu, peak=peak)

CONFIGS = [
    ("fp16 attention", {}),
    ("§6 per-token int8 flash", {"MODIFF_QUANT_ATTN": "1"}),
    ("fused int8 (W8A8->flash)", {"MODIFF_QUANT_ATTN": "1", "MODIFF_QKV_FLASH_FUSED": "8"}),
    ("fused int4 (W4A4->flash)", {"MODIFF_QUANT_ATTN": "1", "MODIFF_QKV_FLASH_FUSED": "4"}),
]
rows = [measure(n, e) for n, e in CONFIGS]
import csv
with open("/tmp/claude-0/-workspace/1150c54c-9325-4a0c-8e13-9708345f7905/scratchpad/f4.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE", flush=True)
