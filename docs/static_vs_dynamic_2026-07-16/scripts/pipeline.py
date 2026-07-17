"""Static vs Dynamic quantization -- full 11-mode e2e pipeline (churches LDM UNet, batch 32).

Modes: fp32 + {fp16,int8,int4} x {dynamic,static} x {base,+MoDiff}. dynamic = every activation
statistic computed at runtime (uncalibrated conv/linear absmax + per-token/per-row attention);
static = calibrated constants everywhere (conv/linear static scales + static-c attention). Linear
runs int_gemm so it participates in the static/dynamic axis. Emits pipeline_speed.csv,
pipeline_io.csv, kernel_profile.csv."""
import os, sys, time, csv, importlib.util, statistics, gc
os.environ.setdefault("MODIFF_ATTN_CALIB_STEPS", "16")   # cover a full DDIM trajectory before freezing c
import torch
from torch.profiler import profile, ProfilerActivity
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int4_optimized import OptimizedInt4Conv2d
OUT = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"

class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 32; args.steps = 12
args.linear_backend = "int_gemm"; args.calibration = None      # int GEMM linear -> static/dynamic axis applies

MODES = [("fp32", "fp32"),
         ("dynamic_fp16", "fp16 dyn"), ("static_fp16", "fp16 static"),
         ("dynamic_int8", "int8 dyn"), ("static_int8", "int8 static"),
         ("dynamic_int8_modiff", "int8 dyn+MoDiff"), ("static_int8_modiff", "int8 static+MoDiff"),
         ("dynamic_int4", "int4 dyn"), ("static_int4", "int4 static"),
         ("dynamic_int4_modiff", "int4 dyn+MoDiff"), ("static_int4_modiff", "int4 static+MoDiff")]

def cache_mib(model):
    tot = 0
    for m in model.modules():
        if isinstance(m, (OptimizedInt8Conv2d, OptimizedInt4Conv2d)):
            for a in ("a_hat_cache", "o_hat_cache"):
                t = getattr(m, a, None)
                if torch.is_tensor(t): tot += t.numel() * t.element_size()
    return tot / (1024**2)

def bucket(name):
    l = name.lower()
    if "flash" in l or "fmha" in l or "scaled_dot" in l or "softmax" in l: return "attention (softmax)"
    if "group_norm" in l or "groupnorm" in l or "rowwisemoments" in l or "computefused" in l: return "GroupNorm"
    if ("gn_accum" in l or "gn_finalize" in l or "fused_gn" in l): return "GroupNorm"
    if "cudnn" in l or "implicit" in l or "fprop" in l or "scudnn" in l or "wgrad" in l or "conv2d" in l: return "conv (GEMM)"
    if "bmm_qk" in l or "bmm_av" in l: return "attn QKᵀ/AV (int GEMM)"   # int8/int4 attn matmuls (named)
    if "gemm" in l or "cutlass" in l or "cublas" in l or "bmm" in l: return "qkv/proj GEMM (+ fp16 attn bmm)"
    if ("quantize" in l or "dequant" in l or "sub_absmax" in l or "delta" in l or "o_hat" in l
            or "step1" in l or "pack" in l or "absmax" in l or "aq_" in l): return "quantize / absmax"
    if "store" in l or "epilogue" in l or "requant" in l: return "conv store epilogue"
    if "upsample" in l or "catarray" in l.replace("_", "") or "cat_" in l: return "upsample / concat"
    if ("elementwise" in l or "vectorized" in l or "functor" in l or "silu" in l
            or "copy" in l or "fill" in l or "index" in l or "add" in l): return "elementwise / copy"
    return "other"

def run(model, sampler, runner, mode):
    cond = runner._cond_kwargs(model, args.batch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(mode != "fp32")):
        sampler.sample(S=args.steps, batch_size=args.batch_size, shape=runner.shape, eta=0.0, verbose=False, **cond)

speed_rows, io_rows, prof_rows = [], [], []
for mode, label in MODES:
    print(f"\n===== {mode} ({label}) =====", flush=True)
    torch.cuda.empty_cache(); gc.collect()
    runner, model, sampler = abb.build(mode, args)
    tw = time.time()
    while time.time() - tw < 7.0:                # heavy warmup: hold boost clock + freeze static calibration
        run(model, sampler, runner, mode)
    torch.cuda.synchronize()
    walls = []
    for _ in range(12):
        torch.cuda.synchronize(); t0 = time.time()
        run(model, sampler, runner, mode)
        torch.cuda.synchronize(); walls.append((time.time() - t0) / args.steps * 1000)
    wall = statistics.median(walls); wmin = min(walls); wsd = statistics.pstdev(walls)
    torch.cuda.reset_peak_memory_stats()
    run(model, sampler, runner, mode); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    reserved = torch.cuda.max_memory_reserved() / (1024**2)
    cache = cache_mib(model)
    NP = 3
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(NP): run(model, sampler, runner, mode)
        torch.cuda.synchronize()
    buckets = {}
    for e in pr.key_averages():
        dt = e.self_device_time_total
        if dt <= 0: continue
        buckets[bucket(e.key)] = buckets.get(bucket(e.key), 0.0) + dt
    buckets = {k: v / (NP * args.steps) / 1e3 for k, v in buckets.items()}
    gpu = sum(buckets.values())
    print(f"  wall med={wall:.2f} min={wmin:.2f} sd={wsd:.2f} ms/step  GPU-busy={gpu:.2f}  peak={peak:.0f}MiB  cache={cache:.1f}MiB", flush=True)
    speed_rows.append({"mode": mode, "label": label, "wall_ms_step": round(wall, 3),
                       "wall_min_ms_step": round(wmin, 3), "wall_stdev": round(wsd, 3),
                       "gpu_busy_ms_step": round(gpu, 3), "overhead_ms_step": round(wall - gpu, 3)})
    io_rows.append({"mode": mode, "label": label, "peak_mem_MiB": round(peak, 1),
                    "reserved_MiB": round(reserved, 1), "modiff_cache_MiB": round(cache, 1)})
    for b, v in buckets.items():
        prof_rows.append({"mode": mode, "bucket": b, "ms_step": round(v, 4)})
    del runner, model, sampler; torch.cuda.empty_cache(); gc.collect()

def wcsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
wcsv(f"{OUT}/pipeline_speed.csv", speed_rows, ["mode", "label", "wall_ms_step", "wall_min_ms_step", "wall_stdev", "gpu_busy_ms_step", "overhead_ms_step"])
wcsv(f"{OUT}/pipeline_io.csv", io_rows, ["mode", "label", "peak_mem_MiB", "reserved_MiB", "modiff_cache_MiB"])
wcsv(f"{OUT}/kernel_profile.csv", prof_rows, ["mode", "bucket", "ms_step"])
print("\nWROTE pipeline_speed.csv, pipeline_io.csv, kernel_profile.csv")
