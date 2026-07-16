"""Ground-truth profile of the quantized attention/linear paths turned ON in the
default int8/int4 pipeline (per user request: turn current impl on, profile, optimize).

For int8_baseline and int4_baseline, measure 3 configs:
  A) default            : fp16 attention + fp16 qkv/proj (baseline)
  B) +QUANT_LINEAR (§7)  : qkv/proj -> W8A8/W4A4 gemm_wxax, attention fp16 math SDPA
  C) +QUANT_ATTN  (§6)   : int8 fused-flash score path (int8 modes only)

Reports GPU-busy ms/step + attention-relevant buckets + wall. Emits quant_attn_profile.csv."""
import os, sys, time, csv, importlib.util, statistics, gc
import torch
from torch.profiler import profile, ProfilerActivity
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
OUT = "/workspace/MoDiff/docs/comprehensive_benchmark_2026-07-15/data"

class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 32; args.steps = 12
args.linear_backend = "fp16"; args.calibration = None

def bucket(name):
    l = name.lower()
    if "flash" in l or "fmha" in l or "scaled_dot" in l or "softmax" in l: return "attention (softmax/flash)"
    if "group_norm" in l or "groupnorm" in l or "rowwisemoments" in l or "computefused" in l or "gn_" in l: return "GroupNorm"
    if "cudnn" in l or "implicit" in l or "fprop" in l or "scudnn" in l or "wgrad" in l or "conv2d" in l: return "conv (GEMM)"
    if "gemm_w" in l or "w8a8" in l or "w4a4" in l or "gemm_wxax" in l: return "quant linear GEMM (ours)"
    if "gemm" in l or "cutlass" in l or "cublas" in l or "bmm" in l: return "GEMM (qkv/proj + attn QK·AV)"
    if ("quantize" in l or "dequant" in l or "sub_absmax" in l or "delta" in l or "o_hat" in l
            or "step1" in l or "pack" in l or "absmax" in l or "quant_act" in l): return "quantize / delta"
    if "store" in l or "epilogue" in l or "requant" in l: return "conv store epilogue"
    if "upsample" in l or "cat" in l: return "upsample / concat"
    if ("elementwise" in l or "vectorized" in l or "functor" in l or "silu" in l
            or "copy" in l or "fill" in l or "index" in l or "add" in l): return "elementwise / copy"
    return "other"

def run(model, sampler, runner, mode):
    cond = runner._cond_kwargs(model, args.batch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(mode != "fp32")):
        sampler.sample(S=args.steps, batch_size=args.batch_size, shape=runner.shape,
                       eta=0.0, verbose=False, **cond)

def measure(mode, cfg):
    for k in ("MODIFF_QUANT_ATTN", "MODIFF_QUANT_LINEAR", "MODIFF_FUSE_GN_QKV"):
        os.environ.pop(k, None)
    if cfg == "lin":  os.environ["MODIFF_QUANT_LINEAR"] = "1"
    if cfg == "attn": os.environ["MODIFF_QUANT_ATTN"] = "1"
    torch.cuda.empty_cache(); gc.collect()
    runner, model, sampler = abb.build(mode, args)
    tw = time.time()
    while time.time() - tw < 6.0: run(model, sampler, runner, mode)
    torch.cuda.synchronize()
    walls = []
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.time(); run(model, sampler, runner, mode)
        torch.cuda.synchronize(); walls.append((time.time() - t0) / args.steps * 1000)
    wall = statistics.median(walls)
    torch.cuda.reset_peak_memory_stats(); run(model, sampler, runner, mode); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    NP = 3; buckets = {}
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(NP): run(model, sampler, runner, mode)
        torch.cuda.synchronize()
    for e in pr.key_averages():
        dt = e.self_device_time_total
        if dt <= 0: continue
        buckets[bucket(e.key)] = buckets.get(bucket(e.key), 0.0) + dt
    buckets = {k: v / (NP*args.steps) / 1e3 for k, v in buckets.items()}
    gpu = sum(buckets.values())
    del runner, model, sampler; torch.cuda.empty_cache(); gc.collect()
    return wall, gpu, peak, buckets

CONFIGS = {"int8_baseline": ["default", "lin", "attn"], "int4_baseline": ["default", "lin"]}
NAME = {"default": "A) default fp16 attn+lin", "lin": "B) +QUANT_LINEAR §7", "attn": "C) +QUANT_ATTN §6 flash"}
rows = []
for mode, cfgs in CONFIGS.items():
    print(f"\n########## {mode} ##########", flush=True)
    for cfg in cfgs:
        wall, gpu, peak, b = measure(mode, cfg)
        att = b.get("attention (softmax/flash)", 0.0)
        gm  = b.get("GEMM (qkv/proj + attn QK·AV)", 0.0)
        ql  = b.get("quant linear GEMM (ours)", 0.0)
        qz  = b.get("quantize / delta", 0.0)
        print(f"  {NAME[cfg]:28s} wall={wall:6.2f} GPU={gpu:6.2f} peak={peak:5.0f} | "
              f"attn(sm/flash)={att:5.2f} GEMM={gm:5.2f} qlinGEMM={ql:5.2f} quant={qz:5.2f}", flush=True)
        rows.append({"mode": mode, "cfg": cfg, "name": NAME[cfg], "wall_ms": round(wall,3),
                     "gpu_ms": round(gpu,3), "peak_MiB": round(peak,1),
                     "attn_flash_ms": round(att,3), "gemm_ms": round(gm,3),
                     "qlin_gemm_ms": round(ql,3), "quant_ms": round(qz,3)})
with open(f"{OUT}/quant_attn_profile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nWROTE quant_attn_profile.csv", flush=True)
