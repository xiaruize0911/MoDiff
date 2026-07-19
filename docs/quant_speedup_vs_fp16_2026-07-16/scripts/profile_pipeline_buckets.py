"""Fresh where-does-the-time-go profile of the CURRENT pipeline (Linear-quantize fusions in),
to pick the next optimization target. torch.profiler device self-time, bucketed by the report's
canonical categories, for fp16 vs int8 at batch 64. Emits:
  data/pipeline_buckets_b{BATCH}.csv    (mode, bucket, ms_step)
  data/pipeline_topkernels_b{BATCH}.csv (mode, kernel, ms_step, bucket)
  data/pipeline_speed_b{BATCH}.csv      (mode, wall_ms_step, gpu_busy_ms_step)
"""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = int(os.environ.get("E2E_BATCH", "64"))
STEPS, WARM_S, NP = 20, 3.0, 2
D = "docs/quant_speedup_vs_fp16_2026-07-16/data"


def bucket(name):   # verbatim from static_vs_dynamic/scripts/pipeline.py (report categories)
    l = name.lower()
    if "flash" in l or "fmha" in l or "scaled_dot" in l or "softmax" in l: return "attention (softmax)"
    if "group_norm" in l or "groupnorm" in l or "rowwisemoments" in l or "computefused" in l: return "GroupNorm"
    if ("gn_accum" in l or "gn_finalize" in l or "fused_gn" in l): return "GroupNorm"
    if "cudnn" in l or "implicit" in l or "fprop" in l or "scudnn" in l or "wgrad" in l or "conv2d" in l: return "conv (GEMM)"
    if "bmm_qk" in l or "bmm_av" in l: return "attn QKᵀ/AV (int GEMM)"
    if "gemm" in l or "cutlass" in l or "cublas" in l or "bmm" in l: return "qkv/proj GEMM (+ fp16 attn bmm)"
    if ("quantize" in l or "dequant" in l or "sub_absmax" in l or "delta" in l or "o_hat" in l
            or "step1" in l or "pack" in l or "absmax" in l or "aq_" in l): return "quantize / absmax"
    if "store" in l or "epilogue" in l or "requant" in l: return "conv store epilogue"
    if "upsample" in l or "catarray" in l.replace("_", "") or "cat_" in l: return "upsample / concat"
    if ("elementwise" in l or "vectorized" in l or "functor" in l or "silu" in l
            or "copy" in l or "fill" in l or "index" in l or "add" in l): return "elementwise / copy"
    return "other"


def run_mode(mode, label):
    # mode = actual _setup_model mode; label = CSV key. Use "int8_baseline" (same int8 kernels,
    # static scales, NO MoDiff a_hat/o_hat temporal caching) — the baseline we care about.
    quant = "int8" in mode or "int4" in mode
    if quant: os.environ["MODIFF_QUANT_LINEAR"] = "1"
    else: os.environ.pop("MODIFF_QUANT_LINEAR", None)
    calib = "integration/calibration/int8_calibration.pt" if quant else None
    runner = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                               "models/ldm/lsun_churches256/model.ckpt",
                               output_dir="integration/results/prof_buckets", batch_size=BATCH,
                               steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                               linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = runner._setup_model(mode)
    cond = runner._cond_kwargs(model, BATCH)
    ac = True

    def sample(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=ac, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False, **cond)

    tw = time.time()                              # heavy warmup (hold boost clock, settle caches)
    while time.time() - tw < WARM_S: sample(STEPS)
    torch.cuda.synchronize()
    walls = []
    for _ in range(8):
        torch.cuda.synchronize(); t0 = time.time(); sample(STEPS)
        torch.cuda.synchronize(); walls.append((time.time() - t0) / STEPS * 1000)
    wall = min(walls)

    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(NP): sample(STEPS)
        torch.cuda.synchronize()
    buckets, kern = {}, {}
    for e in pr.key_averages():
        dt = e.self_device_time_total
        if dt <= 0: continue
        v = dt / (NP * STEPS) / 1e3               # ms/step
        buckets[bucket(e.key)] = buckets.get(bucket(e.key), 0.0) + v
        kern[e.key] = kern.get(e.key, 0.0) + v
    gpu = sum(buckets.values())
    print(f"\n===== {label} ({mode}) =====  wall(min)={wall:.2f} ms/step  GPU-busy={gpu:.2f} ms/step")
    for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"   {v:7.3f} ms/step  {v/gpu*100:5.1f}%  {b}")
    top = sorted(kern.items(), key=lambda kv: -kv[1])[:15]
    del model, sampler; torch.cuda.empty_cache()
    return wall, gpu, buckets, top


def main():
    brows, krows, srows = [], [], []
    for mode, label in (("fp16", "fp16"), ("int8_baseline", "int8")):   # int8 = baseline (no MoDiff caching)
        wall, gpu, buckets, top = run_mode(mode, label)
        srows.append({"mode": label, "wall_ms_step": round(wall, 3), "gpu_busy_ms_step": round(gpu, 3)})
        for b, v in buckets.items(): brows.append({"mode": label, "bucket": b, "ms_step": round(v, 4)})
        for k, v in top: krows.append({"mode": label, "kernel": k[:90], "ms_step": round(v, 4), "bucket": bucket(k)})
    def w(path, rows, cols):
        with open(path, "w", newline="") as f:
            c = csv.DictWriter(f, fieldnames=cols); c.writeheader(); c.writerows(rows)
    w(f"{D}/pipeline_buckets_b{BATCH}.csv", brows, ["mode", "bucket", "ms_step"])
    w(f"{D}/pipeline_topkernels_b{BATCH}.csv", krows, ["mode", "kernel", "ms_step", "bucket"])
    w(f"{D}/pipeline_speed_b{BATCH}.csv", srows, ["mode", "wall_ms_step", "gpu_busy_ms_step"])
    print(f"\nWROTE pipeline_buckets/topkernels/speed_b{BATCH}.csv")


if __name__ == "__main__":
    main()
