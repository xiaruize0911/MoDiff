"""E2E item 3 — per-component timing profile (measured GPU time) across the 5 modes, b128.

torch.profiler (CUDA activity) over profiled steps; sum self_device_time_total per kernel, bucket by
name via cat() (flash-aware), normalize to ms/step. Also records an independent wall ms/step and
gpu_busy = sum(buckets); wall > gpu_busy indicates launch/CPU gaps. Writes data/e2e_timing_profile.csv
(one row per mode, columns = buckets + gpu_busy + wall). cat() mirrors detailed_profile.py.
"""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, STEPS, NP = 50, 25, 6           # profile NP x STEPS steps; per-step normalization
HERE = "docs/benchmark_5mode_2026-07-23"
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]
BUCKETS = ["attention", "conv (int GEMM)", "qkv/proj int GEMM", "attn bmm (fp16)", "other fp16 GEMM",
           "GroupNorm", "quantize/dequant", "modiff cache", "upsample/concat", "elementwise/copy", "other"]


def cat(name):
    l = name.lower()
    if "softmax" in l or "scaled_dot" in l or "flash" in l: return "attention"
    if "implicit" in l or "cudnn" in l or "fprop" in l or "conv2d" in l or "scudnn" in l or "wgrad" in l or "convolution" in l: return "conv (int GEMM)"
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l: return "qkv/proj int GEMM"
    if "wmma_tensorop_f16" in l or "bmm" in l: return "attn bmm (fp16)"
    if "cutlass" in l or "cublas" in l or "ampere_fp16" in l or "s1688gemm" in l: return "other fp16 GEMM"
    if "group_norm" in l or "groupnorm" in l or "gn_" in l or "fused_gn" in l: return "GroupNorm"
    if "scale_accumulate" in l or "o_hat" in l or "dequant_accumulate" in l: return "modiff cache"
    if "quant" in l or "dequant" in l or "requant" in l or "absmax" in l or "ahat" in l or "step1" in l or "aq_" in l or "pack" in l: return "quantize/dequant"
    if "upsample" in l or "interpolate" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "index" in l or "store" in l or "clamp" in l or "round" in l: return "elementwise/copy"
    return "other"


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/bench5mode",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    # independent wall ms/step (min of 3)
    wall = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time(); smp(STEPS); torch.cuda.synchronize()
        wall.append((time.time() - t0) / STEPS * 1000)
    wall = min(wall)
    # profiled per-component GPU time
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        smp(NP * STEPS)
    torch.cuda.synchronize()
    agg = {b: 0.0 for b in BUCKETS}
    for e in p.key_averages():
        if e.self_device_time_total > 0:
            agg[cat(e.key)] += e.self_device_time_total / (NP * STEPS) / 1000.0   # ms/step
    del model, sampler; torch.cuda.empty_cache()
    return agg, wall


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []
print(f"E2E per-component timing profile @ b{BATCH} (ms/step, measured GPU time)\n")
for (label, mode) in VERS:
    agg, wall = run(mode)
    gpu = sum(agg.values())
    row = dict(mode=label, **{b: round(agg[b], 3) for b in BUCKETS}, gpu_busy=round(gpu, 3), wall=round(wall, 3))
    rows.append(row)
    top = sorted(agg.items(), key=lambda x: -x[1])[:5]
    print(f"{label:16} gpu_busy={gpu:6.2f} wall={wall:6.2f} ms/step | top: " +
          ", ".join(f"{k} {v:.1f}" for k, v in top))

with open(f"{HERE}/data/e2e_timing_profile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nWROTE {HERE}/data/e2e_timing_profile.csv")
