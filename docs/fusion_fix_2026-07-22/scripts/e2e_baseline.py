"""Focused e2e benchmark + profile for the BASELINE (cache-free static-quant) modes, b128.

int8_baseline (1 config) and int4_baseline measured with the attention-proj fold OFF vs ON
(MODIFF_FUSE_PROJ_I4, read per-call) to isolate today's int4-proj fix. Same wall methodology as
e2e_speed (30 warm + RUNS x 200 timed, min ms/step) + a torch.profiler per-component pass
(gpu_busy + buckets). Writes data/e2e_baseline_speed.csv and data/e2e_baseline_profile.csv.
"""
import os, sys, csv, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.update(MODIFF_QUANT_LINEAR="1", MODIFF_QUANT_ATTN="1", MODIFF_QUANT_ATTN_STATIC="1", MODIFF_LINEAR_OUT_I8="0")
os.environ.pop("MODIFF_FLASH_ATTN", None)
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH, WARMUP, TIMED, RUNS = 128, 30, 200, 3
PWARM, PS, NP = 30, 20, 4
HERE = "docs/fusion_fix_2026-07-22"
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
    if "scale_accumulate" in l or "o_hat" in l or "dequant_accumulate" in l or "accumulate_from_half" in l: return "modiff cache"
    if "quant" in l or "dequant" in l or "requant" in l or "absmax" in l or "ahat" in l or "step1" in l or "aq_" in l or "pack" in l: return "quantize/dequant"
    if "upsample" in l or "interpolate" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "index" in l or "store" in l or "clamp" in l or "round" in l: return "elementwise/copy"
    return "other"


def build(mode):
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else "integration/calibration/int4_calibration.pt"
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/fusionfix",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib, linear_backend="int_gemm")
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    return r, model, sampler, cond


def smp(sampler, r, cond, n):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=n, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)


def measure(label, r, model, sampler, cond):
    smp(sampler, r, cond, WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(sampler, r, cond, TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    wall = min(ms)
    smp(sampler, r, cond, PWARM); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        smp(sampler, r, cond, NP * PS)
    torch.cuda.synchronize()
    agg = {b: 0.0 for b in BUCKETS}
    for e in p.key_averages():
        if e.self_device_time_total > 0:
            agg[cat(e.key)] += e.self_device_time_total / (NP * PS) / 1000.0
    gpu = sum(agg.values())
    print(f"{label:28} wall={wall:7.2f} gpu_busy={gpu:7.2f} ms/step | conv={agg['conv (int GEMM)']:.1f} "
          f"quant={agg['quantize/dequant']:.1f} elem={agg['elementwise/copy']:.1f} attn={agg['attention']:.1f}")
    return dict(config=label, wall=round(wall, 2), gpu_busy=round(gpu, 2),
                **{b: round(agg[b], 3) for b in BUCKETS})


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []
print(f"E2E baseline benchmark + profile @ b{BATCH} ({RUNS}x{TIMED} timed)\n")
# int8_baseline
r, model, sampler, cond = build("int8_baseline")
rows.append(measure("int8_baseline", r, model, sampler, cond))
del model, sampler; torch.cuda.empty_cache()
# int4_baseline: proj fold OFF then ON (same model)
r, model, sampler, cond = build("int4_baseline")
os.environ["MODIFF_FUSE_PROJ_I4"] = "0"; rows.append(measure("int4_baseline.proj_off", r, model, sampler, cond))
os.environ["MODIFF_FUSE_PROJ_I4"] = "1"; rows.append(measure("int4_baseline.proj_on", r, model, sampler, cond))
del model, sampler; torch.cuda.empty_cache()

with open(f"{HERE}/data/e2e_baseline_profile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nWROTE {HERE}/data/e2e_baseline_profile.csv")
