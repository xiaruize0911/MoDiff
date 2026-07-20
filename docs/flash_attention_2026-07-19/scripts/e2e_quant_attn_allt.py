"""EXPERIMENT (a)+(b): int4 attention, and forcing int8/int4 attention onto the small-T (T=64)
blocks via MODIFF_QUANT_ATTN_ALLT=1. (T=16/4 hd=96 blocks still fall back: T%64 is a hard kernel
limit.) Times fp16 / int8-int8attn (T>=256) / int8-int8attn (ALL-T) / int4-int4attn (T>=256) /
int4-int4attn (ALL-T), + per-category profile of the two ALL-T runs. b128."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
def cat(name):
    l = name.lower()
    if "attn_softmax_requant" in l or "softmax_requant" in l: return "attn softmax+requant (intX)"
    if "softmax" in l or "scaled_dot" in l: return "attention softmax (fp16)"
    if "bmm_qk_s8" in l: return "attn QKᵀ int8"
    if "bmm_qk_s4" in l: return "attn QKᵀ int4"
    if "bmm_av_s8" in l: return "attn AV int8"
    if "bmm_av_s4" in l: return "attn AV int4"
    if "quantize_attn_qkv" in l or "aq_qtok" in l or "aq_vquant" in l or "aq_vscale" in l: return "attn q/k/v quantize"
    if "wmma_tensorop_f16" in l or "bmm" in l: return "attn QKᵀ/AV bmm (fp16 fallback)"
    if "implicit" in l or "cudnn" in l or "scudnn" in l or ("conv" in l and "int" not in l): return "conv"
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l: return "qkv/proj int GEMM"
    if "s1688gemm" in l or "ampere_fp16" in l: return "other fp16 GEMM"
    if "group_norm" in l or "gn_" in l: return "GroupNorm"
    if "quant" in l or "requant" in l or "absmax" in l: return "quantize/dequant"
    if "upsample" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "store" in l: return "elementwise/copy"
    return "other"

def setup(mode, qattn, allt):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if qattn else "0"
    os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"
    os.environ["MODIFF_QUANT_ATTN_ALLT"] = "1" if allt else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = ("integration/calibration/int4_calibration.pt" if "int4" in mode else
             ("integration/calibration/int8_calibration.pt" if "int8" in mode else None))
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/qattn_allt",
                          batch_size=BATCH, steps=40, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    return r, model, sampler, cond

def smp(sampler, r, cond, S):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

CFG = [("fp16", "fp16", False, False),
       ("int8 int8-attn T>=256", "int8_baseline", True, False),
       ("int8 int8-attn ALL-T", "int8_baseline", True, True),
       ("int4 int4-attn T>=256", "int4_baseline", True, False),
       ("int4 int4-attn ALL-T", "int4_baseline", True, True)]
res = {}; prof = {}
for (label, mode, qattn, allt) in CFG:
    r, model, sampler, cond = setup(mode, qattn, allt)
    smp(sampler, r, cond, 20); torch.cuda.synchronize()
    ms = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time(); smp(sampler, r, cond, 40); torch.cuda.synchronize()
        ms.append((time.time() - t0) / 40 * 1000)
    res[label] = statistics.mean(ms)
    if allt:
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            smp(sampler, r, cond, 20)
        torch.cuda.synchronize()
        agg = {}
        for e in p.key_averages():
            if e.self_device_time_total > 0: agg[cat(e.key)] = agg.get(cat(e.key), 0.0) + e.self_device_time_total / 20 / 1000
        prof[label] = agg
    del model, sampler; torch.cuda.empty_cache()

fp16 = res["fp16"]
print(f"\n===== e2e @ b{BATCH} (autocast ON all) =====")
for label, _, _, _ in CFG:
    print(f"  {label:24} {res[label]:8.1f} ms/step   {fp16/res[label]:.3f}x vs fp16")
for label in prof:
    print(f"\n===== profile: {label} (total {sum(prof[label].values()):.1f} ms) =====")
    for k, v in sorted(prof[label].items(), key=lambda x: -x[1]):
        print(f"  {v:7.1f} ms  {k}")
