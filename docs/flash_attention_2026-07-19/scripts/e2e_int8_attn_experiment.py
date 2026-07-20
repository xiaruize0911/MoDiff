"""EXPERIMENT: int8 attention actually used in int8 mode (MODIFF_QUANT_ATTN=1 ->
QuantizedStandardAttentionBlock: attn_qk_int8 + attn_softmax_requant + attn_av_int8).
Not the practical config (int8 attention is slower) — this is to SEE the fully-int8 pipeline.
Times fp16 / int8(fp16 attn) / int8(int8 attn) + per-category profile of the int8-attn run. b128."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
def cat(name):
    l = name.lower()
    if "softmax" in l or "scaled_dot" in l: return "attention softmax"
    if "bmm_qk_s8" in l or "bmm_qk_s4" in l: return "attn QKᵀ (int8)"
    if "bmm_av_s8" in l or "bmm_av_s4" in l: return "attn AV (int8)"
    if "attn_softmax_requant" in l: return "attn softmax+requant (int8)"
    if "quantize_attn_qkv" in l or "aq_qtok" in l or "aq_vquant" in l or "aq_vscale" in l: return "attn q/k/v quantize"
    if "wmma_tensorop_f16" in l or "bmm" in l: return "attn QKᵀ/AV bmm (fp16)"
    if "implicit" in l or "cudnn" in l or "scudnn" in l or ("conv" in l and "int" not in l): return "conv"
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l: return "qkv/proj int GEMM"
    if "s1688gemm" in l or "ampere_fp16" in l: return "other fp16 GEMM"
    if "group_norm" in l or "gn_" in l: return "GroupNorm"
    if "quant" in l or "requant" in l or "absmax" in l: return "quantize/dequant"
    if "upsample" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "store" in l: return "elementwise/copy"
    return "other"

def setup(mode, qattn):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if qattn else "0"
    os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if quant else None
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/i8attn",
                          batch_size=BATCH, steps=40, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    return r, model, sampler, cond, quant

def smp(sampler, r, cond, quant, S):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

results = {}
CFG = [("fp16", "fp16", False), ("int8 (fp16 attn)", "int8_baseline", False), ("int8 (INT8 attn)", "int8_baseline", True)]
for (label, mode, qattn) in CFG:
    r, model, sampler, cond, quant = setup(mode, qattn)
    smp(sampler, r, cond, quant, 20)  # warmup (also calibrates static qattn scales)
    torch.cuda.synchronize()
    ms = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time(); smp(sampler, r, cond, quant, 40); torch.cuda.synchronize()
        ms.append((time.time() - t0) / 40 * 1000)
    results[label] = statistics.mean(ms)
    if qattn:   # profile the fully-int8 run
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            smp(sampler, r, cond, quant, 20)
        torch.cuda.synchronize()
        agg = {}
        for e in p.key_averages():
            if e.self_device_time_total > 0: agg[cat(e.key)] = agg.get(cat(e.key), 0.0) + e.self_device_time_total / 20 / 1000
        results["_prof"] = agg
    del model, sampler; torch.cuda.empty_cache()

fp16 = results["fp16"]
print(f"\n===== e2e @ b{BATCH} (autocast ON all) =====")
for label, _, _ in CFG:
    print(f"  {label:20} {results[label]:8.1f} ms/step   {fp16/results[label]:.3f}x vs fp16")
print(f"\n===== per-category profile: int8 with INT8 attention (total {sum(results['_prof'].values()):.1f} ms) =====")
for k, v in sorted(results["_prof"].items(), key=lambda x: -x[1]):
    print(f"  {v:7.1f} ms  {k}")
