"""EXPERIMENT: FUSED (flash) quantized attention in the pipeline (MODIFF_QATTN_FLASH=1) vs the
materialized 3-kernel path vs fp16. Fusion = QKᵀ+softmax+AV in one kernel, scores in SRAM.
Times + per-category profile of the fused runs. b128."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B
BATCH = 128
def cat(name):
    l = name.lower()
    if "flash_attn_int8" in l or "flash_attn_int4" in l or "flash_attn" in l or "mma_kernel" in l: return "FUSED flash attn (int)"
    if "attn_softmax_requant" in l: return "attn softmax+requant (mat int)"
    if "softmax" in l or "scaled_dot" in l: return "attention softmax (fp16)"
    if "bmm_qk_s" in l: return "attn QKᵀ (mat int)"
    if "bmm_av_s" in l: return "attn AV (mat int)"
    if "quantize_attn_qkv" in l or "aq_qtok" in l or "aq_vquant" in l or "aq_vscale" in l or "pad" in l: return "attn q/k/v quantize"
    if "wmma_tensorop_f16" in l or "bmm" in l: return "attn bmm (fp16 fallback)"
    if "implicit" in l or "cudnn" in l or "scudnn" in l or ("conv" in l and "int" not in l): return "conv"
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l: return "qkv/proj int GEMM"
    if "s1688gemm" in l or "ampere_fp16" in l: return "other fp16 GEMM"
    if "group_norm" in l or "gn_" in l: return "GroupNorm"
    if "quant" in l or "requant" in l or "absmax" in l: return "quantize/dequant"
    if "upsample" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "store" in l or "clamp" in l or "round" in l: return "elementwise/copy/quant-eager"
    return "other"

def setup(mode, qattn, flash):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if qattn else "0"
    os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"
    os.environ["MODIFF_QATTN_FLASH"] = "1" if flash else "0"
    os.environ["MODIFF_QUANT_ATTN_ALLT"] = "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = ("integration/calibration/int4_calibration.pt" if "int4" in mode else
             ("integration/calibration/int8_calibration.pt" if "int8" in mode else None))
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/qattn_fused",
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
       ("int8 attn MATERIALIZED", "int8_baseline", True, False),
       ("int8 attn FUSED (flash)", "int8_baseline", True, True),
       ("int4 attn MATERIALIZED", "int4_baseline", True, False),
       ("int4 attn FUSED (flash)", "int4_baseline", True, True)]
res = {}; prof = {}
for (label, mode, qattn, flash) in CFG:
    r, model, sampler, cond = setup(mode, qattn, flash)
    smp(sampler, r, cond, 20); torch.cuda.synchronize()
    ms = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time(); smp(sampler, r, cond, 40); torch.cuda.synchronize(); ms.append((time.time() - t0) / 40 * 1000)
    res[label] = statistics.mean(ms)
    if flash:
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
    print(f"  {label:26} {res[label]:8.1f} ms/step   {fp16/res[label]:.3f}x vs fp16")
for label in prof:
    print(f"\n===== profile: {label} (total {sum(prof[label].values()):.1f} ms) =====")
    for k, v in sorted(prof[label].items(), key=lambda x: -x[1]):
        print(f"  {v:7.1f} ms  {k}")
