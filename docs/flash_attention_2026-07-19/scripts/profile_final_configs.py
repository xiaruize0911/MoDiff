"""Per-category kernel time breakdown for the current DEFAULT configs: int8 fused-flash and
int4 fused-flash (static). b128, autocast on. Shows where the time goes after all the fusions."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B
BATCH = 128
def cat(name):
    l = name.lower()
    if "flash_attn" in l or "mma_kernel" in l: return "FUSED flash attn"
    if "aq_qtok" in l or "aq_vquant" in l or "aq_vscale" in l or "quantize_attn" in l: return "attn q/k/v quantize (kernel)"
    if "softmax" in l or "scaled_dot" in l: return "attention softmax (fp16 fallback)"
    if "bmm_qk" in l or "bmm_av" in l or "attn_softmax_requant" in l: return "attn materialized (fallback)"
    if "wmma_tensorop_f16" in l or ("bmm" in l): return "attn bmm (fp16 fallback)"
    if "implicit" in l or "cudnn" in l or "scudnn" in l or ("conv" in l and "int" not in l): return "conv (int GEMM)"
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l: return "qkv/proj int GEMM"
    if "s1688gemm" in l or "ampere_fp16" in l: return "other fp16 GEMM"
    if "group_norm" in l or "gn_" in l: return "GroupNorm(+quantize)"
    if "scale_quantize" in l or "quant_attn_out" in l or "requant" in l: return "linear quantize/dequant"
    if "upsample" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "store" in l or "transpose" in l: return "elementwise/copy"
    return "other"

def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    for kk in ("MODIFF_QUANT_ATTN", "MODIFF_QATTN_FLASH", "MODIFF_QUANT_ATTN_ALLT"): os.environ.pop(kk, None)  # defaults
    calib = ("integration/calibration/int4_calibration.pt" if "int4" in mode else
             ("integration/calibration/int8_calibration.pt" if "int8" in mode else None))
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/finalprof",
                          batch_size=BATCH, steps=20, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(20); torch.cuda.synchronize()             # warm (also freezes static attn quantize)
    with profile(activities=[ProfilerActivity.CUDA]) as p: smp(20)
    torch.cuda.synchronize()
    agg = {}
    for e in p.key_averages():
        if e.self_device_time_total > 0: agg[cat(e.key)] = agg.get(cat(e.key), 0.0) + e.self_device_time_total / 20 / 1000
    del model, sampler; torch.cuda.empty_cache(); return agg

bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()
for label, mode in [("int8 fused-flash (DEFAULT)", "int8_baseline"), ("int4 fused-flash STATIC", "int4_baseline")]:
    agg = run(mode); tot = sum(agg.values())
    print(f"\n===== {label}: {tot:.1f} ms/step =====")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {v:7.2f} ms  {100*v/tot:5.1f}%  {k}")
