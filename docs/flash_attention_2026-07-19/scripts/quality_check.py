"""Quality check: sampled-latent rel-L2 vs true fp16 for the quantized-attention configs.
Same seed/noise for all. b16, 50 DDIM steps. Measures end-to-end quality degradation."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B
BATCH, STEPS, SEED = 16, 50, 1234

def latent(x): return x[0] if isinstance(x, (tuple, list)) else x
def setup(mode, qattn, flash):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if qattn else "0"
    os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"; os.environ["MODIFF_QATTN_FLASH"] = "1" if flash else "0"
    os.environ["MODIFF_QUANT_ATTN_ALLT"] = "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = ("integration/calibration/int4_calibration.pt" if "int4" in mode else
             ("integration/calibration/int8_calibration.pt" if "int8" in mode else None))
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/quality",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    return r, model, sampler, cond

def sample(mode, qattn, flash, warm=12):
    r, model, sampler, cond = setup(mode, qattn, flash)
    def smp(S, seed=None):
        if seed is not None: torch.manual_seed(seed)
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            return latent(sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond))
    if qattn: smp(warm)                       # warm calibrates static attn scales
    out = smp(STEPS, seed=SEED).float().clone()
    del model, sampler; torch.cuda.empty_cache()
    return out

def rel(a, b): return (a - b).norm().item() / (b.norm().item() + 1e-9)

CFG = [("fp16 (reference)", "fp16", False, False),
       ("int8 lin + fp16 attn", "int8_baseline", False, False),
       ("int8 lin + int8 FUSED attn", "int8_baseline", True, True),
       ("int4 lin + int4 FUSED attn", "int4_baseline", True, True)]
lats = {}
for (label, mode, qattn, flash) in CFG:
    lats[label] = sample(mode, qattn, flash)
    print(f"  sampled {label}")
ref = lats["fp16 (reference)"]
print(f"\n===== sampled-latent rel-L2 vs true fp16 (b{BATCH}, {STEPS} DDIM steps, seed {SEED}) =====")
for label, _, _, _ in CFG:
    r_fp16 = rel(lats[label], ref)
    extra = ""
    if "int8 FUSED" in label:
        extra = f"  | vs int8-fp16-attn: {rel(lats[label], lats['int8 lin + fp16 attn']):.4f} (attn-quant cost)"
    print(f"  {label:30} rel-L2 {r_fp16:.4f}{extra}")
