"""Name-level profile of the elementwise/copy 'glue' kernels in the int8 fused-flash default config,
so we know what to optimize. b128."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B
BATCH = 128
def short(n):
    n = n.replace("void ", "").replace("(anonymous namespace)::", "").replace("at::native::", "")
    return n[:70]
def is_glue(n):
    l = n.lower()
    return any(t in l for t in ("elementwise", "vectorized", "direct_copy", "copy", "_cat", "catarray",
                                "fill", "add", "store", "silu", "transpose", "contiguous", "index"))
os.environ["MODIFF_QUANT_LINEAR"] = "1"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
for kk in ("MODIFF_QUANT_ATTN", "MODIFF_QATTN_FLASH", "MODIFF_QUANT_ATTN_ALLT"): os.environ.pop(kk, None)
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                      "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/glueprof",
                      batch_size=BATCH, steps=20, shape=(4, 32, 32),
                      calibration_path="integration/calibration/int8_calibration.pt", linear_backend="int_gemm")
model, sampler = r._setup_model("int8_baseline"); cond = r._cond_kwargs(model, BATCH)
def smp(S):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(50): bn = bn @ bn * 1e-4 + 1.0
smp(20); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p: smp(20)
torch.cuda.synchronize()
glue = [(e.self_device_time_total / 20 / 1000, e.key) for e in p.key_averages()
        if e.self_device_time_total > 0 and is_glue(e.key)]
glue.sort(reverse=True)
tot = sum(t for t, _ in glue)
print(f"\n===== int8 fused: elementwise/copy 'glue' kernels (total {tot:.1f} ms/step) =====")
for t, n in glue:
    if t > 0.15: print(f"  {t:7.2f} ms  {short(n)}")
