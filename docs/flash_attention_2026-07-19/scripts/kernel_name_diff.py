"""Name-level top-kernel diff: fp16 vs int8, per-step GPU time. No bucketing — raw kernel
names so conv/linear/attention attribution is exact. b128."""
import os, sys, time
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = 128; NP = 20
def short(n):
    n = n.replace("void ", "").replace("cutlass::", "").replace("at::native::", "")
    for tag in ["cutlass_80_", "_ZN", "(anonymous namespace)::"]:
        n = n.replace(tag, "")
    return n[:82]

def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if quant else None
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/namediff",
                          batch_size=BATCH, steps=NP, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    def smp(S):  # autocast fp16 ON for ALL modes (fixed 2026-07-20: was enabled=quant -> fp16 ran fp32/tf32)
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(20); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        smp(NP)
    torch.cuda.synchronize()
    ks = [(e.self_device_time_total / NP / 1000, e.key) for e in p.key_averages() if e.self_device_time_total > 0]
    ks.sort(reverse=True)
    tot = sum(t for t, _ in ks)
    del model, sampler; torch.cuda.empty_cache()
    return tot, ks

for mode in ["fp16", "int8_baseline"]:
    tot, ks = run(mode)
    shown = [(t, n) for t, n in ks if t >= 0.3]
    print(f"\n================ {mode}: total GPU {tot:.1f} ms/step | {len(ks)} distinct kernels ================")
    print(f"   {'ms':>7} {'%':>6}  kernel")
    acc = 0.0
    for t, name in shown:
        acc += t
        print(f"   {t:7.2f} {100*t/tot:5.1f}%  {short(name)}")
    print(f"   {'-'*7}")
    print(f"   {acc:7.2f} {100*acc/tot:5.1f}%  (shown, ≥0.3 ms)   | rest {tot-acc:.2f} ms in {len(ks)-len(shown)} small kernels")
