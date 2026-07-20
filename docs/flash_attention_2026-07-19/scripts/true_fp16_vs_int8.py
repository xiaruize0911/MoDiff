"""The 'fp16' baseline actually ran fp32/tf32 (no autocast). Measure the TRUE fp16 baseline
(autocast fp16 ON) and compare to int8, to separate precision (fp32->fp16) from quantization
(fp16->int8). b128."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B
BATCH = 128

def run(mode, autocast):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if quant else None
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/truefp16",
                          batch_size=BATCH, steps=40, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=autocast, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(20); torch.cuda.synchronize()
    ms = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time(); smp(40); torch.cuda.synchronize(); ms.append((time.time() - t0) / 40 * 1000)
    del model, sampler; torch.cuda.empty_cache()
    return statistics.mean(ms)

# model dtype check
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml","models/ldm/lsun_churches256/model.ckpt",
                      output_dir="integration/results/truefp16", batch_size=BATCH, steps=40, shape=(4,32,32), linear_backend="fp16")
m, _ = r._setup_model("fp16")
dt = next(m.model.diffusion_model.parameters()).dtype
print(f"diffusion_model param dtype = {dt}")
del m; torch.cuda.empty_cache()

fp16_noac = run("fp16", False)     # as-benchmarked (fp32/tf32)
fp16_ac   = run("fp16", True)      # TRUE fp16 (autocast)
int8_ac   = run("int8_baseline", True)
print(f"\n{'config':32} {'ms/step':>9} {'vs fp32/tf32':>12} {'vs TRUE fp16':>12}")
print(f"{'fp16 no-autocast (fp32/tf32)':32} {fp16_noac:9.1f} {1.0:11.2f}x {'-':>12}")
print(f"{'fp16 autocast (TRUE fp16)':32} {fp16_ac:9.1f} {fp16_noac/fp16_ac:11.2f}x {1.0:11.2f}x")
print(f"{'int8_baseline':32} {int8_ac:9.1f} {fp16_noac/int8_ac:11.2f}x {fp16_ac/int8_ac:11.2f}x")
