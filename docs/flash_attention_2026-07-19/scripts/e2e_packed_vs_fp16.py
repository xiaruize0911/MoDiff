"""Wall-clock e2e: current DEFAULT int8/int4 (packed flash quant attention) vs TRUE fp16.
autocast fp16 ON for ALL modes. b128, DDIM. Measures the packed-quantize glue win end-to-end."""
import os, sys, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128; WARMUP, TIMED, RUNS = 15, 40, 3
VERS = [("fp16 (true, autocast)", "fp16"), ("int8 DEFAULT (packed flash)", "int8_baseline"),
        ("int4 DEFAULT (packed flash static)", "int4_baseline")]

def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    for kk in ("MODIFF_QUANT_ATTN", "MODIFF_QATTN_FLASH", "MODIFF_QUANT_ATTN_ALLT", "MODIFF_FLASH_ATTN"):
        os.environ.pop(kk, None)   # defaults: quant attn + packed flash ON for int8/int4
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/e2e_packed",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize(); ms.append((time.time() - t0) / TIMED * 1000)
    del model, sampler; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms)

bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

fp16 = None
print(f"\ne2e wall-clock (autocast ON for all) @ b{BATCH}\n")
print(f"{'version':36} {'ms/step':>9} {'min':>8} {'vs fp16':>9}")
for (label, mode) in VERS:
    mean, mn = run(mode)
    if fp16 is None: fp16 = mean
    print(f"{label:36} {mean:9.1f} {mn:8.1f} {fp16/mean:8.2f}x")
