"""Driver for the nsys memcpy trace: build one mode, warm up, then run NSTEPS sampling steps inside a
cudaProfilerApi capture range so nsys (--capture-range=cudaProfilerApi) records ONLY the steady-state
region (excludes model build + calibration copies). Usage: python nsys_driver.py <mode> [nsteps].
mode is a _setup_model string: fp16 / int8_baseline / int4_baseline / int8 / int4.
"""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

mode = sys.argv[1]
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
BATCH = 128
quant = mode != "fp16"
os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
        ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                      "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/bench5mode",
                      batch_size=BATCH, steps=NSTEPS, shape=(4, 32, 32), calibration_path=calib,
                      linear_backend=("int_gemm" if quant else "fp16"))
model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)


def smp(S):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)


smp(10); torch.cuda.synchronize()      # warmup outside the capture range
torch.cuda.profiler.start()
smp(NSTEPS)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print(f"NSYS_DRIVER_DONE mode={mode} nsteps={NSTEPS}")
