"""Run ONE mode's UNet for a few DDIM steps (after warming static calibration). Invoked under nsys
by nsys_profile.py. argv[1] = mode string."""
import os, sys, importlib.util
os.environ.setdefault("MODIFF_ATTN_CALIB_STEPS", "16")
import torch
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
mode = sys.argv[1]
class A: pass
a = A(); a.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
a.ckpt = "models/ldm/lsun_churches256/model.ckpt"; a.batch_size = 8; a.steps = 12
a.linear_backend = "int_gemm"; a.calibration = None
runner, model, sampler = abb.build(mode, a)
cond = runner._cond_kwargs(model, 8)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(mode != "fp32")):
    for _ in range(3):                        # warm + freeze static calibration (NOT under capture range)
        sampler.sample(S=12, batch_size=8, shape=runner.shape, eta=0.0, verbose=False, **cond)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()               # nsys --capture-range=cudaProfilerApi profiles only this
    sampler.sample(S=12, batch_size=8, shape=runner.shape, eta=0.0, verbose=False, **cond)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
