"""Export torch.profiler CUDA+CPU traces (Chrome Trace Event JSON) for the fusion-fix configs,
loadable directly in Perfetto UI (ui.perfetto.dev -> Open trace file). One trace per invocation
over a few real DDIM steps (warmup first so the fused kernels are wired/calibrated). The fusion
flag (if any) is set by the CALLER's env before import. Writes data/perfetto/trace_<CFG>.json.

Usage:  CFG=<label> MODE=<int8|int4> [flag envs...] python perfetto_trace.py [--steps N] [--batch B]
  e.g.  CFG=int8_modiff.default MODE=int8 python perfetto_trace.py --steps 3 --batch 64
"""
import os, sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MODIFF_QUANT_LINEAR", "1"); os.environ.setdefault("MODIFF_QUANT_ATTN", "1")
os.environ.setdefault("MODIFF_QUANT_ATTN_STATIC", "1"); os.environ.setdefault("MODIFF_LINEAR_OUT_I8", "0")
os.environ.pop("MODIFF_FLASH_ATTN", None)
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

CFG = os.environ["CFG"]; MODE = os.environ["MODE"]
args = sys.argv[1:]
STEPS = int(args[args.index("--steps") + 1]) if "--steps" in args else 5   # must divide 1000 (DDIM uniform discretization)
BATCH = int(args[args.index("--batch") + 1]) if "--batch" in args else 64
WARMUP = 20   # must divide 1000
HERE = "docs/fusion_fix_2026-07-22"; OUT = f"{HERE}/data/perfetto"; os.makedirs(OUT, exist_ok=True)

calib = "integration/calibration/int8_calibration.pt" if "int8" in MODE else "integration/calibration/int4_calibration.pt"
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                      "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/fusionfix",
                      batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                      linear_backend="int_gemm")
model, sampler = r._setup_model(MODE); cond = r._cond_kwargs(model, BATCH)


def reset():
    B.reset_modiff_state_int8(model.model.diffusion_model) if "int8" in MODE else B.reset_modiff_state_int4(model.model.diffusion_model)
    if "int8" in MODE and B.HAS_INT8_LINEAR: B.reset_modiff_state_linear(model.model.diffusion_model)
    if "int4" in MODE and B.HAS_INT4_LINEAR: B.reset_modiff_state_int4_linear(model.model.diffusion_model)


def smp(S):
    reset()
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(40): bn = bn @ bn * 1e-4 + 1.0
smp(WARMUP); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    smp(STEPS)
torch.cuda.synchronize()
path = f"{OUT}/trace_{CFG}.json"
prof.export_chrome_trace(path)
print(f"WROTE {path}  ({os.path.getsize(path)/1e6:.1f} MB, {STEPS} DDIM steps @ b{BATCH})")
print(f"Load in Perfetto: https://ui.perfetto.dev -> 'Open trace file' -> {path}")
