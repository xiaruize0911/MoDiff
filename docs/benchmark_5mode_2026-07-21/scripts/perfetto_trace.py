"""Export torch.profiler CUDA+CPU traces (Chrome Trace Event JSON) for the current 5-mode pipeline,
loadable directly in Perfetto UI (ui.perfetto.dev -> Open trace file). One trace per mode over a few
real DDIM steps (warmup first so the fused kernels are wired/calibrated). Writes data/perfetto/trace_<mode>.json.

Usage: python perfetto_trace.py [mode1 mode2 ...] [--steps N] [--batch B]
Modes: fp16 int8_baseline int4_baseline int8_modiff int4_modiff (default: all 5).
"""
import os, sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

HERE = "docs/benchmark_5mode_2026-07-21"
OUT = f"{HERE}/data/perfetto"; os.makedirs(OUT, exist_ok=True)
ALL = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
       ("int8_modiff", "int8"), ("int4_modiff", "int4")]

args = [a for a in sys.argv[1:]]
STEPS = int(args[args.index("--steps") + 1]) if "--steps" in args else 3
BATCH = int(args[args.index("--batch") + 1]) if "--batch" in args else 128
sel = [a for a in args if not a.startswith("--") and a not in (str(STEPS), str(BATCH))]
VERS = [(l, m) for (l, m) in ALL if (not sel or l in sel)]
WARMUP = 15


def run(label, mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/bench5mode",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        smp(STEPS)
    torch.cuda.synchronize()
    path = f"{OUT}/trace_{label}.json"
    prof.export_chrome_trace(path)
    mb = os.path.getsize(path) / 1e6
    print(f"WROTE {path}  ({mb:.1f} MB, {STEPS} DDIM steps @ b{BATCH})")
    del model, sampler; torch.cuda.empty_cache()


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(40): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()
for (label, mode) in VERS:
    print(f"\n===== profiling {label} =====")
    run(label, mode)
print(f"\nLoad in Perfetto: open https://ui.perfetto.dev -> 'Open trace file' -> {OUT}/trace_<mode>.json")
