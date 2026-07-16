"""Benchmark the opt-in int8 flash attention-score path (MODIFF_QUANT_ATTN) in the
real int8_baseline mode (batch 32): peak memory (T×T avoidance), speed, latent rel-err
vs fp16 attention. Two int8 configs: default (large-T only) and all attention blocks.
Emits data/attn_quant.csv."""
import os, sys, time, importlib.util, statistics, csv
import torch
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
OUT = "/workspace/MoDiff/docs/comprehensive_benchmark_2026-07-16/data"

class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 32; args.steps = 20
args.linear_backend = "fp16"; args.calibration = None
MODE = "int8_baseline"

def run_once(model, sampler, runner):
    torch.manual_seed(7); cond = runner._cond_kwargs(model, args.batch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = sampler.sample(S=args.steps, batch_size=args.batch_size, shape=runner.shape, eta=0.0, verbose=False, **cond)
    return (out[0] if isinstance(out, (tuple, list)) else out).float()

def measure(quant_attn, min_t=None):
    os.environ["MODIFF_QUANT_ATTN"] = quant_attn
    if min_t is not None: os.environ["MODIFF_FLASH_MIN_T"] = min_t
    elif "MODIFF_FLASH_MIN_T" in os.environ: del os.environ["MODIFF_FLASH_MIN_T"]
    runner, model, sampler = abb.build(MODE, args)
    for _ in range(3): run_once(model, sampler, runner)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    lat = run_once(model, sampler, runner); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    ts = []
    for _ in range(8):
        torch.cuda.synchronize(); t0 = time.time(); run_once(model, sampler, runner)
        torch.cuda.synchronize(); ts.append((time.time() - t0) / args.steps * 1000)
    del runner, model, sampler; torch.cuda.empty_cache()
    return lat, statistics.median(ts), peak

lat0, ms0, pk0 = measure("0")                 # fp16 attention
lat1, ms1, pk1 = measure("1")                 # int8 flash, default (large-T only)
lat2, ms2, pk2 = measure("1", min_t="64")     # int8 flash, all attention blocks
r1 = (lat1 - lat0).norm().item() / (lat0.norm().item() + 1e-12)
r2 = (lat2 - lat0).norm().item() / (lat0.norm().item() + 1e-12)
rows = [("fp16_attn", ms0, pk0, 0.0),
        ("int8_flash_largeT", ms1, pk1, r1),
        ("int8_flash_all", ms2, pk2, r2)]
for name, ms, pk, rel in rows:
    print(f"{name:20s} {ms:7.2f} ms/step  peak {pk:7.0f} MiB  rel {rel:.4f}", flush=True)
with open(f"{OUT}/attn_quant.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["config", "ms_step", "peak_MiB", "rel_err"]); wr.writerows(rows)
print("WROTE attn_quant.csv", flush=True)
