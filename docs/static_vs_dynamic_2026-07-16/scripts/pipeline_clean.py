"""CLEAN static-vs-dynamic comparison that decomposes the full gap and controls noise.

The full dynamic_*/static_* gap conflates two things: (1) conv/linear — whose fused fp16 GN->int8
quantize path is intrinsically calibration-gated, so full-dynamic pays fp32 GroupNorm + a separate
quantize + a slower conv GEMM; and (2) attention — a clean single-variable toggle (same GEMMs, only
the Q/K/V scale source + softmax max source change). To separate them we add a middle config:
'attn-dyn' = conv/linear STATIC (MODIFF_CONVLIN_STATIC=1) but attention DYNAMIC. Then per precision:
  conv/linear effect = full-dynamic − attn-dynamic
  attention effect   = attn-dynamic − full-static
Noise control: sustained warmup + 20 back-to-back timed runs -> report mean/median/min/stdev, plus
GPU-busy (profiler device self-time, throttle-robust). Emits clean_speed.csv."""
import os, sys, time, csv, importlib.util, statistics, gc
os.environ.setdefault("MODIFF_ATTN_CALIB_STEPS", "16")
import torch
from torch.profiler import profile, ProfilerActivity
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
OUT = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"

class A: pass
def mkargs():
    a = A(); a.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
    a.ckpt = "models/ldm/lsun_churches256/model.ckpt"; a.batch_size = 32; a.steps = 12
    a.linear_backend = "int_gemm"; a.calibration = None; return a

# (label, precision, variant, mode, convlin_static_env)
CONFIGS = [
    ("fp16 dyn",         "fp16", "full-dynamic", "dynamic_fp16", False),
    ("fp16 static",      "fp16", "full-static",  "static_fp16",  False),
    ("int8 full-dyn",    "int8", "full-dynamic", "dynamic_int8", False),
    ("int8 attn-dyn",    "int8", "attn-dynamic", "dynamic_int8", True),   # conv/lin static, attn dynamic
    ("int8 full-static", "int8", "full-static",  "static_int8",  False),
    ("int4 full-dyn",    "int4", "full-dynamic", "dynamic_int4", False),
    ("int4 attn-dyn",    "int4", "attn-dynamic", "dynamic_int4", True),
    ("int4 full-static", "int4", "full-static",  "static_int4",  False),
]

def run(model, sampler, runner, a):
    cond = runner._cond_kwargs(model, a.batch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        sampler.sample(S=a.steps, batch_size=a.batch_size, shape=runner.shape, eta=0.0, verbose=False, **cond)

rows = []
print(f"{'config':>18} | {'wall mean':>9} {'median':>7} {'min':>7} {'sd':>5} | {'GPU-busy':>8}")
for label, prec, variant, mode, convlin in CONFIGS:
    if convlin: os.environ["MODIFF_CONVLIN_STATIC"] = "1"
    else: os.environ.pop("MODIFF_CONVLIN_STATIC", None)
    torch.cuda.empty_cache(); gc.collect()
    a = mkargs()
    runner, model, sampler = abb.build(mode, a)
    tw = time.time()                          # sustained warmup (hold boost clock + freeze calibration)
    while time.time() - tw < 9.0:
        run(model, sampler, runner, a)
    torch.cuda.synchronize()
    walls = []                                 # 20 back-to-back timed runs
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.time()
        run(model, sampler, runner, a)
        torch.cuda.synchronize(); walls.append((time.time() - t0) / a.steps * 1000)
    wmean, wmed, wmin, wsd = statistics.mean(walls), statistics.median(walls), min(walls), statistics.pstdev(walls)
    NP = 3                                      # GPU-busy (throttle-robust)
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(NP): run(model, sampler, runner, a)
        torch.cuda.synchronize()
    gpu = sum(e.self_device_time_total for e in pr.key_averages() if e.self_device_time_total > 0) / (NP * a.steps) / 1e3
    print(f"{label:>18} | {wmean:9.2f} {wmed:7.2f} {wmin:7.2f} {wsd:5.2f} | {gpu:8.2f}", flush=True)
    rows.append({"config": label, "precision": prec, "variant": variant,
                 "wall_mean_ms": round(wmean, 3), "wall_median_ms": round(wmed, 3),
                 "wall_min_ms": round(wmin, 3), "wall_stdev": round(wsd, 3), "gpu_busy_ms": round(gpu, 3)})
    del runner, model, sampler; torch.cuda.empty_cache(); gc.collect()
os.environ.pop("MODIFF_CONVLIN_STATIC", None)

with open(f"{OUT}/clean_speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "precision", "variant", "wall_mean_ms", "wall_median_ms", "wall_min_ms", "wall_stdev", "gpu_busy_ms"])
    w.writeheader(); w.writerows(rows)

# decomposition (GPU-busy)
g = {r["config"]: r["gpu_busy_ms"] for r in rows}
print("\n-- decomposition (GPU-busy ms/step) --")
for p in ("int8", "int4"):
    fd, ad, fs = g[f"{p} full-dyn"], g[f"{p} attn-dyn"], g[f"{p} full-static"]
    print(f"{p}: full-dyn {fd:.1f} -> [conv/lin static: -{fd-ad:.1f}] -> attn-dyn {ad:.1f} -> [attn static: -{ad-fs:.1f}] -> full-static {fs:.1f}")
print("WROTE clean_speed.csv")
