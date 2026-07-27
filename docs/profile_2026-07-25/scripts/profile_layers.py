"""Per-layer-type kernel profile (post-fusion-fix state, 2026-07-25).

torch.profiler (CPU+CUDA activities) over profiled DDIM steps for one mode (default:
int8_modiff -- the mode with the most fusion changes this session). Two outputs:

1. A Chrome Trace JSON (docs/profile_2026-07-25/trace/<mode>.json) -- openable directly
   in Perfetto UI (https://ui.perfetto.dev, "Open trace file") or chrome://tracing.
2. A layer-type -> kernel breakdown: every CUDA kernel is bucketed into a layer type
   (attention / conv / linear-GEMM / GroupNorm / quantize-dequant / modiff-cache /
   upsample-concat / elementwise-copy / other fp16 GEMM / other), then within each
   bucket kernels are ranked by % of THAT BUCKET's time (not global %). Writes
   docs/profile_2026-07-25/data/layer_kernel_breakdown.csv and layer_type_summary.csv.

Usage: python docs/profile_2026-07-25/scripts/profile_layers.py [mode]
  mode in {fp16, int8_baseline, int4_baseline, int8_modiff, int4_modiff} (default int8_modiff)
"""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

MODE_LABEL = sys.argv[1] if len(sys.argv) > 1 else "int8_modiff"
_MODE_MAP = {"fp16": "fp16", "int8_baseline": "int8_baseline", "int4_baseline": "int4_baseline",
             "int8_modiff": "int8", "int4_modiff": "int4"}
assert MODE_LABEL in _MODE_MAP, f"unknown mode {MODE_LABEL}"
MODE = _MODE_MAP[MODE_LABEL]

BATCH = 128
WARMUP, STEPS, NP = 40, 20, 5
HERE = "docs/profile_2026-07-25"

# Layer-type classifier (same bucketing convention as
# docs/benchmark_5mode_2026-07-23/scripts/e2e_timing_profile.py's cat(), reused here as
# the "layer type" taxonomy).
LAYER_TYPES_ORDER = ["Attention", "Conv (int GEMM)", "Linear (qkv/proj int GEMM)",
                     "Attention bmm (fp16)", "Other fp16 GEMM", "GroupNorm",
                     "MoDiff cache (o_hat/a_hat)", "Quantize/Dequant", "Upsample/Concat",
                     "Elementwise/Copy", "Other"]


def layer_type(name):
    l = name.lower()
    if "softmax" in l or "scaled_dot" in l or "flash" in l: return "Attention"
    if "implicit" in l or "cudnn" in l or "fprop" in l or "conv2d" in l or "scudnn" in l or "wgrad" in l or "convolution" in l: return "Conv (int GEMM)"
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l: return "Linear (qkv/proj int GEMM)"
    if "wmma_tensorop_f16" in l or "bmm" in l: return "Attention bmm (fp16)"
    if "cutlass" in l or "cublas" in l or "ampere_fp16" in l or "s1688gemm" in l: return "Other fp16 GEMM"
    if "group_norm" in l or "groupnorm" in l or "gn_" in l or "fused_gn" in l: return "GroupNorm"
    if "scale_accumulate" in l or "o_hat" in l or "dequant_accumulate" in l: return "MoDiff cache (o_hat/a_hat)"
    if "quant" in l or "dequant" in l or "requant" in l or "absmax" in l or "ahat" in l or "step1" in l or "aq_" in l or "pack" in l or "upsample2x" in l: return "Quantize/Dequant"
    if "upsample" in l or "interpolate" in l or "catarray" in l.replace("_", ""): return "Upsample/Concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "index" in l or "store" in l or "clamp" in l or "round" in l: return "Elementwise/Copy"
    return "Other"


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/profile_layers",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        smp(NP * STEPS)
    torch.cuda.synchronize()

    trace_path = f"{HERE}/trace/{MODE_LABEL}.json"
    p.export_chrome_trace(trace_path)
    print(f"WROTE Perfetto/Chrome trace: {trace_path}  (open in https://ui.perfetto.dev)")

    # Profiling with CPU+CUDA activities (for a richer Perfetto trace) means key_averages()
    # returns BOTH real CUDA kernel entries AND CPU-side composite op entries (aten::copy_,
    # aten::cat, cudaLaunchKernel, Memcpy/Memset, ...) whose self_device_time_total double-counts
    # the same GPU time already attributed to the underlying kernel's own entry. Keep only actual
    # device-side kernel launches (device_type == CUDA) and drop host-API/composite-op keys, so
    # the aggregate matches the CUDA-only-activity methodology used by the other profiling scripts
    # in this repo (docs/benchmark_5mode_2026-07-23/scripts/e2e_*profile*.py).
    from torch.autograd import DeviceType
    _HOST_PREFIXES = ("aten::", "cuda", "Memcpy", "Memset")
    per_kernel = {}
    for e in p.key_averages():
        if getattr(e, "device_type", None) != DeviceType.CUDA:
            continue
        if e.key.startswith(_HOST_PREFIXES):
            continue
        t = e.self_device_time_total
        if t > 0:
            per_kernel[e.key] = per_kernel.get(e.key, 0.0) + t / (NP * STEPS) / 1000.0  # ms/step
    del model, sampler; torch.cuda.empty_cache()
    return per_kernel


# GPU clock burn-in
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

print(f"Profiling mode={MODE_LABEL} ...", flush=True)
per_kernel = run(MODE)

# Bucket into layer types
buckets = {lt: [] for lt in LAYER_TYPES_ORDER}
for name, ms in per_kernel.items():
    buckets[layer_type(name)].append((name, ms))

grand_total = sum(per_kernel.values())

layer_rows = []
kernel_rows = []
for lt in LAYER_TYPES_ORDER:
    items = sorted(buckets[lt], key=lambda x: -x[1])
    lt_total = sum(ms for _, ms in items)
    if lt_total <= 0:
        continue
    layer_rows.append((lt, lt_total, lt_total / grand_total * 100))
    for name, ms in items:
        kernel_rows.append((lt, name, ms, ms / lt_total * 100))

layer_rows.sort(key=lambda x: -x[1])

with open(f"{HERE}/data/layer_type_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer_type", "ms_per_step", "pct_of_total"])
    for lt, ms, pct in layer_rows:
        w.writerow([lt, round(ms, 4), round(pct, 2)])

with open(f"{HERE}/data/layer_kernel_breakdown.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer_type", "kernel", "ms_per_step", "pct_within_layer_type"])
    # keep layer_type in the same total-desc order as the summary
    order = {lt: i for i, (lt, _, _) in enumerate(layer_rows)}
    for lt, name, ms, pct in sorted(kernel_rows, key=lambda r: (order.get(r[0], 999), -r[2])):
        w.writerow([lt, name, round(ms, 4), round(pct, 2)])

print(f"\nmode={MODE_LABEL}  gpu_busy={grand_total:.3f} ms/step\n")
print(f"{'layer type':<30} {'ms/step':>10} {'% of total':>12}")
print("-" * 54)
for lt, ms, pct in layer_rows:
    print(f"{lt:<30} {ms:10.3f} {pct:11.2f}%")

print(f"\nWROTE {HERE}/data/layer_type_summary.csv")
print(f"WROTE {HERE}/data/layer_kernel_breakdown.csv")
print("\nLAYER_PROFILE_DONE")
