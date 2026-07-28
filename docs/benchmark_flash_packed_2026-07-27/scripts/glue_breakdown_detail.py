"""Detailed per-kernel breakdown of the "Norm / resize / quantize glue" bucket
(everything that's NOT Conv, Attention, or Linear/GEMM -- see make_report_plots.py's
LAYER_TYPES). Same-session, same profiler methodology as final_speedup_and_breakdown.py,
but instead of collapsing into named categories, keeps every individual CUDA kernel name
so we can see exactly what's eating time inside the glue bucket and judge whether it's
already fused or a further-fusion candidate.
"""
import os, sys, json, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, TIMED, RUNS = 30, 150, 5
PROF_STEPS = 20
HERE = "docs/benchmark_flash_packed_2026-07-27"
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]

# Same rule set as final_speedup_and_breakdown.py's CATEGORY_RULES, but we only need
# to know which top-level bucket each kernel maps to (Conv / Attention / Linear-GEMM /
# glue) so we can isolate the glue kernels and keep their individual names.
NON_GLUE_RULES = [
    ("conv", ["modiff", "xmma_fprop", "fprop_optimized", "cudnn_convolution", "implicit_gemm"]),
    ("attention", ["flash_attn_int8", "flash_attn_int4", "flash_attn", "aten::bmm", "aten::_softmax",
                   "softmax_warp_forward", "aten::baddbmm", "flash_fwd", "fmha",
                   "scaled_dot_product", "efficient_attention"]),
    ("linear_gemm", ["gemm_w8a8", "gemm_w4a4", "wmma_tensorop", "addmm", "cublas"]),
]

def top_bucket(name):
    low = name.lower()
    for cat, keys in NON_GLUE_RULES:
        for k in keys:
            if k.lower() in low:
                return cat
    return "glue"


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ.pop("MODIFF_FLASH_PACKED", None)
    os.environ.pop("MODIFF_SDPA_BACKEND", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    mean_ms = statistics.mean(ms)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    kernel_time, total, calls = {}, 0.0, {}
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total
        if t <= 0:
            continue
        kernel_time[evt.key] = kernel_time.get(evt.key, 0.0) + t
        calls[evt.key] = calls.get(evt.key, 0) + evt.count
        total += t

    glue_kernels = {k: v for k, v in kernel_time.items() if top_bucket(k) == "glue"}
    glue_total = sum(glue_kernels.values())
    glue_ms = glue_total / total * mean_ms if total > 0 else 0.0
    # per-kernel ms/step (scale each kernel's share of the profiled-window total time by mean_ms)
    detail = [{"kernel": k, "pct_of_total": round(v / total * 100, 3),
               "ms_step": round(v / total * mean_ms, 4), "calls_per_step": round(calls[k] / PROF_STEPS, 2)}
              for k, v in sorted(glue_kernels.items(), key=lambda x: -x[1])]

    del model, sampler, prof; torch.cuda.empty_cache()
    return dict(ms_step=round(mean_ms, 2), glue_ms_step=round(glue_ms, 2),
                glue_pct_of_total=round(glue_total / total * 100, 2), kernels=detail)


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

results = {}
print(f"Glue-bucket detailed kernel breakdown @ b{BATCH}\n{'mode':16} {'ms/step':>9} {'glue ms':>9} {'glue %':>8}")
for label, mode in VERS:
    r = run(mode)
    print(f"{label:16} {r['ms_step']:9.2f} {r['glue_ms_step']:9.2f} {r['glue_pct_of_total']:7.2f}%")
    results[label] = r

with open(f"{HERE}/data/glue_breakdown_detail.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWROTE {HERE}/data/glue_breakdown_detail.json")

# Print top 12 kernels per mode for a quick look without opening the JSON.
for label, _ in VERS:
    print(f"\n--- {label} top glue kernels (ms/step) ---")
    for k in results[label]["kernels"][:12]:
        print(f"  {k['ms_step']:7.3f} ms  {k['pct_of_total']:6.2f}%  x{k['calls_per_step']:6.1f}/step  {k['kernel']}")
