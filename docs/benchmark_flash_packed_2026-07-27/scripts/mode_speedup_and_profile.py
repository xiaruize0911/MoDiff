"""Fresh same-session 5-mode e2e speedup table + torch.profiler kernel-level breakdown,
run today (after the MODIFF_FLASH_PACKED default flip) so speedups aren't contaminated by
the environment drift seen between the 07-25 run and today's session (heavy pip install).

For each mode: build the model via BenchmarkRunner (same harness as e2e_speed.py), warm up
(lets calibration + flash/packed autotune freeze), then:
  1. time E2E (30 warm + 5x150 steps) -> per-round ms (printed for outlier diagnosis),
     mean/median/min ms/step, speedup vs fp16 (computed from median, robust to a single
     stalled round -- e.g. a cudnn.benchmark re-tune or GPU clock dip on one round).
  2. torch.profiler over 10 more steps -> per-CUDA-KERNEL self time (device-side events only;
     CPU-side `aten::*` dispatcher entries are EXCLUDED because their self_device_time_total
     duplicates the very kernel they launch -- summing both double-counts the same GPU work,
     confirmed by exact-matching pairs like aten::upsample_nearest2d vs
     upsample_nearest2d_nhwc_out_frame). Bucketed into categories (fused GN+SiLU+quantize, int
     conv, flash/SDPA attention, standalone quantize, quantized gemm, fp conv/gemm,
     elementwise/misc) to see where time goes AND whether any category indicates un-fused
     leftover kernels.
"""
import os, sys, csv, json, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, TIMED, RUNS = 30, 150, 5
PROF_STEPS = 10
HERE = "docs/benchmark_flash_packed_2026-07-27"
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]

CATEGORY_RULES = [
    ("attention_flash", ["flash_attn_int8", "flash_attn_int4", "flash_attn"]),
    ("attention_sdpa", ["softmax", "scaled_dot_product", "efficient_attention", "fmha"]),
    ("gn_silu_quantize_fused", ["group_norm_silu_quantize", "group_norm_silu_delta_quantize",
                               "gn_apply_delta_quantize", "gn_group_stats"]),
    ("gn_silu", ["group_norm_silu", "group_norm", "native_group_norm"]),
    ("upsample_fused", ["upsample2x_quantize"]),
    ("upsample_plain", ["upsample_nearest", "upsample_bilinear"]),
    ("conv_int_fused", ["conv2d_int8_fprop", "conv2d_int4_fprop", "conv2d_intx",
                        "modiff26implicitgemmconvolution", "modiffimplicitgemmconvolution"]),
    ("gemm_quant_fused", ["gemm_w8a8", "gemm_w4a4"]),
    ("quantize_standalone", ["quantize_attn_qkv", "quantize_attn_out", "scale_quantize",
                             "quantize_act_int8", "quantize_and_pack", "aq_qtok", "aq_vquant"]),
    ("conv_fp", ["cudnn_convolution", "implicit_gemm", "cutlass_tensorop_f16", "cutlass__5x_cudnn",
                "wmma_tensorop_f16", "xmma_fprop"]),
    ("gemm_fp", ["cutlass_80_tensorop", "cublas", "gemm"]),
    ("copy_cat_misc", ["copy_", "catarraybatched", "cat"]),
    ("elementwise_misc", ["add", "mul", "div", "silu", "mean", "sub", "clamp", "round",
                          "chunk", "permute", "view", "reshape", "empty", "fill", "index",
                          "elementwise", "vectorized"]),
]

def categorize(name):
    low = name.lower()
    for cat, keys in CATEGORY_RULES:
        for k in keys:
            if k.lower() in low:
                return cat
    return "other"


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ.pop("MODIFF_FLASH_PACKED", None)   # use the shipped default (=1) for int8 modes
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
    for i in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        rt = (time.time() - t0) / TIMED * 1000
        ms.append(rt)
        print(f"    round {i}: {rt:.2f} ms/step")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()

    cat_time = {}
    kernel_time = {}
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue          # skip CPU-side aten:: dispatcher entries (double-count guard)
        t = evt.self_device_time_total
        if t <= 0:
            continue
        cat = categorize(evt.key)
        cat_time[cat] = cat_time.get(cat, 0.0) + t
        kernel_time[evt.key] = kernel_time.get(evt.key, 0.0) + t
    total = sum(cat_time.values()) or 1.0
    cat_pct = {k: round(v / total * 100, 2) for k, v in cat_time.items()}
    top_kernels = sorted(kernel_time.items(), key=lambda x: -x[1])[:15]
    top_kernels = [dict(name=k, us_total=round(v, 1)) for k, v in top_kernels]

    del model, sampler, prof; torch.cuda.empty_cache()
    return statistics.mean(ms), statistics.median(ms), min(ms), cat_pct, top_kernels


os.makedirs(f"{HERE}/data", exist_ok=True)
os.makedirs(f"{HERE}/tmp_out", exist_ok=True)

bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

rows, profiles = [], {}
fp16_med = None
print(f"5-mode speed + profile @ b{BATCH} (today's session, shipped defaults)\n")
print(f"{'mode':16} {'mean':>9} {'median':>9} {'min':>8} {'speedup(median)':>16}")
for label, mode in VERS:
    print(f"  -- {label} --")
    mean, med, mn, cat_pct, top_k = run(mode)
    if fp16_med is None:
        fp16_med = med
    sp = fp16_med / med
    print(f"{label:16} {mean:9.2f} {med:9.2f} {mn:8.2f} {sp:16.3f}x")
    rows.append(dict(mode=label, mean_ms=round(mean, 2), median_ms=round(med, 2), min_ms=round(mn, 2),
                     speedup_vs_fp16=round(sp, 3)))
    profiles[label] = dict(category_pct=cat_pct, top_kernels=top_k)

with open(f"{HERE}/data/speedup_today.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(f"{HERE}/data/profile_breakdown.json", "w") as f:
    json.dump(profiles, f, indent=2)
print(f"\nWROTE {HERE}/data/speedup_today.csv and {HERE}/data/profile_breakdown.json")
