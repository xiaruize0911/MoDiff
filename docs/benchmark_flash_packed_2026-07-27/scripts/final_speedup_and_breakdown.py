"""Final speedup + full time-cost breakdown on the post-vectorization build.

Measures fp16 in the SAME session as the 4 quantized modes so the speedup ratios are
apples-to-apples (this repo's fp16 absolute timing has drifted across sessions before
due to environment/thermal state -- see docs/benchmark_flash_packed_2026-07-27 history).
Also computes the FULL kernel-category breakdown (not just quantize-vs-not), reusing the
category rules from mode_speedup_and_profile.py / profile_breakdown_v2.py.
"""
import os, sys, json, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, TIMED, RUNS = 30, 150, 5
PROF_STEPS = 15
HERE = "docs/benchmark_flash_packed_2026-07-27"
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]

CATEGORY_RULES = [
    ("conv_int_fused", ["modiff"]),
    ("attention_flash", ["flash_attn_int8", "flash_attn_int4", "flash_attn"]),
    ("attention_sdpa_math_unfused", ["aten::bmm", "aten::_softmax", "softmax_warp_forward", "aten::baddbmm"]),
    ("attention_sdpa_fused", ["flash_fwd", "fmha", "scaled_dot_product", "efficient_attention"]),
    ("gn_silu_quantize_fused", ["group_norm_silu_quantize", "group_norm_silu_delta_quantize",
                                "gn_apply_delta_quantize", "gn_group_stats", "static_quantize_and_update_ahat"]),
    ("gn_silu", ["group_norm_silu_nhwc", "native_group_norm"]),
    ("resize_unfused", ["upsample_nearest2d", "avg_pool2d"]),
    ("upsample_conv_fused", ["upsample2x_quantize"]),
    ("gemm_quant_fused", ["gemm_w8a8", "gemm_w4a4"]),
    ("quantize_standalone", ["aq_qtok", "aq_vquant", "aq_kquant", "quantize_attn", "scale_quantize",
                            "quantize_act_int8", "quantize_and_pack"]),
    ("conv_fp16", ["xmma_fprop", "fprop_optimized", "cudnn_convolution", "implicit_gemm"]),
    ("gemm_fp16", ["wmma_tensorop", "addmm", "cublas"]),
    ("elementwise_misc", ["aten::add", "aten::mul", "aten::copy", "aten::to", "aten::contiguous",
                          "aten::cat", "catarraybatchedcopy", "aten::div", "aten::silu",
                          "aten::mean", "aten::sub", "aten::clamp", "aten::round", "aten::chunk",
                          "elementwise_kernel", "vectorized", "direct_copy_kernel", "unrolled_elementwise"]),
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
    os.environ.pop("MODIFF_FLASH_PACKED", None)
    os.environ.pop("MODIFF_SDPA_BACKEND", None)  # shipped default (math)
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

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    cat_time, total = {}, 0.0
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total
        if t <= 0:
            continue
        cat = categorize(evt.key)
        cat_time[cat] = cat_time.get(cat, 0.0) + t
        total += t
    cat_pct = {k: round(v / total * 100, 2) for k, v in sorted(cat_time.items(), key=lambda x: -x[1])}

    del model, sampler, prof; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms), cat_pct


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

results = {}
fp16_ms = None
print(f"Final speedup + breakdown (post-vectorization build) @ b{BATCH}\n{'mode':16} {'ms/step':>9} {'min':>8} {'speedup':>9}")
for label, mode in VERS:
    mean, mn, cat_pct = run(mode)
    if fp16_ms is None:
        fp16_ms = mean
    sp = fp16_ms / mean
    print(f"{label:16} {mean:9.2f} {mn:8.2f} {sp:8.3f}x")
    results[label] = dict(ms_step=round(mean, 2), min_ms=round(mn, 2), speedup_vs_fp16=round(sp, 3), category_pct=cat_pct)

with open(f"{HERE}/data/final_speedup_and_breakdown.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWROTE {HERE}/data/final_speedup_and_breakdown.json")
