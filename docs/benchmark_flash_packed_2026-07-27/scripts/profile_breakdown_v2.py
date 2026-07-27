"""Corrected kernel-level profiling breakdown per mode.

Fixes over the first pass (mode_speedup_and_profile.py):
  1. torch.profiler double-counts GPU time if you sum every key_averages() row's
     self_device_time_total: a CPU-side dispatcher op (e.g. aten::_softmax) and its
     underlying CUDA kernel (softmax_warp_forward) both report the SAME device time
     (verified: both show identical self_device_time_total for the same call). Custom
     pybind extension kernels (flash_attn_int8_mma_kernel, group_norm_silu_quantize_nhwc_kernel,
     the CUTLASS conv kernels) have NO such CPU-side duplicate. Summing everything therefore
     inflates any category dominated by native aten:: ops (fp16 mode, MATH-SDPA fallback)
     relative to categories dominated by single custom kernels (int8/int4 fused paths).
     Fix: only sum events where device_type == DeviceType.CUDA (the actual kernel launch).
  2. The int8/int4 CUTLASS conv kernel is a heavily-templated mangled C++ symbol
     ("_ZN7cutlass6KernelIN6modiff26ImplicitGemmConvolution...") that contains the literal
     substring "modiff" (namespace tag) but NOT "conv2d_int8_fprop" (that's the Python-level
     wrapper name, never seen by the profiler) -- the old rule fell through to the generic
     "conv_fp" bucket via the word "convolution". Fixed: match "modiff" first.
  3. MATH-backend SDPA (see token_major_attention.py's _SDPA_CTX -> SDPBackend.MATH) does NOT
     call a fused attention kernel at all -- it decomposes into aten::bmm + aten::_softmax +
     aten::bmm, which matched nothing in the old rules and fell into "other". Fixed: bmm/softmax
     now map to a dedicated attention_sdpa_math_unfused category (real signal: this attention
     path is running unfused regardless of mode).
  4. The Q/K/V quantize kernels for the flash path (aq_qtok_packed_static_qk_kernel,
     aq_vquant_trans_packed_tiled_kernel) didn't match the guessed substrings
     ("quantize_attn_qkv" etc, which are Python-level names) -- added the real kernel name
     patterns ("aq_qtok", "aq_vquant", "aq_kquant").
Saves the top 30 kernels (not 15) per mode for a fuller audit trail.
"""
import os, sys, json
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, PROF_STEPS = 30, 15
HERE = "docs/benchmark_flash_packed_2026-07-27"
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]

CATEGORY_RULES = [
    ("conv_int_fused", ["modiff"]),
    ("attention_flash", ["flash_attn_int8", "flash_attn_int4", "flash_attn"]),
    ("attention_sdpa_math_unfused", ["aten::bmm", "aten::_softmax", "softmax_warp_forward",
                                     "aten::baddbmm", "bmm_kernel"]),
    ("attention_sdpa_fused", ["scaled_dot_product", "efficient_attention", "fmha",
                              "flash_fwd", "attn::"]),
    ("gn_silu_quantize_fused", ["group_norm_silu_quantize", "group_norm_silu_delta_quantize"]),
    ("gn_silu", ["group_norm_silu", "native_group_norm", "aten::group_norm"]),
    ("upsample_fused", ["upsample2x_quantize"]),
    ("upsample_unfused", ["upsample_nearest2d"]),
    ("gemm_quant_fused", ["gemm_w8a8", "gemm_w4a4"]),
    ("quantize_standalone", ["quantize_attn_qkv", "quantize_attn_out", "scale_quantize",
                             "quantize_act_int8", "quantize_and_pack", "aq_qtok", "aq_vquant",
                             "aq_kquant", "aq_qkv"]),
    ("conv_fp", ["cudnn_convolution", "conv2d", "implicit_gemm", "xmma_fprop", "convolution"]),
    ("gemm_fp", ["addmm", "cublas", "aten::linear", "wmma_tensorop", "cutlass_tensorop_f16"]),
    ("elementwise_misc", ["aten::add", "aten::mul", "aten::copy", "aten::to", "aten::contiguous",
                          "aten::cat", "catarraybatchedcopy", "aten::div", "aten::silu",
                          "aten::mean", "aten::sub", "aten::clamp", "aten::round", "aten::chunk",
                          "aten::permute", "aten::view", "aten::reshape", "aten::empty",
                          "aten::fill", "aten::index", "elementwise_kernel", "vectorized",
                          "direct_copy_kernel", "aten::embedding"]),
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
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=WARMUP, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()

    cat_time, kernel_time = {}, {}
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total
        if t <= 0:
            continue
        cat = categorize(evt.key)
        cat_time[cat] = cat_time.get(cat, 0.0) + t
        kernel_time[evt.key] = kernel_time.get(evt.key, 0.0) + t
    total = sum(cat_time.values()) or 1.0
    cat_pct = {k: round(v / total * 100, 2) for k, v in sorted(cat_time.items(), key=lambda x: -x[1])}
    top_kernels = sorted(kernel_time.items(), key=lambda x: -x[1])[:30]
    top_kernels = [dict(name=k[:160], us_total=round(v, 1), pct=round(v / total * 100, 2)) for k, v in top_kernels]

    del model, sampler, prof; torch.cuda.empty_cache()
    return cat_pct, top_kernels, total


os.makedirs(f"{HERE}/data", exist_ok=True)
profiles = {}
print(f"Corrected profiling pass @ b{BATCH} ({WARMUP} warm, {PROF_STEPS} profiled steps)\n")
for label, mode in VERS:
    cat_pct, top_k, total_us = run(mode)
    print(f"=== {label} === total GPU kernel time in window: {total_us/1000:.1f} ms")
    for cat, pct in cat_pct.items():
        print(f"  {cat:30s} {pct:6.2f}%")
    profiles[label] = dict(category_pct=cat_pct, top_kernels=top_k, total_us=round(total_us, 1))

with open(f"{HERE}/data/profile_breakdown_v2.json", "w") as f:
    json.dump(profiles, f, indent=2)
print(f"\nWROTE {HERE}/data/profile_breakdown_v2.json")
