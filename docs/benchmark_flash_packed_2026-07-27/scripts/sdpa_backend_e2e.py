"""E2E test of MODIFF_SDPA_BACKEND (math/flash/efficient) for fp16 mode: speed,
corrected kernel-category profile, and output-level numerical agreement (same seed,
same DDIM trajectory, compare final latents) vs the MATH baseline.
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
BACKENDS = ["math", "flash", "efficient"]

CATEGORY_RULES = [
    ("conv_int_fused", ["modiff"]),
    ("attention_flash", ["flash_attn_int8", "flash_attn_int4", "flash_attn"]),
    ("attention_sdpa_math_unfused", ["aten::bmm", "aten::_softmax", "softmax_warp_forward", "aten::baddbmm"]),
    ("attention_sdpa_fused", ["scaled_dot_product", "efficient_attention", "fmha", "flash_fwd", "attn::",
                             "fused_attn", "attention_kernel"]),
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


def build(backend):
    os.environ["MODIFF_QUANT_LINEAR"] = "0"; os.environ["MODIFF_QUANT_ATTN"] = "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ["MODIFF_SDPA_BACKEND"] = backend
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=None,
                          linear_backend="fp16")
    model, sampler = r._setup_model("fp16"); cond = r._cond_kwargs(model, BATCH)
    return r, model, sampler, cond


def run_speed(backend):
    r, model, sampler, cond = build(backend)

    def smp(S, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            return sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    cat_time = {}
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total
        if t <= 0:
            continue
        cat_time[categorize(evt.key)] = cat_time.get(categorize(evt.key), 0.0) + t
    total = sum(cat_time.values()) or 1.0
    cat_pct = {k: round(v / total * 100, 2) for k, v in sorted(cat_time.items(), key=lambda x: -x[1])}

    # same-seed correctness check: fixed small batch/steps, compare final latent
    small_batch = 8
    torch.manual_seed(1234)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        cond_small = r._cond_kwargs(model, small_batch)
        out, _ = sampler.sample(S=20, batch_size=small_batch, shape=r.shape, eta=0.0, verbose=False, x_T=torch.zeros(small_batch, *r.shape, device='cuda', dtype=torch.float16), **cond_small)

    del model, sampler, prof; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms), cat_pct, out.float().cpu()


os.makedirs(f"{HERE}/data", exist_ok=True)
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

results = {}
outputs = {}
print(f"fp16 SDPA backend A/B @ b{BATCH}\n{'backend':10} {'ms/step':>9} {'min':>8}")
for backend in BACKENDS:
    mean, mn, cat_pct, out = run_speed(backend)
    print(f"{backend:10} {mean:9.2f} {mn:8.2f}")
    results[backend] = dict(ms_step=round(mean, 2), min_ms=round(mn, 2), category_pct=cat_pct)
    outputs[backend] = out

ref = outputs["math"]
for backend in BACKENDS:
    if backend == "math":
        continue
    rel = (outputs[backend] - ref).norm() / ref.norm().clamp_min(1e-8)
    results[backend]["rel_l2_vs_math_x0"] = round(rel.item(), 6)
    print(f"{backend} rel-L2 vs math (fixed x_T, S=20, b=8): {rel.item():.6f}")

with open(f"{HERE}/data/sdpa_backend_e2e.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWROTE {HERE}/data/sdpa_backend_e2e.json")
