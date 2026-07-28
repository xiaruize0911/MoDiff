"""Hierarchical (tree) e2e profile: EVERY CUDA kernel, classified 3 levels deep.

    Level 1  layer type      Conv / Attention / Linear-GEMM / Normalization / Resize /
                             Quantize / Elementwise-Cast / Other
    Level 2  role            what the kernel does inside that layer type
                             (e.g. Conv -> "int8 implicit-GEMM (EVT-fused epilogue)")
    Level 3  kernel          the individual CUDA kernel, with ms/step and calls/step

Unlike glue_breakdown_detail.py (which kept names only for the non-Conv/Attn/Linear
"glue" bucket), this keeps every kernel in every bucket, so the tree accounts for 100%
of measured GPU time and nothing hides in a coarse bucket.

Same measurement methodology as the other scripts here: same-session wall-clock ms/step
(clock burn-in -> warmup -> RUNS x TIMED steps), then a torch.profiler window whose
CUDA-device-only self-times are scaled onto that wall-clock ms/step.

Writes data/profile_tree.json.
"""
import os, sys, json, time, statistics, re
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, TIMED, RUNS = 30, 150, 5
PROF_STEPS = 20
HERE = "docs/final_report_2026-07-28"
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]

# ---------------------------------------------------------------------------
# Classification: ordered (layer_type, role, [substring patterns]).
# FIRST match wins, so put specific patterns before general ones. Every pattern was
# checked against real profiler output; the catch-all at the end is deliberately loud
# ("Other / unclassified") so a new kernel can never silently vanish into a bucket.
# ---------------------------------------------------------------------------
RULES = [
    # ---- Attention -------------------------------------------------------
    ("Attention", "int8/int4 flash kernel (fused QK^T+softmax+AV)",
     ["flash_attn_int8", "flash_attn_int4", "flash_fwd", "flash_attn"]),
    ("Attention", "Q/K/V quantize (packed, static scales)",
     ["aq_qtok", "aq_kquant"]),
    ("Attention", "V quantize + transpose to AV layout",
     ["aq_vquant"]),
    ("Attention", "fp16 SDPA (unfused math backend: BMM + softmax)",
     ["scaled_dot_product", "efficient_attention", "fmha", "aten::bmm", "baddbmm",
      "_softmax", "softmax_warp"]),
    ("Attention", "fused GroupNorm->QKV projection",
     ["fused_gn_qkv"]),
    # ---- Conv ------------------------------------------------------------
    ("Conv", "int8/int4 implicit-GEMM conv (CUTLASS, EVT-fused epilogue)",
     ["implicitgemmconvolutionfusionpersample", "implicit_gemm_conv_evt",
      "conv2devt", "implicitgemmconvolution", "implicit_gemm"]),
    ("Conv", "conv dequant/store epilogue (separate pass)",
     ["scale_accumulate", "scale_store", "conv_epilogue"]),
    ("Conv", "fp16 cuDNN conv",
     ["xmma_fprop", "fprop_optimized", "cudnn_convolution", "nhwcaddpadding",
      "wgrad", "dgrad", "conv2d"]),
    # ---- Linear / GEMM ---------------------------------------------------
    ("Linear-GEMM", "int8/int4 quantized GEMM (W8A8 / W4A4)",
     ["gemm_w8a8", "gemm_w4a4", "dense_kernel0", "awq"]),
    ("Linear-GEMM", "fp16 tensor-core GEMM (cuBLAS)",
     ["ampere_fp16_s1688", "ampere_fp16_s16816", "sm80_xmma_gemm", "xmma_gemm",
      "cublas", "addmm", "wmma_tensorop", "tensorop_f16", "cutlass"]),
    # ---- Normalization ---------------------------------------------------
    ("Normalization", "GN+SiLU+quantize fused (K1 path: one kernel, int8/int4 out)",
     ["group_norm_silu_quantize_nhwc", "group_norm_silu_quantize_pack_nhwc",
      "group_norm_silu_dequant_quantize"]),
    ("Normalization", "MoDiff GN+SiLU+delta-quantize+cache apply",
     ["gn_apply_delta_quantize", "group_norm_silu_delta_quantize"]),
    ("Normalization", "GN group-statistics reduction (mean/var; deliberately scalar)",
     ["gn_group_stats", "gn_stats_"]),
    ("Normalization", "GN+SiLU only (fp16 out; updown blocks + fp16 mode)",
     ["group_norm_silu_nhwc", "group_norm", "layer_norm"]),
    # ---- Resize ----------------------------------------------------------
    ("Resize", "upsample(nearest,2x)+quantize FUSED",
     ["upsample2x_quantize"]),
    ("Resize", "avg_pool(2x2)+quantize FUSED",
     ["avgpool2x_quantize"]),
    ("Resize", "nearest upsample (unfused; x_upd path)",
     ["upsample_nearest"]),
    ("Resize", "avg_pool 2x2 (unfused; x_upd path)",
     ["avg_pool"]),
    # ---- Quantize (standalone, not folded into another op) ---------------
    ("Quantize", "MoDiff delta-quantize + a_hat cache update",
     ["static_quantize_and_update_ahat", "static_quantize_pack_and_update_ahat",
      "update_ahat"]),
    ("Quantize", "static activation quantize (cache-free 'noahat')",
     ["static_quantize_int8_noahat", "static_quantize_pack_int4_noahat",
      "noahat"]),
    ("Quantize", "activation quantize / int4 pack (standalone)",
     ["scale_quantize", "quant_act_int4_pack", "quantize_pack", "quantize"]),
    ("Quantize", "NCHW<->channels-last layout transform for the int kernels",
     ["layout_transform", "ncw_to", "cl_to_"]),
    # ---- Elementwise / cast / copy ---------------------------------------
    ("Elementwise-Cast", "skip-concat (decoder): specialized 2-tensor channels-last",
     ["cat2_channels_last"]),
    ("Elementwise-Cast", "skip-concat (decoder): generic torch.cat",
     ["catarraybatchedcopy"]),
    ("Elementwise-Cast", "residual add",
     ["cudafunctor_add", "functor_add"]),
    ("Elementwise-Cast", "dtype cast / device copy",
     ["direct_copy", "loadwithcast", "storewithcast", "copy_"]),
    ("Elementwise-Cast", "SiLU / activation (standalone)",
     ["silu", "sigmoid", "gelu"]),
    ("Elementwise-Cast", "reduction (amax/absmax for dynamic scales)",
     ["reduce_kernel", "maxnanfunctor", "absmax"]),
    ("Elementwise-Cast", "fill / zero-init",
     ["fillfunctor", "fill_"]),
    ("Elementwise-Cast", "other elementwise",
     ["elementwise_kernel", "vectorized_elementwise", "unrolled_elementwise"]),
]


def classify(name):
    low = name.lower()
    for layer, role, pats in RULES:
        for p in pats:
            if p in low:
                return layer, role
    return "Other / unclassified", "unclassified (investigate)"


def short_kernel_name(k):
    """Trim C++ template/argument noise so the tree is readable, keeping the identifier."""
    s = k
    s = re.sub(r'\bvoid\s+', '', s)
    s = s.split('(')[0]
    s = re.sub(r'<[^<>]*>', '', s)
    s = re.sub(r'<.*', '', s)
    s = s.replace('at::native::', '').replace('(anonymous namespace)::', '')
    s = s.strip(': ')
    return s[-90:] if len(s) > 90 else s


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    for k in ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND"):
        os.environ.pop(k, None)
    calib = ("integration/calibration/int8_calibration.pt" if "int8" in mode else
             "integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32),
                          calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode)
    cond = r._cond_kwargs(model, BATCH)

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

    ktime, kcalls, total = {}, {}, 0.0
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total
        if t <= 0:
            continue
        ktime[evt.key] = ktime.get(evt.key, 0.0) + t
        kcalls[evt.key] = kcalls.get(evt.key, 0) + evt.count
        total += t

    # Build the 3-level tree, converting profiler shares onto wall-clock ms/step.
    tree = {}
    for k, v in ktime.items():
        layer, role = classify(k)
        node = tree.setdefault(layer, {"ms_step": 0.0, "roles": {}})
        rnode = node["roles"].setdefault(role, {"ms_step": 0.0, "kernels": []})
        ms_step = v / total * mean_ms
        node["ms_step"] += ms_step
        rnode["ms_step"] += ms_step
        rnode["kernels"].append({
            "kernel": short_kernel_name(k),
            "kernel_full": k,
            "ms_step": round(ms_step, 4),
            "pct_of_total": round(v / total * 100, 3),
            "calls_per_step": round(kcalls[k] / PROF_STEPS, 2),
        })
    for layer, node in tree.items():
        node["pct_of_total"] = round(node["ms_step"] / mean_ms * 100, 2)
        node["ms_step"] = round(node["ms_step"], 4)
        for role, rnode in node["roles"].items():
            rnode["pct_of_total"] = round(rnode["ms_step"] / mean_ms * 100, 2)
            rnode["ms_step"] = round(rnode["ms_step"], 4)
            rnode["kernels"].sort(key=lambda x: -x["ms_step"])
        node["roles"] = dict(sorted(node["roles"].items(), key=lambda kv: -kv[1]["ms_step"]))
    tree = dict(sorted(tree.items(), key=lambda kv: -kv[1]["ms_step"]))

    accounted = sum(n["ms_step"] for n in tree.values())
    del model, sampler, prof
    torch.cuda.empty_cache()
    return dict(ms_step=round(mean_ms, 2),
                gpu_accounted_ms_step=round(accounted, 3),
                n_distinct_kernels=len(ktime), tree=tree)


def main():
    bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize()
    del bn
    torch.cuda.empty_cache()

    results = {}
    print(f"Hierarchical profile @ b{BATCH}")
    for label, mode in VERS:
        res = run(mode)
        results[label] = res
        print(f"\n=== {label}: {res['ms_step']} ms/step, {res['n_distinct_kernels']} distinct kernels ===")
        for layer, node in res["tree"].items():
            print(f"  {layer:22s} {node['ms_step']:8.2f} ms  {node['pct_of_total']:5.1f}%")
            for role, rnode in node["roles"].items():
                print(f"    - {role[:66]:66s} {rnode['ms_step']:7.2f} ms {rnode['pct_of_total']:5.1f}%")
    with open(f"{HERE}/data/profile_tree.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWROTE {HERE}/data/profile_tree.json")


if __name__ == "__main__":
    main()
