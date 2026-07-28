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
    # fused_gn_qkv.cu's kernels come from ImplicitGemmConvolutionFusionPerSample /
    # ..._evt (csrc/kernels/norm/implicit_gemm_fusion_persample*.h). Their mangled names
    # contain "ImplicitGemm", so this MUST be matched before the Conv rules below or the
    # attention QKV projection gets billed to Conv -- which it was, hiding 14.3 ms of
    # fp16-mode attention time under "Conv" until a ground-truth name dump caught it.
    ("Attention", "fused GroupNorm->QKV projection (CUTLASS per-sample fusion)",
     ["implicitgemmconvolutionfusionpersample", "implicitgemmfpropfusion", "fused_gn_qkv"]),
    ("Attention", "attention output quantize (for the proj GEMM)",
     ["quant_attn_out"]),
    # ---- Conv ------------------------------------------------------------
    # ORDER MATTERS AND IS LOAD-BEARING HERE. cuDNN's own fp16 conv kernels have
    # "implicit_gemm" in their mangled names too (e.g. cudnn::cnn::implicit_gemm<...>,
    # and the *_execute_kernel__5x_cudnn family), so a bare "implicit_gemm" pattern
    # silently mislabels every fp16 cuDNN conv as one of OUR int8/int4 CUTLASS convs --
    # which it did, until this was caught by seeing "int8/int4 CUTLASS conv" show up at
    # 39% inside an fp16-mode layer. "cudnn" appears in cuDNN kernel names and never in
    # ours, so match cuDNN FIRST and use it as the discriminator.
    # "cudnn" appears in every cuDNN kernel name and in none of ours, so it is the
    # discriminator. NOTE: do NOT add "fprop_optimized" here -- our own int8 CUTLASS
    # convs are named cutlass_tensorop_s8_i8816fprop_optimized_*, so that pattern would
    # steal them.
    ("Conv", "fp16 cuDNN conv",
     ["cudnn", "xmma_fprop", "nhwcaddpadding", "wgrad", "dgrad"]),
    ("Conv", "quantized implicit-GEMM conv (CUTLASS, EVT-fused epilogue)",
     ["implicit_gemm_conv_evt", "conv2devt", "implicitgemmconvolution",
      "implicit_gemm", "conv2dproblemsize", "conv2dfprop"]),
    ("Conv", "conv dequant/store epilogue (separate pass)",
     ["scale_accumulate", "scale_store", "conv_epilogue"]),
    ("Conv", "conv (other / generic aten)", ["conv2d", "convolution"]),
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
    ("Normalization", "GN accumulate/finalize (split two-pass helper kernels)",
     ["gn_accum", "gn_finalize"]),
    ("Normalization", "PyTorch native GroupNorm internals (fp16 fallback path)",
     ["rowwisemoments", "computefusedparams"]),
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
    ("Quantize", "MoDiff dequant + accumulate (int4 o_hat return path)",
     ["dequant_accumulate"]),
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
    # ---- Memory ops / sampler-side (small, but real GPU time: keep them out of
    # "unclassified" so that bucket stays a genuine alarm for unknown kernels) ----
    ("Memory-op", "memset / memcpy",
     ["memset", "memcpy"]),
    ("Sampler-side", "DDIM schedule indexing / noise generation",
     ["indexselect", "distribution_elementwise", "index_select"]),
]


def classify(name):
    low = name.lower()
    for layer, role, pats in RULES:
        for p in pats:
            if p in low:
                return layer, role
    return "Other / unclassified", "unclassified (investigate)"


def short_kernel_name(k, maxlen=58):
    """Reduce a CUDA kernel's C++ symbol to the most informative identifier.

    Naive truncation does not work here: the useful name often lives INSIDE the template
    arguments (cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_...> -- the outer
    "Kernel2" is a generic launcher shared by every CUTLASS GEMM), and tail-truncating a
    long at::native::elementwise_kernel<...> symbol yields an empty or meaningless string.
    So: unwrap generic launchers to their instantiation, and for ATen elementwise wrappers
    pull out the functor that says what the kernel actually does.
    """
    s = k.strip()
    s = re.sub(r'^void\s+', '', s)
    # Strip namespace noise BEFORE splitting on '(' -- a symbol beginning with
    # "(anonymous namespace)::" would otherwise split to an empty string.
    for ns in ('(anonymous namespace)::', 'at::native::', 'at::cuda::detail::'):
        s = s.replace(ns, '')

    # Itanium-mangled symbol (no readable arg list): pull out the longest readable
    # CamelCase identifier, which is the instantiated kernel/class name.
    if s.startswith('_Z'):
        # Itanium mangling encodes each identifier as <length><name>, so splitting on
        # digits isolates the real identifiers. Picking the longest CamelCase token that
        # is not mangling scaffolding ("ZN", "KernelIN", ...) recovers e.g.
        # ImplicitGemmConvolutionEVT from _ZN7cutlass6KernelIN6modiff26ImplicitGemm...
        toks = [t for t in re.split(r'\d+', s) if t]
        cand = [t for t in toks
                if re.match(r'^[A-Z][A-Za-z_]{9,}$', t) and not t.startswith(('ZN', 'Kernel'))]
        if cand:
            # FIRST, not longest: the outermost template (the actual kernel/class being
            # launched) appears earliest in the mangled name, while the longest token is
            # typically some inner iterator type (PredicatedScaleBiasVectorAccessIterator).
            best = cand[0]
            best = re.sub(r'(INS_?|ILi|IN|EEE?|ILb)$', '', best)  # trim mangling tail
            return best[:maxlen]
        isl = re.findall(r'[A-Z][A-Za-z0-9_]{9,}', s)
        if isl:
            return re.sub(r'(INS|ILi|IN\d|EEE?).*$', '', max(isl, key=len))[:maxlen]

    # ATen elementwise/unrolled wrappers: the functor name is the informative part.
    m = re.search(r'(CUDAFunctor_\w+|FillFunctor|MaxNanFunctor|[A-Za-z_]*direct_copy_kernel_cuda'
                  r'|LoadWithCast|StoreWithCast)', s)
    if 'elementwise_kernel' in s and m:
        base = 'elementwise_kernel'
        if 'vectorized' in s:
            base = 'vectorized_elementwise_kernel'
        elif 'unrolled' in s:
            base = 'unrolled_elementwise_kernel'
        return f"{base}[{m.group(1)}]"[:maxlen]

    # Generic CUTLASS/cuDNN launchers: unwrap to the instantiated kernel name.
    m = re.match(r'(?:cutlass(?:__\dx_cudnn)?::)?Kernel2?<([^<>,]+)', s)
    if m:
        return m.group(1).strip()[:maxlen]

    # Otherwise: drop the argument list, then strip template args and namespaces.
    s = s.split('(')[0]
    s = re.sub(r'<[^<>]*>', '', s)
    s = re.sub(r'<.*', '', s)
    s = s.strip(': ')
    if not s:
        # Fully mangled/unnameable: fall back to any readable CamelCase island.
        m = re.search(r'([A-Z][A-Za-z0-9_]{12,})', k)
        s = m.group(1) if m else k[:maxlen]
    return s[:maxlen]


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

    # Wall-clock the profiled window too: the sum of kernel self-times divided by this is
    # a REAL measurement (how much of the window the GPU was actually executing kernels,
    # vs idle in launch gaps). Note the profiler adds its own overhead, so treat this as a
    # lower bound on the un-profiled run's busy fraction.
    torch.cuda.synchronize()
    _t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    prof_window_ms = (time.time() - _t0) * 1000.0

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

    # NOT a validation metric -- this is an identity. Every kernel's ms_step is
    # (its share of `total`) x mean_ms and classify() has a catch-all, so the shares always
    # sum to 1 and this always equals mean_ms. Kept only as an arithmetic self-check.
    accounted = sum(n["ms_step"] for n in tree.values())

    # These two ARE measurements:
    #  gpu_busy_frac  = fraction of the profiled window the GPU spent executing kernels
    #                   (the rest is launch gaps / idle). total is in us, window in ms.
    #  unclassified_* = how much time landed in the catch-all bucket, i.e. whether the
    #                   classification actually covered the kernels that ran.
    gpu_busy_frac = (total / 1000.0) / prof_window_ms if prof_window_ms > 0 else None
    unk = tree.get("Other / unclassified")
    del model, sampler, prof
    torch.cuda.empty_cache()
    return dict(ms_step=round(mean_ms, 2),
                gpu_busy_frac_profiled_window=(round(gpu_busy_frac, 4) if gpu_busy_frac else None),
                unclassified_ms_step=round(unk["ms_step"], 4) if unk else 0.0,
                unclassified_n_kernels=sum(len(r["kernels"]) for r in unk["roles"].values()) if unk else 0,
                sum_of_tree_equals_wall_clock=round(accounted, 3),
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
