"""Corrected, fair, per-layer profiling pass.

Two fixes vs the previous rounds:
1. FAIRNESS: MODIFF_SDPA_BACKEND=flash is now set for EVERY mode, not just the fp16
   experiment. int8/int4's _resolve_flash autotune compares the custom flash kernel
   against exactly this shared _SDPA_CTX() fallback -- freezing that decision while
   the fallback was artificially slow (MATH) is what produced the earlier apples-to-
   oranges numbers. Re-running with the fast fallback available lets each block's
   autotune re-decide for real, so int8/int4 get the same "was this actually the
   right call" re-evaluation fp16 did.
2. GRANULARITY: per the request for layer-first-then-subdivide, every ResBlock/
   Attention block's forward is wrapped in a record_function label encoding its
   UNet level (L0-L4/Middle) and block kind (resblock/attention). Kernels are
   attributed to a label via the correlation-id link in the exported chrome trace
   (cudaLaunchKernel CPU event carries the same "correlation" id as the GPU kernel
   it launched; the CPU launch call's timestamp falls inside its enclosing
   record_function span, so matching on that -- not on the kernel's own device
   timestamp, which can lag due to queueing -- gives an unambiguous, non-overlapping
   attribution since blocks run strictly sequentially in this UNet).
"""
import os, sys, json, time, statistics, tempfile
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, record_function
import integration.benchmarks.benchmark_ldm as B
from integration.fused_ops.fused_resblock import FusedResBlock
from integration.fused_ops.quantized_std_attention import QuantizedStandardAttentionBlock
from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock

BATCH = 128
WARMUP, TIMED, RUNS = 30, 150, 5
PROF_STEPS = 8
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


def level_of(name):
    if "middle_block" in name:
        return "Middle"
    parts = name.split(".")
    idx = int(parts[-2]) if parts[-2].isdigit() else int(parts[-1])
    container = parts[-3] if parts[-2].isdigit() else parts[-2]
    if container == "input_blocks":
        return {0: "stem", 1: "L0", 2: "L0", 3: "L0->L1", 4: "L1", 5: "L1", 6: "L1->L2",
               7: "L2", 8: "L2", 9: "L2->L3", 10: "L3", 11: "L3", 12: "L3->L4",
               13: "L4", 14: "L4"}.get(idx, f"in{idx}")
    if container == "output_blocks":
        return {0: "L4", 1: "L4", 2: "L4->L3", 3: "L3", 4: "L3", 5: "L3->L2",
               6: "L2", 7: "L2", 8: "L2->L1", 9: "L1", 10: "L1", 11: "L1->L0",
               12: "L0", 13: "L0", 14: "L0"}.get(idx, f"out{idx}")
    return "other"


def wrap_blocks(model):
    """Monkeypatch forward on every ResBlock/Attention block to wrap it in a
    record_function(f"{level}|{kind}|{name}") span. Returns the list of patched
    (module, orig_forward) pairs so callers can restore them."""
    patched = []
    for name, m in model.named_modules():
        kind = None
        if isinstance(m, FusedResBlock):
            kind = "resblock"
        elif isinstance(m, (QuantizedStandardAttentionBlock, TokenMajorAttentionBlock)):
            kind = "attention"
        if kind is None:
            continue
        lvl = level_of(name)
        label = f"{lvl}|{kind}|{name}"
        orig_forward = m.forward
        def make_wrapped(orig, label=label):
            def wrapped(*args, **kwargs):
                with record_function(label):
                    return orig(*args, **kwargs)
            return wrapped
        m.forward = make_wrapped(orig_forward)
        patched.append((m, orig_forward))
    return patched


def unwrap_blocks(patched):
    for m, orig in patched:
        m.forward = orig


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ.pop("MODIFF_FLASH_PACKED", None)
    os.environ["MODIFF_SDPA_BACKEND"] = "flash"   # fairness fix: same fast fallback for every mode
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

    patched = wrap_blocks(model)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    unwrap_blocks(patched)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        trace_path = f.name
    prof.export_chrome_trace(trace_path)
    with open(trace_path) as f:
        trace = json.load(f)
    os.remove(trace_path)

    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    labels = []   # (label, ts_start, ts_end)
    launch_corr = {}  # correlation id -> ts (cudaLaunchKernel / cudaLaunchKernel_ptsz CPU event)
    kernels = []  # (correlation, name, dur_us)
    known_labels = set()
    for e in events:
        name = e.get("name", "")
        ph = e.get("ph")
        if ph != "X":
            continue
        cat = e.get("cat", "")
        ts, dur = e.get("ts", 0), e.get("dur", 0)
        if "|" in name and name.count("|") == 2 and cat in ("user_annotation", "cpu_op", ""):
            labels.append((name, ts, ts + dur)); known_labels.add(name)
        elif name.startswith("cudaLaunchKernel"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launch_corr[corr] = ts
        elif cat == "kernel" or e.get("args", {}).get("stream") is not None:
            corr = e.get("args", {}).get("correlation")
            kernels.append((corr, name, dur))

    labels.sort(key=lambda x: x[1])
    def find_label(ts):
        for lname, s, en in labels:
            if s <= ts <= en:
                return lname
        return "unattributed"

    level_cat_time = {}
    total_attributed = 0.0
    for corr, kname, dur in kernels:
        if kname in known_labels or corr is None:
            continue
        launch_ts = launch_corr.get(corr)
        lbl = find_label(launch_ts) if launch_ts is not None else "unattributed"
        lvl = lbl.split("|")[0] if "|" in lbl else lbl
        cat = categorize(kname)
        key = (lvl, cat)
        level_cat_time[key] = level_cat_time.get(key, 0.0) + dur
        total_attributed += dur

    breakdown = {}
    for (lvl, cat), dur in sorted(level_cat_time.items(), key=lambda x: -x[1]):
        breakdown.setdefault(lvl, {})[cat] = round(dur / total_attributed * 100, 3) if total_attributed else 0

    del model, sampler, prof; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms), breakdown, total_attributed


os.makedirs(f"{HERE}/data", exist_ok=True)
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

results = {}
print(f"Fair (flash-backend-everywhere) + per-level profiling @ b{BATCH}\n{'mode':16} {'ms/step':>9} {'min':>8}")
for label, mode in VERS:
    mean, mn, breakdown, total_us = run(mode)
    print(f"{label:16} {mean:9.2f} {mn:8.2f}   (attributed {total_us/1000:.1f} ms in profiled window)")
    results[label] = dict(ms_step=round(mean, 2), min_ms=round(mn, 2), level_category_pct=breakdown)

with open(f"{HERE}/data/layer_profile_v3.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWROTE {HERE}/data/layer_profile_v3.json")
