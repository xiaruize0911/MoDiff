"""Hierarchical e2e profile, attributed by the ATen op that LAUNCHED each kernel.

Why this exists (a real misattribution it fixes). Classifying CUDA kernels by name alone
cannot tell you which layer type they belong to, because the same kernel serves different
callers. Concretely, in fp16 mode PyTorch's MATH SDPA computes QK^T and AV as `aten::bmm`,
and cuBLAS dispatches those to `cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_...>`
-- a name indistinguishable from a plain GEMM. Name-based rules therefore billed 44 ms/step
of fp16 ATTENTION work to "Linear-GEMM", which then produced an impossible ~6.4x
"Linear speedup" for int8 (int8/int4 keep QK^T+AV inside their flash kernel, so their
Linear-GEMM bucket holds only qkv/proj -- the two buckets were not the same set of ops).

Fix: export a Chrome trace and join CPU ops to GPU kernels on the trace's `External id`
arg, which PyTorch sets to the same value on an ATen op and the kernels it launches. Then
the top level of the tree is the real caller (aten::bmm -> Attention, aten::conv2d -> Conv,
aten::linear/addmm -> Linear-GEMM, ...), and the kernel name is only used to refine the role
inside it. Kernels with no CPU parent (a few CUDA-graph/internal launches) are reported
separately instead of being silently folded in.

Writes data/profile_tree_by_caller.json.
"""
import os, sys, json, time, statistics, collections, re
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B
from profile_tree import classify as classify_by_name, short_kernel_name

HERE = "docs/final_report_2026-07-28"
BATCH, WARMUP, TIMED, RUNS, PROF_STEPS = 128, 30, 150, 5, 10
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]

# ATen op -> layer type. Checked against the ops that actually appear in these traces.
CALLER_RULES = [
    ("Attention", ["aten::bmm", "aten::baddbmm", "aten::_softmax", "aten::softmax",
                   "scaled_dot_product", "aten::_scaled_dot_product",
                   "flash_attn", "quantize_attn", "attn_"]),
    ("Conv", ["aten::conv2d", "aten::convolution", "aten::_convolution",
              "aten::cudnn_convolution", "conv2d_int8", "conv2d_int4", "conv2d_evt"]),
    ("Linear-GEMM", ["aten::linear", "aten::addmm", "aten::mm", "aten::matmul",
                     "gemm_w8a8", "gemm_w4a4", "w8a8_gemm", "w4a4_gemm", "awq"]),
    ("Normalization", ["aten::group_norm", "aten::native_group_norm", "aten::layer_norm",
                       "group_norm_silu", "gn_"]),
    ("Resize", ["aten::upsample_nearest2d", "aten::avg_pool2d", "upsample2x", "avgpool2x",
                "aten::interpolate"]),
    ("Quantize", ["quantize", "quant_", "dequant", "layout_transform", "pack"]),
    ("Elementwise-Cast", ["aten::cat", "cat2_", "aten::add", "aten::mul", "aten::silu",
                          "aten::sigmoid", "aten::to", "aten::copy_", "aten::_to_copy",
                          "aten::contiguous", "aten::clone", "aten::empty", "aten::fill_",
                          "aten::max", "aten::amax", "aten::abs", "aten::div", "aten::sub",
                          "aten::chunk", "aten::split", "aten::view", "aten::reshape",
                          "aten::permute", "aten::transpose", "aten::index", "aten::slice",
                          "elementwise", "Memcpy", "Memset"]),
]


def classify_caller(op):
    low = op.lower()
    for layer, pats in CALLER_RULES:
        for p in pats:
            if p.lower() in low:
                return layer
    return None


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

    trace = f"/tmp/trace_{mode}.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace)
    ev = json.load(open(trace))["traceEvents"]
    os.remove(trace)

    # Join on "External id": PyTorch tags an ATen op and the kernels it launches with the
    # same value. Prefer the DEEPEST (shortest-duration) CPU op per id so we get e.g.
    # aten::bmm rather than an enclosing aten::matmul wrapper.
    cpu = {}
    for e in ev:
        if e.get("cat") != "cpu_op":
            continue
        eid = (e.get("args") or {}).get("External id")
        if eid is None:
            continue
        prev = cpu.get(eid)
        if prev is None or e.get("dur", 0) < prev[1]:
            cpu[eid] = (e["name"], e.get("dur", 0))

    agg = collections.defaultdict(lambda: {"us": 0.0, "n": 0})
    # tracked for transparency: how much time was attributed by NAME rather than by caller
    no_parent = {"us": 0.0, "n": 0, "kernels": collections.Counter()}
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = e.get("dur", 0.0)
        if dur <= 0:
            continue
        eid = (e.get("args") or {}).get("External id")
        parent = cpu.get(eid, (None, 0))[0] if eid is not None else None
        # HYBRID attribution, and both halves are needed:
        #   * ATen-dispatched kernels HAVE a CPU parent -> use the caller. This is the half
        #     that fixes bmm's GEMMs being billed to Linear-GEMM instead of Attention.
        #   * Our own kernels are launched straight from Python through pybind, so they
        #     never create an aten:: op and have no External id to join on. Left as
        #     "orphan" they were 90% of a quantized step. Their names are unambiguous
        #     (group_norm_silu_*, flash_attn_int8_*, ...), so name-based is correct there.
        via = "caller"
        layer = classify_caller(parent) if parent else None
        if layer is None:
            layer = classify_by_name(e["name"])[0]
            via = "name(no-aten-parent)" if not parent else "name(unknown-caller)"
            no_parent["us"] += dur; no_parent["n"] += 1
            no_parent["kernels"][short_kernel_name(e["name"])] += dur
        _, role = classify_by_name(e["name"])
        key = (layer, role, short_kernel_name(e["name"]), parent or "(pybind, no aten op)", via)
        agg[key]["us"] += dur
        agg[key]["n"] += 1

    total_us = sum(v["us"] for v in agg.values())
    tree = {}
    for (layer, role, kern, parent, via), v in agg.items():
        ms_step = v["us"] / total_us * mean_ms
        n = tree.setdefault(layer, {"ms_step": 0.0, "roles": {}})
        rn = n["roles"].setdefault(role, {"ms_step": 0.0, "kernels": []})
        n["ms_step"] += ms_step; rn["ms_step"] += ms_step
        rn["kernels"].append({"kernel": kern, "called_by": parent, "attributed_via": via,
                              "ms_step": round(ms_step, 4),
                              "calls_per_step": round(v["n"] / PROF_STEPS, 2)})
    for layer, n in tree.items():
        n["pct_of_total"] = round(n["ms_step"] / mean_ms * 100, 2)
        n["ms_step"] = round(n["ms_step"], 4)
        for role, rn in n["roles"].items():
            rn["pct_of_total"] = round(rn["ms_step"] / mean_ms * 100, 2)
            rn["ms_step"] = round(rn["ms_step"], 4)
            rn["kernels"].sort(key=lambda x: -x["ms_step"])
        n["roles"] = dict(sorted(n["roles"].items(), key=lambda kv: -kv[1]["ms_step"]))
    tree = dict(sorted(tree.items(), key=lambda kv: -kv[1]["ms_step"]))

    del model, sampler, prof
    torch.cuda.empty_cache()
    return dict(ms_step=round(mean_ms, 2),
                name_attributed_ms_step=round(no_parent["us"] / total_us * mean_ms, 4),
                name_attributed_pct=round(no_parent["us"] / total_us * 100, 2),
                name_attributed_top=[k for k, _ in no_parent["kernels"].most_common(6)],
                tree=tree)


def main():
    bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()
    out = {}
    for label, mode in VERS:
        res = run(mode)
        out[label] = res
        print(f"\n=== {label}: {res['ms_step']} ms/step "
              f"(name-attributed {res['name_attributed_ms_step']} ms = "
              f"{res['name_attributed_pct']}%) ===")
        for L, n in res["tree"].items():
            print(f"  {L:22s} {n['ms_step']:8.2f} ms {n['pct_of_total']:6.1f}%")
    with open(f"{HERE}/data/profile_tree_by_caller.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {HERE}/data/profile_tree_by_caller.json")


if __name__ == "__main__":
    main()
