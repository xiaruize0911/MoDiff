"""Per-LAYER-TYPE kernel-pipeline benchmark: time the whole kernel pipeline each layer
type executes, at every real shape, in all 5 modes -- not isolated single-kernel
microbenchmarks.

Why pipelines and not single kernels: every optimization in this project fuses kernels
*together* (GN+SiLU+quantize into one launch, resize into the quantize, bias+residual into
the conv epilogue, ...). An isolated per-kernel number can't show that -- fusing two kernels
into one makes the "kernel" faster by definition while the honest question is whether the
LAYER got faster. So this benchmarks the real module forward (the actual production code
path, converted to each mode), and separately records which CUDA kernels fire inside it, so
the report can show both the pipeline's time and its composition.

Layer types measured, each at every distinct real input shape:
  resblock_plain    ResBlock, no resize   (GN+SiLU+quant -> conv -> GN+SiLU+mod+quant -> conv -> skip)
  resblock_updown   ResBlock with resize  (adds interpolate/avg_pool, fused into the quantize)
  attention         AttentionBlock        (GN -> qkv -> Q/K/V quant -> attn -> out quant -> proj)
  Reported per (layer type, shape, mode): pipeline ms, speedup vs the fp16 pipeline, and the
  kernel sequence with per-kernel ms inside that pipeline.

Writes data/layer_pipeline_bench.json.
"""
import os, sys, json, statistics, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src/taming-transformers"))
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B
from integration.fused_ops.fused_resblock import FusedResBlock
# Reuse profile_tree.py's classification so intra-layer roles match the whole-model tree's roles.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_tree import classify as classify_kernel, short_kernel_name
from ck_bench_stats import cuda_bench_stats, stability_verdict

HERE = "docs/final_report_2026-07-28"
BATCH = int(os.environ.get("LBENCH_BATCH", "128"))
# ROUNDS is what the reported CV/CI is computed over, so it went 5 -> 8: at 5 rounds the
# t-based 95% interval is 2.78 SEM wide, at 8 it is 2.36, for 60% more wall time.
WARM, ITERS, ROUNDS = 20, 60, 8
PROF_ITERS = 30
# AttentionBlock gets swapped for one of these wrappers by the mode conversion, so all
# three names must be matched or the quantized modes report zero attention layers.
ATTN_CLASSES = ("AttentionBlock", "TokenMajorAttentionBlock", "QuantizedStandardAttentionBlock")
_ALL_MODES = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"),
              ("int4_baseline", "int4_baseline"), ("int8_modiff", "int8"),
              ("int4_modiff", "int4")]
_MODE_FILTER = {x.strip() for x in os.environ.get("LBENCH_MODES", "").split(",") if x.strip()}
MODES = [m for m in _ALL_MODES if not _MODE_FILTER or m[0] in _MODE_FILTER]


def cuda_bench(fn, warm=WARM, iters=ITERS, rounds=ROUNDS):
    """(stats_dict_us, error). Delegates to ck_bench_stats so this suite reports the same
    distribution as the kernel and e2e suites instead of collapsing to a bare median.

    The previous version returned `median(round_medians)` and threw the samples away, so a
    stable number could not be distinguished from one that happened to land there once. The
    caller still stores that same median as `pipeline_us` for continuity with the data already
    published, and the full distribution alongside it.
    """
    return cuda_bench_stats(fn, warm=warm, iters=iters, rounds=rounds)


def kernel_sequence(fn, iters=PROF_ITERS):
    """Profile INSIDE one layer: every CUDA kernel this pipeline launches, its us per layer
    call, its share of the layer's own GPU time, and its role (same role taxonomy as
    profile_tree.py's whole-model tree, so the two views are directly comparable).

    Returns (kernels, roles, gpu_us_total) where `roles` aggregates the kernels by role so
    the report can show "inside this layer: 62% conv, 24% GN+quantize, 14% elementwise".
    """
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows, total = [], 0.0
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total
        if t <= 0:
            continue
        layer_type, role = classify_kernel(evt.key)
        rows.append(dict(kernel=short_kernel_name(evt.key), layer_type=layer_type, role=role,
                         us_per_layer_call=t / iters, calls=round(evt.count / iters, 2)))
        total += t / iters
    for r in rows:
        r["pct_of_layer"] = round(r["us_per_layer_call"] / total * 100, 2) if total else 0.0
        r["us_per_layer_call"] = round(r["us_per_layer_call"], 2)
    rows.sort(key=lambda r: -r["us_per_layer_call"])
    agg = {}
    for r in rows:
        a = agg.setdefault(r["role"], {"us": 0.0, "pct_of_layer": 0.0, "n_kernels": 0})
        a["us"] += r["us_per_layer_call"]
        a["pct_of_layer"] += r["pct_of_layer"]
        a["n_kernels"] += 1
    for a in agg.values():
        a["us"] = round(a["us"], 2)
        a["pct_of_layer"] = round(a["pct_of_layer"], 2)
    agg = dict(sorted(agg.items(), key=lambda kv: -kv[1]["us"]))
    return rows, agg, round(total, 2)


def collect_layers(mode_key):
    """Build the model in one mode and return the live layer instances + their real input
    shapes, captured by hooking an actual sampling step (so shapes can't drift from reality)."""
    quant = mode_key != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    for k in ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND"):
        os.environ.pop(k, None)
    calib = ("integration/calibration/int8_calibration.pt" if "int8" in mode_key else
             "integration/calibration/int4_calibration.pt" if "int4" in mode_key else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=2, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode_key)
    cond = r._cond_kwargs(model, BATCH)
    unet = model.model.diffusion_model

    seen = {}       # id(module) -> (x_shape, emb_shape)

    def mk(name, mod):
        def hook(m, inp, kw_or_out, out=None):
            if id(m) in seen:
                return
            args = inp
            x = args[0]
            emb = args[1] if len(args) > 1 and torch.is_tensor(args[1]) else None
            seen[id(m)] = (name, tuple(x.shape), None if emb is None else tuple(emb.shape))
        return hook

    handles = []
    for name, m in unet.named_modules():
        cls = type(m).__name__
        if isinstance(m, FusedResBlock) or cls in ATTN_CLASSES:
            handles.append(m.register_forward_hook(mk(name, m)))
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=2, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    for h in handles:
        h.remove()

    layers = []
    for name, m in unet.named_modules():
        if id(m) not in seen:
            continue
        nm, xs, es = seen[id(m)]
        cls = type(m).__name__
        if isinstance(m, FusedResBlock):
            kind = "resblock_updown" if getattr(m, "updown", False) else "resblock_plain"
        elif cls in ATTN_CLASSES:
            kind = "attention"
        else:
            continue
        layers.append(dict(kind=kind, name=nm, module=m, x_shape=xs, emb_shape=es))
    return model, sampler, layers


def main():
    # Wake the CUDA context without driving the board into its power cap before the
    # first (FP16) mode.  Each measured layer already gets WARM dedicated iterations;
    # the former 60 x 4096^3 GEMM burn made results depend strongly on mode order.
    bn = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(8):
        bn = bn @ bn
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    out = {"batch": BATCH, "modes": {}}
    for label, mode_key in MODES:
        print(f"\n########## {label} ##########")
        model, sampler, layers = collect_layers(mode_key)
        # Deduplicate: one representative per (kind, x_shape) -- same shape+kind runs the
        # same pipeline, and calls_per_step is recovered from how many instances collapsed.
        groups = collections.OrderedDict()
        for L in layers:
            key = (L["kind"], L["x_shape"], L["emb_shape"])
            groups.setdefault(key, []).append(L)
        rows = []
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            for (kind, xs, es), insts in groups.items():
                m = insts[0]["module"]
                x = torch.randn(*xs, device="cuda", dtype=torch.float16)
                x = x.contiguous(memory_format=torch.channels_last) if x.dim() == 4 else x
                emb = torch.randn(*es, device="cuda", dtype=torch.float16) if es else None
                fn = (lambda: m(x, emb)) if emb is not None else (lambda: m(x))
                st, err = cuda_bench(fn)
                us = st["median"] if st else None
                row = dict(kind=kind, x_shape=list(xs), emb_shape=list(es) if es else None,
                           n_instances=len(insts), example=insts[0]["name"],
                           pipeline_us=us, stats=st, stability=stability_verdict(st),
                           error=err)
                if us is not None:
                    ks, roles, gpu_us = kernel_sequence(fn)
                    row["kernels"] = ks              # per-kernel, with pct_of_layer
                    row["roles"] = roles             # aggregated by role: the intra-layer split
                    row["gpu_us_sum"] = gpu_us       # sum of kernel self-times in this layer
                    # Wall-clock vs summed kernel time: the gap is launch overhead / gaps.
                    row["gpu_busy_frac"] = round(gpu_us / us, 3) if us else None
                rows.append(row)
                print(f"  {kind:16s} {str(list(xs)):22s} x{len(insts):2d}  "
                      + (f"{us:8.1f} us  CV {st['cv_pct']:5.2f}%  "
                         f"+-{st['ci95_half']:.1f}" if st else 'ERR ' + str(err)))
                del x, emb
                torch.cuda.empty_cache()
        out["modes"][label] = rows
        del model, sampler, layers, groups
        torch.cuda.empty_cache()

    # Attach fp16-relative speedups per (kind, shape)
    base = {}
    for r in out["modes"].get("fp16", []):
        base[(r["kind"], tuple(r["x_shape"]))] = r["pipeline_us"]
    if base:
        for label, rows in out["modes"].items():
            for r in rows:
                b = base.get((r["kind"], tuple(r["x_shape"])))
                r["fp16_us"] = b
                r["speedup_vs_fp16"] = (round(b / r["pipeline_us"], 3)
                                        if b and r.get("pipeline_us") else None)

    out_path = os.environ.get("LBENCH_OUT", f"{HERE}/data/layer_pipeline_bench.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
