"""Kernel-level benchmark + profile for every real kernel call, all modes in ONE process.

Covers three of the final report's five suites -- attention, conv and linear kernels -- plus the
norm/quantize kernels, since excluding them would understate where the time actually goes.

Two design decisions, both forced by what the code actually does:

1. SHAPES ARE CAPTURED, NEVER LISTED. The older per-report scripts carry hardcoded shape
   tables that no longer match this model at all: docs/benchmark_5mode_2026-07-2x/scripts/
   conv_kernel.py benches Cin=128/256/512 at 64x64..8x8, while the churches UNet runs Cin in
   {4,192,384,576,768,1152} at 32x32..2x2 and never sees a 64x64 conv or a 128-channel one;
   integration/benchmarks/bench_attn_kernel.py lists qkv as 192->576 where the real projection
   is 192->768. A hardcoded table reports on shapes the model does not have.

2. KERNELS ARE CAPTURED AT THE C++ ENTRY POINT, NOT AT THE MODULE. A forward hook on conv/
   linear modules cannot see the quantized path: the fused ResBlock calls
   modiff_cutlass.conv2d_int8_evt_bias_residual_fp16 and the gemm_w8a8_awq_* family DIRECTLY,
   bypassing module.forward(), so a module-level hook captures 33 conv shapes in fp16 but only
   13 in int8_baseline -- and those 13 are the leftovers that stayed fp16. Wrapping the entry
   points instead captures all of them, and puts the three modes at the same level.
   (isinstance(nn.Conv2d) is doubly useless here: the conversion REPLACES the classes with
   OptimizedInt8Conv2d/QuantLinearWxAx, which do not subclass the torch types.)

Real call arguments are intercepted during a live sample and REPLAYED verbatim, rather than
synthesized from shapes: each attention shape alone dispatches a different entry with its own
packing convention (qi8_kv_static_qout_hd24 for T=1024/hd=24, qi8_kv_static_qout for
T=256/64/hd=48, qi8packed_small_qout for T=16/4/hd=96 with packed qkv), and hand-building valid
inputs for those is where a kernel benchmark silently starts measuring the wrong thing.
Captured tensors are parked on CPU (~7 GB worst case against ~450 GB free) and moved back one
signature at a time, so GPU memory stays bounded however many signatures exist.

FP16 has no such entry points -- its kernels ARE torch's -- so F.conv2d / F.linear /
scaled_dot_product_attention are wrapped through the same door and land in the same tables.

Statistics come from ck_bench_stats: warmup, then rounds x iters with each round's median as
one sample, reported as mean +- t-based 95% CI with CV and spread.

Writes ONE json (default data/kernel_suites.json): per mode, per suite, one entry per
(entry point, argument shapes) with its full timing distribution and per-kernel profile.
"""
import argparse
import collections
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src/taming-transformers"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, DeviceType

import modiff_cutlass as mc
import integration.benchmarks.benchmark_ldm as B
from ck_bench_stats import cuda_bench_stats, stability_verdict
from profile_tree import classify as classify_kernel, short_kernel_name

#: Selectable so the same suite can be run for the MoDiff modes ("int8"/"int4" in benchmark_ldm's
#: naming) without a second copy of the harness -- same pattern as layer_pipeline_bench's
#: LBENCH_MODES. Default is unchanged, i.e. the baseline modes the 08-01 report used.
_ALL_MODES = ["fp16", "int8_baseline", "int4_baseline", "int8", "int4"]
_MODE_FILTER = [x.strip() for x in os.environ.get("KBENCH_MODES", "").split(",") if x.strip()]
MODES = [m for m in _MODE_FILTER if m in _ALL_MODES] or _ALL_MODES[:3]
#: The MoDiff modes read the same calibration artifact as their baseline; the delta scale is
#: computed at runtime (MODIFF_DELTA_MODE=dynamic) and is not stored in this file.
#: Resolved through CALIBRATION_PREFERENCE rather than hardcoded. The hardcoded value was
#: `int*_calibration.pt`, which benchmark_ldm's own preference comment grades at latent relL2
#: 0.882 (int8) / 3.023 (int4) -- "worse than useless" -- and demoted to LAST RESORT on
#: 2026-08-12 when the qdiff files landed. It was also derived from the 856-byte STUB
#: checkpoint, and this tree has carried the real 2.7 GB checkpoint since 2026-08-04, so the
#: "stub" premise several of these harnesses still state in prose is stale too. Fixed across
#: five harnesses on 2026-08-13 after the same defect was found in e2e_three_mode_bench;
#: for LATENCY a scale is only a multiplier (measured: <=0.33% across all arms), but the stub
#: file also carries 37 emb-Linear scales the qdiff file does not, which puts those layers on
#: a static scale here and a per-call dynamic absmax in the shipped path -- a different kernel
#: route, not a different number.
class CALIB:
    """Drop-in for the dict this replaced, so call sites keep reading CALIB.get(mode)."""
    @staticmethod
    def get(mode):
        return B._default_calibration_path(mode)
QUANT_ENV = {"MODIFF_QUANT_LINEAR": "1", "MODIFF_QUANT_ATTN": "1",
             "MODIFF_QUANT_ATTN_STATIC": "1", "MODIFF_QATTN_FLASH": "1",
             "MODIFF_FLASH_GATE": "on", "MODIFF_LINEAR_OUT_I8": "0"}


def suite_of(entry):
    """Map a kernel entry point to one of the report's suites.

    ORDER MATTERS, and `fused_gn_qkv` is why. It is fp16's qkv PROJECTION with the GroupNorm folded into
    its mainloop, and its name contains none of "gemm", "linear", "conv2d" or "group_norm" -- so it used
    to fall through every test to "other". That put fp16's two largest projections (T=1024 and T=256,
    31.96 ms/sample together) in a different suite from the quantized arms' counterparts, which are
    ordinary GEMMs in `linear`. Three suite ratios were accounting artifacts of that one fallthrough:
    linear read 0.61x, `other` read 3.77x, `norm_quantize` read 0.64x. Classified as `linear` here
    because that is where the work it replaces lands in every other arm.

    It is still not a CLEAN comparison, and no classification can make it one: this kernel also does the
    GroupNorm, whose quantized counterpart is a separate `group_norm_silu_quantize_nhwc` record over in
    `norm_quantize`. Suite totals are therefore not a speedup denominator -- see
    docs/OPEN_ITEMS.md A1/A2. The per-layer matched tables are.

    Captures written before 2026-08-16 have these records under "other"; the report scripts detect them
    by entry name wherever they sit, so old JSON regenerates correctly.
    """
    e = entry.lower()
    if "flash_attn" in e or "sdpa" in e:
        return "attention"
    if "gn_qkv" in e:                        # before the conv/gemm tests: matches none of them
        return "linear"
    if "conv2d" in e:
        return "conv"
    if "gemm" in e or "linear" in e:
        return "linear"
    if "group_norm" in e or "quant" in e:
        return "norm_quantize"
    return "other"


def set_env(mode):
    quant = mode != "fp16"
    for k, v in QUANT_ENV.items():
        os.environ[k] = v if quant else ("0" if k in ("MODIFF_QUANT_LINEAR",
                                                      "MODIFF_QUANT_ATTN") else v)
    for k in ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND"):
        os.environ.pop(k, None)


# ---------------------------------------------------------------- arg parking
def park(args):
    """Move a captured arg list to CPU, remembering how to put each tensor back.

    channels_last has to be recorded explicitly: several conv entries TORCH_CHECK it, and a
    .cpu() round trip does not preserve it.
    """
    out = []
    for a in args:
        if torch.is_tensor(a):
            cl = (a.dim() == 4 and a.is_contiguous(memory_format=torch.channels_last)
                  and not a.is_contiguous())
            out.append(dict(_t=a.detach().to("cpu", copy=True), cl=cl,
                            shape=tuple(a.shape), dtype=str(a.dtype)))
        else:
            out.append(a)
    return out


def unpark(parked, dev="cuda"):
    out = []
    for a in parked:
        if isinstance(a, dict) and "_t" in a:
            t = a["_t"].to(dev)
            if a["cl"]:
                t = t.contiguous(memory_format=torch.channels_last)
            out.append(t)
        else:
            out.append(a)
    return out


def scalar_args(parked):
    return [None if (isinstance(a, dict) and "_t" in a) else
            (a if isinstance(a, (int, float, bool, str)) else None) for a in parked]


def arg_dtypes(parked):
    return [a["dtype"] if isinstance(a, dict) and "_t" in a else None for a in parked]


# ---------------------------------------------------------------- profiling
def kernel_profile(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows, total = [], 0.0
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA or evt.self_device_time_total <= 0:
            continue
        t = evt.self_device_time_total / iters
        layer_type, role = classify_kernel(evt.key)
        rows.append(dict(kernel=short_kernel_name(evt.key), layer_type=layer_type, role=role,
                         us_per_call=round(t, 3), launches=round(evt.count / iters, 2)))
        total += t
    for r in rows:
        r["pct"] = round(r["us_per_call"] / total * 100, 2) if total else 0.0
    rows.sort(key=lambda r: -r["us_per_call"])
    return rows, round(total, 3)


# ---------------------------------------------------------------- capture
def capture(mode, batch, steps, max_sig_per_entry):
    """Run one real sample in `mode`, recording every kernel entry call and its arguments."""
    set_env(mode)
    calls = collections.OrderedDict()
    restore = []

    def wrap(owner, name, orig, label):
        def w(*a, **kw):
            sig = (label, tuple(tuple(x.shape) for x in a if torch.is_tensor(x)))
            rec = calls.get(sig)
            if rec is None:
                if sum(1 for k in calls if k[0] == label) < max_sig_per_entry:
                    calls[sig] = dict(entry=label, fn=orig, args=park(a),
                                      kwargs=dict(kw), count=1)
            else:
                rec["count"] += 1
            return orig(*a, **kw)
        setattr(owner, name, w)
        restore.append((owner, name, orig))

    for n in dir(mc):
        if n.startswith("_"):
            continue
        o = getattr(mc, n)
        if callable(o):
            wrap(mc, n, o, n)
    for name, label in (("conv2d", "torch_conv2d_fp16"), ("linear", "torch_linear_fp16"),
                        ("scaled_dot_product_attention", "torch_sdpa_fp16")):
        wrap(F, name, getattr(F, name), label)

    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir="docs/final_report_2026-07-28/tmp_out",
                          batch_size=batch, steps=steps, shape=(4, 32, 32),
                          calibration_path=CALIB.get(mode), linear_backend="fp16")
    model, sampler = r._setup_model(mode)
    cond = r._cond_kwargs(model, batch)
    # Everything the calibration/observer pass fired is discarded: those entries (plain
    # flash_attn_int8_vt, the dynamic-scale probes) are not what a production step executes.
    calls.clear()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                    dtype=torch.float16):
        sampler.sample(S=steps, batch_size=batch, shape=r.shape, eta=0.0, verbose=False,
                       **cond)
    torch.cuda.synchronize()
    for owner, name, orig in restore:
        setattr(owner, name, orig)
    del model, sampler
    torch.cuda.empty_cache()
    return calls


def bench_calls(calls, warm, iters, rounds, profile_it):
    per_suite = collections.defaultdict(list)
    for (label, shapes), rec in calls.items():
        args = unpark(rec["args"])
        torch.cuda.synchronize()
        fn = lambda f=rec["fn"], a=args, k=rec["kwargs"]: f(*a, **k)
        # Replay under the SAME context production runs in. The captured args are what the
        # caller passed *before* autocast applied its own casts, so torch's own ops
        # (F.conv2d/F.linear) get an fp16 input against an fp32 weight and raise a dtype
        # mismatch if replayed bare. The context is entered once around the whole timing
        # loop, not per call, so its overhead does not land in the measurement.
        with torch.inference_mode(), torch.autocast("cuda", enabled=True,
                                                    dtype=torch.float16):
            st, err = cuda_bench_stats(fn, warm=warm, iters=iters, rounds=rounds)
        row = dict(entry=label, arg_shapes=[list(s) for s in shapes],
                   arg_dtypes=arg_dtypes(rec["args"]), scalar_args=scalar_args(rec["args"]),
                   calls_per_sample=rec["count"], stats=st, error=err,
                   stability=stability_verdict(st))
        if st and profile_it:
            with torch.inference_mode(), torch.autocast("cuda", enabled=True,
                                                        dtype=torch.float16):
                row["kernels"], row["gpu_us_sum"] = kernel_profile(fn)
        per_suite[suite_of(label)].append(row)
        print("  %-14s %-44s %-22s %s" %
              (suite_of(label), label[:44], str(shapes[0]) if shapes else "-",
               ("%8.1f us  CV %5.2f%%  x%d" % (st["mean"], st["cv_pct"], rec["count"]))
               if st else ("ERR " + str(err)[:40])))
        del args
        torch.cuda.empty_cache()
    return per_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--capture-steps", type=int, default=5,
                    help="steps used only to capture calls; must divide 1000")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-sig-per-entry", type=int, default=64)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--output", default="docs/final_report_2026-07-28/data/kernel_suites.json")
    a = ap.parse_args()

    bn = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(8):
        bn = bn @ bn
    torch.cuda.synchronize()
    del bn
    torch.cuda.empty_cache()

    out = {"gpu": torch.cuda.get_device_name(0), "batch": a.batch, "warmup": a.warmup,
           "iters_per_round": a.iters, "rounds": a.rounds,
           "capture_steps": a.capture_steps, "modes": {}}
    for mode in MODES:
        print("\n" + "=" * 96 + "\n%s\n" % mode + "=" * 96)
        calls = capture(mode, a.batch, a.capture_steps, a.max_sig_per_entry)
        entries = collections.Counter(k[0] for k in calls)
        print("captured %d call signatures across %d entry points" % (len(calls), len(entries)))
        per_suite = bench_calls(calls, a.warmup, a.iters, a.rounds, not a.no_profile)
        out["modes"][mode] = dict(per_suite)
        print("  -> " + ", ".join("%s=%d" % (k, len(v)) for k, v in per_suite.items()))
        del calls, per_suite
        torch.cuda.empty_cache()

    path = a.output if os.path.isabs(a.output) else os.path.join(_ROOT, a.output)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("\nWROTE %s" % path)


if __name__ == "__main__":
    main()
