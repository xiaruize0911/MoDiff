"""End-to-end DDIM benchmark + kernel profile for FP16 / INT8 / INT4, all in ONE process.

The repo's benchmark_ldm CLI runs one mode per invocation, which makes cross-mode comparison
depend on the machine being in the same state each time. This drives BenchmarkRunner directly
so all three modes are measured back to back, and adds a torch-profiler pass per mode so the
e2e time can be attributed to kernels rather than only reported as a total.

Writes data/e2e_three_mode.json.

NOTE on the checkpoint: models/ldm/lsun_churches256/model.ckpt in this tree is an 856-byte
stub with an empty state_dict, loaded with strict=False, so all weights are random. Timing is
unaffected (these kernels have no data-dependent control flow) but nothing here is an
image-quality statement.
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/taming-transformers"))

import torch
from torch.profiler import profile, ProfilerActivity, DeviceType

import integration.benchmarks.benchmark_ldm as B

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ck_bench_stats import summarize, stability_verdict

MODES = ["fp16", "int8_baseline", "int4_baseline"]
CALIB = {"int8_baseline": "integration/calibration/int8_calibration.pt",
         "int4_baseline": "integration/calibration/int4_calibration.pt"}

# MODIFF_QUANT_LINEAR=1 is load-bearing and easy to miss: it is what turns the attention block's
# qkv/proj into _QuantLinearWxAx. Without it they stay plain nn.Linear, _qout_eligible() returns
# False, and EVERY fused-epilogue attention route -- the INT8 layout epilogue, the INT4 layout
# epilogue, and the older i4values short-circuit -- is silently skipped in favour of the generic
# score path. The run still completes and still reports a quantized speedup, so the omission does
# not look like an error; it just measures a different, slower configuration. The layer-level
# harness (layer_pipeline_bench.collect_layers) and the quality gates all set it.
QUANT_ENV = {
    "MODIFF_QUANT_LINEAR": "1",
    "MODIFF_QUANT_ATTN": "1",
    "MODIFF_QUANT_ATTN_STATIC": "1",
    "MODIFF_QATTN_FLASH": "1",
    "MODIFF_FLASH_GATE": "on",
    "MODIFF_QUANT_ATTN_ALLT": "0",
    "MODIFF_LINEAR_OUT_I8": "0",
}


def set_env(mode):
    quant = mode != "fp16"
    for k, v in QUANT_ENV.items():
        os.environ[k] = v if quant else ("0" if k in ("MODIFF_QUANT_LINEAR",
                                                      "MODIFF_QUANT_ATTN") else v)
    for k in ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND"):
        os.environ.pop(k, None)


def kernel_table(prof, wall_us):
    """Per-kernel self time, normalized so the parts sum to the unprofiled wall time.

    The profiler adds launch overhead, so its raw totals overstate. Scaling to the
    independently measured wall time keeps the attribution honest -- the same convention the
    layer-level report uses.
    """
    agg = defaultdict(lambda: {"us": 0.0, "calls": 0})
    for e in prof.key_averages():
        if e.device_type != DeviceType.CUDA:
            continue
        us = float(getattr(e, "self_device_time_total", 0) or 0)
        if us <= 0:
            continue
        agg[e.key]["us"] += us
        agg[e.key]["calls"] += int(e.count)
    tot = sum(v["us"] for v in agg.values())
    f = (wall_us / tot) if tot > 0 else 1.0
    rows = [{"kernel": k, "us": v["us"] * f, "calls": v["calls"],
             "pct": v["us"] / tot * 100 if tot else 0.0} for k, v in agg.items()]
    rows.sort(key=lambda r: -r["us"])
    return rows, tot


def main():
    ap = argparse.ArgumentParser()
    # Defaults chosen for robustness, not speed. A 50-step batch-32 sample is short enough that a
    # single scheduler hiccup moves the median: an early 3-repeat run put INT4 at 1064 ms with a
    # 9.77% spread, 6.7% off the 997 ms that 9 repeats converged on. 200 steps at batch 128
    # averages ~16x more work into each individual measurement, which suppresses that noise
    # inside the sample rather than relying on the median to reject it afterwards.
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--output", default="docs/final_report_2026-07-28/data/e2e_three_mode.json")
    a = ap.parse_args()

    out = {"gpu": torch.cuda.get_device_name(0), "batch": a.batch, "steps": a.steps,
           "repeats": a.repeats, "modes": {}}

    for mode in MODES:
        print(f"\n{'='*64}\n{mode}\n{'='*64}")
        set_env(mode)
        runner = B.BenchmarkRunner(
            "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
            "models/ldm/lsun_churches256/model.ckpt",
            output_dir="integration/results/e2e_three_mode",
            batch_size=a.batch, steps=a.steps, shape=(4, 32, 32),
            calibration_path=CALIB.get(mode),
            linear_backend="int_gemm" if mode != "fp16" else "fp16",
        )
        model, sampler = runner._setup_model(mode)
        cond = runner._cond_kwargs(model, a.batch)

        def sample():
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                            dtype=torch.float16):
                sampler.sample(S=a.steps, batch_size=a.batch, shape=runner.shape,
                               eta=0.0, verbose=False, **cond)

        # warm up: settles clocks/caches and, for the quantized modes, lets the static attention
        # scales freeze (that needs MODIFF_ATTN_CALIB_STEPS=8 forwards; one 200-step sample is
        # already far past it, but extra warmups also let the GPU reach a steady thermal/clock
        # state, which is what the first-sample outliers were mostly about).
        for _ in range(a.warmups):
            sample()
        torch.cuda.synchronize()

        # Route check AFTER warmup: _qout_eligible() also requires the static Q/K/V scales to be
        # frozen, which only happens after MODIFF_ATTN_CALIB_STEPS (8) forwards, so checking it
        # straight after _setup_model would always report 0 eligible.
        #
        # This guards a failure mode that is invisible in the summary table: without
        # MODIFF_QUANT_LINEAR=1 the attention qkv/proj stay plain nn.Linear, every fused-epilogue
        # route is skipped, and the run still prints a believable quantized speedup.
        route = {}
        if mode != "fp16":
            blks = [m for m in model.modules()
                    if type(m).__name__ == "QuantizedStandardAttentionBlock"]
            assert blks, f"{mode}: no QuantizedStandardAttentionBlock was installed"
            elig = sum(b._qout_eligible() for b in blks)
            # Both bit widths now reach all 21: INT4 gained an hd=96 route (the dp4a small
            # kernel) and _observe_small_int8_scales was un-gated so those blocks can freeze
            # their scales. This used to be 15 for INT4.
            expected = len(blks)
            qkv_t, proj_t = type(blks[0].qkv).__name__, type(blks[0].proj).__name__
            route = {"attn_blocks": len(blks), "qout_eligible": elig, "expected_eligible": expected,
                     "qkv_type": qkv_t, "proj_type": proj_t,
                     "int4_layout_epilogue": bool(getattr(blks[0], "_int4_layout_epilogue", False))}
            print(f"  route check: {elig}/{len(blks)} qout-eligible (expected {expected}); {route}")
            # The failure this guards against is qkv/proj staying plain nn.Linear, which silently
            # disables every fused-epilogue route while still reporting a plausible speedup.
            assert "QuantLinear" in qkv_t and "QuantLinear" in proj_t, (
                f"{mode}: attention qkv/proj are {qkv_t}/{proj_t}, not quantized linears -- the "
                f"fused epilogue route cannot engage (is MODIFF_QUANT_LINEAR=1 set?)")
            assert elig == expected, (
                f"{mode}: {elig}/{len(blks)} qout-eligible but expected {expected}")

        times = []
        for _ in range(a.repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            sample()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e3)     # us for the whole batch
        wall_us = statistics.median(times)
        mean_us = statistics.mean(times)
        sd_us = statistics.stdev(times) if len(times) > 1 else 0.0

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            sample()
        rows, raw_us = kernel_table(prof, wall_us)

        # Same summary the kernel and layer suites report, so all five suites in the final
        # report quote stability the same way. The hand-rolled fields below are kept because
        # the published data already uses them; `stats` adds the t-based 95% CI on the mean,
        # which the +-sigma fields do not give at these repeat counts.
        st = summarize(times)

        out["modes"][mode] = {
            "stats": st,
            "stability": stability_verdict(st),
            "wall_us_per_batch": wall_us,
            "wall_mean_us": mean_us,
            "wall_stdev_us": sd_us,
            "wall_cv_pct": sd_us / mean_us * 100 if mean_us else 0.0,
            "wall_min_us": min(times),
            "wall_max_us": max(times),
            "wall_all_us": [round(t, 1) for t in times],
            "wall_spread_pct": (max(times) - min(times)) / min(times) * 100,
            "per_sample_ms": wall_us / 1e3 / a.batch,
            "per_step_ms": wall_us / 1e3 / a.steps,
            "profiler_raw_gpu_us": raw_us,
            "profiler_scale": wall_us / raw_us if raw_us else 1.0,
            "route_check": route,
            "kernels": rows,
        }
        print(f"  wall  median {wall_us/1e3:9.1f} ms/batch   "
              f"{wall_us/1e3/a.batch:6.3f} ms/sample   {wall_us/1e3/a.steps:6.2f} ms/step")
        print(f"        mean {mean_us/1e3:.1f}  sd {sd_us/1e3:.1f}  CV {sd_us/mean_us*100:.2f}%  "
              f"min {min(times)/1e3:.1f}  max {max(times)/1e3:.1f}  "
              f"spread {out['modes'][mode]['wall_spread_pct']:.2f}%")
        print(f"  top kernels:")
        for r in rows[:8]:
            print(f"    {r['kernel'][:58]:<60}{r['us']/1e3:>9.2f} ms  {r['pct']:>5.1f}%")

        del model, sampler, runner
        torch.cuda.empty_cache()

    fp = out["modes"]["fp16"]["wall_us_per_batch"]
    for m in MODES:
        out["modes"][m]["speedup_vs_fp16"] = fp / out["modes"][m]["wall_us_per_batch"]

    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWROTE {a.output}")
    print(f"\n{'mode':<16}{'ms/batch':>11}{'ms/sample':>11}{'ms/step':>9}"
          f"{'vs fp16':>9}{'CV':>8}{'spread':>8}")
    for m in MODES:
        d = out["modes"][m]
        print(f"{m:<16}{d['wall_us_per_batch']/1e3:>11.1f}{d['per_sample_ms']:>11.3f}"
              f"{d['per_step_ms']:>9.2f}{d['speedup_vs_fp16']:>8.3f}x"
              f"{d['wall_cv_pct']:>7.2f}%{d['wall_spread_pct']:>7.2f}%")


if __name__ == "__main__":
    main()
