"""Full profile, at MEASUREMENT_REPORT_2026-08-01 depth. Four things the role table alone cannot say.

The role/bucket table already in the report answers "which kernels cost what". It does not answer:

  1. does the kernel time ADD UP to the measured wall clock?  If it does not, the difference is
     launch gap / CPU-bound dispatch, and that difference is a number this report has been quoting
     (~1.5 ms/step for MoDiff) without ever measuring it directly.
  2. where does the time go by LAYER TYPE -- ResBlock vs attention vs updown vs outside any layer?
  3. how much of the timeline is the GPU actually busy, and what does the gap distribution look like?
  4. per attention shape in a REAL run, GPU us/call and the CPU issue cost behind it.

All four come out of one torch.profiler trace per mode, so nsys is not needed:
  - kernel events give self-time totals (role table), the busy span, and the gap distribution
  - record_function ranges wrapped around every ResBlock / AttentionBlock give layer attribution
  - the same ranges give per-shape attention cost and its CPU-side issue time

Kernel self-time is then SCALED to the profiler-free measured wall clock, exactly as the 08-01
report does, so the stage sum is comparable to a real run rather than to a profiled one (the
profiler itself adds overhead).
"""

import json
import os
import statistics as st
import sys
import time
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks
from profile_tree import classify

D = "docs/report_2026-08-04"
BATCH = int(os.environ.get("FP_BATCH", "128"))
STEPS = int(os.environ.get("FP_STEPS", "20"))
WALL_REPS = int(os.environ.get("FP_WALL_REPS", "3"))
MODES = [("int8_baseline", "int8_baseline", "static"), ("int8 dynamic", "int8", "dynamic")]
CALIB = {"int8": "integration/calibration/int8_calibration_realckpt.pt",
         "int4": "integration/calibration/int4_calibration_realckpt.pt"}
GAP_US = 50.0


def build(mode, dm):
    ks.set_env(mode)
    os.environ["MODIFF_DELTA_MODE"] = dm
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{D}/tmp_out", batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
        calibration_path=CALIB["int4" if "int4" in mode else "int8"])
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def layer_kind(mod, name):
    """resblock_plain / resblock_updown / attention, matching the 08-01 report's taxonomy."""
    t = type(mod).__name__
    mro = [c.__name__.lower() for c in type(mod).__mro__]
    if any("attention" in m for m in mro) and hasattr(mod, "norm"):
        return "attention"
    if "resblock" in t.lower() or any("resblock" in m for m in mro):
        return "resblock_updown" if getattr(mod, "updown", False) else "resblock_plain"
    return None


def wrap_layers(model):
    """Wrap each ResBlock / AttentionBlock forward in a record_function range named
    '<kind>|<C>x<H>'. Returns the list of (module, original_forward) so it can be undone."""
    undo = []
    for name, m in model.model.diffusion_model.named_modules():
        kind = layer_kind(m, name)
        if kind is None:
            continue
        orig = m.forward

        def mk(mod=m, f=orig, k=kind):
            def fwd(x, *a, **kw):
                try:
                    tag = f"{k}|C{x.shape[1]}/H{x.shape[2]}"
                except Exception:
                    tag = k
                with record_function("LAYER::" + tag):
                    return f(x, *a, **kw)
            return fwd
        m.forward = mk()
        undo.append((m, orig))
    return undo


def sample(runner, model, sampler):
    from integration.kernels.int4_optimized import reset_modiff_state as r4
    from integration.kernels.int8_optimized import reset_modiff_state as r8
    for r in (r8, r4):
        try:
            r(model.model.diffusion_model)
        except Exception:
            pass
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    cond = runner._cond_kwargs(model, BATCH)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                       verbose=False, **cond)


def analyse(prof, wall_ms_per_step):
    """Kernel roles, busy/idle, gap distribution, layer attribution -- all from one trace."""
    evs = prof.events()
    kern = [e for e in evs if str(getattr(e, "device_type", "")).endswith("CUDA")
            and getattr(e, "device_time", 0) > 0 and getattr(e, "name", None)]
    # role self-time
    roles = defaultdict(float)
    for e in kern:
        roles[classify(e.name)] += e.device_time / 1e3        # us -> ms
    ktot = sum(roles.values())

    # busy / idle from the kernel timeline: merge overlapping [start, start+dur) intervals
    iv = sorted((e.time_range.start, e.time_range.end) for e in kern
                if getattr(e, "time_range", None))
    merged, gaps = [], []
    for s, e_ in iv:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e_))
        else:
            if merged:
                gaps.append(s - merged[-1][1])
            merged.append((s, e_))
    busy = sum(b - a for a, b in merged) / 1e3
    span = (merged[-1][1] - merged[0][0]) / 1e3 if merged else 0.0
    idle = max(0.0, span - busy)

    # layer attribution from the record_function ranges (CPU-side ranges with CUDA children)
    lay = defaultdict(float)
    shape_gpu = defaultdict(float)
    shape_cpu = defaultdict(float)
    shape_n = defaultdict(int)
    for e in evs:
        n = getattr(e, "name", "") or ""
        if not n.startswith("LAYER::"):
            continue
        tag = n[len("LAYER::"):]
        kind = tag.split("|")[0]
        g = getattr(e, "cuda_time_total", 0) or 0
        c = getattr(e, "cpu_time_total", 0) or 0
        lay[kind] += g / 1e3
        shape_gpu[tag] += g / 1e3
        shape_cpu[tag] += c / 1e3
        shape_n[tag] += 1

    scale = (wall_ms_per_step / (ktot / STEPS)) if ktot else 1.0
    return {
        "roles_ms_per_step": {k: v / STEPS for k, v in sorted(roles.items(), key=lambda kv: -kv[1])},
        "roles_scaled_ms_per_step": {k: v / STEPS * scale
                                     for k, v in sorted(roles.items(), key=lambda kv: -kv[1])},
        "kernel_ms_per_step": ktot / STEPS,
        "wall_ms_per_step": wall_ms_per_step,
        "unattributed_ms_per_step": wall_ms_per_step - ktot / STEPS,
        "scale_factor": scale,
        "gpu_busy_pct": busy / span * 100 if span else 0.0,
        "gpu_idle_pct": idle / span * 100 if span else 0.0,
        "median_gap_us": st.median(gaps) if gaps else 0.0,
        "gaps_over_50us": sum(1 for g in gaps if g > GAP_US),
        "gaps_over_50us_span_pct": sum(g for g in gaps if g > GAP_US) / (span * 1e3) * 100
                                   if span else 0.0,
        "n_kernels_per_step": len(kern) / STEPS,
        "layer_ms_per_step": {k: v / STEPS for k, v in sorted(lay.items(), key=lambda kv: -kv[1])},
        "attn_shapes": {t: {"gpu_us_per_call": shape_gpu[t] * 1e3 / max(1, shape_n[t]),
                            "cpu_issue_us_per_call": shape_cpu[t] * 1e3 / max(1, shape_n[t]),
                            "gpu_per_issue": (shape_gpu[t] / shape_cpu[t]) if shape_cpu[t] else 0.0,
                            "calls": shape_n[t]}
                        for t in sorted(shape_gpu) if t.startswith("attention")},
    }


def main():
    out = {}
    for label, mode, dm in MODES:
        runner, model, sampler = build(mode, dm)
        sample(runner, model, sampler)                       # warm-up / steady state

        # profiler-free wall clock first: the profiler itself adds overhead, and the 08-01 report
        # scales kernel self-time to the UNPROFILED wall time.
        ts = []
        for _ in range(WALL_REPS):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            sample(runner, model, sampler)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0 / STEPS)
        wall = st.median(ts)
        print(f"\n{label}: wall {wall:.3f} ms/step (median of {WALL_REPS}, "
              f"spread {(max(ts)-min(ts))/wall*100:.2f}%)", flush=True)

        undo = wrap_layers(model)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, with_stack=False) as prof:
            sample(runner, model, sampler)
        for m, f in undo:
            m.forward = f

        r = analyse(prof, wall)
        out[label] = r
        print(f"  kernel {r['kernel_ms_per_step']:.2f} ms/step vs wall {wall:.2f} -> "
              f"unattributed {r['unattributed_ms_per_step']:+.2f} "
              f"({r['unattributed_ms_per_step']/wall*100:+.1f}%)", flush=True)
        print(f"  GPU busy {r['gpu_busy_pct']:.1f}%  idle {r['gpu_idle_pct']:.1f}%  "
              f"median gap {r['median_gap_us']:.2f} us  gaps>50us {r['gaps_over_50us']} "
              f"({r['gaps_over_50us_span_pct']:.1f}% of span)  "
              f"{r['n_kernels_per_step']:.0f} kernels/step", flush=True)
        print("  by layer type: " + "  ".join(f"{k} {v:.2f}" for k, v in r["layer_ms_per_step"].items()),
              flush=True)
        del model, sampler, runner
        torch.cuda.empty_cache()

    with open(f"{D}/data/full_profile.json", "w") as f:
        json.dump({"batch": BATCH, "steps": STEPS, "results": out}, f, indent=2)
    print(f"\nWROTE {D}/data/full_profile.json")


if __name__ == "__main__":
    main()
