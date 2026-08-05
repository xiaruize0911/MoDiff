"""Export every profiling level as a Perfetto-viewable Chrome trace, for all three modes.

torch.profiler's `export_chrome_trace` writes the Chrome Trace Event JSON that ui.perfetto.dev
opens directly. Each measured unit is wrapped in `record_function`, so a single trace per (mode,
level) shows up in Perfetto as one timeline of named slices rather than an undifferentiated wall of
kernels -- select a slice and the CUDA kernels underneath it are the ones that unit launched.

Levels, mirroring the three suites the measurement report is built from:

  e2e      DDIM steps of the whole UNet, one `step/NNN` slice each
  layers   one forward per (layer kind, input shape), from layer_pipeline_bench.collect_layers
  kernels  one call per captured kernel entry signature, from kernel_suites_bench.capture

3 modes x 3 levels = 9 traces. Iteration counts are deliberately small: these are timelines to
read, not measurements -- the numbers to quote are in docs/MEASUREMENT_REPORT_2026-08-01.md, which
is measured without a profiler attached. Every level warms up past the quantized attention blocks'
static-scale calibration window first, so what the trace shows is the production route.
"""

import argparse
import collections
import gzip
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import kernel_suites_bench as ks
import layer_pipeline_bench as lb

OUT = "docs/perfetto_traces_2026-08-03/traces"
MODES = ["fp16", "int8_baseline", "int4_baseline"]
CALIB = {"fp16": "integration/calibration/int8_calibration.pt",
         "int8_baseline": "integration/calibration/int8_calibration.pt",
         "int4_baseline": "integration/calibration/int4_calibration.pt"}
ACTIVITIES = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
# The quantized attention blocks self-calibrate over their first 8 forwards and then freeze; a
# trace taken before that shows calibration kernels that no production step executes.
WARM_FORWARDS = 12
#: DDIM step count for the warmup sample only. Must divide 1000 (see trace_e2e).
WARM_STEPS = 5


def _write(prof, name, manifest, note):
    """Export one trace, gzip it (Perfetto reads .json.gz directly), and record what it is."""
    raw = os.path.join(OUT, name + ".json")
    prof.export_chrome_trace(raw)
    with open(raw, "rb") as f_in, gzip.open(raw + ".gz", "wb", compresslevel=9) as f_out:
        f_out.writelines(f_in)
    n_events = None
    try:
        with open(raw) as f:
            n_events = len(json.load(f).get("traceEvents", []))
    except Exception:
        pass
    raw_mb = os.path.getsize(raw) / 2 ** 20
    gz_mb = os.path.getsize(raw + ".gz") / 2 ** 20
    os.remove(raw)
    manifest.append(dict(file=name + ".json.gz", note=note, events=n_events,
                         raw_mb=round(raw_mb, 2), gz_mb=round(gz_mb, 2)))
    print(f"  -> {name}.json.gz  {gz_mb:.2f} MiB gz ({raw_mb:.1f} MiB raw)"
          f"{f', {n_events} events' if n_events else ''}")


def trace_e2e(mode, batch, steps, manifest):
    """Whole-model timeline: `steps` DDIM steps, one named slice per step."""
    ks.set_env(mode)
    r = ks.B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                             "models/ldm/lsun_churches256/model.ckpt",
                             output_dir=f"{OUT}/tmp", batch_size=batch, steps=steps,
                             shape=(4, 32, 32), calibration_path=CALIB[mode],
                             linear_backend="fp16")
    model, sampler = r._setup_model(mode)
    cond = r._cond_kwargs(model, batch)
    unet = model.model.diffusion_model

    # One full sample first: warms the allocator and clocks, and freezes the attention scales.
    # WARM_STEPS is fixed at a value that divides 1000 -- make_ddim_timesteps indexes
    # alphas_cumprod at t+1, so an S that does not divide 1000 raises IndexError. The traced step
    # count below is independent of it because the UNet is driven directly.
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=WARM_STEPS, batch_size=batch, shape=r.shape, eta=0.0,
                       verbose=False, **cond)
    torch.cuda.synchronize()

    # Drive the UNet directly rather than sampler.sample() so each step is one labelled slice and
    # the DDIM scheduler's own tensor math does not straddle slice boundaries.
    x = torch.randn(batch, 4, 32, 32, device="cuda", dtype=torch.float16)
    timesteps = torch.linspace(999, 0, steps).round().long().to("cuda")
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        with profile(activities=ACTIVITIES, record_shapes=True) as prof:
            for i in range(steps):
                t = timesteps[i].expand(batch)
                with record_function(f"step/{i:03d}"):
                    x = x + 0.0 * unet(x, t)[:, :4]
            torch.cuda.synchronize()
    _write(prof, f"e2e_{mode}_b{batch}_s{steps}", manifest,
           f"{steps} UNet steps, batch {batch}, {mode}; one 'step/NNN' slice per step")
    del model, sampler, r, unet, x
    torch.cuda.empty_cache()


def trace_layers(mode, iters, manifest):
    """One slice per (layer kind, input shape) -- the layer_pipeline_bench population."""
    model, sampler, layers = lb.collect_layers(mode)
    del sampler
    groups = collections.OrderedDict()
    for L in layers:
        groups.setdefault((L["kind"], L["x_shape"], L["emb_shape"]), []).append(L)

    prepared = []
    for (kind, xs, es), insts in groups.items():
        m = insts[0]["module"]
        x = torch.randn(*xs, device="cuda", dtype=torch.float16)
        x = x.contiguous(memory_format=torch.channels_last) if x.dim() == 4 else x
        emb = torch.randn(*es, device="cuda", dtype=torch.float16) if es else None
        label = f"{kind}/{'x'.join(str(d) for d in xs)}"
        prepared.append((label, m, x, emb, len(insts)))

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for _, m, x, emb, _ in prepared:                       # warm past calibration
            for _ in range(WARM_FORWARDS):
                m(x, emb) if emb is not None else m(x)
        torch.cuda.synchronize()
        with profile(activities=ACTIVITIES, record_shapes=True) as prof:
            for label, m, x, emb, n in prepared:
                with record_function(f"{label} x{n}"):
                    for _ in range(iters):
                        m(x, emb) if emb is not None else m(x)
            torch.cuda.synchronize()
    _write(prof, f"layers_{mode}", manifest,
           f"{len(prepared)} (kind, shape) groups x {iters} forwards, {mode}; "
           f"slice label is 'kind/shape xN', N = instances in the UNet")
    del model, layers, groups, prepared
    torch.cuda.empty_cache()


def trace_kernels(mode, batch, capture_steps, max_sig, iters, manifest):
    """One slice per captured kernel entry signature -- the kernel_suites_bench population."""
    calls = ks.capture(mode, batch, capture_steps, max_sig)
    print(f"  captured {len(calls)} call signatures")
    with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.float16):
        with profile(activities=ACTIVITIES, record_shapes=True) as prof:
            for (label, shapes), rec in calls.items():
                args = ks.unpark(rec["args"])
                shape = "x".join(str(d) for d in shapes[0]) if shapes else "-"
                torch.cuda.synchronize()
                with record_function(f"{ks.suite_of(label)}/{label}/{shape}"):
                    for _ in range(iters):
                        rec["fn"](*args, **rec["kwargs"])
                torch.cuda.synchronize()
                del args
            torch.cuda.synchronize()
    _write(prof, f"kernels_{mode}", manifest,
           f"{len(calls)} kernel entry signatures x {iters} calls, {mode}; "
           f"slice label is 'suite/entry/first-arg-shape'")
    del calls
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--e2e-steps", type=int, default=5)
    ap.add_argument("--layer-iters", type=int, default=3)
    ap.add_argument("--kernel-iters", type=int, default=3)
    ap.add_argument("--capture-steps", type=int, default=5)
    ap.add_argument("--max-sig-per-entry", type=int, default=64)
    ap.add_argument("--levels", default="e2e,layers,kernels")
    ap.add_argument("--modes", default=",".join(MODES))
    a = ap.parse_args()
    levels = [s.strip() for s in a.levels.split(",") if s.strip()]
    modes = [s.strip() for s in a.modes.split(",") if s.strip()]
    os.makedirs(OUT, exist_ok=True)

    # Wake the device without driving it into its power cap (same preamble as the suites).
    bn = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(8):
        bn = bn @ bn
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    manifest = []
    for mode in modes:
        for level in levels:
            print(f"\n=== {mode} / {level} ===", flush=True)
            t0 = time.time()
            try:
                if level == "e2e":
                    trace_e2e(mode, a.batch, a.e2e_steps, manifest)
                elif level == "layers":
                    trace_layers(mode, a.layer_iters, manifest)
                elif level == "kernels":
                    trace_kernels(mode, a.batch, a.capture_steps,
                                  a.max_sig_per_entry, a.kernel_iters, manifest)
                else:
                    raise SystemExit(f"unknown level {level}")
            except Exception as exc:
                # Never silently drop a level: a missing trace must be visible in the manifest.
                print(f"  FAILED {mode}/{level}: {type(exc).__name__}: {exc}")
                manifest.append(dict(file=None, note=f"{mode}/{level} FAILED",
                                     error=f"{type(exc).__name__}: {exc}"))
            print(f"  ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                       batch=a.batch, e2e_steps=a.e2e_steps, layer_iters=a.layer_iters,
                       kernel_iters=a.kernel_iters, traces=manifest), f, indent=2)
    ok = [m for m in manifest if m.get("file")]
    print(f"\n{len(ok)} traces written to {OUT}")
    for m in manifest:
        print(f"  {m.get('file') or 'MISSING'}  {m['note']}")


if __name__ == "__main__":
    main()
