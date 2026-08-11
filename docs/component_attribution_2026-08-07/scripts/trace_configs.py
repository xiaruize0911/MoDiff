"""Method C, part 1: one Perfetto/Chrome trace per configuration, traced once, bucketed offline.

Same idea as `docs/perfetto_traces_2026-08-03/scripts/export_perfetto_traces.py`, pointed at the
MoDiff configurations instead of the three baseline modes. The point of tracing rather than putting
`record_function` around each component is that a trace carries no in-Python instrumentation into
the measured region -- the bias that invalidated `component_profile.py` -- and the bucket rules
live in `bucket_traces.py`, so they can be revised without spending another GPU run.

The configurations are imported from `differential_timing.py` rather than restated, so method A and
method C are provably measuring the same thing. Only the arms whose kernel composition is a
question are traced; the rest are timing-only.

Per config: a full DDIM warm-up sample (past the attention blocks' 8-forward static-scale
calibration window and the conv MoDiff warm-up rounds), then `--steps` UNet forwards driven
directly, one `step/NNN` slice each. Driving the UNet rather than sampler.sample() keeps the DDIM
scheduler's own tensor math out of the slices.

Writes traces/<config>_b128_sN.json.gz plus traces/manifest.json.

A profiler IS attached here, so these numbers carry CUPTI overhead -- read them as composition, and
take the totals from differential_timing.json. `bucket_traces.py` reports the ratio between the two
per config, which is the alignment check this method owes.
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.dirname(os.path.abspath(__file__))]

import torch                                                                    # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function           # noqa: E402

import differential_timing as dt                                                # noqa: E402

OUT = "docs/component_attribution_2026-08-07/traces"
#: The subset of dt.ARMS worth a trace: the ladder plus the two epilogue arms. Timing-only arms
#: (base_no_qattn, base_no_qlinear, ...) answer their question from the wall clock alone.
TRACED = ["fp16", "int8_ptq", "modiff_conv_k4", "modiff_conv_k1", "modiff_full_k1",
          "ptq_no_projquant"]
#: Must divide 1000: make_ddim_timesteps indexes alphas_cumprod at t+1.
WARM_STEPS = 5


def trace_one(arm, a, manifest):
    label, base, mode, over, hook, why = arm
    print(f"\n{'='*72}\n{label}  ({mode})\n{'='*72}", flush=True)
    dt.apply_env(over)
    dt.HOOK_FACTS.clear()

    r = dt.B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{OUT}/tmp", batch_size=a.batch, steps=a.steps, shape=(4, 32, 32),
        calibration_path=None if mode == "fp16" else dt.CALIB8,
        linear_backend="fp16" if mode == "fp16" else "int_gemm")
    model, sampler = r._setup_model(mode)
    cond = r._cond_kwargs(model, a.batch)
    undo = hook(model) if hook is not None else (lambda: None)
    unet = model.model.diffusion_model

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=WARM_STEPS, batch_size=a.batch, shape=r.shape, eta=0.0,
                       verbose=False, **cond)
    torch.cuda.synchronize()
    rc = dt.route_check(model, mode)
    print(f"  route: {rc}", flush=True)

    x = torch.randn(a.batch, 4, 32, 32, device="cuda", dtype=torch.float16)
    timesteps = torch.linspace(999, 0, a.steps).round().long().to("cuda")
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(a.steps):
                t = timesteps[i].expand(a.batch)
                with record_function(f"step/{i:03d}"):
                    x = x + 0.0 * unet(x, t)[:, :4]
            torch.cuda.synchronize()

    name = f"{label}_b{a.batch}_s{a.steps}"
    raw = os.path.join(OUT, name + ".json")
    prof.export_chrome_trace(raw)
    with open(raw, "rb") as fi, gzip.open(raw + ".gz", "wb", compresslevel=9) as fo:
        fo.writelines(fi)
    n_ev = len(json.load(open(raw)).get("traceEvents", []))
    os.remove(raw)
    manifest.append(dict(config=label, mode=mode, delta_from=base, why=why,
                         file=name + ".json.gz", events=n_ev,
                         gz_mb=round(os.path.getsize(raw + ".gz") / 2 ** 20, 2),
                         steps=a.steps, batch=a.batch, route_check=rc,
                         env={k: os.environ.get(k) for k in sorted(dt.BASE_ENV)},
                         hook=getattr(hook, "__name__", None)))
    print(f"  -> {name}.json.gz  {manifest[-1]['gz_mb']:.2f} MiB, {n_ev} events", flush=True)

    undo()
    del model, sampler, r, unet, x
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    #: 8 covers a full MODIFF_DELTA_REFRESH=4 period twice, so a K=4 trace contains both the
    #: refresh step and the three reuse steps in their production proportion.
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--configs", default=",".join(TRACED))
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    want = [x.strip() for x in a.configs.split(",") if x.strip()]
    arms = [x for x in dt.ARMS if x[0] in want]
    missing = set(want) - {x[0] for x in arms}
    assert not missing, f"unknown configs: {missing}"

    # Merge into any existing manifest rather than replacing it, so a single config can be added or
    # re-traced without re-running the five that did not change.
    mpath = os.path.join(OUT, "manifest.json")
    manifest = []
    if os.path.exists(mpath):
        manifest = [t for t in json.load(open(mpath))["traces"] if t["config"] not in want]
    for arm in arms:
        trace_one(arm, a, manifest)
        with open(mpath, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                       "traces": manifest}, f, indent=1)
    print(f"\nWROTE {OUT}/manifest.json ({len(manifest)} traces)")


if __name__ == "__main__":
    main()
