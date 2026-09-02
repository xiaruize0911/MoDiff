"""MODIFF_DELTA_MODE=static (ships) vs =dynamic, on the production W8A8 MoDiff path.

WHY. docs/act_blockwise_2026-09-01 priced the conv-input quantizer at every granularity and
found per-tensor DYNAMIC at 0.0451 vs the shipped per-tensor STATIC at 0.1838 -- a ~3.5x
accuracy gain that is epilogue-expressible, i.e. free, because the scale stays a scalar alpha.
That reproduces an older per-layer result (int8_optimized.py:166-180, 2026-08-04: static 0.1878
-> dynamic 0.0393 latent relL2). Neither measurement was a paired speed+quality run on today's
production path, and blockwise's marginal value should be priced against dynamic rather than
against static. This is that paired run.

Arms, each a fresh model build because `delta_dynamic` is read in OptimizedInt8Conv2d.__init__
(int8_optimized.py:185) -- flipping the env var on a live model is a no-op:
  fp16                          relL2 reference and the speed baseline
  int8 MoDiff DELTA_MODE=static ships today; loads int8_delta_calibration.pt
  int8 MoDiff DELTA_MODE=dynamic candidate; _load_delta_table returns 0 by design
  int8 MoDiff dynamic + AHAT_BLOCK=32  best config available today (a_hat blockwise is landed)

STEADY STATE IS LOAD-BEARING. Quantized attention self-calibrates over its first forwards, so a
first-run quality number is several x worse and reverses the static/dynamic ranking
(int8_optimized.py:179-184). Every arm therefore times first (which includes a warmup sample)
and only then generates the quality batch.

Timing: batch 128, 50 DDIM, CUDA events, median of 2 after 1 warmup.
Quality: n=6, seed 20260805, latent relL2 vs the fp16 arm.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/delta_dynamic_2026-09-02/scripts/delta_dynamic.py
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

# Production path, held fixed across arms. DELTA_MODE and AHAT_BLOCK are the only variables.
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_AHAT_BLOCK"] = "0"
os.environ["MODIFF_ACT_BLOCK"] = "0"          # sim harness OFF: this measures real kernels
os.environ["MODIFF_DELTA_MODE"] = "static"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="delta_dynamic.py")

import torch  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402

SHAPE = (4, 32, 32)
BATCH, STEPS, NQ, SEED = 128, 50, 6, 20260805
OUT_JSON = "docs/delta_dynamic_2026-09-02/data/delta_dynamic.json"

# (label, mode, DELTA_MODE, AHAT_BLOCK)
ARMS = [
    ("fp16",                        "fp16", "static",  0),
    ("W8A8 MoDiff  delta static",   "int8", "static",  0),
    ("W8A8 MoDiff  delta dynamic",  "int8", "dynamic", 0),
    ("W8A8 MoDiff  dynamic + aHat B=32", "int8", "dynamic", 32),
]


def reset(model, quantized):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def sample(model, sampler, n, quantized):
    reset(model, quantized)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        return sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)


def time_n(model, sampler, quantized, n=2, warm=1):
    for _ in range(warm):
        sample(model, sampler, BATCH, quantized)
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        sample(model, sampler, BATCH, quantized)
        e.record()
        torch.cuda.synchronize()
        xs.append(s.elapsed_time(e) / STEPS)
    return statistics.median(xs), xs


def gen_lat(model, sampler, quantized):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    out = sample(model, sampler, NQ, quantized)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={BATCH} steps={STEPS}", flush=True)
    recs, ref = [], None

    for label, mode, dmode, ablk in ARMS:
        os.environ["MODIFF_DELTA_MODE"] = dmode
        os.environ["MODIFF_AHAT_BLOCK"] = str(ablk)
        quantized = mode != "fp16"
        print(f"\n===== {label}  (mode={mode} DELTA_MODE={dmode} AHAT_BLOCK={ablk}) =====",
              flush=True)

        runner = B.BenchmarkRunner(
            config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
            ckpt_path="models/ldm/lsun_churches256/model.ckpt",
            output_dir="docs/delta_dynamic_2026-09-02/tmp",
            batch_size=BATCH, steps=STEPS, shape=SHAPE,
            calibration_path=B._default_calibration_path(mode),
            auto_delta_table=True)
        model, sampler = runner._setup_model(mode)

        ms, trials = time_n(model, sampler, quantized)      # warmup inside -> steady state
        lat = gen_lat(model, sampler, quantized)
        if ref is None:
            ref, rel = lat.clone(), 0.0
        else:
            rel = float((lat - ref).norm() / ref.norm())

        recs.append({"label": label, "mode": mode, "delta_mode": dmode, "ahat_block": ablk,
                     "ms_step": ms, "trials": trials, "relL2_vs_fp16": rel})
        print(f"  {ms:.3f} ms/step  trials={['%.3f' % t for t in trials]}  relL2 {rel:.4f}",
              flush=True)

        del model, sampler, runner
        gc.collect()
        torch.cuda.empty_cache()

    fp16_ms = recs[0]["ms_step"]
    stat_ms = next(r["ms_step"] for r in recs if r["delta_mode"] == "static" and r["mode"] == "int8")
    for r in recs:
        r["speedup_vs_fp16"] = fp16_ms / r["ms_step"]
        r["speedup_vs_shipped"] = stat_ms / r["ms_step"]

    print("\n===== summary =====", flush=True)
    print(f"  {'arm':36s} {'ms/step':>9s} {'vs fp16':>9s} {'vs shipped':>11s} {'relL2':>8s}",
          flush=True)
    for r in recs:
        print(f"  {r['label']:36s} {r['ms_step']:9.3f} {r['speedup_vs_fp16']:8.3f}x "
              f"{r['speedup_vs_shipped']:10.3f}x {r['relL2_vs_fp16']:8.4f}", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": BATCH, "steps": STEPS,
               "seed": SEED, "n_quality": NQ, "arms": recs}, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
