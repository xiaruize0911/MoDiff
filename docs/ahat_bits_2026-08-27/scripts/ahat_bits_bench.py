"""Batch-128 UNet bench: scheme class x MODIFF_AHAT_BITS (+ optional scale refresh).

ms/step = 1000 * wall / 50 DDIM steps. Same process, one model load per mode.

bits=16: fp16 a_hat.
bits=8/4: real int8-code a_hat storage (qmax 127 / 7), in-kernel dequant/quant.
MODIFF_AHAT_REFRESH=1: unpack to fp16 on commit, pack with a fresh absmax (matches
the old Python fake-quant grid). Default 0 holds the t=T scale.
"""
import json
import os
import sys
import time

ROOT = "/workspace/MoDiff"
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"

import torch
import integration.benchmarks.benchmark_ldm as B

STEPS, BATCH, SHAPE = 50, 128, (4, 32, 32)
SEED = 20260827
OUT_JSON = "docs/ahat_bits_2026-08-27/data/ahat_bits_bench.json"

# (label, skip_k, replay_k, bits, refresh)
ARMS_W8A8 = [
    ("full a_hat fp16", 1, 1, 16, 0),
    ("full a_hat int8 held", 1, 1, 8, 0),
    ("full a_hat int8 refresh", 1, 1, 8, 1),
    ("full a_hat int4 held", 1, 1, 4, 0),
    ("full a_hat int4 refresh", 1, 1, 4, 1),
    ("skip-K=4 a_hat fp16", 4, 1, 16, 0),
    ("skip-K=4 a_hat int8 held", 4, 1, 8, 0),
    ("skip-K=4 a_hat int4 held", 4, 1, 4, 0),
    ("replay-K=4 a_hat fp16", 1, 4, 16, 0),
    ("replay-K=4 a_hat int8 held", 1, 4, 8, 0),
    ("replay-K=4 a_hat int4 held", 1, 4, 4, 0),
]
ARMS_W4A4 = [
    ("W4A4 full a_hat fp16", 1, 1, 16, 0),
    ("W4A4 full a_hat int4 held", 1, 1, 4, 0),
    ("W4A4 full a_hat int4 refresh", 1, 1, 4, 1),
    ("W4A4 skip-K=4 a_hat fp16", 4, 1, 16, 0),
    ("W4A4 skip-K=4 a_hat int4 held", 4, 1, 4, 0),
    ("W4A4 replay-K=4 a_hat fp16", 1, 4, 16, 0),
    ("W4A4 replay-K=4 a_hat int4 held", 1, 4, 4, 0),
]


def apply(skip_k, replay_k, bits, refresh=0):
    os.environ["MODIFF_CACHE_SKIP_K"] = str(skip_k)
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_AHAT_REFRESH"] = str(refresh)


def reset(model, mode):
    unet = model.model.diffusion_model
    if "int8" in mode:
        B.reset_modiff_state_int8(unet)
    elif "int4" in mode:
        B.reset_modiff_state_int4(unet)
    B._reset_wxax_modiff_safe(model)


def time_once(model, sampler, mode):
    reset(model, mode)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / STEPS


def run_mode(runner, mode, arms, recs):
    apply(1, 1, 16, 0)
    runner.calibration_path = B._default_calibration_path(mode)
    model, sampler = runner._setup_model(mode)
    for label, skip_k, replay_k, bits, refresh in arms:
        apply(skip_k, replay_k, bits, refresh)
        print(f"warmup {label}", flush=True)
        time_once(model, sampler, mode)
        ms = time_once(model, sampler, mode)
        recs.append({"label": label, "mode": mode, "skip_k": skip_k, "replay_k": replay_k,
                     "ahat_bits": bits, "refresh": refresh, "ms_step": ms})
        print(f"  {label:36s} {ms:.3f} ms/step", flush=True)
    apply(1, 1, 16, 0)
    del model, sampler
    torch.cuda.empty_cache()


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={BATCH} steps={STEPS}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_bits_2026-08-27/tmp_bench",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)
    recs = []
    print("===== W8A8 =====", flush=True)
    run_mode(runner, "int8", ARMS_W8A8, recs)
    print("===== W4A4 =====", flush=True)
    run_mode(runner, "int4", ARMS_W4A4, recs)

    ref = next(r["ms_step"] for r in recs if r["label"] == "full a_hat fp16")
    print("\n===== vs W8A8 full a_hat fp16 =====", flush=True)
    for r in recs:
        r["speedup_vs_fp16_full"] = ref / r["ms_step"]
        r["pct_vs_fp16_full"] = 100.0 * (ref - r["ms_step"]) / ref
        print(f"  {r['label']:36s} {r['ms_step']:7.3f}  {r['speedup_vs_fp16_full']:.3f}x  "
              f"({r['pct_vs_fp16_full']:+.2f}%)", flush=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": BATCH, "steps": STEPS,
               "seed": SEED, "arms": recs}, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
