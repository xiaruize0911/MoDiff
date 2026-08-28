"""One-process replay-K bench: batch 128, 50 DDIM, A40 UNet time (no VAE).

Arms: INT8 K=1, replay 2/4/8, INT8 baseline. ms/step = 1000 * wall / steps.
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
os.environ.setdefault("MODIFF_REPLAY_K", "1")

import torch
import integration.benchmarks.benchmark_ldm as B

STEPS = 50
BATCH = 128
SHAPE = (4, 32, 32)
SEED = 20260827
OUT_JSON = "docs/residual_replay_2026-08-27/data/replay_bench.json"


def reset_int8(model):
    unet = model.model.diffusion_model
    B.reset_modiff_state_int8(unet)
    B._reset_wxax_modiff_safe(model)


def time_once(model, sampler, mode_reset):
    reset_int8(model)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / STEPS


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={BATCH} steps={STEPS}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/residual_replay_2026-08-27/tmp_bench",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)

    os.environ["MODIFF_REPLAY_K"] = "1"
    model, sampler = runner._setup_model("int8")
    arms = [("INT8 K=1 (full compute)", "1"),
            ("INT8 replay-K=2", "2"),
            ("INT8 replay-K=4", "4"),
            ("INT8 replay-K=8", "8")]
    results = []
    for label, k in arms:
        os.environ["MODIFF_REPLAY_K"] = k
        print(f"warmup {label}", flush=True)
        time_once(model, sampler, True)
        ms = time_once(model, sampler, True)
        results.append((label, k, ms))
        print(f"  {label:28s} {ms:.3f} ms/step", flush=True)

    del model, sampler
    torch.cuda.empty_cache()
    os.environ["MODIFF_REPLAY_K"] = "1"
    model, sampler = runner._setup_model("int8_baseline")
    print("warmup INT8 baseline", flush=True)
    time_once(model, sampler, True)
    ms_b = time_once(model, sampler, True)
    results.append(("INT8 baseline", "na", ms_b))
    print(f"  {'INT8 baseline':28s} {ms_b:.3f} ms/step", flush=True)

    ref = results[0][2]
    print("\n===== vs INT8 K=1 =====", flush=True)
    recs = []
    for label, k, ms in results:
        spd = ref / ms
        pct = 100.0 * (ref - ms) / ref
        recs.append({"label": label, "k": k, "ms_step": ms, "speedup_vs_k1": spd, "pct_vs_k1": pct})
        print(f"  {label:28s} {ms:7.3f} ms/step  {spd:.3f}x  ({pct:+.2f}% vs K=1)", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": BATCH, "steps": STEPS,
               "seed": SEED, "arms": recs}, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
