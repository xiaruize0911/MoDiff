"""A/B: per-conv replay vs ResBlock-level early-out (MODIFF_REPLAY_BLOCK).

Same process, W8A8, batch 128, 50 DDIM. Prints ms/step and latent relL2 of
block-replay vs per-conv replay at K=4 (should be ~0 if dead-code only).
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
os.environ.setdefault("MODIFF_AHAT_BITS", "16")

import torch
import integration.benchmarks.benchmark_ldm as B

STEPS = 50
BATCH = 128
SHAPE = (4, 32, 32)
SEED = 20260827
N_QUAL = 2


def reset(model):
    unet = model.model.diffusion_model
    B.reset_modiff_state_int8(unet)
    B._reset_wxax_modiff_safe(model)


def time_once(model, sampler):
    reset(model)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / STEPS


def gen_latent(model, sampler, n, seed):
    reset(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        z = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    if isinstance(z, tuple):
        z = z[0]
    return z.float()


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/residual_replay_2026-08-27/tmp_block",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_REPLAY_BLOCK"] = "1"
    model, sampler = runner._setup_model("int8")

    rows = []
    if os.environ.get("QUAL_ONLY") != "1":
        arms = [
            ("K=1 full", "1", "1"),
            ("K=4 per-conv", "4", "0"),
            ("K=4 block", "4", "1"),
            ("K=8 per-conv", "8", "0"),
            ("K=8 block", "8", "1"),
        ]
        for label, k, block in arms:
            os.environ["MODIFF_REPLAY_K"] = k
            os.environ["MODIFF_REPLAY_BLOCK"] = block
            print(f"warmup {label}", flush=True)
            time_once(model, sampler)
            ms = time_once(model, sampler)
            rows.append({"label": label, "k": k, "block": block, "ms_step": ms})
            print(f"  {label:18s} {ms:.3f} ms/step", flush=True)

    os.environ["MODIFF_REPLAY_K"] = "4"
    os.environ["MODIFF_REPLAY_BLOCK"] = "0"
    print("qual warmup", flush=True)
    gen_latent(model, sampler, N_QUAL, 20260805)
    z00 = gen_latent(model, sampler, N_QUAL, 20260805)
    z01 = gen_latent(model, sampler, N_QUAL, 20260805)
    os.environ["MODIFF_REPLAY_BLOCK"] = "1"
    z1 = gen_latent(model, sampler, N_QUAL, 20260805)
    rel_ctrl = (z01 - z00).pow(2).mean().sqrt() / z00.pow(2).mean().sqrt()
    rel = (z1 - z00).pow(2).mean().sqrt() / z00.pow(2).mean().sqrt()
    print(f"K=4 per-conv vs per-conv relL2: {float(rel_ctrl):.6e}", flush=True)
    print(f"K=4 block vs per-conv relL2:    {float(rel):.6e}", flush=True)

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "arms": rows,
        "block_vs_perconv_relL2": float(rel),
        "perconv_repeat_relL2": float(rel_ctrl),
    }
    path = "docs/residual_replay_2026-08-27/data/replay_block_bench.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
