"""Eager residual-replay vs CUDA-graph full/replay paths.

Installs UNet graphs after model setup: `first`, `modulated` (commit), and
`residual_replay` (skip GN+conv). Times the sample AFTER capture/reset.
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
os.environ.setdefault("MODIFF_AHAT_BITS", "16")

import torch
import integration.benchmarks.benchmark_ldm as B
from integration.kernels.int8_cudagraph import (
    install_cuda_graph_replay_pytorch_int8,
    get_cuda_graph_replay_stats,
)

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
    os.environ["MODIFF_REPLAY_K"] = "4"
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/residual_replay_2026-08-27/tmp_graph",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)
    model, sampler = runner._setup_model("int8")

    rows = []

    os.environ["MODIFF_REPLAY_K"] = "1"
    print("warmup eager K=1", flush=True)
    time_once(model, sampler)
    ms = time_once(model, sampler)
    rows.append({"label": "eager K=1", "ms_step": ms})
    print(f"  eager K=1          {ms:.3f} ms/step", flush=True)

    os.environ["MODIFF_REPLAY_K"] = "4"
    print("warmup eager K=4", flush=True)
    time_once(model, sampler)
    ms = time_once(model, sampler)
    rows.append({"label": "eager K=4", "ms_step": ms})
    print(f"  eager K=4          {ms:.3f} ms/step", flush=True)

    print("installing CUDA graphs (first / modulated / residual_replay)", flush=True)
    mgr = install_cuda_graph_replay_pytorch_int8(model.model, batch_size=BATCH, shape=SHAPE)
    print("precapture graphs", flush=True)
    time_once(model, sampler)
    stats = get_cuda_graph_replay_stats(model.model.diffusion_model) or {}
    print(f"  graph stats after precapture: {stats}", flush=True)
    ms = time_once(model, sampler)
    stats2 = get_cuda_graph_replay_stats(model.model.diffusion_model) or {}
    rows.append({"label": "graph K=4", "ms_step": ms, "stats": stats2})
    print(f"  graph K=4          {ms:.3f} ms/step  stats={stats2}", flush=True)

    print("qual warmup (n=2, dedicated graphs)", flush=True)
    os.environ["MODIFF_REPLAY_K"] = "4"
    model.model.diffusion_model._cuda_graph_manager = None
    gen_latent(model, sampler, N_QUAL, 20260805)
    z_eager = gen_latent(model, sampler, N_QUAL, 20260805)
    mgr2 = install_cuda_graph_replay_pytorch_int8(model.model, batch_size=N_QUAL, shape=SHAPE)
    gen_latent(model, sampler, N_QUAL, 20260805)  # capture
    z_graph = gen_latent(model, sampler, N_QUAL, 20260805)
    rel = (z_graph - z_eager).pow(2).mean().sqrt() / z_eager.pow(2).mean().sqrt()
    qstats = get_cuda_graph_replay_stats(model.model.diffusion_model) or {}
    print(f"graph vs eager K=4 relL2: {float(rel):.6e}  stats={qstats}", flush=True)

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "arms": rows,
        "graph_vs_eager_relL2": float(rel),
        "precapture_stats": stats,
    }
    path = "docs/residual_replay_2026-08-27/data/replay_cudagraph_bench.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
