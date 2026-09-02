"""INT4 PTQ + CUDA graph, and INT4 replay K=8. Target 51.3 ms (2× vs 102.53)."""
import json, os, statistics, sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_DELTA_MODE"] = "static"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_QUANT_LINEAR"] = "1"
os.environ["MODIFF_QUANT_ATTN"] = "1"
os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"
os.environ["MODIFF_QATTN_FLASH"] = "1"
os.environ["MODIFF_FLASH_GATE"] = "on"

import torch
import integration.benchmarks.benchmark_ldm as B
from integration.kernels.int8_cudagraph import (
    install_cuda_graph_replay_pytorch_int8, get_cuda_graph_replay_stats)

BATCH, STEPS = 128, 50
SHAPE = (4, 32, 32)
FP16 = 102.53
OUT = "docs/speedup_2x_2026-08-31/data/int4_graph.json"


def setup(mode, replay_k=1):
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/speedup_2x_2026-08-31/tmp",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def time_n(model, sampler, n=2, warm=1):
    def once():
        mgr = getattr(model.model.diffusion_model, "_cuda_graph_manager", None)
        if mgr is not None:
            mgr.reset_sequence()
        B.reset_modiff_state_int4(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    for _ in range(warm):
        once()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); once(); e.record(); torch.cuda.synchronize()
        xs.append(s.elapsed_time(e) / STEPS)
    return statistics.median(xs), xs


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []

    print("===== int4_baseline + CUDA graph =====", flush=True)
    model, sampler = setup("int4_baseline")
    ms, trials = time_n(model, sampler, warm=1)
    rows.append({"arm": "int4_baseline_eager", "ms_step": ms, "trials": trials,
                 "vs_fp16": FP16 / ms})
    print(f"  eager {ms:.2f}  {FP16/ms:.3f}x  {trials}", flush=True)

    mgr = install_cuda_graph_replay_pytorch_int8(model.model, batch_size=BATCH, shape=SHAPE)
    print("  installed graph, capturing on first sample", flush=True)
    time_n(model, sampler, n=1, warm=0)  # capture
    stats = get_cuda_graph_replay_stats(model.model.diffusion_model)
    print("  stats after capture", stats, flush=True)
    ms, trials = time_n(model, sampler, n=2, warm=0)
    stats = get_cuda_graph_replay_stats(model.model.diffusion_model)
    rows.append({"arm": "int4_baseline_graph", "ms_step": ms, "trials": trials,
                 "vs_fp16": FP16 / ms, "stats": stats})
    print(f"  graph {ms:.2f}  {FP16/ms:.3f}x  {trials}  {stats}", flush=True)
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int4 replay K=8 =====", flush=True)
    model, sampler = setup("int4", replay_k=8)
    ms, trials = time_n(model, sampler)
    rows.append({"arm": "int4_replay_8", "ms_step": ms, "trials": trials,
                 "vs_fp16": FP16 / ms})
    print(f"  {ms:.2f}  {FP16/ms:.3f}x  {trials}", flush=True)

    json.dump({"gpu": torch.cuda.get_device_name(0), "fp16_ms_step": FP16,
               "target_ms": FP16 / 2, "arms": rows}, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
