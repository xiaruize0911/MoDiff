"""INT4 replay K=8/16, leftover skip/out quant, CUDA-graph PTQ.

Target: ≤ fp16/2 ms/step (2.0×). fp16 measured in-process.
"""
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
os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"

import torch
import integration.benchmarks.benchmark_ldm as B
from integration.kernels.int8_cudagraph import (
    install_cuda_graph_replay_pytorch_int8, get_cuda_graph_replay_stats)

BATCH, STEPS = 128, 50
SHAPE = (4, 32, 32)
OUT = "docs/speedup_2x_2026-08-31/data/int4_k_and_graph.json"


def time_n(model, sampler, n=2, warm=1, quantized=True):
    def once():
        mgr = getattr(model.model.diffusion_model, "_cuda_graph_manager", None)
        if mgr is not None:
            mgr.reset_sequence()
        if quantized:
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


def freeze_uncalibrated_int4(model):
    """Give leftover skip/out wrappers a static scale so they take the fused PTQ path."""
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    n = 0
    for m in model.model.diffusion_model.modules():
        if isinstance(m, OptimizedInt4Conv2d) and not m.is_calibrated:
            m.set_static_scale(7.0)  # conservative; activations typically << 1 after GN+SiLU
            n += 1
    return n


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []

    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16")
    fp16, trials = time_n(model, sampler, quantized=False)
    rows.append({"arm": "fp16", "ms_step": fp16, "trials": trials, "vs_fp16": 1.0})
    print(f"  {fp16:.2f}  {trials}", flush=True)
    target = fp16 / 2
    del model, sampler
    torch.cuda.empty_cache()

    for k in (8, 16):
        print(f"===== int4 replay K={k} =====", flush=True)
        os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"
        model, sampler = setup("int4", replay_k=k)
        ms, trials = time_n(model, sampler)
        rows.append({"arm": f"int4_replay_{k}", "ms_step": ms, "trials": trials,
                     "vs_fp16": fp16 / ms, "replay_k": k})
        print(f"  {ms:.2f}  {fp16/ms:.3f}x  target={target:.2f}  {trials}", flush=True)
        del model, sampler
        torch.cuda.empty_cache()
        os.environ["MODIFF_REPLAY_K"] = "1"

    print("===== int4_baseline + leftover skip/out =====", flush=True)
    os.environ["MODIFF_QUANT_SKIP_OUT"] = "1"
    model, sampler = setup("int4_baseline")
    n_fr = freeze_uncalibrated_int4(model)
    print(f"  froze {n_fr} leftover layers", flush=True)
    ms, trials = time_n(model, sampler)
    rows.append({"arm": "int4_baseline_skip_out", "ms_step": ms, "trials": trials,
                 "vs_fp16": fp16 / ms, "froze": n_fr})
    print(f"  {ms:.2f}  {fp16/ms:.3f}x  {trials}", flush=True)

    print("===== leftover PTQ + CUDA graph =====", flush=True)
    try:
        mgr = install_cuda_graph_replay_pytorch_int8(model.model, batch_size=BATCH, shape=SHAPE)
        time_n(model, sampler, n=1, warm=0)
        stats = get_cuda_graph_replay_stats(model.model.diffusion_model)
        print("  capture stats", stats, flush=True)
        if stats and not stats.get("disabled"):
            ms, trials = time_n(model, sampler, n=2, warm=0)
            stats2 = get_cuda_graph_replay_stats(model.model.diffusion_model)
            rows.append({"arm": "int4_baseline_skip_out_graph", "ms_step": ms,
                         "trials": trials, "vs_fp16": fp16 / ms, "stats": stats2})
            print(f"  graph {ms:.2f}  {fp16/ms:.3f}x  {trials}  {stats2}", flush=True)
        else:
            rows.append({"arm": "int4_baseline_skip_out_graph", "error": "disabled", "stats": stats})
            print("  graph disabled", flush=True)
    except Exception as e:
        print("  graph failed", e, flush=True)
        rows.append({"arm": "int4_baseline_skip_out_graph", "error": str(e)})
    del model, sampler
    torch.cuda.empty_cache()
    os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"

    print("===== int4 replay K=8 + leftover =====", flush=True)
    os.environ["MODIFF_QUANT_SKIP_OUT"] = "1"
    model, sampler = setup("int4", replay_k=8)
    n_fr = freeze_uncalibrated_int4(model)
    ms, trials = time_n(model, sampler)
    rows.append({"arm": "int4_replay_8_skip_out", "ms_step": ms, "trials": trials,
                 "vs_fp16": fp16 / ms, "froze": n_fr, "replay_k": 8})
    print(f"  {ms:.2f}  {fp16/ms:.3f}x  froze={n_fr}  {trials}", flush=True)
    del model, sampler
    torch.cuda.empty_cache()
    os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"
    os.environ["MODIFF_REPLAY_K"] = "1"

    json.dump({"gpu": torch.cuda.get_device_name(0), "fp16_ms_step": fp16,
               "target_ms": target, "arms": rows}, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    best = min((r for r in rows if "ms_step" in r), key=lambda r: r["ms_step"])
    print(f"BEST {best['arm']} {best['ms_step']:.2f} ms  {best.get('vs_fp16', 0):.3f}x  "
          f"need {target:.2f} for 2.0x", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
