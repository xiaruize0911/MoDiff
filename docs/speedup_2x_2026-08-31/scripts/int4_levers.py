"""Measure INT4 PTQ ± compile and INT4 MoDiff replay vs fp16.

Target: ≤51.2 ms/step (2× vs fp16 102.53).
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

import torch
import integration.benchmarks.benchmark_ldm as B

BATCH, STEPS = 128, 50
SHAPE = (4, 32, 32)
OUT = "docs/speedup_2x_2026-08-31/data/int4_levers.json"


def time_n(model, sampler, n=2, warm=1, quantized=True):
    def once():
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


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    fp16 = 102.53  # just measured in this session

    print("===== int4_baseline =====", flush=True)
    model, sampler = setup("int4_baseline")
    ms, trials = time_n(model, sampler)
    rows.append({"arm": "int4_baseline", "ms_step": ms, "trials": trials,
                 "vs_fp16": fp16 / ms})
    print(f"  {ms:.2f} ms/step  {fp16/ms:.3f}x  {trials}", flush=True)

    print("===== int4_baseline + torch.compile =====", flush=True)
    try:
        model.model.diffusion_model = torch.compile(
            model.model.diffusion_model, mode="reduce-overhead", fullgraph=False)
        print("  compiled", flush=True)
        ms, trials = time_n(model, sampler, warm=2, n=2)
        rows.append({"arm": "int4_baseline_compile", "ms_step": ms, "trials": trials,
                     "vs_fp16": fp16 / ms})
        print(f"  {ms:.2f} ms/step  {fp16/ms:.3f}x  {trials}", flush=True)
    except Exception as e:
        print("  compile failed", e, flush=True)
        rows.append({"arm": "int4_baseline_compile", "error": str(e)})
    del model, sampler
    torch.cuda.empty_cache()

    for k in (2, 4):
        print(f"===== int4 replay K={k} =====", flush=True)
        model, sampler = setup("int4", replay_k=k)
        ms, trials = time_n(model, sampler)
        rows.append({"arm": f"int4_replay_{k}", "ms_step": ms, "trials": trials,
                     "vs_fp16": fp16 / ms, "replay_k": k})
        print(f"  {ms:.2f} ms/step  {fp16/ms:.3f}x  {trials}", flush=True)
        del model, sampler
        torch.cuda.empty_cache()
        os.environ["MODIFF_REPLAY_K"] = "1"

    json.dump({"gpu": torch.cuda.get_device_name(0), "fp16_ms_step": fp16,
               "target_ms": fp16 / 2, "arms": rows}, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
