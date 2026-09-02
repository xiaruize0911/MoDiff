"""Same-process 4-arm e2e after the zp host-sync fix. No leftover skip/out."""
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

BATCH, STEPS = 128, 50
SHAPE = (4, 32, 32)
OUT = "docs/speedup_2x_2026-08-31/data/e2e_four_arm.json"


def time_n(model, sampler, mode, n=2, warm=1):
    q = mode != "fp16"
    def once():
        if q:
            if "int4" in mode:
                B.reset_modiff_state_int4(model.model.diffusion_model)
            else:
                B.reset_modiff_state_int8(model.model.diffusion_model)
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
    fp16 = None
    for mode, rk, label in [
        ("fp16", 1, "fp16"),
        ("int8_baseline", 1, "int8_baseline"),
        ("int4_baseline", 1, "int4_baseline"),
        ("int4", 1, "int4_modiff_k1"),
        ("int4", 4, "int4_replay_k4"),
    ]:
        print(f"===== {label} =====", flush=True)
        model, sampler = setup(mode, replay_k=rk)
        ms, trials = time_n(model, sampler, mode)
        if fp16 is None:
            fp16 = ms
        rec = {"arm": label, "mode": mode, "replay_k": rk, "ms_step": ms,
               "trials": trials, "vs_fp16": fp16 / ms}
        rows.append(rec)
        print(f"  {ms:.2f}  {fp16/ms:.3f}x  {trials}", flush=True)
        del model, sampler
        torch.cuda.empty_cache()
        os.environ["MODIFF_REPLAY_K"] = "1"
    json.dump({"gpu": torch.cuda.get_device_name(0), "fp16_ms_step": fp16,
               "target_ms": fp16 / 2, "arms": rows}, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
