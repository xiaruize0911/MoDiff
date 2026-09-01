"""Re-time Store Freeze after compile-time write_ahat skip.

Same-process, honest reset. K in {1, 2, 5, 20}. fp16 from k_sweep.json.
"""
import json, os, statistics, sys
ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [
    ROOT, os.path.join(ROOT, "src/taming-transformers"),
    os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts"),
]
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_DELTA_MODE"] = "static"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_DROP_OHAT"] = "0"
os.environ["MODIFF_ATTN_REPLAY_K"] = "1"
os.environ["MODIFF_QUANT_LINEAR"] = "1"
os.environ["MODIFF_QUANT_ATTN"] = "1"
os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"
os.environ["MODIFF_QATTN_FLASH"] = "1"
os.environ["MODIFF_FLASH_GATE"] = "on"
os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"
os.environ["MODIFF_WARMUP_STEPS"] = "1"

import torch
import integration.benchmarks.benchmark_ldm as B
import ahat_fake_quant_grid as G

BATCH, STEPS, NQ, SEED = 128, 50, 4, 20260805
SHAPE = (4, 32, 32)
KS = [1, 2, 5, 20]
OUT = "docs/speedup_w8a8_2x_2026-08-31/data/store_freeze_rebench.json"


def reset_all(model):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def time_n(model, sampler, n=2, warm=1):
    def once():
        reset_all(model)
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


def setup():
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/speedup_w8a8_2x_2026-08-31/tmp",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model("int8")


def gen_lat(model, sampler, n, seed):
    reset_all(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    prev = json.load(open("docs/speedup_w8a8_2x_2026-08-31/data/k_sweep.json"))
    fp16 = prev["fp16_ms_step"]
    print(f"fp16 (from k_sweep) {fp16:.2f} ms", flush=True)
    print("===== int8 MoDiff =====", flush=True)
    model, sampler = setup()
    ref = gen_lat(model, sampler, NQ, SEED)
    rows = []
    for k in KS:
        os.environ["MODIFF_CACHE_SKIP_K"] = str(k)
        os.environ["MODIFF_REPLAY_K"] = "1"
        print(f"===== Store Freeze K={k} =====", flush=True)
        ms, trials = time_n(model, sampler)
        lat = gen_lat(model, sampler, NQ, SEED)
        rel = float((lat - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
        rec = {"K": k, "ms_step": ms, "trials": trials,
               "vs_fp16": fp16 / ms, "vs_k1": None, "relL2_vs_k1": rel}
        rows.append(rec)
        print(f"  {ms:.2f}  {fp16/ms:.3f}x  relL2 vs K=1 {rel:.4f}", flush=True)
    k1 = rows[0]["ms_step"]
    for r in rows:
        r["vs_k1"] = k1 / r["ms_step"]
    payload = {"gpu": torch.cuda.get_device_name(0), "fp16_ms_step": fp16,
               "K": KS, "arms": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
