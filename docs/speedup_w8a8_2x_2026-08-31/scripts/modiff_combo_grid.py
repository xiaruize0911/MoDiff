"""Quality grid for MoDiff-native W8A8 (no attn replay, no CUDA graph)."""
import json, os, sys
ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [
    ROOT, os.path.join(ROOT, "src/taming-transformers"),
    os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts"),
]
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_DELTA_MODE"] = "static"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_ATTN_REPLAY_K"] = "1"
os.environ["MODIFF_QUANT_LINEAR"] = "1"
os.environ["MODIFF_QUANT_ATTN"] = "1"
os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1"
os.environ["MODIFF_QATTN_FLASH"] = "1"
os.environ["MODIFF_FLASH_GATE"] = "on"
os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"

import torch
from PIL import Image, ImageDraw
import integration.benchmarks.benchmark_ldm as B
import ahat_fake_quant_grid as G

BATCH_N, STEPS, SEED = 4, 50, 20260805
SHAPE = (4, 32, 32)
OUT_PNG = "docs/speedup_w8a8_2x_2026-08-31/plots/modiff_combo.png"
OUT_JSON = "docs/speedup_w8a8_2x_2026-08-31/data/modiff_combo.json"

# Times from the same-process combo run that crashed after graph capture.
TIMES = {
    "fp16": 105.8533056640625,
    "int8_ptq": 64.65216796875,
    "modiff_k1_w1": 74.20720458984375,
    "modiff_replay2_w1": 54.937431640625,
    "modiff_replay3_w1": 48.7685107421875,
}


def setup(mode, replay_k=1, warmup=1):
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_WARMUP_STEPS"] = str(warmup)
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/speedup_w8a8_2x_2026-08-31/tmp",
        batch_size=128, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def gen_lat(model, sampler, n, seed, quantized=True):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fp16 = TIMES["fp16"]
    images, quality, rows = [], {}, []
    ref = None
    specs = [
        ("fp16", "fp16", 1),
        ("int8_ptq", "int8_baseline", 1),
        ("modiff_k1_w1", "int8", 1),
        ("modiff_replay2_w1", "int8", 2),
        ("modiff_replay3_w1", "int8", 3),
    ]
    for arm, mode, rk in specs:
        print(f"===== {arm} =====", flush=True)
        model, sampler = setup(mode, replay_k=rk, warmup=1)
        q = mode != "fp16"
        lat = gen_lat(model, sampler, BATCH_N, SEED, quantized=q)
        if ref is None:
            ref = lat.float().clone()
        rel = float((lat.float() - ref).norm() / ref.norm())
        ms = TIMES[arm]
        quality[arm] = {"relL2_vs_fp16": rel}
        rows.append({"arm": arm, "ms_step": ms, "vs_fp16": fp16 / ms,
                     "replay_k": rk, "warmup": 1, "relL2_vs_fp16": rel})
        images.append((f"{arm}  {ms:.1f}ms  {fp16/ms:.2f}x  relL2 {rel:.4f}",
                       G.decode(model, lat)))
        print(f"  relL2 {rel:.4f}", flush=True)
        del model, sampler
        torch.cuda.empty_cache()

    cell, pad, lab = 256, 6, 36
    W = pad + BATCH_N * (cell + pad)
    Hh = len(images) * (cell + lab + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for label, arr in images:
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += lab
        for i in range(min(BATCH_N, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    canvas.save(OUT_PNG, "PNG")
    json.dump({"gpu": "NVIDIA A40", "fp16_ms_step": fp16, "target_ms": fp16 / 2,
               "arms": rows, "quality": quality, "grid": OUT_PNG,
               "note": "times from modiff_combo.py same-process run; grid regenerated after graph capture poisoned CUDA"},
              open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, OUT_PNG, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
