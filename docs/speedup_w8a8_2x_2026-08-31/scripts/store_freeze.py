"""Store-freeze (CACHE_SKIP_K): freeze a_hat/o_hat writes, GN+conv still run."""
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
from PIL import Image, ImageDraw, ImageFont
import integration.benchmarks.benchmark_ldm as B
import ahat_fake_quant_grid as G

BATCH, STEPS, NQ, SEED = 128, 50, 4, 20260805
SHAPE = (4, 32, 32)
OUT_JSON = "docs/speedup_w8a8_2x_2026-08-31/data/store_freeze.json"
OUT_PNG = "docs/speedup_w8a8_2x_2026-08-31/plots/store_freeze.png"
FP16_MS = 105.6582470703125


def reset_all(model, quantized=True):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)


def time_n(model, sampler, n=2, warm=1, quantized=True):
    def once():
        reset_all(model, quantized=quantized)
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


def setup(mode, skip_k=1):
    os.environ["MODIFF_CACHE_SKIP_K"] = str(skip_k)
    os.environ["MODIFF_REPLAY_K"] = "1"
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/speedup_w8a8_2x_2026-08-31/tmp",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def gen_lat(model, sampler, n, seed, quantized=True):
    reset_all(model, quantized=quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    rows, quality, images = [], {}, []
    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16", skip_k=1)
    ref = gen_lat(model, sampler, NQ, SEED, quantized=False)
    images.append(("fp16", G.decode(model, ref)))
    del model, sampler
    torch.cuda.empty_cache()

    specs = [
        ("store_freeze_k3", 3),
        ("store_freeze_k5", 5),
    ]
    for arm, sk in specs:
        print(f"===== {arm} =====", flush=True)
        model, sampler = setup("int8", skip_k=sk)
        ms, trials = time_n(model, sampler)
        lat = gen_lat(model, sampler, NQ, SEED)
        rel = float((lat.float() - ref.float()).norm() / ref.float().norm())
        rec = {"arm": arm, "ms_step": ms, "trials": trials,
               "vs_fp16": FP16_MS / ms, "skip_k": sk, "relL2_vs_fp16": rel}
        rows.append(rec)
        quality[arm] = {"relL2_vs_fp16": rel}
        images.append((arm, G.decode(model, lat)))
        print(f"  {ms:.2f}  {FP16_MS/ms:.3f}x  relL2 {rel:.4f}  {trials}", flush=True)
        del model, sampler
        torch.cuda.empty_cache()

    cell, pad, lab = 256, 6, 36
    W = pad + NQ * (cell + pad)
    Hh = len(images) * (cell + lab + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for label, arr in images:
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += lab
        for i in range(min(NQ, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    canvas.save(OUT_PNG, "PNG")
    json.dump({"gpu": torch.cuda.get_device_name(0), "fp16_ms_step": FP16_MS,
               "arms": rows, "quality": quality, "grid": OUT_PNG},
              open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
