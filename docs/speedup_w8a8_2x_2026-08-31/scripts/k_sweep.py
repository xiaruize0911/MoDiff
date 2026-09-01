"""K sweep: Frozen Residual (REPLAY_K) and Store Freeze (CACHE_SKIP_K).

K in {1,2,3,5,7,10,15,20}. K=1 is Full MoDiff (shared). Same process, honest reset.
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
from PIL import Image, ImageDraw, ImageFont
import integration.benchmarks.benchmark_ldm as B
import ahat_fake_quant_grid as G

BATCH, STEPS, NQ, SEED = 128, 50, 4, 20260805
SHAPE = (4, 32, 32)
KS = [1, 2, 3, 5, 7, 10, 15, 20]
OUT_JSON = "docs/speedup_w8a8_2x_2026-08-31/data/k_sweep.json"
OUT_DIR = "docs/speedup_w8a8_2x_2026-08-31/plots"


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


def setup(mode):
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
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


def grid(path, images, nq=NQ):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    cell, pad, lab = 256, 6, 48
    W = pad + nq * (cell + pad)
    H = pad + len(images) * (cell + lab + pad)
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for title, sub, arr in images:
        dr.text((pad, y + 4), title, fill=(11, 11, 11), font=font)
        dr.text((pad, y + 24), sub, fill=(70, 70, 70), font=font_sm)
        y += lab
        for i in range(min(nq, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path, "PNG")
    print("wrote", path, flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16")
    fp16, trials = time_n(model, sampler, quantized=False)
    ref = gen_lat(model, sampler, NQ, SEED, quantized=False).float().clone()
    fp16_img = G.decode(model, ref)
    rows.append({"scheme": "fp16", "K": None, "ms_step": fp16, "trials": trials,
                 "vs_fp16": 1.0, "relL2_vs_fp16": 0.0})
    print(f"  {fp16:.2f}", flush=True)
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int8 MoDiff (reuse one model) =====", flush=True)
    model, sampler = setup("int8")

    residual_imgs = [("fp16  (reference)", f"{fp16:.1f} ms   1.00x   relL2 0.000", fp16_img)]
    store_imgs = [("fp16  (reference)", f"{fp16:.1f} ms   1.00x   relL2 0.000", fp16_img)]

    # K=1 once = Full MoDiff
    print("===== Full MoDiff K=1 =====", flush=True)
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    ms, trials = time_n(model, sampler)
    lat = gen_lat(model, sampler, NQ, SEED)
    rel = float((lat.float() - ref).norm() / ref.norm())
    rec = {"scheme": "full_modiff", "K": 1, "ms_step": ms, "trials": trials,
           "vs_fp16": fp16 / ms, "relL2_vs_fp16": rel}
    rows.append(rec)
    arr = G.decode(model, lat)
    residual_imgs.append((f"A  Full MoDiff  K=1",
                          f"{ms:.1f} ms   {fp16/ms:.2f}x   relL2 {rel:.3f}", arr))
    store_imgs.append((f"A  Full MoDiff  K=1  (= Store Freeze K=1)",
                       f"{ms:.1f} ms   {fp16/ms:.2f}x   relL2 {rel:.3f}", arr))
    print(f"  {ms:.2f}  {fp16/ms:.3f}x  relL2 {rel:.4f}", flush=True)

    for k in KS:
        if k == 1:
            continue
        print(f"===== Frozen Residual K={k} =====", flush=True)
        os.environ["MODIFF_REPLAY_K"] = str(k)
        os.environ["MODIFF_CACHE_SKIP_K"] = "1"
        ms, trials = time_n(model, sampler)
        lat = gen_lat(model, sampler, NQ, SEED)
        rel = float((lat.float() - ref).norm() / ref.norm())
        rec = {"scheme": "frozen_residual", "K": k, "ms_step": ms, "trials": trials,
               "vs_fp16": fp16 / ms, "relL2_vs_fp16": rel}
        rows.append(rec)
        residual_imgs.append((f"B  Frozen Residual  K={k}",
                              f"{ms:.1f} ms   {fp16/ms:.2f}x   relL2 {rel:.3f}",
                              G.decode(model, lat)))
        print(f"  {ms:.2f}  {fp16/ms:.3f}x  relL2 {rel:.4f}", flush=True)

    for k in KS:
        if k == 1:
            continue
        print(f"===== Store Freeze K={k} =====", flush=True)
        os.environ["MODIFF_REPLAY_K"] = "1"
        os.environ["MODIFF_CACHE_SKIP_K"] = str(k)
        ms, trials = time_n(model, sampler)
        lat = gen_lat(model, sampler, NQ, SEED)
        rel = float((lat.float() - ref).norm() / ref.norm())
        rec = {"scheme": "store_freeze", "K": k, "ms_step": ms, "trials": trials,
               "vs_fp16": fp16 / ms, "relL2_vs_fp16": rel}
        rows.append(rec)
        store_imgs.append((f"D  Store Freeze  K={k}",
                           f"{ms:.1f} ms   {fp16/ms:.2f}x   relL2 {rel:.3f}",
                           G.decode(model, lat)))
        print(f"  {ms:.2f}  {fp16/ms:.3f}x  relL2 {rel:.4f}", flush=True)

    del model, sampler
    torch.cuda.empty_cache()

    grid(f"{OUT_DIR}/k_sweep_frozen_residual.png", residual_imgs)
    grid(f"{OUT_DIR}/k_sweep_store_freeze.png", store_imgs)
    json.dump({"gpu": torch.cuda.get_device_name(0), "fp16_ms_step": fp16,
               "target_ms": fp16 / 2, "K": KS, "arms": rows},
              open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
