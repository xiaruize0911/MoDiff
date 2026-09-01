"""CACHE_SKIP_K sweep on scheme A.

K=1 is full MoDiff (write a_hat/o_hat every step). K>1 still runs GN+Q+conv
every step; commits cache only when step_count % K == 0.

B/C residual replay is gone. This is the only remaining skip-K.

Protocol: LDM-8 LSUN-Churches, A40, batch 128 timing, n=4 latent, DDIM 50,
seed 20260805, W8A8, LINEAR=0, static delta, flash attn.
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
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
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
KS = [1, 2, 3, 5, 7, 10]
OUT_DIR = "docs/scheme_a_only_2026-09-01"
OUT_JSON = f"{OUT_DIR}/data/k_sweep.json"
OUT_PNG = f"{OUT_DIR}/plots/k_sweep.png"


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
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{OUT_DIR}/tmp",
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


def relL2(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    denom = float(b.norm())
    return float((a - b).norm() / denom) if denom > 0 else float("nan")


def grid(path, images, cell=256, pad=12, lab=48):
    nq = images[0][2].shape[0]
    rows = len(images)
    W = pad + nq * (cell + pad)
    H = pad + rows * (cell + lab + pad)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = font_sm = ImageFont.load_default()
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
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

    rows, images, lats = [], [], {}

    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16")
    fp16, trials = time_n(model, sampler, quantized=False)
    ref = gen_lat(model, sampler, NQ, SEED, quantized=False)
    lats["fp16"] = ref
    rows.append({
        "arm": "fp16", "K": None, "ms_step": fp16, "trials": trials,
        "ms_sample": fp16 * STEPS, "vs_fp16": 1.0, "relL2_vs_fp16": 0.0,
    })
    images.append(("fp16  S=50", f"{fp16:.1f} ms/step  1.00x  relL2 0.000",
                   G.decode(model, ref)))
    print(f"  {fp16:.2f} ms/step", flush=True)
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int8 MoDiff CACHE_SKIP_K sweep =====", flush=True)
    model, sampler = setup("int8")
    a_lat = None
    for k in KS:
        print(f"----- K={k} -----", flush=True)
        os.environ["MODIFF_CACHE_SKIP_K"] = str(k)
        ms, trials = time_n(model, sampler, quantized=True)
        lat = gen_lat(model, sampler, NQ, SEED, quantized=True)
        lats[f"K{k}"] = lat
        rel_fp = relL2(lat, ref)
        rec = {
            "arm": f"A_skipK{k}", "K": k, "ms_step": ms, "trials": trials,
            "ms_sample": ms * STEPS, "vs_fp16": fp16 / ms,
            "relL2_vs_fp16": rel_fp, "skip_frac": 0.0 if k <= 1 else (k - 1) / k,
        }
        rows.append(rec)
        print(f"  {ms:.2f} ms/step  {fp16/ms:.3f}x  relL2 {rel_fp:.4f}", flush=True)
        images.append((
            f"A  CACHE_SKIP_K={k}" + ("  (no skip)" if k == 1 else f"  skip {(k-1)/k:.0%}"),
            f"{ms:.1f} ms/step  {fp16/ms:.2f}x  relL2 {rel_fp:.3f}",
            G.decode(model, lat),
        ))
        if k == 1:
            a_lat = lat
    del model, sampler
    torch.cuda.empty_cache()

    for rec in rows:
        if rec["K"] is None:
            rec["relL2_vs_A"] = relL2(lats["fp16"], a_lat)
            rec["vs_A"] = None
            continue
        rec["relL2_vs_A"] = relL2(lats[f"K{rec['K']}"], a_lat)
        rec["vs_A"] = rows[1]["ms_step"] / rec["ms_step"]  # vs K=1

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "protocol": {
            "model": "LDM-8 LSUN-Churches", "batch_time": BATCH, "steps": STEPS,
            "quality_n": NQ, "seed": SEED,
            "knob": "MODIFF_CACHE_SKIP_K",
            "meaning": "still compute; skip a_hat/o_hat store on K-1 of K modulated steps",
        },
        "fp16_ms_step": fp16,
        "arms": rows,
        "grid": OUT_PNG,
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, flush=True)
    grid(OUT_PNG, images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
