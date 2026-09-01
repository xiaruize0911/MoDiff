"""Remeasure scheme A only: full W8A8 MoDiff vs fp16, DDIM 50 and 25.

B (REPLAY_K>1) and C (DROP_OHAT) have been deleted from the kernels. This is
the production path: every step runs GN+Q+INT8 conv and writes a_hat/o_hat.

Protocol matches docs/bc_identity_halfstep_2026-09-01: LDM-8 LSUN-Churches,
A40, batch 128 timing, n=4 latent quality, DDIM η=0, seed 20260805.
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

BATCH, NQ, SEED = 128, 4, 20260805
SHAPE = (4, 32, 32)
OUT_DIR = "docs/scheme_a_only_2026-09-01"
OUT_JSON = f"{OUT_DIR}/data/remeasure_a.json"
OUT_PNG = f"{OUT_DIR}/plots/a_grid.png"

ARMS = [
    ("fp16_s50", "fp16  S=50", "fp16", 50),
    ("fp16_s25", "fp16  S=25", "fp16", 25),
    ("A_s50", "A  Full MoDiff  S=50", "int8", 50),
    ("A_s25", "A  Full MoDiff  S=25", "int8", 25),
]


def reset_all(model, quantized=True):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)


def time_n(model, sampler, steps, n=2, warm=1, quantized=True):
    def once():
        reset_all(model, quantized=quantized)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=steps, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    for _ in range(warm):
        once()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); once(); e.record(); torch.cuda.synchronize()
        xs.append(s.elapsed_time(e) / steps)
    return statistics.median(xs), xs


def setup(mode):
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir=f"{OUT_DIR}/tmp",
        batch_size=BATCH, steps=50, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def gen_lat(model, sampler, n, seed, steps, quantized=True):
    reset_all(model, quantized=quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
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

    lats, images, rows = {}, [], []
    fp16_ms_s50 = None

    for mode in ("fp16", "int8"):
        print(f"===== {mode} =====", flush=True)
        model, sampler = setup(mode)
        quantized = mode != "fp16"
        for aid, label, m, steps in ARMS:
            if m != mode:
                continue
            print(f"----- {aid} -----", flush=True)
            ms, trials = time_n(model, sampler, steps, quantized=quantized)
            lat = gen_lat(model, sampler, NQ, SEED, steps, quantized=quantized)
            lats[aid] = lat
            if aid == "fp16_s50":
                fp16_ms_s50 = ms
            rec = {
                "arm": aid, "label": label, "ms_step": ms, "trials": trials,
                "steps": steps, "ms_sample": ms * steps,
                "vs_fp16_s50_sample": (fp16_ms_s50 * 50) / (ms * steps),
            }
            rows.append(rec)
            print(f"  {ms:.2f} ms/step  {ms*steps:.1f} ms/sample  "
                  f"{rec['vs_fp16_s50_sample']:.3f}x vs fp16-S50", flush=True)
            images.append((label, f"{ms:.1f} ms/step   {ms*steps:.0f} ms/sample",
                           G.decode(model, lat)))
        del model, sampler
        torch.cuda.empty_cache()

    pairwise = {}
    keys = [a[0] for a in ARMS]
    ref = lats["fp16_s50"]
    for k in keys:
        pairwise[k] = {"vs_fp16_s50": relL2(lats[k], ref)}
        for k2 in keys:
            pairwise[k][f"vs_{k2}"] = relL2(lats[k], lats[k2])
        for rec in rows:
            if rec["arm"] == k:
                rec["relL2_vs_fp16_s50"] = pairwise[k]["vs_fp16_s50"]

    highlight = {
        "A_s50_vs_fp16_s50": pairwise["A_s50"]["vs_fp16_s50"],
        "A_s25_vs_fp16_s50": pairwise["A_s25"]["vs_fp16_s50"],
        "A_s25_vs_fp16_s25": pairwise["A_s25"]["vs_fp16_s25"],
        "A_s50_vs_A_s25": pairwise["A_s50"]["vs_A_s25"],
        "fp16_s25_vs_fp16_s50": pairwise["fp16_s25"]["vs_fp16_s50"],
    }
    print("highlights:", json.dumps(highlight, indent=2), flush=True)

    labeled = []
    img_by_arm = {aid: arr for (aid, *_), (_, _, arr) in zip(
        [a for a in ARMS if a[2] == "fp16"] + [a for a in ARMS if a[2] == "int8"],
        images)}
    for aid, label, mode, steps in ARMS:
        rec = next(r for r in rows if r["arm"] == aid)
        labeled.append((
            label,
            f"{rec['ms_step']:.1f} ms/step  {rec['ms_sample']:.0f} ms/sample  "
            f"{rec['vs_fp16_s50_sample']:.2f}x  relL2 {rec['relL2_vs_fp16_s50']:.3f}",
            img_by_arm[aid],
        ))
    grid(OUT_PNG, labeled)

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "protocol": {
            "model": "LDM-8 LSUN-Churches", "batch_time": BATCH, "quality_n": NQ,
            "seed": SEED, "scheme": "A full MoDiff only (B/C deleted)",
        },
        "fp16_s50_ms_step": fp16_ms_s50,
        "arms": rows,
        "highlight": highlight,
        "pairwise_relL2": {k: {kk: vv for kk, vv in v.items()
                               if kk in ("vs_fp16_s50", "vs_fp16_s25", "vs_A_s50", "vs_A_s25")}
                           for k, v in pairwise.items()},
        "grid": OUT_PNG,
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
