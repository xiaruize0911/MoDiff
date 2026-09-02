"""I-MoDiff contact sheet: integer a_hat, frozen s*, o_hat still fp16.

Arms isolate freeze-the-table from integer math (same split as imode_quality.py):
  fp16          unquantized
  w8a8_full     per-step δ, fp16 a_hat
  frozen_s      scale[0] only, fp16 a_hat, IMODE=0
  imode16/8/4   integer a_hat (int16 / int8 / qmax=7), frozen s*

n=6, 50 DDIM, seed 20260805. W8A8 kernels only (int4 has no IMODE).

Run: source setup_cuda_env.sh
     python docs/cache_schemes_report_2026-08-28/scripts/imode_samples.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_DELTA_FREEZE"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="imode_samples.py")

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.kernels.int8_optimized import OptimizedInt8Conv2d  # noqa: E402

SHAPE = (4, 32, 32)
CALIB8 = "integration/calibration/int8_calibration_realckpt.pt"
OUT_PNG = "docs/cache_schemes_report_2026-08-28/plots/imode_samples_grid.png"
OUT_DIR = "docs/cache_schemes_report_2026-08-28/plots/imode_samples"
JSON_OUT = "docs/cache_schemes_report_2026-08-28/data/imode_samples.json"

# name, imode, bits, freeze
ARMS = (
    ("w8a8_full", False, 16, False),
    ("frozen_s", False, 16, True),
    ("imode16", True, 16, False),
    ("imode8", True, 8, False),
    ("imode4", True, 4, False),
)


def _knobs(imode, bits, freeze):
    os.environ["MODIFF_IMODE"] = "1" if imode else "0"
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_DELTA_FREEZE"] = "1" if freeze else "0"
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    os.environ["MODIFF_AHAT_REFRESH"] = "0"


def _overflow(model):
    fracs, dtypes = [], set()
    for m in model.modules():
        if isinstance(m, OptimizedInt8Conv2d) and m.a_hat_cache is not None:
            dtypes.add(str(m.a_hat_cache.dtype).replace("torch.", ""))
            if m.a_hat_cache.dtype in (torch.int8, torch.int16):
                fracs.append(m.ahat_sat_frac())
    if not fracs:
        return {"n_int": 0, "max_sat": 0.0, "n_over": 0, "ahat_dtype": sorted(dtypes)}
    return {"n_int": len(fracs), "max_sat": max(fracs),
            "n_over": sum(1 for f in fracs if f >= 1.0 - 1e-6),
            "ahat_dtype": sorted(dtypes)}


def _reset(model, quantized):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def sample_lat(runner, model, sampler, n, seed, steps, quantized):
    _reset(model, quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float()


def decode(model, lat, chunk=8):
    lat = lat.to("cuda", torch.float16)
    out = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for i in range(0, lat.shape[0], chunk):
            d = model.decode_first_stage(lat[i:i + chunk])
            out.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0).permute(0, 2, 3, 1).cpu())
            del d
    return (torch.cat(out, 0).numpy() * 255).round().astype("uint8")


def save_pngs(arr, folder):
    os.makedirs(folder, exist_ok=True)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(os.path.join(folder, f"{i:06d}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  n={a.n} steps={a.steps}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(OUT_DIR, "_tmp"),
        batch_size=a.n, steps=a.steps, shape=SHAPE,
        calibration_path=None, auto_delta_table=False)

    rows, quality, ref = [], {}, None

    def add(label, lat, arr, extra=None):
        nonlocal ref
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / (ref.norm() + 1e-12))
        rec = {"relL2_vs_fp16": rel}
        if extra:
            rec.update(extra)
        quality[label] = rec
        tag = label if rel == 0.0 else f"{label}    relL2 {rel:.4f}"
        print(f"  {label:12s} relL2 {rel:.4f}"
              + (f"  sat={extra['max_sat']:.3f} n_over={extra['n_over']}/{extra['n_int']}"
                 if extra and extra.get("n_int") else ""),
              flush=True)
        rows.append((tag, arr))
        save_pngs(arr, os.path.join(OUT_DIR, label.replace(" ", "_")))

    print("===== fp16 =====", flush=True)
    _knobs(False, 16, False)
    model, sampler = runner._setup_model("fp16")
    sample_lat(runner, model, sampler, a.n, a.seed, a.steps, quantized=False)
    lat = sample_lat(runner, model, sampler, a.n, a.seed, a.steps, quantized=False)
    add("fp16", lat, decode(model, lat))
    del model, sampler
    torch.cuda.empty_cache()

    print("===== W8A8 I-MoDiff =====", flush=True)
    runner.calibration_path = CALIB8
    runner.auto_delta_table = True
    model, sampler = runner._setup_model("int8")
    for name, imode, bits, freeze in ARMS:
        print(f"===== {name} imode={imode} bits={bits} freeze={freeze} =====", flush=True)
        _knobs(imode, bits, freeze)
        sample_lat(runner, model, sampler, a.n, a.seed, a.steps, quantized=True)
        lat = sample_lat(runner, model, sampler, a.n, a.seed, a.steps, quantized=True)
        ov = _overflow(model)
        add(name, lat, decode(model, lat), ov)
    del model, sampler
    torch.cuda.empty_cache()

    cell, pad, lab = a.cell, 6, 26
    W = pad + a.n * (cell + pad)
    Hh = len(rows) * (cell + lab + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for label, arr in rows:
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += lab
        for i in range(min(a.n, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    canvas.save(OUT_PNG, "PNG")
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    json.dump({"seed": a.seed, "steps": a.steps, "n": a.n, "relL2": quality,
               "out": OUT_PNG, "pngs": OUT_DIR}, open(JSON_OUT, "w"), indent=1)
    print(f"\nwrote {OUT_PNG}  ({W}x{Hh})\nwrote {JSON_OUT}")
    _knobs(False, 16, False)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _knobs(False, 16, False)
