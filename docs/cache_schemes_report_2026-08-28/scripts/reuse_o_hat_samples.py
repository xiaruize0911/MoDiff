"""Generate a few replay samples after wiring reuse_o_hat_add into _replay_out.

Same seeds as fid_cache_schemes.py (seed0=20260805). W8A8, 50 DDIM, batch=n.
Reuse existing fp16 pngs for the contact sheet.

Run: source setup_cuda_env.sh && python docs/cache_schemes_report_2026-08-28/scripts/reuse_o_hat_samples.py
"""
import os
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_REPLAY_BLOCK"] = "1"

from integration.utils.preflight import preflight, MODEL
preflight(*MODEL, what="reuse_o_hat_samples.py")

import torch
from PIL import Image, ImageDraw, ImageFont
import integration.benchmarks.benchmark_ldm as B
import modiff_cutlass as mc

assert hasattr(mc, "reuse_o_hat_add"), "rebuild modiff_cutlass first"

N, STEPS, SEED0 = 6, 50, 20260805
SHAPE = (4, 32, 32)
CALIB8 = "integration/calibration/int8_calibration_realckpt.pt"
OUT = "docs/cache_schemes_report_2026-08-28/plots/reuse_o_hat_samples"
FP16 = "docs/cache_schemes_report_2026-08-28/fid_samples/fp16"


def reset(model, quantized):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def sample(runner, model, sampler, n, seed, quantized):
    reset(model, quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
        lat = lat.to("cuda", torch.float16)
        d = model.decode_first_stage(lat)
        img = torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0).permute(0, 2, 3, 1).cpu()
    return (img.numpy() * 255).round().astype("uint8")


def save_arm(arr, folder):
    os.makedirs(folder, exist_ok=True)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(os.path.join(folder, f"{i:06d}.png"))


def sheet(folders, labels, path):
    idxs = list(range(N))
    cell, pad, label_w, header_h = 128, 6, 120, 22
    W = label_w + N * (cell + pad) + pad
    H = header_h + len(folders) * (cell + pad) + pad
    img = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 3), "same seed  ·  reuse_o_hat_add in pipeline", fill=(200, 200, 200), font=font)
    for j in idxs:
        draw.text((label_w + j * (cell + pad) + pad, 3), f"{j:06d}",
                  fill=(160, 160, 160), font=font)
    for i, (folder, label) in enumerate(zip(folders, labels)):
        y = header_h + i * (cell + pad) + pad
        draw.text((8, y + cell // 2 - 8), label, fill=(240, 240, 240), font=font)
        for j in idxs:
            p = os.path.join(folder, f"{j:06d}.png")
            im = Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS)
            img.paste(im, (label_w + j * (cell + pad) + pad, y))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print("wrote", path, img.size, flush=True)


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  n={N} steps={STEPS}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(OUT, "_tmp"),
        batch_size=N, steps=STEPS, shape=SHAPE,
        calibration_path=CALIB8, auto_delta_table=True)
    print("===== fp16 =====", flush=True)
    runner.calibration_path = None
    os.environ["MODIFF_REPLAY_K"] = "1"
    model, sampler = runner._setup_model("fp16")
    save_arm(sample(runner, model, sampler, N, SEED0, quantized=False),
             os.path.join(OUT, "fp16"))
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int8 MoDiff =====", flush=True)
    runner.calibration_path = CALIB8
    model, sampler = runner._setup_model("int8")
    print("warmup", flush=True)
    sample(runner, model, sampler, min(N, 4), SEED0 - 1, quantized=True)
    for k, name in ((2, "replay2"), (4, "replay4")):
        os.environ["MODIFF_REPLAY_K"] = str(k)
        print(f"===== {name} K={k} =====", flush=True)
        save_arm(sample(runner, model, sampler, N, SEED0, quantized=True),
                 os.path.join(OUT, name))

    sheet(
        [os.path.join(OUT, "fp16"), os.path.join(OUT, "replay2"), os.path.join(OUT, "replay4")],
        ["fp16", "K=2 kernel", "K=4 kernel"],
        os.path.join(OUT, "..", "fig_reuse_o_hat_samples.png"),
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        os.environ["MODIFF_REPLAY_K"] = "1"
