"""Conv-input quantizer granularity: per-tensor vs along-C blockwise B=32.

The quantizer under test is the one feeding the int8 conv -- Q(a_t) in the baseline
arm, Q(a_t - a_hat_{t+1}) in MoDiff. It is per-tensor today because its scale is the
CUTLASS epilogue's scalar alpha, and a blockwise scale along C is a scale along the
conv's reduction axis (see _forward_blockwise_sim). This prices the accuracy that a
real blockwise mainloop would have to buy back in speed.

Every arm runs through the SAME simulation forward (fake-quant + fp32 conv on
dequantized W8 weights), so quantizer granularity is the only variable:

  static   per-tensor, calibrated scale / per-step delta table   (what ships)
  dyn      per-tensor, dynamic absmax
  B=32     dynamic, 32 consecutive channels per pixel

`real` is the shipped fused int8 kernel path, included to show how far the sim
harness sits from the thing it models.

relL2 is against the fp16 model, in latent space, same seed.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/act_blockwise_2026-09-01/scripts/act_block32_quality.py
"""
from __future__ import annotations

import gc
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_AHAT_BLOCK"] = "0"
os.environ["MODIFF_ACT_BLOCK"] = "0"
# Import-time kill switches: every conv must reach OptimizedInt8Conv2d.forward, or the
# sim silently measures the real per-tensor kernels instead (_sim_guard catches it).
for _k in ("MODIFF_DISABLE_GN_MODIFF_FUSION", "MODIFF_DISABLE_GN_INT8_FUSION",
           "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION",
           "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION",
           "MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION"):
    os.environ[_k] = "1"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="act_block32_quality.py")

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
import ahat_fake_quant_grid as G  # noqa: E402

SHAPE = (4, 32, 32)
STEPS, NQ, SEED = 50, 6, 20260805
OUT_JSON = "docs/act_blockwise_2026-09-01/data/act_block32_quality.json"
OUT_PNG = "docs/act_blockwise_2026-09-01/plots/act_block32_quality.png"

# (MODIFF_ACT_BLOCK value, short label)
ARMS = [(0, "real int8 kernels (per-tensor)"),
        (-2, "sim  per-tensor static"),
        (-1, "sim  per-tensor dynamic"),
        (32, "sim  along-C B=32")]


def build(mode):
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/act_blockwise_2026-09-01/tmp",
        batch_size=NQ, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)
    return runner._setup_model(mode)


def gen(model, sampler):
    try:
        B.reset_modiff_state_int8(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)
    except Exception:
        pass
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=NQ, shape=SHAPE, eta=0.0, verbose=False)
        lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  n={NQ} steps={STEPS}", flush=True)
    recs, images = [], []

    model, sampler = build("fp16")
    ref = gen(model, sampler)
    images.append(("fp16", "reference", G.decode(model, ref)))
    print("fp16 reference done", flush=True)
    del model, sampler
    gc.collect()
    torch.cuda.empty_cache()

    for mode, arm_name in [("int8_baseline", "baseline"), ("int8", "MoDiff")]:
        model, sampler = build(mode)
        for blk, label in ARMS:
            os.environ["MODIFF_ACT_BLOCK"] = str(blk)
            lat = gen(model, sampler)
            rel = float((lat - ref).norm() / ref.norm())
            recs.append({"arm": arm_name, "act_block": blk, "label": label,
                         "relL2_vs_fp16": rel})
            print(f"  {arm_name:9s} {label:32s} relL2 {rel:.4f}", flush=True)
            images.append((f"{arm_name}  {label}", f"relL2 {rel:.4f}",
                           G.decode(model, lat)))
        os.environ["MODIFF_ACT_BLOCK"] = "0"
        del model, sampler
        gc.collect()
        torch.cuda.empty_cache()

    print("\n===== relL2 vs fp16 (lower is better) =====", flush=True)
    for arm in ("baseline", "MoDiff"):
        rs = [r for r in recs if r["arm"] == arm]
        base = next(r["relL2_vs_fp16"] for r in rs if r["act_block"] == -1)
        for r in rs:
            d = "" if r["act_block"] != 32 else \
                f"   {100.0 * (base - r['relL2_vs_fp16']) / base:+.1f}% vs per-tensor dyn"
            print(f"  {arm:9s} {r['label']:32s} {r['relL2_vs_fp16']:.4f}{d}", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "n": NQ, "steps": STEPS,
               "seed": SEED, "arms": recs}, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}", flush=True)

    cell, pad, lab = 224, 6, 34
    W = pad + NQ * (cell + pad)
    Hh = len(images) * (cell + lab + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    for title, sub, arr in images:
        dr.text((pad, y + 4), f"{title}    {sub}", fill=(11, 11, 11))
        y += lab
        for i in range(min(NQ, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
        y += cell + pad
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    canvas.save(OUT_PNG, "PNG")
    print(f"wrote {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
