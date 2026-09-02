"""Real along-C B=32 a_hat: int8 cache + in-kernel dynamic per-block scales.

Compares W8A8 fp16 a_hat vs MODIFF_AHAT_BLOCK=32 (int8 storage, ahat_commit_block).

Timing: batch 128, 50 DDIM, CUDA events, median of 2 after 1 warmup.
Quality: n=6, seed 20260805, relL2 vs W8A8 fp16 a_hat.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/ahat_blockwise_2026-09-01/scripts/block32_real.py
"""
from __future__ import annotations

import json
import os
import statistics
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

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="block32_real.py")

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
import ahat_fake_quant_grid as G  # noqa: E402

SHAPE = (4, 32, 32)
BATCH, STEPS, NQ, SEED = 128, 50, 6, 20260805
OUT_JSON = "docs/ahat_blockwise_2026-09-01/data/block32_real.json"
OUT_PNG = "docs/ahat_blockwise_2026-09-01/plots/block32_real.png"


def set_block(b: int) -> None:
    os.environ["MODIFF_AHAT_BLOCK"] = str(b)
    os.environ["MODIFF_AHAT_BITS"] = "16"
    os.environ["MODIFF_IMODE"] = "0"
    os.environ["MODIFF_AHAT_REFRESH"] = "0"


def reset(model) -> None:
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def time_n(model, sampler, n=2, warm=1):
    def once():
        reset(model)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)

    for _ in range(warm):
        once()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        once()
        e.record()
        torch.cuda.synchronize()
        xs.append(s.elapsed_time(e) / STEPS)
    return statistics.median(xs), xs


def gen_lat(model, sampler, n, seed):
    reset(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
        lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={BATCH} steps={STEPS}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_blockwise_2026-09-01/tmp_block32_real",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)

    set_block(0)
    model, sampler = runner._setup_model("int8")

    recs = []
    images = []
    ref = None

    for block, label in [(0, "W8A8  a_hat fp16"), (32, "W8A8  a_hat along-C B=32 int8")]:
        set_block(block)
        print(f"===== {label} =====", flush=True)
        ms, trials = time_n(model, sampler)
        lat = gen_lat(model, sampler, NQ, SEED)
        if ref is None:
            ref = lat.float().clone()
            rel = 0.0
        else:
            rel = float((lat.float() - ref).norm() / ref.norm())
        recs.append({
            "label": label, "ahat_block": block, "ms_step": ms, "trials": trials,
            "relL2_vs_fp16_ahat": rel,
        })
        print(f"  {ms:.3f} ms/step  trials={['%.3f' % t for t in trials]}  relL2 {rel:.4f}",
              flush=True)
        images.append((label, f"{ms:.2f} ms/step   relL2 {rel:.4f}", G.decode(model, lat)))

    ref_ms = recs[0]["ms_step"]
    for r in recs:
        r["speedup_vs_fp16_ahat"] = ref_ms / r["ms_step"]
        r["pct_vs_fp16_ahat"] = 100.0 * (ref_ms - r["ms_step"]) / ref_ms

    print("\n===== vs W8A8 fp16 a_hat =====", flush=True)
    for r in recs:
        print(f"  {r['label']:36s} {r['ms_step']:7.3f}  {r['speedup_vs_fp16_ahat']:.3f}x  "
              f"({r['pct_vs_fp16_ahat']:+.2f}%)  relL2 {r['relL2_vs_fp16_ahat']:.4f}",
              flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": BATCH, "steps": STEPS,
               "seed": SEED, "n_quality": NQ, "arms": recs}, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}", flush=True)

    cell, pad, lab = 256, 6, 40
    W = pad + min(NQ, images[0][2].shape[0]) * (cell + pad)
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
