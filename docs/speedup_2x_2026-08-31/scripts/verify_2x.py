"""Verify the ≥2× arms: leftover-quantized INT4 PTQ (calibrated skip/out) vs fp16.

Times batch 128 / DDIM 50. Quality: n=4, seed 20260805, relL2 + decode grid.
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
os.environ["MODIFF_REPLAY_K"] = "1"
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
from integration.kernels.int4_optimized import OptimizedInt4Conv2d
from integration.kernels.int8_cudagraph import (
    install_cuda_graph_replay_pytorch_int8, get_cuda_graph_replay_stats)

BATCH, STEPS, NQ, SEED = 128, 50, 4, 20260805
SHAPE = (4, 32, 32)
OUT_JSON = "docs/speedup_2x_2026-08-31/data/verify_2x.json"
OUT_PNG = "docs/speedup_2x_2026-08-31/plots/verify_2x_grid.png"


def time_n(model, sampler, n=2, warm=1, quantized=True):
    def once():
        mgr = getattr(model.model.diffusion_model, "_cuda_graph_manager", None)
        if mgr is not None:
            mgr.reset_sequence()
        if quantized:
            B.reset_modiff_state_int4(model.model.diffusion_model)
            B._reset_wxax_modiff_safe(model)
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


def setup(mode, skip_out=False):
    os.environ["MODIFF_QUANT_SKIP_OUT"] = "1" if skip_out else "0"
    os.environ["MODIFF_REPLAY_K"] = "1"
    runner = B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/speedup_2x_2026-08-31/tmp",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def leftover_modules(model):
    out = []
    for m in model.model.diffusion_model.modules():
        if isinstance(m, OptimizedInt4Conv2d) and (
                "skip" in (m.layer_name or "") or (m.layer_name or "").startswith("out.")):
            out.append(m)
    return out


def calib_leftover(model, sampler):
    layers = leftover_modules(model)
    uncal = [m for m in layers if not m.is_calibrated]
    print(f"  leftover {len(layers)}  uncalibrated {len(uncal)}", flush=True)
    for m in uncal:
        m.begin_calibration()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for _ in range(3):
            B.reset_modiff_state_int4(model.model.diffusion_model)
            B._reset_wxax_modiff_safe(model)
            sampler.sample(S=5, batch_size=8, shape=SHAPE, eta=0.0, verbose=False)
    for m in uncal:
        m.end_calibration()
    still = [m.layer_name for m in leftover_modules(model) if not m.is_calibrated]
    print(f"  leftover calibrated; still-open {still}", flush=True)
    return [m.layer_name for m in leftover_modules(model)]


def gen_lat(model, sampler, n, seed, quantized=True):
    if quantized:
        B.reset_modiff_state_int4(model.model.diffusion_model)
        B._reset_wxax_modiff_safe(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    rows, quality, images = [], {}, []
    ref = None

    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16")
    ms, trials = time_n(model, sampler, quantized=False)
    rows.append({"arm": "fp16", "ms_step": ms, "trials": trials, "vs_fp16": 1.0})
    print(f"  {ms:.2f}  {trials}", flush=True)
    lat = gen_lat(model, sampler, NQ, SEED, quantized=False)
    ref = lat.float().clone()
    quality["fp16"] = {"relL2_vs_fp16": 0.0}
    images.append(("fp16", G.decode(model, lat)))
    del model, sampler
    torch.cuda.empty_cache()
    fp16 = ms
    target = fp16 / 2

    print("===== int4_baseline (shipped, no leftover) =====", flush=True)
    model, sampler = setup("int4_baseline", skip_out=False)
    ms, trials = time_n(model, sampler)
    rows.append({"arm": "int4_baseline", "ms_step": ms, "trials": trials, "vs_fp16": fp16 / ms})
    print(f"  {ms:.2f}  {fp16/ms:.3f}x  {trials}", flush=True)
    lat = gen_lat(model, sampler, NQ, SEED)
    rel = float((lat.float() - ref).norm() / ref.norm())
    quality["int4_baseline"] = {"relL2_vs_fp16": rel}
    images.append((f"W4A4 PTQ  relL2 {rel:.4f}", G.decode(model, lat)))
    print(f"  relL2 {rel:.4f}", flush=True)
    del model, sampler
    torch.cuda.empty_cache()

    print("===== int4_baseline leftover skip+out, live calib =====", flush=True)
    model, sampler = setup("int4_baseline", skip_out=True)
    names = calib_leftover(model, sampler)
    ms, trials = time_n(model, sampler)
    rows.append({"arm": "int4_baseline_skip_out_calib", "ms_step": ms, "trials": trials,
                 "vs_fp16": fp16 / ms, "leftover": names})
    print(f"  {ms:.2f}  {fp16/ms:.3f}x  {trials}", flush=True)
    lat = gen_lat(model, sampler, NQ, SEED)
    rel = float((lat.float() - ref).norm() / ref.norm())
    quality["int4_baseline_skip_out_calib"] = {"relL2_vs_fp16": rel}
    images.append((f"W4A4 PTQ+skip/out  relL2 {rel:.4f}", G.decode(model, lat)))
    print(f"  relL2 {rel:.4f}", flush=True)

    print("===== leftover PTQ + CUDA graph =====", flush=True)
    try:
        install_cuda_graph_replay_pytorch_int8(model.model, batch_size=BATCH, shape=SHAPE)
        time_n(model, sampler, n=1, warm=0)
        stats = get_cuda_graph_replay_stats(model.model.diffusion_model)
        print("  capture", stats, flush=True)
        if stats and not stats.get("disabled"):
            ms, trials = time_n(model, sampler, n=2, warm=0)
            rows.append({"arm": "int4_baseline_skip_out_calib_graph", "ms_step": ms,
                         "trials": trials, "vs_fp16": fp16 / ms, "stats": stats})
            print(f"  graph {ms:.2f}  {fp16/ms:.3f}x  {trials}", flush=True)
        else:
            rows.append({"arm": "int4_baseline_skip_out_calib_graph", "error": "disabled",
                         "stats": stats})
    except Exception as e:
        print("  graph failed", e, flush=True)
        rows.append({"arm": "int4_baseline_skip_out_calib_graph", "error": str(e)})
    del model, sampler
    torch.cuda.empty_cache()

    cell, pad, lab = 256, 6, 28
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

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "fp16_ms_step": fp16,
        "target_ms": target,
        "arms": rows,
        "quality": quality,
        "grid": OUT_PNG,
        "hit_2x": any(r.get("vs_fp16", 0) >= 2.0 and "ms_step" in r
                      and r["arm"] != "fp16" for r in rows),
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, OUT_PNG, flush=True)
    for r in rows:
        if "ms_step" in r:
            print(f"  {r['arm']:40s} {r['ms_step']:7.2f}  {r['vs_fp16']:.3f}x", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
