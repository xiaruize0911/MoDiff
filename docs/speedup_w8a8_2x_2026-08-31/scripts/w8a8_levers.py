"""W8A8 PTQ ≥2× vs fp16: attention residual replay, CUDA graph, leftover skip/out.

Quality: n=4 sample grid, images must stay clearly recognizable churches.
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
os.environ["MODIFF_ATTN_REPLAY_K"] = "1"
os.environ["MODIFF_ATTN_REPLAY_HD24"] = "0"

import torch
from PIL import Image, ImageDraw
import integration.benchmarks.benchmark_ldm as B
import ahat_fake_quant_grid as G
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int8_cudagraph import (
    install_cuda_graph_replay_pytorch_int8, get_cuda_graph_replay_stats)
from integration.fused_ops.quantized_std_attention import reset_attn_replay

BATCH, STEPS, NQ, SEED = 128, 50, 4, 20260805
SHAPE = (4, 32, 32)
OUT_JSON = "docs/speedup_w8a8_2x_2026-08-31/data/w8a8_levers.json"
OUT_PNG = "docs/speedup_w8a8_2x_2026-08-31/plots/w8a8_quality.png"


def reset_all(model, quantized=True):
    mgr = getattr(model.model.diffusion_model, "_cuda_graph_manager", None)
    if mgr is not None:
        mgr.reset_sequence()
    reset_attn_replay(model.model.diffusion_model)
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
        output_dir="docs/speedup_w8a8_2x_2026-08-31/tmp",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path(mode),
        linear_backend="int_gemm", auto_delta_table=True)
    return runner._setup_model(mode)


def calib_leftover(model, sampler):
    leftover = [m for m in model.model.diffusion_model.modules()
                if isinstance(m, OptimizedInt8Conv2d) and not m.is_calibrated]
    print(f"  leftover uncalibrated {len(leftover)}", flush=True)
    for m in leftover:
        m.begin_calibration()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for _ in range(3):
            reset_all(model)
            sampler.sample(S=5, batch_size=8, shape=SHAPE, eta=0.0, verbose=False)
    for m in leftover:
        m.end_calibration()
    still = sum(1 for m in leftover if not m.is_calibrated)
    print(f"  leftover still-open {still}", flush=True)


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
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    rows, quality, images = [], {}, []
    ref = None
    fp16 = None

    def record(arm, ms, trials, lat=None, extra=None):
        rec = {"arm": arm, "ms_step": ms, "trials": trials,
               "vs_fp16": (fp16 / ms) if fp16 else 1.0}
        if extra:
            rec.update(extra)
        rows.append(rec)
        print(f"  {ms:.2f}  {rec['vs_fp16']:.3f}x  {trials}", flush=True)
        if lat is not None:
            rel = 0.0 if ref is None else float((lat.float() - ref).norm() / ref.norm())
            quality[arm] = {"relL2_vs_fp16": rel}
            images.append((f"{arm}  relL2 {rel:.4f}", G.decode(model, lat)))
            print(f"  relL2 {rel:.4f}", flush=True)

    print("===== fp16 =====", flush=True)
    model, sampler = setup("fp16")
    ms, trials = time_n(model, sampler, quantized=False)
    fp16 = ms
    lat = gen_lat(model, sampler, NQ, SEED, quantized=False)
    ref = lat.float().clone()
    record("fp16", ms, trials, lat)
    del model, sampler
    torch.cuda.empty_cache()
    target = fp16 / 2
    print(f"target 2.00x = {target:.2f} ms/step", flush=True)

    print("===== int8_baseline PTQ =====", flush=True)
    os.environ["MODIFF_ATTN_REPLAY_K"] = "1"
    os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"
    model, sampler = setup("int8_baseline")
    ms, trials = time_n(model, sampler)
    lat = gen_lat(model, sampler, NQ, SEED)
    record("int8_baseline", ms, trials, lat)

    print("===== int8_baseline + CUDA graph =====", flush=True)
    try:
        install_cuda_graph_replay_pytorch_int8(model.model, batch_size=BATCH, shape=SHAPE)
        time_n(model, sampler, n=1, warm=0)
        stats = get_cuda_graph_replay_stats(model.model.diffusion_model)
        print("  capture", stats, flush=True)
        if stats and not stats.get("disabled"):
            ms, trials = time_n(model, sampler, n=2, warm=0)
            record("int8_baseline_graph", ms, trials, extra={"stats": stats})
        else:
            rows.append({"arm": "int8_baseline_graph", "error": "disabled", "stats": stats})
    except Exception as e:
        print("  graph failed", e, flush=True)
        rows.append({"arm": "int8_baseline_graph", "error": str(e)})
    del model, sampler
    torch.cuda.empty_cache()

    for k, hd24 in ((2, "0"), (3, "0"), (4, "0"), (2, "1"), (3, "1")):
        tag = f"int8_attn_k{k}" + ("_hd24" if hd24 == "1" else "")
        print(f"===== {tag} =====", flush=True)
        os.environ["MODIFF_ATTN_REPLAY_K"] = str(k)
        os.environ["MODIFF_ATTN_REPLAY_HD24"] = hd24
        os.environ["MODIFF_QUANT_SKIP_OUT"] = "0"
        model, sampler = setup("int8_baseline")
        ms, trials = time_n(model, sampler)
        lat = gen_lat(model, sampler, NQ, SEED)
        record(tag, ms, trials, lat, extra={"attn_k": k, "hd24_only": hd24 == "1"})
        del model, sampler
        torch.cuda.empty_cache()
    os.environ["MODIFF_ATTN_REPLAY_K"] = "1"
    os.environ["MODIFF_ATTN_REPLAY_HD24"] = "0"

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
        "hit_2x": any(r.get("vs_fp16", 0) >= 2.0 and "ms_step" in r and r["arm"] != "fp16"
                      for r in rows),
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, OUT_PNG, flush=True)
    for r in rows:
        if "ms_step" in r:
            q = quality.get(r["arm"], {})
            print(f"  {r['arm']:28s} {r['ms_step']:7.2f}  {r['vs_fp16']:.3f}x  "
                  f"relL2 {q.get('relL2_vs_fp16', float('nan')):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
