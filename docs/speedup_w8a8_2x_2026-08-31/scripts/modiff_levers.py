"""W8A8 MoDiff (equations + o_hat replay), not attention-residual skip.

Honest reset per sample. Quality n=4 vs fp16.
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
os.environ["MODIFF_ATTN_REPLAY_K"] = "1"
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
from integration.kernels.int8_cudagraph import (
    install_cuda_graph_replay_pytorch_int8, get_cuda_graph_replay_stats)

BATCH, STEPS, NQ, SEED = 128, 50, 4, 20260805
SHAPE = (4, 32, 32)
OUT_JSON = "docs/speedup_w8a8_2x_2026-08-31/data/modiff_levers.json"
OUT_PNG = "docs/speedup_w8a8_2x_2026-08-31/plots/modiff_quality.png"


def reset_all(model, quantized=True, int4=False):
    mgr = getattr(model.model.diffusion_model, "_cuda_graph_manager", None)
    if mgr is not None:
        mgr.reset_sequence()
    if quantized:
        if int4:
            B.reset_modiff_state_int4(model.model.diffusion_model)
        else:
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


def setup(mode, replay_k=1, warmup=5):
    os.environ["MODIFF_REPLAY_K"] = str(replay_k)
    os.environ["MODIFF_WARMUP_STEPS"] = str(warmup)
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
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    rows, quality, images = [], {}, []
    ref = None
    fp16 = None
    model = sampler = None

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

    specs = [
        ("fp16", "fp16", 1, 5, False),
        ("int8_ptq", "int8_baseline", 1, 5, False),
        ("int8_modiff_k1", "int8", 1, 5, False),
        ("int8_modiff_k1_w1", "int8", 1, 1, False),
        ("int8_modiff_replay2", "int8", 2, 5, False),
        ("int8_modiff_replay4", "int8", 4, 5, False),
        ("int8_modiff_replay2_graph", "int8", 2, 5, True),
    ]

    for arm, mode, rk, wu, graph in specs:
        print(f"===== {arm} =====", flush=True)
        if model is not None:
            del model, sampler
            torch.cuda.empty_cache()
        model, sampler = setup(mode, replay_k=rk, warmup=wu)
        q = mode != "fp16"
        if graph:
            install_cuda_graph_replay_pytorch_int8(model.model, batch_size=BATCH, shape=SHAPE)
            time_n(model, sampler, n=1, warm=0, quantized=q)
            stats = get_cuda_graph_replay_stats(model.model.diffusion_model)
            print("  capture", stats, flush=True)
            if stats and stats.get("disabled"):
                record(arm, float("nan"), [], extra={"error": "graph disabled", "stats": stats})
                continue
            ms, trials = time_n(model, sampler, n=2, warm=0, quantized=q)
            lat = gen_lat(model, sampler, NQ, SEED, quantized=q)
            extra = {"replay_k": rk, "warmup": wu, "stats": stats}
        else:
            ms, trials = time_n(model, sampler, quantized=q)
            lat = gen_lat(model, sampler, NQ, SEED, quantized=q)
            extra = {"replay_k": rk, "warmup": wu}
        if fp16 is None:
            fp16 = ms
            ref = lat.float().clone()
        record(arm, ms, trials, lat, extra=extra)

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
    json.dump({"gpu": torch.cuda.get_device_name(0), "fp16_ms_step": fp16,
               "target_ms": fp16 / 2, "arms": rows, "quality": quality,
               "grid": OUT_PNG}, open(OUT_JSON, "w"), indent=2)
    print("wrote", OUT_JSON, OUT_PNG, flush=True)
    for r in rows:
        if "ms_step" in r:
            qv = quality.get(r["arm"], {})
            print(f"  {r['arm']:28s} {r['ms_step']:7.2f}  {r.get('vs_fp16', 0):.3f}x  "
                  f"relL2 {qv.get('relL2_vs_fp16', float('nan')):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
