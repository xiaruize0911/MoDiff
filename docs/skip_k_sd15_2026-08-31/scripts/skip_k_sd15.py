"""Skip-K on Stable Diffusion 1.5.

Same protocol as docs/skip_k_cadence_2026-08-31: freeze a_hat/o_hat writes on
K-1 of every K steps; GN+quantize+conv still run. Attention is SpatialTransformer
(QKV/proj = Linear + fp16 matmul/SDPA). MODIFF_QUANT_ATTN=0 so we do not convert
OpenAI AttentionBlocks (SD1.5 has none).

K in {1, 2, 5, 10}. No FID.

Run: source setup_cuda_env.sh
     python docs/skip_k_sd15_2026-08-31/scripts/skip_k_sd15.py --stage samples,bench,layer
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ.setdefault("MODIFF_DELTA_MODE", "dynamic")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_QUANT_ATTN"] = "0"
os.environ["MODIFF_QUANT_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ.setdefault("MODIFF_SD_PROMPT", "a photograph of a church on a hill")

from integration.utils.preflight import preflight, MODEL  # noqa: E402

KS = (1, 2, 5, 10, 20, 50)
SHAPE = (4, 64, 64)
CONFIG = "configs/stable-diffusion/v1-inference.yaml"
CKPT = "models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt"
CALIB8 = "integration/calibration/int8_calibration_sd15.pt"
OUT_DIR = "docs/skip_k_sd15_2026-08-31"
SEED_Q = 20260805
SEED_T = 20260827

_STATS = {"write": 0, "skip": 0}
_ORIG_WRITE = None


def _apply(k):
    os.environ["MODIFF_CACHE_SKIP_K"] = str(k)
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_AHAT_BITS"] = "16"
    os.environ["MODIFF_AHAT_REFRESH"] = "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"
    os.environ["MODIFF_LINEAR"] = "0"


def _install_write_counter():
    global _ORIG_WRITE
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    if _ORIG_WRITE is None:
        _ORIG_WRITE = OptimizedInt8Conv2d._write_ahat_now

    def wrapped(self):
        w = _ORIG_WRITE(self)
        _STATS["write" if w else "skip"] += 1
        return w

    OptimizedInt8Conv2d._write_ahat_now = wrapped


def _reset_stats():
    _STATS["write"] = _STATS["skip"] = 0


def _reset(model, quantized):
    import integration.benchmarks.benchmark_ldm as B
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def decode(model, lat, chunk=2):
    import torch
    lat = lat.to("cuda", torch.float16)
    out = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for i in range(0, lat.shape[0], chunk):
            d = model.decode_first_stage(lat[i:i + chunk])
            out.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0).permute(0, 2, 3, 1).cpu())
            del d
    return (torch.cat(out, 0).numpy() * 255).round().astype("uint8")


def _runner(batch, steps, calib):
    import integration.benchmarks.benchmark_ldm as B
    r = B.BenchmarkRunner(
        config_path=CONFIG, ckpt_path=CKPT,
        output_dir=os.path.join(OUT_DIR, "tmp"),
        batch_size=batch, steps=steps, shape=SHAPE,
        calibration_path=calib if (calib and os.path.exists(calib)) else None,
        auto_delta_table=False)
    r.prompt = os.environ.get("MODIFF_SD_PROMPT", "")
    return r


def ensure_calib(runner, model, sampler):
    """Live-calibrate SD1.5 conv/linear scales if the file is missing."""
    if os.path.exists(CALIB8):
        return
    runner.calibration_path = CALIB8
    print(f"===== live INT8 calibration -> {CALIB8} =====", flush=True)
    runner._calibrate_int8(model, sampler, num_runs=2, calib_steps=20, calib_batch=2)


def sample_lat(runner, model, sampler, n, seed, steps, quantized):
    import torch
    _reset(model, quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.float()


def stage_samples(a):
    import torch
    from PIL import Image, ImageDraw
    import integration.benchmarks.benchmark_ldm as B

    _install_write_counter()
    print(f"GPU: {torch.cuda.get_device_name()}  n={a.n} steps={a.steps}", flush=True)
    runner = _runner(a.n, a.steps, None)
    rows, quality, ref = [], {}, None

    def add_arm(label, lat, extra=None):
        nonlocal ref
        if ref is None:
            ref = lat.detach().float().cpu().clone()
            rel = 0.0
        else:
            x = lat.detach().float().cpu()
            rel = float((x - ref).norm() / ref.norm())
        rec = {"relL2_vs_fp16": rel}
        if extra:
            rec.update(extra)
        quality[label] = rec
        extra_s = ""
        if extra:
            extra_s = (f"  write {extra.get('n_write', 0)} skip {extra.get('n_skip', 0)}"
                       f"  skip_frac {extra.get('skip_frac', 0):.3f}")
        print(f"  {label:28s} relL2 {rel:.4f}{extra_s}", flush=True)
        rows.append((f"{label}    relL2 {rel:.4f}" if rel else label, decode(model, lat)))

    print("===== fp16 =====", flush=True)
    _apply(1)
    model, sampler = runner._setup_model("fp16")
    sample_lat(runner, model, sampler, a.n, a.seed, a.steps, False)
    lat = sample_lat(runner, model, sampler, a.n, a.seed, a.steps, False)
    add_arm("fp16 reference", lat)
    del model, sampler
    torch.cuda.empty_cache()

    print("===== W8A8 MoDiff skip-K (SD1.5) =====", flush=True)
    _apply(1)
    model, sampler = runner._setup_model("int8")
    # skip live calib: SD1.5 SmoothQuant produced NaN scales on C=320 in_conv
    # ensure_calib(runner, model, sampler)
    if os.path.exists(CALIB8) and not (runner.calibration_path and os.path.exists(runner.calibration_path)):
        from integration.kernels.int8_optimized import apply_static_scales, get_calibration_config_int8
        scales = torch.load(CALIB8, weights_only=True)
        apply_static_scales(model.model.diffusion_model, scales)
        print(f"✓ Applied {CALIB8} after live calib", flush=True)
    # If we calibrated in this process, scales are already on the modules.
    for k in KS:
        _apply(k)
        _reset_stats()
        sample_lat(runner, model, sampler, a.n, a.seed, a.steps, True)
        _reset_stats()
        lat = sample_lat(runner, model, sampler, a.n, a.seed, a.steps, True)
        n_w, n_s = _STATS["write"], _STATS["skip"]
        extra = {"skip_k": k, "n_write": n_w, "n_skip": n_s,
                 "skip_frac": n_s / max(n_w + n_s, 1)}
        add_arm(f"W8A8 skip-K={k}", lat, extra)
    _apply(1)
    del model, sampler
    torch.cuda.empty_cache()

    cell, pad, lab = a.cell, 6, 26
    W = pad + a.n * (cell + pad)
    Hh = len(rows) * (cell + lab + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    png_dir = os.path.join(OUT_DIR, "samples")
    os.makedirs(png_dir, exist_ok=True)
    for ridx, (label, arr) in enumerate(rows):
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += lab
        for i in range(min(a.n, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
            im.save(os.path.join(png_dir, f"row{ridx:02d}_{i:02d}.png"))
        y += cell + pad
    grid = os.path.join(OUT_DIR, "plots", "skip_k_sd15_grid.png")
    os.makedirs(os.path.dirname(grid), exist_ok=True)
    canvas.save(grid, "PNG")
    payload = {
        "model": "sd1.5", "prompt": os.environ.get("MODIFF_SD_PROMPT", ""),
        "seed": a.seed, "steps": a.steps, "n": a.n, "K": list(KS),
        "gpu": torch.cuda.get_device_name(0), "relL2": quality, "grid": grid,
        "shape": list(SHAPE), "delta_mode": os.environ.get("MODIFF_DELTA_MODE"),
        "quant_attn": os.environ.get("MODIFF_QUANT_ATTN"),
    }
    path = os.path.join(OUT_DIR, "data", "skip_k_sd15_quality.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {grid}  ({W}x{Hh})\nwrote {path}", flush=True)
    return payload


def stage_bench(a):
    import torch
    import integration.benchmarks.benchmark_ldm as B

    batch, steps = a.bench_batch, a.steps
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={batch} steps={steps}", flush=True)
    runner = _runner(batch, steps, None)
    _apply(1)
    model, sampler = runner._setup_model("int8")
    # ensure_calib skipped: SD1.5 C=320 in_conv SmoothQuant NaN scales

    recs = []
    for k in KS:
        _apply(k)

        def once():
            _reset(model, True)
            torch.manual_seed(SEED_T)
            torch.cuda.manual_seed_all(SEED_T)
            cond = runner._cond_kwargs(model, batch)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                sampler.sample(S=steps, batch_size=batch, shape=SHAPE, eta=0.0,
                               verbose=False, **cond)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0 / steps

        print(f"warmup skip-K={k}", flush=True)
        once()
        trials = [once() for _ in range(a.bench_trials)]
        ms = statistics.median(trials)
        recs.append({"skip_k": k, "ms_step": ms, "trials": trials})
        print(f"  skip-K={k:2d}  {ms:.3f} ms/step  trials={[round(t, 3) for t in trials]}",
              flush=True)
    _apply(1)
    del model, sampler
    torch.cuda.empty_cache()

    ref = recs[0]["ms_step"]
    for r in recs:
        r["speedup_vs_k1"] = ref / r["ms_step"]
        r["saved_ms"] = ref - r["ms_step"]
        r["pct_vs_k1"] = 100.0 * (ref - r["ms_step"]) / ref
        print(f"  K={r['skip_k']:2d}  {r['ms_step']:7.3f}  {r['speedup_vs_k1']:.3f}x  "
              f"({r['pct_vs_k1']:+.2f}%)", flush=True)
    payload = {"model": "sd1.5", "gpu": torch.cuda.get_device_name(0),
               "batch": batch, "steps": steps, "shape": list(SHAPE),
               "seed": SEED_T, "trials": a.bench_trials, "arms": recs,
               "delta_mode": os.environ.get("MODIFF_DELTA_MODE"),
               "quant_attn": "0", "linear_modiff": "0"}
    path = os.path.join(OUT_DIR, "data", "skip_k_sd15_bench.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)
    return payload


def stage_layer():
    """One SD1.5-shaped conv: 320->320 at 32x32, batch 32 (similar bytes to churches 192/32/128)."""
    import torch
    import torch.nn as nn
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d

    N, C, H, STEPS, TRIALS = 32, 320, 32, 200, 3
    CL = torch.channels_last
    print(f"GPU {torch.cuda.get_device_name(0)}  one conv {C}->{C} {H}x{H} N={N}", flush=True)
    raw = nn.Conv2d(C, C, 3, padding=1).cuda()
    layer = OptimizedInt8Conv2d(raw, layer_name="sd15_one").cuda()
    layer.enable_modiff(True)
    layer.set_static_scale(16.0)
    layer.static_delta_scale.fill_(16.0)
    layer.static_delta_alpha.fill_(1.0 / 16.0)
    layer.is_delta_calibrated.fill_(True)
    layer._delta_cal = True
    layer.eval()
    xs = [torch.randn(N, C, H, H, device="cuda", dtype=torch.float16).contiguous(memory_format=CL)
          for _ in range(8)]

    def first_step():
        layer.reset_state()
        with torch.inference_mode():
            layer(xs[0])

    def run_200():
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        with torch.inference_mode():
            for t in range(STEPS):
                layer(xs[t % len(xs)])
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / STEPS

    _apply(1)
    first_step()
    run_200()
    rows = []
    for k in KS:
        _apply(k)
        samples = []
        for _ in range(TRIALS):
            first_step()
            samples.append(run_200())
        ms = statistics.median(samples)
        rows.append({"skip_k": k, "ms_step": ms, "trials": samples})
        print(f"  skip-K={k:2d}  {ms:.4f} ms/step  trials={[round(s, 3) for s in samples]}",
              flush=True)
    _apply(1)
    ref = rows[0]["ms_step"]
    for r in rows:
        r["vs_k1"] = ref / r["ms_step"]
        r["saved_ms"] = ref - r["ms_step"]
    payload = {"gpu": torch.cuda.get_device_name(0),
               "shape": {"N": N, "C": C, "H": H, "steps": STEPS},
               "arms": rows}
    path = os.path.join(OUT_DIR, "data", "skip_k_sd15_layer.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="samples,bench,layer")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED_Q)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--bench-batch", type=int, default=8)
    ap.add_argument("--bench-trials", type=int, default=2)
    a = ap.parse_args()
    if not os.path.exists(CKPT):
        raise SystemExit(f"missing SD1.5 checkpoint: {CKPT}")
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    preflight(*MODEL, "transformers", what="skip_k_sd15.py")
    if "samples" in stages:
        print("\n===== STAGE samples =====", flush=True)
        stage_samples(a)
    if "bench" in stages:
        print("\n===== STAGE bench =====", flush=True)
        stage_bench(a)
    if "layer" in stages:
        print("\n===== STAGE layer =====", flush=True)
        stage_layer()
    _apply(1)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _apply(1)
