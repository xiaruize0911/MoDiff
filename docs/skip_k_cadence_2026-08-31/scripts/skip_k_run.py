"""Skip-K cadence: freeze a_hat/o_hat writes, still compute GN+conv every step.

K in {1, 2, 5, 10}. K=1 is shipped MoDiff (write every step). Intermediate steps
keep the last committed caches; only the in-place a_hat/o_hat stores are skipped.
MODIFF_REPLAY_K stays 1.

Stages:
  samples  n=6 contact sheet + latent relL2 vs fp16 (seed 20260805, 50 DDIM)
  bench    batch 128 e2e ms/step (seed 20260827, 50 DDIM)
  layer    one OptimizedInt8Conv2d, 192->192 32x32, 200 modulated steps
  fid      N=2048 Inception FID vs existing fp16 / w8a8_full folders

Run: source setup_cuda_env.sh
     python docs/skip_k_cadence_2026-08-31/scripts/skip_k_run.py --stage samples,bench,layer
     python docs/skip_k_cadence_2026-08-31/scripts/skip_k_run.py --stage fid
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
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "docs/ahat_fake_quant_2026-08-27/scripts")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"

from integration.utils.preflight import preflight, MODEL, FID  # noqa: E402

KS = (1, 2, 5, 10)
SHAPE = (4, 32, 32)
CALIB8 = "integration/calibration/int8_calibration_realckpt.pt"
OUT_DIR = "docs/skip_k_cadence_2026-08-31"
FID_ROOT = "docs/cache_schemes_report_2026-08-28/fid_samples"
SEED_Q = 20260805
SEED_T = 20260827

_STATS = {"write": 0, "skip": 0}
_ORIG_WRITE = None


def _apply(k):
    os.environ["MODIFF_CACHE_SKIP_K"] = str(k)
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_AHAT_BITS"] = "16"
    os.environ["MODIFF_AHAT_REFRESH"] = "0"


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


def decode(model, lat, chunk=8):
    import torch
    lat = lat.to("cuda", torch.float16)
    out = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for i in range(0, lat.shape[0], chunk):
            d = model.decode_first_stage(lat[i:i + chunk])
            out.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0).permute(0, 2, 3, 1).cpu())
            del d
    return (torch.cat(out, 0).numpy() * 255).round().astype("uint8")


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
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(OUT_DIR, "tmp_samples"),
        batch_size=a.n, steps=a.steps, shape=SHAPE,
        calibration_path=None, auto_delta_table=True)

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

    print("===== W8A8 MoDiff skip-K =====", flush=True)
    runner.calibration_path = CALIB8
    _apply(1)
    model, sampler = runner._setup_model("int8")
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
            im.save(os.path.join(png_dir, f"row{ridx:02d}_k{KS[ridx - 1] if ridx else 0}_{i:02d}.png"))
        y += cell + pad
    grid = os.path.join(OUT_DIR, "plots", "skip_k_grid.png")
    os.makedirs(os.path.dirname(grid), exist_ok=True)
    canvas.save(grid, "PNG")
    payload = {"seed": a.seed, "steps": a.steps, "n": a.n, "K": list(KS),
               "gpu": torch.cuda.get_device_name(0), "relL2": quality, "grid": grid}
    path = os.path.join(OUT_DIR, "data", "skip_k_quality.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {grid}  ({W}x{Hh})\nwrote {path}", flush=True)
    return payload


def stage_bench(a):
    import torch
    import integration.benchmarks.benchmark_ldm as B

    batch, steps = a.bench_batch, a.steps
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={batch} steps={steps}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(OUT_DIR, "tmp_bench"),
        batch_size=batch, steps=steps, shape=SHAPE,
        calibration_path=CALIB8, auto_delta_table=True)
    _apply(1)
    model, sampler = runner._setup_model("int8")

    recs = []
    for k in KS:
        _apply(k)

        def once():
            _reset(model, True)
            torch.manual_seed(SEED_T)
            torch.cuda.manual_seed_all(SEED_T)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                sampler.sample(S=steps, batch_size=batch, shape=SHAPE, eta=0.0, verbose=False)
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
    payload = {"gpu": torch.cuda.get_device_name(0), "batch": batch, "steps": steps,
               "seed": SEED_T, "trials": a.bench_trials, "arms": recs}
    path = os.path.join(OUT_DIR, "data", "skip_k_bench.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)
    return payload


def stage_layer():
    import torch
    import torch.nn as nn
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d

    N, C, H, STEPS, TRIALS = 128, 192, 32, 200, 3
    CL = torch.channels_last
    print(f"GPU {torch.cuda.get_device_name(0)}  one conv {C}->{C} {H}x{H} N={N}", flush=True)
    raw = nn.Conv2d(C, C, 3, padding=1).cuda()
    layer = OptimizedInt8Conv2d(raw, layer_name="one").cuda()
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
    path = os.path.join(OUT_DIR, "data", "skip_k_layer.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)
    return payload


def sample_batch_png(runner, model, sampler, n, seed, steps, decode_chunk, quantized):
    import torch
    _reset(model, quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
        lat = lat.to("cuda", torch.float16)
        chunks = []
        for i in range(0, lat.shape[0], decode_chunk):
            d = model.decode_first_stage(lat[i:i + decode_chunk])
            chunks.append(torch.clamp((d.float() + 1.0) / 2.0, 0.0, 1.0)
                          .permute(0, 2, 3, 1).cpu())
            del d
        img = torch.cat(chunks, 0)
    return (img.numpy() * 255).round().astype("uint8")


def generate_folder(runner, model, sampler, folder, n, batch, seed0, steps,
                    decode_chunk, quantized):
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    have = len([f for f in os.listdir(folder) if f.endswith(".png")])
    if have >= n:
        print(f"  {folder}: already {have} pngs, skip", flush=True)
        return
    if have:
        print(f"  {folder}: resume from {have}/{n}", flush=True)
    _reset(model, quantized)
    sample_batch_png(runner, model, sampler, min(batch, 16), seed0 - 1, steps,
                     decode_chunk, quantized)
    written, bi, t0 = have, have // batch, time.time()
    if have % batch:
        raise SystemExit(f"{folder} has {have} pngs, not a multiple of batch={batch}")
    while written < n:
        k = min(batch, n - written)
        arr = sample_batch_png(runner, model, sampler, k, seed0 + bi, steps,
                               decode_chunk, quantized)
        for i in range(arr.shape[0]):
            Image.fromarray(arr[i]).save(os.path.join(folder, f"{written + i:06d}.png"), "PNG")
        written += arr.shape[0]
        bi += 1
        el = time.time() - t0
        print(f"  {os.path.basename(folder)}  {written}/{n}  {el:.0f}s  "
              f"~{el / max(written - have, 1) * (n - written):.0f}s left", flush=True)


def stage_fid(a):
    import torch
    import integration.benchmarks.benchmark_ldm as B
    from pytorch_fid.fid_score import calculate_frechet_distance, compute_statistics_of_path
    from pytorch_fid.inception import InceptionV3

    n, batch, steps, seed0 = a.fid_n, a.bench_batch, a.steps, a.seed
    decode_chunk = a.decode_chunk
    # K=1 reuses w8a8_full. Generate K=2,5,10 only.
    new_arms = [(f"skip{k}", k) for k in KS if k != 1]
    print(f"GPU: {torch.cuda.get_device_name()}  n={n} steps={steps} batch={batch}", flush=True)

    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(FID_ROOT, "_tmp_skipk"),
        batch_size=batch, steps=steps, shape=SHAPE,
        calibration_path=CALIB8, auto_delta_table=True)
    _apply(1)
    model, sampler = runner._setup_model("int8")
    for name, k in new_arms:
        print(f"===== {name} skip-K={k} =====", flush=True)
        _apply(k)
        generate_folder(runner, model, sampler, os.path.join(FID_ROOT, name),
                        n, batch, seed0, steps, decode_chunk, quantized=True)
    _apply(1)
    del model, sampler
    torch.cuda.empty_cache()

    folders = ["fp16", "w8a8_full"] + [name for name, _ in new_arms]
    for name in folders:
        path = os.path.join(FID_ROOT, name)
        have = len([f for f in os.listdir(path) if f.endswith(".png")]) if os.path.isdir(path) else 0
        if have < n:
            raise SystemExit(f"FID folder {path} has {have}/{n} pngs")

    print("===== FID =====", flush=True)
    dev = torch.device("cuda")
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inc = InceptionV3([block]).to(dev)
    cache = {}
    for name in folders:
        path = os.path.join(FID_ROOT, name)
        n_img = len([f for f in os.listdir(path) if f.endswith(".png")])
        print(f"  features {name} n={n_img}", flush=True)
        mu, sigma = compute_statistics_of_path(path, inc, 64, 2048, dev, num_workers=4)
        cache[name] = (mu, sigma, n_img)

    def fid(x, y):
        return float(calculate_frechet_distance(cache[x][0], cache[x][1],
                                                cache[y][0], cache[y][1]))

    rows = {}
    for name in folders:
        rec = {
            "n": cache[name][2],
            "fid_vs_fp16": 0.0 if name == "fp16" else fid("fp16", name),
            "fid_vs_w8a8_full": 0.0 if name == "w8a8_full" else fid("w8a8_full", name),
        }
        rows[name] = rec
        print(f"  {name:16s}  FID vs fp16 {rec['fid_vs_fp16']:.3f}  "
              f"vs W8A8-full {rec['fid_vs_w8a8_full']:.3f}", flush=True)
    payload = {
        "metric": "inception_v3_fid_dims2048",
        "n": n, "steps": steps, "seed0": seed0,
        "note": "N=2048 ranking FID. fp16 and w8a8_full reused from cache_schemes_report.",
        "arms": rows,
    }
    path = os.path.join(OUT_DIR, "data", "skip_k_fid.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="samples,bench,layer",
                    help="comma-separated: samples,bench,layer,fid")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED_Q)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--bench-batch", type=int, default=128)
    ap.add_argument("--bench-trials", type=int, default=2)
    ap.add_argument("--fid-n", type=int, default=2048)
    ap.add_argument("--decode-chunk", type=int, default=32)
    a = ap.parse_args()
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    need_fid = "fid" in stages
    preflight(*MODEL, *(FID if need_fid else ()), what="skip_k_run.py")
    if "samples" in stages:
        print("\n===== STAGE samples =====", flush=True)
        stage_samples(a)
    if "bench" in stages:
        print("\n===== STAGE bench =====", flush=True)
        stage_bench(a)
    if "layer" in stages:
        print("\n===== STAGE layer =====", flush=True)
        stage_layer()
    if "fid" in stages:
        print("\n===== STAGE fid =====", flush=True)
        stage_fid(a)
    _apply(1)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _apply(1)
