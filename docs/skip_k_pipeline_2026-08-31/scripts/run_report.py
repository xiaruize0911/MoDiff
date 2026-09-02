"""Skip-K on the production W8A8 pipeline: quality + e2e profile + one-layer.

Same QUANT_ENV as e2e_three_mode_bench (attention W8A8 fused epilogue,
MODIFF_LINEAR=0, static delta). Arms: fp16, PTQ, MoDiff K=1,2,4,5,10.

  source setup_cuda_env.sh
  python docs/skip_k_pipeline_2026-08-31/scripts/run_report.py
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "docs/skip_k_pipeline_2026-08-31")
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

KS = (1, 2, 4, 5, 10)  # overwritten in main() from --k / --merge extras
SHAPE = (4, 32, 32)
CALIB8 = "integration/calibration/int8_calibration_realckpt.pt"
SEED_Q = 20260805
SEED_T = 20260827

QUANT_ENV = {
    "MODIFF_QUANT_LINEAR": "1",
    "MODIFF_QUANT_ATTN": "1",
    "MODIFF_QUANT_ATTN_STATIC": "1",
    "MODIFF_QATTN_FLASH": "1",
    "MODIFF_FLASH_GATE": "on",
    "MODIFF_QUANT_ATTN_ALLT": "0",
    "MODIFF_LINEAR_OUT_I8": "0",
}

_STATS = {"write": 0, "skip": 0}
_ORIG_WRITE = None


def apply_k(k):
    os.environ["MODIFF_CACHE_SKIP_K"] = str(k)
    os.environ["MODIFF_REPLAY_K"] = "1"


def set_quant(on):
    for k, v in QUANT_ENV.items():
        os.environ[k] = v if on else ("0" if k in ("MODIFF_QUANT_LINEAR",
                                                   "MODIFF_QUANT_ATTN") else v)
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_DELTA_MODE"] = "static"
    os.environ["MODIFF_REPLAY_K"] = "1"
    apply_k(1)
    for k in ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND"):
        os.environ.pop(k, None)


def install_write_counter():
    global _ORIG_WRITE
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    if _ORIG_WRITE is None:
        _ORIG_WRITE = OptimizedInt8Conv2d._write_ahat_now

        def wrapped(self):
            w = _ORIG_WRITE(self)
            _STATS["write" if w else "skip"] += 1
            return w

        OptimizedInt8Conv2d._write_ahat_now = wrapped


def reset_stats():
    _STATS["write"] = _STATS["skip"] = 0


def reset_model(model, quantized):
    import integration.benchmarks.benchmark_ldm as B
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def bucket_kernel(name):
    n = name.lower()
    if "flash" in n or "attn" in n or "attention" in n or "softmax" in n or "sdpa" in n or "bmm" in n:
        return "attention"
    if "cutlass" in n or "gemm" in n or "implicit" in n or "xmma" in n:
        return "GEMM / conv"
    if "group_norm" in n or "groupnorm" in n or "silu" in n or n.startswith("gn_") or "gn_" in n:
        return "GroupNorm+SiLU family"
    if "quant" in n or "absmax" in n or "pack" in n:
        return "quantize (standalone)"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "cat" in n:
        return "elementwise / copy"
    return "other"


BUCKETS = ["GroupNorm+SiLU family", "GEMM / conv", "attention",
           "quantize (standalone)", "elementwise / copy", "other"]


def kernel_table(prof, wall_us):
    from torch.profiler import DeviceType
    agg = defaultdict(lambda: {"us": 0.0, "calls": 0})
    for e in prof.key_averages():
        if e.device_type != DeviceType.CUDA:
            continue
        us = float(getattr(e, "self_device_time_total", 0) or 0)
        if us <= 0:
            continue
        agg[e.key]["us"] += us
        agg[e.key]["calls"] += int(e.count)
    tot = sum(v["us"] for v in agg.values())
    f = (wall_us / tot) if tot > 0 else 1.0
    rows = [{"kernel": k, "us": v["us"] * f, "calls": v["calls"],
             "pct": v["us"] / tot * 100 if tot else 0.0} for k, v in agg.items()]
    rows.sort(key=lambda r: -r["us"])
    return rows, tot


def bucket_ms(kernels, steps):
    acc = defaultdict(float)
    for k in kernels:
        acc[bucket_kernel(k["kernel"])] += k["us"] / 1e3 / steps
    return {b: acc.get(b, 0.0) for b in BUCKETS}


def make_runner(batch, steps, calib):
    import integration.benchmarks.benchmark_ldm as B
    return B.BenchmarkRunner(
        "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        "models/ldm/lsun_churches256/model.ckpt",
        output_dir=os.path.join(OUT, "tmp"),
        batch_size=batch, steps=steps, shape=SHAPE,
        calibration_path=calib, auto_delta_table=True,
        linear_backend="int_gemm" if calib else "fp16")


def sample_once(runner, model, sampler, n, steps, seed, quantized):
    import torch
    reset_model(model, quantized)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.float()


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


def time_arm(sample_fn, steps, warmups, repeats, prepare_fn=None):
    import torch
    prep = prepare_fn or (lambda: None)
    for _ in range(warmups):
        prep()
        sample_fn()
    torch.cuda.synchronize()
    times_us = []
    for _ in range(repeats):
        prep()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        sample_fn()
        e.record()
        torch.cuda.synchronize()
        times_us.append(s.elapsed_time(e) * 1e3)
    wall_us = statistics.median(times_us)
    from torch.profiler import profile, ProfilerActivity
    prep()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        sample_fn()
    rows, raw_us = kernel_table(prof, wall_us)
    return {
        "wall_us_per_batch": wall_us,
        "wall_all_us": [round(t, 1) for t in times_us],
        "wall_cv_pct": (statistics.stdev(times_us) / statistics.mean(times_us) * 100
                        if len(times_us) > 1 else 0.0),
        "per_step_ms": wall_us / 1e3 / steps,
        "kernels": rows,
        "buckets_ms_per_step": bucket_ms(rows, steps),
        "profiler_raw_gpu_us": raw_us,
    }


def stage_e2e(a):
    import torch
    import integration.benchmarks.benchmark_ldm as B

    modes = {}
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU {gpu}  batch={a.batch} steps={a.steps}", flush=True)

    def run_mode(label, mode, k, calib):
        set_quant(mode != "fp16")
        apply_k(k)
        runner = make_runner(a.batch, a.steps, calib)
        model, sampler = runner._setup_model(mode)
        cond = runner._cond_kwargs(model, a.batch)

        def prepare():
            reset_model(model, mode != "fp16")

        def sample():
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                            dtype=torch.float16):
                sampler.sample(S=a.steps, batch_size=a.batch, shape=SHAPE,
                               eta=0.0, verbose=False, **cond)

        rec = time_arm(sample, a.steps, a.warmups, a.repeats, prepare_fn=prepare)
        rec["label"] = label
        rec["mode"] = mode
        rec["skip_k"] = k
        modes[label] = rec
        print(f"  {label:16s} {rec['per_step_ms']:7.2f} ms/step  "
              f"CV {rec['wall_cv_pct']:.2f}%", flush=True)
        del model, sampler, runner
        torch.cuda.empty_cache()
        apply_k(1)

    merge_path = os.path.join(OUT, "data", "e2e.json")
    if a.merge and os.path.exists(merge_path):
        prev = json.load(open(merge_path))
        modes.update(prev.get("modes", {}))
        print(f"  merge: keeping {list(modes)} from {merge_path}", flush=True)
    else:
        run_mode("fp16", "fp16", 1, None)
        run_mode("PTQ", "int8_baseline", 1, CALIB8)

    # One int8 MoDiff load, loop K.
    set_quant(True)
    apply_k(1)
    runner = make_runner(a.batch, a.steps, CALIB8)
    model, sampler = runner._setup_model("int8")
    cond = runner._cond_kwargs(model, a.batch)
    for k in KS:
        apply_k(k)
        label = f"K={k}"

        def prepare(k=k):
            apply_k(k)
            reset_model(model, True)

        def sample():
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True,
                                                            dtype=torch.float16):
                sampler.sample(S=a.steps, batch_size=a.batch, shape=SHAPE,
                               eta=0.0, verbose=False, **cond)

        rec = time_arm(sample, a.steps, a.warmups, a.repeats, prepare_fn=prepare)
        rec["label"] = label
        rec["mode"] = "int8"
        rec["skip_k"] = k
        modes[label] = rec
        print(f"  {label:16s} {rec['per_step_ms']:7.2f} ms/step  "
              f"CV {rec['wall_cv_pct']:.2f}%", flush=True)
    apply_k(1)
    del model, sampler, runner
    torch.cuda.empty_cache()

    fp = modes["fp16"]["wall_us_per_batch"]
    ptq = modes["PTQ"]["wall_us_per_batch"]
    k1 = modes["K=1"]["wall_us_per_batch"]
    for rec in modes.values():
        rec["speedup_vs_fp16"] = fp / rec["wall_us_per_batch"]
        rec["speedup_vs_ptq"] = ptq / rec["wall_us_per_batch"]
        rec["speedup_vs_k1"] = k1 / rec["wall_us_per_batch"]

    k_all = sorted({int(m[2:]) for m in modes if m.startswith("K=")})
    payload = {"gpu": gpu, "batch": a.batch, "steps": a.steps,
               "repeats": a.repeats, "warmups": a.warmups, "K": k_all,
               "modes": modes}
    path = os.path.join(OUT, "data", "e2e.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=1)
    print(f"wrote {path}", flush=True)
    return payload


def stage_quality(a):
    import torch
    from PIL import Image, ImageDraw

    install_write_counter()
    print(f"GPU {torch.cuda.get_device_name(0)}  n={a.n} steps={a.steps}", flush=True)
    rows, quality, ref = [], {}, None

    def add(label, lat, extra=None):
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
        print(f"  {label:16s} relL2 {rel:.4f}{extra_s}", flush=True)
        rows.append((f"{label}  relL2 {rel:.4f}" if rel else label, decode(model, lat)))

    set_quant(False)
    apply_k(1)
    runner = make_runner(a.n, a.steps, None)
    model, sampler = runner._setup_model("fp16")
    sample_once(runner, model, sampler, a.n, a.steps, a.seed, False)
    add("fp16", sample_once(runner, model, sampler, a.n, a.steps, a.seed, False))
    del model, sampler
    torch.cuda.empty_cache()

    merge_q = os.path.join(OUT, "data", "quality.json")
    if a.merge and os.path.exists(merge_q):
        quality.update(json.load(open(merge_q)).get("relL2", {}))
        print(f"  merge quality keys {list(quality)}", flush=True)
        # still need a live fp16 ref for the new K relL2; already sampled above
    else:
        set_quant(True)
        apply_k(1)
        runner.calibration_path = CALIB8
        runner.linear_backend = "int_gemm"
        model, sampler = runner._setup_model("int8_baseline")
        sample_once(runner, model, sampler, a.n, a.steps, a.seed, True)
        add("PTQ", sample_once(runner, model, sampler, a.n, a.steps, a.seed, True),
            {"skip_k": None})
        del model, sampler
        torch.cuda.empty_cache()

    set_quant(True)
    apply_k(1)
    runner.calibration_path = CALIB8
    runner.linear_backend = "int_gemm"
    model, sampler = runner._setup_model("int8")
    for k in KS:
        apply_k(k)
        reset_stats()
        sample_once(runner, model, sampler, a.n, a.steps, a.seed, True)
        reset_stats()
        lat = sample_once(runner, model, sampler, a.n, a.steps, a.seed, True)
        n_w, n_s = _STATS["write"], _STATS["skip"]
        add(f"K={k}", lat, {"skip_k": k, "n_write": n_w, "n_skip": n_s,
                            "skip_frac": n_s / max(n_w + n_s, 1)})
    apply_k(1)
    del model, sampler
    torch.cuda.empty_cache()

    cell, pad, labh = a.cell, 6, 26
    W = pad + a.n * (cell + pad)
    Hh = len(rows) * (cell + labh + pad) + pad
    canvas = Image.new("RGB", (W, Hh), (252, 252, 251))
    dr = ImageDraw.Draw(canvas)
    y = pad
    png_dir = os.path.join(OUT, "samples")
    os.makedirs(png_dir, exist_ok=True)
    grid = os.path.join(OUT, "plots", "quality_grid.png")
    os.makedirs(os.path.dirname(grid), exist_ok=True)
    for ridx, (label, arr) in enumerate(rows):
        dr.text((pad, y + 6), label, fill=(11, 11, 11))
        y += labh
        for i in range(min(a.n, arr.shape[0])):
            im = Image.fromarray(arr[i])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (pad + i * (cell + pad), y))
            im.save(os.path.join(png_dir, f"row{ridx:02d}_{i:02d}.png"))
        y += cell + pad
    grid_path = (os.path.join(OUT, "plots", "quality_grid_highk.png") if a.merge
                 else grid)
    canvas.save(grid_path, "PNG")
    k_all = sorted({int(k[2:]) for k in quality if k.startswith("K=")})
    payload = {"seed": a.seed, "steps": a.steps, "n": a.n, "K": k_all,
               "gpu": torch.cuda.get_device_name(0), "relL2": quality, "grid": grid_path}
    path = os.path.join(OUT, "data", "quality.json")
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {grid_path}\nwrote {path}", flush=True)
    return payload


def stage_layer(a):
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

    apply_k(1)
    first_step()
    run_200()
    rows = []
    merge_l = os.path.join(OUT, "data", "layer.json")
    if a.merge and os.path.exists(merge_l):
        prev = json.load(open(merge_l))
        rows.extend(prev.get("arms", []))
        print(f"  merge layer {len(rows)} existing arms", flush=True)
    for k in KS:
        if any(r.get("skip_k") == k for r in rows):
            continue
        apply_k(k)
        samples = []
        for _ in range(TRIALS):
            first_step()
            samples.append(run_200())
        ms = statistics.median(samples)
        rows.append({"skip_k": k, "ms_step": ms, "trials": samples})
        print(f"  skip-K={k:2d}  {ms:.4f} ms/step", flush=True)
    apply_k(1)
    k1_row = next(r for r in rows if r["skip_k"] == 1)
    ref = k1_row["ms_step"]
    for r in rows:
        r["vs_k1"] = ref / r["ms_step"]
        r["saved_ms"] = ref - r["ms_step"]
    rows.sort(key=lambda r: r["skip_k"])
    payload = {"gpu": torch.cuda.get_device_name(0),
               "shape": {"N": N, "C": C, "H": H, "steps": STEPS},
               "arms": rows}
    path = os.path.join(OUT, "data", "layer.json")
    json.dump(payload, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)
    return payload


def main():
    global KS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="e2e,quality,layer")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED_Q)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--k", default=None, help="comma-separated K list")
    ap.add_argument("--merge", action="store_true",
                    help="keep existing fp16/PTQ/low-K JSON; only time --k")
    a = ap.parse_args()
    if a.k:
        KS = tuple(int(x) for x in a.k.split(",") if x.strip())
    a.merge = bool(a.merge)
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    if "e2e" in stages:
        print("\n===== STAGE e2e =====", flush=True)
        stage_e2e(a)
    if "quality" in stages:
        print("\n===== STAGE quality =====", flush=True)
        stage_quality(a)
    if "layer" in stages:
        print("\n===== STAGE layer =====", flush=True)
        stage_layer(a)
    apply_k(1)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        apply_k(1)
