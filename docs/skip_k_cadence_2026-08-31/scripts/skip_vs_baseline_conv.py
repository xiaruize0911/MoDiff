"""Single conv kernel: PTQ baseline vs MoDiff commit vs MoDiff skip.

Skip still runs the GEMM. It only skips the in-place a_hat / o_hat stores
(o_hat skip writes a separate `out` instead of RMW into the cache).

Two levels:
  kernel  already-quantized int8 activation → EVT conv
  layer   OptimizedInt8Conv2d.forward (static quantize + conv)

Shapes: the 192→192 32×32 one-layer point, plus the 20 UNet residual convs
freq-weighted.

Run: source setup_cuda_env.sh && python docs/skip_k_cadence_2026-08-31/scripts/skip_vs_baseline_conv.py
"""
import json
import os
import statistics
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "build/lib.linux-x86_64-cpython-311")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_WARMUP_STEPS"] = "1"

import torch
import torch.nn as nn
import modiff_cutlass as mc
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

DEV = "cuda"
CL = torch.channels_last
N_BATCH = 128
WARMUP, REPS, TRIALS = 8, 40, 3
STRIDES = (1, 1, 1, 1, 1, 1)
OUT = "docs/skip_k_cadence_2026-08-31/data/skip_vs_baseline_conv.json"

UNET = [
    (768, 768, 2, 2, 12), (384, 384, 8, 8, 8), (192, 192, 32, 32, 7), (384, 384, 16, 16, 7),
    (768, 768, 4, 4, 7), (1536, 768, 2, 2, 3), (1536, 768, 4, 4, 2), (768, 384, 8, 8, 2),
    (768, 384, 16, 16, 2), (384, 192, 32, 32, 2), (192, 192, 16, 16, 1), (192, 384, 16, 16, 1),
    (384, 384, 4, 4, 1), (384, 768, 4, 4, 1), (1152, 768, 4, 4, 1), (768, 768, 8, 8, 1),
    (1152, 384, 8, 8, 1), (576, 384, 16, 16, 1), (384, 384, 32, 32, 1), (576, 192, 32, 32, 1),
]


def cl(t):
    return t.contiguous(memory_format=CL)


def time_fn(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def median_trials(fn):
    return statistics.median(time_fn(fn) for _ in range(TRIALS))


def kernel_tensors(Cin, Cout, H, W):
    x = cl(torch.randint(-8, 8, (N_BATCH, Cin, H, W), device=DEV, dtype=torch.int8))
    w = torch.randint(-8, 8, (Cout, 3, 3, Cin), device=DEV, dtype=torch.int8).contiguous()
    inv = torch.tensor([1.0 / 16.0], device=DEV, dtype=torch.float32)
    wscale = torch.full((Cout,), 0.02, device=DEV, dtype=torch.float32)
    bias = torch.zeros(Cout, device=DEV, dtype=torch.float32)
    empty = torch.empty(0, device=DEV, dtype=torch.float16)
    out = cl(torch.empty(N_BATCH, Cout, H, W, device=DEV, dtype=torch.float16))
    ohat = cl(0.1 * torch.randn(N_BATCH, Cout, H, W, device=DEV, dtype=torch.float16))
    return dict(x=x, w=w, inv=inv, wscale=wscale, bias=bias, empty=empty, out=out, ohat=ohat)


def time_kernels(Cin, Cout, H, W):
    t = kernel_tensors(Cin, Cout, H, W)

    def baseline():
        mc.conv2d_int8_evt_bias_residual_fp16(
            t["x"], t["w"], t["inv"], t["wscale"], t["bias"], t["empty"], t["out"], *STRIDES)

    def commit():
        mc.conv2d_int8_evt_o_hat(
            t["x"], t["w"], t["inv"], t["wscale"], t["ohat"], *STRIDES)

    def skip():
        mc.conv2d_int8_evt_o_hat_skip(
            t["x"], t["w"], t["inv"], t["wscale"], t["ohat"], t["out"], *STRIDES)

    baseline(); commit(); skip(); torch.cuda.synchronize()
    b = median_trials(baseline)
    c = median_trials(commit)
    s = median_trials(skip)
    return {"baseline_us": b * 1000, "commit_us": c * 1000, "skip_us": s * 1000,
            "skip_vs_baseline": b / s, "commit_vs_baseline": b / c,
            "skip_saved_vs_commit_us": (c - s) * 1000}


def build_layer(Cin, Cout, H, W, modiff):
    raw = nn.Conv2d(Cin, Cout, 3, padding=1).to(DEV)
    layer = OptimizedInt8Conv2d(raw, layer_name="bench").to(DEV)
    layer.set_static_scale(16.0)
    layer.static_delta_scale.fill_(16.0)
    layer.static_delta_alpha.fill_(1.0 / 16.0)
    layer.is_delta_calibrated.fill_(True)
    layer._delta_cal = True
    layer.enable_modiff(modiff)
    layer.eval()
    return layer


def time_layer(Cin, Cout, H, W):
    xs = [cl(torch.randn(N_BATCH, Cin, H, W, device=DEV, dtype=torch.float16)) for _ in range(4)]
    base = build_layer(Cin, Cout, H, W, False)
    md = build_layer(Cin, Cout, H, W, True)

    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    md.reset_state()
    with torch.inference_mode():
        md(xs[0])
    assert md.is_first_step is False

    def baseline():
        with torch.inference_mode():
            base(xs[1])

    def commit():
        os.environ["MODIFF_CACHE_SKIP_K"] = "1"
        with torch.inference_mode():
            md(xs[1])

    def skip():
        os.environ["MODIFF_CACHE_SKIP_K"] = "1000000"
        with torch.inference_mode():
            md(xs[1])

    baseline(); commit(); skip(); torch.cuda.synchronize()
    b = median_trials(baseline)
    c = median_trials(commit)
    s = median_trials(skip)
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    return {"baseline_us": b * 1000, "commit_us": c * 1000, "skip_us": s * 1000,
            "skip_vs_baseline": b / s, "commit_vs_baseline": b / c,
            "skip_saved_vs_commit_us": (c - s) * 1000}


def weighted(rows, key):
    num = sum(r[key] * r["freq"] for r in rows)
    den = sum(r["freq"] for r in rows)
    return num / den


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  B={N_BATCH}  "
          f"warmup={WARMUP} reps={REPS} trials={TRIALS}", flush=True)

    focus = (192, 192, 32, 32)
    print("\n===== kernel only (int8 in → EVT conv)  192→192 32×32 =====", flush=True)
    k_focus = time_kernels(*focus)
    print(f"  baseline PTQ     {k_focus['baseline_us']:8.1f} µs", flush=True)
    print(f"  MoDiff commit    {k_focus['commit_us']:8.1f} µs  "
          f"{k_focus['commit_vs_baseline']:.3f}× vs baseline", flush=True)
    print(f"  MoDiff skip      {k_focus['skip_us']:8.1f} µs  "
          f"{k_focus['skip_vs_baseline']:.3f}× vs baseline  "
          f"saved vs commit {k_focus['skip_saved_vs_commit_us']:+.1f} µs", flush=True)

    print("\n===== layer (quantize + conv)  192→192 32×32 =====", flush=True)
    l_focus = time_layer(*focus)
    print(f"  baseline PTQ     {l_focus['baseline_us']:8.1f} µs", flush=True)
    print(f"  MoDiff commit    {l_focus['commit_us']:8.1f} µs  "
          f"{l_focus['commit_vs_baseline']:.3f}× vs baseline", flush=True)
    print(f"  MoDiff skip      {l_focus['skip_us']:8.1f} µs  "
          f"{l_focus['skip_vs_baseline']:.3f}× vs baseline  "
          f"saved vs commit {l_focus['skip_saved_vs_commit_us']:+.1f} µs", flush=True)

    print("\n===== kernel, 20 UNet shapes =====", flush=True)
    k_rows = []
    for Cin, Cout, H, W, freq in UNET:
        rec = time_kernels(Cin, Cout, H, W)
        rec.update({"Cin": Cin, "Cout": Cout, "H": H, "W": W, "freq": freq,
                    "shape": f"{Cin}->{Cout} {H}x{W}"})
        k_rows.append(rec)
        print(f"  {rec['shape']:22s} f{freq:<3}  "
              f"base {rec['baseline_us']:7.1f}  "
              f"commit {rec['commit_us']:7.1f}  "
              f"skip {rec['skip_us']:7.1f}  "
              f"skip/base {rec['skip_vs_baseline']:.3f}×  "
              f"Δcommit {rec['skip_saved_vs_commit_us']:+6.1f} µs", flush=True)

    fw = {
        "baseline_us": weighted(k_rows, "baseline_us"),
        "commit_us": weighted(k_rows, "commit_us"),
        "skip_us": weighted(k_rows, "skip_us"),
    }
    fw["skip_vs_baseline"] = fw["baseline_us"] / fw["skip_us"]
    fw["commit_vs_baseline"] = fw["baseline_us"] / fw["commit_us"]
    fw["skip_saved_vs_commit_us"] = fw["commit_us"] - fw["skip_us"]
    print(f"\n  freq-weighted     base {fw['baseline_us']:7.1f}  "
          f"commit {fw['commit_us']:7.1f}  skip {fw['skip_us']:7.1f}  "
          f"skip/base {fw['skip_vs_baseline']:.3f}×  "
          f"Δcommit {fw['skip_saved_vs_commit_us']:+.1f} µs", flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "batch": N_BATCH,
        "method": "CUDA events, median of 3 trials × 40 reps after 8 warmup",
        "kernel": "int8 activation already quantized; EVT conv only",
        "focus_192_32": {"kernel": k_focus, "layer": l_focus},
        "unet_kernel": k_rows,
        "unet_freq_weighted": fw,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
