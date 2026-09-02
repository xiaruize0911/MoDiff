"""Per-kernel timing: baseline vs MoDiff INT8 conv, swept over (B, H, W, N).

The unfused path is two CUDA kernels:

  k1  CUTLASS ImplicitGemm  (fp32 scratch)     — same binary for both arms
  k2  epilogue
        baseline  scale_store_half_vec2         — write-only dequant store
        MoDiff    scale_accumulate_half_cache   — RMW into o_hat

Production is the fused EVT (one kernel). Timed as a reference, not mixed
into the k1/k2 tables.

CUDA-profiler kernel self-times, mean of TRIALS, each trial = mean over REPS.
Source setup_cuda_env.sh && python docs/conv_kernel_sweep_2026-08-28/scripts/conv_kernel_sweep.py
"""
import json
import os
import statistics
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "build/lib.linux-x86_64-cpython-311")]

import torch
from torch.profiler import ProfilerActivity, profile

import modiff_cutlass as mc

DEV = "cuda"
CL = torch.channels_last
HERE = "docs/conv_kernel_sweep_2026-08-28"
WARMUP, REPS, TRIALS = 8, 24, 3

# Default point: a mid-frequency churches UNet residual (384 ch, 16x16, b128).
DEFAULT = dict(B=128, H=16, W=16, N=384, C=384)

SWEEPS = {
    "B": [8, 16, 32, 64, 128, 256],
    "H": [2, 4, 8, 16, 32],
    "W": [2, 4, 8, 16, 32],
    "N": [128, 192, 256, 384, 512, 768, 1152, 1536],
}

# (Cin, Cout, H, W, freq) — same 20 shapes as conv_layer_microbench.py
UNET = [
    (768, 768, 2, 2, 12), (384, 384, 8, 8, 8), (192, 192, 32, 32, 7), (384, 384, 16, 16, 7),
    (768, 768, 4, 4, 7), (1536, 768, 2, 2, 3), (1536, 768, 4, 4, 2), (768, 384, 8, 8, 2),
    (768, 384, 16, 16, 2), (384, 192, 32, 32, 2), (192, 192, 16, 16, 1), (192, 384, 16, 16, 1),
    (384, 384, 4, 4, 1), (384, 768, 4, 4, 1), (1152, 768, 4, 4, 1), (768, 768, 8, 8, 1),
    (1152, 384, 8, 8, 1), (576, 384, 16, 16, 1), (384, 384, 32, 32, 1), (576, 192, 32, 32, 1),
]


def cl(t):
    return t.contiguous(memory_format=CL)


def make_tensors(B, C, H, W, N):
    x = cl(torch.randint(-8, 8, (B, C, H, W), device=DEV, dtype=torch.int8))
    w = torch.randint(-8, 8, (N, 3, 3, C), device=DEV, dtype=torch.int8).contiguous()
    inv = torch.tensor([1.0 / 16.0], device=DEV, dtype=torch.float32)
    wscale = torch.full((N,), 0.02, device=DEV, dtype=torch.float32)
    bias = torch.zeros(N, device=DEV, dtype=torch.float32)
    empty = torch.empty(0, device=DEV, dtype=torch.float16)
    out = cl(torch.empty(B, N, H, W, device=DEV, dtype=torch.float16))
    ohat = cl(0.1 * torch.randn(B, N, H, W, device=DEV, dtype=torch.float16))
    return dict(x=x, w=w, inv=inv, wscale=wscale, bias=bias, empty=empty, out=out, ohat=ohat)


def parse_prof(prof):
    gemm_us = epi_us = fused_us = 0.0
    epi_kind = None
    for e in prof.key_averages():
        if e.count <= 0:
            continue
        us = (e.self_device_time_total / e.count
              if e.self_device_time_total else e.device_time)
        if us < 0.05:
            continue
        name = e.key
        if "scale_store" in name:
            epi_us += us
            epi_kind = "store"
        elif "scale_accumulate" in name:
            epi_us += us
            epi_kind = "accumulate"
        elif "ImplicitGemm" in name or "implicit_gemm" in name.lower():
            if "modiff" in name:
                fused_us += us
            else:
                gemm_us += us
    return dict(k1_us=gemm_us, k2_us=epi_us, k2_kind=epi_kind, evt_us=fused_us)


def profile_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
    return parse_prof(prof)


def time_shape(B, C, H, W, N):
    t = make_tensors(B, C, H, W, N)
    strides = (1, 1, 1, 1, 1, 1)

    def base_unfused():
        mc.conv2d_int8_fprop_no_ohat_prealloc(
            t["x"], t["w"], t["inv"], t["wscale"], t["out"], *strides)

    def md_unfused():
        mc.conv2d_int8_fprop_o_hat(
            t["x"], t["w"], t["inv"], t["wscale"], t["ohat"], *strides)

    def base_evt():
        mc.conv2d_int8_evt_bias_residual_fp16(
            t["x"], t["w"], t["inv"], t["wscale"], t["bias"], t["empty"], t["out"], *strides)

    def md_evt():
        mc.conv2d_int8_evt_o_hat(
            t["x"], t["w"], t["inv"], t["wscale"], t["ohat"], *strides)

    samples = {k: [] for k in
               ("base_k1", "base_k2", "md_k1", "md_k2", "base_evt", "md_evt")}
    try:
        base_unfused(); md_unfused(); base_evt(); md_evt()
        torch.cuda.synchronize()
    except RuntimeError as ex:
        return {"error": str(ex).split("\n")[0][:160]}

    fns = [("base", base_unfused, "unfused"),
           ("md", md_unfused, "unfused"),
           ("base", base_evt, "evt"),
           ("md", md_evt, "evt")]
    for trial in range(TRIALS):
        order = fns[trial % 4:] + fns[: trial % 4]
        for arm, fn, kind in order:
            p = profile_fn(fn)
            if kind == "unfused":
                samples[f"{arm}_k1"].append(p["k1_us"])
                samples[f"{arm}_k2"].append(p["k2_us"])
            else:
                samples[f"{arm}_evt"].append(p["evt_us"])

    def mean_std(xs):
        xs = [v for v in xs if v and v > 0]
        if not xs:
            return 0.0, 0.0
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return m, s

    out = {}
    for k, xs in samples.items():
        m, s = mean_std(xs)
        out[f"{k}_us"] = round(m, 3)
        out[f"{k}_std"] = round(s, 3)
    bk, mk = out["base_k1_us"], out["md_k1_us"]
    be, me = out["base_k2_us"], out["md_k2_us"]
    out["k1_speedup"] = round(bk / mk, 4) if mk else 0.0
    out["k2_speedup"] = round(be / me, 4) if me else 0.0
    out["unfused_speedup"] = round((bk + be) / (mk + me), 4) if (mk + me) else 0.0
    bv, mv = out["base_evt_us"], out["md_evt_us"]
    out["evt_speedup"] = round(bv / mv, 4) if mv else 0.0
    flops = 2.0 * B * H * W * N * C * 9
    out["k1_tflops"] = round(flops / (bk * 1e-6) / 1e12, 2) if bk else 0.0
    return out


def point(axis, value):
    cfg = dict(DEFAULT)
    cfg[axis] = value
    if axis == "N":
        cfg["C"] = value
    if axis == "H" and "W" not in (axis,):
        pass
    return cfg


def main():
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU {gpu}  WARMUP={WARMUP} REPS={REPS} TRIALS={TRIALS}",
          f"default {DEFAULT}", flush=True)

    sweeps = {}
    for axis, values in SWEEPS.items():
        rows = []
        for v in values:
            cfg = point(axis, v)
            rec = time_shape(**cfg)
            rec.update(cfg)
            rec["axis"] = axis
            rec["value"] = v
            rows.append(rec)
            if "error" in rec:
                print(f"  {axis}={v:<5}  FAIL {rec['error']}", flush=True)
            else:
                print(f"  {axis}={v:<5}  k1 {rec['base_k1_us']:7.1f}/{rec['md_k1_us']:7.1f} "
                      f"({rec['k1_speedup']:.3f}x)  k2 {rec['base_k2_us']:6.1f}/{rec['md_k2_us']:6.1f} "
                      f"({rec['k2_speedup']:.3f}x)  evt {rec['base_evt_us']:6.1f}/{rec['md_evt_us']:6.1f} "
                      f"({rec['evt_speedup']:.3f}x)", flush=True)
        sweeps[axis] = rows

    print("\n===== UNet 20 shapes  B=128 =====", flush=True)
    unet = []
    for Cin, Cout, H, W, freq in UNET:
        rec = time_shape(128, Cin, H, W, Cout)
        rec.update(dict(B=128, C=Cin, N=Cout, H=H, W=W, freq=freq,
                        shape=f"{Cin}->{Cout},{H}x{W}"))
        unet.append(rec)
        if "error" in rec:
            print(f"  {rec['shape']:22s} FAIL", flush=True)
        else:
            print(f"  {rec['shape']:22s} f{freq:<3}  "
                  f"k1 {rec['k1_speedup']:.3f}x  k2 {rec['k2_speedup']:.3f}x  "
                  f"evt {rec['evt_speedup']:.3f}x   "
                  f"k1 {rec['base_k1_us']:.1f}/{rec['md_k1_us']:.1f}  "
                  f"k2 {rec['base_k2_us']:.1f}/{rec['md_k2_us']:.1f}", flush=True)

    out = {
        "gpu": gpu,
        "method": ("torch.profiler CUDA self-time, mean of "
                   f"{TRIALS} trials x {REPS} reps after {WARMUP} warmup"),
        "unit": "us/call",
        "default": DEFAULT,
        "kernels": {
            "k1": "CUTLASS ImplicitGemmConvolution (fp32 scratch)",
            "k2_baseline": "scale_store_half_vec2_kernel",
            "k2_modiff": "scale_accumulate_half_cache_vec2_kernel",
            "evt_baseline": "conv2d_int8_evt_bias_residual_fp16 (D1, no residual)",
            "evt_modiff": "conv2d_int8_evt_o_hat (D2nr, o_hat RMW)",
        },
        "speedup_def": "baseline_us / MoDiff_us  (>1 MoDiff faster)",
        "sweeps": sweeps,
        "unet_shapes": unet,
    }
    path = f"{HERE}/data/conv_kernel_sweep.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)
    plot(out)
    return 0


def plot(data):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    out_dir = f"{HERE}/plots"
    os.makedirs(out_dir, exist_ok=True)
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_b = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    if os.path.exists(font):
        font_manager.fontManager.addfont(font)
        font_manager.fontManager.addfont(font_b)
        mpl.rcParams["font.family"] = "Noto Sans CJK JP"
    mpl.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "text.color": "#1a1a1a",
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    })
    BASE_K1, MD_K1 = "#2563eb", "#1d4ed8"
    BASE_K2, MD_K2 = "#d97706", "#b45309"
    KEEP, DROP = "#1a7f37", "#b42318"
    MUTED = "#666666"
    axis_title = {
        "B": ("B  batch", "B"),
        "H": ("H  height  (W=16 fixed)", "H"),
        "W": ("W  width  (H=16 fixed)", "W"),
        "N": ("N  output channels  (C=N)", "N"),
    }

    def ok(rows):
        return [r for r in rows if "error" not in r and r.get("base_k1_us", 0) > 0]

    def save(fig, name):
        p = os.path.join(out_dir, name)
        fig.savefig(p)
        plt.close(fig)
        print("wrote", p, flush=True)

    # ---- one figure per hyperparameter: time + speedup ----
    for axis, rows in data["sweeps"].items():
        rs = ok(rows)
        if not rs:
            continue
        xs = [r["value"] for r in rs]
        xlab, _ = axis_title[axis]
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))

        ax = axes[0]
        ax.plot(xs, [r["base_k1_us"] for r in rs], "o-", color=BASE_K1, label="baseline k1  ImplicitGemm", ms=5)
        ax.plot(xs, [r["md_k1_us"] for r in rs], "s--", color=MD_K1, label="MoDiff k1  ImplicitGemm", ms=5, alpha=0.85)
        ax.plot(xs, [r["base_k2_us"] for r in rs], "o-", color=BASE_K2, label="baseline k2  scale_store", ms=5)
        ax.plot(xs, [r["md_k2_us"] for r in rs], "s--", color=MD_K2, label="MoDiff k2  scale_accumulate", ms=5, alpha=0.85)
        ax.set_xlabel(xlab)
        ax.set_ylabel("µs / call")
        ax.set_title(f"Kernel time vs {axis}")
        if axis in ("B", "N", "H", "W") and min(xs) > 0 and max(xs) / min(xs) >= 4:
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.legend(fontsize=7.5, loc="upper left")

        ax = axes[1]
        ax.axhline(1.0, color="#cccccc", lw=1)
        ax.plot(xs, [r["k1_speedup"] for r in rs], "o-", color=BASE_K1, label="k1 speedup", ms=5)
        ax.plot(xs, [r["k2_speedup"] for r in rs], "o-", color=BASE_K2, label="k2 speedup", ms=5)
        ax.plot(xs, [r["evt_speedup"] for r in rs], "s--", color="#7c3aed", label="EVT fused", ms=5)
        ax.set_xlabel(xlab)
        ax.set_ylabel("baseline / MoDiff")
        ax.set_title(f"Speedup vs {axis}  (>1 MoDiff faster)")
        if axis in ("B", "N", "H", "W") and min(xs) > 0 and max(xs) / min(xs) >= 4:
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.legend(fontsize=7.5, loc="best")
        fig.suptitle(
            "INT8 conv  ·  3×3  ·  " +
            ", ".join(f"{k}={v}" for k, v in DEFAULT.items() if k != axis and not (axis == "N" and k == "C")),
            fontsize=9, color=MUTED, y=1.02)
        save(fig, f"sweep_{axis}.png")

    # ---- UNet shapes: grouped bars of k1/k2 ----
    urs = [r for r in data["unet_shapes"] if "error" not in r and r.get("base_k1_us", 0) > 0]
    if urs:
        labels = [r["shape"] for r in urs]
        x = list(range(len(urs)))
        fig, ax = plt.subplots(figsize=(12.5, 4.2))
        w = 0.38
        ax.bar([i - w / 2 for i in x], [r["base_k1_us"] for r in urs], w * 0.48,
               color=BASE_K1, label="base k1")
        ax.bar([i - w / 2 + w * 0.48 for i in x], [r["base_k2_us"] for r in urs], w * 0.48,
               color=BASE_K2, label="base k2")
        ax.bar([i + w / 2 - w * 0.48 for i in x], [r["md_k1_us"] for r in urs], w * 0.48,
               color="#93c5fd", label="MoDiff k1")
        ax.bar([i + w / 2 for i in x], [r["md_k2_us"] for r in urs], w * 0.48,
               color="#fbbf24", label="MoDiff k2")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("µs / call")
        ax.set_title("UNet 20 shapes  ·  B=128  ·  unfused k1 + k2")
        ax.legend(ncol=4, fontsize=8, loc="upper right")
        save(fig, "unet_times.png")

        fig, ax = plt.subplots(figsize=(12.5, 3.6))
        ax.axhline(1.0, color="#cccccc", lw=1)
        ax.plot(x, [r["k1_speedup"] for r in urs], "o-", color=BASE_K1, label="k1", ms=4)
        ax.plot(x, [r["k2_speedup"] for r in urs], "o-", color=BASE_K2, label="k2", ms=4)
        ax.plot(x, [r["evt_speedup"] for r in urs], "s--", color="#7c3aed", label="EVT", ms=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("baseline / MoDiff")
        ax.set_title("Per-shape speedup  ·  B=128")
        ax.legend(fontsize=8)
        save(fig, "unet_speedup.png")


if __name__ == "__main__":
    raise SystemExit(main())
