"""Plots for docs/skip_k_pipeline_2026-08-31. No GPU."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
D = "docs/skip_k_pipeline_2026-08-31"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID = "#e3e2df"
BUCKETS = ["GroupNorm+SiLU family", "GEMM / conv", "attention",
           "elementwise / copy", "other"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.bbox": "tight",
})

MODE_ORDER = ["fp16", "PTQ", "K=1", "K=2", "K=4", "K=5", "K=10", "K=20", "K=50", "K=100"]
MODE_LABS = {"fp16": "fp16", "PTQ": "W8A8 PTQ", "K=1": "MoDiff K=1",
             "K=2": "skip K=2", "K=4": "skip K=4", "K=5": "skip K=5",
             "K=10": "skip K=10", "K=20": "skip K=20", "K=50": "skip K=50",
             "K=100": "skip K=100"}


def load(name):
    p = os.path.join(D, "data", name)
    return json.load(open(p)) if os.path.exists(p) else None


def plot_e2e(e2e):
    labs, vals, sp = [], [], []
    for m in MODE_ORDER:
        if m not in e2e["modes"]:
            continue
        labs.append(MODE_LABS[m])
        vals.append(e2e["modes"][m]["per_step_ms"])
        sp.append(e2e["modes"][m].get("speedup_vs_fp16"))
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    y = range(len(labs))
    ax.barh(list(y), vals, height=0.62, color=SERIES[0])
    ax.set_yticks(list(y), labs)
    ax.invert_yaxis()
    ax.set_xlabel("ms per denoising step   (batch 128, DDIM 50, A40)")
    ax.set_xlim(0, max(vals) * 1.32)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, (v, s) in enumerate(zip(vals, sp)):
        t = f"{v:.2f} ms" + (f"   {s:.2f}× vs fp16" if s and abs(s - 1.0) > 1e-6 else "")
        ax.text(v + max(vals) * 0.015, i, t, va="center", ha="left", fontsize=9, color=INK2)
    ax.set_title("End-to-end latency — production W8A8 pipeline", loc="left", pad=10)
    fig.savefig(f"{D}/plots/01_e2e.png")
    plt.close(fig)
    print("wrote 01_e2e.png")


def plot_k_curve(e2e, q):
    ks, ms, rel = [], [], []
    for k in e2e.get("K", [1, 2, 4, 5, 10, 20, 50, 100]):
        lab = f"K={k}"
        if lab not in e2e["modes"]:
            continue
        ks.append(k)
        ms.append(e2e["modes"][lab]["per_step_ms"])
        key = lab
        rel.append(q["relL2"][key]["relL2_vs_fp16"] if q and key in q.get("relL2", {}) else None)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(ks, ms, marker="o", color=SERIES[0], lw=1.8, ms=7)
    ptq = e2e["modes"].get("PTQ", {}).get("per_step_ms")
    if ptq:
        ax.axhline(ptq, color=SERIES[1], ls="--", lw=1.2, label=f"PTQ {ptq:.2f} ms")
        ax.legend(frameon=False)
    ax.set_xlabel("skip K")
    ax.set_ylabel("ms / step")
    ax.set_xticks(ks)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("E2E vs skip cadence  (K=1 writes every step)", loc="left", pad=10)
    fig.savefig(f"{D}/plots/02_e2e_vs_k.png")
    plt.close(fig)
    print("wrote 02_e2e_vs_k.png")

    if all(r is not None for r in rel):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(ks, rel, marker="o", color=SERIES[2], lw=1.8, ms=7)
        ax.set_xlabel("skip K")
        ax.set_ylabel("latent relL2 vs fp16")
        ax.set_xticks(ks)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title("Quality vs skip cadence  (n=6, seed 20260805)", loc="left", pad=10)
        fig.savefig(f"{D}/plots/03_quality_vs_k.png")
        plt.close(fig)
        print("wrote 03_quality_vs_k.png")


def plot_tradeoff(e2e, q):
    pts = []
    for m in MODE_ORDER:
        if m not in e2e["modes"] or m not in q.get("relL2", {}):
            continue
        pts.append((MODE_LABS[m], e2e["modes"][m]["per_step_ms"],
                    q["relL2"][m]["relL2_vs_fp16"]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for lab, ms, rel in pts:
        ax.scatter([ms], [rel], s=70, color=SERIES[0], zorder=3,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.annotate(f"{lab}\n{rel:.3f} @ {ms:.1f} ms", (ms, rel), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=INK2)
    ax.set_xlabel("ms per denoising step  (lower is faster)")
    ax.set_ylabel("latent relL2 vs fp16  (lower is closer)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(min(p[1] for p in pts) * 0.92, max(p[1] for p in pts) * 1.12)
    ymax = max(p[2] for p in pts)
    ax.set_ylim(-0.02, ymax * 1.22 if ymax > 0 else 0.2)
    ax.set_title("Speed against fidelity", loc="left", pad=10)
    fig.savefig(f"{D}/plots/04_quality_vs_speed.png")
    plt.close(fig)
    print("wrote 04_quality_vs_speed.png")


def plot_kernels(e2e):
    order, data = [], {}
    for m in MODE_ORDER:
        d = e2e["modes"].get(m)
        if not d or not d.get("buckets_ms_per_step"):
            continue
        data[MODE_LABS[m]] = d["buckets_ms_per_step"]
        order.append(MODE_LABS[m])
    if not order:
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    y = range(len(order))
    left = [0.0] * len(order)
    buckets = [b for b in BUCKETS if any(data[l].get(b, 0) > 0.05 for l in order)]
    for i, b in enumerate(buckets):
        vals = [data[l].get(b, 0.0) for l in order]
        ax.barh(list(y), vals, left=left, height=0.62, color=SERIES[i % len(SERIES)],
                label=b)
        left = [a + b_ for a, b_ in zip(left, vals)]
    ax.set_yticks(list(y), order)
    ax.invert_yaxis()
    ax.set_xlabel("ms per denoising step")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    ax.set_title("GPU time by kernel bucket", loc="left", pad=10)
    fig.savefig(f"{D}/plots/05_buckets.png")
    plt.close(fig)
    print("wrote 05_buckets.png")


def plot_layer(layer):
    if not layer:
        return
    ks = [r["skip_k"] for r in layer["arms"]]
    ms = [r["ms_step"] * 1000 for r in layer["arms"]]  # µs
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(ks, ms, marker="o", color=SERIES[0], lw=1.8, ms=7)
    ax.set_xlabel("skip K")
    ax.set_ylabel("µs / step  (one 192→192 32×32 conv)")
    ax.set_xticks(ks)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Single-layer OptimizedInt8Conv2d vs K", loc="left", pad=10)
    fig.savefig(f"{D}/plots/06_layer.png")
    plt.close(fig)
    print("wrote 06_layer.png")


def main():
    e2e, q, layer = load("e2e.json"), load("quality.json"), load("layer.json")
    if not e2e:
        sys.exit("missing data/e2e.json")
    plot_e2e(e2e)
    plot_k_curve(e2e, q)
    if q:
        plot_tradeoff(e2e, q)
    plot_kernels(e2e)
    plot_layer(layer)


if __name__ == "__main__":
    main()
