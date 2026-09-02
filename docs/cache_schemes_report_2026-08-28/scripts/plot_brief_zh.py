"""Chinese figures for the cache-schemes brief PDF."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = "/workspace/MoDiff"
OUT = os.path.join(ROOT, "docs/cache_schemes_report_2026-08-28/plots")
os.makedirs(OUT, exist_ok=True)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(FONT_B)
mpl.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "font.sans-serif": ["Noto Sans CJK JP"],
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

INK = "#1a1a1a"
MUTED = "#666666"
REPLAY = "#1f6feb"
SKIP = "#8b949e"
QUANT = "#d97706"
DROP = "#b42318"
KEEP = "#1a7f37"
W4 = "#7c3aed"
FULL = "#2563eb"

W8 = 93.44  # W8A8 full fp16 a_hat ms/step


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_speedup():
    """Ranked e2e speedup vs W8A8 full."""
    rows = [
        ("W8 replay-K=8  fp16", 61.47, KEEP),
        ("W4 replay-K=4  fp16", 64.75, KEEP),
        ("W4 replay-K=4  int4", 65.88, KEEP),
        ("W8 replay-K=4  fp16", 66.43, KEEP),
        ("W8 replay-K=4  int8", 67.75, KEEP),
        ("W8 replay-K=4  int4", 67.84, KEEP),
        ("W8 replay-K=2  fp16", 74.77, KEEP),
        ("W4 skip-K=4  fp16", 86.92, SKIP),
        ("W4 full  fp16", 88.59, W4),
        ("W4 skip-K=4  int4", 88.47, SKIP),
        ("W4 full  int4", 89.87, QUANT),
        ("W8 skip-K=4  fp16", 92.19, SKIP),
        ("W8 full  fp16（基线）", 93.44, FULL),
        ("W8 skip-K=4  int8", 93.86, SKIP),
        ("W8 full  int8 held", 94.35, QUANT),
        ("W8 full  int4 held", 95.11, DROP),
        ("W8 full  int8 refresh", 186.13, DROP),
    ]
    labels = [r[0] for r in rows][::-1]
    xs = [W8 / r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    y = np.arange(len(labels))
    ax.barh(y, xs, color=colors, height=0.72, edgecolor="none")
    ax.axvline(1.0, color=INK, lw=0.8, ls="--")
    ax.axvline(1.20, color=KEEP, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("相对 W8A8 full fp16 a_hat 的加速比（93.4 ms/step）")
    ax.set_xlim(0, 1.75)
    ax.set_title("端到端加速比（A40 · batch 128 · 50 DDIM · 仅 UNet）")
    for yi, x in zip(y, xs):
        ax.text(x + 0.02, yi, f"{x:.2f}×", va="center", fontsize=8, color=INK)
    ax.text(1.01, len(labels) - 0.35, "基线 1.00×", fontsize=8, color=MUTED)
    ax.text(1.21, 0.15, "保留线 1.20×", fontsize=8, color=KEEP)
    fig.tight_layout()
    save(fig, "fig01_speedup.png")


def fig_scheme_grid():
    """Grouped speedup: scheme × a_hat bits, W8 and W4."""
    schemes = ["full", "skip-K=4", "replay-K=4"]
    # W8: fp16, int8, int4
    w8 = {
        "fp16":     [93.44, 92.19, 66.43],
        "int8 held": [94.35, 93.86, 67.75],
        "int4 held": [95.11, 94.07, 67.84],
    }
    w4 = {
        "fp16":     [88.59, 86.92, 64.75],
        "int4 held": [89.87, 88.47, 65.88],
    }
    x = np.arange(len(schemes))
    w = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharey=True)

    ax = axes[0]
    for i, (name, ms) in enumerate(w8.items()):
        xs = [W8 / m for m in ms]
        ax.bar(x + (i - 1) * w, xs, w, label=name, color=[FULL, QUANT, DROP][i], edgecolor="none")
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.set_title("W8A8")
    ax.set_ylabel("加速比（相对 W8A8 full fp16）")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.7)

    ax = axes[1]
    offs = [-0.5 * w, 0.5 * w]
    for i, (name, ms) in enumerate(w4.items()):
        xs = [W8 / m for m in ms]
        ax.bar(x + offs[i], xs, w, label=name, color=[W4, DROP][i], edgecolor="none")
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.set_title("W4A4（无 int8 a_hat 测量）")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("skip / replay × a_hat 量化  加速比网格", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "fig02_scheme_grid.png")


def fig_fid_pareto():
    """Speedup vs FID — the ranking plot."""
    pts = [
        ("full fp16", 1.00, 0.92, FULL, "o"),
        ("skip-K=4", 1.01, 2.68, SKIP, "s"),
        ("skip-K=4+int8", 1.00, 10.77, QUANT, "s"),
        ("replay-K=2", 1.25, 5.40, KEEP, "o"),
        ("replay-K=4", 1.42, 16.35, REPLAY, "o"),
        ("replay-K=4+int8", 1.38, 24.03, QUANT, "D"),
        ("replay-K=8", 1.52, 65.11, DROP, "o"),
        ("int8 held", 0.99, 121.31, DROP, "X"),
        ("skip-K=8", 1.02, 7.97, SKIP, "^"),  # ceiling ~1.02, untimed
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for name, sp, fid, c, m in pts:
        ax.scatter(sp, fid, s=70, c=c, marker=m, zorder=3, edgecolors=INK, linewidths=0.4)
        dy = 4 if name != "int8 held" else -10
        ax.annotate(name, (sp, fid), textcoords="offset points", xytext=(7, dy),
                    fontsize=8, color=INK)
    ax.axvline(1.20, color=KEEP, lw=0.8, ls=":")
    ax.set_xlabel("加速比（相对 W8A8 full fp16 a_hat）")
    ax.set_ylabel("FID vs fp16（N=2048，同 seed）")
    ax.set_title("速度–质量：只有 replay-K=2 同时加速且 FID 仍接近 MoDiff")
    ax.set_xlim(0.92, 1.62)
    ax.set_ylim(-2, 135)
    ax.text(1.205, 128, "加速 ≥1.20×", fontsize=8, color=KEEP)
    fig.tight_layout()
    save(fig, "fig03_fid_pareto.png")


def fig_fid_bars():
    labels = ["full fp16", "skip-K=4", "replay-K=2", "skip-K=8",
              "skip-K=4+int8", "replay-K=4", "replay-K=4+int8", "replay-K=8", "int8 held"]
    fid = [0.92, 2.68, 5.40, 7.97, 10.77, 16.35, 24.03, 65.11, 121.31]
    rel = [0.12, 0.16, 0.19, 0.33, 0.26, 0.29, 0.34, 0.40, 0.69]
    colors = [FULL, SKIP, KEEP, SKIP, QUANT, REPLAY, QUANT, DROP, DROP]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(labels))
    bars = ax1.bar(x, fid, color=colors, width=0.62, edgecolor="none")
    ax1.set_ylabel("FID vs fp16")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=22, ha="right", fontsize=8.5)
    ax1.set_ylim(0, 140)
    ax2 = ax1.twinx()
    ax2.plot(x, rel, color=INK, marker="o", lw=1.4, ms=5, label="relL2（n=6）")
    ax2.set_ylabel("latent relL2 vs fp16")
    ax2.set_ylim(0, 0.85)
    ax2.grid(False)
    ax1.set_title("FID 与 relL2 在 replay-K=4 上唱反调")
    # mark the disagreement
    ax1.annotate("relL2 认为 K=4 优于 skip-K=8\nFID 相反（16.3 vs 8.0）",
                 xy=(5, 16.35), xytext=(3.1, 78),
                 fontsize=8, color=DROP,
                 arrowprops=dict(arrowstyle="->", color=DROP, lw=0.8))
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig04_fid_rell2.png")


def fig_cost_scope():
    """Why e2e replay is only 1.42×: leftover 61 ms."""
    labels = ["单层 200 step", "残差 conv 集合", "端到端 UNet"]
    skip = [1.07, 0.99, 1.01]
    replay = [3.97, 2.65, 1.42]
    x = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w / 2, skip, w, label="skip-K=4", color=SKIP, edgecolor="none")
    ax.bar(x + w / 2, replay, w, label="replay-K=4", color=REPLAY, edgecolor="none")
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("加速比（相对同范围的 full）")
    ax.set_title("从单层到 e2e：replay 被 61 ms 剩余路径稀释")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 4.5)
    for i, v in enumerate(skip):
        ax.text(i - w / 2, v + 0.08, f"{v:.2f}×", ha="center", fontsize=8)
    for i, v in enumerate(replay):
        ax.text(i + w / 2, v + 0.08, f"{v:.2f}×", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig05_cost_scope.png")


def fig_one_layer():
    """Single OptimizedInt8Conv2d, 200 modulated steps."""
    rows = [
        ("replay-K=8", 8.04, REPLAY),
        ("replay-K=4", 3.97, REPLAY),
        ("replay-K=4 int8", 4.28, QUANT),
        ("replay-K=2", 1.98, REPLAY),
        ("skip-K=8", 1.08, SKIP),
        ("skip-K=4", 1.07, SKIP),
        ("skip-K=2", 1.04, SKIP),
        ("int8 held", 1.07, QUANT),
        ("full fp16", 1.00, FULL),
        ("int8 refresh", 0.27, DROP),
    ]
    labels = [r[0] for r in rows][::-1]
    xs = [r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(labels))
    ax.barh(y, xs, color=colors, height=0.72, edgecolor="none")
    ax.axvline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("相对 full fp16 a_hat 的加速比")
    ax.set_title("单层 conv（192→192，32×32，batch 128，200 step）")
    ax.set_xlim(0, 9.2)
    for yi, x in zip(y, xs):
        ax.text(x + 0.08, yi, f"{x:.2f}×", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig07_one_layer.png")


def fig_conv_set():
    """Freq-weighted residual conv path, 20 UNet shapes."""
    labels = ["full", "skip-K=4", "replay-K=2", "replay-K=4", "replay-K=8", "int8 refresh"]
    ms = [32.47, 32.68, 18.98, 12.24, 8.86, 117.29]
    xs = [32.47 / m for m in ms]
    colors = [FULL, SKIP, REPLAY, REPLAY, REPLAY, DROP]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(labels))
    ax.bar(x, xs, color=colors, width=0.62, edgecolor="none")
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("加速比（相对 conv-full 32.5 ms）")
    ax.set_title("残差 conv 集合（20 个 UNet 形状，频次加权）")
    ax.set_ylim(0, 4.2)
    for i, v in enumerate(xs):
        ax.text(i, v + 0.08, f"{v:.2f}×", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig08_conv_set.png")


def fig_step_stack():
    """32 ms conv + 61 ms leftover."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    arms = ["full", "skip-K=4", "replay-K=4", "replay-K=8"]
    conv = [32.5, 32.7, 12.2, 8.9]   # replay: (1/K)*full + ((K-1)/K)*replay primitive
    # leftover ≈ e2e - conv
    e2e = [93.4, 92.2, 66.0, 61.5]
    leftover = [e - c for e, c in zip(e2e, conv)]
    x = np.arange(len(arms))
    ax.bar(x, leftover, 0.55, label="剩余（attention / skip 1×1 / emb）", color="#d0d7de", edgecolor="none")
    ax.bar(x, conv, 0.55, bottom=leftover, label="残差 conv（replay 能砍的）", color=REPLAY, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("ms / step")
    ax.set_title("一步 UNet：replay 只能砍掉残差 conv，~61 ms 仍在")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 115)
    for i, (e, c, L) in enumerate(zip(e2e, conv, leftover)):
        ax.text(i, e + 2.2, f"{e:.0f} ms", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig06_step_stack.png")


def fig_reuse_o_hat():
    """Copy / add primitives vs full, one-layer and conv-set."""
    labels = ["full", "拷贝 reuse_o_hat", "add reuse_o_hat_add", "aten add"]
    one = [1.00, 6.29, 4.27, 4.23]
    conv = [1.00, 8.67, 5.91, 5.87]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w / 2, one, w, label="单层 192→192 32×32", color=REPLAY, edgecolor="none")
    ax.bar(x + w / 2, conv, w, label="残差 conv 集合", color=FULL, edgecolor="none")
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("相对同范围 full 的加速比")
    ax.set_title("reuse_o_hat 核（batch 128；拷贝 ≠ 视图）")
    ax.set_ylim(0, 10.2)
    ax.legend(loc="upper left", fontsize=8)
    for i, (a, b) in enumerate(zip(one, conv)):
        ax.text(i - w / 2, a + 0.12, f"{a:.2f}×", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.12, f"{b:.2f}×", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig09_reuse_o_hat.png")


if __name__ == "__main__":
    fig_one_layer()
    fig_conv_set()
    fig_speedup()
    fig_scheme_grid()
    fig_fid_pareto()
    fig_fid_bars()
    fig_cost_scope()
    fig_step_stack()
    fig_reuse_o_hat()
    print("done")
