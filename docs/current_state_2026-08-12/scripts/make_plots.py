"""Current-state profile figures. Offline: reads this report's data/ (plus the e2e runs from
docs/postsplit_benchmark_2026-08-12/data/), no GPU.

ABSOLUTE state only -- no pre/post comparison anywhere in this report. Four views, coarse to fine:

  00  e2e per arm            which configuration costs what
  01  per attention block    all 21 blocks, four shape tiers, three W8A8 configurations
  02  per conv layer         all 70 called convs in UNet depth order
  03  per kernel             the current default's top kernels and bucket split

Instrument boundaries matter and are stated on each figure: the e2e number is profiler-free wall clock,
the per-layer/per-block numbers are CUDA events covering 0.63-0.89 of the step, and the per-kernel
numbers are a bucketed 8-step Perfetto trace. They do not sum to each other.

Palette copied from docs/profile_kernels_layers_2026-08-11/scripts/make_plots.py.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "docs/current_state_2026-08-12/data")
E2E = os.path.join(ROOT, "docs/postsplit_benchmark_2026-08-12/data")
PLOTS = os.path.join(ROOT, "docs/current_state_2026-08-12/plots")

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, PLUM, ROSE = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cc7", "#d64570"
BUCKET_COLOR = {"conv": BLUE, "norm_quantize": AQUA, "delta_quantize": PLUM,
                "linear_gemm": ORANGE, "attention": ROSE, "attn_quantize": "#c08a2e",
                "elementwise": INK3, "quantize": "#5aa9c4", "other": "#c9c7c1"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})

#: attention block index -> shape tier, from the churches UNet config
#: (model_channels 192, channel_mult [1,2,2,4,4], num_heads 8, attention_resolutions [1,2,4,8]).
TIERS = [("C192 T1024 hd24", {0, 1, 18, 19, 20}, ROSE),
         ("C384 T256 hd48", {2, 3, 15, 16, 17}, ORANGE),
         ("C384 T64 hd48", {4, 5, 12, 13, 14}, AQUA),
         ("C768 T16 hd96", {6, 7, 8, 9, 10, 11}, INK3)]
DEFAULT_ARM = "modiff_full_k4_projk4_qkvi8"
#: the three W8A8 conv+proj configurations, in the order the flags stack up
W8A8_LADDER = [("W8A8 conv+proj", "conv+proj", INK3),
               ("W8A8 conv+proj +projK4", "+ MODIFF_LINEAR_DELTA_REFRESH=4", BLUE),
               ("W8A8 conv+proj +projK4 +routeB", "+ MODIFF_FUSE_QKV_I8=1", ROSE)]
#: the configuration the tree runs by default with both flags set
CURRENT = W8A8_LADDER[-1][0]


def load(name):
    """Layer/trace data is this report's own; the e2e differential runs live in the report that
    produced them and are read from there rather than copied."""
    for d in (SRC, E2E):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def layer_row(cfg):
    rows = load("profile_layers.json") or []
    m = [r for r in rows if r["config"] == cfg]
    return m[0] if m else None


def plot_e2e(out):
    q, f = load("differential_timing_postsplit.json"), load("differential_timing_fp16_postsplit.json")
    if not (q and f):
        return False
    fp = next(iter(f["arms"].values()))["stats"]["median"] / 1e3 / f["steps"]
    order = ["fp16", "int8_ptq", "modiff_conv_k4", "modiff_conv_k1", "modiff_full_k1",
             "modiff_full_k4", "modiff_full_k4_projk4", DEFAULT_ARM]
    SHORT = {"fp16": "fp16", "int8_ptq": "W8A8 PTQ\n(no MoDiff)",
             "modiff_conv_k4": "MoDiff conv\nK=4", "modiff_conv_k1": "MoDiff conv\nK=1",
             "modiff_full_k1": "conv+proj K=1\n(paper)", "modiff_full_k4": "conv+proj\nK=4",
             "modiff_full_k4_projk4": "+ proj refresh\nK=4",
             DEFAULT_ARM: "+ int8 qkv\n-> flash"}
    ms = {"fp16": fp}
    ms.update({a: r["stats"]["median"] / 1e3 / q["steps"] for a, r in q["arms"].items()})
    arms = [a for a in order if a in ms]
    vals = [ms[a] for a in arms]
    cols = [INK3] + [BLUE] * (len(arms) - 1)
    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    ax.bar(range(len(arms)), vals, color=cols, width=0.62)
    for i, a in enumerate(arms):
        ax.text(i, vals[i] * 1.008, f"{vals[i]:.2f}\n{fp / vals[i]:.3f}x", ha="center",
                va="bottom", color=INK2, fontsize=9)
    ax.axhline(fp, color=ORANGE, linewidth=1.5, linestyle="--")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([SHORT[a] for a in arms], fontsize=8.5)
    ax.set_ylabel("ms/step")
    ax.set_title("End to end, batch 128, A40 -- profiler-free wall clock, 200 steps x 5 repeats "
                 "(CV <= 0.38%)", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(vals) * 1.20)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_blocks(out):
    """All 21 attention blocks in the current configuration, and what the two flags move.

    The right panel is the whole point: the flags act on the hd=48 tiers and nowhere else, which is
    exactly what the int8-qkv gate predicts (it needs hd % 16 == 0 and T % 64 == 0).
    """
    rows = {r["config"]: r for r in (load("profile_layers.json") or [])}
    if CURRENT not in rows:
        return False
    L = rows[CURRENT]["layers"]
    idx = sorted(int(k[4:]) for k in L if k.startswith("attn"))
    vals = [L[f"attn{i:02d}"] for i in idx]
    colors = [next((c for _, s_, c in TIERS if i in s_), INK3) for i in idx]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.6, 4.9),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    axl.bar(range(len(idx)), vals, color=colors, width=0.66)
    for i, v in enumerate(vals):
        if v > 1.0:
            axl.text(i, v * 1.01, f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=8)
    axl.set_xticks(range(len(idx)))
    axl.set_xticklabels([str(i) for i in idx], fontsize=8)
    axl.set_xlabel("attention block, UNet order (0-7 down, 8-10 middle, 11-20 up)")
    axl.set_ylabel("ms/step")
    axl.set_title(f"Per attention block -- {sum(vals):.2f} ms/step over 21 blocks", loc="left")
    axl.grid(axis="y", color=GRID, linewidth=0.8)
    axl.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in TIERS]
    axl.legend(handles, [f"{n} ({len(s_)})" for n, s_, _ in TIERS], fontsize=8.5,
               title="shape tier", title_fontsize=8.5)

    names = [n for n, _, _ in TIERS]
    w = 0.26
    for j, (cfg, lab, col) in enumerate(W8A8_LADDER):
        if cfg not in rows:
            continue
        Lj = rows[cfg]["layers"]
        tot = [sum(Lj[f"attn{i:02d}"] for i in s_) for _, s_, _ in TIERS]
        ys = [i + (j - 1) * w for i in range(len(names))]
        axr.barh(ys, tot, color=col, height=w * 0.9, label=f"{lab}  ({sum(tot):.2f} ms)")
        if j == len(W8A8_LADDER) - 1:
            # annotate the int8-qkv fusion alone -- against the row above it, not against conv+proj.
            # The refresh schedule moves every tier (it is a projection-side change and these timers
            # wrap the block's projections); the int8-qkv fusion is the tier-selective one.
            base = [sum(rows[W8A8_LADDER[-2][0]]["layers"][f"attn{i:02d}"] for i in s_)
                    for _, s_, _ in TIERS]
            for i, (t, b) in enumerate(zip(tot, base)):
                d = t - b
                axr.text(max(t, b) + 0.35, i, f"int8 qkv {d:+.2f}",
                         va="center", color=ROSE if abs(d) >= 0.05 else INK3, fontsize=8.5)
    axr.set_yticks(range(len(names)))
    axr.set_yticklabels(names, fontsize=8.5)
    axr.invert_yaxis()
    axr.set_xlabel("ms/step, summed over the tier")
    axr.set_title("The int8-qkv fusion moves the 10 hd48 blocks, nothing else", loc="left")
    axr.grid(axis="x", color=GRID, linewidth=0.8)
    axr.set_axisbelow(True)
    axr.set_xlim(0, 34)
    axr.legend(fontsize=8, loc="lower right")
    fig.suptitle("Batch 128 -- CUDA events on live dispatch targets (coverage 0.87-0.88). "
                 "Left: current configuration, both flags set.", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_conv_layers(out):
    """All called conv layers in depth order.

    Five of the eight configs (every MoDiff conv arm at W8A8 and W8A4) share one conv datapath and
    agree to <=0.055 ms on every single layer, so plotting them as eight curves is eight curves on top
    of each other. They are drawn as one min-max band instead, which is itself the finding.
    """
    rows = {r["config"]: r for r in (load("profile_layers.json") or [])}
    if not rows:
        return False
    SAME = ["W8A8 conv-only", "W8A8 conv+proj", "W8A8 conv+proj +projK4",
            "W8A8 conv+proj +projK4 +routeB", "W8A4 conv+proj"]
    SAME = [c for c in SAME if c in rows]

    def series(cfg):
        conv = {k: v for k, v in rows[cfg]["layers"].items() if k.startswith("conv")}
        return [conv[n] for n in sorted(conv, key=lambda n: int(n[4:]))]

    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    band = [series(c) for c in SAME]
    x = range(len(band[0]))
    lo = [min(col) for col in zip(*band)]
    hi = [max(col) for col in zip(*band)]
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.30, linewidth=0)
    ax.plot(x, [sum(col) / len(col) for col in zip(*band)], color=BLUE, linewidth=1.6,
            label=f"MoDiff conv datapath -- {len(SAME)} configs, W8A8 and W8A4 "
                  f"({sum(band[1]):.1f} ms, max spread 0.055)")
    for cfg, col in (("W8A8 PTQ", INK3), ("W4A4 conv+proj", PLUM)):
        if cfg in rows:
            v = series(cfg)
            ax.plot(range(len(v)), v, color=col, linewidth=1.5,
                    label=f"{cfg} ({sum(v):.1f} ms)")
    ax.set_xlabel("quantized conv layer, UNet depth order (high-res input blocks -> middle -> "
                  "high-res output blocks)")
    ax.set_ylabel("ms/step")
    ax.set_title("Per conv layer -- cost concentrates at the high-resolution ends, "
                 "the low-res middle is nearly free", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_kernels(out):
    d = load("trace_buckets_all.json")
    if not d or DEFAULT_ARM not in d["configs"]:
        return False
    c = d["configs"][DEFAULT_ARM]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.4, 5.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    top = sorted(c["kernels"].items(), key=lambda kv: -kv[1]["ms_per_step"])[:12]

    def short(n):
        if "ImplicitGemmConvolutionEVT" in n:
            return "cutlass ImplicitGemmConvolutionEVT"
        return n.split("(")[0][:44]
    seen = {}
    labels = []
    for n, v in top:
        s = short(n)
        seen[s] = seen.get(s, 0) + 1
        labels.append(f"{s} #{seen[s]}" if seen[s] > 1 else s)
    vals = [v["ms_per_step"] for _, v in top]
    cols = [BUCKET_COLOR.get(v["bucket"], INK3) for _, v in top]
    axl.barh(range(len(vals)), vals, color=cols, height=0.62)
    for i, (n, v) in enumerate(top):
        axl.text(vals[i] + 0.15, i, f"{vals[i]:.2f} ({v['calls_per_step']:.0f} calls)",
                 va="center", color=INK2, fontsize=8.5)
    axl.set_yticks(range(len(vals)))
    axl.set_yticklabels(labels, fontsize=8)
    axl.invert_yaxis()
    axl.set_xlabel("ms/step")
    axl.set_title("Top 12 kernels", loc="left")
    axl.grid(axis="x", color=GRID, linewidth=0.8)
    axl.set_axisbelow(True)
    axl.set_xlim(0, max(vals) * 1.42)

    bk = sorted(c["buckets"].items(), key=lambda kv: -kv[1]["ms_per_step"])
    axr.barh(range(len(bk)), [v["ms_per_step"] for _, v in bk],
             color=[BUCKET_COLOR.get(k, INK3) for k, _ in bk], height=0.6)
    for i, (k, v) in enumerate(bk):
        axr.text(v["ms_per_step"] + 0.25, i,
                 f"{v['ms_per_step']:.2f}  ({v['calls_per_step']:.0f} calls, "
                 f"{v['kernels']} kernels)", va="center", color=INK2, fontsize=8.5)
    axr.set_yticks(range(len(bk)))
    axr.set_yticklabels([k for k, _ in bk], fontsize=8.5)
    axr.invert_yaxis()
    axr.set_xlabel("ms/step")
    axr.set_title(f"By bucket -- {c['gpu_ms_per_step']:.1f} ms/step of GPU time", loc="left")
    axr.grid(axis="x", color=GRID, linewidth=0.8)
    axr.set_axisbelow(True)
    axr.set_xlim(0, max(v["ms_per_step"] for _, v in bk) * 1.95)
    fig.suptitle("Per kernel, current configuration (MoDiff conv+proj K=4, projection refresh K=4, "
                 "route b) -- batch 128, 8-step trace", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def main():
    os.makedirs(PLOTS, exist_ok=True)
    n = 0
    for fn, name in ((plot_e2e, "00_e2e_by_arm.png"),
                     (plot_blocks, "01_per_attention_block.png"),
                     (plot_conv_layers, "02_per_conv_layer.png"),
                     (plot_kernels, "03_per_kernel.png")):
        if fn(os.path.join(PLOTS, name)):
            n += 1
        else:
            print(f"NOTE: skipped {name}")
    print(f"wrote {n} plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
