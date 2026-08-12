"""Current-state profile figures. Offline: reads docs/postsplit_benchmark_2026-08-12/data/, no GPU.

ABSOLUTE state only -- no pre/post comparison anywhere in this report. Four views, coarse to fine:

  00  e2e per arm            which configuration costs what
  01  per attention block    all 21 blocks, and the four shape tiers they fall into
  02  per conv layer         all 70 called convs in UNet depth order
  03  per kernel             the current default's top kernels and bucket split

Instrument boundaries matter and are stated on each figure: the e2e number is profiler-free wall clock,
the per-layer/per-block numbers are CUDA events covering 0.64-0.88 of the step, and the per-kernel
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
SRC = os.path.join(ROOT, "docs/postsplit_benchmark_2026-08-12/data")
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


def load(name):
    p = os.path.join(SRC, name)
    return json.load(open(p)) if os.path.exists(p) else None


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
             "modiff_full_k4_projk4": "+proj refresh\n(opt-in)",
             DEFAULT_ARM: "+route (b)\n(opt-in)"}
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
    """All 21 attention blocks, coloured by shape tier. The four tiers are the whole story."""
    r = layer_row("W8A8 conv+proj")
    if not r:
        return False
    L = r["layers"]
    idx = sorted(int(k[4:]) for k in L if k.startswith("attn"))
    vals = [L[f"attn{i:02d}"] for i in idx]
    colors = []
    for i in idx:
        colors.append(next((c for _, s, c in TIERS if i in s), INK3))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.0, 4.7),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    axl.bar(range(len(idx)), vals, color=colors, width=0.66)
    for i, v in enumerate(vals):
        if v > 1.0:
            axl.text(i, v * 1.01, f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=8)
    axl.set_xticks(range(len(idx)))
    axl.set_xticklabels([str(i) for i in idx], fontsize=8)
    axl.set_xlabel("attention block, UNet order (0-7 down, 8-10 middle, 11-20 up)")
    axl.set_ylabel("ms/step")
    axl.set_title("Per attention block -- 42.85 ms/step over 21 blocks", loc="left")
    axl.grid(axis="y", color=GRID, linewidth=0.8)
    axl.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in TIERS]
    axl.legend(handles, [n for n, _, _ in TIERS], fontsize=8.5, title="shape tier",
               title_fontsize=8.5)

    names = [n for n, _, _ in TIERS]
    tot = [sum(L[f"attn{i:02d}"] for i in s) for _, s, _ in TIERS]
    cnt = [len(s) for _, s, _ in TIERS]
    axr.barh(range(len(names)), tot, color=[c for _, _, c in TIERS], height=0.55)
    for i, (t, n) in enumerate(zip(tot, cnt)):
        axr.text(t + 0.4, i, f"{t:.2f} ms  ({n} blocks, {100 * t / sum(tot):.0f}%)",
                 va="center", color=INK2, fontsize=9)
    axr.set_yticks(range(len(names)))
    axr.set_yticklabels(names, fontsize=8.5)
    axr.invert_yaxis()
    axr.set_xlabel("ms/step, summed over the tier")
    axr.set_title("5 hd24 blocks are 62% of attention", loc="left")
    axr.grid(axis="x", color=GRID, linewidth=0.8)
    axr.set_axisbelow(True)
    axr.set_xlim(0, max(tot) * 1.75)
    fig.suptitle("W8A8 conv+proj, batch 128 -- CUDA events on live dispatch targets (coverage 0.88)",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_conv_layers(out):
    """All called conv layers in depth order, for the arms that have them."""
    rows = load("profile_layers.json") or []
    fig, ax = plt.subplots(figsize=(12.0, 4.7))
    for r, c in zip([x for x in rows if x.get("layers")], (INK3, AQUA, BLUE, PLUM, ROSE)):
        conv = {k: v for k, v in r["layers"].items() if k.startswith("conv")}
        names = sorted(conv, key=lambda n: int(n[4:]))
        ax.plot(range(len(names)), [conv[n] for n in names], linewidth=1.5, color=c,
                label=f"{r['config']} ({sum(conv.values()):.1f} ms)")
    ax.set_xlabel("quantized conv layer, UNet depth order (high-res input blocks -> middle -> "
                  "high-res output blocks)")
    ax.set_ylabel("ms/step")
    ax.set_title("Per conv layer -- cost concentrates at the high-resolution ends, "
                 "the low-res middle is nearly free", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_kernels(out):
    d = load("trace_buckets_postsplit.json")
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
    fig.suptitle("Per kernel: current default + both opt-ins (MoDiff conv+proj K=4, proj refresh, "
                 "route b), batch 128, 8-step trace", x=0.01, ha="left", fontsize=11.5)
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
