"""Figures for the post-split re-measurement. Offline: reads data/*.json, writes plots/*.png, no GPU.

The question this report answers is narrow and worth stating: the csrc/ split into baseline/ and
modiff/ trees claims to have moved code without changing behaviour. The SASS gate proved the device
code is byte-identical and the export manifest proved nothing vanished, but neither says anything about
SPEED -- duplicated CUTLASS instantiations could in principle shift occupancy or cache behaviour. So
every arm, layer and kernel bucket is re-measured against its pre-split record.

Three instruments, and they are not interchangeable (same rule as
docs/profile_kernels_layers_2026-08-11):
  * DIFFERENTIAL, profiler-free wall clock per arm -- authoritative for end-to-end ms/step.
  * PER-LAYER CUDA events on the live dispatch targets -- authoritative for "which layer", and only
    for shares: its coverage is a fraction of the step, so its totals are not the e2e number.
  * PER-KERNEL Perfetto traces bucketed offline -- authoritative for "which CUDA kernel", within a
    capture. Never compare totals across captures; two captures of the same arm drift ~1 ms.

Palette and rcParams copied from docs/profile_kernels_layers_2026-08-11/scripts/make_plots.py so every
report's figures read as one set.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.join(ROOT, "docs/postsplit_benchmark_2026-08-12")
DATA, PLOTS = os.path.join(HERE, "data"), os.path.join(HERE, "plots")

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

#: Pre-split reference, ms/step, from the records this report re-measures against.
#: docs/profile_kernels_layers_2026-08-11/data/ (fp16 .. modiff_full_k4) and
#: docs/aq_fusion_2026-08-12/data/differential_timing_qkvi8.json (the two opt-in arms).
PRE = {"fp16": 106.09, "int8_ptq": 73.31, "modiff_conv_k4": 77.33, "modiff_conv_k1": 83.01,
       "modiff_full_k1": 105.42, "modiff_full_k4": 99.73,
       "modiff_full_k4_projk4": 95.64, "modiff_full_k4_projk4_qkvi8": 94.88}
ORDER = ["fp16", "int8_ptq", "modiff_conv_k4", "modiff_conv_k1", "modiff_full_k1",
         "modiff_full_k4", "modiff_full_k4_projk4", "modiff_full_k4_projk4_qkvi8"]
SHORT = {"fp16": "fp16", "int8_ptq": "W8A8 PTQ", "modiff_conv_k4": "conv K=4",
         "modiff_conv_k1": "conv K=1", "modiff_full_k1": "conv+proj\nK=1",
         "modiff_full_k4": "conv+proj\nK=4",
         "modiff_full_k4_projk4": "+proj K=4\n(opt-in)",
         "modiff_full_k4_projk4_qkvi8": "+route (b)\n(opt-in)"}


def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def e2e_ms():
    """{arm: ms/step} merged from the quantized-ladder run and the separate fp16 run."""
    out = {}
    for fn in ("differential_timing_postsplit.json", "differential_timing_fp16_postsplit.json"):
        d = load(fn)
        if d is None:
            continue
        for arm, rec in d["arms"].items():
            out[arm] = rec["stats"]["median"] / 1e3 / d["steps"]
    return out


def plot_e2e(out):
    """Left: absolute ms/step, post vs the pre-split record. Right: the delta, which is a STEP
    aligned to which FILE each reference came from -- not a trend, and not a code effect.

    The right panel is the point of this figure. Arms 1-5 all read ~1.3 ms faster than their record
    and arms 6-7 read level, and the break is exactly where the reference changes source: arms 1-5 are
    compared against the 2026-08-11 canonical run (a previous session) while arms 6-7 are compared
    against a run made EARLIER THE SAME DAY, in this container. Same-session references reproduce to
    within 0.21 ms; a previous-session reference is offset by ~1.3 ms for every arm alike -- including
    int8_ptq, which contains no MoDiff code at all and which the split cannot have touched.

    So the absolute columns measure session offset, not the split, and the quantity that survives is
    the delta BETWEEN arms inside this one run (see FINDINGS). fp16 is drawn hollow: it ran in its own
    process, so it has no position in the ladder.
    """
    ms = e2e_ms()
    arms = [a for a in ORDER if a in ms]
    if not arms:
        return False
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.2, 4.9),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    x = range(len(arms))
    w = 0.38
    pre = [PRE[a] for a in arms]
    post = [ms[a] for a in arms]
    axl.bar([i - w / 2 for i in x], pre, w, color=INK3, label="pre-split (recorded)")
    axl.bar([i + w / 2 for i in x], post, w, color=BLUE, label="post-split (this run)")
    fp16 = ms.get("fp16")
    for i, a in enumerate(arms):
        axl.text(i + w / 2, post[i] * 1.008, f"{(fp16 / post[i]):.3f}x" if fp16 else "",
                 ha="center", va="bottom", color=INK2, fontsize=8.5)
    if fp16:
        axl.axhline(fp16, color=ORANGE, linewidth=1.4, linestyle="--")
        axl.text(len(arms) - 0.5, fp16 * 1.015, f"fp16 {fp16:.1f}", color=ORANGE, va="bottom",
                 ha="right", fontsize=9)
    axl.set_xticks(list(x))
    axl.set_xticklabels([SHORT[a] for a in arms], fontsize=8.5)
    axl.set_ylabel("ms/step (profiler-free, 200 steps x 5 repeats)")
    axl.set_title("Absolute, with speedup vs this run's own fp16", loc="left")
    axl.grid(axis="y", color=GRID, linewidth=0.8)
    axl.set_axisbelow(True)
    axl.set_ylim(0, max(pre + post) * 1.15)
    axl.legend(loc="lower left", fontsize=9)

    # Right: delta vs run position. The quantized ladder ran in one process, in ARMS order.
    quant = [a for a in arms if a != "fp16"]
    pos = list(range(1, len(quant) + 1))
    dq = [ms[a] - PRE[a] for a in quant]
    axr.axhline(0, color=INK3, linewidth=1.0)
    axr.plot(pos, dq, "o-", color=BLUE, markersize=8, linewidth=1.4, label="quantized ladder")
    if fp16:
        axr.plot([1], [ms["fp16"] - PRE["fp16"]], "o", markerfacecolor="none",
                 markeredgecolor=ORANGE, markersize=9, markeredgewidth=1.8,
                 label="fp16 (separate process)")
    for i, a in enumerate(quant):
        axr.text(pos[i], dq[i] + 0.06, f"{dq[i]:+.2f}", ha="center", va="bottom",
                 color=INK2, fontsize=8)
    axr.set_xticks(pos)
    axr.set_xticklabels([SHORT[a].replace("\n", " ") for a in quant], rotation=35,
                        ha="right", fontsize=8)
    axr.axvspan(0.5, 5.5, color=INK3, alpha=0.07)
    axr.text(3.0, max(dq) * 0.55 if max(dq) > 0 else -0.2,
             "reference: 2026-08-11 run\n(previous session)", ha="center", color=INK3, fontsize=8.5)
    axr.text(6.5, min(dq) * 0.55, "reference: earlier\nTODAY, same container",
             ha="center", color=BLUE, fontsize=8.5)
    axr.set_ylabel("post - pre (ms/step)")
    axr.set_xlabel("arm (ladder order)")
    axr.set_title("A STEP, not a trend: it tracks which session the reference came from", loc="left")
    axr.grid(axis="y", color=GRID, linewidth=0.8)
    axr.set_axisbelow(True)
    axr.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_layers(out):
    """Per-conv-layer time in UNet depth order. Shares only -- see the module docstring."""
    rows = load("profile_layers.json")
    if not rows:
        return False
    rows = [r for r in rows if r.get("layers")]
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    for r, c in zip(rows, (INK3, BLUE, AQUA, PLUM, ROSE, ORANGE)):
        names = sorted(r["layers"], key=lambda n: int("".join(ch for ch in n if ch.isdigit()) or 0))
        vals = [r["layers"][n] for n in names]
        ax.plot(range(len(vals)), vals, linewidth=1.4, color=c, label=r["config"])
    ax.set_xlabel("quantized conv layer, UNet depth order (input blocks -> middle -> output blocks)")
    ax.set_ylabel("ms/step")
    ax.set_title("Per conv layer -- cost concentrates in the high-resolution input/output blocks",
                 loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_model(out):
    """Whole-model wall from the layer harness, per config. NOT the e2e number: this harness
    instruments dispatch targets and its wall includes the profiling overhead."""
    rows = load("profile_layers.json")
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels = [r["config"] for r in rows]
    vals = [r["wall_ms_per_step"] for r in rows]
    ax.bar(labels, vals, color=BLUE, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.01, f"{v:.1f}", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_ylabel("wall ms/step (layer harness)")
    ax.set_title("Whole model, layer harness -- compare shapes across bars, not against the e2e table",
                 loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_buckets(out):
    """Per-kernel bucket breakdown for the traced arms, stacked."""
    d = load("trace_buckets_postsplit.json")
    if d is None:
        return False
    arms = [a for a in ("int8_ptq", "modiff_full_k4_projk4", "modiff_full_k4_projk4_qkvi8")
            if a in d["configs"]]
    keys = []
    for a in arms:
        for k in d["configs"][a]["buckets"]:
            if k not in keys:
                keys.append(k)
    keys = [k for k in ("conv", "delta_quantize", "linear_gemm", "elementwise", "attention",
                        "norm_quantize", "attn_quantize", "quantize", "other") if k in keys]
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    bottoms = [0.0] * len(arms)
    for k in keys:
        vals = [d["configs"][a]["buckets"].get(k, {}).get("ms_per_step", 0.0) for a in arms]
        ax.bar(range(len(arms)), vals, 0.55, bottom=bottoms,
               color=BUCKET_COLOR.get(k, INK3), label=k)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, a in enumerate(arms):
        ax.text(i, bottoms[i] * 1.008, f"{bottoms[i]:.1f}", ha="center", va="bottom",
                color=INK2, fontsize=9)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([SHORT.get(a, a) for a in arms])
    ax.set_ylabel("ms/step of GPU time (bucketed trace, 8 steps)")
    ax.set_title("Per kernel bucket -- read shares within a bar, never totals across captures",
                 loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, fontsize=9, loc="upper center")
    ax.set_ylim(0, max(bottoms) * 1.28)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def main():
    os.makedirs(PLOTS, exist_ok=True)
    made = 0
    for fn, name in ((plot_e2e, "00_e2e_pre_vs_post.png"),
                     (plot_layers, "01_per_conv_layer.png"),
                     (plot_model, "02_model_layer_harness.png"),
                     (plot_buckets, "03_kernel_buckets.png")):
        if fn(os.path.join(PLOTS, name)):
            made += 1
        else:
            print(f"NOTE: skipped {name} -- its data file is not staged yet")
    print(f"wrote {made} plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
