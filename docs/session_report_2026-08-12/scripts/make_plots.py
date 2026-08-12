"""Summary figures for the 2026-08-12 session. Offline: reads other reports' data/, no GPU.

This report does not re-measure anything. It reads the committed data of the two reports it summarises
and draws the three views that only make sense across them:

  1. the session's ONE landed speed change, and the two candidates it refutes, on one axis
  2. what the csrc/ split cost and what it did not change
  3. the verification ladder -- which gate caught what

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
AQ = os.path.join(ROOT, "docs/aq_fusion_2026-08-12/data")
POST = os.path.join(ROOT, "docs/postsplit_benchmark_2026-08-12/data")
PLOTS = os.path.join(ROOT, "docs/session_report_2026-08-12/plots")

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, PLUM, ROSE = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cc7", "#d64570"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def plot_verdicts(out):
    """Every fusion candidate this session touched. TWO panels, because the numbers are not in the
    same unit and putting them on one axis would be wrong.

    Left: end-to-end ms/step at batch 128 -- route (a) and route (b) are paired A/B measurements on the
    whole model; hd=24 is a per-call microbenchmark extrapolated over its 5 blocks, so it is marked
    (est.). Right: the GN-stats candidates, which are KERNEL-level and only ever measured against the
    pass they would replace, expressed as a ratio. A ratio below 1.0 is faster.

    Signed so positive = time saved, and the refutations are drawn at their measured cost rather than
    parked at zero: "built it, it was 4.5 ms worse" is the record that stops someone re-proposing it.
    """
    ab = load(AQ, "ab_route_b.json")
    gn = load(AQ, "gn_stats_tree.json")
    bench8 = load(AQ, "bench_packed_vs_unpacked_load8.json")
    if not (ab and gn and bench8):
        return False
    hd24 = next(r for r in bench8["rows"] if r["hd"] == 24)
    hd24_step = 5 * (hd24["u_total_ms"] - hd24["p_ms"])
    wm = gn["weighted_ms"]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.6, 5.0),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    items = [("route (b)\nqkv int8 -> flash\nLANDED", ab["paired_median"], BLUE, ""),
             ("hd=24 via 8-byte\ncp.async loader\nREFUTED", hd24_step, ROSE, " (est.)"),
             ("route (a)\nfp16 -> flash\nREFUTED 08-11", -18.0, ROSE, "")]
    vals = [i[1] for i in items]
    axl.bar([i[0] for i in items], vals, color=[i[2] for i in items], width=0.55)
    axl.axhline(0, color=INK2, linewidth=1.2)
    for i, (lbl, v, c, suf) in enumerate(items):
        axl.text(i, v + (0.5 if v >= 0 else -0.5), f"{v:+.2f}{suf}", ha="center",
                 va="bottom" if v >= 0 else "top", color=INK2, fontsize=9.5, fontweight="bold")
    axl.set_ylabel("ms/step saved (+) or lost (-), batch 128")
    axl.set_title("End to end: one landed, two refuted", loc="left")
    axl.grid(axis="y", color=GRID, linewidth=0.8)
    axl.set_axisbelow(True)
    axl.set_ylim(min(vals) * 1.30, max(vals) * 4.5)
    plt.setp(axl.get_xticklabels(), fontsize=8.5)

    ratios = [("shared atomics\nREFUTED 08-11", wm["atomics"] / wm["shipped"], ROSE),
              ("warp tree\nMARGIN TOO THIN", wm["tree"] / wm["shipped"], PLUM)]
    axr.bar([r[0] for r in ratios], [r[1] for r in ratios],
            color=[r[2] for r in ratios], width=0.5)
    axr.axhline(1.0, color=ORANGE, linewidth=1.6, linestyle="--")
    axr.text(1.45, 1.02, "the pass it replaces", color=ORANGE, ha="right", va="bottom", fontsize=8.5)
    for i, r in enumerate(ratios):
        axr.text(i, r[1] + 0.04, f"{r[1]:.2f}x", ha="center", va="bottom", color=INK2,
                 fontsize=9.5, fontweight="bold")
    axr.set_ylabel("vs the shipped GN-stats pass (lower is faster)")
    axr.set_title("GN stats -> conv epilogue: kernel level", loc="left")
    axr.grid(axis="y", color=GRID, linewidth=0.8)
    axr.set_axisbelow(True)
    axr.set_ylim(0, 2.15)
    plt.setp(axr.get_xticklabels(), fontsize=8.5)

    fig.suptitle("Every fusion candidate touched this session -- all five BUILT and MEASURED, "
                 "none rejected on a design argument", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_split_cost(out):
    """The csrc/ split: what it cost (build time) against what it did not change (speed)."""
    post = load(POST, "differential_timing_postsplit.json")
    fp16 = load(POST, "differential_timing_fp16_postsplit.json")
    if not (post and fp16):
        return False
    fp = next(iter(fp16["arms"].values()))["stats"]["median"] / 1e3 / fp16["steps"]
    PRE_RATIO = {"int8_ptq": 1.447, "modiff_conv_k4": 1.372, "modiff_conv_k1": 1.278,
                 "modiff_full_k1": 1.006, "modiff_full_k4": 1.064}
    arms = [a for a in PRE_RATIO if a in post["arms"]]
    SHORT = {"int8_ptq": "W8A8\nPTQ", "modiff_conv_k4": "conv\nK=4", "modiff_conv_k1": "conv\nK=1",
             "modiff_full_k1": "conv+proj\nK=1", "modiff_full_k4": "conv+proj\nK=4"}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # left: speedup vs fp16, pre vs post
    x = range(len(arms))
    w = 0.38
    pre = [PRE_RATIO[a] for a in arms]
    postr = [fp / (post["arms"][a]["stats"]["median"] / 1e3 / post["steps"]) for a in arms]
    axl.bar([i - w / 2 for i in x], pre, w, color=INK3, label="pre-split")
    axl.bar([i + w / 2 for i in x], postr, w, color=BLUE, label="post-split")
    for i, a in enumerate(arms):
        axl.text(i, max(pre[i], postr[i]) + 0.012, f"{postr[i] - pre[i]:+.3f}",
                 ha="center", va="bottom", color=INK2, fontsize=8.5)
    axl.set_xticks(list(x))
    axl.set_xticklabels([SHORT[a] for a in arms], fontsize=9)
    axl.set_ylabel("speedup vs fp16 (same-run anchor)")
    axl.set_title("Unchanged: every arm within 0.005x", loc="left")
    axl.set_ylim(0.9, max(pre + postr) * 1.06)
    axl.grid(axis="y", color=GRID, linewidth=0.8)
    axl.set_axisbelow(True)
    axl.legend(fontsize=9, loc="upper right")

    # right: the cost
    names = ["clean build\n246 -> 480 s", ".so size\n25.3 -> 25.9 MiB",
             "translation units\n12 -> 20"]
    pre_c = [246, 26480696 / 2**20, 12]
    post_c = [480, 27116888 / 2**20, 20]
    xs = range(len(names))
    for i, (a, b) in enumerate(zip(pre_c, post_c)):
        axr.bar(i - w / 2, 100.0, w, color=INK3)
        axr.bar(i + w / 2, 100.0 * b / a, w, color=ORANGE)
        axr.text(i + w / 2, 100.0 * b / a + 3, f"{b / a:.2f}x", ha="center", va="bottom",
                 color=ORANGE, fontsize=9.5, fontweight="bold")
    axr.axhline(100, color=INK2, linewidth=1.0)
    axr.set_xticks(list(xs))
    axr.set_xticklabels(names, fontsize=9)
    axr.set_ylabel("% of pre-split")
    axr.set_title("The cost: duplicated CUTLASS instantiations", loc="left")
    axr.set_ylim(0, 230)
    axr.grid(axis="y", color=GRID, linewidth=0.8)
    axr.set_axisbelow(True)
    fig.suptitle("csrc/ split into baseline/ + modiff/ trees: build time nearly doubled, "
                 "runtime untouched", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_gates(out):
    """What each verification gate RETURNED against what it COST.

    Two honest columns, because a gate that produces more false alarms than findings is a real result
    about the gate. The SASS golden is the clearest case: it returned one finding (a pre-existing
    template divergence, not actionable) and cost three false alarms, every one a bug in its own parser
    -- fatbin boundary text attributed to the preceding kernel, branch targets printed as absolute
    addresses, and a comment column right-padded to a width shared across the whole dump. It still
    earned its place, because it is the only instrument that can certify 279 kernels of device code
    unchanged, and it did that for all six family migrations. But the cost belongs in the record.
    """
    rows = [
        ("build\ncompiles + links", 5, 0,
         "3 range-boundary errors in conv/, 2 dependency escapes in quantize/"),
        ("kernel tests\nhost wrappers", 3, 0,
         "3 pre-existing int4/export_apply failures surfaced (verified NOT caused by the split)"),
        ("test_sass_golden\n279 kernels", 1, 3,
         "returned 1 pre-existing template divergence; cost 3 false alarms, all parser bugs"),
        ("test_export_manifest\n130 exports", 0, 0,
         "held 130/130 through every migration -- a guard that never fired, as intended"),
        ("paired A/B\nend-to-end speed", 0, 0,
         "the measurement, not a gate: +0.71 post-split vs +0.79 recorded"),
        ("e2e_output_check\nsampled latents", 0, 2,
         "UNUSABLE as found: MoDiff reference all-NaN, baseline false-fails ~1 run in 3"),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    y = list(range(len(rows)))
    found = [r[1] for r in rows]
    cost = [r[2] for r in rows]
    ax.barh(y, found, color=BLUE, height=0.5, label="real defects returned")
    ax.barh(y, cost, left=found, color=ROSE, height=0.5,
            label="false alarms / defects in the gate itself")
    for i, r in enumerate(rows):
        ax.text(found[i] + cost[i] + 0.15, i, r[3], va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title("The verification ladder: what each gate returned, and what it cost", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 22)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def main():
    os.makedirs(PLOTS, exist_ok=True)
    n = 0
    for fn, name in ((plot_verdicts, "00_fusion_verdicts.png"),
                     (plot_split_cost, "01_split_cost_vs_runtime.png"),
                     (plot_gates, "02_verification_ladder.png")):
        if fn(os.path.join(PLOTS, name)):
            n += 1
        else:
            print(f"NOTE: skipped {name} -- source data missing")
    print(f"wrote {n} plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
