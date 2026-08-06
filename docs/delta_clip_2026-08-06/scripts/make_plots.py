"""Three figures: quality vs activation precision, the clip sweep, and end-to-end speed.

Each figure is drawn from ONE harness's JSON, never two. Cross-script relL2 comparisons in this
project disagree by up to ~20% on identical code (see FINDINGS.md), so mixing sources inside a single
axis would draw a difference that is measurement, not model:

    fig_quality_vs_bits  act_bit_sweep{,_ceiling_k4}.json   -- baseline vs MoDiff, A8..A2, r=1.0
    fig_clip_sweep       clip_e2e_bits.json                 -- relL2 vs clip ratio, A8/A4/A3
    fig_speed            e2e_ceiling_b{128,8}.json          -- ms/step, three modes, two batches

Palette is the dataviz reference instance's first three categorical slots, unmodified
(#2a78d6 blue, #eb6834 orange, #1baf7a aqua), which validate all-pairs in light mode: worst CVD
deltaE 9.2, worst normal-vision 24.0. Aqua sits at 2.74:1 on the surface, below the 3:1 line, so the
relief rule applies and every series carries a direct label as well as a legend -- identity is never
color alone here. Where the same entity appears in two configurations (MoDiff before and after the
ceiling) it keeps its hue and changes linestyle rather than taking a second slot.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                 # noqa: E402
from matplotlib.ticker import FuncFormatter, FixedLocator                       # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PLOTS = os.path.join(ROOT, "docs/delta_clip_2026-08-06/plots")
DATA = os.path.join(ROOT, "docs/delta_clip_2026-08-06/data")
OLD = os.path.join(ROOT, "docs/act_bits_2026-08-05/data")

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#86b6ef"          # same hue, sequential step 250: "same entity, earlier measurement"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})


def load(p):
    with open(p) as f:
        return json.load(f)


def style(ax, ygrid=True):
    """Recessive axes: no top/right spines, one soft grid direction, no tick marks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if ygrid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)


def relfmt(v, _=None):
    return f"{v:g}"


# ----------------------------------------------------------------------------------------------
# Figure 1: quality vs activation precision
# ----------------------------------------------------------------------------------------------
def fig_quality():
    new = load(os.path.join(DATA, "act_bit_sweep_ceiling_k4.json"))
    old = load(os.path.join(OLD, "act_bit_sweep.json"))
    o = {r["act_bits"]: r for r in old["rows"]}
    bits = [r["act_bits"] for r in new["rows"]]                  # 8..2
    x = list(range(len(bits)))
    base = [r["baseline"]["mean"] for r in new["rows"]]
    mod = [r["modiff"]["mean"] for r in new["rows"]]
    mod_pre = [o[b]["modiff"]["mean"] for b in bits]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    style(ax)

    # FID anchors from docs/fid_2026-08-05, as recessive reference lines. They are what make a relL2
    # number mean something; without them the y axis is uncalibrated. Labels sit INSIDE the axes --
    # placed past the last x tick they render outside the clip box and silently vanish.
    for y, label in ((0.039, "FID 7.8 — fp16 parity"), (0.238, "FID 16.4"), (0.456, "FID 200")):
        ax.axhline(y, color=INK3, linewidth=0.8, linestyle=(0, (1, 3)), zorder=1)
        # Below the line, not above: the A8 baseline point sits just above the FID 16.4 anchor.
        ax.annotate(label, (0.02, y), textcoords="offset points", xytext=(0, -3),
                    color=INK3, fontsize=8.5, va="top", ha="left", zorder=1)

    ax.plot(x, base, color=ORANGE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label="baseline (PTQ, MoDiff off)", zorder=3)
    ax.plot(x, mod_pre, color=BLUE_LIGHT, linewidth=2, linestyle=(0, (4, 2)), marker="o",
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="MoDiff, as previously published", zorder=3)
    ax.plot(x, mod, color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label="MoDiff (corrected)", zorder=4)

    # Direct labels, required alongside the legend: the light-blue step is below 3:1 on this surface,
    # so identity cannot rest on color. Placed at the right end and staggered vertically; value
    # labels go to the LEFT of their point so the two never contend for the same space.
    ax.annotate("baseline (PTQ)", (x[-1], base[-1]), textcoords="offset points", xytext=(4, 4),
                color=ORANGE, fontsize=9.5, fontweight="bold", ha="left")
    ax.annotate("MoDiff", (x[-1], mod[-1]), textcoords="offset points", xytext=(4, 6),
                color=BLUE, fontsize=9.5, fontweight="bold", ha="left")
    ax.annotate("as published", (x[-1], mod_pre[-1]), textcoords="offset points", xytext=(4, -12),
                color="#1c5cab", fontsize=8.5, ha="left")

    # Only the rows the ceiling actually corrected get a value label -- a number on every point is
    # noise, and these three are the finding.
    for i, b in enumerate(bits):
        if b in (4, 3, 2):
            ax.annotate(f"{mod[i]:.3f}", (x[i], mod[i]), textcoords="offset points",
                        xytext=(-9, 3), color=INK, fontsize=8.5, ha="right")

    ax.set_yscale("log")
    ax.set_ylim(0.030, 1.25)
    ax.yaxis.set_major_locator(FixedLocator([0.04, 0.06, 0.1, 0.2, 0.4, 0.8]))
    ax.yaxis.set_major_formatter(FuncFormatter(relfmt))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.set_xticks(x)
    ax.set_xticklabels([f"A{b}" for b in bits])
    ax.set_xlim(-0.25, len(bits) - 0.35)          # right margin for the direct labels
    ax.set_xlabel("activation precision in the conv path  (weights fixed at W8)")
    ax.set_ylabel("latent relative L2 vs fp16   (log, lower is better)")
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK2, bbox_to_anchor=(1.0, 0.04))

    fig.suptitle("Activation precision at fixed W8: MoDiff holds to A5, and the old A4–A2 rows were flattered",
                 color=INK, x=0.008, ha="left", fontsize=11.5, y=0.985)
    fig.text(0.008, 0.905,
             "batch 8, DDIM 50, 3 paired seeds, MODIFF_DELTA_REFRESH=4, no clip (r=1.0)\n"
             "dashed: the same rows before the delta quantizer could saturate at Q_b",
             color=INK2, fontsize=8.5, ha="left", va="top")

    out = os.path.join(PLOTS, "fig_quality_vs_bits.png")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------------------------
# Figure 2: the clip sweep
# ----------------------------------------------------------------------------------------------
def fig_clip():
    d = load(os.path.join(DATA, "clip_e2e_bits.json"))
    by = {}
    for r in d["rows"]:
        by.setdefault(r["delta_refresh"], {}).setdefault(r["act_bits"], {})[r["clip_ratio"]] = r
    colors = {8: BLUE, 4: ORANGE, 3: AQUA}

    # Ordinal x, not the ratio's numeric value: the swept ratios bunch up at the low end
    # (0.3/0.25/0.2), and on a linear axis those three tick labels overlap into one smear.
    ratios = sorted({r["clip_ratio"] for r in d["rows"]}, reverse=True)      # 1.0 -> 0.2
    xs = list(range(len(ratios)))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1), sharey=True)
    for ax, k in zip(axes, (1, 4)):
        style(ax)
        rows = by[k]
        for b in (8, 4, 3):
            ys = [rows[b][r]["mean"] for r in ratios]
            ax.plot(xs, ys, color=colors[b], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            # ring the optimum: for A8 that is r=1.0, which is the point of the A8 row
            bi = min(range(len(ratios)), key=lambda i: ys[i])
            ax.plot([xs[bi]], [ys[bi]], marker="o", markersize=14, markerfacecolor="none",
                    markeredgecolor=colors[b], markeredgewidth=2, zorder=4)
            # A4 and A8 converge at the hard-clip end, so stagger the labels rather than let them
            # sit on top of each other.
            dy = {8: -9, 4: 4, 3: 0}[b]
            ax.annotate(f"A{b}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, dy),
                        color=colors[b], fontsize=9.5, fontweight="bold", va="center")
            if b != 8:
                ax.annotate(f"{ys[bi]:.3f} at r={ratios[bi]:g}  ({ys[bi] / rows[b][1.0]['mean']:.2f}x)",
                            (xs[bi], ys[bi]), textcoords="offset points", xytext=(0, 15),
                            color=INK, fontsize=8.5, ha="center")
        ax.set_xlim(-0.35, len(ratios) - 0.35)
        ax.set_xticks(xs)
        ax.set_xticklabels([("1.0\n(no clip)" if r == 1.0 else f"{r:g}") for r in ratios])
        ax.set_xlabel("MODIFF_DELTA_CLIP ratio r   (grid step = r·max|delta| / Q_b)")
        ax.set_title(f"MODIFF_DELTA_REFRESH = {k}" + ("   (shipped default)" if k == 4 else ""),
                     color=INK2, loc="left", fontsize=10.5)

    axes[0].set_yscale("log")
    axes[0].set_ylim(0.05, 0.55)
    axes[0].yaxis.set_major_locator(FixedLocator([0.06, 0.1, 0.2, 0.4]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(relfmt))
    axes[0].yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    axes[0].set_ylabel("latent relative L2 vs fp16   (log, lower is better)")
    handles = [plt.Line2D([], [], color=colors[b], linewidth=2, marker="o", markersize=7,
                          markeredgecolor=SURFACE, markeredgewidth=2, label=f"A{b}")
               for b in (8, 4, 3)]
    axes[0].legend(handles=handles, loc="upper right", fontsize=9, labelcolor=INK2, ncol=3)
    fig.suptitle("A clip helps only where the grid is coarse: A4 halves, A3 drops 2.6x, A8 only loses",
                 color=INK, x=0.008, ha="left", fontsize=11.5, y=0.985)
    fig.text(0.008, 0.905, "batch 8, DDIM 50, 3 paired seeds; every A4/A3 point below r=1.0 wins on "
                           "3 of 3 seeds, every A8 point loses\nringed = best r for that row "
                           "(for A8 that is r=1.0, i.e. no clip)",
             color=INK2, fontsize=8.5, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = os.path.join(PLOTS, "fig_clip_sweep.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------------------------
# Figure 3: end-to-end speed
# ----------------------------------------------------------------------------------------------
def fig_speed():
    files = [(128, os.path.join(DATA, "e2e_ceiling_b128.json")),
             (8, os.path.join(DATA, "e2e_ceiling_b8.json"))]
    files = [(b, p) for b, p in files if os.path.exists(p)]
    if not files:
        return None
    labels = ["fp16", "int8 (PTQ)", "int8 + MoDiff"]
    keys = ["fp16", "int8_baseline", "int8"]
    colors = [INK3, ORANGE, BLUE]

    fig, axes = plt.subplots(1, len(files), figsize=(5.4 * len(files), 4.9))
    axes = [axes] if len(files) == 1 else list(axes)
    for ax, (batch, path) in zip(axes, files):
        d = load(path)
        style(ax)
        ms = [d["modes"][k]["wall_mean_us"] / 1000 / d["steps"] for k in keys]
        cv = [d["modes"][k]["wall_cv_pct"] for k in keys]
        # 2px surface gap between adjacent bars: width 0.62 on unit spacing.
        bars = ax.bar(range(3), ms, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2,
                      zorder=3)
        for i, (b, v) in enumerate(zip(bars, ms)):
            ax.annotate(f"{v:.1f} ms", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        color=INK, fontsize=9.5, fontweight="bold")
            sub = "1.00x" if i == 0 else f"{ms[0] / v:.2f}x fp16"
            if i == 2:
                sub += f"  ·  {ms[1] / v:.2f}x int8"
            ax.annotate(f"{sub}\nCV {cv[i]:.2f}%", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 20), ha="center",
                        color=INK2, fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(ms) * 1.32)
        ax.set_ylabel("ms per DDIM step   (lower is better)")
        ax.set_title(f"batch {batch}", color=INK2, loc="left", fontsize=10.5)

    fig.suptitle("End-to-end speed is a batch-size story, and the quantizer's precision does not enter it",
                 color=INK, x=0.008, ha="left", fontsize=11.5, y=0.985)
    # Two short lines, not one long one: a single line at this fontsize overruns the figure width
    # and matplotlib clips it silently rather than wrapping.
    fig.text(0.008, 0.905, "DDIM 200, 3 warmups + 5 repeats, A40, idle GPU\n"
                           "activations keep their int8 container and the GEMM stays W8A8, so A8/A4/A3 "
                           "all run at these times — A_b is a quality knob, not a speed one",
             color=INK2, fontsize=8.5, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(PLOTS, "fig_speed.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    os.makedirs(PLOTS, exist_ok=True)
    made = [fig_quality(), fig_clip(), fig_speed()]
    for m in made:
        print("wrote" if m else "skipped (no data)", m or "fig_speed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
