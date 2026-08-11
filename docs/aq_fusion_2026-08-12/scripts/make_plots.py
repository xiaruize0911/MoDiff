"""Figures for the route (b) report. Offline: reads data/*.json, writes plots/*.png, no GPU.

Three instruments, and they are not interchangeable:

  * KERNEL microbenchmark (data/bench_packed_vs_unpacked.json) -- per shape, random tensors, no
    model. Authoritative for "which kernel is faster at this shape", and the source of the +0.79
    prediction.
  * PAIRED A/B (data/ab_route_b.json) -- both arms on one model object, alternating. The one to
    trust for end-to-end speed at this effect size.
  * DIFFERENTIAL (data/differential_timing_qkvi8.json) -- separate profiler-free runs per arm. A
    second, independent read on the same delta.

Palette and rcParams copied from docs/profile_kernels_layers_2026-08-11/scripts/make_plots.py so the
two reports' figures read as one set.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.join(ROOT, "docs/aq_fusion_2026-08-12")
DATA, PLOTS = os.path.join(HERE, "data"), os.path.join(HERE, "plots")

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, PLUM, ROSE = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cc7", "#d64570"
AQ_GOLD = "#c08a2e"                                   # the attn_quantize bucket's colour
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})


def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def plot_kernel(out):
    """Per shape: what arm U spends on aq_* + mma flash, against arm P's single gather kernel."""
    d = load("bench_packed_vs_unpacked.json")
    rows = d["rows"]
    labels = [f"C{r['C']}\nT{r['T']}, hd{r['hd']}" for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    w = 0.34
    aq = [r["u_quant_ms"] for r in rows]
    fl = [r["u_flash_ms"] for r in rows]
    ax.bar([i - w / 2 for i in x], aq, w, color=AQ_GOLD, label="arm U: aq_* quantize")
    ax.bar([i - w / 2 for i in x], fl, w, bottom=aq, color=ROSE, label="arm U: mma flash")
    for i, r in enumerate(rows):
        if r.get("p_ms") is None:
            # A rejected shape is a result, not a gap: say why on the axis.
            ax.text(i + w / 2, 0.02, "int8 gather\nREJECTED\nhd%16", ha="center", va="bottom",
                    color=INK3, fontsize=8)
            continue
        ax.bar(i + w / 2, r["p_ms"], w, color=BLUE, label="arm P: packed gather" if i == 1 else None)
        net = r["u_total_ms"] - r["p_ms"]
        ax.text(i + w / 2, r["p_ms"] * 1.02, f"{net:+.3f}\n{r['p_over_u_flash']:.2f}x flash",
                ha="center", va="bottom", color=INK2, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"ms per call, batch {d['batch']} (median of {d['iters']})")
    ax.set_title("Route (b) at the kernel level: the quantize it removes vs the gather it pays for",
                 loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_e2e(out):
    """The two arms end to end, from the differential harness, against the committed fp16 anchor."""
    d = load("differential_timing_qkvi8.json")
    ms = {k: a["stats"]["median"] / 1e3 / d["steps"] for k, a in d["arms"].items()}
    order = ["modiff_full_k4_projk4", "modiff_full_k4_projk4_qkvi8"]
    short = {"modiff_full_k4_projk4": "conv+proj K=4\n+proj K=4",
             "modiff_full_k4_projk4_qkvi8": "…+ route (b)\n(opt-in)"}
    fp16_file = os.path.join(ROOT, "docs/profile_kernels_layers_2026-08-11/data/"
                                   "differential_timing_fp16.json")
    fp16 = None
    if os.path.exists(fp16_file):
        with open(fp16_file) as f:
            g = json.load(f)
        fp16 = g["arms"]["fp16"]["stats"]["median"] / 1e3 / g["steps"]
    vals = [ms[k] for k in order]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar([short[k] for k in order], vals, color=[INK3, BLUE], width=0.5)
    for i, v in enumerate(vals):
        lab = f"{v:.2f}" + (f"\n{fp16 / v:.3f}x" if fp16 else "")
        ax.text(i, v * 1.004, lab, ha="center", va="bottom", color=INK2, fontsize=9)
    ax.annotate(f"{vals[0] - vals[1]:+.2f} ms/step", xy=(1, vals[1]), xytext=(0.5, vals[0] * 1.02),
                ha="center", color=BLUE, fontsize=9)
    if fp16:
        ax.axhline(fp16, color=ORANGE, linewidth=1.6, linestyle="--")
        ax.text(1.45, fp16, f" fp16 {fp16:.1f}", color=ORANGE, va="bottom", fontsize=9)
    ax.set_ylabel(f"ms/step (profiler-free, {d['steps']} steps x {d['repeats']} repeats)")
    ax.set_title(f"End to end, batch {d['batch']}, A40", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # Zoomed y: the whole effect is 0.8% of the step, and a zero-based axis hides it. Labelled as
    # zoomed so the bars are not read as a ratio.
    ax.set_ylim(min(vals) * 0.97, max(vals + ([fp16] if fp16 else [])) * 1.06)
    ax.text(0.01, 0.02, "y axis zoomed -- bar heights are not proportional", transform=ax.transAxes,
            color=INK3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_paired(out):
    """Every paired repeat, against the prediction the microbenchmark made before this ran.

    Individual repeats, not a mean with error bars: with four pairs the spread IS the evidence, and
    this project has twice had a 3-sample mean reverse sign at 8 samples.
    """
    import statistics
    d = load("ab_route_b.json")
    pairs = d["pairs"]
    x = list(range(1, len(pairs) + 1))
    med = statistics.median(pairs)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.axhline(0, color=INK3, linewidth=1.0)
    # The prediction and the measurement can land on the same value -- they did, at +0.79 -- and two
    # overlapping labels then read as a rendering fault rather than as the result. Merge them.
    pred = d["prediction_ms"]
    if abs(med - pred) < 0.03:
        ax.axhline(pred, color=ORANGE, linewidth=1.6, linestyle="--")
        ax.text(len(pairs) + 0.06, pred,
                f" prediction AND measured\n median both {med:+.2f}", color=ORANGE, va="center",
                fontsize=9)
    else:
        ax.axhline(pred, color=ORANGE, linewidth=1.6, linestyle="--")
        ax.text(len(pairs) + 0.06, pred, f" prediction {pred:+.2f}", color=ORANGE, va="center",
                fontsize=9)
        ax.axhline(med, color=BLUE, linewidth=1.2)
        ax.text(len(pairs) + 0.06, med, f" measured median {med:+.2f}", color=BLUE,
                va="center", fontsize=9)
    ax.plot(x, pairs, "o", color=BLUE, markersize=9)
    for i, p in enumerate(pairs):
        ax.text(x[i], p + 0.012, f"{p:+.2f}", ha="center", va="bottom", color=INK2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xlabel("paired repeat (ON differenced against the OFF measured immediately after it)")
    ax.set_ylabel("ms/step recovered (OFF - ON)")
    ax.set_title(f"Paired A/B, batch {d['batch']}, {d['steps']} steps -- one model object, "
                 f"alternating arms", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0.6, len(pairs) + 1.35)
    ax.set_ylim(min(0, min(pairs) - 0.15), max(pairs + [d["prediction_ms"]]) + 0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_quality(out):
    """Per-seed relL2, paired. Returns False when the quality run has not been staged yet."""
    d = load("quality_route_b.json")
    if d is None:
        return False
    r = d["result"]
    seeds = [str(s) for s in d["seeds"]]
    off = [r["per_seed_off"][s] for s in seeds]
    on = [r["per_seed_on"][s] for s in seeds]
    x = list(range(len(seeds)))
    per = [(n - o) / o * 100.0 for o, n in zip(off, on)]
    sem, mean = r["paired_diff_pct_sem"], r["paired_diff_pct_mean"]

    # Two panels because one number cannot carry this. LEFT: the absolute relL2, where seed-to-seed
    # spread (0.018 to 0.099) is 5x the whole range the arms differ over -- which is exactly why an
    # unpaired comparison of these means would be worthless. RIGHT: the paired per-seed difference,
    # where the question actually lives.
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    for i in x:
        axl.plot([i, i], [off[i], on[i]], color=GRID, linewidth=2.0, zorder=1)
    axl.plot(x, off, "o", color=INK3, markersize=9, label="OFF (today)", zorder=2)
    axl.plot(x, on, "o", color=BLUE, markersize=9, label="ON (route b)", zorder=2)
    axl.set_xticks(x)
    axl.set_xticklabels(seeds)
    axl.set_xlabel("seed")
    axl.set_ylabel("latent relL2 vs fp16")
    axl.set_title("Absolute: seed dominates", loc="left")
    axl.grid(axis="y", color=GRID, linewidth=0.8)
    axl.set_axisbelow(True)
    axl.set_ylim(0, max(off + on) * 1.18)
    axl.legend(loc="upper left")

    band = 2 * sem
    axr.axhspan(mean - band, mean + band, color=BLUE, alpha=0.10)
    axr.axhline(0, color=INK3, linewidth=1.0)
    axr.axhline(mean, color=BLUE, linewidth=1.4)
    axr.plot(x, per, "o", color=BLUE, markersize=9)
    for i, p in enumerate(per):
        axr.text(x[i], p + (max(per) - min(per)) * 0.05, f"{p:+.2f}%", ha="center", va="bottom",
                 color=INK2, fontsize=8)
    axr.text(len(seeds) - 0.55, mean + band, f" mean {mean:+.2f}% +- 2 SEM ({band:.2f}%)",
             color=BLUE, va="bottom", ha="right", fontsize=9)
    axr.set_xticks(x)
    axr.set_xticklabels(seeds)
    axr.set_xlabel("seed (same seed and same fp16 reference in both arms)")
    axr.set_ylabel("ON - OFF, % of OFF")
    verdict = ("bit-identical" if r["identical"] else
               ("RESOLVED" if r["resolved"] else "not resolved: the band contains 0"))
    axr.set_title(f"Paired difference -- {verdict}", loc="left")
    axr.grid(axis="y", color=GRID, linewidth=0.8)
    axr.set_axisbelow(True)
    pad = (max(per) - min(per)) * 0.35 + 0.2
    axr.set_ylim(min(per + [mean - band]) - pad, max(per + [mean + band]) + pad)

    fig.suptitle(f"Route (b) quality, batch {d['batch']}, DDIM {d['steps']}, {len(seeds)} paired "
                 f"seeds -- control (OFF twice) was bit-identical", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def main():
    os.makedirs(PLOTS, exist_ok=True)
    plot_kernel(os.path.join(PLOTS, "00_packed_vs_unpacked.png"))
    plot_e2e(os.path.join(PLOTS, "01_e2e_arms.png"))
    plot_paired(os.path.join(PLOTS, "02_paired_ab.png"))
    if not plot_quality(os.path.join(PLOTS, "03_quality_paired.png")):
        print("NOTE: data/quality_route_b.json absent -- 03_quality_paired.png not written")
    print("wrote plots to", PLOTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
