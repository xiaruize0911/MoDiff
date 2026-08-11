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


def plot_kernel(out, fname="bench_packed_vs_unpacked.json", title=None, brk=None):
    """Per shape: what arm U spends on aq_* + mma flash, against arm P's single gather kernel.

    Called twice: once on the 16-byte-only data (hd=24 rejected, the state route (b) shipped in) and
    once on the 8-byte-loader data (hd=24 legal and losing). `brk` labels the break-even ratio per
    shape, which is what each bar has to be read against.
    """
    d = load(fname)
    if d is None:
        return False
    rows = d["rows"]
    labels = [f"C{r['C']}\nT{r['T']}, hd{r['hd']}"
              + (f"\nbreak-even {brk[r['hd']]:.2f}x" if brk and r["hd"] in brk else "")
              for r in rows]
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
        tall = r["p_ms"] > 0.8 * max(max(aq[k] + fl[k] for k in range(len(rows))),
                                     max(q["p_ms"] or 0 for q in rows))
        ax.text(i + w / 2, r["p_ms"] * (0.97 if tall else 1.02),
                f"{net:+.3f}\n{r['p_over_u_flash']:.2f}x flash", ha="center",
                va="top" if tall else "bottom", color="white" if tall else INK2, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"ms per call, batch {d['batch']} (median of {d['iters']})")
    ax.set_title(title or "Route (b) at the kernel level: the quantize it removes vs the gather it "
                 "pays for", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max([a + f for a, f in zip(aq, fl)] + [r["p_ms"] or 0 for r in rows]) * 1.20)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


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


def plot_trace(out):
    """Where the time went, per kernel, from the bucketed Perfetto traces.

    The net (+0.79) is three terms pulling against each other, and only a trace can show them apart.
    What this figure deliberately does NOT show is the trace TOTAL: the two captures are separate
    8-step processes minutes apart, and buckets route (b) cannot touch (conv -0.92, norm_quantize
    -0.11) moved by more than half the effect. That is capture drift, and it is why the headline
    number comes from the paired A/B instead. Read shares and named kernels here, never totals --
    the same limit docs/profile_kernels_layers_2026-08-11 states for its own tables.
    """
    d = load("trace_buckets_qkvi8.json")
    if d is None:
        return False
    a = d["configs"]["modiff_full_k4_projk4"]
    b = d["configs"]["modiff_full_k4_projk4_qkvi8"]

    def kms(c, prefix):
        """Sum every kernel whose name STARTS WITH prefix.

        Not an exact-key lookup: bucket_traces keys the plain GEMM by its full C++ signature
        ("gemm_w8a8_kernel_awq(signed char const*, ...") and an exact match silently returns 0.0,
        which put this figure's GEMM term at -3.92 instead of +0.31 -- the out_i8 half counted and
        the plain half not. Prefix, and assert something matched.
        """
        hit = [v["ms_per_step"] for n, v in c["kernels"].items() if n.startswith(prefix)]
        assert hit, f"no kernel matching {prefix!r} in {c.get('config', '?')}"
        return sum(hit)

    # The plain and out_i8 GEMMs are ONE term: route (b) moves 10 of the 42 qkv/proj GEMM calls from
    # the plain variant to the int8-output one, so counting either alone is meaningless.
    gemm_a = kms(a, "gemm_w8a8_kernel_awq")
    gemm_b = kms(b, "gemm_w8a8_kernel_awq")
    terms = [
        ("aq_* quantize\nremoved", a["buckets"]["attn_quantize"]["ms_per_step"]
         - b["buckets"]["attn_quantize"]["ms_per_step"], AQ_GOLD),
        ("packed gather\npaid back", -(b["buckets"]["attention"]["ms_per_step"]
                                       - a["buckets"]["attention"]["ms_per_step"]), ROSE),
        ("qkv GEMM\nplain -> out_i8", gemm_a - gemm_b, ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    labels = [t[0] for t in terms] + ["sum of the\nthree terms", "paired A/B\n(the number)"]
    vals = [t[1] for t in terms]
    vals += [sum(vals), load("ab_route_b.json")["paired_median"]]
    colors = [t[2] for t in terms] + [INK3, BLUE]
    ax.bar(labels, vals, color=colors, width=0.6)
    ax.axhline(0, color=INK2, linewidth=1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", color=INK2, fontsize=9)
    ax.set_ylabel("ms/step recovered (positive = faster)")
    ax.set_title("What route (b) trades, per kernel, batch 128", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.01, 0.03, "trace attributes the terms; it does not measure the total "
                        "(unrelated buckets drift ~1 ms between captures)",
            transform=ax.transAxes, color=INK3, fontsize=8)
    ax.set_ylim(min(vals) - 0.45, max(vals) + 0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def plot_gn_stats(out):
    """Per shape: the tree reduction against the pass it replaces and against the atomics attempt.

    Per-shape bars rather than the weighted total alone, because the weighted total (0.96x) hides the
    thing that decides whether Stage C is worth starting: 768x4x4 is still 1.54x, worst exactly where
    the tensors are small -- the same shape-dependence the atomics version had at 4.51x.
    """
    d = load("gn_stats_tree.json")
    if d is None:
        return False
    rows = d["rows"]
    labels = [f"{r['C']}\n{r['H']}x{r['W']}\n(x{r['count']})" for r in rows]
    x = list(range(len(rows)))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar([i - w for i in x], [r["shipped_us"] for r in rows], w, color=INK3, label="shipped pass")
    ax.bar(x, [r["tree_us"] for r in rows], w, color=AQUA, label="tree reduction (this)")
    ax.bar([i + w for i in x], [r["atomics_us"] for r in rows], w, color=PLUM,
           label="shared atomics (2026-08-11)")
    for i, r in enumerate(rows):
        ratio = r["tree_us"] / r["shipped_us"]
        ax.text(i, r["tree_us"] * 1.03, f"{ratio:.2f}x", ha="center", va="bottom",
                color=ROSE if ratio > 1 else AQUA, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"us per launch, batch {d['batch']}")
    wm = d["weighted_ms"]
    ax.set_title(f"GN stats from conv tiles: weighted {wm['tree']:.2f} ms vs shipped "
                 f"{wm['shipped']:.2f} ({d['tree_over_shipped']:.2f}x), "
                 f"atomics {wm['atomics']:.2f}", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.text(0.01, 0.006, "deterministic on every shape; the atomics version was not, which is the "
             "gate this rewrite exists for", color=INK3, fontsize=8)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


def main():
    os.makedirs(PLOTS, exist_ok=True)
    plot_kernel(os.path.join(PLOTS, "00_packed_vs_unpacked.png"),
                brk={48: 2.00})
    plot_e2e(os.path.join(PLOTS, "01_e2e_arms.png"))
    plot_paired(os.path.join(PLOTS, "02_paired_ab.png"))
    if not plot_quality(os.path.join(PLOTS, "03_quality_paired.png")):
        print("NOTE: data/quality_route_b.json absent -- 03_quality_paired.png not written")
    if not plot_trace(os.path.join(PLOTS, "04_trace_terms.png")):
        print("NOTE: data/trace_buckets_qkvi8.json absent -- 04_trace_terms.png not written")
    if not plot_kernel(os.path.join(PLOTS, "05_loader_width.png"),
                       "bench_packed_vs_unpacked_load8.json",
                       "With the 8-byte cp.async loader: hd=24 becomes legal, and loses",
                       brk={24: 1.44, 48: 2.00}):
        print("NOTE: data/bench_packed_vs_unpacked_load8.json absent -- 05 not written")
    if not plot_gn_stats(os.path.join(PLOTS, "06_gn_stats_reduction.png")):
        print("NOTE: data/gn_stats_tree.json absent -- 06_gn_stats_reduction.png not written")
    print("wrote plots to", PLOTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
