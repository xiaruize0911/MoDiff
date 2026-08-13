"""Figures for docs/bench_report_2026-08-13_postzp, from the JSON the four measurement steps wrote.

One figure per requested item: e2e latency, per-block attribution, then attention / conv / linear
per-kernel. Data only -- the report carries no interpretation, so these are labelled and left to speak.

PALETTE. The categorical slots are the data-viz reference palette in its documented fixed order,
imported from docs/state_report_2026-08-12/scripts/make_plots.py rather than re-typed so the two
reports cannot drift. node is not installed here, so scripts/validate_palette.js cannot be run; as in
that file the response is to stay strictly inside what the reference already states as validated --
fixed slot order, at most 6 slots on the ADJACENT pairlist (stacked/grouped bars), at most 3 on the
ALL-PAIRS pairlist (lines, scatter). Where a form needs more identities than that, colour stops being
the identity channel and direct labels carry it.

Run: python docs/bench_report_2026-08-13_postzp/scripts/make_plots.py    # no GPU
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs/state_report_2026-08-12/scripts"))

import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from make_plots import SERIES, SURFACE, INK, INK2, GRID, bucket_kernel, BUCKETS   # noqa: E402

D = "docs/bench_report_2026-08-13_postzp"
MODES = [("fp16", "fp16"), ("int8_baseline", "W8A8 PTQ"), ("int8", "W8A8 MoDiff"),
         ("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig1_e2e(e2e):
    """Magnitude across five categories, one measure -> horizontal bars, single hue, no legend."""
    rows = [(lab, e2e["modes"][m]) for m, lab in MODES if m in e2e.get("modes", {})]
    if not rows:
        return
    labs = [r[0] for r in rows]
    vals = [r[1]["per_step_ms"] for r in rows]
    sp = [r[1].get("speedup_vs_fp16") for r in rows]
    cv = [r[1].get("wall_cv_pct") for r in rows]
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    y = range(len(rows))
    ax.barh(list(y), vals, height=0.62, color=SERIES[0])
    ax.set_yticks(list(y), labs)
    ax.invert_yaxis()
    ax.set_xlabel("ms per denoising step   (batch 128, DDIM 200 steps, A40)")
    ax.set_xlim(0, max(vals) * 1.30)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    # Direct labels; the speedup rides in the same label rather than on a second axis, because a
    # dual-axis chart is the one form the method forbids outright.
    for i, (v, s, c) in enumerate(zip(vals, sp, cv)):
        t = f"{v:.2f} ms" + (f"   {s:.3f}x" if s and s != 1.0 else "") + (f"   CV {c:.2f}%" if c else "")
        ax.text(v + max(vals) * 0.013, i, t, va="center", fontsize=9, color=INK2)
    ax.set_title("1. End-to-end latency, five modes at the shipped defaults", loc="left", pad=10)
    fig.savefig(f"{D}/plots/01_e2e.png")
    plt.close(fig)
    print("wrote 01_e2e.png")


def fig2_blocks(rows):
    """Per-kind attribution per config -> stacked bars, ADJACENT pairlist, <=6 slots."""
    rows = [r for r in rows if r.get("kinds")]
    if not rows:
        return
    kinds = list(rows[0]["kinds"].keys())[:6]
    labs = [r["config"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(rows) + 2.4))
    y = range(len(rows))
    left = [0.0] * len(rows)
    for ki, k in enumerate(kinds):
        vals = [r["kinds"].get(k, 0.0) for r in rows]
        # 2px surface gap between adjacent segments, per the mark spec.
        ax.barh(list(y), vals, left=left, height=0.6, color=SERIES[ki % len(SERIES)],
                label=k, edgecolor=SURFACE, linewidth=1.4)
        left = [a + v for a, v in zip(left, vals)]
    for i, r in enumerate(rows):
        ax.text(left[i] * 1.01, i, f"{left[i]:.1f} / {r['wall_ms_per_step']:.1f} wall",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(list(y), labs)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.45)
    ax.set_xlabel("ms per denoising step, attributed to quantized layers by block kind")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("1b. Per-block attribution by kind", loc="left", pad=10)
    fig.savefig(f"{D}/plots/02_blocks.png")
    plt.close(fig)
    print("wrote 02_blocks.png")


def suite_totals(ks, suite):
    """Per-mode us PER SAMPLE for one suite, plus its entry points ranked.

    Each row is one captured call signature, replayed in isolation: stats.median is the time for ONE
    replay and calls_per_sample is how often the model makes that call. The per-sample cost is
    therefore median x calls_per_sample summed over signatures -- summing medians alone would weight a
    signature called 5 times the same as one called 700 times.
    """
    out, tops = {}, {}
    for m, lab in MODES:
        d = (ks.get("modes") or {}).get(m) or {}
        rows = d.get(suite) or []
        tot, per = 0.0, {}
        for r in rows:
            st = r.get("stats") or {}
            med = float(st.get("median") or 0.0)
            n = float(r.get("calls_per_sample") or 0.0)
            us = med * n
            tot += us
            e = r.get("entry", "?")
            per[e] = per.get(e, 0.0) + us
        out[lab] = tot / 1e3
        tops[lab] = sorted(per.items(), key=lambda kv: -kv[1])
    return out, tops


def fig_suite(ks, suite, idx, title):
    """One suite, five modes -> horizontal bars, single hue (one measure, one identity axis)."""
    tot, _ = suite_totals(ks, suite)
    labs = [l for _, l in MODES if l in tot]
    vals = [tot[l] for l in labs]
    if not any(vals):
        print(f"skip {suite}: no data")
        return
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    y = range(len(labs))
    ax.barh(list(y), vals, height=0.62, color=SERIES[0])
    ax.set_yticks(list(y), labs)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.24)
    ax.set_xlabel(f"{suite} kernel time per sample, replayed at the model's real shapes "
                  f"(ms; median x calls_per_sample, summed)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.013, i, f"{v:.2f} ms", va="center", fontsize=9, color=INK2)
    ax.set_title(title, loc="left", pad=10)
    fig.savefig(f"{D}/plots/{idx}_{suite}.png")
    plt.close(fig)
    print(f"wrote {idx}_{suite}.png")


def main():
    os.makedirs(f"{D}/plots", exist_ok=True)
    e2e = load(f"{D}/data/e2e.json")
    blocks = load(f"{D}/data/profile_layers.json")
    ks = load(f"{D}/data/kernel_suites.json")
    if e2e:
        fig1_e2e(e2e)
    else:
        print("skip: data/e2e.json missing")
    if blocks:
        fig2_blocks(blocks)
    else:
        print("skip: data/profile_layers.json missing")
    if ks:
        for idx, (suite, title) in enumerate(
                (("attention", "2. Attention kernels"), ("conv", "3. Conv kernels"),
                 ("linear", "4. Linear kernels")), start=3):
            fig_suite(ks, suite, f"0{idx}", title)
    else:
        print("skip: data/kernel_suites.json missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
