"""Plots for the current-state report, from the JSON the measurement steps wrote.

PALETTE. The categorical slots below are the data-viz reference palette in its documented fixed
order. The rule is that a categorical palette is validated by running the skill's
scripts/validate_palette.js rather than eyeballed -- node is not installed in this container, so
instead of eyeballing anything this file stays strictly inside what that reference already states as
validated: the fixed slot order, at most 6 slots on the ADJACENT pairlist (stacked bars, grouped
bars), and at most 3 on the ALL-PAIRS pairlist (lines, scatter). Where a form needs more identities
than that, colour is dropped as the identity channel and direct labels carry it instead -- which is
the case for the five-mode scatter in plot 2.

Run: python docs/state_report_2026-08-12/scripts/make_plots.py    # no GPU
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402

D = "docs/state_report_2026-08-12"
#: reference palette, light mode, documented fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID = "#e3e2df"
#: display order and short labels; the JSON keys are the mode strings benchmark_ldm uses
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


BUCKETS = ["GEMM / conv", "GroupNorm+SiLU family", "attention", "quantize (standalone)",
           "elementwise / copy", "other"]


def bucket_kernel(name):
    """Coarse buckets. Kernel names are long and mangled; the report needs shape, not identity.

    ORDER MATTERS AND WAS CHECKED AGAINST THE REAL NAMES, not guessed. Three that a plausible
    ordering gets wrong:
      * `pytorch_flash::flash_fwd_kernel` contains no "attn" and would fall to "other";
      * `gn_stats_partials_chanmajor_kernel` contains no "norm" and would too;
      * and the big one -- that same flash kernel's TEMPLATE ARGUMENTS contain `cutlass::half_t`,
        so a GEMM-first ordering files fp16's entire attention cost (1896 ms, 9% of the run) as
        "GEMM / conv" and the fp16 bar shows no attention at all. Attention is tested FIRST
        because "flash"/"attn" are the more specific tokens; no real GEMM kernel carries them.

    The GroupNorm bucket deliberately absorbs the FUSED quantize prologues
    (`group_norm_silu_quantize_*`, `gn_apply_delta_quantize_*`). They do the normalisation pass and
    emit int8/int4 in one kernel, so the time is not separable; filing them under "quantize" would
    invent a quantization overhead the model does not separately pay. "quantize (standalone)" is
    only the kernels that do nothing else.
    """
    n = name.lower()
    if "flash" in n or "attn" in n or "attention" in n or "softmax" in n or "sdpa" in n or "bmm" in n:
        return "attention"
    if "cutlass" in n or "gemm" in n or "implicit" in n or "xmma" in n:
        return "GEMM / conv"
    if "group_norm" in n or "groupnorm" in n or "silu" in n or n.startswith("gn_") or "gn_" in n:
        return "GroupNorm+SiLU family"
    if "quant" in n or "absmax" in n or "pack" in n:
        return "quantize (standalone)"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "cat" in n:
        return "elementwise / copy"
    return "other"


def plot_e2e(e2e):
    """Magnitude across five categories, one measure -> horizontal bars, single hue, no legend."""
    rows = [(lab, e2e["modes"][m]) for m, lab in MODES if m in e2e.get("modes", {})]
    if not rows:
        return
    labs = [r[0] for r in rows]
    vals = [r[1]["per_step_ms"] for r in rows]
    sp = [r[1].get("speedup_vs_fp16") for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    y = range(len(rows))
    ax.barh(list(y), vals, height=0.62, color=SERIES[0])
    ax.set_yticks(list(y), labs)
    ax.invert_yaxis()
    ax.set_xlabel("ms per denoising step   (batch 128, DDIM 200 steps, A40)")
    ax.set_xlim(0, max(vals) * 1.26)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    # Direct labels. Speedup rides in the same label rather than on a second axis -- a dual-axis
    # chart is the one form the method forbids outright.
    for i, (v, s) in enumerate(zip(vals, sp)):
        t = f"{v:.1f} ms" + (f"   {s:.2f}x vs fp16" if s and s != 1.0 else "")
        ax.text(v + max(vals) * 0.015, i, t, va="center", ha="left", fontsize=9, color=INK2)
    ax.set_title("End-to-end latency at the shipped defaults", loc="left", pad=10)
    fig.savefig(f"{D}/plots/01_e2e_speed.png")
    plt.close(fig)
    print("wrote 01_e2e_speed.png")


def plot_tradeoff(e2e, q):
    """Two measures, five identities. Scatter is the ALL-PAIRS pairlist, where only three slots
    validate -- so identity is carried by direct labels and every point takes one hue."""
    pts = []
    for m, lab in MODES:
        if m in e2e.get("modes", {}) and m in q.get("modes", {}):
            pts.append((lab, e2e["modes"][m]["per_step_ms"], q["modes"][m]["relL2_vs_fp16"]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for lab, ms, rel in pts:
        ax.scatter([ms], [rel], s=70, color=SERIES[0], zorder=3,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.annotate(f"{lab}\n{rel:.3f} @ {ms:.0f} ms", (ms, rel), textcoords="offset points",
                    xytext=(9, 6), fontsize=9, color=INK2)
    ax.set_xlabel("ms per denoising step  (lower is faster)")
    ax.set_ylabel("latent relL2 vs fp16  (lower is closer)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(p[1] for p in pts) * 1.35)
    ax.set_ylim(-0.06, max(p[2] for p in pts) * 1.22)
    ax.set_title("Speed against fidelity — every arm at its shipped default", loc="left", pad=10)
    fig.savefig(f"{D}/plots/02_quality_vs_speed.png")
    plt.close(fig)
    print("wrote 02_quality_vs_speed.png")


def plot_kernels(e2e):
    """Composition within each mode -> stacked bars on the adjacent pairlist, <=6 slots."""
    order, data = [], {}
    for m, lab in MODES:
        d = e2e.get("modes", {}).get(m)
        if not d or not d.get("kernels"):
            continue
        agg = {}
        for r in d["kernels"]:
            agg[bucket_kernel(r["kernel"])] = agg.get(bucket_kernel(r["kernel"]), 0.0) + r["us"] / 1e3
        data[lab] = agg
        order.append(lab)
    if not order:
        return
    buckets = [b for b in BUCKETS if any(b in data[l] for l in order)]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    y = range(len(order))
    left = [0.0] * len(order)
    for b in buckets:
        vals = [data[l].get(b, 0.0) for l in order]
        # Colour by the bucket's FIXED index in BUCKETS, not by its rank in the filtered list.
        # "quantize (standalone)" is empty on this model (every quantize is fused into a GN
        # kernel), and colouring by rank would silently repaint everything after it the day a
        # standalone quantize kernel appears. Colour follows the entity, never its position.
        # 2px surface gap between adjacent segments, per the mark spec.
        ax.barh(list(y), vals, left=left, height=0.62, color=SERIES[BUCKETS.index(b) % len(SERIES)],
                label=b, edgecolor=SURFACE, linewidth=1.4)
        left = [a + v for a, v in zip(left, vals)]
    ax.set_yticks(list(y), order)
    ax.invert_yaxis()
    ax.set_xlabel("GPU kernel time per profiled window (ms)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Where the GPU time goes, by kernel bucket", loc="left", pad=10)
    fig.savefig(f"{D}/plots/03_kernel_buckets.png")
    plt.close(fig)
    print("wrote 03_kernel_buckets.png")


def plot_blocks(rows):
    """Per-kind attribution per config -> stacked bars, same pairlist and cap as plot 3."""
    rows = [r for r in rows if r.get("kinds")]
    if not rows:
        return
    kinds = list(rows[0]["kinds"].keys())[:6]
    labs = [r["config"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(rows) + 2.2))
    y = range(len(rows))
    left = [0.0] * len(rows)
    for ki, k in enumerate(kinds):
        vals = [r["kinds"].get(k, 0.0) for r in rows]
        ax.barh(list(y), vals, left=left, height=0.6, color=SERIES[ki % len(SERIES)],
                label=k, edgecolor=SURFACE, linewidth=1.4)
        left = [a + v for a, v in zip(left, vals)]
    for i, r in enumerate(rows):
        ax.text(left[i] * 1.01, i, f"{left[i]:.1f} ms attributed / {r['wall_ms_per_step']:.1f} wall",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(list(y), labs)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.42)
    ax.set_xlabel("ms per denoising step, attributed to quantized layers by kind")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Per-block attribution: what each arm spends where", loc="left", pad=10)
    fig.savefig(f"{D}/plots/04_block_kinds.png")
    plt.close(fig)
    print("wrote 04_block_kinds.png")


def plot_layers(rows):
    """Per-layer cost in depth order. Lines are the ALL-PAIRS pairlist -> at most 3 series."""
    rows = [r for r in rows if r.get("layers")][:3]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    for i, r in enumerate(rows):
        vals = list(r["layers"].values())
        ax.plot(range(len(vals)), vals, linewidth=2.0, color=SERIES[i], label=r["config"])
    ax.set_xlabel("quantized layer, in UNet depth order  (input blocks -> middle -> output blocks)")
    ax.set_ylabel("ms/step")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Per-layer cost, first three arms", loc="left", pad=10)
    fig.savefig(f"{D}/plots/05_per_layer.png")
    plt.close(fig)
    print("wrote 05_per_layer.png")


def main():
    os.makedirs(f"{D}/plots", exist_ok=True)
    e2e, q = load(f"{D}/data/e2e.json"), load(f"{D}/data/samples_quality.json")
    layers = load(f"{D}/data/profile_layers.json")
    if e2e:
        plot_e2e(e2e)
        plot_kernels(e2e)
        if q:
            plot_tradeoff(e2e, q)
    else:
        print("skip: data/e2e.json missing")
    if layers:
        plot_blocks(layers)
        plot_layers(layers)
    else:
        print("skip: data/profile_layers.json missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
