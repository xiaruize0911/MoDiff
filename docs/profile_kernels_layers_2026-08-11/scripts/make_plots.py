"""Plots for the kernel-and-layer profile.

Two granularities, two instruments, and they are not interchangeable:

  * PER KERNEL -- Perfetto traces bucketed offline (docs/component_attribution_2026-08-07's
    trace_configs.py + bucket_traces.py, whose configs are imported from differential_timing.py so
    the arms are provably the same ones the e2e table times). Authoritative for "which CUDA kernel".
  * PER LAYER -- CUDA events on the live dispatch targets
    (integration/tests/profile_layers_and_model.py). Authoritative for "which layer", and only for
    shares: its coverage is 0.64-0.88 of the step.

Reads data/trace_buckets.json and data/profile_layers.json, writes plots/*.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.join(ROOT, "docs/profile_kernels_layers_2026-08-11")
DATA, PLOTS = os.path.join(HERE, "data"), os.path.join(HERE, "plots")

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

ARMS = ["fp16", "int8_ptq", "modiff_conv_k4", "modiff_conv_k1",
        "modiff_full_k4", "modiff_full_k1", "modiff_full_k4_projk4"]
SHORT = {"fp16": "fp16", "int8_ptq": "W8A8 PTQ", "modiff_conv_k4": "conv K=4",
         "modiff_conv_k1": "conv K=1", "modiff_full_k4": "conv+proj K=4",
         "modiff_full_k1": "conv+proj K=1",
         # NON-DEFAULT: MODIFF_LINEAR_DELTA_REFRESH=4. The label says so, because the knob defaults
         # to 1 and its quality is unverified -- a bar that reads like the others would imply a
         # shipped configuration.
         "modiff_full_k4_projk4": "conv+proj K=4\n+proj K=4 (opt-in)"}
BUCKET_COLOR = {"conv": BLUE, "norm_quantize": AQUA, "delta_quantize": PLUM,
                "linear_gemm": ORANGE, "attention": ROSE, "attn_quantize": "#c08a2e",
                "elementwise": INK3, "quantize": "#5aa9c4", "other": "#c9c7c1"}


def load(p):
    with open(p) as f:
        return json.load(f)


def ms(v):
    """bucket_traces.py writes {ms_per_step, calls_per_step, kernels} per entry, not a scalar."""
    return v["ms_per_step"] if isinstance(v, dict) else (v or 0.0)


def plot_e2e(out):
    """Whole-model ms/step and speedup, from the profiler-free differential harness."""
    d = load(os.path.join(DATA, "differential_timing_canonical.json"))
    f = load(os.path.join(DATA, "differential_timing_fp16.json"))
    fp16 = f["arms"]["fp16"]["stats"]["median"] / 1e3 / f["steps"]
    ms_by_arm = {k: a["stats"]["median"] / 1e3 / d["steps"] for k, a in d["arms"].items()}
    extra = os.path.join(DATA, "differential_timing_projk4.json")
    if os.path.exists(extra):
        e = load(extra)
        # Its modiff_full_k4 re-measures an arm canonical already has (99.59 vs 99.73, 0.14 apart),
        # which is the run-to-run check; keep canonical's so the older bars stay as published.
        for k, a in e["arms"].items():
            ms_by_arm.setdefault(k, a["stats"]["median"] / 1e3 / e["steps"])
    arms = [a for a in ARMS if a in ms_by_arm]
    vals = [ms_by_arm[a] for a in arms]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar([SHORT[a] for a in arms], vals, color=BLUE, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.012, f"{v:.1f}\n{fp16 / v:.3f}x", ha="center", va="bottom",
                color=INK2, fontsize=9)
    ax.axhline(fp16, color=ORANGE, linewidth=1.6, linestyle="--")
    ax.text(len(vals) - 0.45, fp16, f"  fp16 {fp16:.1f}", color=ORANGE, va="bottom", fontsize=9)
    ax.set_ylabel("ms/step (profiler-free, 200 steps x 5 repeats)")
    ax.set_title("End to end, batch 128, A40", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(vals + [fp16]) * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_buckets(tb, out):
    """Stacked kernel-bucket ms/step per arm. The whole step, by what the kernel DOES."""
    cfgs = [a for a in ARMS if a in tb["configs"]]
    order = ["conv", "norm_quantize", "delta_quantize", "linear_gemm", "attention",
             "attn_quantize", "quantize", "elementwise", "other"]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    labels = [SHORT[c] for c in cfgs]
    bottom = [0.0] * len(cfgs)
    for b in order:
        vals = [ms(tb["configs"][c]["buckets"].get(b)) for c in cfgs]
        if not any(vals):
            continue
        ax.bar(labels, vals, bottom=bottom, label=b, color=BUCKET_COLOR.get(b, INK3), width=0.64)
        bottom = [x + v for x, v in zip(bottom, vals)]
    for i, c in enumerate(cfgs):
        ax.text(i, bottom[i] * 1.008, f"{bottom[i]:.0f}", ha="center", va="bottom",
                color=INK2, fontsize=9)
    ax.set_ylabel("GPU ms/step (Perfetto trace, bucketed by kernel)")
    ax.set_title("Per kernel, grouped by what the kernel does", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    ax.set_ylim(0, max(bottom) * 1.28)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_top_kernels(tb, out, arm="modiff_full_k1", top=14):
    """The individual kernels that make up one arm's step, largest first."""
    ks = tb["configs"][arm]["kernels"]
    items = sorted(((ms(v), k) for k, v in ks.items()), reverse=True)[:top]
    names = [k[:58] + ("..." if len(k) > 58 else "") for _, k in items][::-1]
    vals = [v for v, _ in items][::-1]
    fig, ax = plt.subplots(figsize=(11.0, 0.42 * len(vals) + 1.5))
    ax.barh(range(len(vals)), vals, color=BLUE, height=0.72)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8.5)
    for i, v in enumerate(vals):
        ax.text(v * 1.01, i, f" {v:.2f}", va="center", color=INK2, fontsize=8.5)
    ax.set_xlabel("GPU ms/step")
    ax.set_title(f"Top {top} kernels — {SHORT.get(arm, arm)}", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(vals) * 1.16)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_kernel_delta(tb, out, a="modiff_conv_k1", b="modiff_full_k1", top=12):
    """Which kernels change when projection MoDiff is switched on. Diverging, largest |delta|."""
    ka, kb = tb["configs"][a]["kernels"], tb["configs"][b]["kernels"]

    keys = set(ka) | set(kb)
    d = sorted(((ms(kb.get(k)) - ms(ka.get(k)), k) for k in keys),
               key=lambda t: -abs(t[0]))[:top]
    d = sorted(d)
    names = [k[:54] + ("..." if len(k) > 54 else "") for _, k in d]
    vals = [v for v, _ in d]
    fig, ax = plt.subplots(figsize=(11.0, 0.42 * len(vals) + 1.7))
    ax.barh(range(len(vals)), vals,
            color=[ORANGE if v > 0 else AQUA for v in vals], height=0.72)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.axvline(0, color=INK2, linewidth=1.0)
    for i, v in enumerate(vals):
        ax.text(v + (0.06 if v > 0 else -0.06), i, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", color=INK2, fontsize=8.5)
    ax.set_xlabel(f"GPU ms/step,  {SHORT.get(b, b)} minus {SHORT.get(a, a)}")
    ax.set_title("What switching projection MoDiff on actually moves", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    lim = max(abs(min(vals)), abs(max(vals))) * 1.35
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_layers(prof, out):
    """Per conv layer in UNet depth order, coloured by channel width."""
    rows = [r for r in prof if r.get("layers")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11.5, 2.2 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    widths = sorted({c for r in rows for c in r["conv_in_channels"].values() if c})
    cmap = {c: [BLUE, AQUA, ORANGE, PLUM, ROSE, INK3][i % 6] for i, c in enumerate(widths)}
    for ax, r in zip(axes, rows):
        keys = sorted(k for k in r["layers"] if k.startswith("conv"))
        vals = [r["layers"][k] for k in keys]
        ch = [r["conv_in_channels"].get(str(int(k[4:]))) for k in keys]
        ax.bar(range(len(vals)), vals, color=[cmap.get(c, INK3) for c in ch], width=0.86)
        ax.set_title(f"{r['config']}   conv layers, {sum(vals):.1f} ms/step", loc="left")
        ax.set_ylabel("ms/step")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[c]) for c in widths]
    axes[0].legend(handles, [f"{c} ch" for c in widths], ncol=len(widths), fontsize=8.5,
                   loc="upper center")
    axes[-1].set_xlabel("quantized conv layer, UNet depth order (input -> middle -> output)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_kinds(prof, out):
    """Per layer KIND. Shares only -- coverage is printed on each bar."""
    rows = [r for r in prof if r.get("kinds")]
    order = ["conv", "updown", "attn (score path)", "proj (42 linears)"]
    col = {"conv": BLUE, "updown": AQUA, "attn (score path)": ORANGE, "proj (42 linears)": PLUM}
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    labels = [r["config"] for r in rows]
    bottom = [0.0] * len(rows)
    for k in order:
        vals = [r["kinds"].get(k, 0.0) for r in rows]
        ax.bar(labels, vals, bottom=bottom, label=k, color=col[k], width=0.62)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, r in enumerate(rows):
        ax.text(i, bottom[i] * 1.01, f"{bottom[i]:.0f}\ncover {r['sum_over_wall']:.2f}",
                ha="center", va="bottom", color=INK2, fontsize=8.5)
    ax.set_ylabel("ms/step attributed to layers of that kind")
    ax.set_title("Per layer kind  (CUDA events on the live dispatch targets; shares, not totals)",
                 loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(bottom) * 1.3)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    os.makedirs(PLOTS, exist_ok=True)
    made = [plot_e2e(os.path.join(PLOTS, "e2e_speedup.png"))]
    tb_p = os.path.join(DATA, "trace_buckets.json")
    if os.path.exists(tb_p):
        tb = load(tb_p)
        made.append(plot_buckets(tb, os.path.join(PLOTS, "kernel_buckets.png")))
        for arm in ("modiff_full_k1", "int8_ptq"):
            if arm in tb["configs"]:
                made.append(plot_top_kernels(tb, os.path.join(PLOTS, f"kernels_{arm}.png"), arm))
        if "modiff_conv_k1" in tb["configs"] and "modiff_full_k1" in tb["configs"]:
            made.append(plot_kernel_delta(tb, os.path.join(PLOTS, "kernel_delta_proj.png")))
    pf_p = os.path.join(DATA, "profile_layers.json")
    if os.path.exists(pf_p):
        prof = load(pf_p)
        made.append(plot_layers(prof, os.path.join(PLOTS, "layers.png")))
        made.append(plot_kinds(prof, os.path.join(PLOTS, "layer_kinds.png")))
    for m in made:
        print(f"  {m}  {os.path.getsize(m) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
