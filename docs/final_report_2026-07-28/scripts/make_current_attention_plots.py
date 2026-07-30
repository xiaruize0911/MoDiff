"""Visualize the current FP16/INT8/INT4 attention benchmark and deep profile."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

COLORS = {
    "fp16": "#6B7280",
    "int8_baseline": "#2563EB",
    "int4_baseline": "#059669",
    "Attention core": "#7C3AED",
    "Projection GEMM": "#2563EB",
    "QKV preparation": "#F59E0B",
    "GroupNorm + input quantization": "#059669",
    "Output quantization": "#DB2777",
    "PyTorch SDPA fallback": "#64748B",
    "Copies, fills, and launch gaps": "#CBD5E1",
}


def shape_label(row):
    return f"C{row['x_shape'][1]}\nT{row['x_shape'][2] * row['x_shape'][3]}"


def attention_rows(data, mode):
    return [row for row in data["modes"][mode] if row["kind"] == "attention"]


def category(name):
    if "flash_attn_int" in name:
        return "Attention core"
    if "pytorch_flash" in name:
        return "PyTorch SDPA fallback"
    if "gemm_w8a8" in name or "gemm_w4a4" in name:
        return "Projection GEMM"
    if name.startswith("aq_"):
        return "QKV preparation"
    if "group_norm_silu_quantize" in name:
        return "GroupNorm + input quantization"
    if "quant_attn_out" in name:
        return "Output quantization"
    return "Copies, fills, and launch gaps"


def plot_latency_and_speedup(layer):
    modes = ["fp16", "int8_baseline", "int4_baseline"]
    rows = {mode: attention_rows(layer, mode) for mode in modes}
    labels = [shape_label(row) for row in rows["fp16"]]
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [1.65, 1]})
    for index, mode in enumerate(modes):
        values = [row["pipeline_us"] for row in rows[mode]]
        bars = ax0.bar(x + (index - 1) * width, values, width,
                       color=COLORS[mode], label=mode.replace("_baseline", "").upper())
        ax0.bar_label(bars, labels=[f"{v:.0f}" for v in values], fontsize=7,
                      rotation=90, padding=2)
    ax0.set_yscale("log")
    ax0.set_ylim(100, 4500)
    ax0.set_xticks(x, labels)
    ax0.set_ylabel("Whole attention-layer latency (µs, log scale)")
    ax0.set_title("Latency at every production attention shape")
    ax0.legend(frameon=False, ncols=3)
    ax0.grid(axis="y", which="both", alpha=0.2)

    totals = {
        mode: sum(row["pipeline_us"] * row["n_instances"] for row in rows[mode]) / 1000
        for mode in modes
    }
    base = totals["fp16"]
    bars = ax1.bar(["FP16", "INT8", "INT4"], [totals[m] for m in modes],
                   color=[COLORS[m] for m in modes], width=0.66)
    labels2 = [
        f"{totals['fp16']:.2f} ms\n1.000×",
        f"{totals['int8_baseline']:.2f} ms\n{base / totals['int8_baseline']:.3f}×",
        f"{totals['int4_baseline']:.2f} ms\n{base / totals['int4_baseline']:.3f}×",
    ]
    ax1.bar_label(bars, labels=labels2, fontsize=10, padding=4)
    ax1.set_ylim(0, max(totals.values()) * 1.22)
    ax1.set_ylabel("Weighted latency across 21 blocks (ms)")
    ax1.set_title("Production-weighted result")
    ax1.grid(axis="y", alpha=0.2)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Current MoDiff attention pipeline — A40, batch 128",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = PLOTS / "fig_current_attention_benchmark.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out}")


def profile_components(row):
    result = {}
    kernel_sum = 0.0
    for kernel in row["kernels"]:
        value = kernel["us"] * kernel.get("calls", 1.0)
        kernel_sum += value
        key = category(kernel["kernel"])
        result[key] = result.get(key, 0.0) + value
    if kernel_sum > row["full_us"]:
        # torch.profiler adds launch overhead. Preserve its component proportions but
        # project them onto the separately measured, unprofiled wall-clock latency.
        scale = row["full_us"] / kernel_sum
        result = {key: value * scale for key, value in result.items()}
    else:
        gaps = row["full_us"] - kernel_sum
        result["Copies, fills, and launch gaps"] = (
            result.get("Copies, fills, and launch gaps", 0.0) + gaps
        )
    return result


def plot_profile(profile, layer):
    modes = ["int8_baseline", "int4_baseline"]
    order = [
        "Attention core",
        "Projection GEMM",
        "QKV preparation",
        "GroupNorm + input quantization",
        "Output quantization",
        "PyTorch SDPA fallback",
        "Copies, fills, and launch gaps",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, mode in zip(axes, modes):
        wall = {
            (row["x_shape"][1], row["x_shape"][2] * row["x_shape"][3]): row["pipeline_us"]
            for row in attention_rows(layer, mode)
        }
        rows = []
        for source in profile["modes"][mode]:
            row = dict(source)
            row["full_us"] = wall[(row["C"], row["T"])]
            rows.append(row)
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        parts = [profile_components(row) for row in rows]
        for key in order:
            values = np.array([part.get(key, 0.0) for part in parts])
            if values.max() == 0:
                continue
            ax.bar(x, values, bottom=bottom, width=0.67, label=key,
                   color=COLORS[key])
            bottom += values
        for index, row in enumerate(rows):
            ax.text(index, bottom[index] + max(bottom) * 0.015,
                    f"{row['full_us']:.0f} µs", ha="center", fontsize=8)
        ax.set_ylabel("µs per layer call")
        ax.set_title(mode.replace("_baseline", "").upper())
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    rows = profile["modes"][modes[0]]
    axes[-1].set_xticks(np.arange(len(rows)),
                        [f"C{row['C']}\nT{row['T']}" for row in rows])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle("Kernel-level profile of the current quantized attention pipeline\n"
                 "Profiler component shares normalized to unprofiled wall time",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = PLOTS / "fig_current_attention_profile.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out}")


def plot_efficiency(profile):
    modes = ["int8_baseline", "int4_baseline"]
    eligible = {
        mode: [row for row in profile["modes"][mode] if row["flash_path"] == "ours"]
        for mode in modes
    }
    labels = [f"C{row['C']}\nT{row['T']}" for row in eligible[modes[0]]]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for index, mode in enumerate(modes):
        values = [row["flash_pct_peak"] for row in eligible[mode]]
        bars = ax.bar(x + (index - 0.5) * width, values, width,
                      color=COLORS[mode], label=mode.replace("_baseline", "").upper())
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], fontsize=9, padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Achieved dense tensor-core peak (%)")
    ax.set_title("Custom attention-core efficiency\n"
                 "INT4 T1024 uses INT8 MMA to avoid head-dimension padding waste",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 55)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = PLOTS / "fig_current_attention_efficiency.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out}")


def main():
    with open(DATA / "layer_pipeline_bench.json") as handle:
        layer = json.load(handle)
    with open(DATA / "qattn_deep_profile.json") as handle:
        profile = json.load(handle)
    plot_latency_and_speedup(layer)
    plot_profile(profile, layer)
    plot_efficiency(profile)


if __name__ == "__main__":
    main()
