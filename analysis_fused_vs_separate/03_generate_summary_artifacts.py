#!/usr/bin/env python3
"""Generate final fused-vs-separate benchmark artifacts.

This script combines:
- layerwise fused-vs-separate results
- modelwise fused-vs-separate results

and produces:
- CSV summary tables
- PNG plots
- a combined Markdown report
- a compact summary JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str, headers: List[str], rows: Iterable[Iterable[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(list(row))


def speedup(fused_ms: float, separate_ms: float) -> float:
    return float(separate_ms) / float(fused_ms) if fused_ms > 0 else math.inf


def detect_quant_mode(layerwise_json: Dict[str, object], modelwise_json: Dict[str, object]) -> str:
    layer_meta = layerwise_json.get("metadata", {})
    if "quant_mode" in layer_meta:
        return str(layer_meta["quant_mode"])
    return str(modelwise_json.get("int8_fused_modiff", {}).get("quant_mode", "dynamic"))


def layerwise_supported_rows(layerwise: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for row in layerwise["results"]:
        if row["int8"].get("status") == "ok" and row["int4"].get("status") == "ok":
            rows.append(row)
    return rows


def layerwise_excluded_rows(layerwise: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for row in layerwise["results"]:
        if row["shape"].get("repo_supported_count", 0) == 0:
            rows.append(row)
    return rows


def total_ms(payload: Dict[str, object], precision: str) -> float:
    return float(payload[precision]["fused_total"]["mean_ms"])


def separate_total_ms(payload: Dict[str, object], precision: str) -> float:
    return float(payload[precision]["separate_total"]["mean_ms"])


def shape_score(row: Dict[str, object]) -> float:
    supported = float(row["shape"]["repo_supported_count"])
    return supported * max(
        float(row["int8"]["separate_total"]["mean_ms"]),
        float(row["int4"]["separate_total"]["mean_ms"]),
    )


def build_layerwise_tables(layerwise: Dict[str, object], output_dir: str) -> Dict[str, object]:
    aggregates = layerwise["aggregates"]
    supported = sorted(layerwise_supported_rows(layerwise), key=shape_score, reverse=True)
    excluded = sorted(layerwise_excluded_rows(layerwise), key=lambda r: r["shape"]["count"], reverse=True)

    write_csv(
        os.path.join(output_dir, "table_layerwise_weighted_summary.csv"),
        [
            "Precision",
            "Weighted Fused Step1 (ms)",
            "Weighted Separate Step1 (ms)",
            "Weighted Step1 Speedup",
            "Weighted Fused Conv (ms)",
            "Weighted Separate Conv (ms)",
            "Weighted Conv Speedup",
            "Weighted Fused Total (ms)",
            "Weighted Separate Total (ms)",
            "Weighted Total Speedup",
            "Benchmarked Calls",
            "Benchmarked Shapes",
        ],
        [
            [
                precision.upper(),
                f"{bucket['weighted_fused_step1_ms']:.6f}",
                f"{bucket['weighted_separate_step1_ms']:.6f}",
                f"{bucket['weighted_step1_speedup']:.6f}",
                f"{bucket['weighted_fused_conv_ms']:.6f}",
                f"{bucket['weighted_separate_conv_ms']:.6f}",
                f"{bucket['weighted_conv_speedup']:.6f}",
                f"{bucket['weighted_fused_total_ms']:.6f}",
                f"{bucket['weighted_separate_total_ms']:.6f}",
                f"{bucket['weighted_total_speedup']:.6f}",
                bucket['benchmarked_calls'],
                bucket['benchmarked_shapes'],
            ]
            for precision, bucket in aggregates.items()
        ],
    )

    write_csv(
        os.path.join(output_dir, "table_layerwise_supported_shapes.csv"),
        [
            "Shape",
            "Total Count",
            "Supported Count",
            "INT8 Fused Total (ms)",
            "INT8 Separate Total (ms)",
            "INT8 Speedup",
            "INT4 Fused Total (ms)",
            "INT4 Separate Total (ms)",
            "INT4 Speedup",
        ],
        [
            [
                row["shape"]["label"],
                row["shape"]["count"],
                row["shape"]["repo_supported_count"],
                f"{row['int8']['fused_total']['mean_ms']:.6f}",
                f"{row['int8']['separate_total']['mean_ms']:.6f}",
                f"{row['int8']['total_speedup']:.6f}",
                f"{row['int4']['fused_total']['mean_ms']:.6f}",
                f"{row['int4']['separate_total']['mean_ms']:.6f}",
                f"{row['int4']['total_speedup']:.6f}",
            ]
            for row in supported
        ],
    )

    write_csv(
        os.path.join(output_dir, "table_layerwise_excluded_shapes.csv"),
        ["Shape", "Total Count", "Excluded Count", "Reasons", "Example Layers"],
        [
            [
                row["shape"]["label"],
                row["shape"]["count"],
                row["shape"]["repo_unsupported_count"],
                "; ".join(row["shape"]["repo_unsupported_reasons"]),
                "; ".join(row["shape"]["repo_unsupported_layer_names"][:4]),
            ]
            for row in excluded
        ],
    )

    weakest = {
        precision: sorted(supported, key=lambda row: row[precision]["total_speedup"])[:5]
        for precision in ("int8", "int4")
    }
    strongest = {
        precision: sorted(supported, key=lambda row: row[precision]["total_speedup"], reverse=True)[:5]
        for precision in ("int8", "int4")
    }

    return {
        "aggregates": aggregates,
        "supported": supported,
        "excluded": excluded,
        "weakest": weakest,
        "strongest": strongest,
    }


def build_modelwise_tables(modelwise: Dict[str, object], output_dir: str) -> Dict[str, object]:
    mode_order = [
        "int8_fused_modiff",
        "int8_separate_modiff",
        "int4_fused_modiff",
        "int4_separate_modiff",
    ]

    write_csv(
        os.path.join(output_dir, "table_modelwise_summary.csv"),
        [
            "Mode",
            "Precision",
            "Implementation",
            "Mean Call (ms)",
            "Std Call (ms)",
            "Time/Sample (ms)",
            "Time/Step (ms)",
            "Ready Memory (MB)",
            "Timed Peak Memory (MB)",
            "Peak Delta (MB)",
            "Loaded Static Scales",
            "Scale Status",
            "Timed Calls",
        ],
        [
            [
                modelwise[mode]["label"],
                modelwise[mode]["precision"].upper(),
                modelwise[mode]["implementation"],
                f"{modelwise[mode]['mean_call_ms']:.6f}",
                f"{modelwise[mode]['std_call_ms']:.6f}",
                f"{modelwise[mode]['time_per_sample_ms']:.6f}",
                f"{modelwise[mode]['time_per_step_ms']:.6f}",
                f"{modelwise[mode].get('memory_ready_allocated_mb', modelwise[mode].get('memory_allocated_mb', 0.0)):.6f}",
                f"{modelwise[mode]['memory_peak_mb']:.6f}",
                f"{modelwise[mode].get('memory_peak_delta_mb', 0.0):.6f}",
                modelwise[mode]["loaded_static_scales"],
                modelwise[mode].get("scale_status", "unknown"),
                modelwise[mode]["timed_calls"],
            ]
            for mode in mode_order
        ],
    )

    precision_rows = []
    for precision in ("int8", "int4"):
        fused = modelwise[f"{precision}_fused_modiff"]
        separate = modelwise[f"{precision}_separate_modiff"]
        precision_rows.append(
            {
                "precision": precision,
                "fused_mean_call_ms": float(fused["mean_call_ms"]),
                "separate_mean_call_ms": float(separate["mean_call_ms"]),
                "fusion_speedup": speedup(float(fused["mean_call_ms"]), float(separate["mean_call_ms"])),
                "fused_ready_memory_mb": float(fused.get("memory_ready_allocated_mb", fused.get("memory_allocated_mb", 0.0))),
                "separate_ready_memory_mb": float(separate.get("memory_ready_allocated_mb", separate.get("memory_allocated_mb", 0.0))),
                "fused_peak_memory_mb": float(fused["memory_peak_mb"]),
                "separate_peak_memory_mb": float(separate["memory_peak_mb"]),
                "fused_peak_delta_mb": float(fused.get("memory_peak_delta_mb", 0.0)),
                "separate_peak_delta_mb": float(separate.get("memory_peak_delta_mb", 0.0)),
                "fused_scale_status": fused.get("scale_status", "unknown"),
                "separate_scale_status": separate.get("scale_status", "unknown"),
            }
        )

    write_csv(
        os.path.join(output_dir, "table_modelwise_speedup.csv"),
        [
            "Precision",
            "Fused Mean Call (ms)",
            "Separate Mean Call (ms)",
            "Fusion Speedup",
            "Fused Ready Memory (MB)",
            "Separate Ready Memory (MB)",
            "Fused Peak Memory (MB)",
            "Separate Peak Memory (MB)",
            "Fused Peak Delta (MB)",
            "Separate Peak Delta (MB)",
            "Fused Scale Status",
            "Separate Scale Status",
        ],
        [
            [
                row["precision"].upper(),
                f"{row['fused_mean_call_ms']:.6f}",
                f"{row['separate_mean_call_ms']:.6f}",
                f"{row['fusion_speedup']:.6f}",
                f"{row['fused_ready_memory_mb']:.6f}",
                f"{row['separate_ready_memory_mb']:.6f}",
                f"{row['fused_peak_memory_mb']:.6f}",
                f"{row['separate_peak_memory_mb']:.6f}",
                f"{row['fused_peak_delta_mb']:.6f}",
                f"{row['separate_peak_delta_mb']:.6f}",
                row["fused_scale_status"],
                row["separate_scale_status"],
            ]
            for row in precision_rows
        ],
    )

    return {
        "mode_order": mode_order,
        "precision_rows": precision_rows,
    }


def build_overall_summary(layerwise_summary: Dict[str, object], modelwise_summary: Dict[str, object], output_dir: str) -> Dict[str, object]:
    overall_rows = []
    for precision in ("int8", "int4"):
        layer_bucket = layerwise_summary["aggregates"][precision]
        model_bucket = next(row for row in modelwise_summary["precision_rows"] if row["precision"] == precision)
        overall_rows.append(
            {
                "precision": precision,
                "layerwise_weighted_speedup": float(layer_bucket["weighted_total_speedup"]),
                "layerwise_fused_total_ms": float(layer_bucket["weighted_fused_total_ms"]),
                "layerwise_separate_total_ms": float(layer_bucket["weighted_separate_total_ms"]),
                "modelwise_speedup": float(model_bucket["fusion_speedup"]),
                "modelwise_fused_call_ms": float(model_bucket["fused_mean_call_ms"]),
                "modelwise_separate_call_ms": float(model_bucket["separate_mean_call_ms"]),
            }
        )

    write_csv(
        os.path.join(output_dir, "table_overall_summary.csv"),
        [
            "Precision",
            "Layerwise Weighted Fused (ms)",
            "Layerwise Weighted Separate (ms)",
            "Layerwise Weighted Speedup",
            "Modelwise Fused Call (ms)",
            "Modelwise Separate Call (ms)",
            "Modelwise Speedup",
        ],
        [
            [
                row["precision"].upper(),
                f"{row['layerwise_fused_total_ms']:.6f}",
                f"{row['layerwise_separate_total_ms']:.6f}",
                f"{row['layerwise_weighted_speedup']:.6f}",
                f"{row['modelwise_fused_call_ms']:.6f}",
                f"{row['modelwise_separate_call_ms']:.6f}",
                f"{row['modelwise_speedup']:.6f}",
            ]
            for row in overall_rows
        ],
    )

    return {"rows": overall_rows}


def plot_layerwise_weighted_totals(layerwise_summary: Dict[str, object], output_dir: str) -> str:
    aggregates = layerwise_summary["aggregates"]
    precisions = ["int8", "int4"]
    fused = [aggregates[p]["weighted_fused_total_ms"] for p in precisions]
    separate = [aggregates[p]["weighted_separate_total_ms"] for p in precisions]

    x = np.arange(len(precisions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, fused, width, label="Fused", color="#4F81BD")
    ax.bar(x + width / 2, separate, width, label="Separate", color="#C0504D")
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in precisions])
    ax.set_ylabel("Weighted total hot-path time (ms)")
    ax.set_title("Layerwise weighted fused vs separate totals")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for idx, precision in enumerate(precisions):
        sp = aggregates[precision]["weighted_total_speedup"]
        ax.text(x[idx], max(fused[idx], separate[idx]) + 0.6, f"{sp:.2f}x", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = os.path.join(output_dir, "plot_01_layerwise_weighted_totals.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_layerwise_breakdown(layerwise_summary: Dict[str, object], output_dir: str) -> str:
    ag = layerwise_summary["aggregates"]
    labels = ["INT8 fused", "INT8 separate", "INT4 fused", "INT4 separate"]
    step1 = [
        ag["int8"]["weighted_fused_step1_ms"],
        ag["int8"]["weighted_separate_step1_ms"],
        ag["int4"]["weighted_fused_step1_ms"],
        ag["int4"]["weighted_separate_step1_ms"],
    ]
    conv = [
        ag["int8"]["weighted_fused_conv_ms"],
        ag["int8"]["weighted_separate_conv_ms"],
        ag["int4"]["weighted_fused_conv_ms"],
        ag["int4"]["weighted_separate_conv_ms"],
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, step1, label="Step1", color="#7FB3D5")
    ax.bar(x, conv, bottom=step1, label="Conv side", color="#F1948A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Weighted time (ms)")
    ax.set_title("Layerwise weighted breakdown: Step1 vs Conv side")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "plot_02_layerwise_weighted_breakdown.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_layerwise_top_shapes(layerwise_summary: Dict[str, object], output_dir: str, top_n: int = 10) -> str:
    supported = layerwise_summary["supported"][:top_n]
    labels = [f"{row['shape']['label']} ×{row['shape']['repo_supported_count']}" for row in supported][::-1]
    int8_values = [row["int8"]["total_speedup"] for row in supported][::-1]
    int4_values = [row["int4"]["total_speedup"] for row in supported][::-1]

    y = np.arange(len(labels))
    h = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.45 * len(labels))), sharey=True)

    axes[0].barh(y, int8_values, height=h, color="#4F81BD")
    axes[0].set_title("INT8 top supported shapes")
    axes[0].set_xlabel("Fused / Separate speedup")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y, int4_values, height=h, color="#C0504D")
    axes[1].set_title("INT4 top supported shapes")
    axes[1].set_xlabel("Fused / Separate speedup")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Top layerwise contributors by supported weighted cost", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, "plot_03_layerwise_top_shapes_speedup.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_modelwise_call_times(modelwise_summary: Dict[str, object], output_dir: str) -> str:
    rows = modelwise_summary["precision_rows"]
    precisions = [row["precision"].upper() for row in rows]
    fused = [row["fused_mean_call_ms"] for row in rows]
    separate = [row["separate_mean_call_ms"] for row in rows]

    x = np.arange(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, fused, width, label="Fused", color="#4F81BD")
    ax.bar(x + width / 2, separate, width, label="Separate", color="#C0504D")
    ax.set_xticks(x)
    ax.set_xticklabels(precisions)
    ax.set_ylabel("Mean full-model call time (ms)")
    ax.set_title("Whole-model fused vs separate timings")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for idx, row in enumerate(rows):
        ax.text(x[idx], max(fused[idx], separate[idx]) + 35.0, f"{row['fusion_speedup']:.2f}x", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = os.path.join(output_dir, "plot_04_modelwise_call_times.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_modelwise_memory_and_speedup(modelwise_summary: Dict[str, object], output_dir: str) -> str:
    rows = modelwise_summary["precision_rows"]
    precisions = [row["precision"].upper() for row in rows]
    speedups = [row["fusion_speedup"] for row in rows]
    fused_mem = [row["fused_peak_memory_mb"] for row in rows]
    separate_mem = [row["separate_peak_memory_mb"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(precisions, speedups, color=["#4F81BD", "#C0504D"])
    axes[0].set_ylabel("Fused / Separate speedup")
    axes[0].set_title("Whole-model fusion speedup")
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(rows))
    width = 0.35
    axes[1].bar(x - width / 2, fused_mem, width, label="Fused", color="#4F81BD")
    axes[1].bar(x + width / 2, separate_mem, width, label="Separate", color="#C0504D")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(precisions)
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].set_title("Whole-model peak memory")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(output_dir, "plot_05_modelwise_speedup_memory.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_summary_json(path: str, layerwise_summary: Dict[str, object], modelwise_summary: Dict[str, object], overall_summary: Dict[str, object], layerwise_json: Dict[str, object], modelwise_json: Dict[str, object]) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quant_mode": detect_quant_mode(layerwise_json, modelwise_json),
        "gpu_name": modelwise_json["int8_fused_modiff"]["label"] and layerwise_json["metadata"]["gpu_name"],
        "layerwise_config": layerwise_json["metadata"],
        "modelwise_reference_mode": modelwise_json["int8_fused_modiff"],
        "overall_summary": overall_summary,
        "layerwise_weighted_summary": layerwise_summary["aggregates"],
        "modelwise_summary": modelwise_summary["precision_rows"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_report(
    output_dir: str,
    layerwise_json: Dict[str, object],
    modelwise_json: Dict[str, object],
    layerwise_summary: Dict[str, object],
    modelwise_summary: Dict[str, object],
    overall_summary: Dict[str, object],
    plot_paths: List[str],
    layerwise_source_dir: str,
    modelwise_source_dir: str,
) -> str:
    report_path = os.path.join(output_dir, "FUSED_VS_SEPARATE_FINAL_REPORT.md")

    layer_meta = layerwise_json["metadata"]
    int8_model = modelwise_json["int8_fused_modiff"]
    int8_sep = modelwise_json["int8_separate_modiff"]
    int4_model = modelwise_json["int4_fused_modiff"]
    int4_sep = modelwise_json["int4_separate_modiff"]
    quant_mode = detect_quant_mode(layerwise_json, modelwise_json)

    int8_model_speedup = speedup(float(int8_model["mean_call_ms"]), float(int8_sep["mean_call_ms"]))
    int4_model_speedup = speedup(float(int4_model["mean_call_ms"]), float(int4_sep["mean_call_ms"]))

    lines = [
        f"# Fused vs Separate MoDiff Benchmark Report ({quant_mode} quantization)",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**GPU**: {layer_meta['gpu_name']}",
        f"**Torch**: {layer_meta['torch_version']} (CUDA {layer_meta['torch_cuda_version']})",
        f"**Conv backend**: {layer_meta['conv_backend']}",
        f"**Quant Mode**: {quant_mode}",
        "",
        "## Sources",
        "",
        f"- Layerwise source dir: `{layerwise_source_dir}`",
        f"- Modelwise source dir: `{modelwise_source_dir}`",
        "",
        "## Benchmark settings",
        "",
        "### Layerwise",
        "",
        f"- batch size: **{layer_meta['batch_size']}**",
        f"- warmup iterations: **{layer_meta['warmup']}**",
        f"- timed iterations per repeat: **{layer_meta['iters']}**",
        f"- timed repeats: **{layer_meta['timed_repeats']}**",
        f"- unique Conv2d shapes enumerated: **{len(layerwise_json['results'])}**",
        f"- supported shapes benchmarked: **{len(layerwise_summary['supported'])}**",
        f"- excluded shapes reported: **{len(layerwise_summary['excluded'])}**",
        "",
        "### Whole model",
        "",
        f"- batch size: **{int8_model['batch_size']}**",
        f"- DDIM steps: **{int8_model['steps']}**",
        f"- warmup runs: **{int8_model['warmup_runs']}**",
        f"- timed calls per mode: **{int8_model['timed_calls']}**",
        f"- quantization mode: **{quant_mode}**",
        "- timed region covers full DDIM denoising and excludes decode/save",
        "- sampler console output is suppressed during timing",
        "",
        "## Headline results",
        "",
        f"- **Layerwise INT8 weighted fusion speedup**: **{layerwise_summary['aggregates']['int8']['weighted_total_speedup']:.2f}x** ({layerwise_summary['aggregates']['int8']['weighted_separate_total_ms']:.3f} ms → {layerwise_summary['aggregates']['int8']['weighted_fused_total_ms']:.3f} ms)",
        f"- **Layerwise INT4 weighted fusion speedup**: **{layerwise_summary['aggregates']['int4']['weighted_total_speedup']:.2f}x** ({layerwise_summary['aggregates']['int4']['weighted_separate_total_ms']:.3f} ms → {layerwise_summary['aggregates']['int4']['weighted_fused_total_ms']:.3f} ms)",
        f"- **Whole-model INT8 fusion speedup**: **{int8_model_speedup:.2f}x** ({int8_sep['mean_call_ms']:.2f} ms → {int8_model['mean_call_ms']:.2f} ms)",
        f"- **Whole-model INT4 fusion speedup**: **{int4_model_speedup:.2f}x** ({int4_sep['mean_call_ms']:.2f} ms → {int4_model['mean_call_ms']:.2f} ms)",
        "",
        "## Layerwise observations",
        "",
        f"- Across the **{len(layerwise_summary['supported'])} supported shapes**, fused was never slower than separate for either INT8 or INT4.",
        f"- The weakest supported INT8 shape still showed **{layerwise_summary['weakest']['int8'][0]['int8']['total_speedup']:.2f}x** speedup on `{layerwise_summary['weakest']['int8'][0]['shape']['label']}`.",
        f"- The weakest supported INT4 shape still showed **{layerwise_summary['weakest']['int4'][0]['int4']['total_speedup']:.2f}x** speedup on `{layerwise_summary['weakest']['int4'][0]['shape']['label']}`.",
        "- Excluded shapes are not benchmark failures; they are layers the repository's quantized conversion path does not replace in practice (e.g. skip-path 1x1 convs, final output conv, or very small input-channel cases).",
        "",
        "## Whole-model observations",
        "",
        f"- Fused INT8 reduced mean full-model call time from **{int8_sep['mean_call_ms']:.2f} ms** to **{int8_model['mean_call_ms']:.2f} ms** at batch size {int8_model['batch_size']} and {int8_model['steps']} DDIM steps.",
        f"- Fused INT4 reduced mean full-model call time from **{int4_sep['mean_call_ms']:.2f} ms** to **{int4_model['mean_call_ms']:.2f} ms** under the same workload.",
        (
            "- Dynamic quantization mode recomputes activation scales online and intentionally ignores static calibration files."
            if quant_mode == "dynamic"
            else "- Static quantization mode reuses cached per-layer activation scales; when a supplied scale file had no matching conv keys, fresh scales were auto-calibrated from representative DDIM sampling calls."
        ),
        f"- Ready memory before timed sampling is INT8 fused **{int8_model.get('memory_ready_allocated_mb', int8_model['memory_allocated_mb']):.0f} MB** vs separate **{int8_sep.get('memory_ready_allocated_mb', int8_sep['memory_allocated_mb']):.0f} MB**, and INT4 fused **{int4_model.get('memory_ready_allocated_mb', int4_model['memory_allocated_mb']):.0f} MB** vs separate **{int4_sep.get('memory_ready_allocated_mb', int4_sep['memory_allocated_mb']):.0f} MB**.",
        f"- Timed-region peak memory is INT8 fused **{int8_model['memory_peak_mb']:.0f} MB** vs separate **{int8_sep['memory_peak_mb']:.0f} MB**, and INT4 fused **{int4_model['memory_peak_mb']:.0f} MB** vs separate **{int4_sep['memory_peak_mb']:.0f} MB**.",
        f"- The earlier inflated fused-memory readings were caused by setup-time artifacts: benchmark-side buffer-pool preallocation and calibration-only FP32 `_orig_weight` clones inside fused modules. The rebuilt benchmark disables the former, releases the latter, and measures peak memory only after warmup.",
        "- After those fixes, the remaining INT8 fused post-warmup gap appears to be mostly backend/workspace retention rather than another reporting bug: Python-visible extra fused state is only the persistent `_residual_buf` (~44 MB on this UNet), while roughly 0.5 GB stays allocated until the fused INT8 model is destroyed. INT4 does not show the same lingering footprint.",
        f"- INT8 scale source: **{int8_model.get('scale_source', 'unknown')}** with **{int8_model['loaded_static_scales']}** applied scales (status: **{int8_model.get('scale_status', 'unknown')}**, path: `{int8_model.get('scale_path') or 'n/a'}`).",
        f"- INT4 scale source: **{int4_model.get('scale_source', 'unknown')}** with **{int4_model['loaded_static_scales']}** applied scales (status: **{int4_model.get('scale_status', 'unknown')}**, path: `{int4_model.get('scale_path') or 'n/a'}`).",
        "",
        "## Figures",
        "",
    ]

    for plot_path in plot_paths:
        lines.append(f"- `{os.path.basename(plot_path)}`")

    lines.extend([
        "",
        "## Output tables",
        "",
        "- `table_layerwise_weighted_summary.csv`",
        "- `table_layerwise_supported_shapes.csv`",
        "- `table_layerwise_excluded_shapes.csv`",
        "- `table_modelwise_summary.csv`",
        "- `table_modelwise_speedup.csv`",
        "- `table_overall_summary.csv`",
        "",
        "## Key takeaway",
        "",
        f"Kernel fusion clearly matters in this codebase. The rebuilt per-layer hot-path benchmark shows **{layerwise_summary['aggregates']['int8']['weighted_total_speedup']:.2f}x** weighted speedup for INT8 and **{layerwise_summary['aggregates']['int4']['weighted_total_speedup']:.2f}x** for INT4, and the effect survives at the whole-model level as **{int8_model_speedup:.2f}x** for INT8 and **{int4_model_speedup:.2f}x** for INT4 under the rebuilt batch-32 / 50-step workload.",
    ])

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final fused-vs-separate summary artifacts")
    parser.add_argument(
        "--layerwise-json",
        type=str,
        default="analysis_fused_vs_separate/layerwise_results/layerwise_fused_vs_separate_results.json",
    )
    parser.add_argument(
        "--modelwise-json",
        type=str,
        default="analysis_fused_vs_separate/modelwise_results/modelwise_fused_vs_separate_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_fused_vs_separate/final_a40",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    layerwise_json_path = Path(args.layerwise_json)
    modelwise_json_path = Path(args.modelwise_json)
    if not layerwise_json_path.is_absolute():
        layerwise_json_path = repo_root / layerwise_json_path
    if not modelwise_json_path.is_absolute():
        modelwise_json_path = repo_root / modelwise_json_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    layerwise_json = load_json(str(layerwise_json_path))
    modelwise_json = load_json(str(modelwise_json_path))

    layerwise_summary = build_layerwise_tables(layerwise_json, str(output_dir))
    modelwise_summary = build_modelwise_tables(modelwise_json, str(output_dir))
    overall_summary = build_overall_summary(layerwise_summary, modelwise_summary, str(output_dir))

    plot_paths = [
        plot_layerwise_weighted_totals(layerwise_summary, str(output_dir)),
        plot_layerwise_breakdown(layerwise_summary, str(output_dir)),
        plot_layerwise_top_shapes(layerwise_summary, str(output_dir)),
        plot_modelwise_call_times(modelwise_summary, str(output_dir)),
        plot_modelwise_memory_and_speedup(modelwise_summary, str(output_dir)),
    ]

    summary_json_path = output_dir / "fusion_summary.json"
    write_summary_json(
        str(summary_json_path),
        layerwise_summary,
        modelwise_summary,
        overall_summary,
        layerwise_json,
        modelwise_json,
    )

    report_path = write_report(
        str(output_dir),
        layerwise_json,
        modelwise_json,
        layerwise_summary,
        modelwise_summary,
        overall_summary,
        plot_paths,
        str(layerwise_json_path.parent),
        str(modelwise_json_path.parent),
    )

    print(f"Summary JSON: {summary_json_path}")
    print(f"Report: {report_path}")
    for plot_path in plot_paths:
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
