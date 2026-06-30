#!/usr/bin/env python3
"""Create CUDA memcpy I/O tables and SVG plots from Nsight SQLite exports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MEMCPY_KIND_NAMES = {
    1: "Host-to-Device",
    2: "Device-to-Host",
    3: "Device-to-Device",
    4: "Peer-to-Peer",
    8: "Device-to-Device",
}


def fmt(value: float) -> str:
    return f"{value:,.1f}"


def mode_sqlite(profile_dir: Path, mode: str, steps: int, batch_size: int) -> Path:
    return profile_dir / f"{mode}_s{steps}_b{batch_size}.sqlite"


def read_memcpy(sqlite_path: Path) -> List[Dict[str, object]]:
    con = sqlite3.connect(sqlite_path)
    try:
        strings = {
            int(row[0]): row[1]
            for row in con.execute("SELECT id, value FROM StringIds")
        }
        runtime = {
            int(row[0]): strings.get(int(row[1]), str(row[1]))
            for row in con.execute(
                "SELECT correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            )
            if row[0] is not None
        }
        rows = []
        for start, end, bytes_, copy_kind, corr in con.execute(
            """
            SELECT start, end, bytes, copyKind, correlationId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        ):
            rows.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "bytes": int(bytes_),
                    "mib": int(bytes_) / 1024 / 1024,
                    "kind": MEMCPY_KIND_NAMES.get(int(copy_kind), str(copy_kind)),
                    "runtime": runtime.get(int(corr), "unknown") if corr is not None else "unknown",
                }
            )
        return rows
    finally:
        con.close()


def summarize_by_kind(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "mib": 0.0, "ms": 0.0})
    for row in rows:
        item = summary[str(row["kind"])]
        item["count"] += 1
        item["mib"] += float(row["mib"])
        item["ms"] += (int(row["end"]) - int(row["start"])) / 1_000_000.0
    return dict(summary)


def summarize_by_runtime(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "mib": 0.0, "ms": 0.0})
    for row in rows:
        item = summary[str(row["runtime"])]
        item["count"] += 1
        item["mib"] += float(row["mib"])
        item["ms"] += (int(row["end"]) - int(row["start"])) / 1_000_000.0
    return dict(summary)


def d2d_size_buckets(rows: Iterable[Dict[str, object]]) -> Counter[int]:
    return Counter(int(row["bytes"]) for row in rows if row["kind"] == "Device-to-Device")


def d2d_size_runtime_names(rows: Iterable[Dict[str, object]]) -> Dict[int, Counter[str]]:
    names: Dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["kind"] == "Device-to-Device":
            names[int(row["bytes"])][str(row["runtime"])] += 1
    return dict(names)


def summarize_name_counts(counter: Counter[str], limit: int = 3) -> str:
    if not counter:
        return "unknown"
    parts = []
    for name, count in counter.most_common(limit):
        parts.append(f"{name} ({count})")
    remaining = sum(counter.values()) - sum(count for _, count in counter.most_common(limit))
    if remaining:
        parts.append(f"other ({remaining})")
    return ", ".join(parts)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_bar_chart(
    path: Path,
    title: str,
    labels: List[str],
    series: List[Tuple[str, List[float], str]],
    ylabel: str,
    width: int = 900,
    height: int = 420,
) -> None:
    margin_l, margin_r, margin_t, margin_b = 80, 30, 60, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_v = max((v for _, values, _ in series for v in values), default=1.0)
    max_v = max(max_v, 1.0)
    group_w = plot_w / max(len(labels), 1)
    bar_w = group_w / (len(series) + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.label{font-size:11px}</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{width/2}" y="28" text-anchor="middle">{html.escape(title)}</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">{html.escape(ylabel)}</text>',
    ]
    for i in range(6):
        y = margin_t + plot_h - (plot_h * i / 5)
        val = max_v * i / 5
        parts.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{margin_l-8}" y="{y+4:.1f}" text-anchor="end">{fmt(val)}</text>')
    parts.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}"/>')
    parts.append(f'<line class="axis" x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}"/>')
    for idx, label in enumerate(labels):
        x0 = margin_l + idx * group_w + (group_w - bar_w * len(series)) / 2
        for sidx, (name, values, color) in enumerate(series):
            value = values[idx]
            h = plot_h * value / max_v
            x = x0 + sidx * bar_w
            y = margin_t + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" fill="{color}"/>')
            if value > 0:
                parts.append(f'<text class="label" x="{x+bar_w*0.41:.1f}" y="{y-4:.1f}" text-anchor="middle">{fmt(value)}</text>')
        lx = margin_l + idx * group_w + group_w / 2
        parts.append(f'<text x="{lx:.1f}" y="{height-margin_b+18}" text-anchor="middle">{html.escape(label)}</text>')
    legend_x = margin_l
    legend_y = height - 35
    for name, _, color in series:
        parts.append(f'<rect x="{legend_x}" y="{legend_y-12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{legend_y}">{html.escape(name)}</text>')
        legend_x += 170
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_stacked_chart(path: Path, labels: List[str], values_by_kind: Dict[str, List[float]]) -> None:
    colors = {
        "Host-to-Device": "#4c78a8",
        "Device-to-Host": "#f58518",
        "Device-to-Device": "#54a24b",
        "Peer-to-Peer": "#b279a2",
    }
    series = [(kind, values_by_kind[kind], colors.get(kind, "#999")) for kind in values_by_kind]
    svg_bar_chart(path, "CUDA memcpy traffic by kind", labels, series, "MiB")


def write_markdown(
    path: Path,
    modes: List[str],
    steps: int,
    by_mode: Dict[str, Dict[str, object]],
    extra_vs_fp16: Dict[str, Dict[str, float]],
    top_d2d: Dict[str, List[Dict[str, object]]],
) -> None:
    lines = [
        "# Nsight CUDA I/O Usage Analysis",
        "",
        f"Profiles: `{by_mode[modes[0]]['sqlite']}` ...",
        f"DDIM steps: `{steps}`",
        "",
        "## Total CUDA I/O",
        "",
        "| Mode | Total MiB | Total Count | Total ms | D2D MiB | D2D Count | Extra MiB vs FP16 | Extra D2D MiB vs FP16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        total = by_mode[mode]["total"]
        d2d = by_mode[mode]["kind"].get("Device-to-Device", {"mib": 0.0, "count": 0})
        extra = extra_vs_fp16.get(mode, {})
        lines.append(
            f"| {mode} | {fmt(total['mib'])} | {int(total['count'])} | {fmt(total['ms'])} | "
            f"{fmt(d2d['mib'])} | {int(d2d['count'])} | {fmt(extra.get('total_mib', 0.0))} | {fmt(extra.get('d2d_mib', 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## Memcpy Kind Breakdown",
            "",
            "| Mode | Kind | MiB | Count | ms |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for mode in modes:
        for kind, row in sorted(by_mode[mode]["kind"].items(), key=lambda kv: kv[1]["mib"], reverse=True):
            lines.append(f"| {mode} | {kind} | {fmt(row['mib'])} | {int(row['count'])} | {fmt(row['ms'])} |")

    lines.extend(
        [
            "",
            "## Runtime Count And Time",
            "",
            "This table names the CUDA runtime API associated with the memcpy events. In this capture all recorded CUDA memcpy traffic is issued through `cudaMemcpyAsync_v3020`.",
            "",
            "| Mode | Runtime name | MiB | Count | ms |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for mode in modes:
        for runtime, row in sorted(by_mode[mode]["runtime"].items(), key=lambda kv: kv[1]["mib"], reverse=True):
            lines.append(
                f"| {mode} | `{runtime}` | {fmt(row['mib'])} | {int(row['count'])} | {fmt(row['ms'])} |"
            )

    lines.extend(
        [
            "",
            "## Largest D2D Size Buckets",
            "",
            "These buckets show repeated GPU-to-GPU copies with identical byte sizes. Repetition across 50 DDIM steps exposes per-step copy patterns.",
            "",
        ]
    )
    for mode in modes:
        lines.extend(
            [
                f"### {mode}",
                "",
                "| Count | Recorded API name | MiB Each | Total MiB | Approx count/step |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in top_d2d[mode]:
            size = int(row["size"])
            count = int(row["count"])
            lines.append(
                f"| {count} | `{row['runtime_names']}` | {fmt(size / 1024 / 1024)} | "
                f"{fmt(size * count / 1024 / 1024)} | {count / steps:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Plots",
            "",
            "### Total CUDA memcpy I/O",
            "",
            "![Total CUDA memcpy I/O](plots/total_cuda_io.png)",
            "",
            "### CUDA memcpy I/O by transfer kind",
            "",
            "![CUDA memcpy I/O by transfer kind](plots/cuda_io_by_kind.png)",
            "",
            "### D2D memcpy event count",
            "",
            "![D2D memcpy event count](plots/d2d_count.png)",
            "",
            "### Baseline D2D copies by repeated tensor size",
            "",
            "![Baseline D2D copies by repeated tensor size](plots/d2d_top_sizes_baselines.png)",
            "",
            "### D2D traffic by repeated tensor-size bucket",
            "",
            "![D2D traffic by repeated tensor-size bucket](plots/d2d_size_heatmap.png)",
            "",
            "## Interpretation",
            "",
            "- `int4` and `int4_baseline` have very different resident tracked memory, but similar total CUDA memcpy I/O.",
            "- Baseline extra I/O versus FP16 is almost entirely D2D, meaning on-GPU tensor movement rather than host transfer.",
            "- INT8 MoDiff has the largest D2D because its MoDiff static path keeps cache updates plus INT8 quantized-island movement.",
            "- INT4 MoDiff and INT4 baseline are close in memcpy bytes because both run the same low-bit island pattern; the MoDiff cache cost appears mainly as resident cache memory, not a large additional memcpy volume.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_matplotlib_plots(output_dir: Path, modes: List[str], by_mode: Dict[str, Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plot_dir = output_dir / "plots"
    sns.set_theme(style="whitegrid", context="talk")

    total_df = pd.DataFrame(
        [
            {
                "mode": mode,
                "total_mib": by_mode[mode]["total"]["mib"],
                "d2d_mib": by_mode[mode]["kind"].get("Device-to-Device", {"mib": 0.0})["mib"],
                "d2d_count": by_mode[mode]["kind"].get("Device-to-Device", {"count": 0})["count"],
            }
            for mode in modes
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=total_df, x="mode", y="total_mib", ax=ax, color="#4c78a8")
    ax.set_title("Total CUDA memcpy I/O")
    ax.set_xlabel("")
    ax.set_ylabel("MiB")
    ax.tick_params(axis="x", rotation=20)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / "total_cuda_io.png", dpi=180)
    plt.close(fig)

    kind_rows = []
    for mode in modes:
        for kind in ("Host-to-Device", "Device-to-Host", "Device-to-Device"):
            kind_rows.append(
                {
                    "mode": mode,
                    "kind": kind,
                    "mib": by_mode[mode]["kind"].get(kind, {"mib": 0.0})["mib"],
                }
            )
    kind_df = pd.DataFrame(kind_rows)
    pivot = kind_df.pivot(index="mode", columns="kind", values="mib").reindex(modes)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    pivot[["Host-to-Device", "Device-to-Host", "Device-to-Device"]].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#4c78a8", "#f58518", "#54a24b"],
    )
    ax.set_title("CUDA memcpy I/O by transfer kind")
    ax.set_xlabel("")
    ax.set_ylabel("MiB")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(plot_dir / "cuda_io_by_kind.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=total_df, x="mode", y="d2d_count", ax=ax, color="#54a24b")
    ax.set_title("D2D memcpy event count")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / "d2d_count.png", dpi=180)
    plt.close(fig)

    size_rows = []
    size_set = set()
    for mode in modes:
        for size, count in by_mode[mode]["d2d_sizes"].most_common(16):
            size_set.add(size)
    top_sizes = sorted(size_set, reverse=True)[:12]
    for mode in modes:
        counts = by_mode[mode]["d2d_sizes"]
        for size in top_sizes:
            size_rows.append(
                {
                    "mode": mode,
                    "size_mib": size / 1024 / 1024,
                    "size_label": f"{size / 1024 / 1024:.3f}",
                    "count": counts.get(size, 0),
                    "total_mib": counts.get(size, 0) * size / 1024 / 1024,
                }
            )
    size_df = pd.DataFrame(size_rows)
    base_df = size_df[size_df["mode"].isin(["int8_baseline", "int4_baseline"]) & (size_df["count"] > 0)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=base_df, x="size_label", y="count", hue="mode", ax=ax)
    ax.set_title("Baseline D2D copies by repeated tensor size")
    ax.set_xlabel("Memcpy size (MiB)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plot_dir / "d2d_top_sizes_baselines.png", dpi=180)
    plt.close(fig)

    heat = size_df.pivot(index="mode", columns="size_label", values="total_mib").reindex(modes).fillna(0)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Total MiB"})
    ax.set_title("D2D traffic by repeated tensor-size bucket")
    ax.set_xlabel("Memcpy size (MiB)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(plot_dir / "d2d_size_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=168)
    parser.add_argument("--modes", nargs="+", default=["fp16", "int8", "int8_baseline", "int4", "int4_baseline"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    by_mode: Dict[str, Dict[str, object]] = {}
    for mode in args.modes:
        sqlite_path = mode_sqlite(args.profile_dir, mode, args.steps, args.batch_size)
        rows = read_memcpy(sqlite_path)
        kind = summarize_by_kind(rows)
        runtime = summarize_by_runtime(rows)
        total = {
            "count": sum(v["count"] for v in kind.values()),
            "mib": sum(v["mib"] for v in kind.values()),
            "ms": sum(v["ms"] for v in kind.values()),
        }
        by_mode[mode] = {
            "sqlite": str(sqlite_path),
            "kind": kind,
            "runtime": runtime,
            "total": total,
            "d2d_sizes": d2d_size_buckets(rows),
            "d2d_size_runtime_names": d2d_size_runtime_names(rows),
        }

    fp16 = by_mode["fp16"]
    extra_vs_fp16 = {}
    fp16_d2d = fp16["kind"].get("Device-to-Device", {"mib": 0.0, "count": 0})
    for mode in args.modes:
        total = by_mode[mode]["total"]
        d2d = by_mode[mode]["kind"].get("Device-to-Device", {"mib": 0.0, "count": 0})
        extra_vs_fp16[mode] = {
            "total_mib": total["mib"] - fp16["total"]["mib"],
            "total_count": total["count"] - fp16["total"]["count"],
            "d2d_mib": d2d["mib"] - fp16_d2d["mib"],
            "d2d_count": d2d["count"] - fp16_d2d["count"],
        }

    summary_json = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "modes": by_mode,
        "extra_vs_fp16": extra_vs_fp16,
    }
    (args.output_dir / "cuda_io_usage_summary.json").write_text(json.dumps(summary_json, indent=2))

    total_rows = []
    kind_rows = []
    runtime_rows = []
    top_d2d: Dict[str, List[Dict[str, object]]] = {}
    for mode in args.modes:
        total = by_mode[mode]["total"]
        d2d = by_mode[mode]["kind"].get("Device-to-Device", {"mib": 0.0, "count": 0})
        extra = extra_vs_fp16[mode]
        total_rows.append({
            "mode": mode,
            "total_mib": f"{total['mib']:.6f}",
            "total_count": int(total["count"]),
            "total_ms": f"{total['ms']:.6f}",
            "d2d_mib": f"{d2d['mib']:.6f}",
            "d2d_count": int(d2d["count"]),
            "extra_mib_vs_fp16": f"{extra['total_mib']:.6f}",
            "extra_d2d_mib_vs_fp16": f"{extra['d2d_mib']:.6f}",
        })
        for kind, row in by_mode[mode]["kind"].items():
            kind_rows.append({
                "mode": mode,
                "kind": kind,
                "mib": f"{row['mib']:.6f}",
                "count": int(row["count"]),
                "ms": f"{row['ms']:.6f}",
            })
        for runtime, row in by_mode[mode]["runtime"].items():
            runtime_rows.append({
                "mode": mode,
                "runtime": runtime,
                "mib": f"{row['mib']:.6f}",
                "count": int(row["count"]),
                "ms": f"{row['ms']:.6f}",
            })
        top_d2d[mode] = [
            {
                "size": size,
                "count": count,
                "runtime_names": summarize_name_counts(
                    by_mode[mode]["d2d_size_runtime_names"].get(size, Counter())
                ),
            }
            for size, count in by_mode[mode]["d2d_sizes"].most_common(12)
        ]

    write_csv(args.output_dir / "cuda_io_total.csv", total_rows, list(total_rows[0].keys()))
    write_csv(args.output_dir / "cuda_io_by_kind.csv", kind_rows, list(kind_rows[0].keys()))
    write_csv(args.output_dir / "cuda_io_by_runtime.csv", runtime_rows, list(runtime_rows[0].keys()))

    labels = args.modes
    svg_bar_chart(
        plot_dir / "total_cuda_io.svg",
        "Total CUDA memcpy I/O",
        labels,
        [("Total MiB", [by_mode[m]["total"]["mib"] for m in labels], "#4c78a8")],
        "MiB",
    )
    kinds = ["Host-to-Device", "Device-to-Host", "Device-to-Device"]
    svg_stacked_chart(
        plot_dir / "cuda_io_by_kind.svg",
        labels,
        {kind: [by_mode[m]["kind"].get(kind, {"mib": 0.0})["mib"] for m in labels] for kind in kinds},
    )
    svg_bar_chart(
        plot_dir / "d2d_count.svg",
        "D2D memcpy event count",
        labels,
        [("D2D Count", [by_mode[m]["kind"].get("Device-to-Device", {"count": 0})["count"] for m in labels], "#54a24b")],
        "Count",
    )

    baseline_labels = []
    int8_values = []
    int4_values = []
    sizes = sorted(
        {int(row["size"]) for row in top_d2d["int8_baseline"]}
        | {int(row["size"]) for row in top_d2d["int4_baseline"]},
        reverse=True,
    )[:8]
    int8_counter = by_mode["int8_baseline"]["d2d_sizes"]
    int4_counter = by_mode["int4_baseline"]["d2d_sizes"]
    for size in sizes:
        baseline_labels.append(f"{size / 1024 / 1024:.2f}MiB")
        int8_values.append(int8_counter.get(size, 0))
        int4_values.append(int4_counter.get(size, 0))
    svg_bar_chart(
        plot_dir / "d2d_top_sizes_baselines.svg",
        "Baseline D2D copies by repeated tensor size",
        baseline_labels,
        [("INT8 baseline count", int8_values, "#4c78a8"), ("INT4 baseline count", int4_values, "#f58518")],
        "Count",
    )

    write_matplotlib_plots(args.output_dir, args.modes, by_mode)
    write_markdown(args.output_dir / "CUDA_IO_USAGE_ANALYSIS.md", args.modes, args.steps, by_mode, extra_vs_fp16, top_d2d)
    print(f"Wrote {args.output_dir / 'CUDA_IO_USAGE_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
