#!/usr/bin/env python3
"""Summarize Nsight Systems CUDA memory/copy data for MoDiff memory redo."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MEMCPY_KIND_NAMES = {
    1: "Host-to-Device",
    2: "Device-to-Host",
    3: "Device-to-Device",
    4: "Peer-to-Peer",
    # Nsight Systems 2024.1.1 exports CUDA device-to-device copies as 8
    # in CUPTI_ACTIVITY_KIND_MEMCPY.copyKind.
    8: "Device-to-Device",
}


def table_names(conn: sqlite3.Connection) -> List[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def find_table(conn: sqlite3.Connection, candidates: Iterable[str]) -> Optional[str]:
    existing = set(table_names(conn))
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def query_memcpy(conn: sqlite3.Connection) -> Dict[str, Dict[str, float]]:
    table = find_table(conn, ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUDA_MEMCPY"])
    if table is None:
        return {}

    cols = set(table_columns(conn, table))
    bytes_col = "bytes" if "bytes" in cols else ("copySize" if "copySize" in cols else None)
    start_col = "start" if "start" in cols else None
    end_col = "end" if "end" in cols else None
    kind_col = "copyKind" if "copyKind" in cols else ("srcKind" if "srcKind" in cols else None)

    if bytes_col is None or kind_col is None:
        return {}

    duration_expr = "0"
    if start_col and end_col:
        duration_expr = f"SUM(({end_col} - {start_col}) / 1000000.0)"

    rows = conn.execute(
        f"""
        SELECT {kind_col} AS kind,
               COUNT(*) AS count,
               SUM({bytes_col}) / 1024.0 / 1024.0 AS total_mib,
               {duration_expr} AS total_ms
        FROM "{table}"
        GROUP BY {kind_col}
        ORDER BY total_mib DESC
        """
    ).fetchall()

    result: Dict[str, Dict[str, float]] = {}
    for kind, count, total_mib, total_ms in rows:
        name = MEMCPY_KIND_NAMES.get(kind, str(kind))
        result[name] = {
            "count": int(count or 0),
            "total_mib": float(total_mib or 0.0),
            "total_ms": float(total_ms or 0.0),
        }
    return result


def total_memcpy(memcpy: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {
        "count": int(sum(row.get("count", 0) for row in memcpy.values())),
        "total_mib": float(sum(row.get("total_mib", 0.0) for row in memcpy.values())),
        "total_ms": float(sum(row.get("total_ms", 0.0) for row in memcpy.values())),
    }


def string_map(conn: sqlite3.Connection) -> Dict[int, str]:
    if "StringIds" not in table_names(conn):
        return {}
    cols = set(table_columns(conn, "StringIds"))
    id_col = "id" if "id" in cols else None
    value_col = "value" if "value" in cols else ("name" if "name" in cols else None)
    if id_col is None or value_col is None:
        return {}
    return {int(row[0]): row[1] for row in conn.execute(f"SELECT {id_col}, {value_col} FROM StringIds")}


def query_kernels(conn: sqlite3.Connection, limit: int = 20) -> List[Dict[str, Any]]:
    table = find_table(conn, ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_KERNEL_NAMED"])
    if table is None:
        return []

    cols = set(table_columns(conn, table))
    if "start" not in cols or "end" not in cols:
        return []

    name_col = None
    name_is_string_id = False
    for candidate in ("demangledName", "shortName", "mangledName", "name"):
        if candidate in cols:
            name_col = candidate
            name_is_string_id = candidate != "name"
            break
    if name_col is None:
        return []

    sid = string_map(conn) if name_is_string_id else {}
    rows = conn.execute(
        f"""
        SELECT {name_col} AS kernel_name,
               COUNT(*) AS count,
               SUM((end - start) / 1000000.0) AS total_ms,
               AVG((end - start) / 1000.0) AS avg_us
        FROM "{table}"
        GROUP BY {name_col}
        ORDER BY total_ms DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    result = []
    for raw_name, count, total_ms, avg_us in rows:
        name = sid.get(int(raw_name), str(raw_name)) if name_is_string_id and raw_name is not None else str(raw_name)
        result.append(
            {
                "kernel": name,
                "count": int(count or 0),
                "total_ms": float(total_ms or 0.0),
                "avg_us": float(avg_us or 0.0),
            }
        )
    return result


def query_osrt_io(conn: sqlite3.Connection) -> Dict[str, Dict[str, float]]:
    if "OSRT_API" not in table_names(conn) or "StringIds" not in table_names(conn):
        return {}

    cols = set(table_columns(conn, "OSRT_API"))
    if not {"start", "end", "nameId"}.issubset(cols):
        return {}

    names = ("read", "write", "fread", "fwrite", "fgets", "fputs", "open", "open64", "fopen", "fopen64")
    placeholders = ", ".join("?" for _ in names)
    rows = conn.execute(
        f"""
        SELECT s.value AS name,
               COUNT(*) AS count,
               SUM((o.end - o.start) / 1000000.0) AS total_ms,
               AVG((o.end - o.start) / 1000.0) AS avg_us
        FROM OSRT_API o
        JOIN StringIds s ON o.nameId = s.id
        WHERE s.value IN ({placeholders})
        GROUP BY s.value
        ORDER BY total_ms DESC
        """,
        names,
    ).fetchall()

    result: Dict[str, Dict[str, float]] = {}
    for name, count, total_ms, avg_us in rows:
        result[str(name)] = {
            "count": int(count or 0),
            "total_ms": float(total_ms or 0.0),
            "avg_us": float(avg_us or 0.0),
        }
    return result


def load_benchmark_result(benchmark_dir: Path, mode: str) -> Dict[str, Any]:
    candidates = [
        benchmark_dir / mode / "results.json",
        benchmark_dir / f"{mode}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        if mode in data and isinstance(data[mode], dict):
            return data[mode]
        if isinstance(data, dict):
            return data
    return {}


def summarize_profile(sqlite_path: Path) -> Dict[str, Any]:
    if not sqlite_path.exists():
        return {"error": f"missing sqlite: {sqlite_path}"}
    conn = sqlite3.connect(str(sqlite_path))
    try:
        memcpy = query_memcpy(conn)
        return {
            "sqlite": str(sqlite_path),
            "memcpy": memcpy,
            "memcpy_total": total_memcpy(memcpy),
            "osrt_io": query_osrt_io(conn),
            "top_kernels": query_kernels(conn),
        }
    finally:
        conn.close()


def fmt_mb(value: Any) -> str:
    try:
        return f"{float(value):,.1f}"
    except Exception:
        return "-"


def write_markdown(summary: Dict[str, Any], output_md: Path) -> None:
    lines = [
        "# Nsight Memory Redo Summary",
        "",
        "This report is generated by `integration/benchmarks/analyze_nsys_memory.py`.",
        "",
        "## Benchmark Memory",
        "",
        "| Mode | Allocated MB | Peak MB | Peak - Allocated MB | Tracked Quant MiB | Cache/Residual MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for mode, data in summary["modes"].items():
        bench = data.get("benchmark", {})
        quant = bench.get("quant_memory_after_warmup", {})
        alloc = bench.get("memory_allocated_mb")
        peak = bench.get("memory_peak_mb")
        delta = None
        if alloc is not None and peak is not None:
            delta = float(peak) - float(alloc)
        lines.append(
            f"| {mode} | {fmt_mb(alloc)} | {fmt_mb(peak)} | {fmt_mb(delta)} | "
            f"{fmt_mb(quant.get('total_tracked_mib'))} | {fmt_mb(quant.get('cache_and_residual_mib'))} |"
        )

    lines.extend(["", "## Total CUDA I/O", ""])
    lines.extend(
        [
            "Total CUDA I/O is the sum of all Nsight CUDA memcpy traffic: Host-to-Device + Device-to-Host + Device-to-Device.",
            "",
            "| Mode | Total Count | Total MiB | Total ms | D2D MiB | D2D Count |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, data in summary["modes"].items():
        nsys = data.get("nsys", {})
        total = nsys.get("memcpy_total", {})
        d2d = nsys.get("memcpy", {}).get("Device-to-Device", {})
        lines.append(
            f"| {mode} | {total.get('count', 0)} | {fmt_mb(total.get('total_mib'))} | "
            f"{fmt_mb(total.get('total_ms'))} | {fmt_mb(d2d.get('total_mib'))} | {d2d.get('count', 0)} |"
        )

    lines.extend(["", "## CUDA Memcpy Breakdown", ""])
    for mode, data in summary["modes"].items():
        lines.extend(
            [
                f"### {mode}",
                "",
                "| Kind | Count | Total MiB | Total ms |",
                "|---|---:|---:|---:|",
            ]
        )
        memcpy = data.get("nsys", {}).get("memcpy", {})
        if not memcpy:
            lines.append("| unavailable | - | - | - |")
        else:
            for kind, row in memcpy.items():
                lines.append(
                    f"| {kind} | {row.get('count', 0)} | {fmt_mb(row.get('total_mib'))} | {fmt_mb(row.get('total_ms'))} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Host OS Runtime I/O",
            "",
            "Nsight OSRT captures API counts and duration here; this export does not include read/write byte counts.",
            "",
        ]
    )
    for mode, data in summary["modes"].items():
        lines.extend(
            [
                f"### {mode}",
                "",
                "| API | Count | Total ms | Avg us |",
                "|---|---:|---:|---:|",
            ]
        )
        osrt_io = data.get("nsys", {}).get("osrt_io", {})
        if not osrt_io:
            lines.append("| unavailable | - | - | - |")
        else:
            for name, row in osrt_io.items():
                lines.append(
                    f"| {name} | {row.get('count', 0)} | {fmt_mb(row.get('total_ms'))} | {fmt_mb(row.get('avg_us'))} |"
                )
        lines.append("")

    lines.extend(["## Top Quant Memory Buckets", ""])
    for mode, data in summary["modes"].items():
        quant = data.get("benchmark", {}).get("quant_memory_after_warmup", {})
        buckets = quant.get("buckets_mib", {})
        lines.extend([f"### {mode}", "", "| Bucket | MiB |", "|---|---:|"])
        if not buckets:
            lines.append("| unavailable | - |")
        else:
            for key, value in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:12]:
                lines.append(f"| `{key}` | {fmt_mb(value)} |")
        lines.append("")

    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", default="integration/results/nsys_memory_redo/profiles")
    parser.add_argument("--benchmark-dir", default="integration/results/nsys_memory_redo/benchmarks")
    parser.add_argument("--output-json", default="integration/results/nsys_memory_redo/nsys_memory_summary.json")
    parser.add_argument("--output-md", default="integration/results/nsys_memory_redo/NSYS_MEMORY_REDO_REPORT.md")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["fp16", "int8", "int8_baseline", "int4", "int4_baseline"],
    )
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    benchmark_dir = Path(args.benchmark_dir)
    summary: Dict[str, Any] = {"modes": {}}

    for mode in args.modes:
        sqlite_matches = sorted(profile_dir.glob(f"{mode}_*.sqlite"))
        sqlite_path = sqlite_matches[-1] if sqlite_matches else profile_dir / f"{mode}.sqlite"
        summary["modes"][mode] = {
            "benchmark": load_benchmark_result(benchmark_dir, mode),
            "nsys": summarize_profile(sqlite_path),
        }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, output_md)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
