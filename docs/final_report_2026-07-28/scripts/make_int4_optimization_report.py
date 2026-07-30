"""Aggregate INT4 optimization phase benchmarks and render the final charts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/MoDiff/docs/final_report_2026-07-28")
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

SHAPES = [(192, 1024), (384, 256), (384, 64), (768, 16), (768, 4)]


def attention_rows(path, mode):
    doc = json.loads(Path(path).read_text())
    rows = [r for r in doc["modes"][mode] if r["kind"] == "attention"]
    by_shape = {(r["x_shape"][1], r["x_shape"][2] * r["x_shape"][3]): r for r in rows}
    return [by_shape[s] for s in SHAPES]


def summarize(path, mode):
    rows = attention_rows(path, mode)
    return {
        "weighted_ms": sum(r["pipeline_us"] * r["n_instances"] for r in rows) / 1000,
        "per_shape_us": [r["pipeline_us"] for r in rows],
        "counts": [r["n_instances"] for r in rows],
    }


common = DATA / "layer_pipeline_bench.json"
series = {
    "FP16": summarize(common, "fp16"),
    "Current INT8": summarize(common, "int8_baseline"),
    "Original INT4": summarize(DATA / "int4_phase_original.json", "int4_baseline"),
    "1. Fast GN/pack": summarize(DATA / "int4_phase_fast_gn.json", "int4_baseline"),
    "4–5. Compact scales + tail clear": summarize(
        DATA / "int4_phase_compact.json", "int4_baseline"),
    "2–3. Fused K/V + Q-in-flash": summarize(
        DATA / "int4_phase_qin_fused_kv.json", "int4_baseline"),
    "6. QKV epilogue (rejected)": summarize(
        DATA / "int4_phase_qkv_epilogue.json", "int4_baseline"),
}

micro = {
    "producer": [
        {"T": 1024, "hd": 24, "reference_ms": 0.4239835, "candidate_ms": 0.3794347},
        {"T": 256, "hd": 48, "reference_ms": 0.1704592, "candidate_ms": 0.1580283},
        {"T": 64, "hd": 48, "reference_ms": 0.0474139, "candidate_ms": 0.0422917},
    ],
    "full_qin_route": [
        {"T": 1024, "hd": 24, "reference_ms": 2.2025200, "candidate_ms": 2.0892634},
        {"T": 256, "hd": 48, "reference_ms": 0.4191317, "candidate_ms": 0.3648645},
        {"T": 64, "hd": 48, "reference_ms": 0.0888549, "candidate_ms": 0.0801387},
    ],
    "fast_gn_us": [
        {"C": 192, "T": 1024, "reference": 461.7, "candidate": 267.4},
        {"C": 384, "T": 256, "reference": 283.2, "candidate": 101.4},
        {"C": 384, "T": 64, "reference": 182.6, "candidate": 35.5},
        {"C": 768, "T": 16, "reference": 59.1, "candidate": 24.2},
        {"C": 768, "T": 4, "reference": 20.7, "candidate": 18.3},
    ],
}

result = {
    "hardware": "NVIDIA A40 (SM86)",
    "batch": 128,
    "protocol": "20 warmups; median of 5 rounds x 60 iterations",
    "shape_order": [{"C": c, "T": t} for c, t in SHAPES],
    "weighted_attention": series,
    "microbenchmarks": micro,
    "quality": {str(s): 0.0 for s in (1234, 5678, 9012)},
    "quality_threshold": 0.02,
}
(DATA / "int4_optimization_final.json").write_text(json.dumps(result, indent=2))

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11, 5.5))
labels = list(series)
vals = [series[k]["weighted_ms"] for k in labels]
colors = ["#8c96a3", "#4c78a8", "#e45756", "#72b7b2", "#f2cf5b", "#54a24b", "#b279a2"]
bars = ax.bar(np.arange(len(vals)), vals, color=colors)
ax.axhline(22.494, color="#222", linestyle="--", linewidth=1.5, label="INT8 target 22.494 ms")
ax.set_ylabel("21-block weighted attention latency (ms)")
ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
ax.set_ylim(0, max(vals) * 1.18)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + .35, f"{v:.2f}", ha="center", fontsize=9)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PLOTS / "int4_weighted_attention_phases.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(SHAPES))
selected = ["FP16", "Current INT8", "Original INT4", "1. Fast GN/pack",
            "2–3. Fused K/V + Q-in-flash", "6. QKV epilogue (rejected)"]
for name in selected:
    ax.plot(x, np.array(series[name]["per_shape_us"]) / 1000, marker="o", label=name)
ax.set_xticks(x, [f"C{c}/T{t}" for c, t in SHAPES])
ax.set_ylabel("Full attention-layer latency (ms)")
ax.set_yscale("log")
ax.legend(ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS / "int4_per_shape_phases.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
names, speedups = [], []
for group, rows in (("Fused K/V", micro["producer"]), ("Q-in full route", micro["full_qin_route"])):
    for r in rows:
        names.append(f"{group}\nT{r['T']}")
        speedups.append(r["reference_ms"] / r["candidate_ms"])
bars = ax.bar(np.arange(len(speedups)), speedups, color=["#72b7b2"]*3 + ["#54a24b"]*3)
ax.axhline(1.0, color="#222", linewidth=1)
ax.set_ylabel("Same-process speedup (×)")
ax.set_xticks(np.arange(len(names)), names)
for b, v in zip(bars, speedups):
    ax.text(b.get_x()+b.get_width()/2, v+.008, f"{v:.2f}×", ha="center", fontsize=9)
ax.set_ylim(.95, max(speedups) * 1.12)
fig.tight_layout()
fig.savefig(PLOTS / "int4_producer_qin_speedups.png", dpi=180)
plt.close(fig)

print(json.dumps({k: round(v["weighted_ms"], 4) for k, v in series.items()}, indent=2))
