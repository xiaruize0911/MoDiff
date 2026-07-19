"""Plots for the fresh pipeline profile (profile_pipeline_buckets.py). Reads
data/pipeline_buckets_b{B}.csv + pipeline_topkernels_b{B}.csv. Emits:
  18_pipeline_buckets.png    -- where the step time goes, fp16 vs int8 (Amdahl)
  19_pipeline_topkernels.png -- top kernels by device self-time (int8), colored by bucket
"""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = int(os.environ.get("E2E_BATCH", "64"))
D = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"
P = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16"
FP16, INT8 = "#888888", "#4C78A8"
# bucket -> color; memory-bound buckets get warm/highlight colors (the optimization targets)
BCOL = {
    "attention (softmax)": "#E45756", "elementwise / copy": "#F58518", "GroupNorm": "#EECA3B",
    "conv (GEMM)": "#54A24B", "qkv/proj GEMM (+ fp16 attn bmm)": "#4C78A8",
    "quantize / absmax": "#B279A2", "conv store epilogue": "#72B7B2",
    "upsample / concat": "#9D755D", "attn QKᵀ/AV (int GEMM)": "#72B7B2", "other": "#BAB0AC",
}
MEMBOUND = {"attention (softmax)", "elementwise / copy", "GroupNorm"}


def rd(name):
    p = f"{D}/{name}"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def save(fig, name):
    fig.savefig(f"{P}/{name}", dpi=120, bbox_inches="tight"); plt.close(fig); print("  wrote", name)


# ---- 18: bucket breakdown fp16 vs int8 (grouped horizontal bars) ----
rows = rd(f"pipeline_buckets_b{B}.csv")
by = {"fp16": {}, "int8": {}}
for r in rows:
    by[r["mode"]][r["bucket"]] = float(r["ms_step"])
buckets = sorted(set(by["fp16"]) | set(by["int8"]), key=lambda b: -max(by["fp16"].get(b, 0), by["int8"].get(b, 0)))
tot = {m: sum(by[m].values()) for m in ("fp16", "int8")}

y = np.arange(len(buckets))[::-1]; h = 0.38
fig, ax = plt.subplots(figsize=(11, 6.6))
for i, b in enumerate(buckets):
    f, q = by["fp16"].get(b, 0), by["int8"].get(b, 0)
    star = "  ★ memory-bound target" if b in MEMBOUND else ""
    ax.barh(y[i] + h / 2, f, h, color=FP16, label="fp16" if i == 0 else None)
    ax.barh(y[i] - h / 2, q, h, color=BCOL.get(b, INT8), label="int8" if i == 0 else None,
            edgecolor="#c00" if b in MEMBOUND else "none", linewidth=1.6 if b in MEMBOUND else 0)
    ax.text(f + 0.15, y[i] + h / 2, f"{f:.1f}  ({f/tot['fp16']*100:.0f}%)", va="center", fontsize=7.5, color="#555")
    ax.text(q + 0.15, y[i] - h / 2, f"{q:.1f}  ({q/tot['int8']*100:.0f}%)", va="center", fontsize=7.5,
            color="#c00" if b in MEMBOUND else "#333", fontweight="bold" if b in MEMBOUND else "normal")
ax.set_yticks(y); ax.set_yticklabels([b + ("  ★" if b in MEMBOUND else "") for b in buckets], fontsize=9)
ax.set_xlabel("device self-time (ms/step) — torch.profiler, batch 64")
ax.set_title(f"Where the step time goes (int8 BASELINE (no MoDiff cache), batch {B})\n"
             f"fp16 wall {[r for r in rd(f'pipeline_speed_b{B}.csv') if r['mode']=='fp16'][0]['wall_ms_step']} · "
             f"int8 wall {[r for r in rd(f'pipeline_speed_b{B}.csv') if r['mode']=='int8'][0]['wall_ms_step']} ms/step   "
             "·   ★ = memory-bound, the next targets", fontsize=10.5)
ax.legend(loc="lower right", fontsize=9); ax.set_xlim(0, max(tot.values()) * 0.30)
ax.margins(y=0.01); save(fig, "18_pipeline_buckets.png")

# ---- 19: top kernels (int8), colored by bucket ----
kr = [r for r in rd(f"pipeline_topkernels_b{B}.csv") if r["mode"] == "int8"][:12]
names, vals, cols = [], [], []
SHORT = {"softmax_warp_forward": "softmax (materialized attn)", "ImplicitGemmConv": "int8 conv (CUTLASS)",
         "wmma_tensorop_f16_s161616gemm_f16_32x32_32x1": "fp16 attn bmm (QKᵀ/AV) 32x1",
         "wmma_tensorop_f16_s161616gemm_f16_32x32_128x2": "fp16 attn bmm (QKᵀ/AV) 128x2",
         "group_norm_silu_nhwc": "GroupNorm(+SiLU)", "scale_accumulate_half_cache": "modiff cache accumulate",
         "static_quantize_and_update_ahat": "conv quantize+a_hat (SiLU)", "gemm_w8a8_kernel_awq": "int8 qkv/proj GEMM",
         "group_norm_silu_quantize_nhwc": "GN->int8 (qkv-fusion, NEW)", "CUDAFunctor_add": "elementwise add"}
def label(k):
    for key, v in SHORT.items():
        if key.lower() in k.lower(): return v
    if "elementwise_kernel" in k: return "elementwise"
    return k[:34]
for r in kr:
    names.append(label(r["kernel"])); vals.append(float(r["ms_step"])); cols.append(BCOL.get(r["bucket"], "#888"))
yy = np.arange(len(names))[::-1]
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(yy, vals, 0.7, color=cols)
gpu = sum(float(r["ms_step"]) for r in rd(f"pipeline_buckets_b{B}.csv") if r["mode"] == "int8")
for i, v in enumerate(vals):
    ax.text(v + 0.1, yy[i], f"{v:.1f} ms  ({v/gpu*100:.0f}%)", va="center", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("device self-time (ms/step)")
ax.set_title(f"Top kernels, int8 baseline pipeline (batch {B}) — the one softmax kernel is the single biggest\n"
             "attention (softmax + fp16 QKᵀ/AV bmm) ≈ 40% of the step; int8 qkv/proj GEMM is only ~3%", fontsize=10.5)
from matplotlib.patches import Patch
seen = {}
for r in kr: seen[r["bucket"]] = BCOL.get(r["bucket"], "#888")
ax.legend(handles=[Patch(color=c, label=b) for b, c in seen.items()], fontsize=8, loc="lower right")
ax.set_xlim(0, max(vals) * 1.22); save(fig, "19_pipeline_topkernels.png")
print("done")
