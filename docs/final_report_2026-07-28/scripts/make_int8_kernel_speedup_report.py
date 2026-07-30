"""Build matched FP16-vs-INT8 attention kernel/stage speedups from the A40 profiles."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/MoDiff/docs/final_report_2026-07-28")
DATA, PLOTS = ROOT / "data", ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
doc = json.loads((DATA / "layer_pipeline_bench.json").read_text())

shapes = [(192, 1024), (384, 256), (384, 64), (768, 16), (768, 4)]


def rows(mode):
    out = {}
    for r in doc["modes"][mode]:
        if r["kind"] == "attention":
            out[(r["x_shape"][1], r["x_shape"][2] * r["x_shape"][3])] = r
    return out


fp, i8 = rows("fp16"), rows("int8_baseline")

# The profiler key averages merge the two identically-named W8A8 GEMMs. An ordered one-call trace
# identifies their split: first is QKV, second is projection. Ratios are applied to the stable
# 30-call aggregate time from layer_pipeline_bench.json.
gemm_qkv_fraction = {
    (192, 1024): 397.349 / (397.349 + 341.476),
    (384, 256): 246.307 / (246.307 + 184.834),
    (384, 64): 79.265 / (79.265 + 67.361),
    (768, 16): 61.440 / (61.440 + 45.216),
    (768, 4): 23.072 / (23.072 + 34.304),
}


def sum_kernel(r, needles):
    return sum(k["us_per_layer_call"] for k in r["kernels"]
               if any(n in k["kernel"] for n in needles))


result_rows = []
for shape in shapes:
    fr, ir = fp[shape], i8[shape]
    C, T = shape
    fp_flash = sum_kernel(fr, ["flash_fwd_kernel"])
    fp_res = sum_kernel(fr, ["CUDAFunctor_add"])
    if T >= 256:
        fp_gn = sum_kernel(fr, ["gn_accum_kernel", "gn_finalize_kernel", "FillFunctor"])
        fp_qkv = sum_kernel(fr, ["ImplicitGemmConvolutionFusionPerSample"])
    else:
        fp_gn = sum_kernel(fr, ["group_norm_silu_nhwc_kernel"])
        # Ordered trace: before flash is QKV; after flash is projection. Kernel names differ.
        nonstage = ("group_norm", "flash_fwd", "CUDAFunctor_add")
        gems = [k["us_per_layer_call"] for k in fr["kernels"]
                if not any(n in k["kernel"] for n in nonstage)]
        fp_qkv = max(gems) if T >= 16 else max(gems)
    # Projection is the non-QKV GEMM. This is explicit for the two fused-GN shapes and inferred
    # from the ordered trace for the remaining shapes.
    fp_proj = sum(k["us_per_layer_call"] for k in fr["kernels"]
                  if ("gemm" in k["kernel"].lower() or "Kernel2" in k["kernel"])
                  and "ImplicitGemmConvolutionFusionPerSample" not in k["kernel"])
    if T < 256:
        fp_proj = sum(k["us_per_layer_call"] for k in fr["kernels"]
                      if ("gemm" in k["kernel"].lower() or "Kernel2" in k["kernel"])) - fp_qkv

    int_gemm_total = sum_kernel(ir, ["gemm_w8a8_kernel_awq"])
    int_qkv = int_gemm_total * gemm_qkv_fraction[shape]
    int_proj = int_gemm_total - int_qkv
    int_gn = sum_kernel(ir, ["group_norm_silu_quantize_nhwc_vec2_kernel"])
    int_prep = sum_kernel(ir, ["aq_kv_packed_static_tiled_vec2_kernel"])
    int_flash = sum_kernel(ir, ["flash_attn_int8", "flash_fwd_kernel"])
    int_outq = sum_kernel(ir, ["quant_attn_out_int8_kernel", "direct_copy_kernel_cuda"])

    stages = {
        "GN(+INT8 quantize)": (fp_gn, int_gn),
        "QKV GEMM": (fp_qkv, int_qkv),
        "K/V preparation": (0.0, int_prep),
        "Attention kernel": (fp_flash, int_flash),
        "Output quantize/copy": (0.0, int_outq),
        "Projection + residual": (fp_proj + fp_res, int_proj),
        "GN + QKV combined": (fp_gn + fp_qkv, int_gn + int_qkv),
        "Score route combined": (fp_flash, int_prep + int_flash),
        "Output route combined": (fp_proj + fp_res, int_outq + int_proj),
    }
    result_rows.append({
        "C": C, "T": T, "count": fr["n_instances"],
        "fp16_layer_us": fr["pipeline_us"], "int8_layer_us": ir["pipeline_us"],
        "layer_speedup": fr["pipeline_us"] / ir["pipeline_us"],
        "stages": {
            name: {
                "fp16_us": a, "int8_us": b,
                "speedup": (a / b if a and b else None),
                "note": ("INT8-only overhead" if not a and b else None),
            } for name, (a, b) in stages.items()
        },
    })


def weighted(stage, side):
    return sum(r["stages"][stage][side] * r["count"] for r in result_rows)


weighted_stages = {}
for stage in result_rows[0]["stages"]:
    a, b = weighted(stage, "fp16_us"), weighted(stage, "int8_us")
    weighted_stages[stage] = {
        "fp16_ms": a / 1000, "int8_ms": b / 1000,
        "speedup": a / b if a and b else None,
    }
fp_total = sum(r["fp16_layer_us"] * r["count"] for r in result_rows) / 1000
i8_total = sum(r["int8_layer_us"] * r["count"] for r in result_rows) / 1000
out = {
    "gpu": "NVIDIA A40 (SM86)", "batch": 128,
    "timing_protocol": "layer: 20 warmups, median 5x60; kernels: 30-call CUDA profile",
    "rows": result_rows, "weighted_stages": weighted_stages,
    "ordered_w8a8_gemm_qkv_fraction": {
        f"C{c}_T{t}": v for (c, t), v in gemm_qkv_fraction.items()
    },
    "gemm_split_method": (
        "The stable 30-call profile merges the two W8A8 kernels. An ordered CUDA trace identifies "
        "the first call as QKV and the second as projection; its ratio splits the stable aggregate."
    ),
    "weighted_overall": {
        "fp16_ms": fp_total, "int8_ms": i8_total, "speedup": fp_total / i8_total,
        "latency_reduction_pct": (1 - i8_total / fp_total) * 100,
    },
}
(DATA / "int8_kernel_speedups.json").write_text(json.dumps(out, indent=2))

fig, ax = plt.subplots(figsize=(10, 5.2))
stage_names = ["GN(+INT8 quantize)", "QKV GEMM", "Attention kernel",
               "GN + QKV combined", "Score route combined", "Projection + residual"]
vals = [weighted_stages[s]["speedup"] for s in stage_names]
colors = ["#e45756" if v < 1 else "#54a24b" for v in vals]
bars = ax.bar(np.arange(len(vals)), vals, color=colors)
ax.axhline(1, color="#222", linewidth=1.3)
ax.set_ylabel("Weighted speedup vs matched FP16 stage (×)")
ax.set_xticks(np.arange(len(vals)), stage_names, rotation=20, ha="right")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+.025, f"{v:.2f}×", ha="center")
ax.set_ylim(0, max(vals)*1.18)
fig.tight_layout()
fig.savefig(PLOTS / "int8_kernel_stage_speedups.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(result_rows))
fpv = [r["fp16_layer_us"]/1000 for r in result_rows]
i8v = [r["int8_layer_us"]/1000 for r in result_rows]
w = .36
ax.bar(x-w/2, fpv, w, label="FP16", color="#8c96a3")
ax.bar(x+w/2, i8v, w, label="INT8", color="#4c78a8")
ax.set_xticks(x, [f"C{r['C']}/T{r['T']}" for r in result_rows])
ax.set_ylabel("Complete attention layer (ms)")
ax.legend(frameon=False)
for xi, r in zip(x, result_rows):
    ax.text(xi, max(r["fp16_layer_us"], r["int8_layer_us"])/1000+.04,
            f"{r['layer_speedup']:.2f}×", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(PLOTS / "int8_layer_speedups_by_shape.png", dpi=180)
plt.close(fig)

print(json.dumps(out["weighted_overall"], indent=2))
print(json.dumps({k: v["speedup"] for k, v in weighted_stages.items()}, indent=2))
