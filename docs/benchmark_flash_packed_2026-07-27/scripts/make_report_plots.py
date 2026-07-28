"""Generate the matplotlib figures for docs/benchmark_flash_packed_2026-07-27/REPORT.md.

Reads:
  - data/final_speedup_and_breakdown.json  (fp16 + 4 quantized modes, same-session speedup + full category %)
  - data/quantize_kernel_audit_precycle0.json / quantize_kernel_audit.json  (before/after vectorization)
Writes PNGs into plots/.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "docs/benchmark_flash_packed_2026-07-27"
os.makedirs(f"{HERE}/plots", exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                      "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10})

C_TEAL = "#1D9E75"
C_GRAY = "#888780"
C_LGRAY = "#B4B2A9"

final = json.load(open(f"{HERE}/data/final_speedup_and_breakdown.json"))
precycle0 = json.load(open(f"{HERE}/data/quantize_kernel_audit_precycle0.json"))
postvec = json.load(open(f"{HERE}/data/quantize_kernel_audit.json"))

# ---------------------------------------------------------------------------
# Figure 1: speedup vs fp16 (same-session, current build)
# ---------------------------------------------------------------------------
quant_modes = ["int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
speedups = [final[m]["speedup_vs_fp16"] for m in quant_modes]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(quant_modes, speedups, color=C_TEAL, width=0.55)
for b, v in zip(bars, speedups):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}x", ha="center", fontsize=11)
ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1)
ax.text(3.4, 1.03, "fp16 baseline", color=C_GRAY, fontsize=9, ha="right")
ax.set_ylabel("speedup vs fp16 (same session)")
ax.set_title("Speedup vs fp16 -- current build, b128, A40")
ax.set_ylim(0, max(speedups) * 1.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{HERE}/plots/fig_speedup.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: time cost broken down PER LAYER TYPE, one small-multiple panel per
# layer type, each panel showing that layer type's absolute ms/step across all
# 5 modes. This replaces the earlier single stacked bar (which lumped conv+
# norm+gemm together into one "fused quantized compute" slice and was hard to
# read) -- here every layer type gets its own clearly-labeled panel, so e.g.
# "how much does Attention cost in each mode" is a single glance, not a
# mental subtraction across a stacked segment.
# ---------------------------------------------------------------------------
LAYER_TYPES = [
    ("Conv", ("conv_int_fused", "conv_fp16", "upsample_conv_fused")),
    ("Attention", ("attention_flash", "attention_sdpa_math_unfused", "attention_sdpa_fused")),
    ("GroupNorm + SiLU", ("gn_silu_quantize_fused", "gn_silu")),
    ("GEMM / Linear", ("gemm_quant_fused", "gemm_fp16")),
    ("Quantize (standalone)", ("quantize_standalone",)),
    ("Resize / Upsample", ("resize_unfused",)),
    ("Elementwise / other", ("elementwise_misc", "other")),
]

modes5 = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
mode_labels = ["fp16", "int8\nbaseline", "int4\nbaseline", "int8\nmodiff", "int4\nmodiff"]
mode_colors = ["#73726c", "#1D9E75", "#2a9d8f", "#2a78d6", "#eda100"]
ms_steps = {m: final[m]["ms_step"] for m in modes5}

def layer_ms(mode, keys):
    pct = final[mode]["category_pct"]
    return sum(pct.get(k, 0.0) for k in keys) / 100.0 * ms_steps[mode]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, (name, keys) in enumerate(LAYER_TYPES):
    ax = axes[i]
    vals = [layer_ms(m, keys) for m in modes5]
    bars = ax.bar(mode_labels, vals, color=mode_colors, width=0.65)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("ms / step")
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
axes[-1].axis("off")
fig.suptitle("Time cost per layer type, absolute ms/step (same-session measurement)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{HERE}/plots/fig_breakdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: vectorization before/after (ms/step and quantize-kernel share)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
x = range(len(quant_modes))
w = 0.35

ms_before = [precycle0[m]["ms_step"] for m in quant_modes]
ms_after = [postvec[m]["ms_step"] for m in quant_modes]
ax = axes[0]
ax.bar([i - w/2 for i in x], ms_before, w, color=C_GRAY, label="before")
ax.bar([i + w/2 for i in x], ms_after, w, color=C_TEAL, label="after")
for i, (b, a) in enumerate(zip(ms_before, ms_after)):
    ax.text(i - w/2, b + 0.6, f"{b:.1f}", ha="center", fontsize=8.5)
    ax.text(i + w/2, a + 0.6, f"{a:.1f}", ha="center", fontsize=8.5)
ax.set_xticks(list(x)); ax.set_xticklabels(quant_modes, rotation=20, ha="right")
ax.set_ylabel("ms / step")
ax.set_title("E2E step time")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)

pct_before = [precycle0[m]["quantize_pct_of_total"] for m in quant_modes]
pct_after = [postvec[m]["quantize_pct_of_total"] for m in quant_modes]
ax = axes[1]
ax.bar([i - w/2 for i in x], pct_before, w, color=C_GRAY, label="before")
ax.bar([i + w/2 for i in x], pct_after, w, color=C_TEAL, label="after")
for i, (b, a) in enumerate(zip(pct_before, pct_after)):
    ax.text(i - w/2, b + 0.4, f"{b:.1f}", ha="center", fontsize=8.5)
    ax.text(i + w/2, a + 0.4, f"{a:.1f}", ha="center", fontsize=8.5)
ax.set_xticks(list(x)); ax.set_xticklabels(quant_modes, rotation=20, ha="right")
ax.set_ylabel("% of total GPU time")
ax.set_title("Quantize-kernel share")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Vectorization effect: before vs after (Cycles 1-2)")
fig.tight_layout()
fig.savefig(f"{HERE}/plots/fig_vectorization_before_after.png", dpi=150)
plt.close(fig)

print("wrote:")
for p in ("fig_speedup.png", "fig_breakdown.png", "fig_vectorization_before_after.png"):
    print(f"  {HERE}/plots/{p}")
