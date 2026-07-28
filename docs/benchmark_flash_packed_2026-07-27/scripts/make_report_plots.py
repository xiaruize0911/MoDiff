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
# Figure 2: time-cost breakdown by category, stacked horizontal bar, all 5 modes.
# Bars are sized in ABSOLUTE ms/step (not normalized to 100%), so bar length itself
# shows that fp16 costs ~1.7-1.9x more wall-clock time than the quantized modes --
# a percent-only stack would hide that a "smaller" percentage slice on a slower mode
# can still be more absolute time than a "bigger" slice on a faster one. Every
# segment worth >=3% of that mode's own total is labeled with its ms value.
# ---------------------------------------------------------------------------
def bucket_ms(cat_pct, ms_step):
    g = lambda *ks: sum(cat_pct.get(k, 0.0) for k in ks) / 100.0 * ms_step
    return {
        "fused quantized compute": g("conv_int_fused", "gn_silu_quantize_fused", "gemm_quant_fused"),
        "flash attention": g("attention_flash"),
        "standalone quantize kernel": g("quantize_standalone"),
        "unquantized fp16 compute": g("conv_fp16", "gemm_fp16"),
        "unfused attention math (fp16 only)": g("attention_sdpa_math_unfused"),
        "structural resize / boundary": g("resize_unfused", "gn_silu"),
        "elementwise / glue / other": g("elementwise_misc", "other"),
    }

modes5 = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
ms_steps = [final[m]["ms_step"] for m in modes5]
buckets = [bucket_ms(final[m]["category_pct"], ms) for m, ms in zip(modes5, ms_steps)]
cat_names = list(buckets[0].keys())
colors = ["#2a78d6", "#199e70", "#eda100", "#73726c", "#e34948", "#b4b2a9", "#d3d1c7"]

fig, ax = plt.subplots(figsize=(11, 5.5))
left = [0.0] * len(modes5)
LABEL_MIN_FRAC = 0.03  # only label segments >=3% of that mode's own total (avoids clutter on slivers)
for cat, color in zip(cat_names, colors):
    vals = [b[cat] for b in buckets]
    bars = ax.barh(modes5, vals, left=left, color=color, label=cat, height=0.62)
    for bar, v, total in zip(bars, vals, ms_steps):
        if v / total >= LABEL_MIN_FRAC:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}", ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
    left = [l + v for l, v in zip(left, vals)]
# total-ms label at the end of each bar, so absolute scale is unambiguous even without a ruler
for i, (mode, total) in enumerate(zip(modes5, ms_steps)):
    ax.text(total + 3, i, f"{total:.0f} ms", va="center", fontsize=10, fontweight="bold", color="#2C2C2A")

ax.set_xlabel("ms / step (absolute -- bar length reflects true e2e time cost)")
ax.set_xlim(0, max(ms_steps) * 1.12)
ax.set_title("Time-cost breakdown by category, in absolute ms/step (same-session measurement)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
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
