"""Sub-breakdown of the 'Norm / resize / quantize glue' bucket into concrete,
actionable categories, based on the per-kernel detail in glue_breakdown_detail.json.
Each category maps to a specific fusion question raised in the follow-up discussion.
"""
import json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "docs/benchmark_flash_packed_2026-07-27"
detail = json.load(open(f"{HERE}/data/glue_breakdown_detail.json"))

SUB_RULES = [
    ("GN+quantize fused (K1 path)",
     ["group_norm_silu_quantize_nhwc", "group_norm_silu_quantize_pack_nhwc",
      "group_norm_silu_delta_quantize", "gn_apply_delta_quantize"]),
    ("GN stats (modiff reduction, unfused)", ["gn_group_stats"]),
    ("Updown-block GN+quantize gap (unfused)",
     ["group_norm_silu_nhwc_kernel", "scale_quantize_int8_kernel", "quant_act_int4_pack_kernel"]),
    ("Ahat-cache update (modiff ancillary)", ["static_quantize_and_update_ahat", "static_quantize_pack_and_update_ahat"]),
    ("Attention quantize (standalone)", ["aq_qtok", "aq_vquant", "aq_kquant"]),
    ("Resize+quantize (baseline: fused both directions; x_upd's calls unfused, feed skip_connection)",
     ["avg_pool2d", "upsample_nearest2d", "upsample2x_quantize", "avgpool2x_quantize"]),
    ("Skip-connection concat", ["catarraybatchedcopy"]),
]

def classify(name):
    low = name.lower()
    for label, keys in SUB_RULES:
        for k in keys:
            if k.lower() in low:
                return label
    return "Residual-add / dtype-cast / other glue"

modes = ["int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
mode_labels = ["int8\nbaseline", "int4\nbaseline", "int8\nmodiff", "int4\nmodiff"]
cat_order = [r[0] for r in SUB_RULES] + ["Residual-add / dtype-cast / other glue"]
colors = ["#1D9E75", "#E24B4A", "#F09595", "#7F77DD", "#BA7517", "#888780", "#5F5E5A", "#D3D1C7"]

buckets = {m: {c: 0.0 for c in cat_order} for m in modes}
for m in modes:
    for k in detail[m]["kernels"]:
        buckets[m][classify(k["kernel"])] += k["ms_step"]

fig, ax = plt.subplots(figsize=(11, 5.5))
left = [0.0] * len(modes)
totals = [detail[m]["glue_ms_step"] for m in modes]
for cat, color in zip(cat_order, colors):
    vals = [buckets[m][cat] for m in modes]
    bars = ax.barh(mode_labels, vals, left=left, color=color, label=cat, height=0.6)
    for bar, v, total in zip(bars, vals, totals):
        if v / total >= 0.04:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    left = [l + v for l, v in zip(left, vals)]
for i, total in enumerate(totals):
    ax.text(total + 1, i, f"{total:.1f} ms", va="center", fontsize=10, fontweight="bold", color="#2C2C2A")

ax.set_xlabel("ms / step (absolute)")
ax.set_xlim(0, max(totals) * 1.2)
ax.set_title("'Norm / resize / quantize glue' bucket, split into fusion-actionable sub-categories")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{HERE}/plots/fig_glue_subbreakdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {HERE}/plots/fig_glue_subbreakdown.png")

for m in modes:
    print(f"\n{m} (glue total {detail[m]['glue_ms_step']:.1f} ms):")
    for c in cat_order:
        if buckets[m][c] > 0.01:
            print(f"  {buckets[m][c]:6.2f} ms  {c}")
