"""Render fusion-fix summary figures from data/fusion_profile.csv (matplotlib, headless).
Writes figs/fig_fusion_gpu_busy.png (grouped before/after gpu_busy) and
figs/fig_fusion_buckets.png (stacked per-component buckets for the key before/after pairs)."""
import os, csv
os.chdir("/workspace/MoDiff")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "docs/fusion_fix_2026-07-22"
rows = {r["config"]: r for r in csv.DictReader(open(f"{HERE}/data/fusion_profile.csv"))}
BUCKETS = ["attention", "conv (int GEMM)", "qkv/proj int GEMM", "attn bmm (fp16)", "other fp16 GEMM",
           "GroupNorm", "quantize/dequant", "modiff cache", "upsample/concat", "elementwise/copy", "other"]

# (title, before_cfg, after_cfg)
PAIRS = [("Phase 2 int8 o_hat", "int8_modiff.ohat_off", "int8_modiff.default"),
         ("Phase 2 int4 o_hat", "int4_modiff.ohat_off", "int4_modiff.default"),
         ("Phase 5 int4 GN->pack", "int4_modiff.gnpack_off", "int4_modiff.default"),
         ("Phase 3 GN->delta (regression)", "int8_modiff.default", "int8_modiff.gndelta_on")]

# fig 1: gpu_busy + wall before/after
fig, ax = plt.subplots(figsize=(9, 4.5))
labels, gb_b, gb_a = [], [], []
for title, b, a in PAIRS:
    if b in rows and a in rows:
        labels.append(title); gb_b.append(float(rows[b]["gpu_busy"])); gb_a.append(float(rows[a]["gpu_busy"]))
x = range(len(labels)); w = 0.38
ax.bar([i - w/2 for i in x], gb_b, w, label="before/off", color="#888")
ax.bar([i + w/2 for i in x], gb_a, w, label="after/on", color="#2a7")
for i, (bb, aa) in enumerate(zip(gb_b, gb_a)):
    ax.text(i, max(bb, aa) + 0.4, f"{(bb-aa):+.1f}ms", ha="center", fontsize=8)
ax.set_ylabel("gpu_busy (ms/step)"); ax.set_title(f"Fusion-fix GPU-work impact @ b{rows[PAIRS[0][2]]['batch']}")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8); ax.legend()
plt.tight_layout(); plt.savefig(f"{HERE}/figs/fig_fusion_gpu_busy.png", dpi=130); plt.close()

# fig 2: stacked buckets for each pair
fig, axes = plt.subplots(1, len(PAIRS), figsize=(4*len(PAIRS), 4.6), sharey=True)
cmap = plt.get_cmap("tab20")
for ax, (title, b, a) in zip(axes, PAIRS):
    if b not in rows or a not in rows: continue
    bottoms = [0.0, 0.0]
    for j, bk in enumerate(BUCKETS):
        vals = [float(rows[b][bk]), float(rows[a][bk])]
        ax.bar(["off", "on"], vals, bottom=bottoms, color=cmap(j % 20), label=bk)
        bottoms = [bottoms[0] + vals[0], bottoms[1] + vals[1]]
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel("ms/step (CUDA self-time)")
axes[-1].legend(fontsize=6, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.savefig(f"{HERE}/figs/fig_fusion_buckets.png", dpi=130); plt.close()
print("WROTE figs/fig_fusion_gpu_busy.png, figs/fig_fusion_buckets.png")
