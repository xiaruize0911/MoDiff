"""Bar chart: e2e per-step GPU time by category, true fp16 vs int8 (b128, churches, corrected
autocast-symmetric profile from cat_profile_clean.py)."""
import os
os.chdir("/workspace/MoDiff")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cats = ["Attention\n(fp16)", "Conv", "Elementwise\n/copy", "GroupNorm", "Linear\n(qkv/proj)", "Quantize", "Upsample\n/other"]
fp16 = [82.0, 43.8, 31.5, 19.9, 6.1, 0.0, 5.8]
int8 = [81.9, 23.5, 29.0, 22.4, 4.8, 3.1, 9.8]
tot_fp16, tot_int8 = sum(fp16), sum(int8)

x = np.arange(len(cats)); w = 0.38
fig, ax = plt.subplots(figsize=(12, 5.8))
b1 = ax.bar(x - w/2, fp16, w, label=f"true fp16  (total {tot_fp16:.0f} ms, 1.00×)", color="#2563eb")
b2 = ax.bar(x + w/2, int8, w, label=f"int8  (total {tot_int8:.1f} ms, 1.08×)", color="#f59e0b")
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        if h > 0.5:
            ax.text(b.get_x() + b.get_width()/2, h + 0.6, f"{h:.0f}", ha="center", va="bottom", fontsize=8)

# annotate the deltas that matter
ax.annotate("−20 ms (1.86×)\nthe real int8 win", xy=(1 + w/2, 23.5), xytext=(1.35, 55),
            fontsize=9, color="#b45309", ha="left",
            arrowprops=dict(arrowstyle="->", color="#b45309"))
ax.annotate("fp16 in both —\nunchanged (43% of step)", xy=(0, 82), xytext=(0.15, 66),
            fontsize=9, color="#1e3a8a", ha="left",
            arrowprops=dict(arrowstyle="->", color="#1e3a8a"))
ax.annotate("int8 overhead\n(+GN +quantize)", xy=(5 + w/2, 3.1), xytext=(4.4, 30),
            fontsize=9, color="#7c2d12", ha="left",
            arrowprops=dict(arrowstyle="->", color="#7c2d12"))

ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("GPU time per step (ms)")
ax.set_title("End-to-end per-step GPU time by category: true fp16 vs int8\n"
             "(LSUN-churches LDM-8, b128, DDIM, A40 — both GPU-bound)\n"
             "int8 = 1.08× e2e: conv quant (−20 ms) diluted by fp16 attention (unchanged) + int8 overhead")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, 95)
plt.tight_layout()
out = "docs/flash_attention_2026-07-19/fig_e2e_profile_fp16_vs_int8_b128.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("WROTE", out)
