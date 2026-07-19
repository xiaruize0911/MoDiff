"""Combined config plot: AWQ/w4a4 Linear with attention in {fp16 SDPA, dynamic-quant, static-quant}.
Reads combined_speed{,_dynamic}_b{B}.csv (quant attention) + bench5_speed_noflash_b{B}.csv (fp16 attn)."""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = int(os.environ.get("E2E_BATCH", "64"))
DD = "/workspace/MoDiff/docs/attention_fused_int8_int4_2026-07-19"
VER = ["fp16", "int8_baseline", "int8_modiff", "int4_baseline", "int4_modiff"]


def rd(name):
    p = f"{DD}/data/{name}"
    return {r["version"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


fp16attn = rd(f"bench5_speed_noflash_b{B}.csv")
dyn = rd(f"combined_speed_dynamic_b{B}.csv")
sta = rd(f"combined_speed_b{B}.csv")
fp = float(fp16attn["fp16"]["wall_ms_step"])

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.45, 1]})
x = np.arange(len(VER)); w = 0.27


def wl(d, v):
    return float(d[v]["wall_ms_step"]) if v in d else 0.0


a = [wl(fp16attn, v) for v in VER]
d = [wl(dyn, v) for v in VER]
s = [wl(sta, v) for v in VER]
ax.bar(x - w, a, w, color="#888888", label="fp16 SDPA attention")
ax.bar(x, d, w, color="#4C78A8", label="dynamic quantized attention")
ax.bar(x + w, s, w, color="#E45756", label="STATIC quantized attention")
for i in range(len(VER)):
    for xi, val, col in [(x[i]-w, a[i], "#555"), (x[i], d[i], "#356"), (x[i]+w, s[i], "#a00")]:
        if val > 0:
            ax.text(xi, val, f"{val:.0f}\n{fp/val:.2f}×", ha="center", va="bottom", fontsize=6.8, color=col)
ax.axhline(fp, ls="--", c="#888", lw=1)
ax.set_xticks(x); ax.set_xticklabels([v.replace("_", "\n") for v in VER], fontsize=8.5)
ax.set_ylabel("wall ms/step (min-of-6)"); ax.set_ylim(0, max(d) * 1.2)
ax.set_title(f"AWQ w8a8 / w4a4 Linear + attention {{fp16 / dynamic-quant / STATIC-quant}} — batch {B}\n"
             "static quant attention is fastest (no runtime reductions); int4_baseline static = 1.11× fp16", fontsize=10)
ax.legend(fontsize=8.5, loc="upper left")

# quality: rel-L2 vs fp16, dynamic vs static
rd_ = [float(dyn[v]["rel_vs_fp16"]) if v in dyn else 0 for v in VER]
rs_ = [float(sta[v]["rel_vs_fp16"]) if v in sta else 0 for v in VER]
ax2.bar(x - w/2, rd_, w, color="#4C78A8", label="dynamic")
ax2.bar(x + w/2, rs_, w, color="#E45756", label="STATIC")
for i in range(len(VER)):
    if rd_[i] > 0: ax2.text(x[i]-w/2, rd_[i], f"{rd_[i]:.2f}"+("*" if "modiff" in VER[i] else ""), ha="center", va="bottom", fontsize=7)
    if rs_[i] > 0: ax2.text(x[i]+w/2, rs_[i], f"{rs_[i]:.2f}"+("*" if "modiff" in VER[i] else ""), ha="center", va="bottom", fontsize=7, color="#a00")
ax2.set_xticks(x); ax2.set_xticklabels([v.replace("_", "\n") for v in VER], fontsize=8)
ax2.set_ylabel("rel-L2 vs fp16 (single forward)")
ax2.set_title("Quality: STATIC costs more (static-c softmax)\n(* modiff = stale-cache, unreliable)", fontsize=10)
ax2.legend(fontsize=8.5)

fig.suptitle("Combined: AWQ w8a8 / modified w4a4 Linear + quantized attention (static vs dynamic) — churches UNet, A40", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{DD}/fig_combined.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("wrote fig_combined.png")
