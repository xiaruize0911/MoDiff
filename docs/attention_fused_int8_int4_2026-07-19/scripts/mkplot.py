"""Plot the 5-version benchmark: flash-attention ON vs OFF, wall ms/step, vs fp16.
Reads data/bench5_speed{,_noflash}_b{B}.csv + bench5_buckets*.csv. Emits fig_bench5.png."""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = int(os.environ.get("E2E_BATCH", "64"))
D = "/workspace/MoDiff/docs/attention_fused_int8_int4_2026-07-19"
VER = ["fp16", "int8_baseline", "int8_modiff", "int4_baseline", "int4_modiff"]


def rd(name):
    p = f"{D}/data/{name}"
    return {r["version"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


on = rd(f"bench5_speed_b{B}.csv")
off = rd(f"bench5_speed_noflash_b{B}.csv")
fp = float(off["fp16"]["wall_ms_step"])

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})

# ---- left: wall ms/step, flash off vs on, per version ----
x = np.arange(len(VER)); w = 0.38
voff = [float(off[v]["wall_ms_step"]) for v in VER]
von = [float(on[v]["wall_ms_step"]) for v in VER]
b1 = ax.bar(x - w/2, voff, w, color="#4C78A8", label="fp16 SDPA attention (flash OFF)")
b2 = ax.bar(x + w/2, von, w, color="#E45756", label="int8 flash attention (flash ON)")
for i, v in enumerate(VER):
    ax.text(x[i] - w/2, voff[i], f"{voff[i]:.0f}\n{fp/voff[i]:.2f}×", ha="center", va="bottom", fontsize=7.5)
    ax.text(x[i] + w/2, von[i], f"{von[i]:.0f}\n{fp/von[i]:.2f}×", ha="center", va="bottom", fontsize=7.5, color="#a00")
ax.axhline(fp, ls="--", c="#888", lw=1); ax.text(len(VER)-0.5, fp, " fp16", va="bottom", ha="right", fontsize=8, color="#888")
ax.set_xticks(x); ax.set_xticklabels([v.replace("_", "\n") for v in VER], fontsize=8.5)
ax.set_ylabel("wall ms/step (min-of-6)"); ax.set_ylim(0, max(von) * 1.18)
ax.set_title(f"5 versions × attention {{fp16 SDPA / int8 flash}} — batch {B}\n"
             "× = speed vs fp16 · flash is a +9–15 ms regression on every quant version", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper left")

# ---- right: attention-bucket decomposition, int8_baseline, flash off vs on ----
bkon = {r["bucket"]: float(r["ms_step"]) for r in csv.DictReader(open(f"{D}/data/bench5_buckets_b{B}.csv")) if r["version"] == "int8_baseline"}
bkoff = {r["bucket"]: float(r["ms_step"]) for r in csv.DictReader(open(f"{D}/data/bench5_buckets_noflash_b{B}.csv")) if r["version"] == "int8_baseline"}
cats = ["attention (softmax/flash)", "qkv/proj GEMM (+ fp16 attn bmm)", "quantize / absmax"]
short = ["attention\n(softmax+bmm / flash)", "qkv/proj GEMM\n(+ attn bmm)", "quantize\n(+ quantize_qkv)"]
xo = np.arange(len(cats))
ax2.bar(xo - w/2, [bkoff.get(c, 0) for c in cats], w, color="#4C78A8", label="flash OFF")
ax2.bar(xo + w/2, [bkon.get(c, 0) for c in cats], w, color="#E45756", label="flash ON")
for i, c in enumerate(cats):
    ax2.text(xo[i]-w/2, bkoff.get(c,0), f"{bkoff.get(c,0):.0f}", ha="center", va="bottom", fontsize=7.5)
    ax2.text(xo[i]+w/2, bkon.get(c,0), f"{bkon.get(c,0):.0f}", ha="center", va="bottom", fontsize=7.5, color="#a00")
ax2.set_xticks(xo); ax2.set_xticklabels(short, fontsize=8)
ax2.set_ylabel("int8_baseline device self-time (ms/step)")
ax2.set_title("Where flash moves the time (int8_baseline)\nflash fuses bmm→attn but the kernel is bigger; quantize grows", fontsize=10)
ax2.legend(fontsize=8.5)

fig.suptitle("Fused int8 flash attention applied to baseline & MoDiff, int8 & int4 — churches UNet, A40", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{D}/fig_bench5.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("wrote fig_bench5.png")
