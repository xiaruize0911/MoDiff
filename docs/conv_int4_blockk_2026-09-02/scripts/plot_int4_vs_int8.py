"""int8 vs int4: what the blockwise tax leaves you. Palette slots 1-3 of the dataviz
reference instance, used unchanged (node unavailable, so validate_palette.js was not run)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

i4 = json.load(open("docs/conv_int4_blockk_2026-09-02/data/int4_shape_sweep.json"))
i8 = json.load(open("docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json"))
el = [r for r in i4["unet"] if "blockk_b64" in r]
key = {(r["C"], r["N"], r["H"], r["W"]) for r in el}
e8 = [r for r in i8["unet"] if (r["C"], r["N"], r["H"], r["W"]) in key and "blockk_b64" in r]
w = lambda rows, k: sum(r[k] * r["freq"] for r in rows if k in r)

groups = [("int8", w(e8, "fp16"), w(e8, "int8_evt"), w(e8, "blockk_ctrl"), w(e8, "blockk_b64")),
          ("int4", w(el, "fp16"), w(el, "cutlass_int4"), w(el, "blockk_ctrl"), w(el, "blockk_b64"))]
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(7.6, 4.2))
labels = ["shipped (per-tensor)", "our tile, scalar", "our tile, blockwise B=64"]
cols = [S1, S2, S3]
bw, gap = 0.24, 0.02
for gi, (nm, fp, ship, ctrl, blk) in enumerate(groups):
    for si, (v, c) in enumerate(zip((ship, ctrl, blk), cols)):
        x = gi + (si - 1) * (bw + gap)
        ax.bar(x, fp / v, bw, color=c, zorder=3,
               edgecolor="white", linewidth=1.5)
        ax.text(x, fp / v + 0.06, f"{fp/v:.2f}x", ha="center", color=INK, fontsize=9)
ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=2)
ax.text(-0.47, 1.06, "fp16", color=MUTED, fontsize=8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["W8A8", "W4A4"], color=INK, fontsize=11)
ax.set_ylabel("conv speedup vs fp16", color=INK)
ax.set_ylim(0, 3.6)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in zip(cols, labels)]
ax.legend(handles=handles, loc="upper left", frameon=False, labelcolor=INK2, fontsize=9)
ax.set_title("The blockwise tax is multiplicative, so it only pays where the baseline is fast\n"
             "conv 3x3 only, frequency-weighted over the 14 churches-UNet shapes with C%128==0, "
             "batch 128, A40", color=INK, fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig("docs/conv_int4_blockk_2026-09-02/plots/int8_vs_int4.png", dpi=170, bbox_inches="tight")
print("wrote plots/int8_vs_int4.png")
