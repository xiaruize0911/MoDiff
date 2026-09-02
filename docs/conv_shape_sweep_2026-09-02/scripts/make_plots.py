"""Plots for the conv shape sweep. Palette slots 1-3 of the dataviz reference instance
(#2a78d6 / #eb6834 / #1baf7a), used unchanged -- node was unavailable so validate_palette.js
could not be run here; these are the skill's own validated defaults rather than new picks."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json"))
OUT = "docs/conv_shape_sweep_2026-09-02/plots"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de"
SERIES = [("int8_evt", "int8 EVT (shipped)", S1),
          ("blockk_ctrl", "our tile, scalar alpha", S2),
          ("blockk_b64", "our tile, blockwise B=64", S3)]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white",
                     "axes.facecolor": "white"})

# ---- Figure 1: speedup vs each shape parameter -------------------------------------------
axes_order = ["B", "N", "H", "W", "C"]
fig, axs = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=True)
for ax, name in zip(axs, axes_order):
    rows = [r for r in D["sweeps"][name] if "error" not in r and r.get("fp16")]
    xs = [r["value"] for r in rows]
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    for key, _lab, col in SERIES:
        ys = [r["fp16"] / r[key] if r.get(key) else None for r in rows]
        px = [x for x, y in zip(xs, ys) if y]; py = [y for y in ys if y]
        ax.plot(px, py, color=col, lw=2, marker="o", ms=5, zorder=3,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log", base=2); ax.set_xticks(xs)
    # thin the tick labels where adjacent powers of 2 collide (N and C run to 1536)
    rot = 45 if len(xs) > 6 else 0
    ax.set_xticklabels([str(x) for x in xs], fontsize=8, rotation=rot,
                       ha="right" if rot else "center")
    ax.set_xlabel(name, color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("speedup vs fp16  (higher = faster)", color=INK)
axs[0].set_ylim(0.85, 2.65)
axs[0].text(0.03, 0.10, "fp16 = 1.0", transform=axs[0].transAxes, color=MUTED, fontsize=8)
handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=5, label=l) for _, l, c in SERIES]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.06), labelcolor=INK2)
fig.suptitle("int8 conv 3x3 speedup vs fp16, by shape  "
             "(one axis varied; others at B=128 C=384 N=384 16x16, A40)",
             y=1.17, color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/speedup_vs_shape.png", dpi=170, bbox_inches="tight")
print("wrote speedup_vs_shape.png")

# ---- Figure 2: the 20 real UNet shapes, dot plot ----------------------------------------
un = [r for r in D["unet"] if "error" not in r and r.get("fp16")]
un.sort(key=lambda r: r["fp16"] * r["freq"])
lab = [f"C{r['C']}→N{r['N']}  {r['H']}x{r['W']}  (x{r['freq']})" for r in un]
fig2, ax = plt.subplots(figsize=(8.6, 6.4))
ax.axvline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
ys = range(len(un))
for key, _l, col in SERIES:
    ax.plot([r["fp16"] / r[key] if r.get(key) else None for r in un], ys,
            "o", color=col, ms=7, zorder=3, markeredgecolor="white", markeredgewidth=1)
ax.set_yticks(list(ys)); ax.set_yticklabels(lab, fontsize=8)
ax.set_xlabel("speedup vs fp16  (higher = faster)", color=INK)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
tw = {k: sum(r[k] * r["freq"] for r in un if r.get(k)) for k, _, _ in SERIES}
tf = sum(r["fp16"] * r["freq"] for r in un)
ax.set_title("The 20 churches-UNet conv shapes, batch 128\n"
             f"frequency-weighted total:  int8 EVT {tf/tw['int8_evt']:.2f}x   "
             f"scalar tile {tf/tw['blockk_ctrl']:.2f}x   blockwise B=64 {tf/tw['blockk_b64']:.2f}x",
             color=INK, fontsize=10, loc="left")
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
          frameon=False, labelcolor=INK2, fontsize=8)
fig2.tight_layout()
fig2.savefig(f"{OUT}/unet_shapes.png", dpi=170, bbox_inches="tight")
print("wrote unet_shapes.png")
