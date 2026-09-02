"""Tile quality vs blockwise tax across every config. Palette slots 1-2 of the dataviz
reference instance, used unchanged (node unavailable, so validate_palette.js was not run)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=json.load(open("docs/conv_int4_blockk_2026-09-02/data/tune_summary.json"))
S1,S2="#2a78d6","#eb6834"
INK,INK2,MUTED,GRID="#0b0b0b","#52514e","#8a8983","#e4e3de"
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"axes.labelcolor":INK2,
                     "xtick.color":INK2,"ytick.color":INK2})
fig,axs=plt.subplots(1,2,figsize=(12.4,4.8))
for ax,(tag,col) in zip(axs,(("int8",S1),("int4",S2))):
    rows=[r for r in S[tag] if r["tax"]<3]
    ax.scatter([r["tax"] for r in rows],[r["tile_pct"] for r in rows],
               s=90,color=col,zorder=3,edgecolor="white",linewidth=1.3)
    for r in rows:
        ax.annotate(f"{r['cfg']}",(r["tax"],r["tile_pct"]),textcoords="offset points",
                    xytext=(0,-3),ha="center",fontsize=7,color="white",zorder=4)
    # iso-lines of achieved % = tile_pct / tax
    import numpy as np
    xs=np.linspace(1.0,2.1,60)
    for lvl,st in ((80,"-"),(65,"--")):
        ax.plot(xs,lvl*xs,color=MUTED,lw=1.2,ls=st,zorder=1)
        # label inside the axes: put it where the iso-line crosses the top of the y range
        xat=min(2.05, 99.0/lvl)
        ax.text(xat, lvl*xat, f"{lvl}% of shipped ", color=MUTED, fontsize=8,
                ha="right", va="bottom", clip_on=True)
    ax.set_xlim(0.98,2.15); ax.set_ylim(50,100)
    ax.set_xlabel("blockwise tax  (blockwise time / same-tile scalar time)",color=INK)
    ax.set_title(f"{tag}", color=INK, loc="left", fontsize=11)
    ax.grid(color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
axs[0].set_ylabel("tile alone, % of shipped conv speed",color=INK)
fig.suptitle("Every config trades tile quality against the blockwise tax, and the trade is set by "
             "shared memory.\nA config must sit ON OR ABOVE the solid line to reach 80% of shipped; "
             "none does. Labels are cfg ids.",
             color=INK, fontsize=10, y=0.985, x=0.02, ha="left")
fig.subplots_adjust(top=0.80, bottom=0.13, left=0.07, right=0.97, wspace=0.18)
fig.savefig("docs/conv_int4_blockk_2026-09-02/plots/tune_tradeoff.png",dpi=170,bbox_inches="tight")
print("wrote plots/tune_tradeoff.png")
