import json, os, math, statistics as st, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
S=json.load(open("docs/ahat_accuracy_2026-09-02/data/shape_grid.json"))
G=json.load(open("docs/ahat_accuracy_2026-09-02/data/grid_2d.json"))
OUT="docs/ahat_accuracy_2026-09-02/plots"; os.makedirs(OUT,exist_ok=True)
INK,MUTED,GRID="#0b0b0b","#8a8983","#e4e3de"
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"figure.facecolor":"white",
                     "axes.facecolor":"white"})
AXES=[("N","batch N"),("C","channels C"),("H","height H"),("W","width W")]
CFG=[(b,B) for b in (4,6,8) for B in (16,32,64)]
CM={4:plt.cm.Reds,6:plt.cm.Oranges,8:plt.cm.Greens}
DEF={"N":4,"C":384,"H":16,"W":16}
def rows(ax): return sorted([r for r in S.values() if r["axis"]==ax],key=lambda r:r["value"])

fig,axs=plt.subplots(2,4,figsize=(16.4,8.4))
for j,(a,lab) in enumerate(AXES):
    rr=rows(a); xs=[r["value"] for r in rr]
    ax=axs[0][j]
    for i,(b,B) in enumerate(CFG):
        k=f"{b}|{B}"
        ref=next((r["eta"][k] for r in rr if r["value"]==DEF[a] and k in r["eta"]),None)
        if ref is None: continue
        y=[(r["eta"][k]/ref if k in r["eta"] else None) for r in rr]
        ax.plot([x for x,v in zip(xs,y) if v],[v for v in y if v],"-o",ms=3.5,lw=1.3,
                color=CM[b](0.4+0.22*(B//32 if B>=32 else 0)+0.15),
                label=f"{b}bit B={B}")
    ax.axhline(1.0,color=INK,lw=0.9,alpha=.5)
    ax.axhspan(0.9,1.1,color="#d6efe4",alpha=.45,lw=0)
    ax.set_xscale("log",base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(0.75,1.35); ax.set_xlabel(lab)
    if j==0:
        ax.set_ylabel("eta_cum / eta_cum at the default shape")
        ax.legend(frameon=False,fontsize=6.4,ncol=3,loc="upper left")
    ax.set_title(f"eta_cum vs {a}  (shaded = +-10%)")
    ax=axs[1][j]
    rB=[(r["value"],r["eta"].get("8|64"),r["eta"].get("8|16")) for r in rr]
    rb=[(r["value"],r["eta"].get("4|32"),r["eta"].get("8|32")) for r in rr]
    ax.plot([v for v,x,y_ in rB if x and y_],[x/y_ for v,x,y_ in rB if x and y_],
            "-o",color="#2a78d6",lw=1.6,ms=4.5,label="eta(B=64)/eta(B=16)  at 8 bit")
    axb=ax.twinx()
    axb.plot([v for v,x,y_ in rb if x and y_],[x/y_ for v,x,y_ in rb if x and y_],
             "--s",color="#eb6834",lw=1.6,ms=4.5,label="eta(4bit)/eta(8bit)  at B=32")
    ax.set_ylim(1.3,1.8); axb.set_ylim(60,100)
    ax.set_xscale("log",base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(lab)
    if j==0:
        ax.set_ylabel("block effect (x)",color="#2a78d6")
        ax.legend(frameon=False,fontsize=7,loc="upper left")
    if j==3:
        axb.set_ylabel("bit-width effect (x)",color="#eb6834")
        axb.legend(frameon=False,fontsize=7,loc="lower right")
    ax.set_title(f"relative effects vs {a}")
for r in axs:
    for ax in r:
        ax.grid(True,color=GRID,lw=0.6); ax.set_axisbelow(True)
        for sp in ("top",): ax.spines[sp].set_visible(False)
fig.suptitle("Is the (bits x block) grid shape-dependent? One axis at a time, synthetic input, "
             "content held fixed, 49 steps.\nTop: eta_cum normalised to the default shape "
             "(N=4 C=384 H=W=16). Bottom: the block effect and the bit effect, each as a ratio.",
             fontsize=10.5,color=INK)
fig.tight_layout(rect=(0,0,1,0.93))
fig.savefig(f"{OUT}/shape_grid.png",dpi=145); print(f"{OUT}/shape_grid.png")
