import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
D=json.load(open("docs/ahat_accuracy_2026-09-02/data/single_layer_sweep.json"))
OUT="docs/ahat_accuracy_2026-09-02/plots"; os.makedirs(OUT,exist_ok=True)
INK,MUTED,GRID="#0b0b0b","#8a8983","#e4e3de"
S1,S2,S3="#2a78d6","#eb6834","#1baf7a"
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"figure.facecolor":"white"})
fig,axs=plt.subplots(1,2,figsize=(13,5.0))
for ax,kind in zip(axs,("walk","iid")):
    arms=D[kind]["arms"]
    def bpe(k):
        if k=="fp16": return 2.0
        b=int(k.split("bit")[0]); B=int(k.split("B=")[1]); return b/8+4/B
    g4=[(bpe(k),v[-1]["eta_cum"],k.split("B=")[1]) for k,v in arms.items() if k.startswith("4bit")]
    g8=[(bpe(k),v[-1]["eta_cum"],k.split("B=")[1]) for k,v in arms.items() if k.startswith("8bit")]
    g4.sort(); g8.sort()
    ax.plot([p[0] for p in g4],[p[1] for p in g4],"-o",color=S2,lw=1.8,ms=6,label="a_hat 4-bit")
    ax.plot([p[0] for p in g8],[p[1] for p in g8],"s",color=S3,ms=9,label="a_hat 8-bit")
    for x,y,lab in g4+g8: ax.annotate(f"B={lab}",(x,y),textcoords="offset points",
                                      xytext=(7,-3),fontsize=7.5,color=MUTED)
    ax.axhline(1.0,color=S2,ls="--",lw=1.1)
    ax.text(0.55,1.06,"accumulated error = signal", fontsize=7.5,color=S2)
    ax.axvline(2.0,color=INK,ls=":",lw=1.1)
    ax.text(2.03,ax.get_ylim()[1]*0.5 if False else 3.0,"fp16 a_hat\n(2.0 B/elem)",
            fontsize=7.5,color=INK)
    ax.set_yscale("log"); ax.set_xscale("log",base=2)
    ax.set_xticks([0.5,0.75,1.0,1.5,2.0,2.5]); ax.set_xticklabels(["0.5","0.75","1.0","1.5","2.0","2.5"])
    ax.set_xlabel("a_hat storage cost (bytes / element, incl. fp32 block scales)")
    ax.set_ylabel("accumulated storage error  ||Σ η|| / ||signal||  @ t=48")
    ax.set_title(f"{kind} input trajectory")
    ax.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False,fontsize=8.5,loc="lower left")
fig.suptitle("Single layer (C=384, 16x16, batch 4), random input, 49 steps. Lower-left is better.\n"
             "Finer blocks do improve 4-bit monotonically, but never reach 8-bit, and below "
             "B=8 the fp32 scale costs more than the codes.",fontsize=10,color=INK)
fig.tight_layout(rect=(0,0,1,0.90))
fig.savefig(f"{OUT}/ahat_pareto.png",dpi=150); print(f"{OUT}/ahat_pareto.png")
