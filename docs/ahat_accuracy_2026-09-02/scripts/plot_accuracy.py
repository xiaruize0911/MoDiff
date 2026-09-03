import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
D=json.load(open("docs/ahat_accuracy_2026-09-02/data/accuracy_int8.json"))["data"]
OUT="docs/ahat_accuracy_2026-09-02/plots"; os.makedirs(OUT, exist_ok=True)
INK,INK2,MUTED,GRID="#0b0b0b","#52514e","#8a8983","#e4e3de"
COL={"fp16":"#0b0b0b","i8 B=16":"#2a78d6","i8 B=32":"#1baf7a","i8 B=64":"#8a8983","i4 B=32":"#eb6834"}
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"axes.labelcolor":INK2,
    "xtick.color":INK2,"ytick.color":INK2,"figure.facecolor":"white","axes.facecolor":"white"})
METRICS=[("eta_cum","ACCUMULATED storage error  ||sum eta_k|| / ||consumed||\n(what the conv output carries; 1.0 = as large as the signal)"),
         ("consumed","relL2 of the activation the conv consumes\n(a_hat_{t-1} + q_t/s_t)"),
         ("state","relL2 of the a_hat state after the write"),
         ("codes","fraction of delta codes differing from reference"),
         ("sat","fraction of a_hat codes pinned at ±limit")]
names=list(D)
fig,axs=plt.subplots(len(METRICS),len(names),figsize=(4.0*len(names),3.0*len(METRICS)),
                     sharex=True)
for r,(mk,mlab) in enumerate(METRICS):
    for c,nm in enumerate(names):
        ax=axs[r][c]; L=D[nm]
        for arm,per in L["arms"].items():
            ax.plot([p["t"] for p in per],[p[mk] for p in per],lw=1.5,
                    color=COL.get(arm,"#888"),label=arm)
        if mk in ("consumed","state","eta_cum"): ax.set_yscale("log")
        if mk=="eta_cum": ax.axhline(1.0,color="#eb6834",lw=1.0,ls="--")
        ax.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        if r==0: ax.set_title(f"C={L['C']} {L['H']}x{L['W']}",fontsize=9,color=INK)
        if c==0: ax.set_ylabel(mlab,fontsize=8)
        if r==len(METRICS)-1: ax.set_xlabel("DDIM step (0 = t=T)")
        if r==0 and c==0: ax.legend(frameon=False,fontsize=7.5)
fig.suptitle("Kernel-1 accuracy, open loop: every arm replays the SAME captured inputs and delta "
             "scales through the real CUDA kernel;\nonly a_hat storage differs. Reference = the "
             "same recurrence in fp32 with a_hat held exactly. W8A8, batch 4, 49 DDIM steps.",
             fontsize=10,color=INK)
fig.tight_layout(rect=(0,0,1,0.955))
fig.savefig(f"{OUT}/kernel1_accuracy.png",dpi=140); print(f"{OUT}/kernel1_accuracy.png")
