import json, os, math, statistics as st, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
R=json.load(open("docs/ahat_accuracy_2026-09-02/data/grid_2d.json"))
OUT="docs/ahat_accuracy_2026-09-02/plots"; os.makedirs(OUT,exist_ok=True)
INK,MUTED,GRID="#0b0b0b","#8a8983","#e4e3de"
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"figure.facecolor":"white",
                     "axes.facecolor":"white"})
M=lambda k,per: st.median([v[k] for v in per.values()])
BITS=[3,4,5,6,7,8,9,10,12]; BLKS=[2,4,8,16,32,64,128]
E={(b,B):M("eta_cum",R[f"{b}|{B}"]) for b in BITS for B in BLKS if f"{b}|{B}" in R}
P={(b,B):M("bpe",R[f"{b}|{B}"])    for b in BITS for B in BLKS if f"{b}|{B}" in R}
SAFE,MARG=0.30,0.70
fig,axs=plt.subplots(2,2,figsize=(14.0,10.4))

# (a) vs bits
ax=axs[0][0]; cm=plt.cm.viridis
for i,B in enumerate(BLKS):
    xs=[b for b in BITS if (b,B) in E]
    ax.plot(xs,[E[(b,B)] for b in xs],"-o",ms=4,lw=1.6,color=cm(i/(len(BLKS)-1)),label=f"B={B}")
ax.axhspan(0,SAFE,color="#d6efe4",alpha=.6,lw=0); ax.axhspan(SAFE,MARG,color="#fbe3cf",alpha=.6,lw=0)
ax.axhspan(MARG,40,color="#f7cdbd",alpha=.6,lw=0)
ax.set_yscale("log"); ax.set_xticks(BITS); ax.set_xlabel("a_hat bit width")
ax.set_ylabel("eta_cum @ t=48"); ax.set_title("(a) vs bit width — one line per block size")
ax.legend(frameon=False,fontsize=7.5,ncol=2)

# (b) vs block
ax=axs[0][1]; cm=plt.cm.plasma
for i,b in enumerate(BITS):
    xs=[B for B in BLKS if (b,B) in E]
    ax.plot(xs,[E[(b,Bx)] for Bx in xs],"-o",ms=4,lw=1.6,color=cm(i/(len(BITS)-1)),label=f"{b} bit")
ax.axhspan(0,SAFE,color="#d6efe4",alpha=.6,lw=0); ax.axhspan(SAFE,MARG,color="#fbe3cf",alpha=.6,lw=0)
ax.axhspan(MARG,40,color="#f7cdbd",alpha=.6,lw=0)
ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.set_xticks(BLKS)
ax.set_xticklabels([str(b) for b in BLKS])
ax.set_xlabel("along-C block size B"); ax.set_ylabel("eta_cum @ t=48")
ax.set_title("(b) vs block size — far weaker slope than bits")
ax.legend(frameon=False,fontsize=7.5,ncol=2)

# (c) heatmap
ax=axs[1][0]
Z=np.full((len(BITS),len(BLKS)),np.nan)
for i,b in enumerate(BITS):
    for j,B in enumerate(BLKS):
        if (b,B) in E: Z[i,j]=E[(b,B)]
im=ax.imshow(np.log10(Z),cmap="RdYlGn_r",aspect="auto",origin="lower")
ax.set_xticks(range(len(BLKS))); ax.set_xticklabels(BLKS)
ax.set_yticks(range(len(BITS))); ax.set_yticklabels(BITS)
for i,b in enumerate(BITS):
    for j,B in enumerate(BLKS):
        if (b,B) not in E: continue
        e=E[(b,B)]
        ax.text(j,i,f"{e:.3f}" if e<10 else f"{e:.1f}",ha="center",va="center",fontsize=6.8,
                color="white" if e>0.7 or e<0.01 else "black")
cs=ax.contour(np.arange(len(BLKS)),np.arange(len(BITS)),Z,levels=[SAFE,MARG],
              colors=["#0b0b0b","#7a2f10"],linewidths=1.8)
ax.clabel(cs,fmt={SAFE:"safe 0.30",MARG:"marginal 0.70"},fontsize=7.5)
ax.set_xlabel("block size B"); ax.set_ylabel("a_hat bit width")
ax.set_title("(c) eta_cum heatmap (log10 colour), contours = the calibrated bands")
plt.colorbar(im,ax=ax,label="log10 eta_cum",fraction=0.046)

# (d) Pareto: eta_cum vs memory
ax=axs[1][1]
pts=[(P[(b,B)],E[(b,B)],b,B) for b in BITS for B in BLKS if (b,B) in E]
front=[p for p in pts if not any(q[0]<=p[0] and q[1]<=p[1] and q!=p for q in pts)]
for bpe,e,b,B in pts:
    ax.plot(bpe,e,"o",ms=4,color=MUTED,alpha=.55,zorder=2)
fs=sorted(front)
ax.plot([p[0] for p in fs],[p[1] for p in fs],"-o",color="#eb6834",lw=2.0,ms=7,zorder=4,
        label="Pareto frontier")
for bpe,e,b,B in fs:
    ax.annotate(f"{b}b/B{B}",(bpe,e),textcoords="offset points",xytext=(6,-9),fontsize=6.8,
                color="#a03c12")
ax.axhline(SAFE,color=INK,ls="--",lw=1.2); ax.text(2.1,SAFE*1.15,"safe limit 0.30",fontsize=7.5)
ax.axvline(1.125,color="#1baf7a",ls=":",lw=1.6)
ax.text(1.15,8,"shipped\n8bit B=32\n1.125 B/elem",fontsize=7.2,color="#1baf7a")
ax.set_xscale("log",base=2); ax.set_yscale("log")
ax.set_xlabel("a_hat storage (B/elem, incl. fp32 block scales)")
ax.set_ylabel("eta_cum @ t=48")
ax.set_title("(d) the trade-off — frontier is 'more bits, coarser blocks'")
ax.legend(frameon=False,fontsize=8)

for r in axs:
    for ax in r:
        ax.grid(True,color=GRID,lw=0.6); ax.set_axisbelow(True)
fig.suptitle("a_hat storage: bit width x along-C block size. Real captured kernel-1 inputs, "
             "5 layers (median), 49 DDIM steps.\nModel validated against the real kernels to "
             "1e-4 relative (validate_kernel.py). Bands anchored on decoded samples.",
             fontsize=10.5,color=INK)
fig.tight_layout(rect=(0,0,1,0.945))
fig.savefig(f"{OUT}/grid_2d.png",dpi=145); print(f"{OUT}/grid_2d.png")
print("Pareto frontier:")
for bpe,e,b,B in fs: print(f"  {bpe:6.4f} B/elem  {b}bit B={B:<4d} eta_cum={e:.4f}")
