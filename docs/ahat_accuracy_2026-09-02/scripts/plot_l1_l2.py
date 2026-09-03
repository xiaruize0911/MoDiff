import json, os, math, statistics as st, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
R=json.load(open("docs/ahat_accuracy_2026-09-02/data/l1_l2.json"))
OUT="docs/ahat_accuracy_2026-09-02/plots"; os.makedirs(OUT,exist_ok=True)
INK,MUTED,GRID="#0b0b0b","#8a8983","#e4e3de"
S1,S2,S3="#2a78d6","#eb6834","#1baf7a"
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"figure.facecolor":"white",
                     "axes.facecolor":"white"})
M=lambda k,per: st.median([v[k] for v in per.values()])
def series(B,key):
    d={int(k.split("bit")[0]):M(key,per) for k,per in R.items() if k.endswith(f"B={B}")}
    xs=sorted(d); return xs,[d[x] for x in xs]
BLKS=[16,32,64]; COL={16:S1,32:S3,64:S2}
FLOOR_L1, FLOOR_L2 = 0.0111, 0.0096      # delta-quantizer floor (consumed), from the same run

fig,axs=plt.subplots(1,3,figsize=(16.0,5.2))

# ---- panel 1: eta_cum, both norms ----
ax=axs[0]
for lo,hi,c,lab in ((0,0.15,"#1baf7a","indistinguishable"),(0.15,0.30,"#a8d5c2","safe"),
                    (0.30,0.70,"#f5c9a8","marginal"),(0.70,30,"#f0a58a","broken")):
    ax.axhspan(lo,hi,color=c,alpha=0.35,lw=0)
    ax.text(12.1,math.sqrt(max(lo,2e-3)*min(hi,20)),lab,fontsize=7,color=MUTED,va="center")
for B in BLKS:
    xs,y2=series(B,"ec2"); _,y1=series(B,"ec1")
    ax.plot(xs,y2,"-o",color=COL[B],lw=1.8,ms=4.5,label=f"B={B}  relL2")
    ax.plot(xs,y1,"--s",color=COL[B],lw=1.2,ms=3.5,alpha=0.75,label=f"B={B}  relL1")
xs,_=series(32,"ec2")
ref=[R["8bit B=32"] and M("ec2",R["8bit B=32"])*2.45**(8-b) for b in xs]
ax.plot(xs,ref,":",color=INK,lw=1.4,label="/2.45 per bit (fit at 8-bit)")
ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xlim(2.6,13.4)
ax.set_xlabel("a_hat bit width"); ax.set_ylabel("accumulated storage error @ t=48")
ax.set_title("eta_cum — the metric that predicts E2E")
ax.legend(frameon=False,fontsize=7.2,ncol=2,loc="lower left")

# ---- panel 2: eta_step (isolated per-step storage error) + the floor ----
ax=axs[1]
for B in BLKS:
    xs,y2=series(B,"es2"); _,y1=series(B,"es1")
    ax.plot(xs,y2,"-o",color=COL[B],lw=1.8,ms=4.5,label=f"B={B}  relL2")
    ax.plot(xs,y1,"--s",color=COL[B],lw=1.2,ms=3.5,alpha=0.75,label=f"B={B}  relL1")
ax.axhline(FLOOR_L2,color=INK,ls="-",lw=1.2)
ax.axhline(FLOOR_L1,color=INK,ls="--",lw=1.0)
ax.text(3.1,FLOOR_L2*1.15,"delta-quantizer floor (relL2 / relL1)",fontsize=7.2,color=INK)
ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xlim(2.6,12.4)
ax.set_xlabel("a_hat bit width"); ax.set_ylabel("per-step storage error  ||eta_t||")
ax.set_title("eta_step — isolated storage term\n(below the floor => a_hat storage is free)")
ax.legend(frameon=False,fontsize=7.2,ncol=2,loc="lower left")

# ---- panel 3: shape indicators ----
ax=axs[2]
for B in BLKS:
    xs,r=series(B,"ec1"); _,r2=series(B,"ec2")
    ax.plot(xs,[a/b for a,b in zip(r,r2)],"-o",color=COL[B],lw=1.8,ms=4.5,label=f"B={B}  relL1/relL2")
    _,cc=series(B,"cc")
    ax.plot(xs,cc,":^",color=COL[B],lw=1.3,ms=4,alpha=0.8,label=f"B={B}  mean|e|/rms")
ax.axhline(0.866,color=MUTED,ls="--",lw=1.0); ax.text(9.4,0.874,"uniform over the step (0.866)",fontsize=7,color=MUTED)
ax.axhline(0.798,color=MUTED,ls=":",lw=1.0);  ax.text(9.4,0.806,"Gaussian (0.798)",fontsize=7,color=MUTED)
ax.axhline(1.0,color=INK,lw=0.8,alpha=0.4)
ax.set_xticks(xs); ax.set_xlim(2.6,12.4); ax.set_ylim(0.65,1.20)
ax.set_xlabel("a_hat bit width"); ax.set_ylabel("ratio")
ax.set_title("error SHAPE, not magnitude\nflat in bits => bits only rescale the error")
ax.legend(frameon=False,fontsize=7.2,ncol=2,loc="center left")

for ax in axs:
    ax.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.suptitle("a_hat storage error vs bit width, both norms. Real captured kernel-1 inputs, "
             "5 layers (median), 49 DDIM steps, W8A8.\n"
             "relL1 = ||e||_1/||x||_1, relL2 = ||e||_2/||x||_2. Only 4-bit and 8-bit exist as "
             "real kernels; the rest is the validated PyTorch storage model.",
             fontsize=10,color=INK)
fig.tight_layout(rect=(0,0,1,0.90))
fig.savefig(f"{OUT}/l1_l2_vs_bits.png",dpi=150); print(f"{OUT}/l1_l2_vs_bits.png")
