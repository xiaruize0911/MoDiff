import json, os, math, statistics as st, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
D=json.load(open("docs/ahat_accuracy_2026-09-02/data/mse.json"))
OUT="docs/ahat_accuracy_2026-09-02/plots"; os.makedirs(OUT,exist_ok=True)
INK,MUTED,GRID="#0b0b0b","#8a8983","#e4e3de"
S1,S2,S3="#2a78d6","#eb6834","#1baf7a"
plt.rcParams.update({"font.size":9,"axes.edgecolor":MUTED,"figure.facecolor":"white",
                     "axes.facecolor":"white"})
T={int(k):v for k,v in D["tensor"].items()}; I=D["image"]
bits=sorted(T)
fig,axs=plt.subplots(1,3,figsize=(16.2,5.2))

# ---- panel 1: absolute tensor MSE per layer, + NMSE ----
ax=axs[0]
layers=list(T[8])
cmap=plt.cm.viridis
for i,L in enumerate(layers):
    ax.plot(bits,[T[b][L]["mse_step"] for b in bits],"-o",ms=4,lw=1.5,
            color=cmap(i/max(1,len(layers)-1)),label=L.split("_")[0])
ax2=ax.twinx()
ax2.plot(bits,[st.median([T[b][L]["mse_step"]/T[b][L]["signal_power"] for L in layers])
               for b in bits],"--k",lw=1.6,label="NMSE (median)")
ax2.set_yscale("log"); ax2.set_ylabel("NMSE = MSE / signal power",color=INK)
ax.set_yscale("log"); ax.set_xticks(bits)
ax.set_xlabel("a_hat bit width"); ax.set_ylabel("absolute MSE of the storage error (per step)")
ax.set_title("(A) tensor level — absolute MSE spans 20x\nacross layers; NMSE collapses them")
ax.legend(frameon=False,fontsize=7.2,loc="lower left",title="layer",title_fontsize=7.2)
ax2.legend(frameon=False,fontsize=7.2,loc="upper right")

# ---- panel 2: image MSE + PSNR vs bits ----
ax=axs[1]
ib=[8,7,6,5,4,3]; lab2arm={7:"sim7",6:"sim6",5:"sim5",4:"sim4",3:"sim3"}
mse=[]; psnr=[]
for b in ib:
    a="int4_ahat32" if b==8 else f"int4_ahat32_{lab2arm[b]}"
    d=next(x for x in I if x["arm"]==a); mse.append(d["mse"]); psnr.append(d["psnr"])
ax.plot(ib,mse,"-o",color=S2,lw=2.0,ms=6,label="image MSE")
ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xticks(ib); ax.set_xlabel("a_hat bit width")
ax.set_ylabel("MSE of the decoded image vs fp16-a_hat", color=S2)
axb=ax.twinx(); axb.plot(ib,psnr,"--s",color=S1,lw=1.6,ms=5,label="PSNR")
axb.set_ylabel("PSNR (dB)",color=S1); axb.invert_xaxis()
axb.axhline(23.4,color=S3,ls=":",lw=1.4)
axb.text(7.9,23.9,"23.4 dB = safe limit (eta_cum 0.30)",fontsize=7.2,color=S3)
for b,p in zip(ib,psnr): axb.annotate(f"{p:.1f}",(b,p),textcoords="offset points",
                                      xytext=(4,5),fontsize=7,color=S1)
ax.set_title("(B) image level — decoded samples\n8 images, same seed, pixel-aligned")
ax.legend(frameon=False,fontsize=7.5,loc="upper left")
axb.legend(frameon=False,fontsize=7.5,loc="lower right")

# ---- panel 3: the calibration — image MSE vs eta_cum ----
ax=axs[2]
pts=[(d["eta_cum"],d["mse"],d["label"]) for d in I if d["mse"]>0]
xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
ax.plot(xs,ys,"o",color=S2,ms=8,zorder=3)
g=[min(xs)*0.7,max(xs)*1.4]
ax.plot(g,[0.0152*x for x in g],"-",color=INK,lw=1.5,
        label="image MSE = 0.0152 x eta_cum\n(log-log slope 0.981)")
for x,y,lab in pts:
    ax.annotate(lab.replace(" B=32","").replace("a_hat ",""),(x,y),
                textcoords="offset points",xytext=(8,-4),fontsize=7.2,color=MUTED)
for lo,hi,c in ((min(g),0.30,"#d6efe4"),(0.30,0.70,"#fbe3cf"),(0.70,max(g),"#f7cdbd")):
    ax.axvspan(lo,hi,color=c,alpha=0.55,lw=0,zorder=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("eta_cum  (open-loop kernel replay, no sampling needed)")
ax.set_ylabel("MSE of the decoded image")
ax.set_title("(C) the calibration — eta_cum predicts image MSE\nto +-12% over two decades")
ax.legend(frameon=False,fontsize=7.8,loc="upper left")

for ax in axs:
    ax.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
    for sp in ("top",): ax.spines[sp].set_visible(False)
fig.suptitle("a_hat storage error as MSE. (A) inside kernel 1, absolute and normalized. "
             "(B) on the decoded images. (C) the two are proportional.\n"
             "W8A8, real captured inputs / batch-128 50-step samples at seed 1234. "
             "Only 4-bit and 8-bit are real kernels; 3/5/6/7-bit is the validated model.",
             fontsize=10,color=INK)
fig.tight_layout(rect=(0,0,1,0.90))
fig.savefig(f"{OUT}/mse_vs_bits.png",dpi=150); print(f"{OUT}/mse_vs_bits.png")
