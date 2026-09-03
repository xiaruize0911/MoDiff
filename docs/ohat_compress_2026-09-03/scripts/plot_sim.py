import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D="docs/ohat_compress_2026-09-03/"
sd=json.load(open(D+"data/sim_sd.json")); a2=json.load(open(D+"data/sim_aligned2.json"))
fig,ax=plt.subplots(1,2,figsize=(12.6,4.6))
cols=plt.cm.viridis([0.05,0.3,0.5,0.7,0.9])
for i,(n,v) in enumerate(sd.items()):
    bb=[(x["bpe"],x["acc_err"]) for k,x in v["arms"].items() if k.startswith("ahat")]
    ss=[(x["bpe"],x["acc_err"]) for k,x in v["arms"].items() if k.startswith("sd")]
    ax[0].plot(*zip(*bb),"o-",color=cols[i],lw=1.8,ms=4,label=f"{n.split('_')[0]} buy bits")
    ax[0].plot(*zip(*ss),"s--",color=cols[i],lw=1.5,ms=6,mfc="none")
ax[0].set_yscale("log"); ax[0].set_xlabel("a_hat storage B/elem"); ax[0].set_ylabel("acc_err at t=48")
ax[0].set_title("(a) sigma-delta drift (squares) vs buying a_hat bits (circles)")
ax[0].axvline(1.125,color="k",ls=":",lw=1); ax[0].text(1.13,0.05,"ships",fontsize=8)
ax[0].legend(fontsize=7,ncol=1); ax[0].grid(alpha=0.3,which="both")
for i,(n,v) in enumerate(a2.items()):
    cur=v["arms"]["current"]["acc_err"]
    for m,ls in ((1.0,"-"),(1.5,"--"),(2.0,":")):
        pts=sorted((int(k.split("K=")[1].split("+")[0]), x["acc_err"])
                   for k,x in v["arms"].items() if k.startswith("aligned") and f"m={m}" in k)
        if len(pts)>1: ax[1].plot(*zip(*pts),ls,color=cols[i],lw=1.6,ms=4,marker="o")
    ax[1].axhline(cur,color=cols[i],lw=0.8,alpha=0.5)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("window K (scale frozen K steps)")
ax[1].set_ylabel("acc_err at t=48")
ax[1].set_title("(b) window-frozen scale: solid m=1, dashed m=1.5, dotted m=2\nthin horizontal = that layer's current")
ax[1].grid(alpha=0.3,which="both")
plt.tight_layout(); plt.savefig(D+"plots/sim_aligned.png",dpi=130); print("ok")
