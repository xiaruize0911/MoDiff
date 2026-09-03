"""Two panels: (a) first-step vs steady-state per arm, (b) MoDiff overhead per step vs step count."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
D="docs/modiff_profile_diff_2026-09-03/"
rows={}
for l in open(D+"data/steps.jsonl"):
    d=json.loads(l); rows.setdefault(d["mode"],{})[d["S"]]=d["total_ms"]
sol={}
for m,r in rows.items():
    B=(r[50]-r[10])/40.0; sol[m]=(r[10]-9*B, B)
order=["int4_baseline","int4","int8_baseline","int8"]
lab={"int4_baseline":"W4A4 PTQ","int4":"W4A4 MoDiff","int8_baseline":"W8A8 PTQ","int8":"W8A8 MoDiff"}
fig,ax=plt.subplots(1,2,figsize=(12.5,4.4))
x=np.arange(4); A=[sol[m][0] for m in order]; Bs=[sol[m][1] for m in order]
c=["#9ecae1","#08519c","#fdbe85","#a63603"]
ax[0].bar(x-0.2,A,0.4,color=c,label="first step A")
ax[0].bar(x+0.2,Bs,0.4,color=c,alpha=0.45,hatch="//",label="steady step B")
for i,(a,b) in enumerate(zip(A,Bs)):
    ax[0].text(i-0.2,a*1.06,f"{a:.0f}",ha="center",fontsize=9)
    ax[0].text(i+0.2,b*1.06,f"{b:.1f}",ha="center",fontsize=9)
    ax[0].text(i,max(a,b)*1.9,f"{a/b:.1f}x",ha="center",fontsize=10,weight="bold")
ax[0].set_yscale("log"); ax[0].set_ylim(30,2200)
ax[0].set_xticks(x); ax[0].set_xticklabels([lab[m] for m in order],fontsize=9)
ax[0].set_ylabel("ms"); ax[0].set_title("(a) total(S) = A + (S-1)B, solved at S=10 and S=50")
ax[0].legend(fontsize=9); ax[0].grid(axis="y",alpha=0.3,which="both")
S=np.arange(4,101)
for p,q,col in [("int4","int4_baseline","#08519c"),("int8","int8_baseline","#a63603")]:
    dA=sol[p][0]-sol[q][0]; dB=sol[p][1]-sol[q][1]
    ax[1].plot(S,dA/S+dB,color=col,lw=2,label=f"{lab[p]} total")
    ax[1].axhline(dB,color=col,ls=":",lw=1.4,label=f"{lab[p]} steady = {dB:.2f}")
    for s0 in (10,50):
        ax[1].plot([s0],[dA/s0+dB],"o",color=col,ms=6)
        ax[1].annotate(f"S={s0}: {dA/s0+dB:.1f}",(s0,dA/s0+dB),textcoords="offset points",
                       xytext=(8,6),fontsize=9,color=col)
ax[1].set_xlabel("sampling steps S"); ax[1].set_ylabel("MoDiff overhead vs PTQ (ms/step)")
ax[1].set_title("(b) the overhead is a one-time cost, amortized"); ax[1].set_ylim(0,60)
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(D+"plots/first_step_vs_steady.png",dpi=130)
print("wrote plots/first_step_vs_steady.png")
