"""Absolute MSE, at two levels.

(A) TENSOR level: MSE of the a_hat storage error inside kernel 1, per layer, vs bit width.
    Absolute MSE depends on each layer's activation scale, so the signal power mean(c^2) is
    reported next to it; NMSE = MSE / mean(c^2) = relL2^2 collapses the layers onto one curve.

(B) IMAGE level: MSE / PSNR between each arm's DECODED samples and the fp16-a_hat reference.
    Same seed for every arm, so the 8 images are pixel-aligned and directly comparable. This is
    the number that corresponds to what the eye sees in the sample grids.
"""
import os, sys, json, math, statistics as st
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
import numpy as np
from PIL import Image

# ---------------- (A) tensor level ----------------
CAP=torch.load("docs/ahat_accuracy_2026-09-02/data/capture_int8.pt", weights_only=False)
DEV="cuda"; DLIM=127.0; BITS=[3,4,5,6,7,8,9,10,12]; B=32
def gn_silu(x,meta,mod,N,C,H,W):
    G=meta["num_groups"]; xg=x.float().view(N,G,C//G,H,W)
    mu=xg.mean(dim=(2,3,4),keepdim=True); var=xg.var(dim=(2,3,4),unbiased=False,keepdim=True)
    n=((xg-mu)*(var+meta["eps"]).rsqrt()).view(N,C,H,W)
    n=n*meta["weight"].to(DEV).view(1,C,1,1)+meta["bias"].to(DEV).view(1,C,1,1)
    if mod is not None: n=n*(1.0+mod[0].to(DEV).view(N,C,1,1))+mod[1].to(DEV).view(N,C,1,1)
    n=n.half().float(); o=n*torch.sigmoid(n) if meta["apply_silu"] else n
    if meta["smooth_inv"] is not None: o=o*meta["smooth_inv"].to(DEV).view(1,C,1,1)
    return o
def blk_q(v,bits,N,C,H,W):
    lim=float(2**(bits-1)-1); x=v.permute(0,2,3,1).reshape(N,H,W,C//B,B)
    s=x.abs().amax(-1,keepdim=True).clamp_min(1e-12)/lim
    return ((x/s).round().clamp_(-lim,lim)*s).reshape(N,H,W,C).permute(0,3,1,2).contiguous()

T_OUT={}
for name,L in CAP["layers"].items():
    C,H,W,N=L["C"],L["H"],L["W"],L["batch"]; T=L["x"].shape[0]
    o=[gn_silu(L["x"][t].to(DEV),L["meta"],L["mod"][t],N,C,H,W) for t in range(T)]
    ref=torch.zeros(N,C,H,W,device=DEV); cbar=[]
    for t in range(T):
        q=torch.clamp(torch.round((o[t]-ref)*L["scale"][t]),-DLIM,DLIM)
        ref=ref+q/L["scale"][t]; cbar.append(ref.clone())
    pw=st.median([(c*c).mean().item() for c in cbar[5:]])
    for bits in BITS:
        a=torch.zeros(N,C,H,W,device=DEV); eta=torch.zeros(N,C,H,W,device=DEV)
        ms,mc=[],[]
        for t in range(T):
            s=L["scale"][t]
            q=torch.clamp(torch.round((o[t]-a)*s),-DLIM,DLIM); cons=a+q/s
            new=blk_q(cons,bits,N,C,H,W); e=new-cons; eta+=e
            ms.append((e*e).mean().item()); mc.append((eta*eta).mean().item()); a=new
        T_OUT.setdefault(bits,{})[name]={"mse_step":st.median(ms[5:]),"mse_cum":mc[-1],
                                         "signal_power":pw}
    del o,cbar; torch.cuda.empty_cache()

print("(A) TENSOR level, a_hat storage error, B=32.  MSE is absolute; NMSE = MSE/signal power")
print(f"{'bits':>4} | " + " | ".join(f"{n.split('_')[0]:>12}" for n in T_OUT[8]) + " |  NMSE(step)  NMSE(cum)")
print("-"*118)
for bits in BITS:
    per=T_OUT[bits]
    cells=" | ".join(f"{v['mse_step']:12.3e}" for v in per.values())
    ns=st.median([v["mse_step"]/v["signal_power"] for v in per.values()])
    nc=st.median([v["mse_cum"]/v["signal_power"] for v in per.values()])
    print(f"{bits:>4} | {cells} |  {ns:10.3e}  {nc:9.3e}")
print(f"{'power':>4} | " + " | ".join(f"{v['signal_power']:12.3e}" for v in T_OUT[8].values()))

# ---------------- (B) image level ----------------
D="docs/ahat_only_conv_2026-09-02/samples"
def load(a): return np.asarray(Image.open(f"{D}/{a}.png").convert("RGB"),dtype=np.float64)/255.0
ARMS=[("int4_ahat0","a_hat fp16 (reference)"),("int4_ahat32","a_hat i8 B=32  REAL"),
      ("int4_ahat32_sim7","a_hat 7-bit B=32"),("int4_ahat32_sim6","a_hat 6-bit B=32"),
      ("int4_ahat32_sim5","a_hat 5-bit B=32"),("int4_ahat32_sim4","a_hat 4-bit B=32"),
      ("int4_ahat32_sim3","a_hat 3-bit B=32")]
ref=load(ARMS[0][0]); I_OUT=[]
print("\n(B) IMAGE level, 8 decoded samples vs the fp16-a_hat reference (same seed, pixel-aligned)")
print(f"{'arm':>26} | {'MSE':>10} | {'RMSE':>8} | {'PSNR dB':>8} | {'eta_cum':>8}")
print("-"*76)
ETA={"int4_ahat0":0.001,"int4_ahat32":0.053,"int4_ahat32_sim7":0.110,
     "int4_ahat32_sim6":0.254,"int4_ahat32_sim5":0.657,"int4_ahat32_sim4":1.982,
     "int4_ahat32_sim3":7.33}
for a,lab in ARMS:
    x=load(a); mse=float(((x-ref)**2).mean())
    psnr=10*math.log10(1.0/mse) if mse>0 else float("inf")
    I_OUT.append({"arm":a,"label":lab,"mse":mse,"rmse":math.sqrt(mse),"psnr":psnr,
                  "eta_cum":ETA[a]})
    print(f"{lab:>26} | {mse:10.3e} | {math.sqrt(mse):8.4f} | "
          f"{psnr:8.2f} | {ETA[a]:8.3f}")
json.dump({"tensor":T_OUT,"image":I_OUT},
          open("docs/ahat_accuracy_2026-09-02/data/mse.json","w"),indent=1)
