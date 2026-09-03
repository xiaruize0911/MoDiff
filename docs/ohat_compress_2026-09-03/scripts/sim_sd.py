"""The scheme that actually breaks the accumulation, priced against simply buying a_hat bits.

v2's `accref` arm was a tautology (I never quantized the increment) -- discard it.  The correct
construction:  the conv accumulates codes, so let it accumulate a CORRECTED code.  Track the drift
D = (what the conv accumulated) - (the stored state).  D is bounded by construction once you
correct with it, so it can be stored in a few bits:

  q_t  = clamp(round((O_t - a_{t-1}) * s_t))          the delta, unchanged
  cons = a_{t-1} + q_t/s_t                            the activation, unchanged
  q'_t = clamp(round(q_t - D_hat_{t-1} * s_t))        what the CONV is fed  <-- the only change
  a_t  = Q_ahat(cons)                                 8-bit blockwise, unchanged
  D_t  = D_{t-1} + q'_t/s_t - q_t/s_t - eta_t         stored at R bits of one a_hat LSB

  =>  D_t = -e_{t-1} + rho/s_t - eta_t   : bounded, no sum.  Output error = D_T, one term.

Arms: current | sd R in {2,3,4,8} | ahat at 9..12 bits (the "just buy bits" alternative, priced
at 0.125 B/elem per bit, which is what D costs too).  Same validated PyTorch model.
"""
import os, sys, json
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
CAP=torch.load("docs/ahat_accuracy_2026-09-02/data/capture_int8.pt", weights_only=False)
DEV="cuda"; DLIM=127.0; BLK=32
def gn_silu(x, meta, mod):
    G=meta["num_groups"]; N,C,H,W=x.shape
    xg=x.view(N,G,C//G,H,W)
    mu=xg.mean(dim=(2,3,4),keepdim=True); var=xg.var(dim=(2,3,4),unbiased=False,keepdim=True)
    n=((xg-mu)*(var+meta["eps"]).rsqrt()).view(N,C,H,W)
    n=n*meta["weight"].to(DEV).view(1,C,1,1)+meta["bias"].to(DEV).view(1,C,1,1)
    if mod is not None:
        ms,sh=mod; n=n*(1.0+ms.to(DEV).view(N,C,1,1))+sh.to(DEV).view(N,C,1,1)
    n=n.half().float()
    o=n*torch.sigmoid(n) if meta["apply_silu"] else n
    if meta["smooth_inv"] is not None: o=o*meta["smooth_inv"].to(DEV).view(1,C,1,1)
    return o
def to_b(v,N,C,H,W): return v.permute(0,2,3,1).reshape(N,H,W,C//BLK,BLK)
def fr_b(v,N,C,H,W): return v.reshape(N,H,W,C).permute(0,3,1,2).contiguous()
def qa(v,N,C,H,W,bits):
    lim=float(2**(bits-1)-1)
    vb=to_b(v,N,C,H,W); S=vb.abs().amax(-1,keepdim=True).clamp_min(1e-12)/lim
    return fr_b(torch.clamp(torch.round(vb/S),-lim,lim)*S,N,C,H,W), S

out={}
for name,L in CAP["layers"].items():
    C,H,W,N=L["C"],L["H"],L["W"],L["batch"]; T=L["x"].shape[0]
    meta,scales,mods=L["meta"],L["scale"],L["mod"]
    if C%BLK: continue
    O=[gn_silu(L["x"][t].to(DEV).float(),meta,mods[t]) for t in range(T)]
    ra=torch.zeros(N,C,H,W,device=DEV); ref=[]
    for t in range(T):
        s=scales[t]; q=torch.clamp(torch.round((O[t]-ra)*s),-DLIM,DLIM); ra=ra+q/s; ref.append(ra.clone())
    refn=ref[-1].norm(); rows={}
    # ---- plain a_hat at B bits (current is B=8) ----
    for bits in (8,9,10,11,12):
        a=torch.zeros(N,C,H,W,device=DEV); acc=torch.zeros_like(a); per=[]
        for t in range(T):
            s=scales[t]; q=torch.clamp(torch.round((O[t]-a)*s),-DLIM,DLIM)
            cons=a+q/s; acc+=q/s; a,_=qa(cons,N,C,H,W,bits)
            per.append(((acc-cons).norm()/refn).item())
        rows[f"ahat {bits}b"]={"acc_err":per[-1],"bpe":bits/8+4/BLK,"curve":per}
    # ---- sigma-delta drift correction, D stored at R bits of one a_hat LSB ----
    for R in (2,3,4,8):
        a=torch.zeros(N,C,H,W,device=DEV); acc=torch.zeros_like(a)
        D=torch.zeros(N,C,H,W,device=DEV); per=[]; clip=0.0
        for t in range(T):
            s=scales[t]; q=torch.clamp(torch.round((O[t]-a)*s),-DLIM,DLIM)
            cons=a+q/s
            qp=torch.clamp(torch.round(q-D*s),-DLIM,DLIM)          # corrected codes -> the conv
            clip+=(qp!=(q-D*s)).float().mean().item()*0             # (clamp bookkeeping below)
            acc+=qp/s
            a,S=qa(cons,N,C,H,W,8)
            Dn=acc-a                                               # true drift after this step
            Sf=fr_b(S.expand(-1,-1,-1,-1,BLK),N,C,H,W)             # one a_hat LSB per element
            D=torch.round(Dn/Sf*(2**R))/(2**R)*Sf                  # store D at R bits of an LSB
            per.append(((acc-cons).norm()/refn).item())
        rows[f"sd R={R}"]={"acc_err":per[-1],"bpe":1.0+R/8+4/BLK,"curve":per}
    out[name]={"C":C,"H":H,"W":W,"steps":T,"arms":rows}
    print(f"{name}: " + "  ".join(f"{k}={v['acc_err']:.4f}@{v['bpe']:.3f}B" for k,v in rows.items()), flush=True)
    del O,ref; torch.cuda.empty_cache()
json.dump(out,open("docs/ohat_compress_2026-09-03/data/sim_sd.json","w"))
print("\nwrote data/sim_sd.json")
