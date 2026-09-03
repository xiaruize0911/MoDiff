"""Repair pass on the windowed-frozen scheme, plus the strong alternative.

v1 found: clip_i == 0 (the increment transport is fine) but clip_a explodes with K (10.9% at K=1
to 87% at K=49) and even K=1 is 1.66x worse than current -- because telescoping needs a CONSTANT
scale, and re-gridding the state at a boundary costs a full eta.

Arms here:
  current                baseline, as v1
  alignedK+fc            window K, and the re-grid residual is CARRIED (not just the clamp one)
  alignedK+fc,m=M        same, frozen scale given M x headroom so the stale grid stops clipping
  accref rb=R            the strong version: reference the delta to what the CONV ACTUALLY
                         ACCUMULATED rather than to a_hat.  Requires storing that value; store it
                         as a_hat(8b blockwise) + an R-bit residual of one a_hat LSB.  R=0 is
                         plain a_hat (== current's reference), R=inf is exact.
Same PyTorch model as v1 (validated to 1e-4 vs the real kernel, REPORT.md section 3).
"""
import os, sys, json
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
CAP=torch.load("docs/ahat_accuracy_2026-09-02/data/capture_int8.pt", weights_only=False)
DEV="cuda"; DLIM=127.0; ALIM=127.0; BLK=32

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
def amax_b(v,N,C,H,W): return to_b(v,N,C,H,W).abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)

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
    # ---- current ----
    a=torch.zeros(N,C,H,W,device=DEV); acc=torch.zeros_like(a); per=[]
    for t in range(T):
        s=scales[t]; q=torch.clamp(torch.round((O[t]-a)*s),-DLIM,DLIM)
        cons=a+q/s; acc+=q/s
        S=amax_b(cons,N,C,H,W)/ALIM
        a=fr_b(torch.clamp(torch.round(to_b(cons,N,C,H,W)/S),-ALIM,ALIM)*S,N,C,H,W)
        per.append(((acc-cons).norm()/refn).item())
    rows["current"]={"acc_err":per[-1],"curve":per}
    # ---- aligned, window K, full carry, headroom m ----
    for K,m in [(2,1.0),(5,1.0),(10,1.0),(49,1.0),(5,1.5),(10,1.5),(49,1.5),(10,2.0),(49,2.0),(49,4.0)]:
        a=torch.zeros(N,C,H,W,device=DEV); ac=torch.zeros(N,H,W,C//BLK,BLK,device=DEV)
        S=None; accs=torch.zeros(N,C,H,W,device=DEV)
        carry=torch.zeros(N,H,W,C//BLK,BLK,device=DEV); ca=ci=0.0; per=[]
        for t in range(T):
            s=scales[t]; q=torch.clamp(torch.round((O[t]-a)*s),-DLIM,DLIM); cons=a+q/s
            if t%K==0:
                Sn=amax_b(cons,N,C,H,W)*m/ALIM
                if S is not None:
                    tgt=ac*S/Sn; ac=torch.round(tgt); carry=carry*S/Sn+(tgt-ac)   # CARRY the re-grid
                S=Sn
            code=torch.clamp(torch.round(to_b(cons,N,C,H,W)/S),-ALIM,ALIM)
            ca+=(code.abs()==ALIM).float().mean().item()
            inc=code-ac+carry; incc=torch.clamp(torch.round(inc),-DLIM,DLIM)
            ci+=(incc!=inc).float().mean().item(); carry=inc-incc
            ac=ac+incc; accs+=fr_b(incc*S,N,C,H,W)
            a=fr_b(ac*S,N,C,H,W)
            per.append(((accs-cons).norm()/refn).item())
        rows[f"aligned K={K}+fc,m={m}"]={"acc_err":per[-1],"clip_a":ca/T,"clip_i":ci/T,"curve":per}
    # ---- accref: delta referenced to what the conv accumulated; state = a_hat 8b + R-bit residual
    for R in (0,2,3,4,6):
        a=torch.zeros(N,C,H,W,device=DEV); accs=torch.zeros_like(a); per=[]
        for t in range(T):
            s=scales[t]
            q=torch.clamp(torch.round((O[t]-a)*s),-DLIM,DLIM)      # delta vs the STORED state
            tgt=a+q/s                                              # where the conv should land
            inc=tgt-accs; accs=accs+inc                            # exact by construction
            S=amax_b(tgt,N,C,H,W)/ALIM
            cb=torch.clamp(torch.round(to_b(tgt,N,C,H,W)/S),-ALIM,ALIM)
            res=to_b(tgt,N,C,H,W)/S-cb                             # in a_hat LSB units, |res|<=0.5
            rq=torch.round(res*(2**R))/(2**R) if R>0 else torch.zeros_like(res)
            a=fr_b((cb+rq)*S,N,C,H,W)
            per.append(((accs-tgt).norm()/refn).item())
        rows[f"accref rb={R}"]={"acc_err":per[-1],"curve":per}
    out[name]={"C":C,"H":H,"W":W,"steps":T,"arms":rows}
    b=rows["current"]["acc_err"]
    print(f"{name}: current {b:.4f} | " + " ".join(
        f"{k.replace('aligned ','A').replace('+fc','').replace('accref ','R')}={v['acc_err']:.4f}"
        for k,v in rows.items() if k!="current"), flush=True)
    del O,ref; torch.cuda.empty_cache()
json.dump(out,open("docs/ohat_compress_2026-09-03/data/sim_aligned2.json","w"))
print("\nwrote data/sim_aligned2.json")
