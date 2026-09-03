"""Full 2D sweep: a_hat bit width x along-C block size, on the real captured kernel-1 inputs.

Validated against the real kernels at (8,B) and (4,32) in validate_kernel.py (agreement 1e-4),
so the model stands in for the kernel everywhere on this grid.
Per-tensor (B = C) is included as the B -> infinity limit, i.e. no blockwise at all.
"""
import os, sys, json, math, statistics as st
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
CAP=torch.load("docs/ahat_accuracy_2026-09-02/data/capture_int8.pt", weights_only=False)
DEV="cuda"; DLIM=127.0
BITS=[3,4,5,6,7,8,9,10,12]
BLKS=[2,4,8,16,32,64,128,"C"]

def gn_silu(x,meta,mod,N,C,H,W):
    G=meta["num_groups"]; xg=x.float().view(N,G,C//G,H,W)
    mu=xg.mean(dim=(2,3,4),keepdim=True); var=xg.var(dim=(2,3,4),unbiased=False,keepdim=True)
    n=((xg-mu)*(var+meta["eps"]).rsqrt()).view(N,C,H,W)
    n=n*meta["weight"].to(DEV).view(1,C,1,1)+meta["bias"].to(DEV).view(1,C,1,1)
    if mod is not None: n=n*(1.0+mod[0].to(DEV).view(N,C,1,1))+mod[1].to(DEV).view(N,C,1,1)
    n=n.half().float(); o=n*torch.sigmoid(n) if meta["apply_silu"] else n
    if meta["smooth_inv"] is not None: o=o*meta["smooth_inv"].to(DEV).view(1,C,1,1)
    return o
def blk_q(v,B,bits,N,C,H,W):
    lim=float(2**(bits-1)-1); x=v.permute(0,2,3,1).reshape(N,H,W,C//B,B)
    s=x.abs().amax(-1,keepdim=True).clamp_min(1e-12)/lim
    q=(x/s).round().clamp_(-lim,lim)
    return ((q*s).reshape(N,H,W,C).permute(0,3,1,2).contiguous(),
            (q.abs()==lim).float().mean().item())

res={}
for name,L in CAP["layers"].items():
    C,H,W,N=L["C"],L["H"],L["W"],L["batch"]; T=L["x"].shape[0]
    o=[gn_silu(L["x"][t].to(DEV),L["meta"],L["mod"][t],N,C,H,W) for t in range(T)]
    ref=torch.zeros(N,C,H,W,device=DEV); cbar=[]
    for t in range(T):
        q=torch.clamp(torch.round((o[t]-ref)*L["scale"][t]),-DLIM,DLIM)
        ref=ref+q/L["scale"][t]; cbar.append(ref.clone())
    pw=st.median([(c*c).mean().item() for c in cbar[5:]])
    for bits in BITS:
        for Bs in BLKS:
            B = C if Bs=="C" else Bs
            if C%B: continue
            a=torch.zeros(N,C,H,W,device=DEV); eta=torch.zeros(N,C,H,W,device=DEV)
            es,sa=[],[]
            for t in range(T):
                s=L["scale"][t]
                q=torch.clamp(torch.round((o[t]-a)*s),-DLIM,DLIM); cons=a+q/s
                new,sat=blk_q(cons,B,bits,N,C,H,W); e=new-cons; eta+=e
                es.append((e*e).mean().item()); sa.append(sat); a=new
            res.setdefault(f"{bits}|{Bs}",{})[name]={
                "eta_cum": (eta.norm()/cbar[-1].norm()).item(),
                "mse_step": st.median(es[5:]), "sat": st.median(sa[5:]),
                "nmse_step": st.median(es[5:])/pw, "bpe": bits/8 + 4/B}
    del o,cbar; torch.cuda.empty_cache()
json.dump(res,open("docs/ahat_accuracy_2026-09-02/data/grid_2d.json","w"),indent=1)

M=lambda k,per: st.median([v[k] for v in per.values()])
print("eta_cum（5 层中位数）。行=bit，列=block。'C'=per-tensor（无 blockwise）")
hdr="bits |" + "".join(f"{('B='+str(b)) if b!='C' else 'per-tensor':>12}" for b in BLKS)
print(hdr); print("-"*len(hdr))
for bits in BITS:
    row=f"{bits:>4} |"
    for Bs in BLKS:
        k=f"{bits}|{Bs}"
        row += f"{M('eta_cum',res[k]):12.4f}" if k in res else f"{'--':>12}"
    print(row)
print("\nB/elem（bits/8 + 4/B，fp32 block scale）")
print(hdr)
for bits in BITS:
    row=f"{bits:>4} |"
    for Bs in BLKS:
        k=f"{bits}|{Bs}"
        row += f"{M('bpe',res[k]):12.4f}" if k in res else f"{'--':>12}"
    print(row)
