"""Same metrics under BOTH norms, plus a concentration indicator.

relL2 = ||e||_2 / ||x||_2   is dominated by the largest errors.
relL1 = ||e||_1 / ||x||_1   weights every element equally.

Their ratio says how the error is distributed. For an error spread uniformly over a quantization
step Delta, E|e| = Delta/4 and rms = Delta/sqrt(12), so E|e|/rms = 0.866. Gaussian gives 0.798.
Well below that means the error is concentrated in a minority of elements -- which is exactly what
amax block scaling should do, since ~1/B of the codes sit at the limit by construction.

    conc = mean|e| / sqrt(mean e^2)      0.866 = uniform, 0.798 = Gaussian, lower = concentrated
"""
import os, sys, json, math, statistics as st
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
CAP=torch.load("docs/ahat_accuracy_2026-09-02/data/capture_int8.pt", weights_only=False)
DEV="cuda"; DLIM=127.0
BITS=[3,4,5,6,7,8,9,10,12]; BLKS=[16,32,64]

def gn_silu(x, meta, mod, N,C,H,W):
    G=meta["num_groups"]; xg=x.float().view(N,G,C//G,H,W)
    mu=xg.mean(dim=(2,3,4),keepdim=True); var=xg.var(dim=(2,3,4),unbiased=False,keepdim=True)
    n=((xg-mu)*(var+meta["eps"]).rsqrt()).view(N,C,H,W)
    n=n*meta["weight"].to(DEV).view(1,C,1,1)+meta["bias"].to(DEV).view(1,C,1,1)
    if mod is not None: n=n*(1.0+mod[0].to(DEV).view(N,C,1,1))+mod[1].to(DEV).view(N,C,1,1)
    n=n.half().float(); o=n*torch.sigmoid(n) if meta["apply_silu"] else n
    if meta["smooth_inv"] is not None: o=o*meta["smooth_inv"].to(DEV).view(1,C,1,1)
    return o
def blk_q(v,B,bits,N,C,H,W):
    lim=float(2**(bits-1)-1)
    x=v.permute(0,2,3,1).reshape(N,H,W,C//B,B)
    s=x.abs().amax(-1,keepdim=True).clamp_min(1e-12)/lim
    q=(x/s).round().clamp_(-lim,lim)
    return (q*s).reshape(N,H,W,C).permute(0,3,1,2).contiguous()
def rel(e,x):
    return ((e.abs().sum()/x.abs().sum()).item(), (e.norm()/x.norm()).item())
def conc(e):
    a=e.abs().mean().item(); r=math.sqrt((e*e).mean().item())
    return a/r if r>0 else 0.0

res={}
for name,L in CAP["layers"].items():
    C,H,W,N=L["C"],L["H"],L["W"],L["batch"]; T=L["x"].shape[0]
    o=[gn_silu(L["x"][t].to(DEV),L["meta"],L["mod"][t],N,C,H,W) for t in range(T)]
    ref=torch.zeros(N,C,H,W,device=DEV); cbar=[]
    for t in range(T):
        q=torch.clamp(torch.round((o[t]-ref)*L["scale"][t]),-DLIM,DLIM)
        ref=ref+q/L["scale"][t]; cbar.append(ref.clone())
    for bits in BITS:
        for B in BLKS:
            if C%B: continue
            a=torch.zeros(N,C,H,W,device=DEV); eta=torch.zeros(N,C,H,W,device=DEV)
            m={k:[] for k in ("ec1","ec2","es1","es2","co1","co2","st1","st2","cc")}
            for t in range(T):
                s=L["scale"][t]
                q=torch.clamp(torch.round((o[t]-a)*s),-DLIM,DLIM); cons=a+q/s
                new=blk_q(cons,B,bits,N,C,H,W); e=new-cons; eta+=e
                x=cbar[t]
                for tag,err in (("ec",eta),("es",e),("co",cons-x),("st",new-x)):
                    l1,l2=rel(err,x); m[tag+"1"].append(l1); m[tag+"2"].append(l2)
                m["cc"].append(conc(e)); a=new
            res.setdefault(f"{bits}bit B={B}",{})[name]={
                "ec1":m["ec1"][-1],"ec2":m["ec2"][-1],
                **{k:st.median(v[5:]) for k,v in m.items() if k not in ("ec1","ec2")}}
    del o,cbar; torch.cuda.empty_cache()
json.dump(res,open("docs/ahat_accuracy_2026-09-02/data/l1_l2.json","w"),indent=1)
M=lambda k,per: st.median([v[k] for v in per.values()])
print(f"{'config':>11} | {'eta_cum L1':>10} {'L2':>7} {'L1/L2':>6} | "
      f"{'eta_step L1':>11} {'L2':>7} {'L1/L2':>6} | {'state L1':>8} {'L2':>7} | {'conc':>5}")
print("-"*106)
for k,per in res.items():
    print(f"{k:>11} | {M('ec1',per):10.4f} {M('ec2',per):7.4f} {M('ec1',per)/M('ec2',per):6.3f} | "
          f"{M('es1',per):11.4f} {M('es2',per):7.4f} {M('es1',per)/M('es2',per):6.3f} | "
          f"{M('st1',per):8.4f} {M('st2',per):7.4f} | {M('cc',per):5.3f}")
print(f"\n{'(delta-quantizer floor, consumed)':>36} L1={M('co1',res['8bit B=32']):.4f} "
      f"L2={M('co2',res['8bit B=32']):.4f}")
print("conc reference: 0.866 = error uniform over the quantization step, 0.798 = Gaussian")
