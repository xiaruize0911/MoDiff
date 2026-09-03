"""Test the REAL kernels against the PyTorch model that produced the calibration curves.

Three levels:
 (1) SINGLE STEP, a_hat = 0. No trajectory divergence possible, so any difference is the GN
     reduction order or the quantizers themselves. Compares the delta codes q element-by-element
     and the stored a_hat codes.
 (2) TRAJECTORY, 49 steps. eta_cum / MSE per layer, real kernel vs model.
 (3) CALIBRATION. Does the real kernel's eta_cum, put through image_MSE = 0.0152*eta_cum,
     reproduce the MSE actually measured on its decoded samples?

Real kernels available for a_hat storage: fp16, int8 B=16/32/64, packed int4 B=32 -- on both the
int8 conv path (group_norm_silu_delta_quantize_nhwc) and the int4 one (..._pack_nhwc).
"""
import os, sys, json, math, statistics as st
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, modiff_cutlass as mc
CAP=torch.load("docs/ahat_accuracy_2026-09-02/data/capture_int8.pt", weights_only=False)
DEV,CL="cuda",torch.channels_last
ef=torch.empty(0,device=DEV,dtype=torch.float32); eh=torch.empty(0,device=DEV,dtype=torch.float16)
ei=torch.empty(0,device=DEV,dtype=torch.int32)

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
    return ((q*s).reshape(N,H,W,C).permute(0,3,1,2).contiguous(), q)
def mk_state(kind,B,N,C,H,W):
    if kind=="fp16":
        return torch.empty(N,C,H,W,device=DEV,dtype=torch.float16,memory_format=CL).zero_(), ef
    ch=C//2 if kind=="i4" else C
    return (torch.empty(N,ch,H,W,device=DEV,dtype=torch.int8,memory_format=CL).zero_(),
            torch.ones(N,H,W,C//B,device=DEV,dtype=torch.float32))
def deq(A,S,kind,N,C,H,W,B):
    if kind=="fp16": return A.float()
    if kind=="i8": q=A.permute(0,2,3,1).float()
    else:
        by=A.permute(0,2,3,1).contiguous().reshape(-1).to(torch.uint8)
        lo=(by&0xF).to(torch.int16); lo=torch.where(lo>7,lo-16,lo)
        hi=((by>>4)&0xF).to(torch.int16); hi=torch.where(hi>7,hi-16,hi)
        q=torch.stack([lo,hi],-1).reshape(N,H,W,C).float()
    return (q.view(N,H,W,C//B,B)*S.view(N,H,W,C//B,1)).reshape(N,H,W,C).permute(0,3,1,2)
def call(path,x,meta,mod,s,A,S,N,C,H,W):
    w=meta["weight"].to(DEV).half(); b=meta["bias"].to(DEV).half()
    ms,sh=(eh,eh) if mod is None else (mod[0].to(DEV).half().contiguous(),
                                       mod[1].to(DEV).half().contiguous())
    si=ef if meta["smooth_inv"] is None else meta["smooth_inv"].to(DEV).contiguous()
    sc=torch.full((1,),float(s),device=DEV,dtype=torch.float32)
    DL=127.0 if path=="int8" else 7.0
    a=(x.contiguous(memory_format=CL),w,b,A,meta["num_groups"],meta["eps"],meta["apply_silu"],
       sc,si,ms,sh,ef,ef,ef,ei,DL,False,1.0)
    if path=="int8":
        return mc.group_norm_silu_delta_quantize_nhwc(*a,False,True,S).float(), DL
    yqp=mc.group_norm_silu_delta_quantize_pack_nhwc(*a,True,S)
    by=yqp.reshape(-1).to(torch.uint8)
    lo=(by&0xF).to(torch.int16); lo=torch.where(lo>7,lo-16,lo)
    hi=((by>>4)&0xF).to(torch.int16); hi=torch.where(hi>7,hi-16,hi)
    return torch.stack([lo,hi],-1).reshape(N,H,W,C).float().permute(0,3,1,2), DL

CFG=[("fp16",16,0),("i8",8,16),("i8",8,32),("i8",8,64),("i4",4,32)]
out={}
print("(1) SINGLE STEP with a_hat=0 — delta-code disagreement between real kernel and model")
print(f"{'path':>5} {'a_hat':>10} | " + " | ".join(f"{n.split('_')[0]:>11}" for n in CAP["layers"]))
for path in ("int8","int4"):
    for kind,bits,B in CFG:
        row=[]
        for name,L in CAP["layers"].items():
            C,H,W,N=L["C"],L["H"],L["W"],L["batch"]
            if kind!="fp16" and C%B: row.append("   n/a"); continue
            A,S=mk_state(kind,B,N,C,H,W)
            q_k,DL=call(path,L["x"][0].to(DEV),L["meta"],L["mod"][0],L["scale"][0],A,S,N,C,H,W)
            o=gn_silu(L["x"][0].to(DEV),L["meta"],L["mod"][0],N,C,H,W)
            q_m=torch.clamp(torch.round(o*L["scale"][0]),-DL,DL)
            row.append(f"{(q_k!=q_m).float().mean().item():11.2e}")
            del A,S
        print(f"{path:>5} {kind+(f' B={B}' if kind!='fp16' else ''):>10} | " + " | ".join(row))

print("\n(2) TRAJECTORY, 49 steps — eta_cum: real kernel vs model (int8 conv path)")
print(f"{'a_hat':>10} | {'kernel':>8} {'model':>8} {'ratio':>6} | {'kernel MSE':>11} {'model MSE':>11}")
for kind,bits,B in CFG:
    ks,ms_,kmse,mmse=[],[],[],[]
    for name,L in CAP["layers"].items():
        C,H,W,N=L["C"],L["H"],L["W"],L["batch"]; T=L["x"].shape[0]
        if kind!="fp16" and C%B: continue
        o=[gn_silu(L["x"][t].to(DEV),L["meta"],L["mod"][t],N,C,H,W) for t in range(T)]
        ref=torch.zeros(N,C,H,W,device=DEV); cbar=[]
        for t in range(T):
            q=torch.clamp(torch.round((o[t]-ref)*L["scale"][t]),-127.,127.)
            ref=ref+q/L["scale"][t]; cbar.append(ref.clone())
        # real kernel
        A,S=mk_state(kind,B,N,C,H,W); eta=torch.zeros(N,C,H,W,device=DEV); step=[]
        for t in range(T):
            ap=deq(A,S,kind,N,C,H,W,B if kind!="fp16" else 32)
            q,_=call("int8",L["x"][t].to(DEV),L["meta"],L["mod"][t],L["scale"][t],A,S,N,C,H,W)
            cons=ap+q/L["scale"][t]
            new=deq(A,S,kind,N,C,H,W,B if kind!="fp16" else 32)
            eta+=new-cons; step.append(((new-cons)**2).mean().item())
        ks.append((eta.norm()/cbar[-1].norm()).item()); kmse.append(st.median(step[5:]))
        del A,S
        # model
        a=torch.zeros(N,C,H,W,device=DEV); eta=torch.zeros(N,C,H,W,device=DEV); step=[]
        for t in range(T):
            q=torch.clamp(torch.round((o[t]-a)*L["scale"][t]),-127.,127.); cons=a+q/L["scale"][t]
            new=cons.half().float() if kind=="fp16" else blk_q(cons,B,bits,N,C,H,W)[0]
            eta+=new-cons; step.append(((new-cons)**2).mean().item()); a=new
        ms_.append((eta.norm()/cbar[-1].norm()).item()); mmse.append(st.median(step[5:]))
        del o,cbar; torch.cuda.empty_cache()
    k,m=st.mean(ks),st.mean(ms_)
    out[f"{kind} B={B}" if kind!="fp16" else "fp16"]={"kernel_eta_cum":k,"model_eta_cum":m,
        "kernel_mse":st.mean(kmse),"model_mse":st.mean(mmse)}
    print(f"{(kind+(f' B={B}' if kind!='fp16' else '')):>10} | {k:8.4f} {m:8.4f} {k/max(m,1e-12):6.3f} | "
          f"{st.mean(kmse):11.3e} {st.mean(mmse):11.3e}")

print("\n(3) CALIBRATION — real-kernel eta_cum -> predicted image MSE vs the MSE actually measured")
IMG=json.load(open("docs/ahat_accuracy_2026-09-02/data/mse.json"))["image"]
meas={"i8 B=32":next(d["mse"] for d in IMG if d["arm"]=="int4_ahat32")}
print(f"{'arm':>10} | {'kernel eta_cum':>14} | {'predicted MSE':>13} | {'measured MSE':>12} | {'ratio':>6}")
for k,v in out.items():
    if k not in meas: continue
    pred=0.0152*v["kernel_eta_cum"]
    print(f"{k:>10} | {v['kernel_eta_cum']:14.4f} | {pred:13.3e} | {meas[k]:12.3e} | {meas[k]/pred:6.2f}")
json.dump(out,open("docs/ahat_accuracy_2026-09-02/data/validate_kernel.json","w"),indent=1)
