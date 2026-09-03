"""Correctness gate for the generic-block-index change: for every B, the kernel's delta codes and
its re-stored a_hat must match an exact fp32 per-block reference."""
import os,sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, modiff_cutlass as mc
DEV,CL="cuda",torch.channels_last
G,EPS,DLIM=32,1e-6,127.0
ef=torch.empty(0,device=DEV,dtype=torch.float32); eh=torch.empty(0,device=DEV,dtype=torch.float16)
ei=torch.empty(0,device=DEV,dtype=torch.int32)
torch.manual_seed(0)
def gn_silu(x,w,b):
    N,C,H,W=x.shape; xg=x.float().view(N,G,C//G,H,W)
    mu=xg.mean((2,3,4),keepdim=True); var=xg.var((2,3,4),unbiased=False,keepdim=True)
    n=((xg-mu)*(var+EPS).rsqrt()).view(N,C,H,W)
    n=n*w.float().view(1,C,1,1)+b.float().view(1,C,1,1)
    n=n.half().float(); return n*torch.sigmoid(n)
print(f"{'B':>4} {'code mismatch':>14} {'a_hat relL2':>12} {'scale relL2':>12}")
for blk in (2,4,8,16,32,64):
    N,C,H,W=2,384,8,8
    x=torch.empty(N,C,H,W,device=DEV,dtype=torch.float16,memory_format=CL).normal_()
    gw=torch.randn(C,device=DEV,dtype=torch.float16); gb=torch.randn(C,device=DEV,dtype=torch.float16)
    s=8.0; sc=torch.full((1,),s,device=DEV,dtype=torch.float32)
    # seed a_hat with a real quantized state so the READ path is exercised too
    a0=torch.empty(N,C,H,W,device=DEV,dtype=torch.float32).normal_()
    ab=a0.permute(0,2,3,1).reshape(N,H,W,C//blk,blk)
    S=(ab.abs().amax(-1,keepdim=True)/127.0).clamp_min(1e-12)
    code=torch.clamp(torch.round(ab/S),-127,127)
    A=code.reshape(N,H,W,C).permute(0,3,1,2).to(torch.int8).contiguous(memory_format=CL)
    As=S.squeeze(-1).contiguous()
    a_prev=(code*S).reshape(N,H,W,C).permute(0,3,1,2).contiguous()
    o=gn_silu(x,gw,gb)
    q_ref=torch.clamp(torch.round((o-a_prev)*s),-DLIM,DLIM)
    cons=a_prev+q_ref/s
    cb=cons.permute(0,2,3,1).reshape(N,H,W,C//blk,blk)
    S2=(cb.abs().amax(-1,keepdim=True)/127.0).clamp_min(1e-12)
    a_ref=(torch.clamp(torch.round(cb/S2),-127,127)*S2).reshape(N,H,W,C).permute(0,3,1,2)
    Ak=A.clone(); Ask=As.clone()
    yq=mc.group_norm_silu_delta_quantize_nhwc(x,gw,gb,Ak,G,EPS,True,sc,ef,eh,eh,
                                              ef,ef,ef,ei,DLIM,False,1.0,False,True,Ask)
    mism=(yq.float()!=q_ref).float().mean().item()
    got=(Ak.permute(0,2,3,1).reshape(N,H,W,C//blk,blk).float()*Ask.unsqueeze(-1)) \
        .reshape(N,H,W,C).permute(0,3,1,2)
    e=((got-a_ref).norm()/a_ref.norm()).item()
    es=((Ask-S2.squeeze(-1)).norm()/S2.norm()).item()
    flag="OK" if (mism==0 and e<2e-3 and es<1e-5) else "**FAIL**"
    print(f"{blk:>4} {mism:14.2e} {e:12.3e} {es:12.3e}  {flag}")
