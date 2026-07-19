import os,sys,math; os.chdir("/workspace/MoDiff"); sys.path.insert(0,"/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
torch.manual_seed(0); dev="cuda"
def bench(fn,it=50,warm=20,reps=5):
    ts=[]
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e)/it*1e3)
    ts.sort(); return ts[len(ts)//2]
def relL2(a,b): return (a.float()-b.float()).norm().item()/(b.float().norm().item()+1e-9)
def pack_i4(qi,hdp4):  # qi int8 [N,H,T,hd] in [-8,7] -> packed uint8->int8 [N,H,T,hdp4/2]
    hd=qi.shape[-1]; qi=F.pad(qi,(0,hdp4-hd))
    lo=(qi[...,0::2].int()&0xF); hi=(qi[...,1::2].int()&0xF)
    return (lo|(hi<<4)).to(torch.uint8).view(torch.int8).contiguous()
for (N,H,T,hd) in [(128,8,1024,24),(128,8,256,48)]:
    hdp4=64; hdp_v=((hd+31)//32)*32
    q=torch.randn(N,H,T,hd,device=dev,dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    sc=1.0/math.sqrt(hd)
    sq=(q.abs().amax(-1).clamp_min(1e-8)/7.0).float(); sk=(k.abs().amax(-1).clamp_min(1e-8)/7.0).float()
    sv=(v.abs().amax(2).clamp_min(1e-8)/127.0).float()
    qi=torch.round(q/sq.unsqueeze(-1)).clamp(-8,7).to(torch.int8)
    ki=torch.round(k/sk.unsqueeze(-1)).clamp(-8,7).to(torch.int8)
    vi=F.pad(torch.round(v/sv.unsqueeze(2)).clamp(-127,127).to(torch.int8),(0,hdp_v-hd))
    q4=pack_i4(qi,hdp4); k4=pack_i4(ki,hdp4)
    out=mc.flash_attn_int4(q4,k4,vi,sq,sk,sv,hdp4,sc)
    S=torch.einsum("nhid,nhjd->nhij",q.float(),k.float())*sc
    ref=torch.einsum("nhij,nhjd->nhid",torch.softmax(S,-1),v.float())
    rel=relL2(out,ref)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        tf=bench(lambda: F.scaled_dot_product_attention(q.reshape(1,N*H,T,hd),k.reshape(1,N*H,T,hd),v.reshape(1,N*H,T,hd),scale=sc))
    ti=bench(lambda: mc.flash_attn_int4(q4,k4,vi,sq,sk,sv,hdp4,sc))
    print(f"T{T:4d} hd{hd}: int4-flash {ti:7.1f}us  fp16-flash {tf:7.1f}us  ({tf/ti:.2f}x)  rel-L2 {rel:.4f}")
