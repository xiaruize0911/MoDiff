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
# --- mma_smoke sanity ---
K,Nn=64,16; A=torch.randint(-8,8,(16,K),device=dev,dtype=torch.int8); Bm=torch.randint(-8,8,(Nn,K),device=dev,dtype=torch.int8)
Csm=mc.mma_smoke(A,Bm); ref_sm=(A.float()@Bm.float().T).to(torch.int32)
print("mma_smoke exact:", torch.equal(Csm, ref_sm))
# --- flash int8 at level 0: N=128,H=8,T=1024,hd=24 ---
for (N,H,T,hd) in [(128,8,1024,24),(128,8,256,48),(128,8,64,48)]:
    hd_pad=((hd+31)//32)*32
    q=torch.randn(N,H,T,hd,device=dev,dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
    sc=1.0/math.sqrt(hd)
    # quantize: per-token Q/K, per-channel V
    sq=(q.abs().amax(-1).clamp_min(1e-8)/127.0).float()       # [N,H,T]
    sk=(k.abs().amax(-1).clamp_min(1e-8)/127.0).float()
    sv=(v.abs().amax(2).clamp_min(1e-8)/127.0).float()        # [N,H,hd]
    def q_i8(x,s,perch=False):
        if perch: xi=torch.round(x/s.unsqueeze(2)).clamp(-127,127).to(torch.int8)
        else: xi=torch.round(x/s.unsqueeze(-1)).clamp(-127,127).to(torch.int8)
        return F.pad(xi,(0,hd_pad-hd))
    qi=q_i8(q,sq); ki=q_i8(k,sk); vi=q_i8(v,sv,perch=True)
    out=mc.flash_attn_int8(qi,ki,vi,sq,sk,sv,sc)              # [N,H,T,hd]
    # reference fp32
    qf,kf,vf=q.float(),k.float(),v.float()
    S=torch.einsum("nhid,nhjd->nhij",qf,kf)*sc
    ref=torch.einsum("nhij,nhjd->nhid",torch.softmax(S,-1),vf)
    rel=relL2(out,ref)
    # fp16 flash bar
    q4=q.reshape(1,N*H,T,hd)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        tf=bench(lambda: F.scaled_dot_product_attention(q4,k.reshape(1,N*H,T,hd),v.reshape(1,N*H,T,hd),scale=sc))
    ti=bench(lambda: mc.flash_attn_int8(qi,ki,vi,sq,sk,sv,sc))
    print(f"T{T:4d} hd{hd}: int8-flash {ti:7.1f}us  fp16-flash {tf:7.1f}us  ({tf/ti:.2f}x)  rel-L2 {rel:.4f}")
