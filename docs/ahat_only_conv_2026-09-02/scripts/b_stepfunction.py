"""Is the B!=32 slowdown a step function at 32 (compile-time specialization) or continuous in B
(reduction width)?  And does it live in the a_hat WRITE?  Sweep B and toggle write_ahat."""
import os,sys,json,statistics
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, modiff_cutlass as mc
DEV,CL="cuda",torch.channels_last
WU,R,G,EPS=8,25,32,1e-6
ef=torch.empty(0,device=DEV,dtype=torch.float32); eh=torch.empty(0,device=DEV,dtype=torch.float16)
ei=torch.empty(0,device=DEV,dtype=torch.int32)
def bench(fn):
    for _ in range(WU): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(R):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)
def one(N,C,H,W,blk,write):
    x=torch.empty(N,C,H,W,device=DEV,dtype=torch.float16,memory_format=CL).normal_()
    gw=torch.randn(C,device=DEV,dtype=torch.float16); gb=torch.randn(C,device=DEV,dtype=torch.float16)
    sc=torch.full((1,),16.0,device=DEV,dtype=torch.float32)
    A=torch.empty(N,C,H,W,device=DEV,dtype=torch.int8,memory_format=CL).zero_()
    As=ef if blk==0 else torch.ones(N,H,W,C//blk,device=DEV,dtype=torch.float32)
    fn=lambda: mc.group_norm_silu_delta_quantize_nhwc(
        x,gw,gb,A,G,EPS,True,sc,ef,eh,eh,ef,ef,ef,ei,127.0,False,1.0,False,write,As)
    fn(); torch.cuda.synchronize(); t=bench(fn)
    del x,A,As,fn; torch.cuda.empty_cache(); return t
SH=[(128,384,32,32),(128,768,16,16),(128,192,32,32),(128,1536,4,4)]
BS=[2,4,8,16,32,64]
for write in (True,False):
    print(f"\n===== write_ahat = {write} =====")
    print(f"{'shape':>18} " + " ".join(f"{'B='+str(b):>9}" for b in BS) )
    print(f"{'':>18} " + " ".join(f"{'':>9}" for b in BS) )
    for (N,C,H,W) in SH:
        ts={}
        for b in BS:
            ts[b]=one(N,C,H,W,b,write) if C%b==0 else None
        pt=float("nan")
        base=ts[32]
        print(f"{f'C{C} {H}x{W}':>18} " + " ".join(
            ("%9s"%"-") if ts[b] is None else f"{ts[b]/base:8.3f}x" for b in BS) )
        print(f"{'  ms':>18} " + " ".join(
            ("%9s"%"-") if ts[b] is None else f"{ts[b]:9.4f}" for b in BS) )
