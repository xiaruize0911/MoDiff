"""Why the single-layer blockwise ratios differ so much: bandwidth-bound? vec4-eligible?

Prints, per layer, the achieved bandwidth of the baseline (to show launch-bound vs
bandwidth-bound) and CPG%4 (blk32_vec4 needs CPG%4==0; group_norm_silu.cu:2589).
"""
import json, os, statistics, sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, modiff_cutlass as mc
DEV,CL="cuda",torch.channels_last
W_,R,G,EPS,BLK=8,25,32,1e-6,32
def bench(fn):
    for _ in range(W_): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(R):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)
def arms(B,C,H,W):
    ef=torch.empty(0,device=DEV,dtype=torch.float32); eh=torch.empty(0,device=DEV,dtype=torch.float16)
    ei=torch.empty(0,device=DEV,dtype=torch.int32)
    x=torch.empty(B,C,H,W,device=DEV,dtype=torch.float16,memory_format=CL).normal_()
    gw=torch.randn(C,device=DEV,dtype=torch.float16); gb=torch.randn(C,device=DEV,dtype=torch.float16)
    sc=torch.full((1,),16.0,device=DEV,dtype=torch.float32)
    out={}
    out["base"]=bench(lambda: mc.group_norm_silu_quantize_nhwc_fast(x,gw,gb,G,EPS,True,sc,ef,eh,eh))
    A=torch.empty(B,C,H,W,device=DEV,dtype=torch.float16,memory_format=CL).zero_()
    out["fp16"]=bench(lambda: mc.group_norm_silu_delta_quantize_nhwc(
        x,gw,gb,A,G,EPS,True,sc,ef,eh,eh,ef,ef,ef,ei,127.0,False,1.0,False,True,ef))
    A8=torch.empty(B,C,H,W,device=DEV,dtype=torch.int8,memory_format=CL).zero_()
    As=torch.ones(B,H,W,C//BLK,device=DEV,dtype=torch.float32)
    out["b32"]=bench(lambda: mc.group_norm_silu_delta_quantize_nhwc(
        x,gw,gb,A8,G,EPS,True,sc,ef,eh,eh,ef,ef,ef,ei,127.0,False,1.0,False,True,As))
    del x,A,A8,As; torch.cuda.empty_cache()
    return out
LAYERS=[(768,2,2),(768,4,4),(384,8,8),(384,16,16),(768,16,16),(384,32,32),
        (192,32,32),(576,32,32),(1152,8,8),(1536,4,4)]
B=128
print(f"{'C':>5} {'HxW':>7} {'CPG':>4} {'vec4':>5} | {'base GB/s':>9} | "
      f"{'fp16 a_hat':>10} {'b32':>8} | {'b32/fp16':>9}")
print("-"*76)
for C,H,W in LAYERS:
    a=arms(B,C,H,W)
    e=B*C*H*W
    gbs=(5*e/1e9)/(a["base"]/1e3)          # baseline moves ~5 B/elem (2 passes read x, write int8)
    cpg=C//G
    print(f"{C:>5} {f'{H}x{W}':>7} {cpg:>4} {'yes' if cpg%4==0 else 'NO':>5} | {gbs:9.0f} | "
          f"{a['fp16']/a['base']:9.3f}x {a['b32']/a['base']:7.3f}x | {a['b32']/a['fp16']:8.3f}x")
