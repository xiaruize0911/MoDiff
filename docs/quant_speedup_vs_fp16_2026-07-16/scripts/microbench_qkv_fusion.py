"""Microbench the qkv-side quantize fusion: unfused = native GN (fp16) + quantize_act_int8;
fused = group_norm_silu_quantize_nhwc (GN in fp32 -> int8 in one kernel). Real b64 attention
GN shapes (N,C,H,W with T=H*W). Measures the per-step saving from folding qkv's quantize into GN."""
import os, sys; os.chdir("/workspace/MoDiff"); sys.path.insert(0,"/workspace/MoDiff")
import torch, modiff_cutlass as mc
def bench(fn,it=200,warm=50,reps=5):
    ts=[]
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e)/it*1e3)
    ts.sort(); return ts[len(ts)//2]
NG=32
# (C, side, count) : attention GN inputs at b64, T=side^2
SH=[(192,32,5),(384,16,5),(384,8,5),(768,4,5),(768,2,1)]
B=64
tot_un=tot_fu=0.0
print(f"{'C':>5}{'HxW':>7}{'cnt':>4} | {'GN+quant(un)':>13} {'GN-quant(fu)':>13} {'save/inst':>10}")
for (C,side,cnt) in SH:
    x=torch.randn(B,C,side,side,device="cuda",dtype=torch.float16).to(memory_format=torch.channels_last)
    w=torch.randn(C,device="cuda",dtype=torch.float16); bnorm=torch.randn(C,device="cuda",dtype=torch.float16)
    sc=torch.tensor([8.0],device="cuda",dtype=torch.float32); empty=x.new_empty(0)
    def un():
        y=mc.group_norm_silu_nhwc(x,w,bnorm,NG,1e-5,False,empty,empty)          # GN -> fp16
        return mc.quantize_act_int8(y.permute(0,2,3,1).reshape(B*side*side,C).contiguous(), 0.125)
    def fu():
        return mc.group_norm_silu_quantize_nhwc(x,w,bnorm,NG,1e-5,False,sc,empty,empty,empty)  # GN -> int8
    u=bench(un); f=bench(fu); tot_un+=u*cnt; tot_fu+=f*cnt
    print(f"{C:>5}{f'{side}x{side}':>7}{cnt:>4} | {u:>13.2f} {f:>13.2f} {u-f:>10.2f}  (us)")
print(f"\nper-step qkv GN+quant: unfused {tot_un:.1f}us -> fused {tot_fu:.1f}us  (saved {tot_un-tot_fu:.1f}us/step, {(1-tot_fu/tot_un)*100:.0f}%)")
