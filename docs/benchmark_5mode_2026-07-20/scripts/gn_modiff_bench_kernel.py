"""Kernel microbench: fused group_norm_silu_delta_quantize_nhwc vs the two-kernel
reference (group_norm_silu_nhwc + step1_static_quantize_fprop_silu), at churches
modiff conv shapes. CUDA-event median of 200 iters, 50 warm."""
import torch, statistics, modiff_cutlass as M
dev='cuda'
def cl(t): return t.contiguous(memory_format=torch.channels_last)
def bench(fn, warm=50, iters=200):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000)  # us
    return statistics.median(ts)

SHAPES=[("res_128_64",128,64),("res_256_32",256,32),("down_256_512_16",512,16),
        ("mid_512_8",512,8),("up_512_256_16",256,16)]
N,ng=128,32
print(f"{'shape':18} {'2-kernel us':>12} {'fused us':>10} {'speedup':>8}")
for name,C,H in SHAPES:
    W=H
    x=cl(torch.randn(N,C,H,W,device=dev,dtype=torch.float16))
    g=torch.randn(C,device=dev,dtype=torch.float16); b=torch.randn(C,device=dev,dtype=torch.float16)
    scale=torch.tensor([127.0/3],device=dev,dtype=torch.float32)
    smooth=torch.empty(0,device=dev,dtype=torch.float32)
    ms=sh=torch.empty(0,device=dev,dtype=torch.float16)
    a1=cl(torch.zeros(N,C,H,W,device=dev,dtype=torch.float16))
    a2=cl(torch.zeros(N,C,H,W,device=dev,dtype=torch.float16))
    def two():
        nm=M.group_norm_silu_nhwc(x,g,b,ng,1e-5,False,ms,sh)
        M.step1_static_quantize_fprop_silu(nm,a1,scale,smooth)
    def fused():
        M.group_norm_silu_delta_quantize_nhwc(x,g,b,a2,ng,1e-5,True,scale,smooth,ms,sh)
    t2=bench(two); tf=bench(fused)
    print(f"{name:18} {t2:12.1f} {tf:10.1f} {t2/tf:7.2f}x")
