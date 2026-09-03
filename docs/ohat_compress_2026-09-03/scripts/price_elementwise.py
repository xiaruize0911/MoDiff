"""Price the no-EVT route for int8 o_hat.

CUTLASS EVT cannot broadcast a per-(pixel x K-block) scale, which is what B=32-along-K needs. The
route that avoids EVT surgery reuses conv2d_int8_evt_o_hat_skip (out = o_hat_old + conv, no store)
and puts the format change in two elementwise passes around it:

  1  dequant   int8 o_hat + scales  ->  ONE transient fp16 buffer, reused across layers
  2  conv      the existing _skip kernel, unchanged
  3  quantize  out -> int8 o_hat + scales, stochastic rounding   (conv_quantize_block_nhwc + SR)

Persistent storage becomes int8 (636 MB) plus one transient fp16 buffer at the largest layer's
size, instead of 70 persistent fp16 buffers. This measures what passes 1 and 3 cost.
"""
import os,sys,json,statistics
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, modiff_cutlass as mc
DEV,CL="cuda",torch.channels_last
WU,R,BLK=8,25,32
SHAPES=[(s["C"],s["H"],s["W"],s["B"],s["freq"]) for s in
        json.load(open("docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json"))["unet"]]
def bench(fn):
    for _ in range(WU): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(R):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)
tot={"deq_eager":0.,"quant_kernel":0.,"quant_eager":0.,"bw_bound":0.,"ohat_MB":0.}
PEAK=696.0
print(f"{'shape':>16} {'deq eager':>10} {'quant krn':>10} {'quant eag':>10} {'bw bound':>9}")
for C,H,W,B,f in SHAPES:
    oh=torch.empty(B,C,H,W,device=DEV,dtype=torch.float16,memory_format=CL).normal_()
    q=torch.randint(-127,128,(B,C,H,W),device=DEV,dtype=torch.int8).contiguous(memory_format=CL)
    sc=torch.rand(B,H,W,C//BLK,device=DEV,dtype=torch.float32)*0.01+1e-3
    n=B*C*H*W
    # 1: dequant int8+scales -> fp16 (eager; a fused kernel would be at the bandwidth bound)
    def deq():
        v=q.permute(0,2,3,1).reshape(B,H,W,C//BLK,BLK).float()*sc.unsqueeze(-1)
        oh.copy_(v.reshape(B,H,W,C).permute(0,3,1,2))
    t_deq=bench(deq)
    # 3: quantize fp16 -> int8 + scales. The real kernel (no SR yet) and the eager+SR version.
    t_qk=bench(lambda: mc.conv_quantize_block_nhwc(oh,BLK))
    def qe():
        v=oh.permute(0,2,3,1).reshape(B,H,W,C//BLK,BLK).float()
        s=v.abs().amax(-1,keepdim=True).clamp_min(1e-12)/127.0
        torch.floor(v/s+torch.rand_like(v)).clamp_(-127,127)
    t_qe=bench(qe)
    # bandwidth bound for the two passes: (1.125 read + 2 write) + (2 read + 1.125 write)
    bw=n*6.25/2**30/(PEAK/1000)
    print(f"{f'C{C} {H}x{W}':>16} {t_deq:10.4f} {t_qk:10.4f} {t_qe:10.4f} {bw:9.4f}")
    for k,v in (("deq_eager",t_deq),("quant_kernel",t_qk),("quant_eager",t_qe),("bw_bound",bw)):
        tot[k]+=v*f
    tot["ohat_MB"]+=n*2/2**20
    del oh,q,sc; torch.cuda.empty_cache()
print(f"\nfreq-weighted totals (ms/step):")
print(f"  dequant, eager           {tot['deq_eager']:.2f}")
print(f"  quantize, real kernel    {tot['quant_kernel']:.2f}   (no SR yet)")
print(f"  quantize, eager + SR     {tot['quant_eager']:.2f}")
print(f"  --> eager pair           {tot['deq_eager']+tot['quant_eager']:.2f}")
print(f"  --> best case (2 fused kernels at the bandwidth bound)  {tot['bw_bound']:.2f}")
print(f"\no_hat total at batch 128: {tot['ohat_MB']:.0f} MB fp16")
