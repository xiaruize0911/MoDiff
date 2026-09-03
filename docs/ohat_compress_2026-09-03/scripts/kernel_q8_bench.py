"""conv2d_int8_evt_o_hat_q8 (int8 o_hat) vs conv2d_int8_evt_o_hat (fp16), frequency-weighted over
the 20 UNet conv shapes at batch 128. The int8 o_hat moves 1+1 B/elem instead of 2+2."""
import os,sys,json,statistics
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, modiff_cutlass as mc
DEV,CL="cuda",torch.channels_last
WU,R=6,20
SHAPES=[(s["C"],s["H"],s["W"],s["B"],s["freq"],s.get("N",s["C"])) for s in
        json.load(open("docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json"))["unet"]]
def bench(fn):
    for _ in range(WU): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(R):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)
tot={"fp16":0.,"q8":0.,"q8r":0.}; peak={"fp16":0.,"q8":0.}
print(f"{'C->K':>12} {'HxW':>7} {'fp16 ms':>9} {'q8 ms':>9} {'q8r ms':>9} {'fp16/q8r':>8}")
for C,H,W,B,f,K in SHAPES:
    x=torch.randint(-127,128,(B,C,H,W),device=DEV,dtype=torch.int8).contiguous(memory_format=CL)
    w=torch.randint(-127,128,(K,3,3,C),device=DEV,dtype=torch.int8).contiguous()
    alpha=torch.tensor([1/64.],device=DEV,dtype=torch.float32)
    ws=(torch.rand(K,device=DEV)*0.01+0.002).float().contiguous()
    oh=torch.empty(B,K,H,W,device=DEV,dtype=torch.float16,memory_format=CL).normal_()
    q=torch.randint(-127,128,(B,K,H,W),device=DEV,dtype=torch.int8).contiguous(memory_format=CL)
    sr=(torch.rand(K,device=DEV)*0.05+0.01).float().contiguous(); si=(1.0/sr).contiguous()
    t16=bench(lambda: mc.conv2d_int8_evt_o_hat(x,w,alpha,ws,oh,1,1,1,1,1,1))
    t8 =bench(lambda: mc.conv2d_int8_evt_o_hat_q8(x,w,alpha,ws,q,sr,si,1,1,1,1,1,1))
    am=torch.zeros(K,device=DEV,dtype=torch.float32)
    def _q8r():
        am.zero_(); mc.conv2d_int8_evt_o_hat_q8r(x,w,alpha,ws,q,sr,si,am,1,1,1,1,1,1)
    t8r=bench(_q8r)
    mb16=oh.numel()*2/2**20; mb8=q.numel()/2**20+K*8/2**20
    print(f"{f'{C}->{K}':>12} {f'{H}x{W}':>7} {t16:9.4f} {t8:9.4f} {t8r:9.4f} {t16/t8r:7.3f}x")
    tot["fp16"]+=t16*f; tot["q8"]+=t8*f; tot["q8r"]+=t8r*f; peak["fp16"]+=mb16*f; peak["q8"]+=mb8*f
    del x,w,oh,q; torch.cuda.empty_cache()
print(f"\nfreq-weighted (ms):  fp16 {tot['fp16']:.3f}   q8 {tot['q8']:.3f} ({tot['fp16']/tot['q8']:.3f}x)   "
      f"q8r {tot['q8r']:.3f} ({tot['fp16']/tot['q8r']:.3f}x)   reduction costs {tot['q8r']/tot['q8']:.3f}x")
print(f"o_hat bytes (freq-weighted MB): fp16 {peak['fp16']:.0f}  int8 {peak['q8']:.0f}  "
      f"ratio {peak['fp16']/peak['q8']:.2f}x")
