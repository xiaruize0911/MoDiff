import os, sys; os.chdir("/workspace/MoDiff"); sys.path.insert(0,"/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
def bench(fn,it=200,warm=50,reps=5):
    ts=[]
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e)/it*1e3)
    ts.sort(); return ts[len(ts)//2]
# real b64 proj shapes: (b,nh,T,hd,C,count)
SH=[(64,8,1024,24,192,5),(64,8,256,48,384,5),(64,8,64,48,384,5),(64,8,16,96,768,5),(64,8,4,96,768,1)]
tot_un=tot_fu=0.0
print(f"{'C':>5}{'T':>6}{'cnt':>4} | {'copy+quant(un)':>15} {'fused_quant':>12} {'save/inst':>10}")
for (b,nh,T,hd,C,cnt) in SH:
    a=torch.randn(b,nh,T,hd,device="cuda",dtype=torch.float16)
    sc=a.abs().max().item()/127.0
    un=bench(lambda: mc.quantize_act_int8(a.transpose(1,2).reshape(b,T,C).contiguous(), sc))
    fu=bench(lambda: mc.quantize_attn_out_int8(a, sc))
    tot_un+=un*cnt; tot_fu+=fu*cnt
    print(f"{C:>5}{T:>6}{cnt:>4} | {un:>15.2f} {fu:>12.2f} {un-fu:>10.2f}  (us)")
print(f"\nper-step proj input-prep: unfused {tot_un:.1f}us -> fused {tot_fu:.1f}us  (saved {tot_un-tot_fu:.1f}us/step, {(1-tot_fu/tot_un)*100:.0f}% of the prep)")
