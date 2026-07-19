import torch, math
# exp-floor microbench: BH*T*T exponentials at level 0 (b128), precision-independent SFU EX2
def bench(fn,it=50,warm=20,reps=5):
    ts=[]
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e)/it*1e3)
    ts.sort(); return ts[len(ts)//2]
dev="cuda"
BH,T=1024,1024
N=BH*T*T
print(f"level-0 score elements = BH*T*T = {N:,} ({N/1e9:.2f}e9)")
# fp16 exp over the full [BH,T,T] tensor (what softmax does): time exp + a reduction proxy
x=torch.randn(BH,T,T,device=dev,dtype=torch.float16)
t_exp=bench(lambda: torch.exp(x))
# full softmax (max + exp + sum + div) as the true floor
t_sm=bench(lambda: torch.softmax(x,dim=-1))
BW=696e9
mem_floor=(N*2*2)/BW*1e6  # read+write fp16
print(f"raw exp() over [BH,T,T] fp16 : {t_exp:7.1f} us   (mem floor read+write {mem_floor:.0f} us)")
print(f"full softmax(dim=-1) fp16    : {t_sm:7.1f} us")
print(f"\nfp16 flash bar = 1809 us. exp floor {'<' if t_exp<1809 else '>='} bar -> {'flash can win' if t_exp<1809 else 'GATE FAIL'}")
