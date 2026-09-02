import torch, statistics, modiff_cutlass as MC
dev='cuda'; torch.manual_seed(0)
def pack(q):
    lo=q[...,0::2].to(torch.int16)&0xF; hi=q[...,1::2].to(torch.int16)&0xF
    return ((hi<<4)|lo).to(torch.uint8).view(torch.int8).contiguous()
def bench(fn,n=50,w=10):
    for _ in range(w): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        s,e=torch.cuda.Event(True),torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
print(f"{'shape':26s} {'CUTLASS':>8s} {'ctrl':>8s} {'B=64':>8s} {'B=128':>9s} {'B=256':>9s}   (ms, b128)")
for (N,C,H,W,K,R,st,pd) in [(128,384,16,16,384,3,1,1),(128,768,8,8,768,3,1,1),(128,384,32,32,384,3,1,1)]:
    a=torch.randint(-7,8,(N,H,W,C),device=dev,dtype=torch.int8); xp=pack(a)
    w=torch.randint(-7,8,(K,R,R,C),device=dev,dtype=torch.int8); wp=pack(w)
    ws=(torch.rand(K,device=dev)*0.01+0.001).float()
    em=torch.empty(0,device=dev,dtype=torch.float32)
    try:
        sc=torch.full((1,),0.01,device=dev)
        eb=torch.empty(0,device=dev,dtype=torch.float16)
        cut=bench(lambda: MC.conv2d_int4_fprop(xp,wp,sc,eb,st,st,pd,pd,1,1))
    except Exception as ex:
        cut=float('nan')
    row=[cut, bench(lambda: MC.conv2d_int4_blockk(xp,wp,ws,em,0.01,64,st,pd,None,None,None))]
    for blk in (64,128,256):
        if C%blk: row.append(float('nan')); continue
        sb=(torch.rand(N,H,W,C//blk,device=dev)*0.02+0.005).float()
        row.append(bench(lambda: MC.conv2d_int4_blockk(xp,wp,ws,sb,0.0,blk,st,pd,None,None,None)))
    print(f"C{C} {H}x{W} K{K} R{R}        " + " ".join(f"{v:8.3f}" for v in row))
    print(f"{'  vs CUTLASS':26s} " + " ".join(f"{v/row[0]:7.2f}x" for v in row))
    print(f"{'  vs ctrl':26s} {'':8s} " + " ".join(f"{v/row[1]:7.2f}x" for v in row[1:]))
