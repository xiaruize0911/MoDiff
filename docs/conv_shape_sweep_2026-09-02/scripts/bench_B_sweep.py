import torch, modiff_cutlass as MC, statistics
dev='cuda'; torch.manual_seed(0)
def bench(fn,n=50,w=10):
    for _ in range(w): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        s,e=torch.cuda.Event(True),torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
print(f"{'shape':22s} {'CUTLASS':>9s} {'ctrl':>9s} {'B=64':>9s} {'B=128':>9s} {'B=256':>9s}")
for (N,C,H,W,K,R,st,pd) in [(128,384,16,16,384,3,1,1),(128,768,8,8,768,3,1,1),(128,384,32,32,384,3,1,1)]:
    x=torch.randn(N,C,H,W,device=dev,dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    wq=torch.randint(-127,127,(K,R,R,C),device=dev,dtype=torch.int8).contiguous()
    ws=(torch.rand(K,device=dev)*0.01+0.001).float()
    em=torch.empty(0,device=dev,dtype=torch.float32)
    row=[]
    # shipped CUTLASS per-tensor conv for reference
    q8,_=MC.conv_quantize_block_nhwc(x,64)
    try:
        t_cut=bench(lambda: MC.conv2d_int8_fprop(q8,wq,ws,0.01,st,pd))
    except Exception:
        t_cut=float('nan')
    row.append(t_cut)
    row.append(bench(lambda: MC.conv2d_int8_blockk(q8,wq,ws,em,0.01,64,st,pd,None,None)))
    for blk in (64,128,256):
        if C % blk: row.append(float('nan')); continue
        q,sb=MC.conv_quantize_block_nhwc(x,blk)
        row.append(bench(lambda: MC.conv2d_int8_blockk(q,wq,ws,sb,0.0,blk,st,pd,None,None)))
    print(f"C{C} {H}x{W} K{K} R{R}     " + " ".join(f"{v:9.3f}" for v in row))
