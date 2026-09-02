import torch, modiff_cutlass as MC
torch.manual_seed(0); dev='cuda'

print("=== conv_quantize_block_nhwc vs torch reference ===")
for (N,C,H,W) in [(2,192,32,32),(3,384,16,16),(1,576,8,8),(2,128,7,5)]:
    for blk in (32,64):
        if C % blk: continue
        x=(torch.randn(N,C,H,W,device=dev,dtype=torch.float16)*3).contiguous(memory_format=torch.channels_last)
        q,s=MC.conv_quantize_block_nhwc(x,blk)
        t=x.permute(0,2,3,1).float().reshape(N,H,W,C//blk,blk)
        amax=t.abs().amax(-1).clamp_min(1e-12)
        sref=amax/127.0
        qref=(t/sref.unsqueeze(-1)).round().clamp(-127,127)
        qg=q.permute(0,2,3,1).reshape(N,H,W,C//blk,blk).float()
        bit=(qg==qref).float().mean().item()
        ds=(s-sref).abs().max().item()/max(sref.max().item(),1e-12)
        print(f"  N{N} C{C} {H}x{W} blk{blk}: codes bit-identical {bit*100:.3f}%  scale relerr {ds:.2e}")

print("=== ACCUM epilogue: o_hat += A(Q(d)) ===")
for (N,C,H,W,K,R,st,pd) in [(2,192,32,32,192,3,1,1),(2,384,16,16,384,1,1,0),(1,576,8,8,576,3,2,1),(2,128,9,7,64,3,1,1)]:
    blk=64
    x=(torch.randn(N,C,H,W,device=dev,dtype=torch.float16)).contiguous(memory_format=torch.channels_last)
    q,sb=MC.conv_quantize_block_nhwc(x,blk)
    wq=torch.randint(-127,127,(K,R,R,C),device=dev,dtype=torch.int8).contiguous()
    ws=(torch.rand(K,device=dev)*0.01+0.001).float()
    empty=torch.empty(0,device=dev,dtype=torch.float32)
    # fresh-output reference
    fresh=MC.conv2d_int8_blockk(q,wq,ws,sb,0.0,blk,st,pd,None,None)
    # accumulate into a seeded o_hat
    seed=(torch.randn(*fresh.shape,device=dev,dtype=torch.float16)*0.5).contiguous(memory_format=torch.channels_last)
    oh=seed.clone()
    got=MC.conv2d_int8_blockk(q,wq,ws,sb,0.0,blk,st,pd,None,oh)
    exp=(seed.float()+fresh.float()).half()
    rel=((got.float()-exp.float()).norm()/exp.float().norm()).item()
    same=(got==exp).float().mean().item()
    inplace = got.data_ptr()==oh.data_ptr()
    print(f"  C{C} {H}x{W} K{K} R{R} s{st}: relL2 {rel:.3e}  bit-identical {same*100:.3f}%  in-place {inplace}")
