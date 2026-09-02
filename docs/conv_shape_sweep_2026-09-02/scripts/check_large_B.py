import torch, modiff_cutlass as MC
torch.manual_seed(0); dev='cuda'
print("BLK>64: blockwise vs exact fp32 per-block reference, and vs scalar control when all scales equal")
for blk in (64,128,256):
    for (N,C,H,W,K,R,st,pd) in [(2,256,16,16,128,3,1,1),(1,512,8,8,256,3,1,1),(2,256,16,16,128,1,1,0)]:
        if C % blk: continue
        x=torch.randn(N,C,H,W,device=dev,dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        q,sb=MC.conv_quantize_block_nhwc(x,blk)
        wq=torch.randint(-127,127,(K,R,R,C),device=dev,dtype=torch.int8).contiguous()
        ws=(torch.rand(K,device=dev)*0.01+0.001).float()
        got=MC.conv2d_int8_blockk(q,wq,ws,sb,0.0,blk,st,pd,None,None)
        # exact fp32 reference: dequantize A blockwise, dequantize W, fp32 conv
        t=q.permute(0,2,3,1).reshape(N,H,W,C//blk,blk).float()
        a=(t*sb.reshape(N,H,W,C//blk,1)).reshape(N,H,W,C).permute(0,3,1,2).contiguous()
        wf=(wq.permute(0,3,1,2).float()*ws.reshape(K,1,1,1))
        ref=torch.nn.functional.conv2d(a,wf,None,st,pd)
        rel=((got.float()-ref).norm()/ref.norm()).item()
        floor=((ref.half().float()-ref).norm()/ref.norm()).item()
        # equal-scale degeneration must reproduce the scalar control bitwise
        eq=torch.full_like(sb, 0.004)
        bw=MC.conv2d_int8_blockk(q,wq,ws,eq,0.0,blk,st,pd,None,None)
        ct=MC.conv2d_int8_blockk(q,wq,ws,torch.empty(0,device=dev,dtype=torch.float32),0.004,blk,st,pd,None,None)
        bit=(bw==ct).float().mean().item()
        print(f"  blk{blk:3d} C{C} {H}x{W} R{R}: relL2 {rel:.3e} (fp16 floor {floor:.3e})  equal-scale==ctrl {bit*100:7.3f}%")
