import torch, torch.nn.functional as F, modiff_cutlass as MC
torch.manual_seed(0); dev='cuda'
def ref_gn(x,w,b,G,eps,silu):
    y=F.group_norm(x.float(),G,w.float(),b.float(),eps)
    return F.silu(y) if silu else y
print("fused GN+blockwise-quantize  vs  (GN kernel) then conv_quantize_block_nhwc")
for (N,C,H,W,G) in [(2,192,32,32,32),(2,384,16,16,32),(1,576,8,8,32),(2,768,4,4,32)]:
    for silu in (True,False):
        x=torch.randn(N,C,H,W,device=dev,dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        w=torch.randn(C,device=dev,dtype=torch.float16); b=torch.randn(C,device=dev,dtype=torch.float16)
        em=torch.empty(0,device=dev,dtype=torch.float32); eh=torch.empty(0,device=dev,dtype=torch.float16)
        # BASELINE arm
        q,s=MC.gn_silu_blockk_quantize_b32(x,w,b,eh,G,1e-6,silu,em,eh,eh)
        gn=ref_gn(x,w,b,G,1e-6,silu).half().contiguous(memory_format=torch.channels_last)
        q2,s2=MC.conv_quantize_block_nhwc(gn,32)
        bit=(q==q2).float().mean().item(); sd=(s-s2).abs().max().item()/max(s2.max().item(),1e-12)
        print(f"  base C{C} {H}x{W} silu={int(silu)}: codes {bit*100:7.3f}%  scale relerr {sd:.2e}")
print()
print("MoDiff arm: delta codes + fused a_hat update")
for (N,C,H,W,G) in [(2,192,32,32,32),(2,576,8,8,32)]:
    x=torch.randn(N,C,H,W,device=dev,dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    w=torch.randn(C,device=dev,dtype=torch.float16); b=torch.randn(C,device=dev,dtype=torch.float16)
    ah=(torch.randn(N,C,H,W,device=dev,dtype=torch.float16)*0.3).contiguous(memory_format=torch.channels_last)
    em=torch.empty(0,device=dev,dtype=torch.float32); eh=torch.empty(0,device=dev,dtype=torch.float16)
    ah0=ah.clone()
    q,s=MC.gn_silu_blockk_quantize_b32(x,w,b,ah,G,1e-6,True,em,eh,eh)
    gn=ref_gn(x,w,b,G,1e-6,True).half().contiguous(memory_format=torch.channels_last)
    d=(gn.float()-ah0.float()).half().contiguous(memory_format=torch.channels_last)
    q2,s2=MC.conv_quantize_block_nhwc(d,32)
    bit=(q==q2).float().mean().item(); sd=(s-s2).abs().max().item()/max(s2.max().item(),1e-12)
    # a_hat update check
    t=q.permute(0,2,3,1).reshape(N,H,W,C//32,32).float()
    dq=(t*s.reshape(N,H,W,C//32,1)).reshape(N,H,W,C).permute(0,3,1,2)
    exp=(ah0.float()+dq).half()
    ar=((ah.float()-exp.float()).norm()/exp.float().norm()).item()
    print(f"  modiff C{C} {H}x{W}: delta codes {bit*100:7.3f}%  scale relerr {sd:.2e}  a_hat relL2 {ar:.2e}")
