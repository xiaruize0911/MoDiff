import torch, torch.nn.functional as F, modiff_cutlass as MC
torch.manual_seed(0); dev='cuda'
N,C,H,W,G=2,192,16,16,32
w=torch.randn(C,device=dev,dtype=torch.float16); b=torch.randn(C,device=dev,dtype=torch.float16)
em=torch.empty(0,device=dev,dtype=torch.float32); eh=torch.empty(0,device=dev,dtype=torch.float16)
def gn(x): return F.silu(F.group_norm(x.float(),G,w.float(),b.float(),1e-6)).half().contiguous(memory_format=torch.channels_last)
def deq(q,s):
    t=q.permute(0,2,3,1).reshape(N,H,W,C//32,32).float()
    return (t*s.reshape(N,H,W,C//32,1)).reshape(N,H,W,C).permute(0,3,1,2).contiguous(memory_format=torch.channels_last)
ah=torch.zeros(N,C,H,W,device=dev,dtype=torch.float16).contiguous(memory_format=torch.channels_last)
ah_ref=ah.clone().float()
print("step | delta-code match | a_hat relL2 (kernel vs reference recursion)")
for step in range(4):
    x=torch.randn(N,C,H,W,device=dev,dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    g=gn(x)
    # reference: delta against the reference a_hat, blockwise quantize, accumulate
    d_ref=(g.float()-ah_ref).half().contiguous(memory_format=torch.channels_last)
    q_ref,s_ref=MC.conv_quantize_block_nhwc(d_ref,32)
    ah_ref = ah_ref + deq(q_ref,s_ref).float()
    # kernel
    q,s=MC.gn_silu_blockk_quantize_b32(x,w,b,ah,G,1e-6,True,em,eh,eh)
    m=(q==q_ref).float().mean().item()
    r=((ah.float()-ah_ref).norm()/ah_ref.norm()).item()
    print(f"  {step}  | {m*100:8.3f}%        | {r:.4e}")
