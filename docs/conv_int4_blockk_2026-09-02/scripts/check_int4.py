import torch, torch.nn.functional as F, modiff_cutlass as MC
torch.manual_seed(0); dev='cuda'

def pack(q):                       # q int8 codes in [-7,7], last dim = C -> [..., C/2]
    lo = q[..., 0::2].to(torch.int16) & 0xF
    hi = q[..., 1::2].to(torch.int16) & 0xF
    return ((hi << 4) | lo).to(torch.uint8).view(torch.int8).contiguous()

print("int4 blockk vs exact fp32 per-block reference")
print(f"{'case':34s} {'relL2':>10s} {'fp16 floor':>11s} {'eq-scale==ctrl':>15s}")
for blk in (64,128,256):
    for (N,C,H,W,K,R,st,pd) in [(2,384,16,16,384,3,1,1),(1,768,8,8,384,3,1,1),(2,384,16,16,384,1,1,0)]:
        if C % blk: continue
        # activations: int4 codes + per-(pixel, C-block) scales, NHWC
        a = torch.randint(-7,8,(N,H,W,C),device=dev,dtype=torch.int8)
        sb = (torch.rand(N,H,W,C//blk,device=dev)*0.02+0.005).float()
        xp = pack(a)                                            # [N,H,W,C/2]
        w = torch.randint(-7,8,(K,R,R,C),device=dev,dtype=torch.int8)
        wp = pack(w)                                            # [K,R,S,C/2]
        ws = (torch.rand(K,device=dev)*0.01+0.001).float()
        got = MC.conv2d_int4_blockk(xp,wp,ws,sb,0.0,blk,st,pd,None,None,None)
        # exact fp32 reference
        af = (a.float().reshape(N,H,W,C//blk,blk)*sb.unsqueeze(-1)).reshape(N,H,W,C)
        af = af.permute(0,3,1,2).contiguous()
        wf = (w.permute(0,3,1,2).float()*ws.reshape(K,1,1,1))
        ref = F.conv2d(af,wf,None,st,pd)
        rel=((got.float()-ref).norm()/ref.norm()).item()
        floor=((ref.half().float()-ref).norm()/ref.norm()).item()
        # equal scales must reproduce the scalar-alpha control bitwise
        eq=torch.full_like(sb,0.01)
        bw=MC.conv2d_int4_blockk(xp,wp,ws,eq,0.0,blk,st,pd,None,None,None)
        ct=MC.conv2d_int4_blockk(xp,wp,ws,torch.empty(0,device=dev,dtype=torch.float32),0.01,blk,st,pd,None,None,None)
        bit=(bw==ct).float().mean().item()
        print(f"  blk{blk:3d} C{C} {H}x{W} K{K} R{R}      {rel:10.3e} {floor:11.3e} {bit*100:14.3f}%")
