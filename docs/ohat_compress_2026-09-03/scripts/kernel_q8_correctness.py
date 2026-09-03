"""Correctness gate for conv2d_intX_evt_o_hat_q8:
   codes_new = sat_i8( (codes*s_read + conv_i32*alpha*ws[k]) * s_write_inv )
The int8 conv accumulates in int32 (exact), so an fp32 torch reference should match to the float
multiply order only."""
import os,sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, torch.nn.functional as F, modiff_cutlass as mc
DEV,CL="cuda",torch.channels_last
torch.manual_seed(0)
print(f"{'N':>3} {'C':>5} {'K':>5} {'HxW':>7} {'mismatch':>10} {'max|dcode|':>11} {'dequant relL2':>13}")
for (N,C,K,H,W) in [(2,192,192,16,16),(2,384,384,8,8),(1,768,768,4,4),(4,576,384,16,16),(2,1536,768,4,4)]:
    x=torch.randint(-127,128,(N,C,H,W),device=DEV,dtype=torch.int8).contiguous(memory_format=CL)
    w=torch.randint(-127,128,(K,3,3,C),device=DEV,dtype=torch.int8).contiguous()
    alpha=torch.tensor([1/64.],device=DEV,dtype=torch.float32)
    ws=(torch.rand(K,device=DEV)*0.01+0.002).float().contiguous()
    codes=torch.randint(-127,128,(N,K,H,W),device=DEV,dtype=torch.int8).contiguous(memory_format=CL)
    s_read=(torch.rand(K,device=DEV)*0.05+0.01).float().contiguous()
    s_wr  =(torch.rand(K,device=DEV)*0.05+0.01).float().contiguous()
    s_winv=(1.0/s_wr).contiguous()
    # reference in fp32
    conv=F.conv2d(x.float(), w.permute(0,3,1,2).float(), None, 1, 1)          # [N,K,H,W]
    o_new=codes.float()*s_read.view(1,K,1,1) + conv*float(alpha)*ws.view(1,K,1,1)
    ref=torch.clamp(torch.round(o_new*s_winv.view(1,K,1,1)), -128, 127)
    got=mc.conv2d_int8_evt_o_hat_q8(x,w,alpha,ws,codes.clone(),s_read,s_winv,1,1,1,1,1,1)
    g=got.float()
    mm=(g!=ref).float().mean().item(); mx=(g-ref).abs().max().item()
    # what the next step would read back
    dq_ref=ref*s_wr.view(1,K,1,1); dq_got=g*s_wr.view(1,K,1,1)
    rel=((dq_got-dq_ref).norm()/dq_ref.norm()).item()
    flag="OK" if mm<2e-3 and mx<=1 else "**FAIL**"
    print(f"{N:>3} {C:>5} {K:>5} {f'{H}x{W}':>7} {mm:10.2e} {mx:11.1f} {rel:13.2e}  {flag}")
