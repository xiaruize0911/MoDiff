import os,sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch, torch.nn.functional as F, modiff_cutlass as mc
DEV,CL="cuda",torch.channels_last
torch.manual_seed(0)
print(f"{'C->K':>10} {'HxW':>7} {'code mism':>10} {'q8r vs q8':>10} {'amax relL2':>11} {'amax maxrel':>12}")
for (N,C,K,H,W) in [(2,192,192,16,16),(2,384,384,8,8),(4,576,384,16,16),(2,1536,768,4,4),(1,768,768,8,8)]:
    x=torch.randint(-127,128,(N,C,H,W),device=DEV,dtype=torch.int8).contiguous(memory_format=CL)
    w=torch.randint(-127,128,(K,3,3,C),device=DEV,dtype=torch.int8).contiguous()
    alpha=torch.tensor([1/64.],device=DEV,dtype=torch.float32)
    ws=(torch.rand(K,device=DEV)*0.01+0.002).float().contiguous()
    codes=torch.randint(-127,128,(N,K,H,W),device=DEV,dtype=torch.int8).contiguous(memory_format=CL)
    sr=(torch.rand(K,device=DEV)*0.05+0.01).float().contiguous()
    swi=(1.0/(torch.rand(K,device=DEV)*0.05+0.01)).float().contiguous()
    conv=F.conv2d(x.float(), w.permute(0,3,1,2).float(), None,1,1)
    o_new=codes.float()*sr.view(1,K,1,1) + conv*float(alpha)*ws.view(1,K,1,1)
    ref=torch.clamp(torch.round(o_new*swi.view(1,K,1,1)),-128,127)
    amax_ref=o_new.abs().amax(dim=(0,2,3))
    a=torch.zeros(K,device=DEV,dtype=torch.float32)
    g_r=mc.conv2d_int8_evt_o_hat_q8r(x,w,alpha,ws,codes.clone(),sr,swi,a,1,1,1,1,1,1).float()
    g_q=mc.conv2d_int8_evt_o_hat_q8 (x,w,alpha,ws,codes.clone(),sr,swi,  1,1,1,1,1,1).float()
    mm=(g_r!=ref).float().mean().item(); dq=(g_r!=g_q).float().mean().item()
    rel=((a-amax_ref).norm()/amax_ref.norm()).item()
    mxr=((a-amax_ref).abs()/amax_ref.clamp_min(1e-9)).max().item()
    flag="OK" if mm==0 and dq==0 and mxr<1e-4 else "**FAIL**"
    print(f"{f'{C}->{K}':>10} {f'{H}x{W}':>7} {mm:10.2e} {dq:10.2e} {rel:11.2e} {mxr:12.2e}  {flag}")
