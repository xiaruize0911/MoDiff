"""Correctness gate for every config in conv2d_blockk_tune's table, int8 and int4.
A config that does not reproduce an exact fp32 per-block reference to the fp16 store floor is
excluded from the benchmark rather than reported as fast."""
import torch, torch.nn.functional as F, modiff_cutlass as MC
torch.manual_seed(0); dev='cuda'
def pack(q):
    lo=q[...,0::2].to(torch.int16)&0xF; hi=q[...,1::2].to(torch.int16)&0xF
    return ((hi<<4)|lo).to(torch.uint8).view(torch.int8).contiguous()
NC=MC.blockk_tune_num_cfgs()
print(f"{NC} configs\n")
good={}
for i4 in (False,True):
    tag="int4" if i4 else "int8"
    for cfg in range(NC):
        nm=MC.blockk_tune_cfg_name(cfg)
        okall=True; worst=0.0; why=""
        for (N,C,H,W,K) in [(2,768,16,16,128),(1,384,8,8,256)]:
            try:
                if i4:
                    a=torch.randint(-7,8,(N,H,W,C),device=dev,dtype=torch.int8); xp=pack(a)
                    w=torch.randint(-7,8,(K,3,3,C),device=dev,dtype=torch.int8); wp=pack(w)
                    aq=a.float()
                else:
                    a=torch.randint(-127,128,(N,C,H,W),device=dev,dtype=torch.int8).contiguous(memory_format=torch.channels_last)
                    xp=a; wp=torch.randint(-127,128,(K,3,3,C),device=dev,dtype=torch.int8).contiguous()
                    w=wp; aq=a.permute(0,2,3,1).float()
                ws=(torch.rand(K,device=dev)*0.01+0.001).float()
                blk=int(nm.split("_B")[1])
                if C%blk: continue
                sb=(torch.rand(N,H,W,C//blk,device=dev)*0.02+0.005).float()
                got=MC.conv2d_blockk_tune(xp,wp,ws,sb,0.0,cfg,i4,1,1)
                af=(aq.reshape(N,H,W,C//blk,blk)*sb.unsqueeze(-1)).reshape(N,H,W,C).permute(0,3,1,2).contiguous()
                ref=F.conv2d(af,(w.permute(0,3,1,2).float()*ws.reshape(K,1,1,1)),None,1,1)
                rel=((got.float()-ref).norm()/ref.norm()).item()
                fl=((ref.half().float()-ref).norm()/ref.norm()).item()
                worst=max(worst,rel)
                if rel > 5*fl: okall=False; why=f"relL2 {rel:.2e} vs floor {fl:.2e}"
            except Exception as ex:
                okall=False; why=str(ex).split("\n")[0][:70]
        good[(tag,cfg)]=okall
        print(f"  {'PASS' if okall else 'FAIL'}  {tag}  cfg{cfg:2d} {nm:30s} " +
              (f"relL2 {worst:.3e}" if okall else why))
import json
json.dump({f"{k[0]}_{k[1]}":v for k,v in good.items()},
          open("docs/conv_int4_blockk_2026-09-02/data/tune_valid.json","w"),indent=1)
