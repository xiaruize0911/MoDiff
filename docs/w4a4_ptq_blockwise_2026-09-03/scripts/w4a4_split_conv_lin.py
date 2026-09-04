"""Can W4A4 PTQ survive with BLOCKWISE activations? Pure-accuracy sim: run in fp16 mode and
fake-quantize every 3x3 conv's weight and input, with configurable activation granularity and
SmoothQuant-style migration. Isolates the conv activation-granularity axis, which
docs/wa_budget_2026-09-02 measured at 12.5x (A4 per-tensor 0.5181 vs blockwise B=64 0.0415) --
larger than migration's 2.40x or SVDQuant's low-rank 1.035x, and never tested end to end.

argv: TAG  ACTMODE(pt|blk)  BLK  MIGRATE(0|1)   [BITS_W BITS_A]
"""
import os,sys,json,statistics
ROOT="/workspace/MoDiff"; sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]; os.chdir(ROOT)
import torch, torch.nn.functional as F, integration.benchmarks.benchmark_ldm as B
TAG,ACT,BLK,MIG = sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]=="1"
WHICH = sys.argv[7] if len(sys.argv)>7 else "both"   # conv | lin | both
WB = int(sys.argv[5]) if len(sys.argv)>5 else 4
AB = int(sys.argv[6]) if len(sys.argv)>6 else 4
BATCH,STEPS=32,50
AMAX=torch.load("/tmp/act_amax.pt",map_location="cpu",weights_only=False)
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/oh",
 batch_size=BATCH, steps=STEPS, shape=(4,32,32), calibration_path=None, auto_delta_table=False)
m,s=r._setup_model("fp16"); unet=m.model.diffusion_model
def qw(w,bits,lam):
    if lam is not None: w = w*lam.view(1,-1,1,1)
    lim=2**(bits-1)-1
    sc=w.abs().amax(dim=(1,2,3),keepdim=True).clamp_min(1e-12)/lim
    return torch.clamp(torch.round(w/sc),-lim,lim)*sc
hooks=[]; n_hook=0; n_blk=0
n_lin=0
for name,mod in unet.named_modules():
    is_c = isinstance(mod,torch.nn.Conv2d) and mod.kernel_size==(3,3)
    is_l = isinstance(mod,torch.nn.Linear)
    if not (is_c or is_l): continue
    if is_c and WHICH not in ("conv","both"): continue
    if is_l and WHICH not in ("lin","both"): continue
    if is_c and name not in AMAX: continue
    if is_l:
        # Linear: [out, in]. Weight per-output-row, activation per-tensor or blockwise along `in`.
        C=mod.in_features
        with torch.no_grad():
            w=mod.weight.detach().float(); lim=2**(WB-1)-1
            sc=w.abs().amax(dim=1,keepdim=True).clamp_min(1e-12)/lim
            mod.weight.copy_((torch.clamp(torch.round(w/sc),-lim,lim)*sc).to(mod.weight.dtype))
        def mkl(C):
            def pre(_m,inp):
                x=inp[0].float(); lim=2**(AB-1)-1
                if ACT=="blk" and C % BLK == 0:
                    sh=x.shape; v=x.reshape(-1,C//BLK,BLK)
                    q=v.abs().amax(-1,keepdim=True).clamp_min(1e-12)/lim
                    x=(torch.clamp(torch.round(v/q),-lim,lim)*q).reshape(sh)
                else:
                    q=x.abs().amax().clamp_min(1e-12)/lim
                    x=torch.clamp(torch.round(x/q),-lim,lim)*q
                return (x.to(inp[0].dtype),)+inp[1:]
            return pre
        hooks.append(mod.register_forward_pre_hook(mkl(C))); n_lin+=1; continue
    C=mod.in_channels
    lam=None
    if MIG:
        ax=AMAX[name].to("cuda").clamp_min(1e-8)
        aw=mod.weight.detach().abs().amax(dim=(0,2,3)).float().clamp_min(1e-8)
        lam=(ax.sqrt()/aw.sqrt()).clamp(1e-3,1e3)
    with torch.no_grad():
        mod.weight.copy_(qw(mod.weight.detach().float(),WB,lam).to(mod.weight.dtype))
    blk_ok = (ACT=="blk" and C % BLK == 0)
    if blk_ok: n_blk+=1
    def mk(lam,blk_ok,C):
        def pre(_m,inp):
            x=inp[0].float()
            if lam is not None: x=x/lam.view(1,C,1,1)
            lim=2**(AB-1)-1
            if blk_ok:
                n,c,h,w=x.shape
                v=x.permute(0,2,3,1).reshape(n,h,w,c//BLK,BLK)
                sc=v.abs().amax(-1,keepdim=True).clamp_min(1e-12)/lim
                v=torch.clamp(torch.round(v/sc),-lim,lim)*sc
                x=v.reshape(n,h,w,c).permute(0,3,1,2)
            else:
                sc=x.abs().amax().clamp_min(1e-12)/lim
                x=torch.clamp(torch.round(x/sc),-lim,lim)*sc
            return (x.to(inp[0].dtype),)+inp[1:]
        return pre
    hooks.append(mod.register_forward_pre_hook(mk(lam,blk_ok,C))); n_hook+=1
print(f"hooked {n_hook} convs (blockwise {n_blk}) + {n_lin} linears, which={WHICH}",flush=True)
torch.manual_seed(1234)
with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
    out=s.sample(S=STEPS,batch_size=BATCH,shape=(4,32,32),eta=0.0,verbose=False)
lat=(out[0] if isinstance(out,(tuple,list)) else out).float().cpu()
torch.save(lat,f"/tmp/w4_lat_{TAG}.pt")
print("W4JSON:"+json.dumps({"tag":TAG,"act":ACT,"blk":BLK,"migrate":MIG,"wb":WB,"ab":AB,
 "hooked":n_hook,"blockwise":n_blk,"linears":n_lin,"which":WHICH,"finite":bool(torch.isfinite(lat).all()),
 "absmax":float(lat.abs().max()),"std":float(lat.std()),"mean":float(lat.mean())}))
