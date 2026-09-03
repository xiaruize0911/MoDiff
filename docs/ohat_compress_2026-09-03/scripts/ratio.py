"""r_t = ||o_hat_t - o_hat_(t-1)|| / ||o_hat_t||  -- the increment-to-accumulator ratio.
Decides whether a low-precision o_hat can represent the update at all: if r_t << 2^-bits the
increment is swallowed by the quantization step. Snapshotted per DDIM step (forward hooks do not
fire -- the fused ResBlock calls forward_gn_fused_modiff directly)."""
import os,sys,json
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
 "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0","MODIFF_DELTA_MODE":"static",
 "MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0","MODIFF_AHAT_BLOCK":"32"})
ROOT="/workspace/MoDiff"; sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
import torch, integration.benchmarks.benchmark_ldm as B
MODE="int8"; BATCH,STEPS=8,50
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/oh",
 batch_size=BATCH, steps=STEPS, shape=(4,32,32),
 calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE); unet=m.model.diffusion_model
B.reset_modiff_state_int8(unet)
convs=[(n,mod) for n,mod in unet.named_modules()
       if hasattr(mod,"o_hat_cache") and hasattr(mod,"a_hat_cache")]
# instrument the UNet forward so we snapshot once per model call
orig=unet.forward; state={"t":0,"prev":{},"rec":{}}
SUB=[c for i,c in enumerate(convs) if i % max(1,len(convs)//10)==0][:10]
def fwd(*a,**k):
    out=orig(*a,**k)
    for n,mod in SUB:
        oh=mod.o_hat_cache
        if oh is None: continue
        cur=oh.detach().float()
        p=state["prev"].get(n)
        if p is not None:
            d=(cur-p).norm().item(); mag=cur.norm().item()
            amax=cur.abs().max().item()
            state["rec"].setdefault(n,[]).append((d,mag,amax))
        state["prev"][n]=cur.clone()
    state["t"]+=1
    return out
unet.forward=fwd
B._reset_wxax_modiff_safe(m); torch.manual_seed(1234)
with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
    s.sample(S=STEPS,batch_size=BATCH,shape=(4,32,32),eta=0.0,verbose=False)
unet.forward=orig

print(f"\n{'layer':46s} {'r med':>8} {'r min':>8} {'r@t=45':>8} {'amax/rms':>9} | bits needed")
out={}
for n,mod in SUB:
    v=state["rec"].get(n)
    if not v: continue
    rs=sorted(d/mg for d,mg,_ in v)
    med=rs[len(rs)//2]; mn=rs[0]
    late=v[min(len(v)-1,44)]; rl=late[0]/late[1]
    # crest factor of the accumulator: sets the quantization step for a given bit width
    crest=[a/(mg/ (mod.o_hat_cache.numel()**0.5)) for _,mg,a in v]
    cr=sorted(crest)[len(crest)//2]
    # bits so that the step delta = 2*amax/2^bits stays below the smallest increment element-wise
    import math
    need=math.log2(2*cr/mn) if mn>0 else float('inf')
    print(f"{n[-46:]:46s} {med:8.4f} {mn:8.4f} {rl:8.4f} {cr:9.2f} | {need:5.1f}")
    out[n]={"r_med":med,"r_min":mn,"r_late":rl,"crest":cr,"bits_need":need,
            "series":[(d,mg) for d,mg,_ in v]}
json.dump(out,open("/tmp/oh/ratio.json","w"),indent=1)
print("\nwrote /tmp/oh/ratio.json")
