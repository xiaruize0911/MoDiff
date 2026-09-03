"""(a) exact a_hat vs o_hat cache bytes over every conv layer; (b) the accumulator ratio
   r_t = ||Delta_t|| / ||o_hat_t||, which decides whether low-precision o_hat is possible at all."""
import os,sys,json,collections
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
 "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0","MODIFF_DELTA_MODE":"static",
 "MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0","MODIFF_AHAT_BLOCK":"32"})
ROOT="/workspace/MoDiff"; sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
import torch, integration.benchmarks.benchmark_ldm as B
MODE="int8"; BATCH,STEPS=32,8
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/oh",
 batch_size=BATCH, steps=STEPS, shape=(4,32,32),
 calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE)
unet=m.model.diffusion_model
B.reset_modiff_state_int8(unet)
convs=[(n,mod) for n,mod in unet.named_modules() if hasattr(mod,"o_hat_cache")]
print(f"{len(convs)} modules with an o_hat_cache")

# (b) snapshot o_hat before/after each conv forward on a subset of layers
SUB=set(range(0,len(convs),max(1,len(convs)//8)))
ratio=collections.defaultdict(list); hooks=[]
prev={}
def mk(i,n,mod):
    def pre(_m,_inp):
        oh=mod.o_hat_cache
        prev[i]=None if oh is None else oh.detach().float().norm().item()
    def post(_m,_inp,out):
        oh=mod.o_hat_cache
        if oh is None or prev.get(i) is None: return
        # Delta_t norm via the increment: ||o_hat_t - o_hat_{t-1}|| needs the old tensor, so use
        # the identity ||out|| for o_hat_t and the stored pre-norm for o_hat_{t-1}; record both
        ratio[(i,n)].append((prev[i], oh.detach().float().norm().item()))
    return pre,post
for i,(n,mod) in enumerate(convs):
    if i in SUB:
        pre,post=mk(i,n,mod)
        hooks.append(mod.register_forward_pre_hook(pre)); hooks.append(mod.register_forward_hook(post))

B._reset_wxax_modiff_safe(m); torch.manual_seed(1234)
with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
    s.sample(S=STEPS,batch_size=BATCH,shape=(4,32,32),eta=0.0,verbose=False)
for h in hooks: h.remove()

tot_a=tot_o=0; rows=[]
for n,mod in convs:
    a=getattr(mod,"a_hat_cache",None); o=getattr(mod,"o_hat_cache",None)
    qs=getattr(mod,"_ahat_block_qscale",None) or getattr(mod,"ahat_qscale",None)
    ab=0 if a is None else a.numel()*a.element_size()
    if qs is not None and torch.is_tensor(qs): ab+=qs.numel()*qs.element_size()
    ob=0 if o is None else o.numel()*o.element_size()
    tot_a+=ab; tot_o+=ob
    rows.append({"name":n,"a_MB":ab/2**20,"o_MB":ob/2**20,
                 "a_shape":None if a is None else list(a.shape),
                 "o_shape":None if o is None else list(o.shape),
                 "o_dtype":None if o is None else str(o.dtype)})
print(f"\nbatch={BATCH}  a_hat total {tot_a/2**20:.1f} MB   o_hat total {tot_o/2**20:.1f} MB"
      f"   o/a = {tot_o/max(tot_a,1):.2f}x")
print(f"scaled to batch 128:  a_hat {tot_a/2**20*4:.0f} MB   o_hat {tot_o/2**20*4:.0f} MB")
print("\naccumulator ratio  ||o_hat_(t-1)|| -> ||o_hat_t||  per step, subset of layers:")
for (i,n),v in ratio.items():
    seq=" ".join(f"{b:.3g}" for _,b in v[:8])
    print(f"  [{i:2d}] {n[-46:]:46s} {seq}")
json.dump({"rows":rows,"tot_a_MB":tot_a/2**20,"tot_o_MB":tot_o/2**20,
           "ratio":{f"{i}|{n}":v for (i,n),v in ratio.items()},"batch":BATCH},
          open("/tmp/oh/probe.json","w"),indent=1)
