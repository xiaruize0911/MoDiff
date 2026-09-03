"""Where does a_hat-blockwise peak memory actually go, per block size?

Reports, after one warmup sample: currently-allocated total, the summed a_hat cache bytes,
the summed block-scale bytes, and the peak during a second sample. argv: int8|int4, block.
"""
import os, sys, json
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers")]
PREC, BLK = sys.argv[1], sys.argv[2]
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
    "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0",
    "MODIFF_DELTA_MODE":"static","MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0",
    "MODIFF_AHAT_BLOCK":BLK})
import torch
import integration.benchmarks.benchmark_ldm as B
SHAPE,BATCH,STEPS=(4,32,32),128,50
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="docs/ahat_only_conv_2026-09-02/tmp",
    batch_size=BATCH, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path(PREC), auto_delta_table=True)
m,s=r._setup_model(PREC)
def once():
    (B.reset_modiff_state_int8 if PREC=="int8" else B.reset_modiff_state_int4)(m.model.diffusion_model)
    B._reset_wxax_modiff_safe(m)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        s.sample(S=STEPS,batch_size=BATCH,shape=SHAPE,eta=0.0,verbose=False)
once(); torch.cuda.synchronize()
ah=sc=oh=0; nlay=0; dts={}
for mod in m.model.diffusion_model.modules():
    a=getattr(mod,"a_hat_cache",None)
    if a is not None:
        nlay+=1; ah+=a.numel()*a.element_size()
        dts[str(a.dtype)]=dts.get(str(a.dtype),0)+1
        q=getattr(mod,"_ahat_qscale",None)
        if q is not None: sc+=q.numel()*q.element_size()
        o=getattr(mod,"o_hat_cache",None)
        if o is not None: oh+=o.numel()*o.element_size()
alloc=torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats(); once(); torch.cuda.synchronize()
print("MEMJSON:"+json.dumps({"prec":PREC,"block":int(BLK),"layers":nlay,"ahat_dtypes":dts,
  "ahat_MB":ah/2**20,"scale_MB":sc/2**20,"ohat_MB":oh/2**20,
  "allocated_after_warmup_MB":alloc/2**20,
  "peak_during_sample_MB":torch.cuda.max_memory_allocated()/2**20}))
