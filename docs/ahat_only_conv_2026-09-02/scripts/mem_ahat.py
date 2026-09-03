"""Peak memory and actual a_hat cache footprint/traffic for fp16 vs blockwise-int8 a_hat.

The claim under test: a_hat at 1.125 B/elem (int8 codes + one fp32 scale per 32 channels)
against 2 B/elem for fp16 is a 1.78x smaller cache. But the blockwise kernels REINTERPRET the
fp16 buffer as int8 (`reinterpret_cast<int8_t*>(a_hat_cache)`), so if the allocation is still
fp16-sized the saving is traffic only, not capacity. This measures both.

argv: ahat0 | ahat32
"""
import os, sys, json
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
ARM=sys.argv[1]
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
    "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0",
    "MODIFF_DELTA_MODE":"static","MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0",
    "MODIFF_AHAT_BLOCK":"32" if ARM=="ahat32" else "0"})
import torch, integration.benchmarks.benchmark_ldm as B
SHAPE,BATCH,STEPS=(4,32,32),128,6
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="docs/ahat_only_conv_2026-09-02/tmp",
    batch_size=BATCH, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path("int8"), auto_delta_table=True)
m,s=r._setup_model("int8")
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
base=torch.cuda.memory_allocated()
B.reset_modiff_state_int8(m.model.diffusion_model); B._reset_wxax_modiff_safe(m)
with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
    s.sample(S=STEPS,batch_size=BATCH,shape=SHAPE,eta=0.0,verbose=False)
peak_a=torch.cuda.max_memory_allocated(); peak_r=torch.cuda.max_memory_reserved()

from integration.kernels.int8_optimized import OptimizedInt8Conv2d
ah_b=q_b=oh_b=0; n=0; dt=set(); ex=None
for mod in m.model.diffusion_model.modules():
    if not isinstance(mod,OptimizedInt8Conv2d): continue
    c=getattr(mod,"a_hat_cache",None)
    if c is not None:
        n+=1; ah_b+=c.numel()*c.element_size(); dt.add(str(c.dtype))
        if ex is None: ex=(tuple(c.shape), str(c.dtype), c.numel()*c.element_size())
    qs=getattr(mod,"_ahat_qscale",None)
    if qs is not None and qs.dim()==4: q_b+=qs.numel()*qs.element_size()
    o=getattr(mod,"o_hat_cache",None)
    if o is not None: oh_b+=o.numel()*o.element_size()
elems=ah_b//2 if "float16" in str(dt) else ah_b
print("MEMJSON:"+json.dumps({"arm":ARM,"peak_alloc_GB":peak_a/2**30,"peak_reserved_GB":peak_r/2**30,
    "model_base_GB":base/2**30,"n_layers_with_ahat":n,"ahat_dtypes":sorted(dt),
    "ahat_alloc_GB":ah_b/2**30,"qscale_GB":q_b/2**30,"ohat_GB":oh_b/2**30,
    "ahat_elems":elems,"bytes_per_elem_allocated":ah_b/max(elems,1),
    "example_layer":ex}))
