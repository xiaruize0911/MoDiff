import os, sys, collections
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers")]
for k,v in {"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
            "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0",
            "MODIFF_AHAT_BLOCK":"0","MODIFF_DELTA_MODE":"static","MODIFF_ACT_BLOCK":"0",
            "MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION":"1",
            "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION":"1",
            "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION":"1",
            "MODIFF_CONV_BLOCKK":"32","MODIFF_CONV_BLOCKK_CTRL":"0"}.items():
    os.environ[k]=v
import torch, integration.benchmarks.benchmark_ldm as B
import integration.kernels.int8_optimized as I8
C=I8.OptimizedInt8Conv2d
cnt=collections.Counter(); perlayer=collections.defaultdict(collections.Counter)
def wrap(name, fn, ok=None):
    def w(self,*a,**kw):
        r=fn(self,*a,**kw)
        tag=name if (ok is None or r is not None) else name+"_MISS"
        cnt[tag]+=1; perlayer[getattr(self,'layer_name','?')][tag]+=1
        return r
    return w
C.blockk_gn_fused=wrap("blockk_fused",C.blockk_gn_fused,ok=1)
C._forward_conv_blockk=wrap("blockk_twopass",C._forward_conv_blockk)
for nm in ["_forward_standard","_forward_first_step","_forward_modulated",
           "forward_gn_fused_modiff","_forward_modulated_static_fused_silu",
           "forward_from_int8","_conv_from_int8_o_hat","_evt_ohat"]:
    if hasattr(C,nm): setattr(C,nm,wrap(nm,getattr(C,nm)))
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/conv_blockk_e2e_2026-09-02/tmp", batch_size=2, steps=5, shape=(4,32,32),
    calibration_path=B._default_calibration_path("int8"), auto_delta_table=True)
m,s=r._setup_model("int8")
B.reset_modiff_state_int8(m.model.diffusion_model); B._reset_wxax_modiff_safe(m)
cnt.clear(); perlayer.clear()
with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
    s.sample(S=5,batch_size=2,shape=(4,32,32),eta=0.0,verbose=False)
print("=== path counts over 5 steps ===")
for k,v in cnt.most_common(): print(f"  {k:38s} {v}")
print("\n=== layers that took a NON-blockwise conv path ===")
bad=[(l,dict(c)) for l,c in perlayer.items()
     if any(t.startswith(('_forward_','forward_gn','forward_from','_conv_from','_evt')) for t in c)]
print(f"  {len(bad)} layers")
for l,c in bad[:12]: print(f"    {l}: {c}")
