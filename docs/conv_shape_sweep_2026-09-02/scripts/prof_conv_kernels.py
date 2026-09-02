import os, sys, collections, re
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
ARM=sys.argv[1]
BASE={"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1","MODIFF_AHAT_BITS":"16",
      "MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0","MODIFF_AHAT_BLOCK":"0","MODIFF_DELTA_MODE":"static",
      "MODIFF_ACT_BLOCK":"0","MODIFF_CONV_BLOCKK_CTRL":"0","MODIFF_CONV_BLOCKK":"0"}
if ARM=="b64fused":
    BASE.update({"MODIFF_CONV_BLOCKK":"64","MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION":"1",
                 "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION":"1","MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION":"1"})
os.environ.update(BASE)
import torch, integration.benchmarks.benchmark_ldm as B
from torch.profiler import profile, ProfilerActivity
STEPS,BATCH=10,128
mode="fp16" if ARM=="fp16" else "int8_baseline"
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/x",
    batch_size=BATCH, steps=STEPS, shape=(4,32,32),
    calibration_path=B._default_calibration_path(mode), auto_delta_table=True)
m,s=r._setup_model(mode)
def run():
    if mode!="fp16": B.reset_modiff_state_int8(m.model.diffusion_model)
    B._reset_wxax_modiff_safe(m)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        s.sample(S=STEPS,batch_size=BATCH,shape=(4,32,32),eta=0.0,verbose=False)
run()
with profile(activities=[ProfilerActivity.CUDA]) as p:   # CUDA ONLY -- no double counting
    run()
CONV=re.compile(r"fprop|ImplicitGemm|blockk_kernel|scale_store|scale_accumulate|implicit", re.I)
QUANT=re.compile(r"group_norm_silu|gn_silu_blockk|gn_stats|quantize|_pack", re.I)
tot=0.0; conv=[]; quant=[]
for e in p.key_averages():
    v=e.self_device_time_total
    if v<=0: continue
    tot+=v
    if CONV.search(e.key): conv.append((e.key,v))
    elif QUANT.search(e.key): quant.append((e.key,v))
f=lambda v: v/1e3/STEPS
print(f"ARM={ARM}  traced GPU {f(tot):.2f} ms/step")
print("  -- CONV kernels --")
for k,v in sorted(conv,key=lambda z:-z[1]): print(f"    {f(v):7.2f} ms/step  {k[:76]}")
print(f"    {f(sum(v for _,v in conv)):7.2f} ms/step  == CONV TOTAL ({len(conv)} kernels)")
print("  -- GN / quantize kernels --")
for k,v in sorted(quant,key=lambda z:-z[1])[:4]: print(f"    {f(v):7.2f} ms/step  {k[:76]}")
print(f"    {f(sum(v for _,v in quant)):7.2f} ms/step  == GN/QUANT TOTAL")
