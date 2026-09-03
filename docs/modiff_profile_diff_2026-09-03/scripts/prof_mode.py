"""Per-kernel CUDA time for one MODE, so MoDiff and PTQ can be diffed. argv: mode out.json"""
import os,sys,json
MODE=sys.argv[1]
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
 "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0","MODIFF_DELTA_MODE":"static",
 "MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0","MODIFF_AHAT_BLOCK":"0"})
ROOT="/workspace/MoDiff"; sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
import torch, integration.benchmarks.benchmark_ldm as B
from torch.profiler import profile, ProfilerActivity
BATCH,STEPS=128,6
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="docs/ahat_only_conv_2026-09-02/tmp",
 batch_size=BATCH, steps=STEPS, shape=(4,32,32),
 calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE)
reset = (B.reset_modiff_state_int8 if MODE.startswith("int8") else B.reset_modiff_state_int4)
def run():
    reset(m.model.diffusion_model); B._reset_wxax_modiff_safe(m)
    torch.manual_seed(1234)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        s.sample(S=STEPS,batch_size=BATCH,shape=(4,32,32),eta=0.0,verbose=False)
run(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    run(); torch.cuda.synchronize()
agg={}
for e in prof.key_averages():
    t=getattr(e,"self_device_time_total",0) or getattr(e,"self_cuda_time_total",0)
    if t>0: agg[e.key]=agg.get(e.key,0.0)+t/1000.0/STEPS
json.dump(agg, open(sys.argv[2],"w"))
print(f"{MODE}: 总 CUDA {sum(agg.values()):.2f} ms/step, {len(agg)} kernels")
