"""ms/step at two step counts -> separate the one-time first step from the steady state.
total(S) = A + (S-1)*B  =>  measure S=10 and S=50, solve for A (first step) and B (steady)."""
import os,sys,statistics,json
MODE=sys.argv[1]; S=int(sys.argv[2])
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
 "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0","MODIFF_DELTA_MODE":"static",
 "MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0","MODIFF_AHAT_BLOCK":"0"})
ROOT="/workspace/MoDiff"; sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
import torch, integration.benchmarks.benchmark_ldm as B
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="docs/ahat_only_conv_2026-09-02/tmp",
 batch_size=128, steps=S, shape=(4,32,32),
 calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE)
reset=(B.reset_modiff_state_int8 if MODE.startswith("int8") else B.reset_modiff_state_int4)
def run():
    reset(m.model.diffusion_model); B._reset_wxax_modiff_safe(m)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        s.sample(S=S,batch_size=128,shape=(4,32,32),eta=0.0,verbose=False)
run(); torch.cuda.synchronize(); ts=[]
for _ in range(2):
    a,b=torch.cuda.Event(True),torch.cuda.Event(True)
    a.record(); run(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
print("TOTJSON:"+json.dumps({"mode":MODE,"S":S,"total_ms":statistics.median(ts)}))
