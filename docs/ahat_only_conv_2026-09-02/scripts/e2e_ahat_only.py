"""Wall-clock E2E for the a_hat-only configuration (blockwise a_hat kept, conv-input quantizer
reverted to per-tensor). One process per arm; batch 128, 50 DDIM, CUDA events, median of 2
after 1 warmup. argv: fp16 | ptq | ahat0 | ahat32"""
import os, statistics, sys, json
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
ARM=sys.argv[1]
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
    "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0",
    "MODIFF_DELTA_MODE":"static","MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0",
    "MODIFF_AHAT_BLOCK":"32" if ARM=="ahat32" else "0"})
MODE={"fp16":"fp16","ptq":"int8_baseline","ahat0":"int8","ahat32":"int8"}[ARM]
import torch, integration.benchmarks.benchmark_ldm as B
SHAPE,BATCH,STEPS=(4,32,32),128,50
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="docs/ahat_only_conv_2026-09-02/tmp",
    batch_size=BATCH, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE)
def once():
    if MODE!="fp16": B.reset_modiff_state_int8(m.model.diffusion_model)
    B._reset_wxax_modiff_safe(m)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        s.sample(S=STEPS,batch_size=BATCH,shape=SHAPE,eta=0.0,verbose=False)
once(); torch.cuda.synchronize(); ts=[]
for _ in range(2):
    a,b=torch.cuda.Event(True),torch.cuda.Event(True)
    a.record(); once(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/STEPS)
print("E2EJSON:"+json.dumps({"arm":ARM,"mode":MODE,"ms_step":statistics.median(ts),"trials":ts}))
