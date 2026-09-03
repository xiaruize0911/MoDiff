"""E2E for one o_hat sim config. argv: mode BITS SR   (BITS=0 -> reference, fp16 o_hat)"""
import os,statistics,sys,json
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
MODE,BITS,SR=sys.argv[1],sys.argv[2],sys.argv[3]
TAG=sys.argv[4] if len(sys.argv)>4 else f"b{BITS}sr{SR}"
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
 "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0","MODIFF_DELTA_MODE":"static",
 "MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0","MODIFF_AHAT_BLOCK":"32",
 "MODIFF_OHAT_SIM_BITS":BITS,"MODIFF_OHAT_SIM_BLOCK":"32","MODIFF_OHAT_SIM_SR":SR})
import torch, integration.benchmarks.benchmark_ldm as B
SHAPE,BATCH,STEPS=(4,32,32),128,50
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
 ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="docs/ahat_only_conv_2026-09-02/tmp",
 batch_size=BATCH, steps=STEPS, shape=SHAPE,
 calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE)
reset=B.reset_modiff_state_int8 if MODE=="int8" else B.reset_modiff_state_int4
def once():
    reset(m.model.diffusion_model); B._reset_wxax_modiff_safe(m)
    torch.manual_seed(1234)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        return s.sample(S=STEPS,batch_size=BATCH,shape=SHAPE,eta=0.0,verbose=False)
out=once(); torch.cuda.synchronize()
lat=(out[0] if isinstance(out,(tuple,list)) else out).float().cpu()
torch.save(lat, f"/tmp/oh_lat_{MODE}_{TAG}.pt")
print("OHJSON:"+json.dumps({"mode":MODE,"bits":int(BITS),"sr":int(SR),"tag":TAG,
 "finite":bool(torch.isfinite(lat).all()),"absmax":float(lat.abs().max()),
 "mean":float(lat.mean()),"std":float(lat.std())}))
