"""One arm: MEASURED E2E wall clock, MEASURED peak memory, and REAL decoded samples.

Nothing in the output of this script is derived from another number -- ms/step comes from CUDA
events around a full 50-step sample at batch 128, peak memory from torch.cuda.max_memory_*
during that same run, and the images from decoding an actual generated latent.

argv: fp16 | ptq | ahat0 | ahat32 | blockk64
  fp16      fp16 baseline
  ptq       W8A8 PTQ, no a_hat
  ahat0     W8A8 MoDiff, fp16 a_hat
  ahat32    W8A8 MoDiff, blockwise int8 a_hat            <-- the shipping config
  blockk64  ahat32 + blockwise conv-input quantizer B=64 <-- the arm the 80% goal is about
"""
import json, os, statistics, sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers"),
              os.path.join(ROOT,"docs/ahat_fake_quant_2026-08-27/scripts")]
ARM=sys.argv[1]
E={"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1","MODIFF_AHAT_BITS":"16",
   "MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0","MODIFF_DELTA_MODE":"static",
   "MODIFF_ACT_BLOCK":"0","MODIFF_CONV_BLOCKK":"0","MODIFF_AHAT_BLOCK":"0"}
if ARM in ("ahat32","blockk64"): E["MODIFF_AHAT_BLOCK"]="32"
if ARM=="blockk64":
    E.update({"MODIFF_CONV_BLOCKK":"64","MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION":"1",
              "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION":"1",
              "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION":"1"})
MODE={"fp16":"fp16","ptq":"int8_baseline","ahat0":"int8","ahat32":"int8","blockk64":"int8_baseline"}[ARM]
os.environ.update(E)
from integration.utils.preflight import preflight, MODEL
preflight(*MODEL, what="measure_all.py")
import torch, numpy as np
import integration.benchmarks.benchmark_ldm as B
import ahat_fake_quant_grid as G

SHAPE, BATCH, STEPS, NQ, SEED = (4,32,32), 128, 50, 6, 20260805
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/ahat_only_conv_2026-09-02/tmp", batch_size=BATCH, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE)
q = MODE!="fp16"

def once(n=BATCH):
    if q: B.reset_modiff_state_int8(m.model.diffusion_model)
    B._reset_wxax_modiff_safe(m)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        o=s.sample(S=STEPS,batch_size=n,shape=SHAPE,eta=0.0,verbose=False)
    return o[0] if isinstance(o,(tuple,list)) else o

once()                                    # warm up; also settles attention self-calibration
torch.cuda.synchronize()
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
ts=[]
for _ in range(3):                        # MEASURED wall clock
    a,b=torch.cuda.Event(True),torch.cuda.Event(True)
    a.record(); once(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/STEPS)
peak_alloc=torch.cuda.max_memory_allocated()/2**30   # MEASURED, during the timed batch-128 runs
peak_resv =torch.cuda.max_memory_reserved()/2**30

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
lat=once(NQ).detach().float().cpu()                  # REAL latent, fixed seed
img=G.decode(m,lat)                                  # REAL decode through the VAE
np.save(f"docs/ahat_only_conv_2026-09-02/data/img_{ARM}.npy", img)
torch.save(lat, f"docs/ahat_only_conv_2026-09-02/data/lat_{ARM}.pt")
print("MEASJSON:"+json.dumps({"arm":ARM,"mode":MODE,
  "ms_step":statistics.median(ts),"trials":ts,
  "peak_alloc_GB":peak_alloc,"peak_reserved_GB":peak_resv,
  "img_shape":list(img.shape)}))
