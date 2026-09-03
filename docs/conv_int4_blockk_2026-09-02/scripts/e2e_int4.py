"""MEASURED W4A4 end-to-end: wall clock, peak memory, real decoded samples.

There is no fused GN->int4-blockwise-quantize kernel, so the blockwise arm must run with the
fusion kill switches set and pays a separate quantize pass. The `unfused` arm is the same
configuration with the per-tensor quantizer, which is what separates the fusion handicap from the
blockwise cost -- read `blockwise vs unfused` for the blockwise tax and `vs shipped` for what a
user would actually see today.

argv: fp16 | shipped | unfused | b64 | b128 | b64fused | b64gnonly

b64fused keeps the GN->conv fusion ON and injects gn_silu_blockk_quantize_pack_int4 into it,
so it does NOT pay the separate GN + separate quantize passes the other blockwise arms do.
"""
import json, os, statistics, sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers"),
              os.path.join(ROOT,"docs/ahat_fake_quant_2026-08-27/scripts")]
ARM=sys.argv[1]
E={"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1","MODIFF_AHAT_BITS":"16",
   "MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0","MODIFF_DELTA_MODE":"static",
   "MODIFF_ACT_BLOCK":"0","MODIFF_AHAT_BLOCK":"0","MODIFF_CONV_BLOCKK":"0"}
if ARM in ("unfused","b64","b128"):
    E.update({"MODIFF_DISABLE_GN_MODIFF_FUSION":"1","MODIFF_DISABLE_GN_INT8_FUSION":"1",
              "MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION":"1",
              "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION":"1",
              "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION":"1"})
if ARM=="b64gnonly":
    # ONLY the blockwise GN->quantize kernel is fused; every layer it cannot serve runs unfused,
    # so the result attributes the gain to that kernel alone.
    E.update({"MODIFF_CONV_BLOCKK":"64","MODIFF_CONV_BLOCKK_GN_ONLY":"1",
              "MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION":"1",
              "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION":"1",
              "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION":"1"})
if ARM=="b64fused":
    # only the three folds the blockwise path cannot serve; the GN->conv fusion stays ON
    E.update({"MODIFF_CONV_BLOCKK":"64","MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION":"1",
              "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION":"1",
              "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION":"1"})
if ARM=="b64":  E["MODIFF_CONV_BLOCKK"]="64"
if ARM=="b128": E["MODIFF_CONV_BLOCKK"]="128"
MODE="fp16" if ARM=="fp16" else "int4_baseline"
os.environ.update(E)
from integration.utils.preflight import preflight, MODEL
preflight(*MODEL, what="e2e_int4.py")
import torch, numpy as np
import integration.benchmarks.benchmark_ldm as B
import ahat_fake_quant_grid as G
SHAPE,BATCH,STEPS,NQ,SEED=(4,32,32),128,50,6,20260805
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/conv_int4_blockk_2026-09-02/tmp", batch_size=BATCH, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE)
q=MODE!="fp16"
def once(n=BATCH):
    if q: B.reset_modiff_state_int8(m.model.diffusion_model)
    B._reset_wxax_modiff_safe(m)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        o=s.sample(S=STEPS,batch_size=n,shape=SHAPE,eta=0.0,verbose=False)
    return o[0] if isinstance(o,(tuple,list)) else o
once(); torch.cuda.synchronize()
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
ts=[]
for _ in range(3):
    a,b=torch.cuda.Event(True),torch.cuda.Event(True)
    a.record(); once(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/STEPS)
pa=torch.cuda.max_memory_allocated()/2**30; pr=torch.cuda.max_memory_reserved()/2**30
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
lat=once(NQ).detach().float().cpu()
np.save(f"docs/conv_int4_blockk_2026-09-02/data/img_{ARM}.npy", G.decode(m,lat))
torch.save(lat, f"docs/conv_int4_blockk_2026-09-02/data/lat_{ARM}.pt")
print("I4JSON:"+json.dumps({"arm":ARM,"mode":MODE,"ms_step":statistics.median(ts),"trials":ts,
                            "peak_alloc_GB":pa,"peak_reserved_GB":pr}))
