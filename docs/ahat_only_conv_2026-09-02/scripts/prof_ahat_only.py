"""Conv speedup with the a_hat blockwise cache KEPT and the conv-input quantizer REVERTED to
per-tensor. This is the shipping decision that follows from the 2026-09-02 measurements:
`a_hat` blockwise is a net win (docs/ahat_blockwise_2026-09-01), the conv-input blockwise is not
at W8A8 (docs/wa_budget_2026-09-02, docs/conv_blockk_e2e_2026-09-02).

No code revert is needed -- every MODIFF_CONV_BLOCKK / MODIFF_ACT_BLOCK path added on 2026-09-02
is inert at its default, so the configuration is env only.

Four arms, conv kernels bucketed from a CUDA-only profile (CPU+CUDA double-counts, because both
the aten op and the kernel carry self device time):
  fp16
  int8_baseline   PTQ, no a_hat at all
  int8 AHAT=0     MoDiff with an fp16 a_hat cache
  int8 AHAT=32    MoDiff with the blockwise int8 a_hat cache   <-- the requested config

argv: fp16 | ptq | ahat0 | ahat32
"""
import os, re, sys, collections
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,"src/taming-transformers")]
ARM=sys.argv[1]
BASE={"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
      "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0",
      "MODIFF_DELTA_MODE":"static",
      # reverted to the original per-tensor conv-input quantizer
      "MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0"}
MODE={"fp16":"fp16","ptq":"int8_baseline","ahat0":"int8","ahat32":"int8"}[ARM]
BASE["MODIFF_AHAT_BLOCK"] = "32" if ARM=="ahat32" else "0"
os.environ.update(BASE)
import torch, integration.benchmarks.benchmark_ldm as B
from torch.profiler import profile, ProfilerActivity
STEPS,BATCH=10,128
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="docs/ahat_only_conv_2026-09-02/tmp",
    batch_size=BATCH, steps=STEPS, shape=(4,32,32),
    calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m,s=r._setup_model(MODE)
def run():
    if MODE!="fp16": B.reset_modiff_state_int8(m.model.diffusion_model)
    B._reset_wxax_modiff_safe(m)
    with torch.inference_mode(), torch.amp.autocast("cuda",enabled=True,dtype=torch.float16):
        s.sample(S=STEPS,batch_size=BATCH,shape=(4,32,32),eta=0.0,verbose=False)
run()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    run()
CONV=re.compile(r"fprop|ImplicitGemm|blockk_kernel|scale_store|scale_accumulate|implicit",re.I)
QUANT=re.compile(r"group_norm_silu|gn_|ahat_|quantize|_pack|delta",re.I)
tot=0.0; buckets=collections.Counter(); detail=collections.Counter()
for e in p.key_averages():
    v=e.self_device_time_total
    if v<=0: continue
    tot+=v
    b = "conv" if CONV.search(e.key) else ("gn_quant" if QUANT.search(e.key) else "other")
    buckets[b]+=v; detail[(b,e.key[:64])]+=v
f=lambda v: v/1e3/STEPS
print(f"ARM={ARM} mode={MODE} AHAT_BLOCK={BASE['MODIFF_AHAT_BLOCK']}  traced GPU {f(tot):.2f} ms/step")
for b in ("conv","gn_quant","other"):
    print(f"  {f(buckets[b]):8.2f} ms/step  {b}")
print("  -- top conv kernels --")
for (b,k),v in sorted(((kk,vv) for kk,vv in detail.items() if kk[0]=="conv"),key=lambda z:-z[1])[:4]:
    print(f"     {f(v):7.2f}  {k}")
