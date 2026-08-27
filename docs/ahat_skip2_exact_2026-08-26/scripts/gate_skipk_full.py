"""Full gate: for every K, run the PATCHED path TWICE and compare both to one unpatched baseline.
The patched-twice control is the check the committed validate_e2e.py lacked (it ran baseline twice
but the patched path only once), which is why patch_skip2's cross-generation divergence went
unnoticed. Also reports refresh-delegated call counts so a vacuous run cannot pass silently.
"""
import os, shutil, sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers"),
              os.path.dirname(os.path.abspath(__file__))]
import numpy as np, torch
from PIL import Image
import integration.benchmarks.benchmark_ldm as B
from integration.utils import attention_identity_guard as guard
import patch_skipk
torch.backends.cudnn.benchmark=False
BATCH,STEPS,N=4,12,4
KS=[int(k) for k in os.environ.get("KS","2,3,4,5,6,8").split(",")]
OUT=os.environ.get("GATE_OUT", "/tmp/gate_skipk_out")
def run(tag,K=None):
    if K is not None: patch_skipk.install(K)
    guard.seed_model_construction(); torch.manual_seed(777)
    r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir=OUT,
        batch_size=BATCH, steps=STEPS, shape=(4,32,32))
    r.run_mode("int8", num_samples=N, calibrate=True, force_recalibrate=False)
    st=dict(patch_skipk._stats) if K is not None else None
    if K is not None: patch_skipk.uninstall()
    s,d=os.path.join(OUT,"int8"),os.path.join(OUT,tag)
    if os.path.exists(d): shutil.rmtree(d)
    os.rename(s,d)
    return np.stack([np.array(Image.open(os.path.join(d,f"{i:05d}.png"))) for i in range(N)]), st
def diff(a,b):
    return int(np.abs(a.astype(np.int32)-b.astype(np.int32)).max()), float((a!=b).mean())
base,_=run("baseline")
print(f"gate: steps={STEPS} batch={BATCH} N={N}  refresh={os.environ.get('MODIFF_DELTA_REFRESH','default 4')}")
rows=[]
for K in KS:
    r1,s1=run(f"k{K}_a",K); r2,s2=run(f"k{K}_b",K)
    d12,f12=diff(r1,r2); d1b,f1b=diff(r1,base); d2b,f2b=diff(r2,base)
    ok=(d12==0 and d1b==0 and d2b==0 and s1["patched_calls"]>0)
    rows.append((K,d12,d1b,d2b,s1["patched_calls"],s1.get("refresh_delegated",0),ok))
    print(f"  K={K}: run1vs2={d12}/255  run1vsbase={d1b}/255  run2vsbase={d2b}/255  "
          f"patched={s1['patched_calls']} delegated={s1.get('refresh_delegated',0)}  "
          f"{'PASS' if ok else 'FAIL'}")
print("\n=== GATE SUMMARY ===")
allok=all(r[6] for r in rows)
for K,d12,d1b,d2b,p,dg,ok in rows:
    print(f"  K={K}: {'BIT-EXACT + deterministic' if ok else f'FAIL (run1vs2={d12}, vsbase={d1b}/{d2b})'}")
print("VERDICT:", "ALL PASS -- timings trustworthy" if allok else "FAILURES PRESENT")
