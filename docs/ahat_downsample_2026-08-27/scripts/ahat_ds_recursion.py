"""Does the downsampled-a_hat spatial error ACCUMULATE in conv(a_hat) - o_hat?

The single-step screen (ahat_downsample_screen.py) says f=2 inflates the delta's absmax 1.82x.
That is only the entry price. The structural question is the one that killed the int8 a_hat cache
(C15): o_hat only ever adds conv(code)*dequant and never sees a_hat's own storage error, so any
error in a_hat's stored value accumulates in conv(a_hat) - o_hat instead of cancelling.

This captures the REAL per-step target activation silu(gn(x)) for one layer over a real generation,
then replays two schemes on it:
  full : a_hat_t = a_hat_{t-1} + dequant(Q(target_t - a_hat_{t-1}))          <- production
  ds(f): ref = up(down(a_hat_{t-1})); a_hat_t = ref + dequant(Q(target_t - ref))
and tracks, per step, the invariant residual  ||a_hat_t - o_hat_equivalent||  where
o_hat_equivalent = a_hat_0 + sum of dequantized codes (what o_hat's accumulation represents; conv is
linear so the scalar/elementwise form transfers, as simulate_drift.py established).

If the ds residual grows without bound the scheme is dead for the same reason int8 storage was.
"""
import os, sys
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers")]
import torch, torch.nn.functional as F
torch.backends.cudnn.benchmark=False
import integration.kernels.int8_optimized as i8
import integration.benchmarks.benchmark_ldm as B
from integration.utils import attention_identity_guard as guard

TARGET_LAYER=os.environ.get("DS_LAYER","input_blocks.4.0.in_conv")
STEPS=int(os.environ.get("DS_STEPS","30")); BATCH=2
C=i8.OptimizedInt8Conv2d; ORIG=C.forward_gn_fused_modiff
caps=[]   # (target fp32 cpu, scale used that step)

def shim(self,x,gw,gb,ng,eps,ms,sh,residual=None):
    if (self.layer_name or "")==TARGET_LAYER and not self.delta_dynamic and x.dtype==torch.float16 and len(caps)<STEPS:
        if not x.is_contiguous(memory_format=torch.channels_last):
            x=x.contiguous(memory_format=torch.channels_last)
        self._ensure_state_buffers(x)
        with torch.no_grad():
            xf=x.float(); n=F.group_norm(xf,ng,gw.float(),gb.float(),eps)
            if ms is not None and ms.numel()>0:
                N,Cc=x.shape[0],x.shape[1]
                n=n*(1.0+ms.float().view(N,Cc,1,1))+sh.float().view(N,Cc,1,1)
            o=F.silu(n.half().float())
            if not self._smooth_is_identity and self._smooth_inv_flat.numel()>0:
                o=o*self._smooth_inv_flat.float().view(1,-1,1,1)
            sc,_=self._delta_scale_args(x.device)
            caps.append((o.clone(), float(sc.view(-1)[0])))
    return ORIG(self,x,gw,gb,ng,eps,ms,sh,residual)

C.forward_gn_fused_modiff=shim
guard.seed_model_construction(); torch.manual_seed(777)
r=B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir=os.environ.get("DS_OUT", "/tmp/ahat_dsr_out"),
    batch_size=BATCH, steps=STEPS, shape=(4,32,32))
r.run_mode("int8", num_samples=BATCH, calibrate=True, force_recalibrate=False)
C.forward_gn_fused_modiff=ORIG
print(f"\ncaptured {len(caps)} steps for {TARGET_LAYER}, shape {tuple(caps[0][0].shape)}")

def replay(f=None):
    a=torch.zeros_like(caps[0][0]); acc=a.clone()   # acc = a_hat_0 + sum dequant(codes) = o_hat's view
    H,W=a.shape[2],a.shape[3]; out=[]
    for t,(tgt,sc) in enumerate(caps):
        ref = a if f is None else F.interpolate(F.avg_pool2d(a,f),size=(H,W),mode="nearest")
        d = tgt-ref
        q = torch.clamp(torch.round(d*sc),-127,127)
        deq = q/sc
        acc = acc + deq                     # o_hat's accumulation: only ever sees dequantized codes
        a = (ref + deq).half().float()      # a_hat as STORED (fp16), then coarsened on next read
        resid=(a-acc).abs().max().item()
        out.append((t+1, d.abs().max().item(), resid, (tgt-acc).abs().max().item()))
    return out
full=replay(None); ds2=replay(2); ds4=replay(4)
print("\n=== invariant residual max|a_hat - (a_hat_0 + sum dequant codes)| , i.e. what conv(a_hat)-o_hat sees ===")
print(f"{'step':>5}{'full resid':>13}{'ds2 resid':>12}{'ds4 resid':>12}   |{'full |tgt-acc|':>16}{'ds2':>10}{'ds4':>10}")
for i in list(range(min(5,len(full))))+list(range(len(full)//2,len(full)//2+1))+list(range(len(full)-3,len(full))):
    print(f"{full[i][0]:>5}{full[i][2]:>13.6g}{ds2[i][2]:>12.6g}{ds4[i][2]:>12.6g}   |{full[i][3]:>16.6g}{ds2[i][3]:>10.6g}{ds4[i][3]:>10.6g}")
print(f"\n  delta absmax, median over steps : full {sorted(x[1] for x in full)[len(full)//2]:.4g}"
      f" | ds2 {sorted(x[1] for x in ds2)[len(ds2)//2]:.4g} | ds4 {sorted(x[1] for x in ds4)[len(ds4)//2]:.4g}")
print(f"  FINAL invariant residual        : full {full[-1][2]:.6g} | ds2 {ds2[-1][2]:.6g} | ds4 {ds4[-1][2]:.6g}")
print(f"  FINAL |target - o_hat_view|     : full {full[-1][3]:.6g} | ds2 {ds2[-1][3]:.6g} | ds4 {ds4[-1][3]:.6g}")
