"""Pack before/after + fp16 vs int8 first-modulated yq stats."""
from __future__ import annotations
import os, sys
ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]
os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_AHAT_BLOCK"] = "32"

from integration.utils.preflight import preflight, MODEL
preflight(*MODEL, what="probe_pack.py")

import torch
import modiff_cutlass as mc
import integration.benchmarks.benchmark_ldm as B
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/ahat_blockwise_2026-09-01/tmp_probe",
    batch_size=2, steps=50, shape=(4, 32, 32),
    calibration_path=B._default_calibration_path("int8"),
    auto_delta_table=True)
model, sampler = runner._setup_model("int8")
unet = model.model.diffusion_model

pack_logs = []
_orig_pack = OptimizedInt8Conv2d._pack_ahat_along_c

def pack_hook(self):
    a = self.a_hat_cache
    if a is not None and a.dtype != torch.int8:
        before = float(a.float().abs().max())
        _orig_pack(self)
        q, s = self.a_hat_cache, self._ahat_qscale
        n, c, h, w = q.shape
        g = s.shape[-1]
        bsz = c // g
        recon = (q.permute(0,2,3,1).reshape(n,h,w,g,bsz).float() * s.unsqueeze(-1))
        after = float(recon.abs().max())
        rel = float((recon.reshape(n,h,w,c).permute(0,3,1,2) - a.float()).norm() / (a.float().norm()+1e-8))
        if len(pack_logs) < 6:
            pack_logs.append((getattr(self, "layer_name", "?"), tuple(a.shape), before, after, rel,
                              float(s.mean()), float(s.max())))
    else:
        _orig_pack(self)

OptimizedInt8Conv2d._pack_ahat_along_c = pack_hook

orig = mc.group_norm_silu_delta_quantize_nhwc
yq_stats = []

def wrap(*a, **k):
    yq = orig(*a, **k)
    if len(yq_stats) < 70:
        yq_stats.append((float(yq.abs().max()), float(yq.abs().float().mean()),
                         float(a[7].reshape(-1)[0]), tuple(a[0].shape),
                         a[3].dtype, None if a[-1] is None else tuple(getattr(a[-1],'shape',()))))
    return yq

mc.group_norm_silu_delta_quantize_nhwc = wrap

B.reset_modiff_state_int8(unet)
B._reset_wxax_modiff_safe(model)
torch.manual_seed(0)
with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
    sampler.sample(S=50, batch_size=2, shape=(4, 32, 32), eta=0.0, verbose=False)

print("=== pack t=T (first 6) ===")
for row in pack_logs:
    print(" ", row)

print("\n=== first modulated GN yq (int8 B=32), first 10 ===")
for i, row in enumerate(yq_stats[:10]):
    print(f"  {i} absmax={row[0]:.1f} meanabs={row[1]:.3f} dscale={row[2]:.1f} x={row[3]} cache={row[4]} s={row[5]}")
sat = sum(1 for r in yq_stats[:62] if r[0] >= 127)
print(f"  first-mod 62 layers: n_sat127={sat} mean_meanabs={sum(r[1] for r in yq_stats[:62])/max(len(yq_stats[:62]),1):.3f}")

print("\n===== fp16 a_hat arm =====")
os.environ["MODIFF_AHAT_BLOCK"] = "0"
yq_stats.clear()
B.reset_modiff_state_int8(unet)
B._reset_wxax_modiff_safe(model)
torch.manual_seed(0)
with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
    sampler.sample(S=50, batch_size=2, shape=(4, 32, 32), eta=0.0, verbose=False)
print("=== first modulated GN yq (fp16 a_hat), first 10 ===")
for i, row in enumerate(yq_stats[:10]):
    print(f"  {i} absmax={row[0]:.1f} meanabs={row[1]:.3f} dscale={row[2]:.1f} x={row[3]} cache={row[4]} s={row[5]}")
sat = sum(1 for r in yq_stats[:62] if r[0] >= 127)
print(f"  first-mod 62 layers: n_sat127={sat} mean_meanabs={sum(r[1] for r in yq_stats[:62])/max(len(yq_stats[:62]),1):.3f}")
print(f"  n_yq_stats={len(yq_stats)}")
