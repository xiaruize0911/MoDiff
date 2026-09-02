"""Capture real silu(gn(x)) targets, a_hat, and per-step delta scales.

One W8A8 generation. Keeps the first layer that hits each of the plan's
shapes (192x32x32, 192x16x16, 384x16x16) for STEPS modulated calls.

Run from repo root:
    python docs/ahat_svd_residual_2026-09-01/scripts/capture.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = False

import integration.kernels.int8_optimized as i8
import integration.benchmarks.benchmark_ldm as B
from integration.utils import attention_identity_guard as guard

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Named layers so the 192×16×16 arm is the same conv C20 used.
TARGET_LAYERS = (
    "input_blocks.1.0.in_conv",    # 192×32×32
    "input_blocks.4.0.in_conv",    # 192×16×16 (C20 layer)
    "input_blocks.4.0.out_conv",   # 384×16×16
)
STEPS = int(os.environ.get("SVD_STEPS", "30"))
BATCH = int(os.environ.get("SVD_BATCH", "2"))

C = i8.OptimizedInt8Conv2d
ORIG = C.forward_gn_fused_modiff
caps = defaultdict(list)
bound = {}  # (C,H,W) -> layer_name


def target_activation(self, x, gn_weight, gn_bias, num_groups, eps, ms2d, sh2d):
    xf = x.float()
    n = F.group_norm(xf, num_groups, gn_weight.float(), gn_bias.float(), eps)
    if ms2d is not None and ms2d.numel() > 0:
        N, Cc = x.shape[0], x.shape[1]
        n = n * (1.0 + ms2d.float().view(N, Cc, 1, 1)) + sh2d.float().view(N, Cc, 1, 1)
    o = F.silu(n.half().float())
    if not self._smooth_is_identity and self._smooth_inv_flat.numel() > 0:
        o = o * self._smooth_inv_flat.float().view(1, -1, 1, 1)
    return o


def shim(self, x, gn_weight, gn_bias, num_groups, eps, mod_scale2d, mod_shift2d,
         residual=None):
    if self.delta_dynamic or x.dtype != torch.float16:
        return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                    mod_scale2d, mod_shift2d, residual)
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
    name = self.layer_name or ""
    if name not in TARGET_LAYERS or len(caps[name]) >= STEPS:
        return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                    mod_scale2d, mod_shift2d, residual)
    key = (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
    bound[key] = name
    self._ensure_state_buffers(x)
    with torch.no_grad():
        tgt = target_activation(self, x, gn_weight, gn_bias, num_groups, eps,
                                mod_scale2d, mod_shift2d)
        ah = self.a_hat_cache.detach().float()
        sc, _ = self._delta_scale_args(x.device)
        caps[name].append({
            "tgt": tgt.cpu().half(),
            "ahat": ah.cpu().half(),
            "scale": float(sc.view(-1)[0]),
            "step": int(self.step_count) + 1,
            "shape": key,
        })
    return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                mod_scale2d, mod_shift2d, residual)


C.forward_gn_fused_modiff = shim
guard.seed_model_construction()
torch.manual_seed(777)
runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir=os.environ.get("SVD_OUT", "/tmp/ahat_svd_cap"),
    batch_size=BATCH, steps=STEPS, shape=(4, 32, 32))
runner.run_mode("int8", num_samples=BATCH, calibrate=True, force_recalibrate=False)
C.forward_gn_fused_modiff = ORIG

payload = {
    "steps": STEPS,
    "batch": BATCH,
    "bound": {f"{c},{h}x{w}": n for (c, h, w), n in bound.items()},
    "layers": {},
}
for name, rows in caps.items():
    payload["layers"][name] = {
        "shape": rows[0]["shape"],
        "n_steps": len(rows),
        "scales": [r["scale"] for r in rows],
        "steps": [r["step"] for r in rows],
        "tgt": torch.stack([r["tgt"] for r in rows]),     # T,N,C,H,W half
        "ahat": torch.stack([r["ahat"] for r in rows]),
    }
    print(f"captured {name} shape={rows[0]['shape']} n={len(rows)}")

out_path = os.path.join(OUT_DIR, "capture.pt")
torch.save(payload, out_path)
print(f"wrote {out_path}")
