"""Feasibility screen: can a_hat be stored at REDUCED SPATIAL RESOLUTION?

Every refuted a_hat storage idea attacked bytes-PER-ELEMENT (int8/fp8/companding: refuted on a
~3.6-bit budget shortfall) or the number of WRITES (skip-K, deferred-write: refuted structurally and
at kernel level). This attacks the number of ELEMENTS, which reduces a_hat's READ and WRITE together
and is bounded by neither of those results.

The scheme under test: store a_hat at 1/f spatial resolution; on read, upsample it; quantize
`delta = silu(gn(x)) - upsample(a_hat_small)` as usual. a_hat's own dynamic range is untouched, so
the bit-budget argument that killed int8 storage does not apply. What is at risk instead is the
whole premise of MoDiff: that |delta| is much smaller than |activation|. A coarser reference makes
the delta bigger, and the quantizer's scale follows the delta's absmax directly.

So the screen measures exactly that: absmax(delta) with a downsampled reference, over absmax(delta)
with the real full-resolution reference, on REAL captured a_hat and real activations. The traffic
prize for f=2 is 4x less a_hat traffic (a_hat read+write is 8 of the apply kernel's 10 sectors/warp
per C17, so 10 -> 4 sectors, far beyond the 2.024 ms/step write-only ceiling).

REPORTED HONESTLY AS A SINGLE-STEP SCREEN. It does not simulate the recursion (a_hat_{t+1} built
from a downsampled a_hat_t compounds the coarsening across 200 steps), so the numbers here are the
OPTIMISTIC case. If the inflation is already large in one step there is no point simulating more.

Run: python ahat_downsample_screen.py
"""
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

FACTORS = [2, 4]
MODE = os.environ.get("DS_MODE", "int8")
BATCH = int(os.environ.get("DS_BATCH", "4"))
STEPS = int(os.environ.get("DS_STEPS", "10"))
MAX_REC = int(os.environ.get("DS_MAX_REC", "400"))

C = i8.OptimizedInt8Conv2d
ORIG = C.forward_gn_fused_modiff
rec = defaultdict(list)   # (C,H,W) -> list of dicts


def target_activation(self, x, gn_weight, gn_bias, num_groups, eps, ms2d, sh2d):
    """Replicate silu(gn(x)) [+mod] [*smooth_inv] -- what the fused kernel quantizes against a_hat.

    fp32 throughout: this screen compares absmax RATIOS, so matching the kernel's fp16 rounding is
    not needed and fp32 avoids adding its own noise to the comparison.
    """
    xf = x.float()
    n = F.group_norm(xf, num_groups, gn_weight.float(), gn_bias.float(), eps)
    if ms2d is not None and ms2d.numel() > 0:
        N, Cc = x.shape[0], x.shape[1]
        n = n * (1.0 + ms2d.float().view(N, Cc, 1, 1)) + sh2d.float().view(N, Cc, 1, 1)
    n = n.half().float()                      # the kernel rounds to fp16 before SiLU
    o = F.silu(n)
    if not self._smooth_is_identity and self._smooth_inv_flat.numel() > 0:
        o = o * self._smooth_inv_flat.float().view(1, -1, 1, 1)
    return o


def shim(self, x, gn_weight, gn_bias, num_groups, eps, mod_scale2d, mod_shift2d, residual=None):
    if (self.delta_dynamic or x.dtype != torch.float16
            or sum(len(v) for v in rec.values()) >= MAX_REC):
        return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                    mod_scale2d, mod_shift2d, residual)
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
    self._ensure_state_buffers(x)
    a_hat = self.a_hat_cache.detach().float().clone()
    with torch.no_grad():
        tgt = target_activation(self, x, gn_weight, gn_bias, num_groups, eps,
                                mod_scale2d, mod_shift2d)
        d_full = tgt - a_hat
        row = {"layer": self.layer_name, "step": int(self.step_count) + 1,
               "act_absmax": tgt.abs().max().item(),
               "d_full_absmax": d_full.abs().max().item(),
               "d_full_rms": d_full.pow(2).mean().sqrt().item(),
               "ahat_absmax": a_hat.abs().max().item()}
        H, W = x.shape[2], x.shape[3]
        for f in FACTORS:
            if H % f or W % f or H // f < 1:
                row[f"d_ds{f}_absmax"] = None
                continue
            small = F.avg_pool2d(a_hat, f)                       # store at 1/f resolution
            up = F.interpolate(small, size=(H, W), mode="nearest")  # cheapest possible read path
            d = tgt - up
            row[f"d_ds{f}_absmax"] = d.abs().max().item()
            row[f"d_ds{f}_rms"] = d.pow(2).mean().sqrt().item()
            up_b = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
            row[f"d_ds{f}b_absmax"] = (tgt - up_b).abs().max().item()
        rec[(x.shape[1], H, W)].append(row)
    return ORIG(self, x, gn_weight, gn_bias, num_groups, eps,
                mod_scale2d, mod_shift2d, residual)


C.forward_gn_fused_modiff = shim
guard.seed_model_construction()
torch.manual_seed(777)
runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir=os.environ.get("DS_OUT", "/tmp/ahat_ds_out"),
    batch_size=BATCH, steps=STEPS, shape=(4, 32, 32))
runner.run_mode(MODE, num_samples=BATCH, calibrate=True, force_recalibrate=False)
C.forward_gn_fused_modiff = ORIG

print("\n" + "=" * 100)
print("a_hat SPATIAL DOWNSAMPLING SCREEN -- inflation of the delta's absmax (sets the quant scale)")
print("=" * 100)
print(f"{'shape':<16}{'n':>4}{'gain_full':>11}{'x2 near':>10}{'x2 bilin':>10}{'x4 near':>10}"
      f"{'gain@x2':>10}{'gain@x4':>10}")
tot = defaultdict(list)
for key in sorted(rec, key=lambda k: -(k[1] * k[2])):
    rows = rec[key]
    def med(f):
        v = [r[f] for r in rows if r.get(f)]
        return sorted(v)[len(v) // 2] if v else None
    dfull = med("d_full_absmax"); act = med("act_absmax")
    if not dfull:
        continue
    r2 = med("d_ds2_absmax"); r2b = med("d_ds2b_absmax"); r4 = med("d_ds4_absmax")
    gain_full = act / dfull
    g2 = act / r2 if r2 else None
    g4 = act / r4 if r4 else None
    s = f"{key[0]},{key[1]}x{key[2]}"
    print(f"{s:<16}{len(rows):>4}{gain_full:>11.2f}"
          f"{(r2/dfull if r2 else float('nan')):>9.2f}x{(r2b/dfull if r2b else float('nan')):>9.2f}x"
          f"{(r4/dfull if r4 else float('nan')):>9.2f}x"
          f"{(g2 if g2 else float('nan')):>10.2f}{(g4 if g4 else float('nan')):>10.2f}")
    if r2: tot["x2"].append(r2 / dfull)
    if r2b: tot["x2b"].append(r2b / dfull)
    if r4: tot["x4"].append(r4 / dfull)
    tot["gain_full"].append(gain_full)
    if g2: tot["gain2"].append(g2)
    if g4: tot["gain4"].append(g4)


def m(k):
    v = sorted(tot[k]); return v[len(v) // 2] if v else float("nan")


print("-" * 100)
print(f"  median delta-absmax inflation : x2 nearest {m('x2'):.2f}x | x2 bilinear {m('x2b'):.2f}x "
      f"| x4 nearest {m('x4'):.2f}x")
print(f"  median MoDiff gain (act/delta): full-res {m('gain_full'):.2f}  ->  "
      f"x2 {m('gain2'):.2f}  ->  x4 {m('gain4'):.2f}")
print(f"\n  traffic prize: a_hat read+write is 8 of the apply kernel's 10 sectors/warp (C17);")
print(f"    f=2 -> a_hat traffic /4  => 10 sectors -> 4  (-60% of the kernel's bytes)")
print(f"    f=4 -> a_hat traffic /16 => 10 sectors -> 2.5")
print(f"\n  CAVEAT: single-step screen. The real scheme rebuilds a_hat FROM the downsampled copy every")
print(f"  step, so coarsening compounds over the schedule -- these are the OPTIMISTIC numbers.")
