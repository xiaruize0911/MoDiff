"""Probe why along-C B=32 int8 a_hat is rainbow (relL2 2.2)."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
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
preflight(*MODEL, what="probe_block32.py")

import torch
import torch.nn.functional as F
import modiff_cutlass as mc
import integration.benchmarks.benchmark_ldm as B
from integration.kernels.int8_optimized import OptimizedInt8Conv2d

print("=== layout ===")
for dt in (torch.float16, torch.int8):
    z = torch.zeros(2, 192, 8, 8, dtype=dt, device="cuda")
    cl = z.contiguous(memory_format=torch.channels_last)
    print(f"  zeros {dt} stride={tuple(cl.stride())} cl={cl.is_contiguous(memory_format=torch.channels_last)}")
    x = torch.randn(2, 192, 8, 8, device="cuda").contiguous(memory_format=torch.channels_last).to(dt)
    q = x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    print(f"  pack-style {dt} stride={tuple(q.stride())} cl={q.is_contiguous(memory_format=torch.channels_last)}")

print("\n=== isolated step1 + GN ===")
torch.manual_seed(0)
N, C, H, W, Bsz = 2, 192, 8, 8, 32
x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
# pack like Python
xf = x.permute(0, 2, 3, 1).contiguous().float()
g = C // Bsz
blk = xf.view(N, H, W, g, Bsz)
amax = blk.abs().amax(-1).clamp_min(1e-12)
scale = (amax / 127.0).contiguous()
qi = (blk / scale.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
q = qi.view(N, H, W, C).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
print(f"  packed stride={tuple(q.stride())} scale={tuple(scale.shape)} scale_mean={scale.mean().item():.5f}")
recon = (qi.float() * scale.unsqueeze(-1)).reshape(N, H, W, C).permute(0, 3, 1, 2)
print(f"  pack relL2 vs x {(recon.cuda() - x.float()).norm() / x.float().norm():.6f}")

dscale = torch.tensor([10.0], device="cuda")  # large => tiny residual codes if x≈ahat
empty = torch.empty(0, device="cuda", dtype=torch.float32)
yq = mc.step1_static_quantize_fprop(x, q, dscale, empty, False, True, scale)
print(f"  step1 yq absmax={yq.abs().max().item()} meanabs={yq.abs().float().mean().item():.4f}")

# GN: x slightly different from packed a_hat; weight/bias identity-ish
w = torch.ones(C, device="cuda", dtype=torch.float16)
b = torch.zeros(C, device="cuda", dtype=torch.float16)
ms = torch.empty(0, device="cuda", dtype=torch.float16)
# make a_hat ≈ silu(gn(x)) by packing x itself (GN of randn isn't identity)
yq2 = mc.group_norm_silu_delta_quantize_nhwc(
    x, w, b, q.clone(), 32, 1e-5, True, dscale, empty, ms, ms,
    empty, empty, empty, empty, 127.0, False, 1.0, False, True, scale)
print(f"  GN    yq absmax={yq2.abs().max().item()} meanabs={yq2.abs().float().mean().item():.4f}")

print("\n=== model 2-step inspect ===")
runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/ahat_blockwise_2026-09-01/tmp_probe",
    batch_size=2, steps=2, shape=(4, 32, 32),
    calibration_path=B._default_calibration_path("int8"),
    auto_delta_table=True)
model, sampler = runner._setup_model("int8")
unet = model.model.diffusion_model

calls = {"gn": 0, "step1": 0, "s1silu": 0, "resize": 0, "up": 0, "bad_scale": []}
orig_gn = mc.group_norm_silu_delta_quantize_nhwc
orig_s1 = mc.step1_static_quantize_fprop
orig_s1s = mc.step1_static_quantize_fprop_silu
orig_rs = mc.group_norm_silu_delta_quantize_resize_nhwc
orig_up = mc.upsample2x_quantize_noahat_fprop

def _chk(tag, cache, scale_t, narg):
    if cache.dtype != torch.int8:
        calls["bad_scale"].append(f"{tag} cache {cache.dtype} nargs={narg}")
        return
    if not isinstance(scale_t, torch.Tensor) or scale_t.dim() != 4:
        calls["bad_scale"].append(
            f"{tag} scale dim={getattr(scale_t,'dim',lambda:None)()} "
            f"shape={getattr(scale_t,'shape',None)} nargs={narg}")

def wrap_gn(*a, **k):
    calls["gn"] += 1
    _chk("gn", a[3], a[-1], len(a))
    return orig_gn(*a, **k)
def wrap_s1(*a, **k):
    calls["step1"] += 1
    _chk("s1", a[1], a[-1], len(a))
    return orig_s1(*a, **k)
def wrap_s1s(*a, **k):
    calls["s1silu"] += 1
    _chk("s1s", a[1], a[-1], len(a))
    return orig_s1s(*a, **k)
def wrap_rs(*a, **k):
    calls["resize"] += 1
    _chk("rs", a[13], a[-1], len(a))
    return orig_rs(*a, **k)
def wrap_up(*a, **k):
    calls["up"] += 1
    _chk("up", a[3], a[-1], len(a))
    return orig_up(*a, **k)

mc.group_norm_silu_delta_quantize_nhwc = wrap_gn
mc.step1_static_quantize_fprop = wrap_s1
mc.step1_static_quantize_fprop_silu = wrap_s1s
mc.group_norm_silu_delta_quantize_resize_nhwc = wrap_rs
mc.upsample2x_quantize_noahat_fprop = wrap_up

B.reset_modiff_state_int8(unet)
B._reset_wxax_modiff_safe(model)
torch.manual_seed(0)
with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
    sampler.sample(S=2, batch_size=2, shape=(4, 32, 32), eta=0.0, verbose=False)

print(f"  kernel calls gn={calls['gn']} step1={calls['step1']} s1silu={calls['s1silu']} "
      f"resize={calls['resize']} up={calls['up']}")
print(f"  bad_scale n={len(calls['bad_scale'])} sample={calls['bad_scale'][:8]}")

print("\n=== per-layer a_hat after 2 steps ===")
rows = []
for name, m in unet.named_modules():
    if not isinstance(m, OptimizedInt8Conv2d):
        continue
    a = m.a_hat_cache
    s = getattr(m, "_ahat_qscale", None)
    cin = m.in_channels
    want = m._ahat_want_int8()
    info = dict(name=name, C=cin, cal=m.is_calibrated, want=want,
                adt=None if a is None else str(a.dtype).replace("torch.", ""),
                ashape=None if a is None else tuple(a.shape),
                astride=None if a is None else tuple(a.stride()),
                sdim=None if s is None else s.dim(),
                sshape=None if s is None else tuple(s.shape),
                smean=None if s is None or s.numel()==0 else float(s.float().mean()),
                amax=None if a is None else float(a.abs().float().max()),
                cl=None if a is None else a.is_contiguous(memory_format=torch.channels_last))
    rows.append(info)

from collections import Counter
print("  n_conv", len(rows))
print("  dtype", Counter(r["adt"] for r in rows))
print("  sdim", Counter(r["sdim"] for r in rows))
print("  cl", Counter(r["cl"] for r in rows))
print("  C%32", Counter(r["C"] % 32 for r in rows))
print("  want", Counter(r["want"] for r in rows))
bad = [r for r in rows if r["adt"]=="int8" and r["sdim"]!=4]
print("  int8 without 4D scale", len(bad))
for r in bad[:6]:
    print("   ", r["name"], r)
notcl = [r for r in rows if r["adt"]=="int8" and not r["cl"]]
print("  int8 not channels_last", len(notcl))
for r in notcl[:6]:
    print("   ", r["name"], r["ashape"], r["astride"])
c4 = [r for r in rows if r["C"] % 32 != 0]
print("  C not div 32", [(r["name"], r["C"], r["adt"], r["cal"], r["want"]) for r in c4])
# show a few typical
for r in rows[:4]:
    print("  ", r["name"], "C", r["C"], r["adt"], "scale", r["sshape"], "smean", r["smean"],
          "amax", r["amax"], "stride", r["astride"])
