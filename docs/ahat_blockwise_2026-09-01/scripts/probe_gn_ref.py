"""Compare first modulated GN call: kernel yq vs Python held-block reference."""
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
preflight(*MODEL, what="probe_gn_ref.py")

import torch
import torch.nn.functional as F
import modiff_cutlass as mc
import integration.benchmarks.benchmark_ldm as B

runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/ahat_blockwise_2026-09-01/tmp_probe",
    batch_size=2, steps=50, shape=(4, 32, 32),
    calibration_path=B._default_calibration_path("int8"),
    auto_delta_table=True)
model, sampler = runner._setup_model("int8")
unet = model.model.diffusion_model

orig = mc.group_norm_silu_delta_quantize_nhwc
dumps = []

def silu_gn(x, w, b, ng, eps, ms, sh, apply_silu=True):
    # x [N,C,H,W] channels_last
    n, c, h, ww = x.shape
    xt = x.float()
    xtn = F.group_norm(xt, int(ng), w.float(), b.float(), float(eps))
    if ms is not None and ms.numel() > 0:
        ms2 = ms.float().reshape(n, c, 1, 1)
        sh2 = sh.float().reshape(n, c, 1, 1)
        xtn = xtn * (1 + ms2) + sh2
    # kernel does fp16 round-trip before silu
    xtn = xtn.half().float()
    if apply_silu:
        xtn = xtn * torch.sigmoid(xtn)
    return xtn

def unpack(q, s):
    n, c, h, w = q.shape
    g = s.shape[-1]
    bsz = c // g
    codes = q.permute(0, 2, 3, 1).reshape(n, h, w, g, bsz).float()
    return (codes * s.unsqueeze(-1)).reshape(n, h, w, c).permute(0, 3, 1, 2)

def py_delta(out, q, s, dscale):
    cfp = unpack(q, s)
    d = out - cfp
    y = (d * float(dscale)).round().clamp(-127, 127)
    return y, d, cfp

def wrap(*a, **k):
    if len(dumps) >= 8:
        return orig(*a, **k)
    x, w, b, ah = a[0], a[1], a[2], a[3]
    ng, eps, apply_silu = a[4], a[5], a[6]
    dscale, smooth, ms, sh = a[7], a[8], a[9], a[10]
    write_ahat, ascale = a[-2], a[-1]
    ah0 = ah.clone()
    sc0 = ascale.clone()
    yq = orig(*a, **k)
    with torch.no_grad():
        out = silu_gn(x, w, b, ng, eps, ms, sh, apply_silu)
        if smooth is not None and smooth.numel() > 0:
            out = out * smooth.view(1, -1, 1, 1)
        ds = float(dscale.reshape(-1)[0].item())
        yref, d, cfp = py_delta(out, ah0, sc0, ds)
        yq_f = yq.float()
        rec = unpack(ah, sc0)  # AFTER kernel write, held scales unchanged
        dumps.append({
            "ng": int(ng), "C": x.shape[1], "xsh": tuple(x.shape),
            "dscale": float(dscale.view(-1)[0]),
            "write": bool(write_ahat),
            "smooth": int(smooth.numel()),
            "yq_absmax": float(yq.abs().max()),
            "yq_meanabs": float(yq.abs().float().mean()),
            "yref_absmax": float(yref.abs().max()),
            "yq_vs_yref_max": float((yq_f - yref).abs().max()),
            "yq_vs_yref_rel": float((yq_f - yref).norm() / (yref.norm() + 1e-8)),
            "c_vs_out_rel": float((cfp - out).norm() / (out.norm() + 1e-8)),
            "out_absmax": float(out.abs().max()),
            "c_absmax": float(cfp.abs().max()),
            "d_absmax": float(d.abs().max()),
            "s_mean": float(ascale.mean()),
            "s_max": float(ascale.max()),
            "code_absmax_after": float(ah.abs().max()),
            "rec_after_vs_out_rel": float((rec - out).norm() / (out.norm() + 1e-8)),
        })
    return yq

mc.group_norm_silu_delta_quantize_nhwc = wrap

B.reset_modiff_state_int8(unet)
B._reset_wxax_modiff_safe(model)
torch.manual_seed(0)
with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
    sampler.sample(S=50, batch_size=2, shape=(4, 32, 32), eta=0.0, verbose=False)

for i, d in enumerate(dumps):
    print(f"\n--- GN call {i} ---")
    for k, v in d.items():
        print(f"  {k}: {v}")
