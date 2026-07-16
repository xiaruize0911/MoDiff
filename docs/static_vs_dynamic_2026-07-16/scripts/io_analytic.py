"""Analytical total DRAM IO per DDIM step (churches LDM UNet, batch 32), DYNAMIC vs STATIC.

Dynamic quantization computes every activation statistic at runtime, which costs extra DRAM reads:
  - conv/linear: one extra full read of the activation to compute its absmax scale,
  - attention Q/K: an extra read of Q,K for the per-token absmax; V: an extra full read for the
    per-channel absmax; softmax: a SECOND read of the T*T score matrix (2-pass max+exp).
Static reads calibrated constants instead -> those extra passes vanish (1 conv/linear read, 1 V
read, 1-pass softmax). This models the LOGICAL DRAM bytes; small per-row re-reads are partly
L2-absorbed in practice, so treat ncu dram__bytes.sum (ncu_profile.py) as the measured truth. The
analytic model shows the *shape* of the static saving. Emits pipeline_io_analytic.csv."""
import os, sys, csv, importlib.util
import torch, torch.nn as nn
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
OUT = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"
MiB = 1024**2

class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 32; args.steps = 2
args.linear_backend = "fp16"; args.calibration = None
N = args.batch_size

runner, model, sampler = abb.build("dynamic_fp16", args)
unet = model.model.diffusion_model
convs, attns = [], []
def conv_hook(m):
    def h(mod, inp):
        x = inp[0]; H, W = x.shape[2], x.shape[3]; st = mod.stride[0]
        convs.append((mod.in_channels, mod.out_channels, mod.kernel_size[0], H, W, H // st, W // st))
    return h
def attn_hook(m):
    def h(mod, inp):
        x = inp[0]; C = mod.channels; T = x.shape[2] * x.shape[3]
        attns.append((C, T, mod.num_heads, C // mod.num_heads))
    return h
hs = []
for m in unet.modules():
    if isinstance(m, nn.Conv2d): hs.append(m.register_forward_pre_hook(conv_hook(m)))
    elif isinstance(m, TokenMajorAttentionBlock): hs.append(m.register_forward_pre_hook(attn_hook(m)))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    x0 = torch.randn(N, 4, 32, 32, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    unet(x0, torch.randint(0, 1000, (N,), device="cuda").long(), None)
for h in hs: h.remove()

def conv_io(e, dyn):                 # in + weight at e; output fp16 (2B); dynamic: +1 activation read (absmax)
    return sum(N * Cin * H * W * e * (2 if dyn else 1) + Cout * Cin * k * k * e + N * Cout * Ho * Wo * 2.0
               for (Cin, Cout, k, H, W, Ho, Wo) in convs)
def lin_io(e, dyn):                  # qkv (C->3C) + proj (C->C); dynamic: +1 activation read for absmax
    return sum((N * T * C * e * (2 if dyn else 1) + C * 3 * C * e + N * T * 3 * C * e)
               + (N * T * C * e * (2 if dyn else 1) + C * C * e + N * T * C * e)
               for (C, T, nh, hd) in attns)
def attn_io(eq, dyn):                # materialized attention; dynamic adds absmax reads + 2nd score read
    tot = 0.0
    for (C, T, nh, hd) in attns:
        BH = N * nh
        qkv_read = 3 * BH * T * hd * 2 * (2 if dyn else 1)     # dyn: reread Q,K,V for absmax
        s_write = BH * T * T * 2                                # fp16 scores
        s_read = BH * T * T * 2 * (2 if dyn else 1)             # dyn softmax: max-pass + exp-pass
        p_rw = 2 * BH * T * T * eq                              # write P + read P in AV
        o_write = BH * T * hd * 2
        tot += qkv_read + s_write + s_read + p_rw + o_write
    return tot

# (precision, e_conv, e_lin, e_qkv_P). fp16 conv/linear are NOT quantized -> no absmax, so fp16
# dynamic==static for conv/linear; only softmax differs. int8/int4 differ on every layer.
PREC = [("fp16", 2.0, 2.0, 2.0), ("int8", 1.0, 1.0, 1.0), ("int4", 0.5, 0.5, 0.5)]
rows = []
for prec, ec, el, eq in PREC:
    for dyn in (True, False):
        # fp16 conv/linear have no runtime absmax regardless of dyn -> force dyn=False for those
        cdyn = dyn if prec != "fp16" else False
        c = conv_io(ec, cdyn) / MiB; l = lin_io(el, cdyn) / MiB; a = attn_io(eq, dyn) / MiB
        rows.append({"precision": prec, "variant": "dynamic" if dyn else "static",
                     "conv_MiB": round(c, 1), "linear_MiB": round(l, 1), "attn_MiB": round(a, 1),
                     "total_MiB": round(c + l + a, 1)})
        print(f"{prec:5s} {'dyn' if dyn else 'sta':>3}: conv={c:7.0f} lin={l:6.0f} attn={a:8.0f} total={c+l+a:8.0f} MiB/step")
with open(f"{OUT}/pipeline_io_analytic.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["precision", "variant", "conv_MiB", "linear_MiB", "attn_MiB", "total_MiB"])
    w.writeheader(); w.writerows(rows)
print(f"\n{len(convs)} convs, {len(attns)} attn blocks -> WROTE pipeline_io_analytic.csv")
