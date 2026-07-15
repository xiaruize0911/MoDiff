"""Analytical total DRAM IO usage per DDIM step (churches LDM UNet, batch 32).

Allocator traffic (torch.cuda.memory_stats) is a poor "IO usage" proxy: the
quantized convs store fp16 outputs and the whole attention path is fp16 in every
mode, so measured allocator bytes barely move with dtype and wrongly suggest
int8/int4 use ~the same IO as fp16.

Instead we sum, over every conv / attention-linear / attention-SDPA op in the UNet,
the analytical DRAM bytes moved = (bytes_in + bytes_weight + bytes_out), using each
op's REAL operand dtype (same model as the kernel-IO section §2). Shapes are
mode-independent, so we collect them once on the fp16 build and apply the per-
precision byte model. Emits pipeline_io_analytic.csv with a conv/linear/attn split
so the quantization effect (conv operands shrink; fp16 outputs + fp16 attention
stay put) is explicit.

Note: convs keep fp16 outputs (standard_output_fp16=True), and only conv operands
are quantized -- attention QKV/SDPA run in fp16/fp32 in all modes. So int8/int4
shrink the conv bytes (~0.64x / 0.45x) but not the dtype-invariant attention score
traffic, which dominates the total. This mirrors the Amdahl bound on speed (§4)."""
import os, sys, csv, importlib.util
import torch, torch.nn as nn
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
OUT = "/workspace/MoDiff/docs/comprehensive_benchmark_2026-07-15/data"
MiB = 1024**2

class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 32; args.steps = 2
args.linear_backend = "fp16"; args.calibration = None
N = args.batch_size

runner, model, sampler = abb.build("fp16", args)
unet = model.model.diffusion_model
convs, attns = [], []
def conv_hook(m):
    def h(mod, inp):
        x = inp[0]; H, W = x.shape[2], x.shape[3]; st = mod.stride[0]
        convs.append((mod.in_channels, mod.out_channels, mod.kernel_size[0], H, W, H//st, W//st))
    return h
def attn_hook(m):
    def h(mod, inp):
        x = inp[0]; C = mod.channels; T = x.shape[2]*x.shape[3]
        attns.append((C, T, mod.num_heads, C//mod.num_heads))
    return h
hs = []
for m in unet.modules():
    if isinstance(m, nn.Conv2d): hs.append(m.register_forward_pre_hook(conv_hook(m)))
    elif isinstance(m, TokenMajorAttentionBlock): hs.append(m.register_forward_pre_hook(attn_hook(m)))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    x0 = torch.randn(N, 4, 32, 32, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    unet(x0, torch.randint(0, 1000, (N,), device="cuda").long(), None)
for h in hs: h.remove()

def conv_io(e_operand):          # in + weight at e_operand bytes; output always fp16 (2B)
    return sum(N*Cin*H*W*e_operand + Cout*Cin*k*k*e_operand + N*Cout*Ho*Wo*2.0
               for (Cin, Cout, k, H, W, Ho, Wo) in convs)
def lin_io(e):                    # qkv (C->3C) + proj (C->C) per attn block
    return sum((N*T*C*e + C*3*C*e + N*T*3*C*e) + (N*T*C*e + C*C*e + N*T*C*e)
               for (C, T, nh, hd) in attns)
def attn_io(e):                   # math SDPA: read Q,K,V + write&read scores[N,h,T,T] + write out
    return sum(3*N*nh*T*hd*e + 2*(N*nh*T*T*e) + N*nh*T*hd*e for (C, T, nh, hd) in attns)

# (precision, conv-operand bytes/elem, everything-else bytes/elem). Attention + the
# qkv/proj linears run in fp16 for fp16/int8/int4 (int linear unused), fp32 for fp32.
PREC = [("fp32", 4.0, 4.0), ("fp16", 2.0, 2.0), ("int8", 1.0, 2.0), ("int4", 0.5, 2.0)]
rows = []
for prec, e_conv, e_rest in PREC:
    c = conv_io(e_conv)/MiB; l = lin_io(e_rest)/MiB; a = attn_io(e_rest)/MiB
    rows.append({"precision": prec, "conv_MiB_step": round(c, 1), "linear_MiB_step": round(l, 1),
                 "attn_MiB_step": round(a, 1), "total_MiB_step": round(c+l+a, 1)})
    print(f"{prec:5s} conv={c:7.0f} linear={l:7.0f} attn={a:8.0f} total={c+l+a:8.0f} MiB/step", flush=True)

with open(f"{OUT}/pipeline_io_analytic.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["precision", "conv_MiB_step", "linear_MiB_step", "attn_MiB_step", "total_MiB_step"])
    w.writeheader(); w.writerows(rows)
print(f"\n{len(convs)} convs, {len(attns)} attn blocks -> WROTE pipeline_io_analytic.csv")
