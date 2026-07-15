"""Kernel-level speed + IO micro-benchmarks for the churches LDM UNet operators:
conv (fp16 cuDNN / int8 / int4, base + MoDiff), linear (qkv/proj: fp16 / int8 / int4),
attention (fused GN->qkv CUTLASS vs GroupNorm+cuBLAS, + flash SDPA).
IO = analytical bytes moved (read in + read weight + write out) / measured time -> effective GB/s.
Emits kernel_speed.csv and kernel_io.csv."""
import os, sys, csv, importlib.util
import torch, torch.nn as nn, torch.nn.functional as F
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
from integration.kernels.int8_optimized import OptimizedInt8Conv2d, reset_modiff_state as reset_i8
from integration.kernels.int4_optimized import OptimizedInt4Conv2d, reset_modiff_state as reset_i4
from integration.fused_ops.fused_resblock import _group_norm_silu
import modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
OUT = "/workspace/MoDiff/docs/comprehensive_benchmark_2026-07-15/data"
torch.backends.cudnn.benchmark = True
PEAK_BW = 696e9; N = 32

def bench(fn, it=60, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it / 1e3   # seconds

def gbps(nbytes, t_s): return nbytes / t_s / 1e9

# ---- collect real conv + attn shapes from a forward pass ----
class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = N; args.steps = 2
args.linear_backend = "fp16"; args.calibration = None
runner, model, sampler = abb.build("fp16", args)
unet = model.model.diffusion_model
conv_shapes = {}
def cpre(m):
    def h(mod, inp):
        x = inp[0]
        conv_shapes[(mod.in_channels, mod.out_channels, mod.kernel_size[0], mod.stride[0], x.shape[2], x.shape[3])] = \
            conv_shapes.get((mod.in_channels, mod.out_channels, mod.kernel_size[0], mod.stride[0], x.shape[2], x.shape[3]), 0) + 1
    return h
hs = [m.register_forward_pre_hook(cpre(m)) for m in unet.modules() if isinstance(m, nn.Conv2d)]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    x0 = torch.randn(N, 4, 32, 32, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    unet(x0, torch.randint(0, 1000, (N,), device="cuda").long(), None)
for h in hs: h.remove()
del runner, model, sampler; torch.cuda.empty_cache()

# ================= CONV kernels =================
conv_speed, conv_io = [], []
BE = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
# representative shapes get the MoDiff timing too. Pick the top-6 by actual FLOPs
# (count * Cout*Cin*k*k * Ho*Wo) -> the genuinely expensive convs, which are the
# large-spatial 3x3s. (Selecting by param-count instead biases toward tiny-spatial
# deep convs where int4's weight-unpack overhead makes it look slower than int8.)
def flops(k): return conv_shapes[k] * k[1] * k[0] * k[2] * k[2] * (k[4]//k[3]) * (k[5]//k[3])
shapes = sorted([k for k in conv_shapes if k[0] >= 8], key=lambda k: -flops(k))
modiff_set = set(shapes[:6])
print(f"{len(shapes)} conv shapes; MoDiff on top {len(modiff_set)}", flush=True)
for (Cin, Cout, k, st, H, W) in shapes:
    cnt = conv_shapes[(Cin, Cout, k, st, H, W)]; pad = k // 2
    Ho, Wo = H // st, W // st
    xf = torch.randn(N, Cin, H, W, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    fc = nn.Conv2d(Cin, Cout, k, stride=st, padding=pad, bias=False).cuda().half().to(memory_format=torch.channels_last)
    with torch.no_grad(): t16 = bench(lambda: fc(xf))
    # int8 base
    q8 = OptimizedInt8Conv2d(nn.Conv2d(Cin, Cout, k, stride=st, padding=pad, bias=False).cuda().float()).cuda().eval()
    q8.set_static_scale(4.0); q8.set_standard_output_fp16(True); q8.enable_modiff(False)
    xi = q8.quantize_input(xf); q8._ensure_tuned_config(xi, (N, Cout, Ho, Wo))
    with torch.no_grad(): t8 = bench(lambda: q8.forward_from_int8(xi))
    # int4 base
    t4 = float("nan")
    if Cin % 2 == 0:
        q4 = OptimizedInt4Conv2d(nn.Conv2d(Cin, Cout, k, stride=st, padding=pad, bias=False).cuda().float()).cuda().eval()
        q4.set_static_scale(4.0); q4.set_standard_output_fp16(True); q4.enable_modiff(False)
        xp = q4.quantize_input(xf); q4._ensure_tuned_config(xp)
        with torch.no_grad(): t4 = bench(lambda: q4.forward_from_int4(xp, H, W))
    row = {"Cin": Cin, "Cout": Cout, "k": k, "stride": st, "H": H, "W": W, "count": cnt,
           "fp16_us": round(t16*1e6, 2), "int8_us": round(t8*1e6, 2),
           "int4_us": round(t4*1e6, 2) if t4 == t4 else ""}
    io = {"Cin": Cin, "Cout": Cout, "k": k, "H": H, "W": W}
    for nm, t, ein in [("fp16", t16, 2.0), ("int8", t8, 1.0), ("int4", t4, 0.5)]:
        if t != t: continue
        nb = N*Cin*H*W*ein + Cout*Cin*k*k*ein + N*Cout*Ho*Wo*2.0
        io[f"{nm}_GBps"] = round(gbps(nb, t), 1)
    # MoDiff conv (steady-state modulated step) for representative shapes
    if (Cin, Cout, k, st, H, W) in modiff_set:
        for tag, cls, reset in [("int8", OptimizedInt8Conv2d, reset_i8), ("int4", OptimizedInt4Conv2d, reset_i4)]:
            if tag == "int4" and Cin % 2: continue
            qm = cls(nn.Conv2d(Cin, Cout, k, stride=st, padding=pad, bias=False).cuda().float()).cuda().eval()
            qm.set_calibrating(True); _ = qm(xf); qm.calibrating = False; qm._act_channel_max = None
            qm.end_calibration(); qm.set_standard_output_fp16(True); qm.enable_modiff(True)
            base = xf.clone()
            def mstep(i=[0]):
                i[0] += 1
                return qm(base + 0.01 * (1.0 / (i[0] + 1)) * torch.randn_like(base))
            try:
                reset(qm); _ = qm(base)                       # first step (warmup cache)
                tmod = bench(mstep, it=30, warm=10)
                row[f"{tag}_modiff_us"] = round(tmod*1e6, 2)
            except Exception as ex:
                row[f"{tag}_modiff_us"] = f"ERR"
    conv_speed.append(row); conv_io.append(io)
    print(f"  conv Cin{Cin} Cout{Cout} k{k} {H}x{W}: fp16 {t16*1e6:.1f} int8 {t8*1e6:.1f} int4 {(t4*1e6 if t4==t4 else 0):.1f} us", flush=True)

# ================= LINEAR kernels (qkv/proj) =================
from integration.kernels.int8_linear import OptimizedInt8Linear
lin_speed, lin_io = [], []
for (C, Cout, tag, T) in [(192, 576, "qkv", 1024), (192, 192, "proj", 1024),
                          (384, 1152, "qkv", 256), (384, 384, "proj", 256),
                          (768, 2304, "qkv", 16), (768, 768, "proj", 16)]:
    M = N * T
    x = torch.randn(M, C, device="cuda", dtype=torch.float16)
    lin = nn.Linear(C, Cout).cuda().half()
    with torch.no_grad(): t16 = bench(lambda: F.linear(x, lin.weight, lin.bias))
    q8 = OptimizedInt8Linear(lin, backend="int_gemm", int_gemm_min_m=1).cuda()
    q8.set_static_scale(127.0 / x.abs().max().item())
    with torch.no_grad(): t8 = bench(lambda: q8(x))
    r = {"role": tag, "C": C, "Cout": Cout, "M": M, "fp16_us": round(t16*1e6, 2), "int8_us": round(t8*1e6, 2)}
    lin_speed.append(r)
    io = {"role": tag, "C": C, "Cout": Cout, "M": M}
    for nm, t, ein in [("fp16", t16, 2.0), ("int8", t8, 1.0)]:
        nb = M*C*ein + C*Cout*ein + M*Cout*2.0
        io[f"{nm}_GBps"] = round(gbps(nb, t), 1)
    lin_io.append(io)
    print(f"  linear {tag} C{C}->{Cout} M{M}: fp16 {t16*1e6:.1f} int8 {t8*1e6:.1f} us", flush=True)

# ================= ATTENTION kernels =================
attn_rows = []
G, eps, SHIFT = 32, 1e-6, 16.0
for (C, H, W) in [(192, 32, 32), (384, 16, 16), (384, 8, 8), (768, 4, 4)]:
    T = H*W; M = N*T; Cg = C//G; nh = C // (C // 32 if C >= 32 else 1)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    qkv = nn.Linear(C, 3*C).cuda().half(); gn = nn.GroupNorm(G, C, eps=eps).cuda().half()
    gw, gb = gn.weight.detach().half(), gn.bias.detach().half()
    W16, b16 = qkv.weight.detach().half(), qkv.bias.detach().half()
    Wf = (W16 * gw[None, :]).contiguous(); conv_w = Wf.view(3*C, 1, 1, C).contiguous()
    epi = (b16 + W16 @ gb - SHIFT * Wf.sum(1)).contiguous().half()
    def base():
        xn = _group_norm_silu(x, G, gw, gb, eps, apply_silu=False)
        return F.linear(xn.permute(0, 2, 3, 1).reshape(M, C), W16, b16)
    fuseable = (T % 128 == 0)
    tb = bench(base)
    tf = bench(lambda: mc.fused_gn_qkv(x, conv_w, epi, G, eps, SHIFT)) if fuseable else float("nan")
    # SDPA on this shape: flash (removed from the model) vs math (now used).
    heads = 8; hd = C // heads
    q = torch.randn(N, heads, T, hd, device="cuda", dtype=torch.float16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION): tflash = bench(lambda: F.scaled_dot_product_attention(q, q, q))
    with sdpa_kernel(SDPBackend.MATH): tmath = bench(lambda: F.scaled_dot_product_attention(q, q, q))
    attn_rows.append({"C": C, "HxW": f"{H}x{W}", "T": T,
                      "gn+qkv_base_us": round(tb*1e6, 2),
                      "gn+qkv_fused_us": round(tf*1e6, 2) if tf == tf else "n/a(T%128)",
                      "fused_speedup": round(tb/tf, 3) if tf == tf else "",
                      "sdpa_flash_us": round(tflash*1e6, 2), "sdpa_math_us": round(tmath*1e6, 2),
                      "math/flash": round(tmath/tflash, 2)})
    print(f"  attn C{C} {H}x{W}: base {tb*1e6:.1f} fused {(tf*1e6 if tf==tf else 0):.1f} flash {tflash*1e6:.1f} math {tmath*1e6:.1f} us", flush=True)

def wcsv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
wcsv(f"{OUT}/kernel_conv_speed.csv", conv_speed)
wcsv(f"{OUT}/kernel_conv_io.csv", conv_io)
wcsv(f"{OUT}/kernel_linear_speed.csv", lin_speed)
wcsv(f"{OUT}/kernel_linear_io.csv", lin_io)
wcsv(f"{OUT}/kernel_attn.csv", attn_rows)
print("\nWROTE kernel_*.csv")
