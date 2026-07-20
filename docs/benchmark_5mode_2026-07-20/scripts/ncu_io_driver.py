"""ncu harness driver — launches each (family, mode, shape) kernel ONCE, wrapped in an NVTX range
tagged `family|mode|shape`, so Nsight Compute can measure real per-kernel DRAM read/write bytes and
map each profiled kernel back to its shape. No warmup: ncu replays each kernel internally for accurate
counters. Run under ncu via run_ncu_io.sh (NOT a speed benchmark).

Requires unlocked GPU perf counters (RmProfilingAdminOnly=0 or CAP_SYS_ADMIN). Standalone (no ncu) it
just launches everything once and prints the config list — used to validate the driver is correct.

Usage: python ncu_io_driver.py [conv|linear|attn|all]   (default all)
"""
import os, sys, math, copy
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int4_optimized import OptimizedInt4Conv2d
from integration.kernels.wxax_linear import QuantLinearWxAx

torch.manual_seed(0); dev = "cuda"
B = 128
FAM = sys.argv[1] if len(sys.argv) > 1 else "all"
rng = torch.cuda.nvtx.range

CONV_SHAPES = [  # (name, Cin, H, W, Cout, K, stride, pad)
    ("res_128_64", 128, 64, 64, 128, 3, 1, 1), ("res_256_32", 256, 32, 32, 256, 3, 1, 1),
    ("down_256_512_16", 256, 16, 16, 512, 3, 1, 1), ("mid_512_8", 512, 8, 8, 512, 3, 1, 1),
    ("up_512_256_16", 512, 16, 16, 256, 3, 1, 1),
]
LIN_SHAPES = [(192, 1024), (384, 256), (384, 64), (768, 16)]        # (C, T) -> qkv C->3C, proj C->C ; M=B*T
ATTN_SHAPES = [(192, 8, 24, 1024), (384, 8, 48, 256), (384, 8, 48, 64), (768, 8, 96, 16)]  # (C,nh,hd,T)

configs = []   # (family, mode, shape_tag) in launch order


def tag(family, mode, shape):
    configs.append((family, mode, shape)); return f"{family}|{mode}|{shape}"


# ---------------- CONV ----------------
def run_conv():
    for (name, Cin, H, W, Cout, K, st, pad) in CONV_SHAPES:
        x = torch.randn(B, Cin, H, W, device=dev, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        conv = nn.Conv2d(Cin, Cout, K, stride=st, padding=pad, bias=True).cuda().eval()   # fp32 master
        # fp16 uses a SEPARATE halved copy so the fp32 master stays intact for the int wrappers
        cf = copy.deepcopy(conv).half().to(memory_format=torch.channels_last)
        with rng(tag("conv", "fp16", name)): cf(x)
        for mode, Wrap in (("int8", OptimizedInt8Conv2d), ("int4", OptimizedInt4Conv2d)):
            opt = Wrap(conv, layer_name="io").cuda().eval()
            opt.set_static_scale(32.0); opt.set_standard_output_fp16(True); opt.enable_modiff(False)
            opt(x); torch.cuda.synchronize()                      # 1 untagged setup call (lazy caches)
            with rng(tag("conv", mode, name)): opt(x)
        torch.cuda.synchronize()


# ---------------- LINEAR (qkv/proj GEMM) ----------------
def run_linear():
    for (C, T) in LIN_SHAPES:
        M = B * T
        for kind, K, Nout in (("qkv", C, 3 * C), ("proj", C, C)):
            x = torch.randn(M, K, device=dev, dtype=torch.float16)
            lin = nn.Linear(K, Nout).to(dev).half()
            sh = f"{kind}_{K}-{Nout}_M{M}"
            with rng(tag("linear", "fp16", sh)): F.linear(x, lin.weight, lin.bias)
            for bits in (8, 4):
                ql = QuantLinearWxAx(lin, bits).to(dev)
                a = x.abs().max().item() / (127.0 if bits == 8 else 7.0); ql.set_a_scale(a)
                xp = F.pad(x, (0, ql._awqt_K - K)).contiguous() if ql._awqt_K != K else x
                if bits == 8:
                    xq = mc.quantize_act_int8(xp, a)
                    g = (lambda: mc.gemm_w8a8_awq_nout(xq, ql.qweight, ql.w_scale, a, Nout)) if ql._awqt_N != Nout \
                        else (lambda: mc.gemm_w8a8_awq(xq, ql.qweight, ql.w_scale, a))
                else:
                    xq = mc.quantize_act_int4_pack(xp, a)
                    g = (lambda: mc.gemm_w4a4_awq_nout(xq, ql.qweight, ql.w_scale, a, ql._awqt_K, Nout)) if ql._awqt_N != Nout \
                        else (lambda: mc.gemm_w4a4_awq(xq, ql.qweight, ql.w_scale, a, ql._awqt_K))
                with rng(tag("linear", f"int{bits}", sh)): g()
            torch.cuda.synchronize()


# ---------------- ATTENTION (fp16 SDPA / int8 flash / int4 flash) ----------------
def run_attn():
    for (C, nh, hd, T) in ATTN_SHAPES:
        N = B; H = nh; BH = N * H; sc = 1.0 / math.sqrt(hd)
        q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
        q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
        sh = f"hd{hd}_T{T}"
        with rng(tag("attn", "fp16", sh)):
            with sdpa_kernel(SDPBackend.MATH): F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc)
        if hd <= 48 and T % 64 == 0:                             # flash-eligible
            qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
            hd_pad = ((hd + 31) // 32) * 32
            qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(qm, km, vm, hd_pad, hd_pad, 8)
            qi = qi.view(N, H, T, hd_pad); ki = ki.view(N, H, T, hd_pad); vt = vt.view(N, H, hd_pad, T)
            sq = sq.view(N, H, T).contiguous(); sk = sk.view(N, H, T).contiguous(); sv = sv[..., :hd].contiguous().view(N, H, hd)
            with rng(tag("attn", "int8", sh)): mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc)
            hdp4, hdpv = 64, ((hd + 31) // 32) * 32
            q4, k4, vt4, sq4, sk4, sv4 = mc.quantize_attn_qkv_i4qk_i8v(qm, km, vm, hdp4, hdpv)
            q4 = q4.view(N, H, T, -1); k4 = k4.view(N, H, T, -1); vt4 = vt4.view(N, H, hdpv, T)
            sq4 = sq4.view(N, H, T).contiguous(); sk4 = sk4.view(N, H, T).contiguous(); sv4 = sv4[..., :hd].contiguous().view(N, H, hd)
            with rng(tag("attn", "int4", sh)): mc.flash_attn_int4_vt(q4, k4, vt4, sq4, sk4, sv4, hdp4, sc)
        torch.cuda.synchronize()


if FAM in ("conv", "all"): run_conv()
if FAM in ("linear", "all"): run_linear()
if FAM in ("attn", "all"): run_attn()
torch.cuda.synchronize()
print(f"NCU_DRIVER_DONE family={FAM} launched {len(configs)} tagged configs:")
for i, c in enumerate(configs):
    print(f"  [{i}] {c[0]}|{c[1]}|{c[2]}")
