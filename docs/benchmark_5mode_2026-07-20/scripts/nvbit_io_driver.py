"""NVBit driver for measured per-kernel/per-shape DRAM read/write bytes (via tools/mem_bytes).
Config registry: each (family, mode, shape). `--list` prints all tags. `--one <tag>` builds that
config, does setup/warmup OUTSIDE the profiler range, then runs the op once inside
cudaProfilerStart/Stop — so with the mem_bytes tool (ACTIVE_FROM_START=0) MEMBYTES_TOTAL = that
config's DRAM bytes. Run one config per process (run_nvbit_io.sh) for unambiguous mapping.
"""
import os, sys, math, copy
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int4_optimized import OptimizedInt4Conv2d
from integration.kernels.wxax_linear import QuantLinearWxAx

torch.manual_seed(0); dev = "cuda"; B = 128
CONV = [("res_128_64", 128, 64, 64, 128, 3, 1, 1), ("res_256_32", 256, 32, 32, 256, 3, 1, 1),
        ("down_256_512_16", 256, 16, 16, 512, 3, 1, 1), ("mid_512_8", 512, 8, 8, 512, 3, 1, 1),
        ("up_512_256_16", 512, 16, 16, 256, 3, 1, 1)]
LIN = [(192, 1024), (384, 256), (384, 64), (768, 16)]
ATTN = [(192, 8, 24, 1024), (384, 8, 48, 256), (384, 8, 48, 64), (768, 8, 96, 16)]


def conv_setup(mode, Cin, H, W, Cout, K, st, pad):
    # mode: fp16 | int8_baseline | int4_baseline | int8_modiff | int4_modiff.
    # baseline = no temporal cache (enable_modiff False); modiff = a_hat/o_hat delta cache (True).
    x = torch.randn(B, Cin, H, W, device=dev, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    conv = nn.Conv2d(Cin, Cout, K, stride=st, padding=pad, bias=True).cuda().eval()
    if mode == "fp16":
        cf = copy.deepcopy(conv).half().to(memory_format=torch.channels_last)
        return lambda: cf(x)
    Wrap = OptimizedInt8Conv2d if "int8" in mode else OptimizedInt4Conv2d
    opt = Wrap(conv, layer_name="io").cuda().eval()
    opt.set_static_scale(32.0); opt.set_standard_output_fp16(True)
    opt.enable_modiff("modiff" in mode)
    opt(x); opt(x); torch.cuda.synchronize()         # 2 warmups OUTSIDE range: lazy caches + leave
    return lambda: opt(x)                             # modiff's is_first_step (-> steady-state delta)


def lin_setup(mode, C, T, kind):
    M = B * T; K = C; Nout = 3 * C if kind == "qkv" else C
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    lin = nn.Linear(K, Nout).to(dev).half()
    if mode == "fp16":
        return lambda: F.linear(x, lin.weight, lin.bias)
    bits = 8 if "int8" in mode else 4     # linear has no modiff variant -> baseline label only
    ql = QuantLinearWxAx(lin, bits).to(dev); a = x.abs().max().item() / (127.0 if bits == 8 else 7.0); ql.set_a_scale(a)
    xp = F.pad(x, (0, ql._awqt_K - K)).contiguous() if ql._awqt_K != K else x
    if bits == 8:
        xq = mc.quantize_act_int8(xp, a)
        return (lambda: mc.gemm_w8a8_awq_nout(xq, ql.qweight, ql.w_scale, a, Nout)) if ql._awqt_N != Nout \
            else (lambda: mc.gemm_w8a8_awq(xq, ql.qweight, ql.w_scale, a))
    xq = mc.quantize_act_int4_pack(xp, a)
    return (lambda: mc.gemm_w4a4_awq_nout(xq, ql.qweight, ql.w_scale, a, ql._awqt_K, Nout)) if ql._awqt_N != Nout \
        else (lambda: mc.gemm_w4a4_awq(xq, ql.qweight, ql.w_scale, a, ql._awqt_K))


def attn_setup(mode, C, nh, hd, T):
    N = B; H = nh; BH = N * H; sc = 1.0 / math.sqrt(hd)
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    if mode == "fp16":
        q4 = q.reshape(1, BH, T, hd); k4 = k.reshape(1, BH, T, hd); v4 = v.reshape(1, BH, T, hd)
        def f():
            with sdpa_kernel(SDPBackend.MATH): return F.scaled_dot_product_attention(q4, k4, v4, scale=sc)
        return f
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    if "int8" in mode:                    # attention has no modiff variant -> baseline label only
        hp = ((hd + 31) // 32) * 32
        qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(qm, km, vm, hp, hp, 8)
        qi = qi.view(N, H, T, hp); ki = ki.view(N, H, T, hp); vt = vt.view(N, H, hp, T)
        sq = sq.view(N, H, T).contiguous(); sk = sk.view(N, H, T).contiguous(); sv = sv[..., :hd].contiguous().view(N, H, hd)
        return lambda: mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc)
    hdp4, hdpv = 64, ((hd + 31) // 32) * 32
    q4, k4, vt4, sq4, sk4, sv4 = mc.quantize_attn_qkv_i4qk_i8v(qm, km, vm, hdp4, hdpv)
    q4 = q4.view(N, H, T, -1); k4 = k4.view(N, H, T, -1); vt4 = vt4.view(N, H, hdpv, T)
    sq4 = sq4.view(N, H, T).contiguous(); sk4 = sk4.view(N, H, T).contiguous(); sv4 = sv4[..., :hd].contiguous().view(N, H, hd)
    return lambda: mc.flash_attn_int4_vt(q4, k4, vt4, sq4, sk4, sv4, hdp4, sc)


REG = {}   # tag -> (setup callable)
# conv: 5 modes — baseline vs modiff differ (modiff adds a_hat/o_hat cache traffic)
for (nm, Cin, Hh, Ww, Co, K, st, pad) in CONV:
    for m in ("fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"):
        REG[f"conv|{m}|{nm}"] = (lambda m=m, Cin=Cin, Hh=Hh, Ww=Ww, Co=Co, K=K, st=st, pad=pad: conv_setup(m, Cin, Hh, Ww, Co, K, st, pad))
# linear/attention: no modiff variant (baseline == modiff) -> baseline label only
for (C, T) in LIN:
    for kind in ("qkv", "proj"):
        for m in ("fp16", "int8_baseline", "int4_baseline"):
            REG[f"linear|{m}|{kind}_{C}_{3*C if kind=='qkv' else C}_M{B*T}"] = (lambda m=m, C=C, T=T, kind=kind: lin_setup(m, C, T, kind))
for (C, nh, hd, T) in ATTN:
    modes = ("fp16", "int8_baseline", "int4_baseline") if (hd <= 48 and T % 64 == 0) else ("fp16",)
    for m in modes:
        REG[f"attn|{m}|hd{hd}_T{T}"] = (lambda m=m, C=C, nh=nh, hd=hd, T=T: attn_setup(m, C, nh, hd, T))

arg = sys.argv[1] if len(sys.argv) > 1 else "--list"
if arg == "--list":
    for t in REG: print(t)
elif arg == "--one":
    tag = sys.argv[2]
    launch = REG[tag]()                    # build + setup/warmup OUTSIDE the profiled range
    launch(); torch.cuda.synchronize()     # one more warmup, still outside range
    torch.cuda.profiler.start()
    launch()                               # the single measured launch
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"NVBIT_ONE_DONE {tag}")
