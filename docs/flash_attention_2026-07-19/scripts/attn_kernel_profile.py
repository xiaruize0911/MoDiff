"""Kernel-level profile of the ATTENTION path in isolation (no conv/GN): where does the time go
inside fp16-MATH vs int8-materialized vs int4-materialized attention, per churches shape, b128.
Uses torch.profiler; reports per-kernel us/call."""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile, ProfilerActivity
B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0); dev = "cuda"

def short(n):
    n = n.replace("void ", "").replace("(anonymous namespace)::", "").replace("at::native::", "")
    return n[:52]

def prof(fn, reps=40):
    for _ in range(15): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps): fn()
    torch.cuda.synchronize()
    ks = [(e.self_device_time_total / reps, e.key) for e in p.key_averages() if e.self_device_time_total > 0]
    ks.sort(reverse=True); return ks

_bn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(40): _bn = _bn @ _bn * 1e-4 + 1.0
torch.cuda.synchronize()

SHAPES = [(192, 8, 24, 1024), (384, 8, 48, 256), (384, 8, 48, 64)]
for (C, H, hd, T) in SHAPES:
    N = B; BH = N * H; sc = 1.0 / math.sqrt(hd)
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64; hp4 = (hd + 63) // 64 * 64

    def fp16():
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc)
    qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
    def i8():
        qi2, ki2, vt2, sq2, sk2, sv2 = mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)  # include quantize
        S = mc.attn_qk_int8(qi2, ki2, sq2, sk2, sc); P, spp = mc.attn_softmax_requant(S); return mc.attn_av_int8(P, vt2, spp, sv2)
    qi4, ki4, vt4, sq4, sk4, sv4 = mc.quantize_attn_qkv(qm, km, vm, hp4, hpa, 4)
    def i4():
        a, b_, c_, d, e, f = mc.quantize_attn_qkv(qm, km, vm, hp4, hpa, 4)
        S = mc.attn_qk_int4(a, b_, hp4, d, e, sc); P, spp = mc.attn_softmax_requant4(S); return mc.attn_av_int4(P, c_, spp, f, T)

    print(f"\n################ hd{hd} / T{T}  (BH={BH}, b{B}) ################")
    for name, fn in [("fp16 MATH", fp16), ("int8 materialized", i8), ("int4 materialized", i4)]:
        ks = prof(fn); tot = sum(t for t, _ in ks)
        print(f"  --- {name}: {tot:.1f} us/call ---")
        for t, kn in ks:
            if t > tot * 0.02:
                print(f"      {t:7.1f} us  {100*t/tot:5.1f}%  {short(kn)}")
