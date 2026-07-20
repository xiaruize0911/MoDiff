"""Optimization harness for the fused int8/int4 flash attention kernels.
Kernel-only time + rel-L2 correctness at every flash-eligible churches shape,
vs fp16 MATH (current default) and fp16 flash (FA-2 reference). Rerun after each
kernel change to track progress. Usage: python flash_opt_bench.py [batch]
"""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
FL = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0); dev = "cuda"

def bench(fn, it=100, warm=30, reps=7):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]
def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)
_bn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _bn = _bn @ _bn * 1e-4 + 1.0
torch.cuda.synchronize()

def pack4(qi, hdp4):
    hd = qi.shape[-1]; qi = F.pad(qi, (0, hdp4 - hd))
    lo = (qi[..., 0::2].int() & 0xF); hi = (qi[..., 1::2].int() & 0xF)
    return (lo | (hi << 4)).to(torch.uint8).view(torch.int8).contiguous()

SHAPES = [(192, 8, 24, 1024), (384, 8, 48, 256), (384, 8, 48, 64)]
print(f"flash opt bench @ b{B}   (kernel-only us; vsM=vs fp16 MATH, vsF=vs fp16 flash)")
print(f"{'hd/T':>10} | {'fp16 MATH':>9} {'fp16 flash':>10} | {'int8':>8} {'vsM':>5} {'vsF':>5} {'relL2':>6} | {'int4':>8} {'vsM':>5} {'vsF':>5} {'relL2':>6}")
for (C, H, hd, T) in SHAPES:
    N = B; BH = N * H; sc = 1.0 / math.sqrt(hd)
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    ref = torch.einsum("nhij,nhjd->nhid", torch.softmax(torch.einsum("nhid,nhjd->nhij", q.float(), k.float()) * sc, -1), v.float())
    with sdpa_kernel(SDPBackend.MATH):
        tM = bench(lambda: F.scaled_dot_product_attention(q.reshape(1, BH, T, hd), k.reshape(1, BH, T, hd), v.reshape(1, BH, T, hd), scale=sc))
    with sdpa_kernel(FL):
        tF = bench(lambda: F.scaled_dot_product_attention(q.reshape(1, BH, T, hd), k.reshape(1, BH, T, hd), v.reshape(1, BH, T, hd), scale=sc))
    # int8
    hp = (hd + 31) // 32 * 32
    sq = (q.abs().amax(-1).clamp_min(1e-8) / 127.0).float(); skq = (k.abs().amax(-1).clamp_min(1e-8) / 127.0).float()
    sv = (v.abs().amax(2).clamp_min(1e-8) / 127.0).float()
    qi = F.pad(torch.round(q / sq.unsqueeze(-1)).clamp(-127, 127).to(torch.int8), (0, hp - hd)).contiguous()
    ki = F.pad(torch.round(k / skq.unsqueeze(-1)).clamp(-127, 127).to(torch.int8), (0, hp - hd)).contiguous()
    vi = F.pad(torch.round(v / sv.unsqueeze(2)).clamp(-127, 127).to(torch.int8), (0, hp - hd)).contiguous()
    o8 = mc.flash_attn_int8(qi, ki, vi, sq, skq, sv, sc); r8 = relL2(o8[..., :hd], ref)
    t8 = bench(lambda: mc.flash_attn_int8(qi, ki, vi, sq, skq, sv, sc))
    # int4
    hdp4 = 64; hdv = (hd + 31) // 32 * 32
    sq4 = (q.abs().amax(-1).clamp_min(1e-8) / 7.0).float(); sk4 = (k.abs().amax(-1).clamp_min(1e-8) / 7.0).float()
    sv4 = (v.abs().amax(2).clamp_min(1e-8) / 127.0).float()
    q4 = pack4(torch.round(q / sq4.unsqueeze(-1)).clamp(-8, 7).to(torch.int8), hdp4)
    k4 = pack4(torch.round(k / sk4.unsqueeze(-1)).clamp(-8, 7).to(torch.int8), hdp4)
    v4 = F.pad(torch.round(v / sv4.unsqueeze(2)).clamp(-127, 127).to(torch.int8), (0, hdv - hd)).contiguous()
    o4 = mc.flash_attn_int4(q4, k4, v4, sq4, sk4, sv4, hdp4, sc); r4 = relL2(o4[..., :hd], ref)
    t4 = bench(lambda: mc.flash_attn_int4(q4, k4, v4, sq4, sk4, sv4, hdp4, sc))
    print(f"{f'{hd}/{T}':>10} | {tM:9.1f} {tF:10.1f} | {t8:8.1f} {tM/t8:4.2f}x {tF/t8:4.2f}x {r8:6.3f} | {t4:8.1f} {tM/t4:4.2f}x {tF/t4:4.2f}x {r4:6.3f}")
