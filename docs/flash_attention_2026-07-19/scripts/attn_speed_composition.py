"""Speed composition of fp16 MATH vs int8 materialized attention at the two shapes
where int8 loses (hd48/T256, hd48/T64). Per-CUDA-kernel breakdown via torch.profiler
(ncu counters are blocked on this box, so this is the finest breakdown available).
Shows WHY int8 loses at small T: the fixed-cost quantize + softmax-requant don't shrink
with T, so they dominate once the O(T^2) matmuls get cheap.
"""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile, ProfilerActivity

dev = "cuda"; torch.manual_seed(0)
_burn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _burn = _burn @ _burn * 1e-4 + 1.0
torch.cuda.synchronize()

def prof(fn, reps=50):
    for _ in range(15): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps): fn()
    torch.cuda.synchronize()
    out = {}
    for e in p.key_averages():
        if e.self_device_time_total > 0:
            out[e.key] = out.get(e.key, 0.0) + e.self_device_time_total / reps
    return out

def tag(name):
    n = name.lower()
    if "s1688gemm" in n or "gemm" in n and "s8" not in n and "s4" not in n: return "fp16 QKᵀ/PV matmul"
    if "softmax_warp" in n: return "fp16 softmax"
    if "bmm_qk_s8" in n: return "int8 QKᵀ matmul"
    if "bmm_av_s8" in n: return "int8 AV matmul"
    if "softmax_requant" in n: return "int8 softmax+requant"
    if "aq_qtok" in n: return "quantize Q/K (per-token)"
    if "aq_vscale" in n or "aq_vquant" in n: return "quantize V (+transpose)"
    if "elementwise" in n or "fill" in n: return "elementwise/scale"
    return name[:34]

N, H = 128, 8
for (T, hd) in [(256, 48), (64, 48)]:
    BH = N * H; sc = 1.0 / math.sqrt(hd)
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64

    def fp16():
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc)
    def quant():
        return mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
    qi, ki, vt, sq, sk, sv = quant()
    def i8_core():
        S = mc.attn_qk_int8(qi, ki, sq, sk, sc); P, sp = mc.attn_softmax_requant(S); return mc.attn_av_int8(P, vt, sp, sv)
    def i8_full():
        qi2, ki2, vt2, sq2, sk2, sv2 = mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
        S = mc.attn_qk_int8(qi2, ki2, sq2, sk2, sc); P, sp = mc.attn_softmax_requant(S); return mc.attn_av_int8(P, vt2, sp, sv2)

    pf = prof(fp16); pi = prof(i8_full)
    # aggregate by tag
    def agg(pr):
        out = {}
        for k_, v_ in pr.items(): out[tag(k_)] = out.get(tag(k_), 0.0) + v_
        return out
    af = agg(pf); ai = agg(pi)
    tf = sum(af.values()); ti = sum(ai.values())
    print(f"\n================= hd{hd}/T{T}  (BH={BH}, b128) =================")
    print(f"  fp16 MATH total = {tf:7.1f} us      int8 materialized total = {ti:7.1f} us   ({tf/ti:.2f}x)")
    print(f"  --- fp16 MATH composition ---")
    for k_, v_ in sorted(af.items(), key=lambda x: -x[1]):
        print(f"     {v_:7.1f} us  {100*v_/tf:5.1f}%  {k_}")
    print(f"  --- int8 materialized composition ---")
    for k_, v_ in sorted(ai.items(), key=lambda x: -x[1]):
        print(f"     {v_:7.1f} us  {100*v_/ti:5.1f}%  {k_}")
