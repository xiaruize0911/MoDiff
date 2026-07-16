"""Attention-kernel speed: fp16 standard (bmm QKᵀ + softmax + bmm AV) vs int8/int4
materialized quantized attention (quantize + attn_qk + softmax_requant + attn_av), per real
churches attention shape. 'Effective' = int8/int4 beats fp16 STANDARD (math) attention.
Emits data/attn_kernel_speed.csv."""
import os, sys, math, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
torch.backends.cudnn.benchmark = True
OUT = "/workspace/MoDiff/docs/comprehensive_benchmark_2026-07-16/data"

def bench(fn, it=30, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3  # us

def pack(t): lo = t[..., 0::2] & 0xF; hi = t[..., 1::2] & 0xF; return (lo | (hi << 4)).to(torch.int8).contiguous()

# real churches attention: (BH=N*nh, T, hd) at batch 32, nh=8
SHAPES = [(256, 1024, 24), (256, 256, 48), (256, 64, 96)]
rows = []
print(f"{'BH,T,hd':>14} | {'fp16 us':>8} {'int8 us':>8} {'int4 us':>8} | i8/16 i4/16")
for (BH, T, hd) in SHAPES:
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(hd)
    def fp16_attn():
        s = torch.bmm(Q, K.transpose(1, 2)) * scale
        p = F.softmax(s.float(), -1).half()
        return torch.bmm(p, V)
    if T % 64 == 0:
        # Effective path: fused CUDA quantize + fp16 scores + fused softmax (matches the block).
        hpq8 = (hd + 31) // 32 * 32; hpq4 = (hd + 63) // 64 * 64; hpa = (hd + 63) // 64 * 64
        def int8_attn():
            qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hpq8, hpa, 8)
            S = mc.attn_qk_int8(qi, ki, sq, sk, scale); P, sp = mc.attn_softmax_requant(S)
            return mc.attn_av_int8(P, vt, sp, sv)
        def int4_attn():
            qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hpq4, hpa, 4)
            S = mc.attn_qk_int4(qi, ki, hpq4, sq, sk, scale); P, sp = mc.attn_softmax_requant4(S)
            return mc.attn_av_int4(P, vt, sp, sv, T)
        t16 = bench(fp16_attn); t8 = bench(int8_attn); t4 = bench(int4_attn)
    else:
        t16 = bench(fp16_attn); t8 = t4 = float("nan")
    print(f"{BH},{T:>4},{hd:>3} | {t16:8.1f} {t8:8.1f} {t4:8.1f} | {t16/t8:5.2f} {t16/t4:5.2f}")
    rows.append(dict(BH=BH, T=T, hd=hd, fp16_us=round(t16, 1), int8_us=round(t8, 1), int4_us=round(t4, 1),
                     int8_vs_fp16=round(t16 / t8, 3) if t8 == t8 else "", int4_vs_fp16=round(t16 / t4, 3) if t4 == t4 else ""))
with open(f"{OUT}/attn_kernel_speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE attn_kernel_speed.csv")
