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
        # int8
        hpq = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64
        def q8(x, hp):
            sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
            return F.pad(torch.round(x / sc).clamp(-127, 127).to(torch.int8), (0, hp - hd)).contiguous(), sc.squeeze(-1).float().contiguous()
        qi, sq = q8(Q, hpq); ki, sk = q8(K, hpq)
        svc = V.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
        vt = F.pad(torch.round(V / svc).clamp(-127, 127).to(torch.int8).transpose(1, 2).contiguous(), (0, 0, 0, hpa - hd)).contiguous()
        sv = F.pad(svc.squeeze(1), (0, hpa - hd)).float().contiguous()
        def int8_attn():
            S = mc.attn_qk_int8(qi, ki); P, sp = mc.attn_softmax_requant(S, sq, sk, scale)
            return mc.attn_av_int8(P, vt, sp, sv)
        # int4
        hp4 = (hd + 63) // 64 * 64
        def q4(x, hp):
            sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
            return pack(F.pad(torch.round(x / sc).clamp(-7, 7).to(torch.int8), (0, hp - hd))), sc.squeeze(-1).float().contiguous()
        qi4, sq4 = q4(Q, hp4); ki4, sk4 = q4(K, hp4)
        sv4c = V.abs().amax(1, keepdim=True).clamp_min(1e-8) / 7
        vt4 = pack(F.pad(torch.round(V / sv4c).clamp(-7, 7).to(torch.int8).transpose(1, 2).contiguous(), (0, 0, 0, hp4 - hd)))
        sv4 = F.pad(sv4c.squeeze(1), (0, hp4 - hd)).float().contiguous()
        def int4_attn():
            S = mc.attn_qk_int4(qi4, ki4, hp4); P, sp = mc.attn_softmax_requant4(S, sq4, sk4, scale)
            return mc.attn_av_int4(P, vt4, sp, sv4, T)
        t16 = bench(fp16_attn); t8 = bench(int8_attn); t4 = bench(int4_attn)
    else:
        t16 = bench(fp16_attn); t8 = t4 = float("nan")
    print(f"{BH},{T:>4},{hd:>3} | {t16:8.1f} {t8:8.1f} {t4:8.1f} | {t16/t8:5.2f} {t16/t4:5.2f}")
    rows.append(dict(BH=BH, T=T, hd=hd, fp16_us=round(t16, 1), int8_us=round(t8, 1), int4_us=round(t4, 1),
                     int8_vs_fp16=round(t16 / t8, 3) if t8 == t8 else "", int4_vs_fp16=round(t16 / t4, 3) if t4 == t4 else ""))
with open(f"{OUT}/attn_kernel_speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE attn_kernel_speed.csv")
