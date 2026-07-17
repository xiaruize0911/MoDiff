"""Attention sub-op speed by precision (fp16 vs int8 vs int4), per real churches shape. Splits the
attention matmuls (QKᵀ, AV) and softmax out from the qkv/proj linear so the quantization benefit that
the merged e2e 'GEMM' bucket hides (behind the slow int qkv/proj linear) is visible. Same algorithm
per op (materialized), only precision varies. Emits attn_ops.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, modiff_cutlass as mc
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=50, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

rows = []
print(f"{'shape':>10} {'op':>8} | {'fp16':>8} {'int8':>8} {'int4':>8} | {'i8/16':>5} {'i4/16':>5}")
for (BH, T, hd) in [(256, 1024, 24), (256, 256, 48), (256, 64, 96)]:
    scale = 1.0 / math.sqrt(hd)
    hp8, hp4 = (hd + 31) // 32 * 32, (hd + 63) // 64 * 64
    Qf = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    Kf = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    Vf = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    # int operands
    Qi = torch.randint(-127, 127, (BH, T, hp8), device="cuda", dtype=torch.int8)
    Ki = torch.randint(-127, 127, (BH, T, hp8), device="cuda", dtype=torch.int8)
    Qi4 = torch.randint(-7, 7, (BH, T, hp4 // 2), device="cuda", dtype=torch.int8)
    Ki4 = torch.randint(-7, 7, (BH, T, hp4 // 2), device="cuda", dtype=torch.int8)
    sq = torch.rand(BH, T, device="cuda"); sk = torch.rand(BH, T, device="cuda")
    S = torch.randn(BH, T, T, device="cuda", dtype=torch.float16); cc = S.float().amax(-1).mean().item()
    Pf = torch.randn(BH, T, T, device="cuda", dtype=torch.float16)   # fp16 attention probs (precomputed)
    P8, sp8 = mc.attn_softmax_requant(S); P4, sp4 = mc.attn_softmax_requant4(S)
    Vt8 = torch.randint(-100, 100, (BH, hp4, T), device="cuda", dtype=torch.int8)
    Vt4 = torch.randint(-7, 7, (BH, hp4, T // 2), device="cuda", dtype=torch.int8)
    sv = torch.rand(BH, hp4, device="cuda")
    ops = {
        "QKT": (lambda: (torch.bmm(Qf, Kf.transpose(1, 2)) * scale).half(),
                lambda: mc.attn_qk_int8(Qi, Ki, sq, sk, scale),
                lambda: mc.attn_qk_int4(Qi4, Ki4, hp4, sq, sk, scale)),
        "softmax": (lambda: mc.attn_softmax_fp16(S, False, 0.0),
                    lambda: mc.attn_softmax_requant(S),
                    lambda: mc.attn_softmax_requant4(S)),
        "AV": (lambda: torch.bmm(Pf, Vf),
               lambda: mc.attn_av_int8(P8, Vt8, sp8, sv),
               lambda: mc.attn_av_int4(P4, Vt4, sp4, sv, T)),
    }
    for op, (ff, f8, f4) in ops.items():
        t16, t8, t4 = bench(ff), bench(f8), bench(f4)
        print(f"{BH},{T:>4} {op:>8} | {t16:8.1f} {t8:8.1f} {t4:8.1f} | {t16/t8:5.2f} {t16/t4:5.2f}")
        rows.append(dict(BH=BH, T=T, op=op, fp16_us=round(t16, 1), int8_us=round(t8, 1), int4_us=round(t4, 1),
                         int8_speedup=round(t16 / t8, 3), int4_speedup=round(t16 / t4, 3)))
with open(f"{OUT}/attn_ops.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE attn_ops.csv")
