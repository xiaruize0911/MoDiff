"""Full int8-SCORE attention vs fp16-score (the §7 'fewer T*T bytes' lever, completed).
fp16-S path: attn_qk_int8 (fp16 S) -> softmax_static -> AV.
int8-S path: attn_qk_int8_s8out (INT8 S, both T*T passes at 1B) -> softmax_requant_s8 -> AV.
Measures QKᵀ, softmax, and full-attention time + quality (rel-err vs fp32). Emits int8_score.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=50, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

rows = []
torch.manual_seed(0)
print(f"{'shape':>10} | {'QKT 16':>7} {'QKT i8':>7} | {'smax16':>7} {'smax i8':>7} | "
      f"{'full16':>7} {'full i8':>7} {'spdup':>5} | {'rel16':>7} {'rel i8':>7} {'S rel':>6}")
for (BH, T, hd) in [(256, 1024, 24), (256, 256, 48), (256, 64, 96)]:
    scale = 1.0 / math.sqrt(hd); hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    ref = torch.bmm(F.softmax(torch.bmm(Q.float(), K.float().transpose(1, 2)) * scale, -1), V.float())
    qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hp, hpa, 8)
    Sf = mc.attn_qk_int8(qi, ki, sq, sk, scale)                     # fp16 scores (reference for sS/c)
    sS = Sf.float().abs().max().item() / 127.0
    c = Sf.float().amax(-1).mean().item()
    Si = mc.attn_qk_int8_s8out(qi, ki, sq, sk, scale, sS)           # int8 scores
    Srel = ((Si.float() * sS - Sf.float()).norm() / Sf.float().norm()).item()   # QKᵀ int8-out accuracy
    # full attention both ways
    def full16():
        S = mc.attn_qk_int8(qi, ki, sq, sk, scale); P, sp = mc.attn_softmax_requant_static(S, c)
        return mc.attn_av_int8(P, vt, sp, sv)
    def full8():
        S = mc.attn_qk_int8_s8out(qi, ki, sq, sk, scale, sS); P, sp = mc.attn_softmax_requant_s8(S, sS, c)
        return mc.attn_av_int8(P, vt, sp, sv)
    rel16 = ((full16().float()[:, :, :hd] - ref).norm() / ref.norm()).item()
    rel8 = ((full8().float()[:, :, :hd] - ref).norm() / ref.norm()).item()
    tqk16 = bench(lambda: mc.attn_qk_int8(qi, ki, sq, sk, scale)); tqk8 = bench(lambda: mc.attn_qk_int8_s8out(qi, ki, sq, sk, scale, sS))
    tsm16 = bench(lambda: mc.attn_softmax_requant_static(Sf, c)); tsm8 = bench(lambda: mc.attn_softmax_requant_s8(Si, sS, c))
    tf16 = bench(full16); tf8 = bench(full8)
    print(f"{BH},{T:>4} | {tqk16:7.1f} {tqk8:7.1f} | {tsm16:7.1f} {tsm8:7.1f} | "
          f"{tf16:7.1f} {tf8:7.1f} {tf16/tf8:5.2f} | {rel16:7.4f} {rel8:7.4f} {Srel:6.3f}")
    rows.append(dict(BH=BH, T=T, qkT_fp16_us=round(tqk16, 1), qkT_int8_us=round(tqk8, 1),
                     softmax_fp16_us=round(tsm16, 1), softmax_int8_us=round(tsm8, 1),
                     full_fp16_us=round(tf16, 1), full_int8_us=round(tf8, 1), full_speedup=round(tf16 / tf8, 3),
                     rel_fp16S=round(rel16, 4), rel_int8S=round(rel8, 4), S_rel=round(Srel, 4)))
with open(f"{OUT}/int8_score.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE int8_score.csv")
