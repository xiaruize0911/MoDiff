"""Prototype: int8 attention SCORES (S) instead of fp16, per §7's 'fewer T*T bytes' lever. Stores the
[BH,T,T] score matrix as int8 (1B) -> halves the softmax read (and, with an int8-S QKᵀ, the write).
Measures softmax speed (fp16-S vs int8-S) and full-attention quality (rel-err vs fp32). Emits
int8_score.csv."""
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
print(f"{'shape':>10} | {'smax fp16-S':>11} {'smax int8-S':>11} {'spdup':>5} | {'rel fp16-S':>10} {'rel int8-S':>10}")
for (BH, T, hd) in [(256, 1024, 24), (256, 256, 48), (256, 64, 96)]:
    scale = 1.0 / math.sqrt(hd); hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    ref = torch.bmm(F.softmax(torch.bmm(Q.float(), K.float().transpose(1, 2)) * scale, -1), V.float())
    qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hp, hpa, 8)
    Sf = mc.attn_qk_int8(qi, ki, sq, sk, scale)                     # fp16 pre-scaled logits [BH,T,T]
    c = Sf.float().amax(-1).mean().item()
    # ---- fp16-score path (current best): static-c softmax on fp16 S ----
    Pf, spf = mc.attn_softmax_requant_static(Sf, c)
    Of = mc.attn_av_int8(Pf, vt, spf, sv)[:, :, :hd]
    rel_f = ((Of.float() - ref).norm() / ref.norm()).item()
    t_f = bench(lambda: mc.attn_softmax_requant_static(Sf, c))
    # ---- int8-score path: quantize S -> int8 (per-tensor sS), softmax reads int8 ----
    sS = Sf.float().abs().max().item() / 127.0
    Si8 = torch.round(Sf.float() / sS).clamp(-127, 127).to(torch.int8).contiguous()
    P8, sp8 = mc.attn_softmax_requant_s8(Si8, sS, c)
    O8 = mc.attn_av_int8(P8, vt, sp8, sv)[:, :, :hd]
    rel_8 = ((O8.float() - ref).norm() / ref.norm()).item()
    t_8 = bench(lambda: mc.attn_softmax_requant_s8(Si8, sS, c))
    print(f"{BH},{T:>4} | {t_f:11.1f} {t_8:11.1f} {t_f/t_8:5.2f} | {rel_f:10.4f} {rel_8:10.4f}")
    rows.append(dict(BH=BH, T=T, softmax_fp16S_us=round(t_f, 1), softmax_int8S_us=round(t_8, 1),
                     softmax_speedup=round(t_f / t_8, 3), rel_fp16S=round(rel_f, 4), rel_int8S=round(rel_8, 4)))
with open(f"{OUT}/int8_score.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE int8_score.csv")
