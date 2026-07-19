"""Kernel-level benchmark of the FULL attention (QKᵀ + softmax + AV) at the dominant churches shape
(b128 level-0: BH=b*nh=1024, T=1024, hd=24), comparing the fp16 SDPA path actually used against every
quantized-attention kernel option. Shows WHY attention stays fp16: all quantized paths are slower.
Writes data/attn_kernel_bench_b128.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend


def bench(fn, it=50, warm=20, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]


def rel(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)

torch.manual_seed(0)
BH, T, hd = 1024, 1024, 24     # b128, nh=8 -> BH=1024; C192 level, the dominant attention block
scale = 1.0 / math.sqrt(hd)
hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64; hp4 = (hd + 63) // 64 * 64
q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
k = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
v = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
q4 = q.view(1, BH, T, hd)  # SDPA wants [N,H,T,hd]
ref = torch.einsum("nij,njd->nid", torch.softmax(torch.einsum("nid,njd->nij", q.float(), k.float()) * scale, -1), v.float())

# fp16 SDPA (MATH) -- what the model actually runs
def fp16_sdpa():
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.view(1, BH, T, hd), k.view(1, BH, T, hd), v.view(1, BH, T, hd), scale=scale)

# int8 "our kernel" materialized (quantize + int8 QKᵀ + dynamic requant + int8 AV)
qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(q, k, v, hp, hpa, 8)
Sf = mc.attn_qk_int8(qi, ki, sq, sk, scale); sS = Sf.float().abs().max().item() / 127.0
def quant8(): return mc.quantize_attn_qkv(q, k, v, hp, hpa, 8)
def int8_mat():
    S = mc.attn_qk_int8(qi, ki, sq, sk, scale); P, sp = mc.attn_softmax_requant(S); return mc.attn_av_int8(P, vt, sp, sv)
def int8_score():
    S = mc.attn_qk_int8_s8out(qi, ki, sq, sk, scale, sS); P, sp = mc.attn_softmax_requant_s8_dyn(S, sS); return mc.attn_av_int8(P, vt, sp, sv)
# int4
qi4, ki4, vt4, sq4, sk4, sv4 = mc.quantize_attn_qkv(q, k, v, hp4, hpa, 4)
def quant4(): return mc.quantize_attn_qkv(q, k, v, hp4, hpa, 4)
def int4_mat():
    S = mc.attn_qk_int4(qi4, ki4, hp4, sq4, sk4, scale); P, sp = mc.attn_softmax_requant4(S); return mc.attn_av_int4(P, vt4, sp, sv4, T)

t_fp16 = bench(fp16_sdpa)
tq8 = bench(quant8); t8 = bench(int8_mat); t8s = bench(int8_score)
tq4 = bench(quant4); t4 = bench(int4_mat)
r8 = rel(int8_mat()[:, :, :hd], ref); r8s = rel(int8_score()[:, :, :hd], ref); r4 = rel(int4_mat()[:, :, :hd], ref)

rows = [
    dict(path="fp16 SDPA (used)", attn_us=round(t_fp16, 1), quantize_us=0.0, total_us=round(t_fp16, 1), vs_fp16=1.0, rel_vs_fp32=""),
    dict(path="int8 materialized (our kernel)", attn_us=round(t8, 1), quantize_us=round(tq8, 1), total_us=round(t8 + tq8, 1), vs_fp16=round(t_fp16 / (t8 + tq8), 2), rel_vs_fp32=round(r8, 3)),
    dict(path="int8-score (int8 QKᵀ+dyn softmax)", attn_us=round(t8s, 1), quantize_us=round(tq8, 1), total_us=round(t8s + tq8, 1), vs_fp16=round(t_fp16 / (t8s + tq8), 2), rel_vs_fp32=round(r8s, 3)),
    dict(path="int4 materialized", attn_us=round(t4, 1), quantize_us=round(tq4, 1), total_us=round(t4 + tq4, 1), vs_fp16=round(t_fp16 / (t4 + tq4), 2), rel_vs_fp32=round(r4, 3)),
]
print(f"FULL attention @ BH={BH} T={T} hd={hd} (b128 C192 level) — full = QKᵀ+softmax+AV (+quantize for int)")
print(f"{'path':34s} {'attn us':>9} {'quant us':>9} {'total us':>9} {'vs fp16':>8} {'rel/fp32':>9}")
for r in rows:
    print(f"{r['path']:34s} {r['attn_us']:9.1f} {r['quantize_us']:9.1f} {r['total_us']:9.1f} {r['vs_fp16']:7}x {str(r['rel_vs_fp32']):>9}")
with open("docs/layer_roofline_2026-07-19/data/attn_kernel_bench_b128.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nWROTE attn_kernel_bench_b128.csv")
