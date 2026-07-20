"""Fair attention-block comparison per shape: BOTH paths pay GroupNorm; the quant
path additionally pays the Q/K/V quantize. Components:
  fp16   = GroupNorm + fp16 MATH SDPA attention
  int8   = GroupNorm + quantize_attn_qkv(int8) + int8 attn (QKᵀ+softmax_requant+AV)
  int4   = GroupNorm + quantize_attn_qkv(int4) + int4 attn
GroupNorm is measured on the real block input [b,C,H,W] (channels_last, no SiLU, 32 groups);
quantize + attention on the post-projection q/k/v [BH,T,hd]. Weighted by per-forward counts.
Writes data/attn_fair_b<B>.csv
"""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from integration.fused_ops.fused_resblock import _group_norm_silu

B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0); dev = "cuda"
# (C, nh, hd, T, H, count); H*H = T (square latents)
SHAPES = [(192, 8, 24, 1024, 32, 5), (384, 8, 48, 256, 16, 5), (384, 8, 48, 64, 8, 5),
          (768, 8, 96, 16, 4, 5), (768, 8, 96, 4, 2, 1)]
NG = 32

def bench(fn, it=100, warm=30, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]

_burn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _burn = _burn @ _burn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []
tot = dict(fp16=0.0, int8=0.0, int4=0.0, gn=0.0, q8=0.0, q4=0.0, a_fp16=0.0, a8=0.0, a4=0.0)
print(f"Fair attention-block (norm+quantize+attn) @ b{B}")
print(f"{'C/hd/T':>12} {'cnt':>3} | {'GN':>7} {'q8':>7} {'q4':>7} | {'attn16':>7} {'attn8':>7} {'attn4':>7} | "
      f"{'fp16tot':>8} {'int8tot':>8} {'int4tot':>8} | {'i8/fp16':>7} {'i4/fp16':>7}")
for (C, nh, hd, T, Hs, cnt) in SHAPES:
    N = B; H = nh; BH = N * H; sc = 1.0 / math.sqrt(hd)
    # GroupNorm on real block input [b,C,H,W] channels_last, no SiLU
    xg = torch.randn(N, C, Hs, Hs, device=dev, dtype=torch.float16).to(memory_format=torch.channels_last)
    gw = torch.randn(C, device=dev, dtype=torch.float16); gb = torch.randn(C, device=dev, dtype=torch.float16)
    t_gn = bench(lambda: _group_norm_silu(xg, NG, gw, gb, 1e-5, False))
    # attention operands
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
    with sdpa_kernel(SDPBackend.MATH):
        t_a16 = bench(lambda: F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc))
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64; hp4 = (hd + 63) // 64 * 64
    t_q8 = t_a8 = t_q4 = t_a4 = None
    try:
        t_q8 = bench(lambda: mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8))
        qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
        def i8():
            S = mc.attn_qk_int8(qi, ki, sq, sk, sc); P, sp = mc.attn_softmax_requant(S); return mc.attn_av_int8(P, vt, sp, sv)
        t_a8 = bench(i8)
    except Exception:
        pass
    try:
        t_q4 = bench(lambda: mc.quantize_attn_qkv(qm, km, vm, hp4, hpa, 4))
        qi4, ki4, vt4, sq4, sk4, sv4 = mc.quantize_attn_qkv(qm, km, vm, hp4, hpa, 4)
        def i4():
            S = mc.attn_qk_int4(qi4, ki4, hp4, sq4, sk4, sc); P, sp = mc.attn_softmax_requant4(S); return mc.attn_av_int4(P, vt4, sp, sv4, T)
        t_a4 = bench(i4)
    except Exception:
        pass
    fp16tot = t_gn + t_a16
    int8tot = (t_gn + t_q8 + t_a8) if t_a8 is not None else None
    int4tot = (t_gn + t_q4 + t_a4) if t_a4 is not None else None
    def f(v): return f"{v:7.1f}" if v is not None else "    N/A"
    print(f"{f'{C}/{hd}/{T}':>12} {cnt:3d} | {t_gn:7.1f} {f(t_q8)} {f(t_q4)} | {t_a16:7.1f} {f(t_a8)} {f(t_a4)} | "
          f"{fp16tot:8.1f} {f(int8tot)} {f(int4tot)} | "
          f"{(fp16tot/int8tot if int8tot else float('nan')):6.2f}x {(fp16tot/int4tot if int4tot else float('nan')):6.2f}x")
    rows.append(dict(C=C, hd=hd, T=T, count=cnt, gn_us=round(t_gn, 1),
                     q8_us=round(t_q8, 1) if t_q8 else "", q4_us=round(t_q4, 1) if t_q4 else "",
                     attn_fp16_us=round(t_a16, 1), attn8_us=round(t_a8, 1) if t_a8 else "", attn4_us=round(t_a4, 1) if t_a4 else "",
                     fp16_tot_us=round(fp16tot, 1), int8_tot_us=round(int8tot, 1) if int8tot else "",
                     int4_tot_us=round(int4tot, 1) if int4tot else ""))
    tot["gn"] += cnt * t_gn; tot["a_fp16"] += cnt * t_a16; tot["fp16"] += cnt * fp16tot
    tot["int8"] += cnt * (int8tot if int8tot else fp16tot); tot["int4"] += cnt * (int4tot if int4tot else fp16tot)

print(f"\n=== weighted totals per forward (21 blocks) @ b{B} ===")
print(f"fp16  (GN + fp16 attn)            {tot['fp16']:9.1f} us   1.00x")
print(f"int8  (GN + quant + int8 attn)    {tot['int8']:9.1f} us   {tot['fp16']/tot['int8']:.2f}x   (hd=96 blocks fall back to fp16)")
print(f"int4  (GN + quant + int4 attn)    {tot['int4']:9.1f} us   {tot['fp16']/tot['int4']:.2f}x")
print(f"  [GroupNorm alone (shared)       {tot['gn']:9.1f} us]")
with open(f"docs/flash_attention_2026-07-19/data/attn_fair_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"WROTE data/attn_fair_b{B}.csv")
