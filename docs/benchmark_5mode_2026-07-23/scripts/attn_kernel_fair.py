"""Kernel item 5 (speed) — attention kernel WITH the norm layer for fairness, 5 modes, b128.
Both paths pay GroupNorm on the real [b,C,H,W] block input; the quant path additionally pays the
Q/K/V quantize. Cores:
  fp16 = GroupNorm + fp16 MATH SDPA
  int8 = GroupNorm + quantize_attn_qkv(int8) + flash_attn_int8_vt   (FUSED flash, sole quant-attn path)
  int4 = GroupNorm + quantize_attn_qkv_i4qk_i8v + flash_attn_int4_vt (int4 Q/K + int8 V)
Only hd<=48 & T%64==0 blocks run flash (24/1024, 48/256, 48/64 = 15/21 blocks); the hd=96 blocks
(16, 4) fall back to fp16. NOTE: attention has no modiff variant, so int8_baseline == int8_modiff
(int4 likewise). CUDA-event median warm 30 + 100 iters x 5 reps, burn-in. rel-L2 vs fp32 sanity per
eligible shape. Writes data/attn_kernel_fair_speed.csv.
"""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from integration.fused_ops.fused_resblock import _group_norm_silu

torch.manual_seed(0); dev = "cuda"
B = 128; NG = 32
HERE = "docs/benchmark_5mode_2026-07-23"
# (C, nh, hd, T, H, count); H*H = T
SHAPES = [(192, 8, 24, 1024, 32, 5), (384, 8, 48, 256, 16, 5), (384, 8, 48, 64, 8, 5),
          (768, 8, 96, 16, 4, 5), (768, 8, 96, 4, 2, 1)]


def bench(fn, it=200, warm=60, reps=7):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]


def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)


_burn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _burn = _burn @ _burn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []
tot = dict(fp16=0.0, int8=0.0, int4=0.0, gn=0.0)
print(f"Attention kernel WITH norm (fair) @ b{B} (us, median)")
print(f"{'C/hd/T':>13} {'cnt':>3} | {'GN':>7} {'q8':>7} {'q4':>7} {'a16':>7} {'a8':>7} {'a4':>7} | "
      f"{'fp16tot':>8} {'int8tot':>8} {'int4tot':>8} | {'i8/16':>6} {'i4/16':>6} | {'r8':>6} {'r4':>6}")
for (C, nh, hd, T, Hs, cnt) in SHAPES:
    N = B; H = nh; BH = N * H; sc = 1.0 / math.sqrt(hd)
    xg = torch.randn(N, C, Hs, Hs, device=dev, dtype=torch.float16).to(memory_format=torch.channels_last)
    gw = torch.randn(C, device=dev, dtype=torch.float16); gb = torch.randn(C, device=dev, dtype=torch.float16)
    t_gn = bench(lambda: _group_norm_silu(xg, NG, gw, gb, 1e-5, False))
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
    with sdpa_kernel(SDPBackend.MATH):
        t_a16 = bench(lambda: F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc))
    ref = None
    with torch.no_grad():
        S = torch.einsum("nhid,nhjd->nhij", q.float(), k.float()) * sc
        ref = torch.einsum("nhij,nhjd->nhid", torch.softmax(S, -1), v.float())
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    eligible = (hd <= 48 and T % 64 == 0)
    t_q8 = t_a8 = t_q4 = t_a4 = None; r8 = r4 = None
    if eligible:
        # int8 fused flash
        hd_pad = ((hd + 31) // 32) * 32
        def quant8():
            return mc.quantize_attn_qkv(qm, km, vm, hd_pad, hd_pad, 8)
        t_q8 = bench(quant8)
        qi, ki, vt, sq, sk, sv = quant8()
        qi = qi.view(N, H, T, hd_pad); ki = ki.view(N, H, T, hd_pad); vt = vt.view(N, H, hd_pad, T)
        sq = sq.view(N, H, T).contiguous(); sk = sk.view(N, H, T).contiguous(); sv = sv[..., :hd].contiguous().view(N, H, hd)
        def flash8():
            return mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc)
        r8 = relL2(flash8().reshape(N, H, T, hd), ref)
        t_a8 = bench(flash8)
        # int4 (int4 Q/K + int8 V) fused flash
        hdp4, hdp_v = 64, ((hd + 31) // 32) * 32
        def quant4():
            return mc.quantize_attn_qkv_i4qk_i8v(qm, km, vm, hdp4, hdp_v)
        t_q4 = bench(quant4)
        q4, k4, vt4, sq4, sk4, sv4 = quant4()
        q4 = q4.view(N, H, T, -1); k4 = k4.view(N, H, T, -1); vt4 = vt4.view(N, H, hdp_v, T)
        sq4 = sq4.view(N, H, T).contiguous(); sk4 = sk4.view(N, H, T).contiguous(); sv4 = sv4[..., :hd].contiguous().view(N, H, hd)
        def flash4():
            return mc.flash_attn_int4_vt(q4, k4, vt4, sq4, sk4, sv4, hdp4, sc)
        r4 = relL2(flash4().reshape(N, H, T, hd), ref)
        t_a4 = bench(flash4)
    fp16tot = t_gn + t_a16
    int8tot = (t_gn + t_q8 + t_a8) if eligible else fp16tot     # hd=96 blocks stay fp16
    int4tot = (t_gn + t_q4 + t_a4) if eligible else fp16tot

    def g(x): return f"{x:7.1f}" if x is not None else "    N/A"
    print(f"{f'{C}/{hd}/{T}':>13} {cnt:3d} | {t_gn:7.1f} {g(t_q8)} {g(t_q4)} {t_a16:7.1f} {g(t_a8)} {g(t_a4)} | "
          f"{fp16tot:8.1f} {int8tot:8.1f} {int4tot:8.1f} | {fp16tot/int8tot:5.2f}x {fp16tot/int4tot:5.2f}x | "
          f"{(r8 if r8 is not None else float('nan')):6.3f} {(r4 if r4 is not None else float('nan')):6.3f}")
    rows.append(dict(C=C, hd=hd, T=T, count=cnt, eligible=int(eligible), gn_us=round(t_gn, 1),
                     q8_us=round(t_q8, 1) if t_q8 else "", q4_us=round(t_q4, 1) if t_q4 else "",
                     attn16_us=round(t_a16, 1), attn8_us=round(t_a8, 1) if t_a8 else "", attn4_us=round(t_a4, 1) if t_a4 else "",
                     fp16_tot_us=round(fp16tot, 1), int8_tot_us=round(int8tot, 1), int4_tot_us=round(int4tot, 1),
                     int8_vs_fp16=round(fp16tot / int8tot, 3), int4_vs_fp16=round(fp16tot / int4tot, 3),
                     relL2_int8=round(r8, 4) if r8 is not None else "", relL2_int4=round(r4, 4) if r4 is not None else ""))
    tot["gn"] += cnt * t_gn; tot["fp16"] += cnt * fp16tot; tot["int8"] += cnt * int8tot; tot["int4"] += cnt * int4tot

print(f"\n=== weighted totals per forward (21 blocks, GN+quant+attn) @ b{B} ===")
print(f"fp16  {tot['fp16']:9.1f} us  1.00x")
print(f"int8  {tot['int8']:9.1f} us  {tot['fp16']/tot['int8']:.2f}x   (hd=96 blocks fall back to fp16)")
print(f"int4  {tot['int4']:9.1f} us  {tot['fp16']/tot['int4']:.2f}x")
print(f"  [GroupNorm alone (shared) {tot['gn']:9.1f} us]")
rows.append(dict(C="WEIGHTED_TOTAL", hd="", T="", count=21, eligible="", gn_us=round(tot["gn"], 1),
                 q8_us="", q4_us="", attn16_us="", attn8_us="", attn4_us="",
                 fp16_tot_us=round(tot["fp16"], 1), int8_tot_us=round(tot["int8"], 1), int4_tot_us=round(tot["int4"], 1),
                 int8_vs_fp16=round(tot["fp16"] / tot["int8"], 3), int4_vs_fp16=round(tot["fp16"] / tot["int4"], 3),
                 relL2_int8="", relL2_int4=""))
with open(f"{HERE}/data/attn_kernel_fair_speed.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"WROTE {HERE}/data/attn_kernel_fair_speed.csv")
