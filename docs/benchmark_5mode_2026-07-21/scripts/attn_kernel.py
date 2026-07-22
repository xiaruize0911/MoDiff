"""Attention kernel speed (WITH GroupNorm, fair) — every attention block the churches UNet runs,
5 blocks, b128. Both paths pay GroupNorm on the real [b,C,H,W] input; the quant path additionally
pays the fused Q/K/V quantize. Cores (latest fused flash kernels):
  fp16 = GroupNorm + fp16 MATH SDPA
  int8 = GroupNorm + quantize_attn_qkv(int8)          + flash_attn_int8_vt
  int4 = GroupNorm + quantize_attn_qkv_i4qk_i8v       + flash_attn_int4_vt   (int4 Q/K + int8 V)
Only hd<=48 & T%64==0 blocks run flash (24/1024, 48/256, 48/64); hd=96 blocks fall back to fp16 SDPA.
Attention has NO modiff variant (baseline == modiff). Shapes + per-step counts from the real-model
enumeration (shapes.attn_shapes()). rel-L2 vs fp32 reference per eligible shape.
Writes data/attn_kernel_speed.csv (per-call + per-step rollup).
"""
import os, sys, csv, math
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "docs/benchmark_5mode_2026-07-21/scripts")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from integration.fused_ops.fused_resblock import _group_norm_silu
import shapes as S

torch.manual_seed(0); dev = "cuda"
B = 128; NG = 32
HERE = "docs/benchmark_5mode_2026-07-21"


def bench(fn, it=100, warm=30, reps=5):
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

attn = sorted(S.attn_shapes(), key=lambda s: -s["T"])
rows = []
tot = dict(fp16=0.0, int8=0.0, int4=0.0, gn=0.0)
print(f"Attention kernel WITH norm (fair) @ b{B} (us/call, median)")
print(f"{'C/hd/T':>13}{'cnt':>4} | {'GN':>7}{'q8':>7}{'q4':>7}{'a16':>7}{'a8':>7}{'a4':>7} | "
      f"{'fp16':>8}{'int8':>8}{'int4':>8} | {'i8x':>5}{'i4x':>5} | {'r8':>6}{'r4':>6}")
for sh in attn:
    C, nh, hd, T, Hs, cnt = sh["C"], sh["nh"], sh["hd"], sh["T"], sh["Hspatial"], sh["count"]
    N = B; H = nh; BH = N * H; sc = 1.0 / math.sqrt(hd)
    xg = torch.randn(N, C, Hs, Hs, device=dev, dtype=torch.float16).to(memory_format=torch.channels_last)
    gw = torch.randn(C, device=dev, dtype=torch.float16); gb = torch.randn(C, device=dev, dtype=torch.float16)
    t_gn = bench(lambda: _group_norm_silu(xg, NG, gw, gb, 1e-5, False))
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
    with sdpa_kernel(SDPBackend.MATH):
        t_a16 = bench(lambda: F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc))
    with torch.no_grad():
        Smat = torch.einsum("nhid,nhjd->nhij", q.float(), k.float()) * sc
        ref = torch.einsum("nhij,nhjd->nhid", torch.softmax(Smat, -1), v.float())
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    eligible = (hd <= 48 and T % 64 == 0)
    t_q8 = t_a8 = t_q4 = t_a4 = None; r8 = r4 = None
    if eligible:
        hd_pad = ((hd + 31) // 32) * 32
        def quant8(): return mc.quantize_attn_qkv(qm, km, vm, hd_pad, hd_pad, 8)
        t_q8 = bench(quant8)
        qi, ki, vt, sq, sk, sv = quant8()
        qi = qi.view(N, H, T, hd_pad); ki = ki.view(N, H, T, hd_pad); vt = vt.view(N, H, hd_pad, T)
        sq = sq.view(N, H, T).contiguous(); sk = sk.view(N, H, T).contiguous(); sv = sv[..., :hd].contiguous().view(N, H, hd)
        def flash8(): return mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc)
        r8 = relL2(flash8().reshape(N, H, T, hd), ref); t_a8 = bench(flash8)
        hdp4, hdp_v = 64, ((hd + 31) // 32) * 32
        def quant4(): return mc.quantize_attn_qkv_i4qk_i8v(qm, km, vm, hdp4, hdp_v)
        t_q4 = bench(quant4)
        q4, k4, vt4, sq4, sk4, sv4 = quant4()
        q4 = q4.view(N, H, T, -1); k4 = k4.view(N, H, T, -1); vt4 = vt4.view(N, H, hdp_v, T)
        sq4 = sq4.view(N, H, T).contiguous(); sk4 = sk4.view(N, H, T).contiguous(); sv4 = sv4[..., :hd].contiguous().view(N, H, hd)
        def flash4(): return mc.flash_attn_int4_vt(q4, k4, vt4, sq4, sk4, sv4, hdp4, sc)
        r4 = relL2(flash4().reshape(N, H, T, hd), ref); t_a4 = bench(flash4)
    fp16tot = t_gn + t_a16
    int8tot = (t_gn + t_q8 + t_a8) if eligible else fp16tot
    int4tot = (t_gn + t_q4 + t_a4) if eligible else fp16tot

    def g(x): return f"{x:7.1f}" if x is not None else "    N/A"
    print(f"{f'{C}/{hd}/{T}':>13}{cnt:>4} | {t_gn:7.1f}{g(t_q8)}{g(t_q4)}{t_a16:7.1f}{g(t_a8)}{g(t_a4)} | "
          f"{fp16tot:8.1f}{int8tot:8.1f}{int4tot:8.1f} | {fp16tot/int8tot:4.2f}x{fp16tot/int4tot:4.2f}x | "
          f"{(r8 if r8 is not None else float('nan')):6.3f}{(r4 if r4 is not None else float('nan')):6.3f}")
    rows.append(dict(C=C, nh=nh, hd=hd, T=T, count_per_step=cnt, flash_eligible=int(eligible),
                     gn_us=round(t_gn, 1), q8_us=round(t_q8, 1) if t_q8 else "", q4_us=round(t_q4, 1) if t_q4 else "",
                     attn16_us=round(t_a16, 1), attn8_us=round(t_a8, 1) if t_a8 else "", attn4_us=round(t_a4, 1) if t_a4 else "",
                     fp16_us=round(fp16tot, 1), int8_us=round(int8tot, 1), int4_us=round(int4tot, 1),
                     fp16_us_per_step=round(fp16tot * cnt, 1), int8_us_per_step=round(int8tot * cnt, 1),
                     int4_us_per_step=round(int4tot * cnt, 1),
                     int8_vs_fp16=round(fp16tot / int8tot, 3), int4_vs_fp16=round(fp16tot / int4tot, 3),
                     relL2_int8=round(r8, 4) if r8 is not None else "", relL2_int4=round(r4, 4) if r4 is not None else ""))
    tot["gn"] += cnt * t_gn; tot["fp16"] += cnt * fp16tot; tot["int8"] += cnt * int8tot; tot["int4"] += cnt * int4tot

print(f"\n=== attention per-step total (GN+quant+attn, ms/step) ===")
for k, lab in (("fp16", "fp16"), ("int8", "int8"), ("int4", "int4")):
    print(f"  {lab:6} {tot[k]/1000:8.3f} ms/step  {tot['fp16']/tot[k]:.2f}x")
print(f"  [GroupNorm alone (shared): {tot['gn']/1000:.3f} ms/step]")
rows.append(dict(C="TOTAL_PER_STEP", nh="", hd="", T="", count_per_step=sum(s["count"] for s in attn),
                 flash_eligible="", gn_us=round(tot["gn"], 1), q8_us="", q4_us="", attn16_us="", attn8_us="", attn4_us="",
                 fp16_us="", int8_us="", int4_us="",
                 fp16_us_per_step=round(tot["fp16"], 1), int8_us_per_step=round(tot["int8"], 1),
                 int4_us_per_step=round(tot["int4"], 1),
                 int8_vs_fp16=round(tot["fp16"] / tot["int8"], 3), int4_vs_fp16=round(tot["fp16"] / tot["int4"], 3),
                 relL2_int8="", relL2_int4=""))
cols = ["C", "nh", "hd", "T", "count_per_step", "flash_eligible", "gn_us", "q8_us", "q4_us",
        "attn16_us", "attn8_us", "attn4_us", "fp16_us", "int8_us", "int4_us",
        "fp16_us_per_step", "int8_us_per_step", "int4_us_per_step",
        "int8_vs_fp16", "int4_vs_fp16", "relL2_int8", "relL2_int4"]
with open(f"{HERE}/data/attn_kernel_speed.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"WROTE {HERE}/data/attn_kernel_speed.csv")
