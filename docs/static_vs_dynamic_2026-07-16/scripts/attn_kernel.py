"""Full quantized attention kernel micro-benchmark: dynamic vs static, per precision, per real
churches shape. Dynamic = runtime per-token Q/K + per-channel V absmax + per-row softmax max.
Static = calibrated per-tensor Q/K + per-channel V + single softmax c (no runtime reductions).
fp16 = materialized bmm -> softmax(dyn 2-pass / static 1-pass) -> bmm. Emits attn_kernel_speed.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, modiff_cutlass as mc
OUT = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"

def bench(fn, it=30, warm=12):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

SHAPES = [(256, 1024, 24), (256, 256, 48), (256, 64, 96)]
rows = []
print(f"{'BH,T,hd':>14} {'prec':>5} | {'dyn us':>8} {'static us':>9} | s/d")
for (BH, T, hd) in SHAPES:
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(hd)

    def fp16_dyn():
        S = (torch.bmm(Q, K.transpose(1, 2)) * scale).half()
        P, rs = mc.attn_softmax_fp16(S, False, 0.0); return torch.bmm(P, V) / rs.unsqueeze(-1).half()
    # calibrate fp16 static c (global max -> lossless)
    Sf = (torch.bmm(Q, K.transpose(1, 2)) * scale).half(); cf = Sf.float().max().item()
    def fp16_sta():
        S = (torch.bmm(Q, K.transpose(1, 2)) * scale).half()
        P, rs = mc.attn_softmax_fp16(S, True, cf); return torch.bmm(P, V) / rs.unsqueeze(-1).half()

    variants = [("fp16", fp16_dyn, fp16_sta)]
    for bits in (8, 4):
        Qm = 127.0 if bits == 8 else 7.0
        hpq = (hd + 31) // 32 * 32 if bits == 8 else (hd + 63) // 64 * 64
        hpa = (hd + 63) // 64 * 64
        sq_c = Q.abs().max().item() / Qm; sk_c = K.abs().max().item() / Qm
        sv_vec = torch.ones(hpa, device="cuda"); sv_vec[:hd] = (V.abs().amax(dim=(0, 1)).float() / Qm).clamp_min(1e-8)
        qi2, ki2, vt2, sq2, sk2, sv2 = mc.quantize_attn_qkv_static(Q, K, V, hpq, hpa, bits, sq_c, sk_c, sv_vec)
        Ss = mc.attn_qk_int8(qi2, ki2, sq2, sk2, scale) if bits == 8 else mc.attn_qk_int4(qi2, ki2, hpq, sq2, sk2, scale)
        c = Ss.float().amax(-1).mean().item()

        def dyn(bits=bits, hpq=hpq, hpa=hpa):
            qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hpq, hpa, bits)
            if bits == 8:
                S = mc.attn_qk_int8(qi, ki, sq, sk, scale); P, sp = mc.attn_softmax_requant(S); return mc.attn_av_int8(P, vt, sp, sv)
            S = mc.attn_qk_int4(qi, ki, hpq, sq, sk, scale); P, sp = mc.attn_softmax_requant4(S); return mc.attn_av_int4(P, vt, sp, sv, T)

        def sta(bits=bits, hpq=hpq, hpa=hpa, sq_c=sq_c, sk_c=sk_c, sv_vec=sv_vec, c=c):
            qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv_static(Q, K, V, hpq, hpa, bits, sq_c, sk_c, sv_vec)
            if bits == 8:
                S = mc.attn_qk_int8(qi, ki, sq, sk, scale); P, sp = mc.attn_softmax_requant_static(S, c); return mc.attn_av_int8(P, vt, sp, sv)
            S = mc.attn_qk_int4(qi, ki, hpq, sq, sk, scale); P, sp = mc.attn_softmax_requant4_static(S, c); return mc.attn_av_int4(P, vt, sp, sv, T)
        variants.append((f"int{bits}", dyn, sta))

    for prec, dyn, sta in variants:
        td = bench(dyn); ts = bench(sta)
        print(f"{BH},{T:>4},{hd:>3} {prec:>5} | {td:8.1f} {ts:9.1f} | {td/ts:5.2f}")
        rows.append(dict(BH=BH, T=T, hd=hd, precision=prec, dyn_us=round(td, 1), static_us=round(ts, 1),
                         static_speedup=round(td / ts, 3)))
with open(f"{OUT}/attn_kernel_speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE attn_kernel_speed.csv")
