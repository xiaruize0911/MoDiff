"""Layer-level roofline profile: measure the actual production kernels (qkv/proj GEMM; attention
QKᵀ / softmax / AV) in fp16/int8/int4 at the real churches shapes (batch 64), and compare to the
A40 compute roofline (FLOPs / peak-TOPS) and memory roofline (bytes / bandwidth). Efficiency =
roofline_us / measured_us. Writes data/roofline_gemm_b64.csv and data/roofline_attn_b64.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc

# ---- A40 (GA102) peaks ----
TF = {"fp16": 149.7e12, "int8": 299.3e12, "int4": 598.7e12}   # tensor-core OP/s
BW = 696e9                                                    # HBM bytes/s
EB = {"fp16": 2, "int8": 1, "int4": 0.5}                      # bytes/element


def bench(fn, it=200, warm=50, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)   # us
    ts.sort(); return ts[len(ts) // 2]


def pack4(q):
    q = q.to(torch.int8); lo = q[..., 0::2] & 0xF; hi = q[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8).contiguous()


# ================= 1. qkv/proj GEMM: [M,K]x[N,K]->[M,N] =================
GEMM = [  # (name, M, K, N, count)  -- real b64 qkv/proj shapes
    ("32² C192 qkv", 65536, 192, 576, 5), ("32² C192 proj", 65536, 192, 192, 5),
    ("16² C384 qkv", 16384, 384, 1152, 5), ("16² C384 proj", 16384, 384, 384, 5),
    ("8² C384 qkv", 4096, 384, 1152, 5), ("8² C384 proj", 4096, 384, 384, 5),
    ("4² C768 qkv", 1024, 768, 2304, 5), ("4² C768 proj", 1024, 768, 768, 5),
]
grows = []
print("== GEMM roofline (measured us | compute-roofline | mem-roofline | eff%) ==")
for (nm, M, K, N, cnt) in GEMM:
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    Wf = torch.randn(N, K, device="cuda", dtype=torch.float16)
    flops = 2.0 * M * K * N
    for prec in ("fp16", "int8", "int4"):
        if prec == "fp16":
            t = bench(lambda: F.linear(x, Wf))
            byts = (M * K + K * N + M * N) * EB["fp16"]
        elif prec == "int8":
            Kp = ((K + 63) // 64) * 64; Np = ((N + 127) // 128) * 128
            xq = torch.randint(-127, 127, (M, Kp), device="cuda", dtype=torch.int8)
            Wq = torch.randint(-127, 127, (Np, Kp), device="cuda", dtype=torch.int8)
            ws = torch.randn(Np, device="cuda").abs().float() / 127
            t = bench(lambda: mc.gemm_w8a8_awq(xq, Wq, ws, 0.01))
            byts = (M * K + K * N) * 1 + M * N * 2       # int8 in, fp16 out
        else:
            Kp = ((K + 127) // 128) * 128; Np = ((N + 127) // 128) * 128
            xq = pack4(torch.randint(-7, 7, (M, Kp), device="cuda", dtype=torch.int8))
            Wq = pack4(torch.randint(-7, 7, (Np, Kp), device="cuda", dtype=torch.int8))
            ws = torch.randn(Np, device="cuda").abs().float() / 7
            t = bench(lambda: mc.gemm_w4a4_awq(xq, Wq, ws, 0.01, Kp))
            byts = (M * K + K * N) * 0.5 + M * N * 2     # int4 in, fp16 out
        comp = flops / TF[prec] * 1e6                    # us
        mem = byts / BW * 1e6                            # us
        rf = max(comp, mem)
        grows.append(dict(shape=nm, M=M, K=K, N=N, count=cnt, prec=prec, meas_us=round(t, 2),
                          compute_us=round(comp, 2), mem_us=round(mem, 2), roofline_us=round(rf, 2),
                          bound=("compute" if comp > mem else "memory"), eff_pct=round(rf / t * 100, 1)))
    r8 = grows[-2]; print(f"  {nm:16s} int8 {r8['meas_us']:7.1f} | c{r8['compute_us']:6.1f} m{r8['mem_us']:6.1f} | {r8['bound']:6s} {r8['eff_pct']:5.1f}%")

with open("docs/layer_roofline_2026-07-19/data/roofline_gemm_b64.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(grows[0].keys())); w.writeheader(); w.writerows(grows)

# ================= 2. attention QKᵀ / softmax / AV =================
# BH = b*nh = 64*8 = 512; (label, T, hd, count)
ATTN = [("32² T1024 hd24", 1024, 24, 5), ("16² T256 hd48", 256, 48, 5),
        ("8² T64 hd48", 64, 48, 5), ("4² T16 hd96", 16, 96, 5), ("2² T4 hd96", 4, 96, 1)]
BH = 512
arows = []
print("\n== attention roofline (per op: measured us | roofline | bound | eff%) ==")
for (nm, T, hd, cnt) in ATTN:
    scale = 1.0 / math.sqrt(hd)
    hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    flops_qk = 2.0 * BH * T * T * hd    # QKᵀ
    flops_av = 2.0 * BH * T * T * hd    # AV

    def add(op, prec, t, flops, byts, comp_prec):
        comp = flops / TF[comp_prec] * 1e6 if flops else 0.0
        mem = byts / BW * 1e6
        rf = max(comp, mem)
        arows.append(dict(shape=nm, T=T, hd=hd, count=cnt, op=op, prec=prec, meas_us=round(t, 2),
                          compute_us=round(comp, 2), mem_us=round(mem, 2), roofline_us=round(rf, 2),
                          bound=("compute" if comp > mem else "memory"), eff_pct=round(rf / t * 100, 1) if t else 0))
        return rf, comp > mem

    # -- fp16 path (bmm QK, softmax, bmm AV) --
    Sf16 = torch.randn(BH, T, T, device="cuda", dtype=torch.float16)
    add("QKᵀ", "fp16", bench(lambda: torch.bmm(Q, K.transpose(1, 2))),
        flops_qk, (2 * BH * T * hd + BH * T * T) * 2, "fp16")
    add("softmax", "fp16", bench(lambda: F.softmax(Sf16, -1)), 0, BH * T * T * 2 * 2, "fp16")
    add("AV", "fp16", bench(lambda: torch.bmm(Sf16, V)), flops_av, (BH * T * T + BH * T * hd) * 2 + BH * T * hd * 2, "fp16")

    # -- int8 path (needs T%64==0; else the real model SDPA-falls-back) --
    if T % 64 == 0:
        qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hp, hpa, 8)
        add("QKᵀ", "int8", bench(lambda: mc.attn_qk_int8(qi, ki, sq, sk, scale)),
            flops_qk, (2 * BH * T * hp) * 1 + BH * T * T * 2, "int8")    # int8 Q/K in, fp16 S out
        S8 = mc.attn_qk_int8(qi, ki, sq, sk, scale)
        add("softmax", "int8", bench(lambda: mc.attn_softmax_requant(S8)), 0, BH * T * T * 2 + BH * T * T * 1, "int8")
        P8, sp8 = mc.attn_softmax_requant(S8)
        add("AV", "int8", bench(lambda: mc.attn_av_int8(P8, vt, sp8, sv)),
            flops_av, (BH * T * T + BH * hpa * T) * 1 + BH * T * hd * 2, "int8")

    # -- int4 path (needs T%64==0 and hp4%64==0) --
    if T % 64 == 0:
        hp4 = (hd + 63) // 64 * 64
        qi4, ki4, vt4, sq4, sk4, sv4 = mc.quantize_attn_qkv(Q, K, V, hp4, hpa, 4)
        add("QKᵀ", "int4", bench(lambda: mc.attn_qk_int4(qi4, ki4, hp4, sq4, sk4, scale)),
            flops_qk, (2 * BH * T * hp4) * 0.5 + BH * T * T * 2, "int4")
        S4 = mc.attn_qk_int4(qi4, ki4, hp4, sq4, sk4, scale)
        add("softmax", "int4", bench(lambda: mc.attn_softmax_requant4(S4)), 0, BH * T * T * 2 + BH * T * T * 0.5, "int4")
        P4, sp4 = mc.attn_softmax_requant4(S4)
        add("AV", "int4", bench(lambda: mc.attn_av_int4(P4, vt4, sp4, sv4, T)),
            flops_av, (BH * T * T * 0.5 + BH * hpa * T * 0.5) + BH * T * hd * 2, "int4")
    r = {x['op'] + x['prec']: x for x in arows if x['shape'] == nm}
    if 'QKᵀint8' in r:
        print(f"  {nm:16s} QKᵀ i8 {r['QKᵀint8']['meas_us']:6.1f}({r['QKᵀint8']['bound']} {r['QKᵀint8']['eff_pct']:.0f}%) "
              f"smax i8 {r['softmaxint8']['meas_us']:6.1f}({r['softmaxint8']['eff_pct']:.0f}%) "
              f"AV i8 {r['AVint8']['meas_us']:6.1f}({r['AVint8']['eff_pct']:.0f}%)  [fp16 QKᵀ {r['QKᵀfp16']['meas_us']:.1f}]")
    else:
        print(f"  {nm:16s} (T<64: quantized attn N/A -> fp16; QKᵀ fp16 {r['QKᵀfp16']['meas_us']:.1f}us)")

with open("docs/layer_roofline_2026-07-19/data/roofline_attn_b64.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(arows[0].keys())); w.writeheader(); w.writerows(arows)
print("\nWROTE roofline_gemm_b64.csv, roofline_attn_b64.csv")
