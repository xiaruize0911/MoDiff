"""Kernel-level benchmark + profile across EVERY attention AND linear (qkv/proj) shape
the churches UNet runs, at the benchmark batch (default b128). Flash is gone — attention
is fp16 MATH; the quantized option is the materialized int8/int4 attention.

ATTENTION (per shape): fp16 MATH SDPA (the real) vs int8 / int4 MATERIALIZED quant attention
  (quantize_attn_qkv -> attn_qk_int{8,4} -> softmax_requant -> attn_av_int{8,4}).
LINEAR (qkv C->3C, proj C->C, per shape): fp16 F.linear (the real) vs int8 / int4 AWQ GEMM
  (QuantLinearWxAx._gemm = quantize + gemm_w{8,4}a{8,4}_awq), the production Linear path.

Each path: kernel time (us), rel-L2 vs fp16, speedup vs the real. Then weight by per-forward
block counts -> expected total time per forward per family and speedup vs real.

Writes data/kernel_attn_b<B>.csv, data/kernel_linear_b<B>.csv, data/kernel_policy_b<B>.csv
Usage: python kernel_bench_all_shapes.py [batch]   (default 128)
"""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
from integration.kernels.wxax_linear import QuantLinearWxAx

B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0); dev = "cuda"

# (C, nh, hd, T, count) captured from the UNet (capture_attn_shapes.py); nh=8 everywhere.
SHAPES = [
    (192, 8, 24, 1024, 5),
    (384, 8, 48,  256, 5),
    (384, 8, 48,   64, 5),
    (768, 8, 96,   16, 5),
    (768, 8, 96,    4, 1),
]

def bench(fn, it=100, warm=30, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)  # us
    ts.sort(); return ts[len(ts) // 2]

def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)

# clock burn-in
_burn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _burn = _burn @ _burn * 1e-4 + 1.0
torch.cuda.synchronize()

# ============================ ATTENTION ============================
attn_rows = []
attn_shp = {}
print(f"===== ATTENTION kernels @ b{B} (BH=b*nh=1024) =====")
print(f"{'hd/T':>10} {'count':>5} {'path':26} {'us':>9} {'vs_real':>7} {'relL2':>7}")
for (C, nh, hd, T, count) in SHAPES:
    N = B; H = nh; BH = N * H; sc = 1.0 / math.sqrt(hd)
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
    S = torch.einsum("nhid,nhjd->nhij", q.float(), k.float()) * sc
    ref = torch.einsum("nhij,nhjd->nhid", torch.softmax(S, -1), v.float())
    tag = f"{hd}/{T}"; shp = {}

    with sdpa_kernel(SDPBackend.MATH):
        t_math = bench(lambda: F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc))
    shp["fp16 MATH (real)"] = t_math

    def emit(family_rows, family_shp, tagx, cnt, path, us, rel):
        real = family_shp["fp16 MATH (real)"] if family_rows is attn_rows else family_shp["fp16 (real)"]
        vs = real / us if us else float("nan")
        print(f"{tagx:>10} {cnt:5d} {path:26} {us:9.1f} {vs:6.2f}x {str(rel):>7}")
        family_rows.append(dict(hd_or_K=tagx, count=cnt, path=path, us=round(us, 1),
                                vs_real=round(vs, 3), relL2=rel))
    emit(attn_rows, shp, tag, count, "fp16 MATH (real)", t_math, "")

    # int8 / int4 materialized quant attention
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64; hp4 = (hd + 63) // 64 * 64
    try:
        qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
        def i8():
            Sm = mc.attn_qk_int8(qi, ki, sq, sk, sc); P, sp = mc.attn_softmax_requant(Sm); return mc.attn_av_int8(P, vt, sp, sv)
        r8 = relL2(i8().reshape(BH, T, -1)[..., :hd].reshape(N, H, T, hd), ref)
        t8 = bench(i8); shp["int8 mat attn"] = t8
        emit(attn_rows, shp, tag, count, "int8 materialized attn", t8, round(r8, 4))
    except Exception as ex:
        shp["int8 mat attn"] = None; print(f"{tag:>10} {count:5d} {'int8 materialized attn':26}  N/A ({type(ex).__name__})")
    try:
        qi4, ki4, vt4, sq4, sk4, sv4 = mc.quantize_attn_qkv(qm, km, vm, hp4, hpa, 4)
        def i4():
            Sm = mc.attn_qk_int4(qi4, ki4, hp4, sq4, sk4, sc); P, sp = mc.attn_softmax_requant4(Sm); return mc.attn_av_int4(P, vt4, sp, sv4, T)
        r4 = relL2(i4().reshape(BH, T, -1)[..., :hd].reshape(N, H, T, hd), ref)
        t4 = bench(i4); shp["int4 mat attn"] = t4
        emit(attn_rows, shp, tag, count, "int4 materialized attn", t4, round(r4, 4))
    except Exception as ex:
        shp["int4 mat attn"] = None; print(f"{tag:>10} {count:5d} {'int4 materialized attn':26}  N/A ({type(ex).__name__})")
    attn_shp[(C, nh, hd, T, count)] = shp

# ============================ LINEAR (qkv/proj) ============================
lin_rows = []
lin_shp = {}   # (K,N,M,count) -> dict
print(f"\n===== LINEAR (qkv/proj) GEMM kernels @ b{B} =====")
print(f"{'K->N':>12} {'M':>8} {'count':>5} {'path':22} {'us':>9} {'vs_real':>7} {'relL2':>7}")
def lin_layers():
    out = []
    for (C, nh, hd, T, count) in SHAPES:
        M = B * T
        out.append(("qkv", C, 3 * C, M, count))
        out.append(("proj", C, C, M, count))
    return out
for (kind, K, Nout, M, count) in lin_layers():
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    lin = nn.Linear(K, Nout).to(dev).half()
    W = lin.weight; b = lin.bias
    def fp16(): return F.linear(x, W, b)
    ref = fp16().float()
    tag = f"{K}->{Nout}"; shp = {"fp16 (real)": bench(fp16)}
    emit(lin_rows, shp, tag, count, "fp16 (real)", shp["fp16 (real)"], "")
    lin_rows[-1]["M"] = M; lin_rows[-1]["kind"] = kind
    for bits, name in ((8, "int8 AWQ GEMM"), (4, "int4 AWQ GEMM")):
        try:
            ql = QuantLinearWxAx(lin, bits).to(dev)
            a_scale = x.abs().max().item() / (127.0 if bits == 8 else 7.0)
            ql.set_a_scale(a_scale)
            out = ql._gemm(x, a_scale)[:, :Nout].float() + (b.float() if b is not None else 0)
            # NOTE: _gemm excludes bias; add for the rel check only
            rel = relL2(ql._gemm(x, a_scale)[:, :Nout].float() + (b.float() if b is not None else 0), ref)
            t = bench(lambda: ql._gemm(x, a_scale)); shp[name] = t
            emit(lin_rows, shp, tag, count, name, t, round(rel, 4))
            lin_rows[-1]["M"] = M; lin_rows[-1]["kind"] = kind
        except Exception as ex:
            shp[name] = None; print(f"{tag:>12} {M:8d} {count:5d} {name:22}  N/A ({type(ex).__name__})")
    lin_shp[(kind, K, Nout, M, count)] = shp

# ============================ POLICY AGGREGATION ============================
def total(shps, real_key, pick_key):
    tot = 0.0
    for key, shp in shps.items():
        cnt = key[-1]
        us = shp.get(pick_key) or shp[real_key]
        tot += cnt * us
    return tot

a_real = total(attn_shp, "fp16 MATH (real)", "fp16 MATH (real)")
a_i8 = total(attn_shp, "fp16 MATH (real)", "int8 mat attn")
a_i4 = total(attn_shp, "fp16 MATH (real)", "int4 mat attn")
l_real = total(lin_shp, "fp16 (real)", "fp16 (real)")
l_i8 = total(lin_shp, "fp16 (real)", "int8 AWQ GEMM")
l_i4 = total(lin_shp, "fp16 (real)", "int4 AWQ GEMM")

print(f"\n===== EXPECTED total per forward (sum over blocks) @ b{B} =====")
print(f"{'family / policy':40} {'us/fwd':>10} {'vs real':>8}")
def pr(n, us, real): print(f"{n:40} {us:10.1f} {real/us:7.2f}x")
print("-- ATTENTION (21 blocks) --")
pr("fp16 MATH (real)", a_real, a_real); pr("int8 materialized attn", a_i8, a_real); pr("int4 materialized attn", a_i4, a_real)
print("-- LINEAR qkv/proj (42 GEMMs) --")
pr("fp16 (real)", l_real, l_real); pr("int8 AWQ GEMM", l_i8, l_real); pr("int4 AWQ GEMM", l_i4, l_real)
print("-- ATTENTION + LINEAR combined --")
pr("fp16 everywhere (real)", a_real + l_real, a_real + l_real)
pr("int8 linear + fp16 attn (shipped int8)", l_i8 + a_real, a_real + l_real)
pr("int4 linear + fp16 attn (shipped int4)", l_i4 + a_real, a_real + l_real)

D = "docs/flash_attention_2026-07-19/data"
with open(f"{D}/kernel_attn_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(attn_rows[0].keys())); w.writeheader(); w.writerows(attn_rows)
with open(f"{D}/kernel_linear_b{B}.csv", "w", newline="") as f:
    keys = ["kind", "hd_or_K", "M", "count", "path", "us", "vs_real", "relL2"]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(lin_rows)
prows = [
    dict(family="attention", policy="fp16 MATH (real)", us_per_fwd=round(a_real, 1), vs_real=1.0),
    dict(family="attention", policy="int8 materialized", us_per_fwd=round(a_i8, 1), vs_real=round(a_real / a_i8, 3)),
    dict(family="attention", policy="int4 materialized", us_per_fwd=round(a_i4, 1), vs_real=round(a_real / a_i4, 3)),
    dict(family="linear", policy="fp16 (real)", us_per_fwd=round(l_real, 1), vs_real=1.0),
    dict(family="linear", policy="int8 AWQ", us_per_fwd=round(l_i8, 1), vs_real=round(l_real / l_i8, 3)),
    dict(family="linear", policy="int4 AWQ", us_per_fwd=round(l_i4, 1), vs_real=round(l_real / l_i4, 3)),
    dict(family="attn+linear", policy="int8 linear+fp16 attn", us_per_fwd=round(l_i8 + a_real, 1), vs_real=round((a_real + l_real) / (l_i8 + a_real), 3)),
    dict(family="attn+linear", policy="int4 linear+fp16 attn", us_per_fwd=round(l_i4 + a_real, 1), vs_real=round((a_real + l_real) / (l_i4 + a_real), 3)),
]
with open(f"{D}/kernel_policy_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prows[0].keys())); w.writeheader(); w.writerows(prows)
print(f"\nWROTE kernel_attn_b{B}.csv, kernel_linear_b{B}.csv, kernel_policy_b{B}.csv")
