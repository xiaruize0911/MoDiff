"""Kernel item 3 (speed) — qkv/proj Linear GEMM across modes, churches shapes, b128.
fp16 F.linear vs int8/int4 AWQ GEMM (QuantLinearWxAx). GEMM-only (activation quantize fused into
upstream GroupNorm in the real model) AND +quant (standalone). NOTE: linear has NO modiff variant
(benchmark_ldm uses static W/A linear for both baseline and modiff), so int8_baseline == int8_modiff
and int4_baseline == int4_modiff here — 3 distinct kernels (fp16/int8/int4). CUDA-event median,
warm 50 + 200 iters x 5 reps, GPU clock burn-in. Writes data/linear_kernel_speed.csv.
"""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from integration.kernels.wxax_linear import QuantLinearWxAx

torch.manual_seed(0); dev = "cuda"
B = 128
HERE = "docs/benchmark_5mode_2026-07-20"
SHAPES = [(192, 1024, 5), (384, 256, 5), (384, 64, 5), (768, 16, 5), (768, 4, 1)]   # (C, T, count)


def bench(fn, it=300, warm=80, reps=7):
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
tot = {"fp16": 0.0, "i8_gemm": 0.0, "i8_full": 0.0, "i4_gemm": 0.0, "i4_full": 0.0}
print(f"Linear qkv/proj GEMM @ b{B} (us, median 5x200)")
print(f"{'K->N':>12} {'M':>8} {'cnt':>3} | {'fp16':>8} {'i8 gemm':>8} {'i8 full':>8} {'i4 gemm':>8} {'i4 full':>8} | {'i8g/fp16':>8} {'i4g/fp16':>8}")
for (C, T, cnt) in SHAPES:
    M = B * T
    for (kind, K, Nout) in [("qkv", C, 3 * C), ("proj", C, C)]:
        x = torch.randn(M, K, device=dev, dtype=torch.float16)
        lin = nn.Linear(K, Nout).to(dev).half()
        t_fp16 = bench(lambda: F.linear(x, lin.weight, lin.bias))
        rec = dict(kind=kind, K=K, N=Nout, M=M, count=cnt, fp16_us=round(t_fp16, 1))
        for bits, gk, fk in ((8, "i8_gemm", "i8_full"), (4, "i4_gemm", "i4_full")):
            ql = QuantLinearWxAx(lin, bits).to(dev)
            a = x.abs().max().item() / (127.0 if bits == 8 else 7.0); ql.set_a_scale(a)
            xp = F.pad(x, (0, ql._awqt_K - K)).contiguous() if ql._awqt_K != K else x
            if bits == 8:
                xq = mc.quantize_act_int8(xp, a)
                gemm = (lambda xq=xq, ql=ql, a=a, Nout=Nout: mc.gemm_w8a8_awq_nout(xq, ql.qweight, ql.w_scale, a, Nout)) if ql._awqt_N != Nout \
                    else (lambda xq=xq, ql=ql, a=a: mc.gemm_w8a8_awq(xq, ql.qweight, ql.w_scale, a))
            else:
                xq = mc.quantize_act_int4_pack(xp, a)
                gemm = (lambda xq=xq, ql=ql, a=a, Nout=Nout: mc.gemm_w4a4_awq_nout(xq, ql.qweight, ql.w_scale, a, ql._awqt_K, Nout)) if ql._awqt_N != Nout \
                    else (lambda xq=xq, ql=ql, a=a: mc.gemm_w4a4_awq(xq, ql.qweight, ql.w_scale, a, ql._awqt_K))
            rec[gk + "_us"] = round(bench(gemm), 1)
            rec[fk + "_us"] = round(bench(lambda ql=ql, x=x, a=a: ql._gemm(x, a)), 1)
        rec["i8_gemm_vs_fp16"] = round(t_fp16 / rec["i8_gemm_us"], 3)
        rec["i4_gemm_vs_fp16"] = round(t_fp16 / rec["i4_gemm_us"], 3)
        rows.append(rec)
        for kk, col in (("fp16", "fp16_us"), ("i8_gemm", "i8_gemm_us"), ("i8_full", "i8_full_us"), ("i4_gemm", "i4_gemm_us"), ("i4_full", "i4_full_us")):
            tot[kk] += cnt * rec[col]
        print(f"{f'{K}->{Nout}':>12} {M:8d} {cnt:3d} | {rec['fp16_us']:8.1f} {rec['i8_gemm_us']:8.1f} {rec['i8_full_us']:8.1f} {rec['i4_gemm_us']:8.1f} {rec['i4_full_us']:8.1f} | {rec['i8_gemm_vs_fp16']:7.2f}x {rec['i4_gemm_vs_fp16']:7.2f}x")

print(f"\n=== weighted total qkv/proj per forward (42 GEMMs) @ b{B} ===")
for k, lab in [("fp16", "fp16"), ("i8_gemm", "int8 GEMM-only"), ("i8_full", "int8 +quant"), ("i4_gemm", "int4 GEMM-only"), ("i4_full", "int4 +quant")]:
    print(f"{lab:20} {tot[k]:9.1f} us  {tot['fp16']/tot[k]:.2f}x")
rows.append(dict(kind="WEIGHTED_TOTAL", K="", N="", M="", count="", fp16_us=round(tot["fp16"], 1),
                 i8_gemm_us=round(tot["i8_gemm"], 1), i8_full_us=round(tot["i8_full"], 1),
                 i4_gemm_us=round(tot["i4_gemm"], 1), i4_full_us=round(tot["i4_full"], 1),
                 i8_gemm_vs_fp16=round(tot["fp16"] / tot["i8_gemm"], 3), i4_gemm_vs_fp16=round(tot["fp16"] / tot["i4_gemm"], 3)))
with open(f"{HERE}/data/linear_kernel_speed.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"WROTE {HERE}/data/linear_kernel_speed.csv")
