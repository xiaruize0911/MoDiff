"""GEMM-only ceiling for the qkv/proj linear shapes: in production the activation
quantize is fused into GroupNorm->qkv (group_norm_silu_quantize), so the realistic
int GEMM cost EXCLUDES the standalone quantize_act_int8/int4 that _gemm includes.
Here we pre-quantize once and time only the GEMM kernel, vs fp16 F.linear.
Writes data/linear_gemm_only_b<B>.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from integration.kernels.wxax_linear import QuantLinearWxAx

B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0); dev = "cuda"
SHAPES = [(192, 1024, 5), (384, 256, 5), (384, 64, 5), (768, 16, 5), (768, 4, 1)]

def bench(fn, it=200, warm=50, reps=5):
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
print(f"GEMM-only vs quantize-included, qkv/proj @ b{B}")
print(f"{'K->N':>12} {'M':>8} {'cnt':>3} | {'fp16':>7} {'i8 gemm':>7} {'i8 full':>7} {'i4 gemm':>7} {'i4 full':>7} | {'i8g/fp16':>8} {'i4g/fp16':>8}")
for (C, T, cnt) in SHAPES:
    M = B * T
    for (kind, K, Nout) in [("qkv", C, 3 * C), ("proj", C, C)]:
        x = torch.randn(M, K, device=dev, dtype=torch.float16)
        lin = nn.Linear(K, Nout).to(dev).half()
        t_fp16 = bench(lambda: F.linear(x, lin.weight, lin.bias))
        rec = dict(kind=kind, K=K, N=Nout, M=M, count=cnt, fp16_us=round(t_fp16, 1))
        for bits, gk, fk in ((8, "i8_gemm", "i8_full"), (4, "i4_gemm", "i4_full")):
            ql = QuantLinearWxAx(lin, bits).to(dev); a = x.abs().max().item() / (127.0 if bits == 8 else 7.0); ql.set_a_scale(a)
            xp = F.pad(x, (0, ql._awqt_K - K)).contiguous() if ql._awqt_K != K else x
            if bits == 8:
                xq = mc.quantize_act_int8(xp, a)
                gemm = (lambda xq=xq, ql=ql, a=a, Nout=Nout: mc.gemm_w8a8_awq_nout(xq, ql.qweight, ql.w_scale, a, Nout)) if ql._awqt_N != Nout \
                    else (lambda xq=xq, ql=ql, a=a: mc.gemm_w8a8_awq(xq, ql.qweight, ql.w_scale, a))
            else:
                xq = mc.quantize_act_int4_pack(xp, a)
                gemm = (lambda xq=xq, ql=ql, a=a, Nout=Nout: mc.gemm_w4a4_awq_nout(xq, ql.qweight, ql.w_scale, a, ql._awqt_K, Nout)) if ql._awqt_N != Nout \
                    else (lambda xq=xq, ql=ql, a=a: mc.gemm_w4a4_awq(xq, ql.qweight, ql.w_scale, a, ql._awqt_K))
            t_g = bench(gemm)
            t_f = bench(lambda ql=ql, x=x, a=a: ql._gemm(x, a))
            rec[gk] = round(t_g, 1); rec[fk] = round(t_f, 1)
        rows.append(rec)
        for kk in tot: tot[kk] += cnt * rec[{"fp16": "fp16_us", "i8_gemm": "i8_gemm", "i8_full": "i8_full", "i4_gemm": "i4_gemm", "i4_full": "i4_full"}[kk]]
        print(f"{f'{K}->{Nout}':>12} {M:8d} {cnt:3d} | {rec['fp16_us']:7.1f} {rec['i8_gemm']:7.1f} {rec['i8_full']:7.1f} {rec['i4_gemm']:7.1f} {rec['i4_full']:7.1f} | {t_fp16/rec['i8_gemm']:7.2f}x {t_fp16/rec['i4_gemm']:7.2f}x")

print(f"\n=== weighted total qkv/proj per forward @ b{B} ===")
print(f"fp16                         {tot['fp16']:8.1f} us  1.00x")
print(f"int8 GEMM-only (fused quant) {tot['i8_gemm']:8.1f} us  {tot['fp16']/tot['i8_gemm']:.2f}x")
print(f"int8 full (+standalone quant){tot['i8_full']:8.1f} us  {tot['fp16']/tot['i8_full']:.2f}x")
print(f"int4 GEMM-only (fused quant) {tot['i4_gemm']:8.1f} us  {tot['fp16']/tot['i4_gemm']:.2f}x")
print(f"int4 full (+standalone quant){tot['i4_full']:8.1f} us  {tot['fp16']/tot['i4_full']:.2f}x")
with open(f"docs/flash_attention_2026-07-19/data/linear_gemm_only_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"WROTE data/linear_gemm_only_b{B}.csv")
