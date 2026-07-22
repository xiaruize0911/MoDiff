"""Linear GEMM speed — EVERY qkv/proj + time-embed linear shape the churches UNet runs, b128.

Shapes + per-step counts from the real-model enumeration (shapes.linear_shapes()). Linear has NO
modiff variant (all modes use static W/A quant), so the 5 modes collapse to 3 distinct linear
kernels: fp16 / int8 / int4  (int8_baseline == int8_modiff, int4_baseline == int4_modiff).

Two linear families in the model:
  * qkv / proj  -> QuantLinearWxAx (W8A8 / W4A4 AWQ GEMM). Since MODIFF_FUSE_GN_QKV=0 in quant modes,
                   the activation quantize runs INSIDE the linear, so the per-step cost is the FULL
                   forward (quantize + GEMM). We report full forward (faithful) AND gemm-only (ref).
  * time-embed  -> OptimizedInt{8,4}Linear, but in_features (192/768) < K_INT8_GATE(2048), so these
    ("other")     K-gate to fp16 F.linear in quant modes -> int8/int4 time == fp16 time.

CUDA-event median, warm 50 + 200 x 5 reps, GPU clock burn-in. Writes data/linear_kernel_speed.csv.
"""
import os, sys, csv
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "docs/benchmark_5mode_2026-07-21/scripts")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from integration.kernels.wxax_linear import QuantLinearWxAx
import shapes as S

torch.manual_seed(0); dev = "cuda"
HERE = "docs/benchmark_5mode_2026-07-21"


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

lins = sorted(S.linear_shapes(), key=lambda s: (s["role"] != "qkv", s["role"] != "proj", -s["M"]))
rows = []
tot = {"fp16": 0.0, "int8": 0.0, "int4": 0.0}
print("Linear GEMM @ b128 (us/call, median 5x200)")
print(f"{'role':6}{'K->N':>12}{'M':>9}{'cnt':>4} | {'fp16':>8}{'i8 full':>9}{'i4 full':>9}{'i8 gemm':>9}{'i4 gemm':>9} | {'i8x':>6}{'i4x':>6}")
for sh in lins:
    role, K, Nout, M, cnt = sh["role"], sh["K"], sh["N"], sh["M"], sh["count"]
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    lin = nn.Linear(K, Nout).to(dev).half()
    t_fp16 = bench(lambda: F.linear(x, lin.weight, lin.bias))
    rec = dict(role=role, K=K, N=Nout, M=M, count_per_step=cnt, fp16_us=round(t_fp16, 1))
    is_attn = role in ("qkv", "proj")     # QuantLinearWxAx (truly quantized)
    for bits, tag in ((8, "int8"), (4, "int4")):
        if is_attn:
            ql = QuantLinearWxAx(lin, bits).to(dev)
            a = x.abs().max().item() / (127.0 if bits == 8 else 7.0); ql.set_a_scale(a)
            t_full = bench(lambda ql=ql, x=x: ql.forward(x))               # what the model runs: quantize + GEMM (+bias/res epi)
            # TRUE gemm-only ("if activation quantize were fused away"): quantize once, time the raw AWQ GEMM.
            xf = F.pad(x, (0, ql._awqt_K - K)).contiguous() if ql._awqt_K != K else x
            if bits == 8:
                t_quant = bench(lambda xf=xf, a=a: mc.quantize_act_int8(xf, a))
                xq = mc.quantize_act_int8(xf, a)
                g = (lambda: mc.gemm_w8a8_awq_nout(xq, ql.qweight, ql.w_scale, a, Nout)) if ql._awqt_N != Nout \
                    else (lambda: mc.gemm_w8a8_awq(xq, ql.qweight, ql.w_scale, a))
            else:
                t_quant = bench(lambda xf=xf, a=a: mc.quantize_act_int4_pack(xf, a))
                xq = mc.quantize_act_int4_pack(xf, a)
                g = (lambda: mc.gemm_w4a4_awq_nout(xq, ql.qweight, ql.w_scale, a, ql._awqt_K, Nout)) if ql._awqt_N != Nout \
                    else (lambda: mc.gemm_w4a4_awq(xq, ql.qweight, ql.w_scale, a, ql._awqt_K))
            t_gemm = bench(g)
        else:
            # time-embed: K < K_INT8_GATE(2048) -> the model K-gates these to fp16, so int8/int4 == fp16.
            t_full = t_gemm = t_fp16; t_quant = 0.0
        rec[f"{tag}_full_us"] = round(t_full, 1)
        rec[f"{tag}_gemmonly_us"] = round(t_gemm, 1)
        rec[f"{tag}_quant_us"] = round(t_quant, 1)
        rec[f"{tag}_vs_fp16"] = round(t_fp16 / t_full, 3) if t_full else ""
        rec[f"{tag}_gemmonly_vs_fp16"] = round(t_fp16 / t_gemm, 3) if t_gemm else ""
    rows.append(rec)
    tot["fp16"] += cnt * t_fp16; tot["int8"] += cnt * rec["int8_full_us"]; tot["int4"] += cnt * rec["int4_full_us"]
    tot["int8_g"] = tot.get("int8_g", 0.0) + cnt * rec["int8_gemmonly_us"]
    tot["int4_g"] = tot.get("int4_g", 0.0) + cnt * rec["int4_gemmonly_us"]
    print(f"{role:6}{f'{K}->{Nout}':>12}{M:>9}{cnt:>4} | {rec['fp16_us']:8.1f}{rec['int8_full_us']:9.1f}"
          f"{rec['int4_full_us']:9.1f}{rec['int8_gemmonly_us']:9.1f}{rec['int4_gemmonly_us']:9.1f} | "
          f"{rec['int8_vs_fp16']:5.2f}x{rec['int4_vs_fp16']:5.2f}x")

print(f"\n=== linear per-step total (full = quantize+GEMM, what the model runs) ===")
roll = dict(role="TOTAL_PER_STEP", K="", N="", M="", count_per_step=sum(s["count"] for s in lins),
            fp16_us=round(tot["fp16"], 1), int8_full_us=round(tot["int8"], 1), int4_full_us=round(tot["int4"], 1),
            int8_gemmonly_us=round(tot["int8_g"], 1), int4_gemmonly_us=round(tot["int4_g"], 1),
            int8_quant_us="", int4_quant_us="")
roll["int8_vs_fp16"] = round(tot["fp16"] / tot["int8"], 3)
roll["int4_vs_fp16"] = round(tot["fp16"] / tot["int4"], 3)
roll["int8_gemmonly_vs_fp16"] = round(tot["fp16"] / tot["int8_g"], 3)
roll["int4_gemmonly_vs_fp16"] = round(tot["fp16"] / tot["int4_g"], 3)
for k, lab in (("fp16", "fp16"), ("int8", "int8 full"), ("int8_g", "int8 gemm-only"),
               ("int4", "int4 full"), ("int4_g", "int4 gemm-only")):
    print(f"  {lab:16} {tot[k]/1000:8.3f} ms/step  {tot['fp16']/tot[k]:.2f}x")
rows.append(roll)

cols = ["role", "K", "N", "M", "count_per_step", "fp16_us",
        "int8_full_us", "int8_gemmonly_us", "int8_quant_us", "int8_vs_fp16", "int8_gemmonly_vs_fp16",
        "int4_full_us", "int4_gemmonly_us", "int4_quant_us", "int4_vs_fp16", "int4_gemmonly_vs_fp16"]
with open(f"{HERE}/data/linear_kernel_speed.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"WROTE {HERE}/data/linear_kernel_speed.csv")
