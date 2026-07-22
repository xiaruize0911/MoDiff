"""Fuse the qkv activation-quantize into the GroupNorm kernel (GN->qkv path), and validate it.

The model's qkv path currently runs, per attention block:
    GroupNorm (fp16 kernel) -> QuantLinearWxAx.forward  [ = quantize_act_int8(GN_out) + AWQ W8A8 GEMM ]
i.e. the activation quantize is a SEPARATE kernel. The conv path already fuses GN+quantize via
group_norm_silu_quantize_nhwc -> forward_from_int8; the qkv path never reused it.

FUSED path (this script): emit the int8 qkv-GEMM input straight out of the GroupNorm kernel:
    q_i8 = group_norm_silu_quantize_nhwc(x, gn_w, gn_b, NG, eps, apply_silu=False, scale=127/absmax)
    out  = gemm_w8a8_awq_bias_res(q_i8_tokens, qweight, w_scale, a, N_out, bias, -)
All attention C in {192,384,768} are multiples of 64 -> the int8 activation needs no K-pad, so the
GN-kernel output feeds the AWQ GEMM directly. Validates rel-L2 vs the non-fused path (must be ~0) and
vs an fp32 reference, then times the front-end (GN + quantize + qkv GEMM) non-fused vs fused, per shape
and per DDIM step (x count). int8 only (quality-safe; C%64==0 so no pad). Writes data/fuse_gn_qkv_quant.csv.
"""
import os, sys, csv, math
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "docs/benchmark_5mode_2026-07-21/scripts")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
from integration.kernels.wxax_linear import QuantLinearWxAx
from integration.fused_ops.fused_resblock import _group_norm_silu
import shapes as S

torch.manual_seed(0); dev = "cuda"
B = 128; NG = 32; EPS = 1e-5
HERE = "docs/benchmark_5mode_2026-07-21"
# qkv attention blocks: (C, T, Hspatial, count/step) from the real-model enumeration
ATTN = [(a["C"], a["T"], a["Hspatial"], a["count"]) for a in sorted(S.attn_shapes(), key=lambda s: -s["T"])]


def bench(fn, it=200, warm=50, reps=5):
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
tot = dict(fp16=0.0, nonfused=0.0, fused=0.0, quant=0.0)
print(f"Fuse GN->qkv activation-quantize (int8) @ b{B}  (front-end = GN + quantize + qkv GEMM, us/call)")
print(f"{'C/T':>10}{'x':>4} | {'relL2':>8}{'relL2vs':>9} | {'fp16 FE':>8}{'nonfus':>8}{'quant':>7}{'fused':>7} | {'fus/nonf':>9}{'save/step':>10}")
for (C, T, Hs, cnt) in ATTN:
    N = B; M = N * T; Nout = 3 * C
    x = torch.randn(N, C, Hs, Hs, device=dev, dtype=torch.float16).to(memory_format=torch.channels_last)
    gw = torch.randn(C, device=dev, dtype=torch.float16); gb = torch.randn(C, device=dev, dtype=torch.float16)
    lin = nn.Linear(C, Nout).to(dev).half()
    ql = QuantLinearWxAx(lin, 8).to(dev)
    empty = x.new_empty(0)

    # GN (fp16, no SiLU) -> tokens; calibrate static activation scale a = absmax/127
    xn = _group_norm_silu(x, NG, gw, gb, EPS, False)                 # [N,C,Hs,Hs] fp16 CL
    xn_tok = xn.permute(0, 2, 3, 1).reshape(M, C).contiguous()
    a = (xn.abs().max().item() / 127.0) or 1e-8
    ql.set_a_scale(a)
    scale_mult = torch.tensor([1.0 / a], device=dev, dtype=torch.float32)   # 127/absmax (multiplier)

    # ---- correctness ----
    out_ref = ql.forward(xn_tok)                                      # model non-fused: quantize + AWQ GEMM(+bias)
    q_i8 = mc.group_norm_silu_quantize_nhwc(x, gw, gb, NG, EPS, False, scale_mult, empty, empty, empty)
    q_tok = q_i8.permute(0, 2, 3, 1).reshape(M, C).contiguous()
    out_fused = mc.gemm_w8a8_awq_bias_res(q_tok, ql.qweight, ql.w_scale, a, Nout, ql.bias, empty)
    out_fp32 = F.linear(xn.permute(0, 2, 3, 1).reshape(M, C).float(), lin.weight.float(), lin.bias.float())
    rl_vs_nonfused = relL2(out_fused, out_ref)
    rl_vs_fp32 = relL2(out_fused, out_fp32)

    # ---- timing ----
    t_fp16 = bench(lambda: F.linear(_group_norm_silu(x, NG, gw, gb, EPS, False).permute(0, 2, 3, 1).reshape(M, C), lin.weight, lin.bias))
    def nonfused():
        xn_ = _group_norm_silu(x, NG, gw, gb, EPS, False)
        return ql.forward(xn_.permute(0, 2, 3, 1).reshape(M, C).contiguous())
    t_nonfused = bench(nonfused)
    t_quant = bench(lambda: mc.quantize_act_int8(xn_tok, a))          # the separate quantize the fusion removes
    def fused():
        qi = mc.group_norm_silu_quantize_nhwc(x, gw, gb, NG, EPS, False, scale_mult, empty, empty, empty)
        return mc.gemm_w8a8_awq_bias_res(qi.permute(0, 2, 3, 1).reshape(M, C).contiguous(), ql.qweight, ql.w_scale, a, Nout, ql.bias, empty)
    t_fused = bench(fused)

    tot["fp16"] += cnt * t_fp16; tot["nonfused"] += cnt * t_nonfused
    tot["fused"] += cnt * t_fused; tot["quant"] += cnt * t_quant
    rows.append(dict(C=C, T=T, count_per_step=cnt, relL2_vs_nonfused=round(rl_vs_nonfused, 5),
                     relL2_vs_fp32=round(rl_vs_fp32, 4), fp16_fe_us=round(t_fp16, 1),
                     nonfused_fe_us=round(t_nonfused, 1), quant_us=round(t_quant, 1), fused_fe_us=round(t_fused, 1),
                     fused_vs_nonfused=round(t_nonfused / t_fused, 3),
                     save_per_step_us=round((t_nonfused - t_fused) * cnt, 1)))
    print(f"{f'{C}/{T}':>10}{cnt:>4} | {rl_vs_nonfused:8.5f}{rl_vs_fp32:9.4f} | {t_fp16:8.1f}{t_nonfused:8.1f}{t_quant:7.1f}{t_fused:7.1f} | "
          f"{t_nonfused/t_fused:8.2f}x{(t_nonfused-t_fused)*cnt:9.1f}")

print(f"\n=== qkv front-end per step (GN + quantize + qkv GEMM, ms/step) ===")
for k, lab in (("fp16", "fp16"), ("nonfused", "int8 non-fused (model today)"), ("fused", "int8 GN->quant fused")):
    print(f"  {lab:30} {tot[k]/1000:7.3f} ms/step  {tot['fp16']/tot[k]:.2f}x vs fp16")
print(f"  removed separate quantize kernel:  {tot['quant']/1000:.3f} ms/step")
print(f"  fused vs non-fused qkv front-end:  {tot['nonfused']/tot['fused']:.2f}x  (saves {(tot['nonfused']-tot['fused'])/1000:.3f} ms/step)")
rows.append(dict(C="TOTAL_PER_STEP", T="", count_per_step=sum(a[3] for a in ATTN), relL2_vs_nonfused="",
                 relL2_vs_fp32="", fp16_fe_us=round(tot["fp16"], 1), nonfused_fe_us=round(tot["nonfused"], 1),
                 quant_us=round(tot["quant"], 1), fused_fe_us=round(tot["fused"], 1),
                 fused_vs_nonfused=round(tot["nonfused"] / tot["fused"], 3),
                 save_per_step_us=round(tot["nonfused"] - tot["fused"], 1)))
cols = ["C", "T", "count_per_step", "relL2_vs_nonfused", "relL2_vs_fp32", "fp16_fe_us",
        "nonfused_fe_us", "quant_us", "fused_fe_us", "fused_vs_nonfused", "save_per_step_us"]
with open(f"{HERE}/data/fuse_gn_qkv_quant.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"WROTE {HERE}/data/fuse_gn_qkv_quant.csv")
