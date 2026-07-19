"""SPLIT projection stage, architecture-matched (identical 3-op chain on every path: a norm[+quantize]
kernel, a plain bias-less GEMM, then a separate bias-add): isolates the GEMM's own cost, no path gets
a free kernel fusion (e.g. cuBLAS addmm silently fusing bias into the GEMM) that another lacks.

  fp16: group_norm_silu_nhwc (GN, fp16 out) -> cuBLAS matmul (fp16xfp16->fp16) -> + bias
  int8: group_norm_silu_quantize_nhwc (GN+quant fused, int8 out) -> gemm_w8a8 (ours) -> + bias
  AWQ:  group_norm_silu_quantize_nhwc (same kernel as int8)      -> AWQ's w8a8_gemm_forward_cuda -> + bias

This intentionally does NOT compare against fp16's fused_gn_qkv (GN folded into the qkv GEMM's
mainloop): that comparison pits a fused fp16 kernel against a split int8 pipeline, crediting fp16
with an architectural advantage (no separate norm kernel, no intermediate write) that the int8 side
structurally cannot have without giving up its GEMM's native cp.async loader (see NEXT_STEPS.md --
fusing GN+quantize into either our own or AWQ's GEMM mainloop was tried and lost badly, because it
forces a fully synchronous quantizing loader into a pipeline that only gets its speed from cp.async's
zero-compute async copies). Matching the fusion depth (2 kernels, both sides) isolates the actual
question -- is int8 compute cheaper here -- from that confound.

fp16_fused is still measured, but only reported as reference context, not as "the" comparison.
Emits split_stage.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
sys.path.insert(0, "/workspace/llm-awq/awq/kernels")
try: import awq_inference_engine as _awq
except Exception as _e: _awq = None; print("AWQ import failed:", repr(_e)[:200])
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"
DEV = "cuda"

def bench(fn, it=200, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

def rel_err(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)

CFGS = [("C192 qkv", 32, 192, 32, 32), ("C384 qkv", 32, 384, 16, 16), ("C768 qkv", 32, 768, 8, 8)]
EMPTY = torch.empty(0, device=DEV, dtype=torch.float16)
G, eps = 32, 1e-6
rows = []
print(f"{'shape':>10} | {'fp16-split':>10} {'int8-split':>10} {'awq-split':>10} | {'fp16-fused(ref)':>15} | "
      f"{'i8×split':>8} {'awq×split':>9} | {'i8_err':>7} {'awq_err':>7}")
for name, N, C, H, W in CFGS:
    Nout = 3 * C; T = H * W; M = N * T
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    qkv = nn.Linear(C, Nout).to(DEV).half(); gn = nn.GroupNorm(G, C, eps=eps).to(DEV).half()
    Wt, bt = qkv.weight.detach().float(), qkv.bias.detach().float()
    gw, gb = gn.weight.detach().float(), gn.bias.detach().float()
    gwh, gbh = gw.half().contiguous(), gb.half().contiguous()

    # ---- fp16 split: GN (fp16 out) -> cuBLAS matmul -> bias add.
    # Symmetric op count with int8/AWQ below (norm[+quant], GEMM, separate bias-add) --
    # NOT torch.addmm, which would silently fuse the bias into cuBLAS's single GEMM call and
    # give fp16 a free kernel that int8/AWQ's plain (bias-less) GEMMs structurally can't match. ----
    Wf16 = Wt.t().half().contiguous()   # [C, Nout]
    bt16 = bt.half().contiguous()
    def s_fp16_split():
        h = mc.group_norm_silu_nhwc(x, gwh, gbh, G, eps, False, EMPTY, EMPTY)   # fp16 [N,C,H,W] cl
        return torch.matmul(h.permute(0, 2, 3, 1).reshape(M, C), Wf16) + bt16

    # ---- int8 / AWQ split: GN+quantize fused (int8 out) -> GEMM ----
    # gamma folds into the int8 weight (per-output-channel w_scale); beta+qkv.bias -> epi_bias.
    Wf = Wt * gw[None, :]
    xn_pure = F.group_norm(x.float(), G, None, None, eps)
    s_a = float(127.0 / xn_pure.abs().max().clamp_min(1e-8))
    ws = (Wf.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    wi8 = torch.clamp(torch.round(Wf / ws[:, None]), -127, 127).to(torch.int8).contiguous()
    epi_bias = (bt + Wt @ gb).contiguous()
    ones_g = torch.ones(C, device=DEV, dtype=torch.float16); zeros_g = torch.zeros(C, device=DEV, dtype=torch.float16)
    scale_t = torch.tensor([s_a], device=DEV, dtype=torch.float32); smooth = torch.empty(0, device=DEV, dtype=torch.float32)
    def gnq8():
        xi = mc.group_norm_silu_quantize_nhwc(x, ones_g, zeros_g, G, eps, False, scale_t, smooth, EMPTY, EMPTY)
        return xi.permute(0, 2, 3, 1).reshape(M, C)
    def s_int8_split():
        return mc.gemm_w8a8(gnq8(), wi8, ws.contiguous(), 1.0 / s_a) + epi_bias.half()

    Npad = ((Nout + 127) // 128) * 128
    wi8_pad = F.pad(wi8, (0, 0, 0, Npad - Nout)).contiguous()
    ws_half_pad = F.pad(ws.half(), (0, Npad - Nout), value=1.0).contiguous()
    asc = torch.full((M,), 1.0 / s_a, device=DEV, dtype=torch.float16)
    outp = torch.empty(M, Npad, device=DEV, dtype=torch.float16)
    epi_bias_h = epi_bias.half()
    def s_awq_split():
        if _awq is None: return outp
        xq = gnq8()
        _awq.w8a8_gemm_forward_cuda(xq, wi8_pad, ws_half_pad, asc, outp)
        return outp[:, :Nout] + epi_bias_h

    # ---- fp16 fused: reference context only, NOT the comparison ----
    conv_w = Wf.half().view(Nout, 1, 1, C).contiguous()
    SHIFT = 16.0
    epi16 = (bt + Wt @ gb - SHIFT * Wf.sum(dim=1)).half().contiguous()
    def s_fp16_fused(): return mc.fused_gn_qkv(x, conv_w, epi16, G, eps, SHIFT)

    # correctness (vs fp32 GN->Linear reference) for the two int8 paths
    xn = F.group_norm(x.float(), G, gw, gb, eps)
    ref = F.linear(xn.permute(0, 2, 3, 1).reshape(M, C), Wt, bt)
    re8 = rel_err(s_int8_split(), ref)
    rea = rel_err(s_awq_split(), ref) if _awq is not None else float("nan")

    t_fp16sp = bench(s_fp16_split)
    t_int8sp = bench(s_int8_split)
    t_awqsp = bench(s_awq_split) if _awq is not None else float("nan")
    t_fp16fu = bench(s_fp16_fused)
    sp = lambda t: t_fp16sp / t if t == t else float("nan")
    print(f"{name:>10} | {t_fp16sp:10.1f} {t_int8sp:10.1f} {t_awqsp:10.1f} | {t_fp16fu:15.1f} | "
          f"{sp(t_int8sp):8.2f} {sp(t_awqsp):9.2f} | {re8:7.4f} {rea:7.4f}")
    rows.append(dict(shape=name, M=M,
                      fp16_split_us=round(t_fp16sp, 1), int8_split_us=round(t_int8sp, 1),
                      awq_split_us=round(t_awqsp, 1) if t_awqsp == t_awqsp else "",
                      fp16_fused_ref_us=round(t_fp16fu, 1),
                      int8_vs_fp16_split=round(sp(t_int8sp), 3),
                      awq_vs_fp16_split=round(sp(t_awqsp), 3) if t_awqsp == t_awqsp else "",
                      int8_rel_err=round(re8, 4), awq_rel_err=round(rea, 4) if rea == rea else ""))

with open(f"{OUT}/split_stage.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE split_stage.csv")
