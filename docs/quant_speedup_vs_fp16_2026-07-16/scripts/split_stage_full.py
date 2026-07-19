"""FULL split-stage benchmark across every real AWQ-eligible GEMM shape in the UNet (not just the
3 qkv shapes from split_stage.py -- corrects a shape error: C768 was previously benchmarked at a
synthetic T=64; the real model runs C768 at T=16 (level3) and T=4 (middle block); T=64 belongs to
C384's second occurrence). Adds the 5 attention-projection (C,T) combos x{qkv,proj} = 10 shapes, plus
5 time-embedding MLP shapes that run at M=batch-size only (a completely different, tiny-M regime).

Same architecture-matched op count as split_stage.py: norm[+quant] kernel -> plain GEMM -> bias-add
(qkv/proj use GroupNorm; time-embed MLPs have no norm, so norm-quant kernels are skipped there --
only the GEMM+dequant+bias is timed, mirroring how they actually run in the model).
Emits split_stage_full.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
sys.path.insert(0, "/workspace/llm-awq/awq/kernels")
import torch, torch.nn as nn, torch.nn.functional as F, modiff_cutlass as mc
try: import awq_inference_engine as _awq
except Exception as _e: _awq = None; print("AWQ import failed:", repr(_e)[:200])
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"
DEV = "cuda"
N_BATCH = 32   # matches the rest of this report's benchmarks

def bench(fn, it=200, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

def rel_err(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)

def awq_pad_weight(wi8, ws, Nout, M, s_a):
    """One-time setup (matches what wxax_linear.py's __init__ does for weight padding, PLUS an
    optimization wxax_linear.py itself does NOT do: caching the per-token ascale + output buffers.
    wxax_linear.py rebuilds `asc = torch.full((M,), a_scale, ...)` and `out = torch.empty(...)` on
    EVERY forward call, assuming M and a_scale might change call-to-call (true for dynamic-shape
    LLM decoding, AWQ's original target). In this UNet, both are constant across calls for a given
    layer (a_scale is static-calibrated, M is fixed by the resolution level / batch size) -- so both
    can be cached once, eliminating a `torch.full` kernel launch (~3.5us CPU dispatch + ~1.2us GPU)
    every call. This matters a lot at tiny M (time-embedding MLPs): see NEXT_STEPS.md."""
    Npad = ((Nout + 127) // 128) * 128
    wi8_pad = F.pad(wi8, (0, 0, 0, Npad - Nout)).contiguous() if Npad != Nout else wi8.contiguous()
    ws_half_pad = F.pad(ws.half(), (0, Npad - Nout), value=1.0).contiguous() if Npad != Nout else ws.half().contiguous()
    asc_cached = torch.full((M,), 1.0 / s_a, device=DEV, dtype=torch.float16)
    out_cached = torch.empty(M, Npad, device=DEV, dtype=torch.float16)
    return wi8_pad, ws_half_pad, Npad, asc_cached, out_cached

def awq_gemm(xq, wi8_pad, ws_half_pad, Nout, Npad, asc_cached, out_cached, bias):
    """AWQ split GEMM+dequant+bias, with ascale/out reused from the one-time cache above."""
    _awq.w8a8_gemm_forward_cuda(xq, wi8_pad, ws_half_pad, asc_cached, out_cached)
    return (out_cached[:, :Nout] if Npad != Nout else out_cached) + bias

# ---- Attention qkv/proj shapes: (label, Cin, Cout, T) at the 5 real (C,T) combos ----
ATTN_CFGS = [
    ("qkv  C192 T1024 (level0)",  192, 576, 1024),
    ("proj C192 T1024 (level0)",  192, 192, 1024),
    ("qkv  C384 T256  (level1)",  384, 1152, 256),
    ("proj C384 T256  (level1)",  384, 384, 256),
    ("qkv  C384 T64   (level2)",  384, 1152, 64),
    ("proj C384 T64   (level2)",  384, 384, 64),
    ("qkv  C768 T16   (level3)",  768, 2304, 16),
    ("proj C768 T16   (level3)",  768, 768, 16),
    ("qkv  C768 T4    (middle)",  768, 2304, 4),
    ("proj C768 T4    (middle)",  768, 768, 4),
]
# ---- time-embedding MLP shapes: (label, Cin, Cout), M = N_BATCH only (no token multiplication) ----
TEMB_CFGS = [
    ("time_embed[0]",       192, 768),
    ("time_embed[2]",       768, 768),
    ("emb_layers Cch192",   768, 384),
    ("emb_layers Cch384",   768, 768),
    ("emb_layers Cch768",   768, 1536),
]

G, eps = 32, 1e-6
EMPTY = torch.empty(0, device=DEV, dtype=torch.float16)
rows = []


def run_attn(label, C, Nout, T):
    N, M = N_BATCH, N_BATCH * T
    torch.manual_seed(0)
    x = torch.randn(N, C, T, 1, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    # H,W factored as (T,1) is fine for GroupNorm (only groups/channels matter, spatial shape doesn't)
    lin = nn.Linear(C, Nout).to(DEV).half(); gn = nn.GroupNorm(G, C, eps=eps).to(DEV).half()
    Wt, bt = lin.weight.detach().float(), lin.bias.detach().float()
    gw, gb = gn.weight.detach().float(), gn.bias.detach().float()
    gwh, gbh = gw.half().contiguous(), gb.half().contiguous()

    Wf16 = Wt.t().half().contiguous(); bt16 = bt.half().contiguous()
    def s_fp16():
        h = mc.group_norm_silu_nhwc(x, gwh, gbh, G, eps, False, EMPTY, EMPTY)
        return torch.matmul(h.permute(0, 2, 3, 1).reshape(M, C), Wf16) + bt16

    Wf = Wt * gw[None, :]
    xn_pure = F.group_norm(x.float(), G, None, None, eps)
    s_a = float(127.0 / xn_pure.abs().max().clamp_min(1e-8))
    ws = (Wf.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    wi8 = torch.clamp(torch.round(Wf / ws[:, None]), -127, 127).to(torch.int8).contiguous()
    epi_bias = (bt + Wt @ gb).half().contiguous()
    ones_g = torch.ones(C, device=DEV, dtype=torch.float16); zeros_g = torch.zeros(C, device=DEV, dtype=torch.float16)
    scale_t = torch.tensor([s_a], device=DEV, dtype=torch.float32); smooth = torch.empty(0, device=DEV, dtype=torch.float32)
    def gnq8():
        xi = mc.group_norm_silu_quantize_nhwc(x, ones_g, zeros_g, G, eps, False, scale_t, smooth, EMPTY, EMPTY)
        return xi.permute(0, 2, 3, 1).reshape(M, C)
    def s_int8(): return mc.gemm_w8a8(gnq8(), wi8, ws.contiguous(), 1.0 / s_a) + epi_bias
    wi8_pad, ws_half_pad, Npad, asc_c, out_c = awq_pad_weight(wi8, ws, Nout, M, s_a) if _awq is not None else (None,) * 5  # ONE-TIME, not timed
    def s_awq(): return awq_gemm(gnq8(), wi8_pad, ws_half_pad, Nout, Npad, asc_c, out_c, epi_bias) if _awq is not None else None

    xn = F.group_norm(x.float(), G, gw, gb, eps)
    ref = F.linear(xn.permute(0, 2, 3, 1).reshape(M, C), Wt, bt)
    re8 = rel_err(s_int8(), ref); rea = rel_err(s_awq(), ref) if _awq is not None else float("nan")
    t_fp, t8, ta = bench(s_fp16), bench(s_int8), (bench(s_awq) if _awq is not None else float("nan"))
    return dict(shape=label, kind="attn", M=M, fp16_split_us=round(t_fp, 1), int8_split_us=round(t8, 1),
                awq_split_us=round(ta, 1) if ta == ta else "", int8_vs_fp16=round(t_fp / t8, 3),
                awq_vs_fp16=round(t_fp / ta, 3) if ta == ta else "", int8_rel_err=round(re8, 4),
                awq_rel_err=round(rea, 4) if rea == rea else "")


def run_temb(label, Cin, Cout):
    M = N_BATCH
    torch.manual_seed(0)
    x = torch.randn(M, Cin, device=DEV, dtype=torch.float16)
    lin = nn.Linear(Cin, Cout).to(DEV).half()
    Wt, bt = lin.weight.detach().float(), lin.bias.detach().float()
    Wf16 = Wt.t().half().contiguous(); bt16 = bt.half().contiguous()
    def s_fp16(): return torch.matmul(x, Wf16) + bt16

    s_a = float(127.0 / x.float().abs().max().clamp_min(1e-8))
    ws = (Wt.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    wi8 = torch.clamp(torch.round(Wt / ws[:, None]), -127, 127).to(torch.int8).contiguous()
    bt16b = bt.half().contiguous()
    def q(): return mc.quantize_act_int8(x, 1.0 / s_a)   # quantize_act_int8 takes the DEQUANT scale (absmax/127)
    def s_int8(): return mc.gemm_w8a8(q(), wi8, ws.contiguous(), 1.0 / s_a) + bt16b
    wi8_pad, ws_half_pad, Npad, asc_c, out_c = awq_pad_weight(wi8, ws, Cout, M, s_a) if _awq is not None else (None,) * 5  # ONE-TIME, not timed
    def s_awq(): return awq_gemm(q(), wi8_pad, ws_half_pad, Cout, Npad, asc_c, out_c, bt16b) if _awq is not None else None

    ref = F.linear(x.float(), Wt, bt)
    re8 = rel_err(s_int8(), ref); rea = rel_err(s_awq(), ref) if _awq is not None else float("nan")
    t_fp, t8, ta = bench(s_fp16), bench(s_int8), (bench(s_awq) if _awq is not None else float("nan"))
    return dict(shape=label, kind="temb", M=M, fp16_split_us=round(t_fp, 1), int8_split_us=round(t8, 1),
                awq_split_us=round(ta, 1) if ta == ta else "", int8_vs_fp16=round(t_fp / t8, 3),
                awq_vs_fp16=round(t_fp / ta, 3) if ta == ta else "", int8_rel_err=round(re8, 4),
                awq_rel_err=round(rea, 4) if rea == rea else "")


print(f"{'shape':<28} | {'fp16':>8} {'int8':>8} {'awq':>8} | {'i8x':>6} {'awqx':>6} | {'i8err':>7} {'awqerr':>7}")
for label, C, Nout, T in ATTN_CFGS:
    r = run_attn(label, C, Nout, T); rows.append(r)
    print(f"{label:<28} | {r['fp16_split_us']:8.1f} {r['int8_split_us']:8.1f} {r['awq_split_us'] if r['awq_split_us']!='' else float('nan'):8.1f} | "
          f"{r['int8_vs_fp16']:6.2f} {r['awq_vs_fp16'] if r['awq_vs_fp16']!='' else float('nan'):6.2f} | {r['int8_rel_err']:7.4f} {r['awq_rel_err'] if r['awq_rel_err']!='' else float('nan'):7.4f}")
for label, Cin, Cout in TEMB_CFGS:
    r = run_temb(label, Cin, Cout); rows.append(r)
    print(f"{label:<28} | {r['fp16_split_us']:8.1f} {r['int8_split_us']:8.1f} {r['awq_split_us'] if r['awq_split_us']!='' else float('nan'):8.1f} | "
          f"{r['int8_vs_fp16']:6.2f} {r['awq_vs_fp16'] if r['awq_vs_fp16']!='' else float('nan'):6.2f} | {r['int8_rel_err']:7.4f} {r['awq_rel_err'] if r['awq_rel_err']!='' else float('nan'):7.4f}")

with open(f"{OUT}/split_stage_full.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE split_stage_full.csv")
