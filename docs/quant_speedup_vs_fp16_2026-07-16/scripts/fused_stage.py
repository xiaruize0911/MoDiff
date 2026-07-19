"""FUSED projection stage (the realistic block path): GN is FUSED into the projection, not run as a
separate kernel. fp16 = fused_gn_qkv (GN+qkv in one kernel); quantized = group_norm_silu_quantize
(GN+quantize fused -> int8/int4 activation, no separate GN write) -> GEMM. Compares fp16 / ours-w8a8 /
AWQ-w8a8 / ours-w4a4 with GN fused in every mode. Emits fused_stage.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch, torch.nn.functional as F, modiff_cutlass as mc
try: import awq_inference_engine as _awq
except Exception: _awq = None
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=80, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

CFGS = [("C192 qkv", 32, 192, 32, 32), ("C384 qkv", 32, 384, 16, 16), ("C768 qkv", 32, 768, 8, 8)]
EMPTY = torch.empty(0, device="cuda", dtype=torch.float16)
rows = []
print(f"{'shape':>10} | {'fp16(fused GN+qkv)':>18} {'int8':>7} {'awq':>7} {'int4':>7} | {'i8×':>4} {'awq×':>4} {'i4×':>4}")
for name, N, C, H, W in CFGS:
    Ncol = 3 * C; M = N * H * W; eps = 1e-6
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    gw = torch.ones(C, device="cuda", dtype=torch.float16); gb = torch.zeros(C, device="cuda", dtype=torch.float16)
    scale_t = torch.tensor([50.0], device="cuda", dtype=torch.float32)      # scalar quant multiplier (127/absmax)
    smooth = torch.ones(C, device="cuda", dtype=torch.float32)
    ws = torch.rand(Ncol, device="cuda").float() / 127
    Wq = torch.randint(-127, 127, (Ncol, C), device="cuda", dtype=torch.int8)
    Wq4 = torch.randint(-7, 7, (Ncol, C // 2), device="cuda", dtype=torch.int8)
    Np = ((Ncol + 127) // 128) * 128; Wqp = F.pad(Wq, (0, 0, 0, Np - Ncol)); wsh = F.pad(ws, (0, Np - Ncol), value=1).half()
    ascv = torch.full((M,), 0.02, device="cuda", dtype=torch.float16); outp = torch.empty(M, Np, device="cuda", dtype=torch.float16)
    # fp16: GN+qkv fused
    conv_w = (torch.randn(Ncol, 1, 1, C, device="cuda", dtype=torch.float16) * 0.05).contiguous()
    epi = (torch.randn(Ncol, device="cuda", dtype=torch.float16) * 0.1)
    def s_fp16(): return mc.fused_gn_qkv(x, conv_w, epi, 32, eps, 16.0)
    def gnq8():   # GN+quantize fused -> int8 [M,C]
        xi = mc.group_norm_silu_quantize_nhwc(x, gw, gb, 32, eps, False, scale_t, smooth, EMPTY, EMPTY)
        return xi.permute(0, 2, 3, 1).reshape(M, C)
    def s_int8(): return mc.gemm_w8a8(gnq8(), Wq, ws, 0.02)
    def s_awq():  xq = gnq8(); _awq.w8a8_gemm_forward_cuda(xq, Wqp, wsh, ascv, outp); return outp
    def gnq4():
        xi = mc.group_norm_silu_quantize_pack_nhwc(x, gw, gb, 32, eps, False, scale_t, smooth, EMPTY, EMPTY)
        return xi.permute(0, 2, 3, 1).reshape(M, C // 2)
    def s_int4(): return mc.gemm_w4a4(gnq4(), Wq4, ws, 0.02, C)
    tf = bench(s_fp16); t8 = bench(s_int8); ta = bench(s_awq) if _awq else float("nan")
    try: t4 = bench(s_int4)
    except Exception as ex: t4 = float("nan"); print("  int4 skip:", repr(ex)[:80])
    sp = lambda t: tf / t if t == t else float("nan")
    print(f"{name:>10} | {tf:18.1f} {t8:7.1f} {ta:7.1f} {t4:7.1f} | {sp(t8):4.2f} {sp(ta):4.2f} {sp(t4):4.2f}")
    rows.append(dict(shape=name, M=M, fp16_fused_us=round(tf, 1), int8_us=round(t8, 1),
                     awq_us=round(ta, 1) if ta == ta else "", int4_us=round(t4, 1) if t4 == t4 else "",
                     int8_vs_fp16=round(sp(t8), 3), awq_vs_fp16=round(sp(ta), 3) if ta == ta else "",
                     int4_vs_fp16=round(sp(t4), 3) if t4 == t4 else ""))
with open(f"{OUT}/fused_stage.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE fused_stage.csv")
