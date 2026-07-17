"""FAIR projection-stage benchmark INCLUDING the GroupNorm layer (which the bare-GEMM benchmark omits,
though it costs IO in every mode). Stage = GroupNorm -> [quantize] -> qkv GEMM, for fp16 / ours-w8a8 /
AWQ-w8a8 / ours-w4a4, on the real churches qkv shapes. Also reports GN-alone and the fused GN->qkv path
(fused_gn_qkv / fused_gn_qkv_int8) which folds GN into the GEMM. Emits stage_with_norm.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from integration.fused_ops.fused_resblock import _group_norm_silu
try: import awq_inference_engine as _awq
except Exception: _awq = None
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=80, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

# (name, N(batch), C, H, W) ; qkv: C->3C ; M=N*H*W
CFGS = [("C192 qkv", 32, 192, 32, 32), ("C384 qkv", 32, 384, 16, 16), ("C768 qkv", 32, 768, 8, 8)]
rows = []
print(f"{'shape':>10} | {'GN':>6} {'fp16':>7} {'int8':>7} {'awq':>7} {'int4':>7} | {'i8×':>4} {'awq×':>4} {'i4×':>4}  (stage = GN+quant+GEMM)")
for name, N, C, H, W in CFGS:
    Ncol = 3 * C; M = N * H * W
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    gw = torch.ones(C, device="cuda", dtype=torch.float16); gb = torch.zeros(C, device="cuda", dtype=torch.float16)
    W16 = torch.randn(Ncol, C, device="cuda", dtype=torch.float16)
    ws = torch.rand(Ncol, device="cuda").float() / 127; Wq = torch.randint(-127, 127, (Ncol, C), device="cuda", dtype=torch.int8)
    Wq4 = torch.randint(-7, 7, (Ncol, C // 2), device="cuda", dtype=torch.int8)
    Np = ((Ncol + 127) // 128) * 128; Wqp = F.pad(Wq, (0, 0, 0, Np - Ncol)); wsh = F.pad(ws, (0, Np - Ncol), value=1).half()
    ascv = torch.full((M,), 0.02, device="cuda", dtype=torch.float16); outp = torch.empty(M, Np, device="cuda", dtype=torch.float16)

    def gn():  # GroupNorm(+reshape to [M,C]) -- the norm layer, common to all modes
        xn = _group_norm_silu(x, 32, gw, gb, 1e-6, apply_silu=False)
        return xn.permute(0, 2, 3, 1).reshape(M, C)
    def s_fp16():
        return F.linear(gn(), W16)
    def s_int8():
        xf = gn(); xq = mc.quantize_act_int8(xf, 0.02); return mc.gemm_w8a8(xq, Wq, ws, 0.02)
    def s_awq():
        xf = gn(); xq = mc.quantize_act_int8(xf, 0.02); _awq.w8a8_gemm_forward_cuda(xq, Wqp, wsh, ascv, outp); return outp
    def s_int4():
        xf = gn(); xq = mc.quantize_act_int4_pack(xf, 0.02 / 18); return mc.gemm_w4a4(xq, Wq4, ws, 0.02, C)
    tgn = bench(gn); tf = bench(s_fp16); t8 = bench(s_int8); ta = bench(s_awq) if _awq else float("nan"); t4 = bench(s_int4)
    sp = lambda t: tf / t if t == t else float("nan")
    print(f"{name:>10} | {tgn:6.1f} {tf:7.1f} {t8:7.1f} {ta:7.1f} {t4:7.1f} | {sp(t8):4.2f} {sp(ta):4.2f} {sp(t4):4.2f}")
    rows.append(dict(shape=name, M=M, GN_us=round(tgn, 1), stage_fp16_us=round(tf, 1),
                     stage_int8_us=round(t8, 1), stage_awq_us=round(ta, 1) if ta == ta else "", stage_int4_us=round(t4, 1),
                     int8_vs_fp16=round(sp(t8), 3), awq_vs_fp16=round(sp(ta), 3) if ta == ta else "", int4_vs_fp16=round(sp(t4), 3),
                     GN_frac_fp16=round(tgn / tf, 3)))
with open(f"{OUT}/stage_with_norm.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE stage_with_norm.csv")
