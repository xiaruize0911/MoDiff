"""Fusion CEILING: how fast could a *perfectly* fused GN-prologue int8/int4 GEMM be?
A fused kernel (GN+quant folded into the GEMM prologue, no int8 intermediate, one launch) converges
toward the BARE GEMM time (activation already int8/int4 in regs, dequant in epilogue). So bare-GEMM-only
is the lower bound on the fused stage. If bare-GEMM-only is still >= fp16 fused stage, NO fuse strategy
can win -> keep fp16. Also reports the output-write floor (fp16 [M,3C], shared by every mode). fuse_ceiling.csv"""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
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
rows = []
print(f"{'shape':>10} | {'M':>6} {'K':>4} {'N':>5} | {'outWr':>6} {'fp16stg':>7} | {'i8gemm':>6} {'awq':>6} {'i4gemm':>6} | ceiling vs fp16 stage")
for name, N, C, H, W in CFGS:
    Ncol = 3 * C; M = N * H * W
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    conv_w = (torch.randn(Ncol, 1, 1, C, device="cuda", dtype=torch.float16) * 0.05).contiguous()
    epi = (torch.randn(Ncol, device="cuda", dtype=torch.float16) * 0.1)
    def s_fp16(): return mc.fused_gn_qkv(x, conv_w, epi, 32, 1e-6, 16.0)     # fp16 fused stage (GN+qkv)
    # bare GEMMs (activation ALREADY quantized in HBM) = the fuse ceiling
    xq8 = torch.randint(-127, 127, (M, C), device="cuda", dtype=torch.int8)
    xq4 = torch.randint(-7, 7, (M, C // 2), device="cuda", dtype=torch.int8)
    ws = torch.rand(Ncol, device="cuda").float() / 127
    Wq = torch.randint(-127, 127, (Ncol, C), device="cuda", dtype=torch.int8)
    Wq4 = torch.randint(-7, 7, (Ncol, C // 2), device="cuda", dtype=torch.int8)
    Np = ((Ncol + 127) // 128) * 128; Wqp = F.pad(Wq, (0, 0, 0, Np - Ncol)); wsh = F.pad(ws, (0, Np - Ncol), value=1).half()
    ascv = torch.full((M,), 0.02, device="cuda", dtype=torch.float16); outp = torch.empty(M, Np, device="cuda", dtype=torch.float16)
    def g8():  return mc.gemm_w8a8(xq8, Wq, ws, 0.02)
    def gaw(): _awq.w8a8_gemm_forward_cuda(xq8, Wqp, wsh, ascv, outp); return outp
    def g4():  return mc.gemm_w4a4(xq4, Wq4, ws, 0.02, C)
    # output-write floor: just writing the fp16 [M,3C] result (shared by every mode)
    out16 = torch.empty(M, Ncol, device="cuda", dtype=torch.float16); src = out16.clone()
    def owr(): out16.copy_(src); return out16
    tf = bench(s_fp16); t8 = bench(g8); ta = bench(gaw) if _awq else float("nan"); t4 = bench(g4); tow = bench(owr)
    best = min([t for t in (t8, ta, t4) if t == t])
    print(f"{name:>10} | {M:6d} {C:4d} {Ncol:5d} | {tow:6.1f} {tf:7.1f} | {t8:6.1f} {ta:6.1f} {t4:6.1f} | best gemm {best:.0f}µs = {tf/best:.2f}× fp16 stage")
    rows.append(dict(shape=name, M=M, K=C, N=Ncol, out_write_floor_us=round(tow, 1), fp16_fused_stage_us=round(tf, 1),
                     int8_gemm_only_us=round(t8, 1), awq_gemm_only_us=round(ta, 1) if ta == ta else "",
                     int4_gemm_only_us=round(t4, 1), best_gemm_ceiling_vs_fp16=round(tf / best, 3)))
with open(f"{OUT}/fuse_ceiling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE fuse_ceiling.csv")
