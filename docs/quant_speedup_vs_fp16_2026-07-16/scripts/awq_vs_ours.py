"""Kernel-level GEMM benchmark: fp16 cuBLAS vs OUR gemm_w8a8 vs AWQ w8a8 vs OUR gemm_w4a4, on the
real churches qkv/proj shapes. Kernel-only (inputs pre-quantized) so it isolates the GEMM from the
quantize/dequant plumbing. Reports time, speedup vs fp16, and effective TFLOPS (useful 2*M*K*N).
Emits awq_vs_ours.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/llm-awq/awq/kernels")
import torch, torch.nn.functional as F, modiff_cutlass as mc
try: import awq_inference_engine as _awq
except Exception: _awq = None
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=100, warm=40):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3  # us

# (name, M=batch*T, K=in, N=out)
SHAPES = [("C192 qkv", 32768, 192, 576), ("C192 proj", 32768, 192, 192),
          ("C384 qkv", 8192, 384, 1152), ("C384 proj", 8192, 384, 384),
          ("C768 qkv", 2048, 768, 2304), ("C768 proj", 2048, 768, 768)]
print(f"awq: {_awq is not None}")
rows = []
print(f"{'shape':>10} {'M,K,N':>16} | {'fp16':>7} {'ours8':>7} {'awq8':>7} {'ours4':>7} | "
      f"{'o8×':>4} {'awq×':>4} {'o4×':>4} | {'fp16 TF':>7} {'awq TF':>7}")
for name, M, K, N in SHAPES:
    flops = 2.0 * M * K * N
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    asc = x.abs().max().item() / 127.0; ws = torch.randn(N, device="cuda").abs().float() / 127
    xq = mc.quantize_act_int8(x, asc); Wq = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
    tf = bench(lambda: F.linear(x, W))
    tw8 = bench(lambda: mc.gemm_w8a8(xq, Wq, ws, asc))
    if _awq is not None:
        Np = ((N + 127) // 128) * 128; Wqp = F.pad(Wq, (0, 0, 0, Np - N)); wsh = F.pad(ws, (0, Np - N), value=1).half()
        ascv = torch.full((M,), asc, device="cuda", dtype=torch.float16); outp = torch.empty(M, Np, device="cuda", dtype=torch.float16)
        tawq = bench(lambda: _awq.w8a8_gemm_forward_cuda(xq, Wqp, wsh, ascv, outp))
    else: tawq = float("nan")
    xq4 = mc.quantize_act_int4_pack(x, asc / 18); Wq4 = torch.randint(-7, 7, (N, K // 2), device="cuda", dtype=torch.int8)
    tw4 = bench(lambda: mc.gemm_w4a4(xq4, Wq4, ws, asc, K))
    sp = lambda t: tf / t if t == t else float("nan")
    tflop = lambda t: flops / (t / 1e6) / 1e12 if t == t else float("nan")
    print(f"{name:>10} {f'{M},{K},{N}':>16} | {tf:7.1f} {tw8:7.1f} {tawq:7.1f} {tw4:7.1f} | "
          f"{sp(tw8):4.2f} {sp(tawq):4.2f} {sp(tw4):4.2f} | {tflop(tf):7.0f} {tflop(tawq):7.0f}")
    rows.append(dict(shape=name, M=M, K=K, N=N, fp16_us=round(tf, 1), ours_w8a8_us=round(tw8, 1),
                     awq_w8a8_us=round(tawq, 1) if tawq == tawq else "", ours_w4a4_us=round(tw4, 1),
                     ours8_vs_fp16=round(sp(tw8), 3), awq_vs_fp16=round(sp(tawq), 3) if tawq == tawq else "",
                     ours4_vs_fp16=round(sp(tw4), 3),
                     fp16_TFLOPS=round(tflop(tf), 1), awq_TFLOPS=round(tflop(tawq), 1) if tawq == tawq else "",
                     ours8_TFLOPS=round(tflop(tw8), 1), ours4_TFLOPS=round(tflop(tw4), 1)))
with open(f"{OUT}/awq_vs_ours.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE awq_vs_ours.csv")
