"""GEMM KERNEL-ONLY speed (inputs pre-quantized; no activation-quantize / fp16-dequant / slice
overhead) vs fp16 cuBLAS, on the qkv shapes. Compared with linear_backends.py (full deployed op),
this isolates 'is the kernel slow' from 'is the quantize/dequant plumbing slow'. Emits
linear_kernel_only.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
try: import awq_inference_engine as _awq
except Exception: _awq = None
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=60, warm=25):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

SHAPES = [("C192 qkv", 32768, 192, 576), ("C384 qkv", 8192, 384, 1152), ("C768 qkv", 2048, 768, 2304)]
rows = []
print(f"{'shape':>12} | {'fp16':>7} {'w8a8':>7} {'awq':>7} {'w4a4':>7} | {'w8a8':>5} {'awq':>5} {'w4a4':>5}")
for name, M, K, N in SHAPES:
    W = torch.randn(N, K, device="cuda", dtype=torch.float16); x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    asc = x.abs().max().item() / 127.0; ws = torch.randn(N, device="cuda").abs().float() / 127
    xq = mc.quantize_act_int8(x, asc); Wq = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
    tf = bench(lambda: F.linear(x, W))
    tw8 = bench(lambda: mc.gemm_w8a8(xq, Wq, ws, asc))
    if _awq is not None:
        Npad = ((N + 127) // 128) * 128; Wqp = F.pad(Wq, (0, 0, 0, Npad - N)); wsh = F.pad(ws, (0, Npad - N), value=1).half()
        ascv = torch.full((M,), asc, device="cuda", dtype=torch.float16); outp = torch.empty(M, Npad, device="cuda", dtype=torch.float16)
        tawq = bench(lambda: _awq.w8a8_gemm_forward_cuda(xq, Wqp, wsh, ascv, outp))
    else: tawq = float("nan")
    xq4 = mc.quantize_act_int4_pack(x, asc / 18); Wq4 = torch.randint(-7, 7, (N, K // 2), device="cuda", dtype=torch.int8)
    tw4 = bench(lambda: mc.gemm_w4a4(xq4, Wq4, ws, asc, K))
    sp = lambda t: tf / t if t == t else float("nan")
    print(f"{name:>12} | {tf:7.1f} {tw8:7.1f} {tawq:7.1f} {tw4:7.1f} | {sp(tw8):5.2f} {sp(tawq):5.2f} {sp(tw4):5.2f}")
    rows.append(dict(shape=name, M=M, K=K, N=N, fp16_us=round(tf, 1), w8a8_us=round(tw8, 1),
                     awq_us=round(tawq, 1) if tawq == tawq else "", w4a4_us=round(tw4, 1),
                     w8a8_vs_fp16=round(sp(tw8), 3), awq_vs_fp16=round(sp(tawq), 3) if tawq == tawq else "",
                     w4a4_vs_fp16=round(sp(tw4), 3)))
with open(f"{OUT}/linear_kernel_only.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE linear_kernel_only.csv")
