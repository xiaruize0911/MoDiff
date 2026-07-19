"""Kernel benchmark (post-consolidation): the production AWQ-tiling kernels
(gemm_w8a8_awq / gemm_w4a4_awq, ldmatrix+XOR-swizzle+128-wide-N) vs AWQ's reference w8a8, vs fp16
cuBLAS, on the real churches qkv/proj shapes. Kernel-only (inputs pre-quantized). Repeated trials
(the AWQ-tiling kernels need N%128; int4 needs K%128 too -- N/K padded to the next 128-multiple).
The prior hand-written gemm_w8a8/gemm_w4a4 were retired 2026-07-18 (see backup/); this bench no
longer includes them -- their numbers live in the earlier stage3 report. Emits data/stage3_kernel_bench.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/llm-awq/awq/kernels")
import torch, torch.nn.functional as F, modiff_cutlass as mc
try: import awq_inference_engine as _awq
except Exception: _awq = None
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=200, warm=60, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)  # us
    ts.sort(); return ts[len(ts)//2]  # median

def pack4(q):
    q = q.to(torch.int8); lo = q[..., 0::2] & 0xF; hi = q[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8).contiguous()

SHAPES = [("C192 qkv", 32768, 192, 576), ("C192 proj", 32768, 192, 192),
          ("C384 qkv", 8192, 384, 1152), ("C384 proj", 8192, 384, 384),
          ("C768 qkv", 2048, 768, 2304), ("C768 proj", 2048, 768, 768)]
print(f"awq available: {_awq is not None}")
hdr = ["shape","M","K","N","fp16_us","o8awq_us","awqref_us","o4awq_us",
       "o8awq_vs_awqref","o8awq_vs_fp16","o4awq_vs_fp16"]
rows = []
print(f"{'shape':>10} {'M,K,N':>16} | {'fp16':>7} {'o8awq':>7} {'awqref':>7} {'o4awq':>7} | "
      f"{'8awq/awq':>8} {'8awq/fp16':>9} {'4awq/fp16':>9}")
for name, M, K, N in SHAPES:
    torch.manual_seed(0)
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    asc = x.abs().max().item() / 127.0
    ws = torch.randn(N, device="cuda").abs().float() / 127
    xq = mc.quantize_act_int8(x, asc)
    Wq = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
    Np = ((N + 127) // 128) * 128
    Wqp = F.pad(Wq, (0, 0, 0, Np - N)); wsp = F.pad(ws, (0, Np - N), value=1.0)

    tf = bench(lambda: F.linear(x, W))
    tw8awq = bench(lambda: mc.gemm_w8a8_awq(xq, Wqp, wsp, asc))
    if _awq is not None:
        wsh = wsp.half(); ascv = torch.full((M,), asc, device="cuda", dtype=torch.float16)
        outp = torch.empty(M, Np, device="cuda", dtype=torch.float16)
        tawq = bench(lambda: _awq.w8a8_gemm_forward_cuda(xq, Wqp, wsh, ascv, outp))
    else: tawq = float("nan")

    # int4: needs K%128 and N%128. All real K here are %128 except K=192 (192%128=64) -> pad K to 256.
    Kp = ((K + 127) // 128) * 128
    x4 = torch.randint(-7, 7, (M, Kp), device="cuda", dtype=torch.int8)
    W4 = torch.randint(-7, 7, (Np, Kp), device="cuda", dtype=torch.int8)
    xq4 = pack4(x4); Wq4 = pack4(W4)
    asc4 = asc / 18
    ws4n = torch.randn(N, device="cuda").abs().float() / 7
    ws4p = F.pad(ws4n, (0, Np - N), value=1.0)
    tw4awq = bench(lambda: mc.gemm_w4a4_awq(xq4, Wq4, ws4p, asc4, Kp))

    sp = lambda a, b: (a / b) if (a == a and b == b and b > 0) else float("nan")
    print(f"{name:>10} {f'{M},{K},{N}':>16} | {tf:7.1f} {tw8awq:7.1f} {tawq:7.1f} {tw4awq:7.1f} | "
          f"{sp(tawq,tw8awq):8.2f} {sp(tf,tw8awq):9.2f} {sp(tf,tw4awq):9.2f}")
    rows.append([name, M, K, N, round(tf,1), round(tw8awq,1),
                 round(tawq,1) if tawq==tawq else "", round(tw4awq,1),
                 round(sp(tawq,tw8awq),3) if tawq==tawq else "",
                 round(sp(tf,tw8awq),3), round(sp(tf,tw4awq),3)])
with open(f"{OUT}/stage3_kernel_bench.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
print(f"WROTE {OUT}/stage3_kernel_bench.csv")
print("\nNOTE: int4 K=192 shapes benchmark K padded to 256 (int4 kernel needs K%128).")
