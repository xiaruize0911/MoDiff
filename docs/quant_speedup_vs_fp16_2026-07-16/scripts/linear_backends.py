"""Linear GEMM backends on the real qkv/proj shapes: fp16 cuBLAS vs our gemm_w8a8 vs AWQ w8a8 vs
our gemm_w4a4. Answers 'are the int (incl AWQ) linear kernels actually faster than fp16 here?'.
Static a_scale (no dynamic sync). Emits linear_backends.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn as nn, torch.nn.functional as F
import integration.kernels.wxax_linear as wl
from integration.kernels.wxax_linear import QuantLinearWxAx, _eligible
_awq = wl._awq
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=50, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3  # us

# (name, M=batch*T, K=in, N=out). batch 32. qkv: C->3C, proj: C->C.
SHAPES = [("C192 qkv", 32768, 192, 576), ("C192 proj", 32768, 192, 192),
          ("C384 qkv", 8192, 384, 1152), ("C384 proj", 8192, 384, 384),
          ("C768 qkv", 2048, 768, 2304), ("C768 proj", 2048, 768, 768)]
print(f"awq available: {_awq is not None}")
rows = []
print(f"{'shape':>10} {'M,K,N':>16} | {'fp16':>7} {'w8a8':>7} {'awq':>7} {'w4a4':>7} | {'w8a8':>5} {'awq':>5} {'w4a4':>5}")
for name, M, K, N in SHAPES:
    lin = nn.Linear(K, N).cuda().half()
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    a_scale = x.abs().max().item() / 127.0
    t_fp16 = bench(lambda: F.linear(x, lin.weight, lin.bias))

    def mk(bits, no_awq):
        wl._awq = None if no_awq else _awq          # control AWQ N-pad at construction
        m = QuantLinearWxAx(lin, bits); wl._awq = _awq
        m.set_a_scale(x.abs().max().item() / m.Q)
        return m
    # our gemm_w8a8 (AWQ off, no pad)
    m8 = mk(8, True); t_w8a8 = bench(lambda: m8(x))
    # AWQ w8a8 (N-padded)
    m8a = mk(8, False)
    t_awq = bench(lambda: m8a(x)) if m8a._use_awq else float("nan")
    # our gemm_w4a4
    if _eligible(K, N, 4):
        m4 = mk(4, True); t_w4a4 = bench(lambda: m4(x))
    else:
        t_w4a4 = float("nan")
    sp = lambda t: (t_fp16 / t) if t == t else float("nan")
    print(f"{name:>10} {f'{M},{K},{N}':>16} | {t_fp16:7.1f} {t_w8a8:7.1f} {t_awq:7.1f} {t_w4a4:7.1f} | "
          f"{sp(t_w8a8):5.2f} {sp(t_awq):5.2f} {sp(t_w4a4):5.2f}")
    rows.append(dict(shape=name, M=M, K=K, N=N, fp16_us=round(t_fp16, 1), w8a8_us=round(t_w8a8, 1),
                     awq_us=round(t_awq, 1) if t_awq == t_awq else "", w4a4_us=round(t_w4a4, 1) if t_w4a4 == t_w4a4 else "",
                     w8a8_vs_fp16=round(sp(t_w8a8), 3), awq_vs_fp16=round(sp(t_awq), 3) if t_awq == t_awq else "",
                     w4a4_vs_fp16=round(sp(t_w4a4), 3) if t_w4a4 == t_w4a4 else ""))
with open(f"{OUT}/linear_backends.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE linear_backends.csv")
