"""Headline fairness micro-benchmark: SOFTMAX kernel, dynamic (2-pass, per-row max) vs static
(1-pass, calibrated c), for fp16 / int8 / int4, per real churches attention shape.

This isolates the static-vs-dynamic *algorithm* delta from precision: if the static 1-pass speedup
is comparable across fp16/int8/int4, static is a precision-independent optimization (fair to fp16),
not a quantization effect. Softmax is HBM-bandwidth bound on the T*T score matrix; static removes the
max pass -> 1 read instead of 2. Emits data/softmax_kernel.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, modiff_cutlass as mc
OUT = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"

def bench(fn, it=50, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3  # us

SHAPES = [(256, 1024), (256, 256), (256, 64)]      # (BH, T) real churches score matrices
rows = []
print(f"{'BH,T':>10} {'prec':>5} | {'dyn us':>8} {'static us':>9} | speedup")
for (BH, T) in SHAPES:
    S = torch.randn(BH, T, T, device="cuda", dtype=torch.float16)
    c = S.float().amax(-1).mean().item()
    variants = [
        ("fp16", lambda: mc.attn_softmax_fp16(S, False, 0.0), lambda: mc.attn_softmax_fp16(S, True, c)),
        ("int8", lambda: mc.attn_softmax_requant(S),          lambda: mc.attn_softmax_requant_static(S, c)),
        ("int4", lambda: mc.attn_softmax_requant4(S),         lambda: mc.attn_softmax_requant4_static(S, c)),
    ]
    for prec, dyn, sta in variants:
        td = bench(dyn); ts = bench(sta)
        print(f"{BH},{T:>4} {prec:>5} | {td:8.1f} {ts:9.1f} | {td/ts:5.2f}x")
        rows.append(dict(BH=BH, T=T, precision=prec, dyn_us=round(td, 1), static_us=round(ts, 1),
                         static_speedup=round(td / ts, 3)))
with open(f"{OUT}/softmax_kernel.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE softmax_kernel.csv")
