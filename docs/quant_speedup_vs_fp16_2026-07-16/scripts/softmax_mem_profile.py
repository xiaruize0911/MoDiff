"""Memory-traffic roofline of the softmax / score-matrix / elementwise kernels (the pipeline's
memory-bound tail). ncu HW counters are blocked here, so we compute ANALYTICAL DRAM bytes from the
known [BH,T,T] shapes and divide by measured kernel time to get achieved GB/s vs A40 peak (696 GB/s).
Shows the T*T score matrix dominates and how close each kernel is to the bandwidth roofline.
Emits softmax_mem.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"
PEAK = 696.0  # A40 GB/s

def bench(fn, it=50, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3  # us

rows = []
print(f"{'shape':>10} {'kernel':>26} | {'us':>7} {'MiB':>7} {'GB/s':>6} {'%peak':>5}")
for (BH, T, hd) in [(256, 1024, 24), (256, 256, 48), (256, 64, 96)]:
    scale = 1.0 / math.sqrt(hd); hp = 32
    S = torch.randn(BH, T, T, device="cuda", dtype=torch.float16)
    Q = torch.randint(-127, 127, (BH, T, hp), device="cuda", dtype=torch.int8)
    K = torch.randint(-127, 127, (BH, T, hp), device="cuda", dtype=torch.int8)
    Vt = torch.randint(-100, 100, (BH, 64, T), device="cuda", dtype=torch.int8)
    sq = torch.rand(BH, T, device="cuda"); sk = torch.rand(BH, T, device="cuda")
    c = S.float().amax(-1).mean().item()
    P8, sp8 = mc.attn_softmax_requant_static(S, c); sv = torch.rand(BH, 64, device="cuda")
    STT = BH * T * T                                    # score-matrix elements
    # (name, callable, analytical bytes)
    ks = [
        ("score *scale (fp16 EW)", lambda: (S * scale),                       STT * 2 + STT * 2),      # read+write fp16 S
        ("softmax int8 static",    lambda: mc.attn_softmax_requant_static(S, c), STT * 2 + STT * 1),   # read S fp16 + write P int8
        ("softmax int8 dynamic",   lambda: mc.attn_softmax_requant(S),        STT * 2 + STT * 1),      # (2-pass; 2nd S read L2)
        ("softmax fp16 dynamic",   lambda: mc.attn_softmax_fp16(S, False, 0.0), STT * 2 + STT * 2),    # read S + write P fp16
        ("QKᵀ int8 (write S)",     lambda: mc.attn_qk_int8(Q, K, sq, sk, scale), 2 * BH * T * hp + STT * 2),
        ("AV int8 (read P)",       lambda: mc.attn_av_int8(P8, Vt, sp8, sv),   STT * 1 + BH * 64 * T + BH * T * 64 * 2),
    ]
    for name, fn, byts in ks:
        t = bench(fn); gbs = byts / 1e9 / (t / 1e6); pct = 100 * gbs / PEAK
        print(f"{BH},{T:>4} {name:>26} | {t:7.1f} {byts/2**20:7.0f} {gbs:6.0f} {pct:5.0f}")
        rows.append(dict(BH=BH, T=T, kernel=name, us=round(t, 1), MiB=round(byts / 2**20, 1),
                         GBps=round(gbs, 0), pct_peak=round(pct, 0)))
with open(f"{OUT}/softmax_mem.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE softmax_mem.csv")
