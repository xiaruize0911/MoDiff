import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
"""Isolated int8/int4 flash-attention benchmark for the quantized-attention optimization work.

Measures our kernels against PyTorch fp16 SDPA (flash backend) on the shapes the model
actually uses, and reports utilization against each precision's A40 dense peak so the
"1.5x int8 / 2x int4 vs fp16" target is checkable against a hardware ceiling rather than
against a moving baseline.

Optimization rounds tracked in the table below:
  base : hd_pad a runtime arg -> Oreg spilled to local memory (cuobjdump STACK:128)
  r1   : templated on HD_PAD  -> STACK:0, exact smem
  r2   : + WARPS templated 4/8 (fewer CTAs per (n,h) -> fewer K/V re-reads)
  r3   : + Q A-fragment hoisted out of the key loop into registers
"""
import json, statistics, sys
import torch
import torch.nn.functional as F
import modiff_cutlass as mc

DEV = "cuda"
PEAK = {"fp16": 149.7, "int8": 299.3, "int4": 598.7}   # A40 dense TFLOPS/TOPS

# (T, hd) -> us, measured on this GPU at earlier rounds; used only to show the trend.
HIST = {
    (1024, 24): {"base": 5506.9, "r1": 3036.4, "r2": 2956.2},
    (256,  48): {"base": 1007.4, "r1":  384.0, "r2":  380.6},
    (64,   48): {"base":  142.7, "r1":   63.0, "r2":   62.0},
    (1024, 32): {"base": 6180.4, "r1": 3235.9, "r2": 3147.9},
    (1024, 64): {"base": 18995.3, "r1": 5130.0, "r2": 5068.5},
}
SHAPES = [(128, 8, 1024, 24), (128, 8, 256, 48), (128, 8, 64, 48),
          (128, 8, 1024, 32), (128, 8, 1024, 64)]


def bench(fn, it=25, reps=4):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / it * 1e3)      # us
    return statistics.median(out)


def make_int8(N, H, T, hd):
    hp = ((hd + 31) // 32) * 32
    return dict(
        q=torch.randint(-127, 127, (N, H, T, hp), device=DEV, dtype=torch.int8),
        k=torch.randint(-127, 127, (N, H, T, hp), device=DEV, dtype=torch.int8),
        vt=torch.randint(-127, 127, (N, H, hp, T), device=DEV, dtype=torch.int8).contiguous(),
        sq=torch.full((N, H, T), 0.01, device=DEV, dtype=torch.float32),
        sk=torch.full((N, H, T), 0.01, device=DEV, dtype=torch.float32),
        sv=torch.full((N, H, hd), 0.01, device=DEV, dtype=torch.float32),
        hp=hp,
    )


def make_int4(N, H, T, hd):
    hdp4 = 64                                  # int4 QK K-pad: kernel requires hdp4 % 64 == 0
    hdp_v = ((hd + 31) // 32) * 32
    return dict(
        q4=torch.randint(-127, 127, (N, H, T, hdp4 // 2), device=DEV, dtype=torch.int8),
        k4=torch.randint(-127, 127, (N, H, T, hdp4 // 2), device=DEV, dtype=torch.int8),
        vt=torch.randint(-127, 127, (N, H, hdp_v, T), device=DEV, dtype=torch.int8).contiguous(),
        sq=torch.full((N, H, T), 0.01, device=DEV, dtype=torch.float32),
        sk=torch.full((N, H, T), 0.01, device=DEV, dtype=torch.float32),
        sv=torch.full((N, H, hd), 0.01, device=DEV, dtype=torch.float32),
        hdp4=hdp4, hdp_v=hdp_v,
    )


def main():
    rows = []
    for N, H, T, hd in SHAPES:
        a = make_int8(N, H, T, hd)
        sc = 1.0 / (hd ** 0.5)
        u8 = bench(lambda: mc.flash_attn_int8_vt(a["q"], a["k"], a["vt"],
                                                 a["sq"], a["sk"], a["sv"], sc))
        b = make_int4(N, H, T, hd)
        try:
            u4 = bench(lambda: mc.flash_attn_int4_vt(b["q4"], b["k4"], b["vt"], b["sq"],
                                                     b["sk"], b["sv"], b["hdp4"], sc))
        except Exception as ex:                       # shape may be structurally ineligible
            u4, u4err = None, str(ex)[:80]
        q = torch.randn(N, H, T, hd, device=DEV, dtype=torch.float16)
        k = torch.randn(N, H, T, hd, device=DEV, dtype=torch.float16)
        v = torch.randn(N, H, T, hd, device=DEV, dtype=torch.float16)
        upt = bench(lambda: F.scaled_dot_product_attention(q, k, v, scale=sc))

        # "useful" flops use the real hd; the padded variant shows what the mma units actually issue
        f_real = 4 * N * H * T * T * hd / 1e9        # GFLOP
        f_pad8 = 4 * N * H * T * T * a["hp"] / 1e9
        h = HIST.get((T, hd), {})
        r = dict(N=N, H=H, T=T, hd=hd, hd_pad=a["hp"], hdp4=b["hdp4"], hdp_v=b["hdp_v"],
                 int8_us=u8, int4_us=u4, pt_fp16_us=upt,
                 gflop_real=f_real, gflop_pad_int8=f_pad8,
                 int8_vs_pt=upt / u8, int4_vs_pt=(upt / u4) if u4 else None,
                 int8_pct_peak=f_pad8 / (u8 * 1e-6) / 1e3 / PEAK["int8"] * 100,
                 pt_pct_peak=f_real / (upt * 1e-6) / 1e3 / PEAK["fp16"] * 100,
                 hist=h)
        rows.append(r)
        for t in (a, b):
            t.clear()
        del q, k, v
        torch.cuda.empty_cache()

    W = f"{'T':>5s} {'hd':>3s} | {'base':>8s} {'r1':>7s} {'r2':>7s} {'r3(now)':>8s} | {'总':>6s} | {'PT fp16':>8s} {'i8 vs':>6s} {'i4 vs':>6s} | {'i8峰值':>7s}"
    print(W)
    print("-" * len(W))
    for r in rows:
        h = r["hist"]
        print(f"{r['T']:5d} {r['hd']:3d} | {h.get('base',0):8.1f} {h.get('r1',0):7.1f} "
              f"{h.get('r2',0):7.1f} {r['int8_us']:8.1f} | "
              f"{(h.get('base',0)/r['int8_us'] if h else 0):5.2f}x | {r['pt_fp16_us']:8.1f} "
              f"{r['int8_vs_pt']:5.2f}x " + (f"{r['int4_vs_pt']:5.2f}x" if r['int4_vs_pt'] else "    --") +
              f" | {r['int8_pct_peak']:6.1f}%")

    out = "docs/final_report_2026-07-28/data/qattn_qhoist_bench.json"
    with open(out, "w") as f:
        json.dump({"peak": PEAK, "rows": rows}, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
