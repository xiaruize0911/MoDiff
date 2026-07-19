"""Verify + benchmark the int8-OUTPUT GEMM fix (output-fusion) vs the fp16-output production kernel,
for int8 (gemm_w8a8_awq) and int4 (gemm_w4a4_awq) at the real qkv/proj shapes. Correctness: dequant
the int8 output and compare rel-L2 to the fp16 output (expect ~1/127 int8-rounding). Speed: measured
us + memory roofline (int8-out halves the M*N output write). Writes data/gemm_outi8_b64.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, modiff_cutlass as mc
BW = 696e9


def bench(fn, it=200, warm=50, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]


def pack4(q):
    q = q.to(torch.int8); lo = q[..., 0::2] & 0xF; hi = q[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8).contiguous()


def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


SH = [("32²C192 qkv", 65536, 192, 576, 5), ("32²C192 proj", 65536, 192, 192, 5),
      ("16²C384 qkv", 16384, 384, 1152, 5), ("16²C384 proj", 16384, 384, 384, 5),
      ("8²C384 qkv", 4096, 384, 1152, 5), ("4²C768 qkv", 1024, 768, 2304, 5), ("4²C768 proj", 1024, 768, 768, 5)]
rows = []
torch.manual_seed(0)
a_scale = 0.02
print(f"{'shape':14s}{'prec':5s}| {'fp16out':>8} {'i8out':>7} {'speedup':>7} | {'rf_fp16':>7} {'rf_i8':>6} {'eff_i8%':>7} | {'rel':>7}")
for (nm, M, K, N, cnt) in SH:
    for prec in ("int8", "int4"):
        if prec == "int8":
            Kp = ((K + 63) // 64) * 64; Np = ((N + 127) // 128) * 128
            A = torch.randint(-127, 127, (M, Kp), device="cuda", dtype=torch.int8)
            B = torch.randint(-127, 127, (Np, Kp), device="cuda", dtype=torch.int8)
            ws = (torch.randn(Np, device="cuda").abs().float() / 127 + 1e-3)
            fp16 = mc.gemm_w8a8_awq(A, B, ws, a_scale)                    # [M,Np] fp16 reference
            absmax = fp16.float().abs().amax(0).clamp_min(1e-6)          # per-column
            inv_out = (127.0 / absmax).contiguous()
            out_scale = (absmax / 127.0)
            i8out = mc.gemm_w8a8_awq_out_i8(A, B, ws, a_scale, inv_out)  # [M,Np] int8
            deq = i8out.float() * out_scale.unsqueeze(0)
            t16 = bench(lambda: mc.gemm_w8a8_awq(A, B, ws, a_scale))
            t8 = bench(lambda: mc.gemm_w8a8_awq_out_i8(A, B, ws, a_scale, inv_out))
            ein = (M * Kp + Kp * Np) * 1
        else:
            Kp = ((K + 127) // 128) * 128; Np = ((N + 127) // 128) * 128
            A = pack4(torch.randint(-7, 7, (M, Kp), device="cuda", dtype=torch.int8))
            B = pack4(torch.randint(-7, 7, (Np, Kp), device="cuda", dtype=torch.int8))
            ws = (torch.randn(Np, device="cuda").abs().float() / 7 + 1e-3)
            fp16 = mc.gemm_w4a4_awq(A, B, ws, a_scale, Kp)
            absmax = fp16.float().abs().amax(0).clamp_min(1e-6)
            inv_out = (127.0 / absmax).contiguous(); out_scale = (absmax / 127.0)
            i8out = mc.gemm_w4a4_awq_out_i8(A, B, ws, a_scale, Kp, inv_out)
            deq = i8out.float() * out_scale.unsqueeze(0)
            t16 = bench(lambda: mc.gemm_w4a4_awq(A, B, ws, a_scale, Kp))
            t8 = bench(lambda: mc.gemm_w4a4_awq_out_i8(A, B, ws, a_scale, Kp, inv_out))
            ein = (M * Kp + Kp * Np) * 0.5
        rel = relL2(deq, fp16)
        rf16 = (ein + M * Np * 2) / BW * 1e6
        rf8 = (ein + M * Np * 1) / BW * 1e6
        rows.append(dict(shape=nm, M=M, K=K, N=N, count=cnt, prec=prec, fp16out_us=round(t16, 2),
                         i8out_us=round(t8, 2), speedup=round(t16 / t8, 3), roofline_fp16_us=round(rf16, 2),
                         roofline_i8_us=round(rf8, 2), eff_i8_pct=round(rf8 / t8 * 100, 1), rel_dequant=round(rel, 4)))
        r = rows[-1]
        print(f"{nm:14s}{prec:5s}| {r['fp16out_us']:8.1f} {r['i8out_us']:7.1f} {r['speedup']:7.2f} | "
              f"{r['roofline_fp16_us']:7.1f} {r['roofline_i8_us']:6.1f} {r['eff_i8_pct']:6.1f}% | {r['rel_dequant']:7.4f}")

with open("docs/layer_roofline_2026-07-19/data/gemm_outi8_b64.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nWROTE gemm_outi8_b64.csv")
