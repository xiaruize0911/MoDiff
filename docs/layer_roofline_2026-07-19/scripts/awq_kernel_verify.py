"""Verify + benchmark the vendored AWQ w8a8 kernel (awq_w8a8_gemm) vs our gemm_w8a8_awq and fp16,
on the real churches qkv/proj shapes. Correctness: AWQ output vs int-matmul reference (rel-L2).
Writes data/awq_kernel_verify.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc


def bench(fn, it=200, warm=50, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]


def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


SH = [("32²C192 qkv", 65536, 192, 576), ("32²C192 proj", 65536, 192, 192),
      ("16²C384 qkv", 16384, 384, 1152), ("16²C384 proj", 16384, 384, 384),
      ("8²C384 qkv", 4096, 384, 1152), ("4²C768 qkv", 1024, 768, 2304), ("4²C768 proj", 1024, 768, 768)]
rows = []
torch.manual_seed(0)
print(f"{'shape':14s} {'M,K,N':>16} | {'fp16':>7} {'ours':>7} {'AWQ':>7} | {'AWQ/ours':>8} {'AWQ/fp16':>8} | {'rel':>7}")
for (nm, M, K, N) in SH:
    Np = ((N + 127) // 128) * 128
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    a_scale = x.abs().max().item() / 127.0
    xq = mc.quantize_act_int8(x, a_scale)                                 # [M,K] int8
    Wq = torch.randint(-127, 127, (Np, K), device="cuda", dtype=torch.int8)
    ws = (torch.randn(Np, device="cuda").abs().float() / 127 + 1e-3)      # [Np] f32 (ours)
    W = torch.randn(N, K, device="cuda", dtype=torch.float16)
    # reference dequant matmul (real N cols)
    ref = (xq.float() @ Wq[:N].float().T) * a_scale * ws[:N].unsqueeze(0)
    # ours
    t_ours = bench(lambda: mc.gemm_w8a8_awq(xq, Wq, ws, a_scale))
    ours = mc.gemm_w8a8_awq(xq, Wq, ws, a_scale)[:, :N]
    # AWQ (preallocated out, per-token ascale, half wscale)
    wsh = ws.half().contiguous(); ascv = torch.full((M,), a_scale, device="cuda", dtype=torch.float16)
    outp = torch.empty(M, Np, device="cuda", dtype=torch.float16)
    mc.awq_w8a8_gemm(xq, Wq, wsh, ascv, outp)
    awq = outp[:, :N]
    t_awq = bench(lambda: mc.awq_w8a8_gemm(xq, Wq, wsh, ascv, outp))
    t_fp16 = bench(lambda: F.linear(x, W))
    rel = relL2(awq, ref)
    rows.append(dict(shape=nm, M=M, K=K, N=N, fp16_us=round(t_fp16, 1), ours_us=round(t_ours, 1),
                     awq_us=round(t_awq, 1), awq_vs_ours=round(t_ours / t_awq, 3),
                     awq_vs_fp16=round(t_fp16 / t_awq, 3), rel_vs_ref=round(rel, 4)))
    r = rows[-1]
    print(f"{nm:14s} {f'{M},{K},{N}':>16} | {r['fp16_us']:7.1f} {r['ours_us']:7.1f} {r['awq_us']:7.1f} | "
          f"{r['awq_vs_ours']:7.2f}x {r['awq_vs_fp16']:7.2f}x | {r['rel_vs_ref']:7.4f}")
with open("docs/layer_roofline_2026-07-19/data/awq_kernel_verify.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
tot_o = sum(r["ours_us"] for r in rows); tot_a = sum(r["awq_us"] for r in rows)
print(f"\nsum ours {tot_o:.0f}us -> AWQ {tot_a:.0f}us = {tot_o/tot_a:.2f}x  | max rel {max(r['rel_vs_ref'] for r in rows):.4f}")
print("WROTE awq_kernel_verify.csv")
