"""Stage B: is the QKV shape actually amenable to speedup?

The advisor's point, and he is right about the premise: `openaimodel.py:337,345` build the attention
block's qkv and proj_out as **nn.Conv1d with kernel_size=1**, not Linear. `token_major_attention.py
:147-155` reshapes them to nn.Linear ([3C,C,1] -> [3C,C], a pure axis drop, bit-identical), and
`convert_linears_to_wxax` then makes them QuantLinearWxAx. His question is whether the resulting
shape can be sped up at all.

WHAT IS TIMED, per shape, all fp16 in / fp16 out so the comparison is like-for-like:
  linear_fp16   nn.Linear                      -- cuBLAS, the thing int8 has to beat
  conv1d_fp16   nn.Conv1d(C, N, 1)             -- the advisor's framing, to check it is not different
  int8_awq      the PRODUCTION path             -- quantize_act_int8 + gemm_w8a8_awq[_nout],
                                                  including the K/N zero-padding _gemm does

Padding is included ON PURPOSE. wxax_linear.py:64-68 pads K up to a multiple of 64 and N up to a
multiple of 128, and for the C=192 blocks that is 576->640 on qkv (+11%) and 192->256 on proj (+33%).
Timing the padded call is what the model actually pays.

Run: python docs/qdiff_bridge_2026-08-12/scripts/qkv_shape_bench.py [--batch 128] [--iters 50]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch                                                                # noqa: E402
import torch.nn as nn                                                       # noqa: E402
import torch.nn.functional as F                                             # noqa: E402
import modiff_cutlass as _mc                                                # noqa: E402

OUT = "docs/qdiff_bridge_2026-08-12/data/qkv_shape_bench.json"

#: (C, T, n_blocks, head_dim) for LSUN-churches: model_channels 192, channel_mult [1,2,2,4,4],
#: attention_resolutions [1,2,4,8], num_heads 8, latent 32x32. 21 attention blocks in total.
#: The middle block sits at a 2x2 feature map, hence T=4 rather than 16.
TIERS = [(192, 1024, 5, 24), (384, 256, 5, 48), (384, 64, 5, 48),
         (768, 16, 5, 96), (768, 4, 1, 96)]


def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def make_int8(K, N, dev):
    """Mirror QuantLinearWxAx.__init__'s padding and weight layout (wxax_linear.py:64-68)."""
    Kp = ((K + 63) // 64) * 64
    Np = ((N + 127) // 128) * 128
    qw = torch.randint(-127, 127, (Np, Kp), dtype=torch.int8, device=dev)
    ws = torch.rand(Np, device=dev, dtype=torch.float32) * 0.01 + 0.001
    return Kp, Np, qw, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(1234)
    rows = []

    print(f"batch {a.batch}, median of {a.iters} iters, all fp16 in/out\n")
    hdr = (f"{'shape':>22} {'blk':>4} {'pad':>11} {'linear':>8} {'conv1d':>8} "
           f"{'i8 total':>8} {'=quant':>7} {'+gemm':>7} {'tot/fp16':>9} {'gemm/fp16':>10}")
    print(hdr)
    print("-" * len(hdr))

    for C, T, nblk, hd in TIERS:
        M = a.batch * T
        for kind, N in (("qkv", 3 * C), ("proj", C)):
            K = C
            Kp, Np, qw, ws = make_int8(K, N, dev)
            x = torch.randn(M, K, device=dev, dtype=torch.float16)

            lin = nn.Linear(K, N).to(dev, torch.float16)
            t_lin = bench(lambda: lin(x), a.iters)

            # Conv1d k=1 over [B, C, T] -- the model's original form, to confirm the reshape to
            # Linear is not leaving performance on the table.
            xc = x.view(a.batch, T, K).transpose(1, 2).contiguous()
            cv = nn.Conv1d(K, N, 1).to(dev, torch.float16)
            t_cv = bench(lambda: cv(xc), a.iters)

            a_scale = 0.05

            def run_int8():
                xf = F.pad(x, (0, Kp - K)).contiguous() if Kp != K else x
                xq = _mc.quantize_act_int8(xf, a_scale)
                if Np != N:
                    return _mc.gemm_w8a8_awq_nout(xq, qw, ws, a_scale, N)
                return _mc.gemm_w8a8_awq(xq, qw, ws, a_scale)

            t_i8 = bench(run_int8, a.iters)

            # Split the int8 cost. The quantize is a full extra pass over [M,Kp] and at C=192 that
            # tensor is 131072x192; whether the GEMM or the quantize is the problem changes what a
            # fix would look like (a better tile vs fusing the quantize into the producer, which the
            # conv path already does via step1_static_quantize_fprop).
            xf_pre = F.pad(x, (0, Kp - K)).contiguous() if Kp != K else x
            t_q = bench(lambda: _mc.quantize_act_int8(xf_pre, a_scale), a.iters)
            xq_pre = _mc.quantize_act_int8(xf_pre, a_scale)
            if Np != N:
                t_g = bench(lambda: _mc.gemm_w8a8_awq_nout(xq_pre, qw, ws, a_scale, N), a.iters)
            else:
                t_g = bench(lambda: _mc.gemm_w8a8_awq(xq_pre, qw, ws, a_scale), a.iters)
            padpct = 100.0 * (Kp * Np - K * N) / (K * N)
            rows.append({"C": C, "T": T, "hd": hd, "blocks": nblk, "kind": kind,
                         "M": M, "K": K, "N": N, "Kpad": Kp, "Npad": Np,
                         "pad_pct": padpct, "linear_fp16_ms": t_lin, "conv1d_fp16_ms": t_cv,
                         "int8_awq_ms": t_i8, "int8_quantize_ms": t_q, "int8_gemm_ms": t_g,
                         "speedup_vs_fp16": t_lin / t_i8,
                         "speedup_gemm_only": t_lin / t_g})
            print(f"{f'{M}x{K}x{N}':>22} {nblk:>4} {f'{Kp}x{Np} +{padpct:.0f}%':>11} "
                  f"{t_lin:8.4f} {t_cv:8.4f} {t_i8:8.4f} {t_q:7.4f} {t_g:7.4f} "
                  f"{t_lin / t_i8:8.2f}x {t_lin / t_g:9.2f}x")

    print("\nWeighted by how many blocks run each shape, ms/step over all 21 blocks:")
    tot_l = sum(r["linear_fp16_ms"] * r["blocks"] for r in rows)
    tot_c = sum(r["conv1d_fp16_ms"] * r["blocks"] for r in rows)
    tot_i = sum(r["int8_awq_ms"] * r["blocks"] for r in rows)
    tot_g = sum(r["int8_gemm_ms"] * r["blocks"] for r in rows)
    tot_q = sum(r["int8_quantize_ms"] * r["blocks"] for r in rows)
    print(f"  linear fp16 {tot_l:7.3f}   conv1d fp16 {tot_c:7.3f}   int8 AWQ {tot_i:7.3f} "
          f"(= quantize {tot_q:.3f} + gemm {tot_g:.3f})")
    print(f"  int8 total vs fp16 {tot_l / tot_i:.2f}x     int8 GEMM ALONE vs fp16 {tot_l / tot_g:.2f}x")

    slower = [r for r in rows if r["speedup_vs_fp16"] < 1.0]
    print(f"\nshapes where int8 is SLOWER than fp16: {len(slower)}/{len(rows)}")
    for r in sorted(slower, key=lambda r: r["speedup_vs_fp16"])[:6]:
        print(f"  {r['M']}x{r['K']}x{r['N']:<5} ({r['kind']}, {r['blocks']} blocks) "
              f"{r['speedup_vs_fp16']:.2f}x  pad +{r['pad_pct']:.0f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"batch": a.batch, "iters": a.iters, "rows": rows,
               "weighted_ms": {"linear_fp16": tot_l, "conv1d_fp16": tot_c, "int8_awq": tot_i,
                               "int8_quantize": tot_q, "int8_gemm": tot_g}},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
