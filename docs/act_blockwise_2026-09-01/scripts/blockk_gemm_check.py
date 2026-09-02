"""Correctness + cost of the blockwise-along-K int8 GEMM mainloop.

`gemm_w8a8_blockk` dequantizes once per K-block INSIDE the mainloop, which is the only way a
scale along the reduction axis can be applied at all. The same kernel, same tile config, same
everything, instantiated with BLOCKWISE=false and a scalar alpha, is the control -- so the time
difference is the mainloop dequant and not a tiling change.

CUTLASS `gemm_w8a8_awq` is also timed, to show where the hand-written tile sits independently.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/act_blockwise_2026-09-01/scripts/blockk_gemm_check.py
"""
from __future__ import annotations

import os
import statistics
import sys

sys.path[:0] = ["/workspace/MoDiff"]
os.chdir("/workspace/MoDiff")

import torch  # noqa: E402
import modiff_cutlass as MC  # noqa: E402

QMAX = 127.0


def quant_blockwise(x: torch.Tensor, blk: int):
    """x [M,K] fp32 -> int8 codes + [K/blk, M] fp32 scales (K-block major)."""
    M, K = x.shape
    xb = x.reshape(M, K // blk, blk)
    s = xb.abs().amax(-1).clamp_min(1e-12) / QMAX          # [M, K/blk]
    q = (xb / s.unsqueeze(-1)).round().clamp_(-QMAX, QMAX).to(torch.int8).reshape(M, K)
    return q.contiguous(), s.t().contiguous()               # [K/blk, M]


def ref(q, w, sblk, wscale, blk):
    """Exact reference: sum over K-blocks of (int32 partial * block scale) * w_scale."""
    M, K = q.shape
    N = w.shape[0]
    out = torch.zeros(M, N, device=q.device, dtype=torch.float32)
    for kb in range(K // blk):
        p = (q[:, kb * blk:(kb + 1) * blk].float()
             @ w[:, kb * blk:(kb + 1) * blk].float().t())    # [M,N] int32-exact in fp32
        out += p * sblk[kb].unsqueeze(1)
    return out * wscale.unsqueeze(0)


def bench(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"GPU {torch.cuda.get_device_name(0)}\n")
    empty = torch.empty(0, device=dev, dtype=torch.float32)

    shapes = [(4096, 1024, 1024), (8192, 1152, 1152), (16384, 1024, 512), (32768, 512, 512)]
    print(f"{'M':>6} {'K':>6} {'N':>5} {'blk':>4} | {'max rel err':>11} | "
          f"{'ctrl ms':>8} {'blkw ms':>8} {'cost':>6} | {'cutlass ms':>10}")
    for (M, K, N) in shapes:
        x = torch.randn(M, K, device=dev) * torch.rand(M, K // 32, device=dev).repeat_interleave(
            32, 1).clamp_min(0.02) * 4          # per-block-varying dynamic range, the case blockwise is for
        w = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        wscale = (torch.rand(N, device=dev) * 0.01 + 0.001)

        for blk in (32, 64):
            q, sblk = quant_blockwise(x, blk)
            got = MC.gemm_w8a8_blockk(q, w, wscale, sblk, 0.0, blk)
            exp = ref(q, w, sblk, wscale, blk)
            err = float(((got.float() - exp).abs() / exp.abs().clamp_min(1e-3)).max())

            t_b = bench(lambda: MC.gemm_w8a8_blockk(q, w, wscale, sblk, 0.0, blk))
            t_c = bench(lambda: MC.gemm_w8a8_blockk(q, w, wscale, empty, 0.01, blk))
            t_cut = bench(lambda: MC.gemm_w8a8_awq(q, w, wscale, 0.01)) if N % 128 == 0 else float("nan")
            print(f"{M:>6} {K:>6} {N:>5} {blk:>4} | {err:>11.2e} | "
                  f"{t_c:>8.3f} {t_b:>8.3f} {t_b / t_c:>5.2f}x | {t_cut:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
