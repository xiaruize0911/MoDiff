"""Correctness + cost of the blockwise-along-C int8 CONV mainloop.

`conv2d_int8_blockk` applies a per-(input pixel, channel block) activation scale inside the
mainloop. For R,S>1 that scale depends on the reduction index (r,s) as well as on the output
pixel, so no epilogue can express it -- see the kernel header.

Controls, all timed on the same tensors:
  ctrl     the same kernel with BLOCKWISE=false and a scalar alpha (isolates the dequant)
  cutlass  conv2d_int8_fprop, the shipped scalar-alpha CUTLASS conv (is the tile competitive?)

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/act_blockwise_2026-09-01/scripts/blockk_conv_check.py
"""
from __future__ import annotations

import os
import statistics
import sys

sys.path[:0] = ["/workspace/MoDiff"]
os.chdir("/workspace/MoDiff")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import modiff_cutlass as MC  # noqa: E402

QMAX = 127.0


def quant_blockwise_nhwc(x: torch.Tensor, blk: int):
    """x [N,C,H,W] fp32 -> int8 channels_last codes + [N,H,W,C/blk] fp32 scales."""
    n, c, h, w = x.shape
    t = x.permute(0, 2, 3, 1).contiguous()                       # [N,H,W,C]
    tb = t.reshape(n, h, w, c // blk, blk)
    s = tb.abs().amax(-1).clamp_min(1e-12) / QMAX                # [N,H,W,C/blk]
    q = (tb / s.unsqueeze(-1)).round().clamp_(-QMAX, QMAX).to(torch.int8).reshape(n, h, w, c)
    q = q.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    return q, s.contiguous()


def ref_conv(q, sblk, wq, wscale, blk, stride, pad):
    """Exact reference: dequantize the activation blockwise in fp32, then a fp32 conv."""
    n, c, h, w = q.shape
    s = sblk.repeat_interleave(blk, dim=-1)                      # [N,H,W,C]
    xdq = q.float().permute(0, 2, 3, 1) * s
    xdq = xdq.permute(0, 3, 1, 2).contiguous()
    wf = wq.float().permute(0, 3, 1, 2).contiguous()             # [K,R,S,C] -> [K,C,R,S]
    out = F.conv2d(xdq, wf, None, stride, pad)
    return out * wscale.view(1, -1, 1, 1)


def bench(fn, iters=30, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"GPU {torch.cuda.get_device_name(0)}\n")
    empty = torch.empty(0, device=dev, dtype=torch.float32)

    # (N, C, H, W, Kout, R, stride, pad) -- the churches UNet's real conv shapes at batch 32.
    shapes = [
        (32, 192, 32, 32, 192, 3, 1, 1),
        (32, 384, 16, 16, 384, 3, 1, 1),
        (32, 576, 8, 8, 576, 3, 1, 1),
        (32, 192, 32, 32, 192, 1, 1, 0),
        (16, 384, 32, 32, 384, 3, 1, 1),
    ]
    print(f"{'N':>3} {'C':>4} {'HxW':>7} {'K':>4} {'R':>2} {'blk':>4} | {'relL2':>9} {'fp16 floor':>10} | "
          f"{'ctrl ms':>8} {'blkw ms':>8} {'cost':>6} | {'cutlass':>8}")
    for (n, c, h, w, kout, r, stride, pad) in shapes:
        x = torch.randn(n, c, h, w, device=dev) * torch.rand(
            n, c // 32, h, w, device=dev).repeat_interleave(32, 1).clamp_min(0.02) * 4
        wq = torch.randint(-127, 128, (kout, r, r, c), device=dev, dtype=torch.int8)
        wscale = torch.rand(kout, device=dev) * 0.01 + 0.001

        for blk in (32, 64):
            q, sblk = quant_blockwise_nhwc(x, blk)
            got = MC.conv2d_int8_blockk(q, wq, wscale, sblk, 0.0, blk, stride, pad)
            exp = ref_conv(q, sblk, wq, wscale, blk, stride, pad)
            rel = float((got.float() - exp).norm() / exp.norm())
            floor = float((exp.half().float() - exp).norm() / exp.norm())

            t_b = bench(lambda: MC.conv2d_int8_blockk(q, wq, wscale, sblk, 0.0, blk, stride, pad))
            t_c = bench(lambda: MC.conv2d_int8_blockk(q, wq, wscale, empty, 0.01, blk, stride, pad))
            try:
                alpha = torch.tensor([0.01], device=dev)
                t_cut = bench(lambda: MC.conv2d_int8_fprop(
                    q, wq, alpha, torch.empty(0, device=dev), stride, stride, pad, pad, 1, 1))
            except Exception:
                t_cut = float("nan")
            print(f"{n:>3} {c:>4} {f'{h}x{w}':>7} {kout:>4} {r:>2} {blk:>4} | {rel:>9.2e} "
                  f"{floor:>10.2e} | {t_c:>8.3f} {t_b:>8.3f} {t_b / t_c:>5.2f}x | {t_cut:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
