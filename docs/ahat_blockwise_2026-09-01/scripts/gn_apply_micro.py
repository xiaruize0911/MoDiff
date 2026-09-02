"""Isolated gn_apply timing: fp16 a_hat vs along-C B=32 int8, one real layer shape.

The E2E profile can only say gn_apply is +1.2 ms/step in aggregate. This drives the
kernel directly so a single change can be timed in seconds instead of a 5-minute
rebuild plus a 40-second sample, and so ncu has one clean kernel to look at.
"""
from __future__ import annotations
import os, sys, statistics
ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT]
import torch  # noqa: E402
import modiff_cutlass  # noqa: E402

# input_blocks mid-resolution shape: batch 128, C=384, 16x16.
N, C, H, W, G = 128, 384, 16, 16, 32
REPEAT = 50


def run(int8_ahat: bool):
    dev = "cuda"
    x = torch.randn(N, C, H, W, device=dev, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    gamma = torch.randn(C, device=dev, dtype=torch.float16)
    beta = torch.randn(C, device=dev, dtype=torch.float16)
    scale = torch.tensor([8.0], device=dev, dtype=torch.float32)
    empty = torch.empty(0, device=dev)
    empty_i = torch.empty(0, device=dev, dtype=torch.int32)
    if int8_ahat:
        cache = torch.zeros(N, C, H, W, device=dev, dtype=torch.int8).contiguous(
            memory_format=torch.channels_last)
        ahat_scale = torch.ones(N, H, W, C // 32, device=dev, dtype=torch.float32)
    else:
        cache = torch.zeros(N, C, H, W, device=dev, dtype=torch.float16).contiguous(
            memory_format=torch.channels_last)
        ahat_scale = empty

    def once():
        modiff_cutlass.group_norm_silu_delta_quantize_nhwc(
            x, gamma, beta, cache, G, 1e-5, True, scale, empty, empty, empty,
            empty, empty, empty, empty_i, 127.0, False, 1.0, False, True, ahat_scale)

    for _ in range(10):
        once()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(REPEAT):
            once()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / REPEAT)
    return statistics.median(ts)


def main():
    fp = run(False)
    i8 = run(True)
    elems = N * C * H * W
    print(f"shape {N}x{C}x{H}x{W}  {elems/1e6:.1f} M elem", flush=True)
    print(f"  fp16 a_hat  {fp*1e3:8.1f} us   {elems*7/fp*1e-9:6.0f} GB/s-equiv", flush=True)
    print(f"  int8 B=32   {i8*1e3:8.1f} us   {elems*5.25/i8*1e-9:6.0f} GB/s-equiv", flush=True)
    print(f"  ratio       {i8/fp:.3f}x", flush=True)


if __name__ == "__main__":
    main()
