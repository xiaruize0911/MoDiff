#!/usr/bin/env python3
"""Kernel-to-kernel conv speed: best-of-N tuned-tile INT8 (deep-fuse fp16 out) and
INT4 vs cuDNN FP16, across representative ResNet-50 shapes. This is the Phase-0
gate: does per-shape tile selection let our int8 conv beat cuDNN fp16 on the
shapes where the single-tile kernel loses?

  python integration/benchmarks/microbench_conv_tuned.py --batch 64
"""
import os, sys, argparse, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import modiff_cutlass as mc
from integration.kernels.int8_optimized import OptimizedInt8Conv2d
from integration.kernels.int4_optimized import OptimizedInt4Conv2d

# (Cin, Cout, HW, kernel, stride) — a spread of ResNet-50 shapes (losing + winning)
SHAPES = [
    (64,  256, 56, 1, 1),   # K=64   1x1  (was losing)
    (64,   64, 56, 3, 1),   # K=576  3x3  (was 0.63x)
    (256,  64, 56, 1, 1),   # K=256  1x1
    (128, 512, 28, 1, 1),   # K=128  1x1
    (128, 128, 28, 3, 1),   # K=1152 3x3
    (256, 256, 14, 3, 1),   # K=2304 3x3  (was 1.70x)
    (1024,256, 14, 1, 1),   # K=1024 1x1
    (256,1024, 14, 1, 1),   # K=256  1x1
    (512, 512,  7, 3, 1),   # K=4608 3x3
    (512,2048,  7, 1, 1),   # K=512  1x1
    (2048,512,  7, 1, 1),   # K=2048 1x1
]


def bench(fn, iters=60, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    dev = "cuda"
    B = a.batch
    ncfg = mc.conv2d_int8_num_tuned_configs()
    print(f"batch={B}  tuned int8 configs={ncfg}\n")
    print(f"{'shape (Cin,Cout,HW,k,s)':30}{'K':>6}{'fp16':>9}{'int8*':>9}{'cfg':>4}{'int4':>9}"
          f"{'i8/fp16':>9}{'i4/fp16':>9}")
    tot = {"fp16": 0.0, "int8": 0.0, "int4": 0.0}
    for (Cin, Cout, HW, k, st) in SHAPES:
        pad = k // 2
        x = torch.randn(B, Cin, HW, HW, device=dev, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        # cuDNN fp16 reference
        fp16_conv = nn.Conv2d(Cin, Cout, k, stride=st, padding=pad, bias=False).cuda().half().to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t_fp16 = bench(lambda: fp16_conv(x))
        # int8: build calibrated conv, quantize input, time each tuned tile
        base = nn.Conv2d(Cin, Cout, k, stride=st, padding=pad, bias=False)
        q8 = OptimizedInt8Conv2d(base).cuda()
        q8.set_static_scale(4.0); q8.set_standard_output_fp16(True); q8.enable_modiff(False)
        q8._ensure_conv_caches(dev)
        xi = q8.quantize_input(x)
        Hout = (HW + 2*pad - (k-1) - 1)//st + 1
        out16 = torch.empty(B, Cout, Hout, Hout, device=dev, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        wscale_h = q8.weight_scale_channel_half.view(-1)
        best, best_cfg = float("inf"), -1
        for cid in range(ncfg):
            try:
                mc.conv2d_int8_dequant_fp16_tuned(xi, q8.weight_int8, q8._cached_alpha_tensor,
                                                  wscale_h, out16, cid, st, st, pad, pad, 1, 1)
            except Exception:
                continue
            t = bench(lambda: mc.conv2d_int8_dequant_fp16_tuned(
                xi, q8.weight_int8, q8._cached_alpha_tensor, wscale_h, out16, cid, st, st, pad, pad, 1, 1))
            if t < best:
                best, best_cfg = t, cid
        # int4 (existing single-config kernel)
        try:
            q4 = OptimizedInt4Conv2d(base).cuda()
            q4.set_static_scale(4.0); q4.set_standard_output_fp16(True); q4.enable_modiff(False)
            q4._ensure_conv_caches(dev)
            packed = mc.scale_quantize_and_pack(x.float(), q4._cached_scale_tensor)
            t_int4 = bench(lambda: q4.forward_from_int4(packed, HW, HW))
        except Exception as ex:
            t_int4 = float("nan")
        K = Cin * k * k
        tot["fp16"] += t_fp16; tot["int8"] += best; tot["int4"] += (t_int4 if t_int4 == t_int4 else 0)
        print(f"{str((Cin,Cout,HW,k,st)):30}{K:>6}{t_fp16:>9.3f}{best:>9.3f}{best_cfg:>4}{t_int4:>9.3f}"
              f"{t_fp16/best:>8.2f}x{(t_fp16/t_int4 if t_int4==t_int4 else 0):>8.2f}x")
    print(f"\n{'TOTAL':30}{'':>6}{tot['fp16']:>9.3f}{tot['int8']:>9.3f}{'':>4}{tot['int4']:>9.3f}"
          f"{tot['fp16']/tot['int8']:>8.2f}x{tot['fp16']/tot['int4']:>8.2f}x")


if __name__ == "__main__":
    main()
