#!/usr/bin/env python3
"""
Focused INT8 kernel benchmark for the exact modules used by benchmark_ldm.py.

This script benchmarks:
  - integration.kernels.int8_optimized.OptimizedInt8Conv2d baseline path
  - integration.kernels.int8_linear.OptimizedInt8Linear backends: fp16, int_gemm, awq

Profile examples:
  env -u RUNPOD_API_KEY -u VSCODE_CLI_REQUIRE_TOKEN \
    nsys profile -t cuda,nvtx -o integration/results/ldm_int8_kernel_compare/nsys/linear_awq \
    python integration/benchmarks/benchmark_ldm_int8_kernels.py --profile linear_awq

  ncu --target-processes all --launch-skip 20 --launch-count 1 --set roofline \
    python integration/benchmarks/benchmark_ldm_int8_kernels.py --profile linear_awq
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from integration.kernels.int8_linear import OptimizedInt8Linear  # noqa: E402
from integration.kernels.int8_optimized import OptimizedInt8Conv2d  # noqa: E402


CONV_SHAPES = [
    # name, B, Cin, H, W, Cout, K, stride, padding
    ("res_128_64", 42, 128, 64, 64, 128, 3, 1, 1),
    ("res_128_32", 42, 128, 32, 32, 128, 3, 1, 1),
    ("down_128_256_32", 42, 128, 32, 32, 256, 3, 1, 1),
    ("res_256_32", 42, 256, 32, 32, 256, 3, 1, 1),
    ("res_256_16", 42, 256, 16, 16, 256, 3, 1, 1),
    ("down_256_512_16", 42, 256, 16, 16, 512, 3, 1, 1),
    ("mid_512_8", 42, 512, 8, 8, 512, 3, 1, 1),
    ("up_512_256_16", 42, 512, 16, 16, 256, 3, 1, 1),
    ("up_256_128_32", 42, 256, 32, 32, 128, 3, 1, 1),
    ("up_128_64", 42, 128, 64, 64, 128, 3, 1, 1),
    ("pointwise_320_16", 42, 320, 16, 16, 320, 1, 1, 0),
]

LINEAR_SHAPES = [
    # representative flattened attention/MLP-ish shapes
    ("attn_proj_320_m2688", 2688, 320, 320),
    ("attn_proj_320_m5376", 5376, 320, 320),
    ("attn_proj_512", 5376, 512, 512),
    ("attn_proj_640_m2688", 2688, 640, 640),
    ("ffn_512_2048", 5376, 512, 2048),
    ("ffn_640_2560", 2688, 640, 2560),
    ("large_2048", 2048, 2048, 2048),
    ("large_4096", 1024, 4096, 4096),
    ("small_m_4096", 128, 4096, 4096),
]


def cuda_bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def conv_ops(b: int, cin: int, h: int, w: int, cout: int, k: int, stride: int, pad: int) -> int:
    hout = (h + 2 * pad - k) // stride + 1
    wout = (w + 2 * pad - k) // stride + 1
    return 2 * b * hout * wout * cout * cin * k * k


def make_conv_case(b: int, cin: int, h: int, w: int, cout: int, k: int, stride: int, pad: int):
    conv = nn.Conv2d(cin, cout, k, stride=stride, padding=pad, bias=True).cuda().eval()
    opt = OptimizedInt8Conv2d(conv, layer_name="bench").cuda().eval()
    opt.set_static_scale(32.0)
    opt.enable_modiff(False)
    opt.set_standard_output_fp16(True)
    x = torch.randn((b, cin, h, w), device="cuda", dtype=torch.float16)
    x = x.contiguous(memory_format=torch.channels_last)
    conv = conv.half().to(memory_format=torch.channels_last)
    return conv, opt, x


def make_linear_case(m: int, k: int, n: int, backend: str):
    linear = nn.Linear(k, n, bias=True).cuda().half().eval()
    opt = OptimizedInt8Linear(linear, layer_name="bench", backend=backend, int_gemm_min_m=1).cuda().eval()
    opt.enable_modiff(False)
    opt.set_standard_output_fp16(True)
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    return opt, x


def _aggregate_rounds(samples: list[dict]) -> dict:
    keys = ("median_ms", "mean_ms", "min_ms", "max_ms")
    out = {
        "rounds": len(samples),
        "round_median_ms_min": min(s["median_ms"] for s in samples),
        "round_median_ms_max": max(s["median_ms"] for s in samples),
        "round_median_ms_stdev": statistics.stdev([s["median_ms"] for s in samples]) if len(samples) > 1 else 0.0,
    }
    for key in keys:
        values = [s[key] for s in samples]
        out[key] = statistics.median(values)
    return out


def _bench_rounds(fn: Callable[[], torch.Tensor], warmup: int, iters: int, rounds: int) -> dict:
    samples = []
    for _ in range(rounds):
        samples.append(cuda_bench(fn, warmup, iters))
    return _aggregate_rounds(samples)


def benchmark_all(warmup: int, iters: int, rounds: int) -> list[dict]:
    rows: list[dict] = []
    torch.manual_seed(0)
    for name, b, cin, h, w, cout, k, stride, pad in CONV_SHAPES:
        conv, opt, x = make_conv_case(b, cin, h, w, cout, k, stride, pad)
        ops = conv_ops(b, cin, h, w, cout, k, stride, pad)
        for backend, fn in (
            ("torch_fp16_cudnn_conv2d", lambda conv=conv, x=x: conv(x)),
            ("OptimizedInt8Conv2d benchmark_ldm baseline", lambda opt=opt, x=x: opt(x)),
        ):
            timing = _bench_rounds(fn, warmup, iters, rounds)
            rows.append({
                "kind": "conv",
                "shape": name,
                "backend": backend,
                "M_or_B": b,
                "N_or_Cout": cout,
                "K_or_Cin": cin,
                "H": h,
                "W": w,
                "kernel": k,
                "stride": stride,
                "padding": pad,
                **timing,
                "tops": ops / (timing["median_ms"] * 1e-3) / 1e12,
            })
    for name, m, k, n in LINEAR_SHAPES:
        refs = {}
        for backend in ("fp16", "int_gemm", "awq"):
            opt, x = make_linear_case(m, k, n, backend)
            timing = _bench_rounds(lambda opt=opt, x=x: opt(x), warmup, iters, rounds)
            out = opt(x).float()
            if "fp16" not in refs:
                refs["fp16"] = out
            err = (out - refs["fp16"]).abs().max().item()
            rows.append({
                "kind": "linear",
                "shape": name,
                "backend": backend,
                "M_or_B": m,
                "N_or_Cout": n,
                "K_or_Cin": k,
                **timing,
                "tops": (2 * m * n * k) / (timing["median_ms"] * 1e-3) / 1e12,
                "max_abs_err_vs_fp16_backend": err,
            })
    return rows


def run_profile(target: str, repeats: int):
    torch.manual_seed(0)
    if target == "conv":
        opt, x = make_conv_case(32, 128, 32, 32, 128, 3, 1, 1)
        fn = lambda: opt(x)
    elif target == "linear_int_gemm":
        opt, x = make_linear_case(1024, 4096, 4096, "int_gemm")
        fn = lambda: opt(x)
    elif target == "linear_awq":
        opt, x = make_linear_case(1024, 4096, 4096, "awq")
        fn = lambda: opt(x)
    else:
        raise ValueError(target)

    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(target)
    for _ in range(repeats):
        fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "integration/results/ldm_int8_kernel_compare"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--profile", choices=["conv", "linear_int_gemm", "linear_awq"])
    parser.add_argument("--profile-repeats", type=int, default=200)
    args = parser.parse_args()

    if args.profile:
        run_profile(args.profile, args.profile_repeats)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = benchmark_all(args.warmup, args.iters, args.rounds)
    with (out_dir / "ldm_int8_kernel_compare.json").open("w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "results": rows}, f, indent=2)
    with (out_dir / "ldm_int8_kernel_compare.csv").open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_dir}")
    for row in rows:
        print(f"{row['kind']:6s} {row['shape']:14s} {row['backend']:42s} {row['median_ms']:.4f} ms {row['tops']:.2f} TOPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
