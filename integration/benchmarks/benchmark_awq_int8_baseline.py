#!/usr/bin/env python3
"""
Benchmark MoDiff W8A8 GEMM kernels against llm-awq W8A8 GEMM.

The default mode writes JSON/CSV timing results. The profile mode runs one
selected kernel many times so Nsight Compute can attach cleanly, for example:

  ncu --target-processes all --set full --kernel-name regex:dense_kernel0 \
    python integration/benchmarks/benchmark_awq_int8_baseline.py --profile awq
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from modiff_triton.kernels.awq_w8a8 import (  # noqa: E402
    awq_fused_quant_gemm_w8a8,
    awq_gemm_w8a8,
    is_awq_available,
    quantize_awq_per_token,
)
from modiff_triton.kernels.gemm_w8a8 import gemm_w8a8  # noqa: E402
from modiff_triton.kernels.gemm_w8a8_fused import gemm_w8a8_fused  # noqa: E402
from modiff_triton.nn.awq_linear import AWQW8A8BaselineLinear  # noqa: E402
from modiff_triton.nn.linear import W8A8MoDiffLinear  # noqa: E402


DEFAULT_SHAPES = [
    # name, M, N, K
    ("linear_4096", 1024, 4096, 4096),
    ("linear_2048", 2048, 2048, 2048),
    ("ldm_attn_proj", 5376, 512, 512),
    ("ldm_ffn", 5376, 2048, 512),
    ("small_decode", 128, 4096, 4096),
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


def make_case(m: int, n: int, k: int, device: str = "cuda") -> dict:
    x = torch.randn((m, k), device=device, dtype=torch.float16)
    weight_fp = torch.randn((k, n), device=device, dtype=torch.float16) / (k**0.5)
    bias = torch.randn((n,), device=device, dtype=torch.float16)

    weight_max = weight_fp.float().abs().max(dim=0).values
    scale_w = torch.clamp(weight_max / 127.0, min=1e-8).float()
    weight_int8 = torch.round(weight_fp.float() / scale_w.unsqueeze(0)).clamp(-128, 127).to(torch.int8)

    x_int8, scale_a_awq = quantize_awq_per_token(x)
    scale_a_scalar = torch.clamp(x.float().abs().max() / 127.0, min=1e-8).float()
    x_int8_scalar = torch.round(x.float() / scale_a_scalar).clamp(-128, 127).to(torch.int8)

    return {
        "x": x,
        "x_int8_awq": x_int8,
        "scale_a_awq": scale_a_awq,
        "x_int8_scalar": x_int8_scalar,
        "scale_a_scalar": scale_a_scalar,
        "weight_int8": weight_int8,
        "weight_int8_awq": weight_int8.t().contiguous(),
        "scale_w": scale_w,
        "bias": bias,
    }


def tops(m: int, n: int, k: int, ms: float) -> float:
    return (2 * m * n * k) / (ms * 1e-3) / 1e12


def benchmark_shape(name: str, m: int, n: int, k: int, warmup: int, iters: int) -> list[dict]:
    case = make_case(m, n, k)
    x = case["x"]
    weight_int8 = case["weight_int8"]
    weight_int8_awq = case["weight_int8_awq"]
    scale_w = case["scale_w"]
    bias = case["bias"]

    methods: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("ours_triton_fused_quant_gemm", lambda: gemm_w8a8_fused(x, weight_int8, scale_w, bias)),
        (
            "ours_triton_gemm_only_scalar_quant",
            lambda: gemm_w8a8(case["x_int8_scalar"], weight_int8, case["scale_a_scalar"], scale_w, bias.float()),
        ),
        (
            "awq_raw_quant_plus_gemm",
            lambda: awq_fused_quant_gemm_w8a8(
                x,
                weight_int8_awq,
                scale_w,
                bias,
                weight_is_awq_layout=True,
                allow_unsafe_small_m=True,
            ),
        ),
        (
            "awq_integrated_safe_quant_plus_gemm",
            lambda: awq_fused_quant_gemm_w8a8(
                x, weight_int8_awq, scale_w, bias, weight_is_awq_layout=True
            ),
        ),
        (
            "awq_gemm_only_awq_quant",
            lambda: awq_gemm_w8a8(
                case["x_int8_awq"],
                weight_int8_awq,
                scale_w,
                case["scale_a_awq"],
                bias,
                weight_is_awq_layout=True,
                allow_unsafe_small_m=True,
            ),
        ),
        (
            "awq_gemm_only_scalar_quant",
            lambda: awq_gemm_w8a8(
                case["x_int8_scalar"],
                weight_int8_awq,
                scale_w,
                case["scale_a_scalar"].expand(m).half(),
                bias,
                weight_is_awq_layout=True,
                allow_unsafe_small_m=True,
            ),
        ),
        (
            "awq_integrated_safe_gemm_only_scalar",
            lambda: awq_gemm_w8a8(
                case["x_int8_scalar"],
                weight_int8_awq,
                scale_w,
                case["scale_a_scalar"].expand(m).half(),
                bias,
                weight_is_awq_layout=True,
            ),
        ),
    ]

    rows = []
    reference = None
    for method, fn in methods:
        timing = cuda_bench(fn, warmup, iters)
        out = fn()
        if reference is None:
            reference = out.float()
        max_abs_err = (out.float() - reference).abs().max().item()
        row = {
            "shape": name,
            "M": m,
            "N": n,
            "K": k,
            "method": method,
            **timing,
            "tops": tops(m, n, k, timing["median_ms"]),
            "max_abs_err_vs_ours_fused": max_abs_err,
        }
        rows.append(row)
    return rows


def run_profile_target(target: str, shape: tuple[int, int, int], repeats: int) -> None:
    m, n, k = shape
    case = make_case(m, n, k)
    x = case["x"]
    weight_int8 = case["weight_int8"]
    weight_int8_awq = case["weight_int8_awq"]
    scale_w = case["scale_w"]
    bias = case["bias"]

    targets = {
        "ours_fused": lambda: gemm_w8a8_fused(x, weight_int8, scale_w, bias),
        "ours_gemm": lambda: gemm_w8a8(case["x_int8_awq"], weight_int8, case["scale_a_awq"].float(), scale_w, bias.float()),
        "awq": lambda: awq_gemm_w8a8(
            case["x_int8_awq"],
            weight_int8_awq,
            scale_w,
            case["scale_a_awq"],
            bias,
            weight_is_awq_layout=True,
            allow_unsafe_small_m=True,
        ),
        "awq_quant_plus_gemm": lambda: awq_fused_quant_gemm_w8a8(
            x, weight_int8_awq, scale_w, bias, weight_is_awq_layout=True
        ),
    }
    fn = targets[target]
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()


def smoke_linear_modules() -> dict:
    linear = torch.nn.Linear(512, 512, bias=True).cuda().half()
    ours = W8A8MoDiffLinear.from_linear(linear)
    ours.set_modulation(False)
    awq = AWQW8A8BaselineLinear.from_linear(linear)
    x = torch.randn((256, 512), device="cuda", dtype=torch.float16)
    out_ours = ours(x).float()
    out_awq = awq(x).float()
    return {
        "ours_shape": list(out_ours.shape),
        "awq_shape": list(out_awq.shape),
        "max_abs_diff": (out_ours - out_awq).abs().max().item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "integration/results/awq_int8_baseline"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", choices=["ours_fused", "ours_gemm", "awq", "awq_quant_plus_gemm"])
    parser.add_argument("--profile-shape", default="1024,4096,4096", help="M,N,K")
    parser.add_argument("--profile-repeats", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not is_awq_available():
        raise RuntimeError("AWQ extension is not available")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    if args.profile:
        shape = tuple(int(x) for x in args.profile_shape.split(","))
        if len(shape) != 3:
            raise ValueError("--profile-shape must be M,N,K")
        run_profile_target(args.profile, shape, args.profile_repeats)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, m, n, k in DEFAULT_SHAPES:
        print(f"Benchmarking {name}: M={m}, N={n}, K={k}")
        rows.extend(benchmark_shape(name, m, n, k, args.warmup, args.iters))

    smoke = smoke_linear_modules()
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "awq_available": True,
        "linear_module_smoke": smoke,
        "note": "AWQ output is FP16; MoDiff Triton GEMM outputs FP32 except fused path uses input dtype.",
    }

    json_path = out_dir / "awq_int8_baseline_results.json"
    csv_path = out_dir / "awq_int8_baseline_results.csv"
    with json_path.open("w") as f:
        json.dump({"metadata": metadata, "results": rows}, f, indent=2)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
