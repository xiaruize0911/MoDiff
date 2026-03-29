#!/usr/bin/env python3
"""
Focused A40 layerwise benchmark for INT8 vs INT4.

This script measures three increasingly realistic stages for the representative
LDM convolution shapes used elsewhere in this repository:

1. Raw CUTLASS convolution only
   - pre-quantized input + pre-quantized weight
   - measures only `conv2d_int8_fprop` vs `conv2d_int4_fprop`

2. Fused baseline computation (no MoDiff)
   - static activation quantize(+pack) -> CUTLASS conv -> dequant-by-weight-scale
   - mirrors the steady-state baseline path in `OptimizedInt8Conv2d` /
     `OptimizedInt4Conv2d` when temporal caching is disabled

3. Fused MoDiff hot path
   - static MoDiff: `step1_static_quantize_*` + `conv2d_*_fprop_o_hat`
   - dynamic MoDiff: `step1_quantize_*` + `conv2d_*_fprop_o_hat`

The goal is to answer the practical question behind the benchmark report:
where does the expected INT4/INT8 speedup shrink, and by how much on this GPU?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import modiff_cutlass
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Failed to import modiff_cutlass. Build the extension first and ensure "
        "Torch shared libraries are on LD_LIBRARY_PATH."
    ) from exc


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


Shape = Tuple[int, int, int, int]


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


def benchmark_cuda(fn: Callable[[], None], warmup: int = 20, iters: int = 100) -> TimingStats:
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [float(s.elapsed_time(e)) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    return TimingStats(
        median_ms=statistics.median(times_ms),
        mean_ms=sum(times_ms) / len(times_ms),
        min_ms=times_ms[0],
        max_ms=times_ms[-1],
    )


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values stored in an int8 tensor, 2 per byte."""
    last_dim = tensor.shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(f"Last dimension must be even for INT4 packing, got {last_dim}")

    reshaped = tensor.view(*tensor.shape[:-1], last_dim // 2, 2)
    low = reshaped[..., 0] & 0x0F
    high = (reshaped[..., 1] & 0x0F) << 4
    return (low | high).to(torch.int8)


def format_shape(shape: Shape) -> str:
    n, c, h, w = shape
    return f"N={n}, C={c}, H=W={h}"


def speedup(int8_ms: float, int4_ms: float) -> float:
    return float(int8_ms) / float(int4_ms) if int4_ms > 0.0 else math.inf


def make_weight_quantized(conv: nn.Conv2d):
    """Prepare quantized weights/scales for both INT8 and INT4 CUTLASS paths."""
    w_data = conv.weight.detach().float()
    out_channels = w_data.shape[0]
    empty_bias = torch.empty(0, device=w_data.device)

    w_flat = w_data.reshape(out_channels, -1)

    ch_scale_int8 = torch.clamp(w_flat.abs().max(dim=1).values / 127.0, min=1e-8)
    w_q8 = (w_flat / ch_scale_int8.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    w_q8 = w_q8.reshape_as(w_data).permute(0, 2, 3, 1).contiguous()
    w_scale_int8 = ch_scale_int8.view(1, out_channels, 1, 1).contiguous()

    ch_scale_int4 = torch.clamp(w_flat.abs().max(dim=1).values / 7.0, min=1e-8)
    w_q4 = (w_flat / ch_scale_int4.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
    w_q4 = w_q4.reshape_as(w_data).permute(0, 2, 3, 1).contiguous()
    w_packed = pack_int4(w_q4).contiguous()
    w_scale_int4 = ch_scale_int4.view(1, out_channels, 1, 1).contiguous()

    return {
        "int8": {"weight": w_q8, "weight_scale": w_scale_int8, "empty_bias": empty_bias},
        "int4": {"weight": w_packed, "weight_scale": w_scale_int4, "empty_bias": empty_bias},
    }


def make_random_tensors(shape: Shape, device: str = "cuda"):
    n, c, h, w = shape
    x = torch.randn(n, c, h, w, device=device, dtype=torch.float32).contiguous(
        memory_format=torch.channels_last
    )
    cache = torch.randn_like(x)
    cache_zero = torch.zeros_like(x)
    o_hat = torch.randn(n, c, h, w, device=device, dtype=torch.float32).contiguous(
        memory_format=torch.channels_last
    )
    return x, cache, cache_zero, o_hat


def measure_shape(shape: Shape, warmup: int, iters: int) -> Dict[str, object]:
    n, c, h, w = shape
    conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False).cuda().eval()
    conv = conv.to(memory_format=torch.channels_last)

    x, cache, cache_zero, o_hat = make_random_tensors(shape)
    weight_data = make_weight_quantized(conv)

    # Shared buffers for MoDiff step1 kernels
    residual_buf = torch.empty_like(x)
    absmax_buf = torch.zeros(1, device="cuda", dtype=torch.float32)
    scale_buf = torch.empty(1, device="cuda", dtype=torch.float32)
    inv_scale_buf = torch.empty(1, device="cuda", dtype=torch.float32)
    retire_count = torch.zeros(1, device="cuda", dtype=torch.int32)
    smooth_inv = torch.empty(0, device="cuda", dtype=torch.float32)

    # Static scales based on the current input tensor.
    static_scale_int8 = torch.tensor(
        [127.0 / max(float(x.abs().amax().item()), 1e-6)],
        device="cuda",
        dtype=torch.float32,
    )
    static_inv_int8 = 1.0 / static_scale_int8

    static_scale_int4 = torch.tensor(
        [7.0 / max(float(x.abs().amax().item()), 1e-6)],
        device="cuda",
        dtype=torch.float32,
    )
    static_inv_int4 = 1.0 / static_scale_int4

    # Prequantized inputs for raw conv-only benchmarking
    x_int8 = modiff_cutlass.scale_quantize_int8(x, static_scale_int8)
    x_int4 = modiff_cutlass.scale_quantize_and_pack(x, static_scale_int4)

    int8_weight = weight_data["int8"]["weight"]
    int8_weight_scale = weight_data["int8"]["weight_scale"]
    int4_weight = weight_data["int4"]["weight"]
    int4_weight_scale = weight_data["int4"]["weight_scale"]
    empty_bias = weight_data["int8"]["empty_bias"]

    # ------------------------------------------------------------------
    # 1) Raw conv-only speed
    # ------------------------------------------------------------------
    raw_int8 = benchmark_cuda(
        lambda: modiff_cutlass.conv2d_int8_fprop(
            x_int8, int8_weight, static_inv_int8.view(1), empty_bias, 1, 1, 1, 1, 1, 1
        ),
        warmup=warmup,
        iters=iters,
    )

    raw_int4 = benchmark_cuda(
        lambda: modiff_cutlass.conv2d_int4_fprop(
            x_int4, int4_weight, static_inv_int4.view(1), empty_bias, 1, 1, 1, 1, 1, 1
        ),
        warmup=warmup,
        iters=iters,
    )

    # ------------------------------------------------------------------
    # 2) Baseline fused (no MoDiff)
    # Mirrors OptimizedInt{8,4}Conv2d steady-state baseline:
    #   quantize -> conv -> dequant-by-weight-scale
    # ------------------------------------------------------------------
    def baseline_int8():
        q = modiff_cutlass.scale_quantize_int8(x, static_scale_int8)
        out_raw = modiff_cutlass.conv2d_int8_fprop(
            q, int8_weight, static_inv_int8.view(1), empty_bias, 1, 1, 1, 1, 1, 1
        )
        _ = out_raw * int8_weight_scale

    def baseline_int4():
        q = modiff_cutlass.scale_quantize_and_pack(x, static_scale_int4)
        out_raw = modiff_cutlass.conv2d_int4_fprop(
            q, int4_weight, static_inv_int4.view(1), empty_bias, 1, 1, 1, 1, 1, 1
        )
        _ = out_raw * int4_weight_scale

    baseline8 = benchmark_cuda(baseline_int8, warmup=warmup, iters=iters)
    baseline4 = benchmark_cuda(baseline_int4, warmup=warmup, iters=iters)

    # ------------------------------------------------------------------
    # 3a) MoDiff fused static hot path
    # Mirrors calibrated static MoDiff:
    #   step1_static_quantize_* -> conv2d_*_fprop_o_hat
    # ------------------------------------------------------------------
    def modiff_static_int8_step1():
        return modiff_cutlass.step1_static_quantize_fprop(x, cache, static_scale_int8.view(1), smooth_inv)

    x8_mod_static = modiff_cutlass.step1_static_quantize_fprop(
        x, cache, static_scale_int8.view(1), smooth_inv
    )

    def modiff_static_int8_conv():
        modiff_cutlass.conv2d_int8_fprop_o_hat(
            x8_mod_static,
            int8_weight,
            static_inv_int8.view(1),
            int8_weight_scale.view(-1),
            o_hat,
            1, 1, 1, 1, 1, 1,
        )

    def modiff_static_int4_step1():
        return modiff_cutlass.step1_static_quantize_pack_int4_fprop(
            x, cache, static_scale_int4.view(1), smooth_inv
        )

    x4_mod_static = modiff_cutlass.step1_static_quantize_pack_int4_fprop(
        x, cache, static_scale_int4.view(1), smooth_inv
    )

    def modiff_static_int4_conv():
        modiff_cutlass.conv2d_int4_fprop_o_hat(
            x4_mod_static,
            int4_weight,
            static_inv_int4.view(1),
            int4_weight_scale.view(-1),
            o_hat,
            1, 1, 1, 1, 1, 1,
        )

    modiff_static8_step1 = benchmark_cuda(modiff_static_int8_step1, warmup=warmup, iters=iters)
    modiff_static8_conv = benchmark_cuda(modiff_static_int8_conv, warmup=warmup, iters=iters)
    modiff_static4_step1 = benchmark_cuda(modiff_static_int4_step1, warmup=warmup, iters=iters)
    modiff_static4_conv = benchmark_cuda(modiff_static_int4_conv, warmup=warmup, iters=iters)

    # ------------------------------------------------------------------
    # 3b) MoDiff fused dynamic hot path
    # Mirrors current dynamic hot path used in the existing microbenchmark:
    #   step1_quantize_* -> conv2d_*_fprop_o_hat
    # ------------------------------------------------------------------
    def modiff_dynamic_int8_step1():
        absmax_buf.zero_()
        retire_count.zero_()
        return modiff_cutlass.step1_quantize_fprop(
            x,
            cache,
            residual_buf,
            absmax_buf,
            scale_buf,
            inv_scale_buf,
            retire_count,
            127.0,
            smooth_inv,
        )

    x8_mod_dynamic = modiff_cutlass.step1_quantize_fprop(
        x,
        cache,
        residual_buf,
        absmax_buf,
        scale_buf,
        inv_scale_buf,
        retire_count,
        127.0,
        smooth_inv,
    )

    def modiff_dynamic_int8_conv():
        modiff_cutlass.conv2d_int8_fprop_o_hat(
            x8_mod_dynamic,
            int8_weight,
            inv_scale_buf.view(1),
            int8_weight_scale.view(-1),
            o_hat,
            1, 1, 1, 1, 1, 1,
        )

    def modiff_dynamic_int4_step1():
        absmax_buf.zero_()
        retire_count.zero_()
        return modiff_cutlass.step1_quantize_pack_int4_fprop(
            x,
            cache,
            residual_buf,
            absmax_buf,
            scale_buf,
            inv_scale_buf,
            retire_count,
            7.0,
            smooth_inv,
        )

    x4_mod_dynamic = modiff_cutlass.step1_quantize_pack_int4_fprop(
        x,
        cache,
        residual_buf,
        absmax_buf,
        scale_buf,
        inv_scale_buf,
        retire_count,
        7.0,
        smooth_inv,
    )

    def modiff_dynamic_int4_conv():
        modiff_cutlass.conv2d_int4_fprop_o_hat(
            x4_mod_dynamic,
            int4_weight,
            inv_scale_buf.view(1),
            int4_weight_scale.view(-1),
            o_hat,
            1, 1, 1, 1, 1, 1,
        )

    modiff_dynamic8_step1 = benchmark_cuda(modiff_dynamic_int8_step1, warmup=warmup, iters=iters)
    modiff_dynamic8_conv = benchmark_cuda(modiff_dynamic_int8_conv, warmup=warmup, iters=iters)
    modiff_dynamic4_step1 = benchmark_cuda(modiff_dynamic_int4_step1, warmup=warmup, iters=iters)
    modiff_dynamic4_conv = benchmark_cuda(modiff_dynamic_int4_conv, warmup=warmup, iters=iters)

    results = {
        "shape": {"n": n, "c": c, "h": h, "w": w},
        "raw_conv_only": {
            "int8": raw_int8.to_dict(),
            "int4": raw_int4.to_dict(),
            "int4_over_int8_speedup": speedup(raw_int8.median_ms, raw_int4.median_ms),
        },
        "baseline_fused_static": {
            "int8": baseline8.to_dict(),
            "int4": baseline4.to_dict(),
            "int4_over_int8_speedup": speedup(baseline8.median_ms, baseline4.median_ms),
        },
        "modiff_fused_static": {
            "int8": {
                "step1": modiff_static8_step1.to_dict(),
                "conv": modiff_static8_conv.to_dict(),
                "total_median_ms": modiff_static8_step1.median_ms + modiff_static8_conv.median_ms,
            },
            "int4": {
                "step1": modiff_static4_step1.to_dict(),
                "conv": modiff_static4_conv.to_dict(),
                "total_median_ms": modiff_static4_step1.median_ms + modiff_static4_conv.median_ms,
            },
            "step1_speedup": speedup(modiff_static8_step1.median_ms, modiff_static4_step1.median_ms),
            "conv_speedup": speedup(modiff_static8_conv.median_ms, modiff_static4_conv.median_ms),
            "total_speedup": speedup(
                modiff_static8_step1.median_ms + modiff_static8_conv.median_ms,
                modiff_static4_step1.median_ms + modiff_static4_conv.median_ms,
            ),
        },
        "modiff_fused_dynamic": {
            "int8": {
                "step1": modiff_dynamic8_step1.to_dict(),
                "conv": modiff_dynamic8_conv.to_dict(),
                "total_median_ms": modiff_dynamic8_step1.median_ms + modiff_dynamic8_conv.median_ms,
            },
            "int4": {
                "step1": modiff_dynamic4_step1.to_dict(),
                "conv": modiff_dynamic4_conv.to_dict(),
                "total_median_ms": modiff_dynamic4_step1.median_ms + modiff_dynamic4_conv.median_ms,
            },
            "step1_speedup": speedup(modiff_dynamic8_step1.median_ms, modiff_dynamic4_step1.median_ms),
            "conv_speedup": speedup(modiff_dynamic8_conv.median_ms, modiff_dynamic4_conv.median_ms),
            "total_speedup": speedup(
                modiff_dynamic8_step1.median_ms + modiff_dynamic8_conv.median_ms,
                modiff_dynamic4_step1.median_ms + modiff_dynamic4_conv.median_ms,
            ),
        },
    }

    del conv, x, cache, cache_zero, o_hat, residual_buf, absmax_buf, scale_buf, inv_scale_buf, retire_count
    torch.cuda.empty_cache()
    return results


def render_markdown_report(all_results: Dict[str, object], output_path: str) -> None:
    gpu_name = all_results["metadata"]["gpu_name"]
    warmup = all_results["metadata"]["warmup"]
    iters = all_results["metadata"]["iters"]

    lines: List[str] = [
        "# A40 Layerwise INT8 vs INT4 Speedup Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**GPU:** {gpu_name}",
        f"**PyTorch:** {all_results['metadata']['torch_version']}",
        f"**CUDA (PyTorch):** {all_results['metadata']['torch_cuda_version']}",
        f"**Iterations:** {iters}",
        f"**Warmup:** {warmup}",
        "",
        "## What MoDiff is",
        "",
        "MoDiff is **error-compensated modulated quantization across diffusion timesteps**.",
        "Instead of quantizing the full activation $a_t$ at every step, it caches the next-step approximation and quantizes the residual:",
        "",
        "$$",
        "\\hat{a}_t = Q(a_t - \\hat{a}_{t+1}) + \\hat{a}_{t+1}, \\qquad",
        "\\hat{o}_t = A(Q(a_t - \\hat{a}_{t+1})) + \\hat{o}_{t+1}",
        "$$",
        "",
        "That makes quantization **more accurate**, especially at lower activation bitwidths. It does **not** automatically guarantee a 2x latency speedup, because the residual path, cache traffic, quantization, and dequant/accumulate work still cost time.",
        "",
        "## Summary table",
        "",
        "| Shape | Raw conv-only | Baseline fused | MoDiff fused (static) | MoDiff fused (dynamic) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for entry in all_results["results"]:
        shape = format_shape((entry["shape"]["n"], entry["shape"]["c"], entry["shape"]["h"], entry["shape"]["w"]))
        raw = entry["raw_conv_only"]["int4_over_int8_speedup"]
        base = entry["baseline_fused_static"]["int4_over_int8_speedup"]
        mod_static = entry["modiff_fused_static"]["total_speedup"]
        mod_dynamic = entry["modiff_fused_dynamic"]["total_speedup"]
        lines.append(f"| {shape} | {raw:.2f}x | {base:.2f}x | {mod_static:.2f}x | {mod_dynamic:.2f}x |")

    lines.extend([
        "",
        "## Detailed measurements",
        "",
    ])

    for entry in all_results["results"]:
        shape = format_shape((entry["shape"]["n"], entry["shape"]["c"], entry["shape"]["h"], entry["shape"]["w"]))
        lines.extend([
            f"### {shape}",
            "",
            "| Stage | INT8 (ms) | INT4 (ms) | INT4 / INT8 speedup |",
            "| --- | --- | --- | --- |",
            f"| Raw conv-only | {entry['raw_conv_only']['int8']['median_ms']:.3f} | {entry['raw_conv_only']['int4']['median_ms']:.3f} | {entry['raw_conv_only']['int4_over_int8_speedup']:.2f}x |",
            f"| Baseline fused static | {entry['baseline_fused_static']['int8']['median_ms']:.3f} | {entry['baseline_fused_static']['int4']['median_ms']:.3f} | {entry['baseline_fused_static']['int4_over_int8_speedup']:.2f}x |",
            f"| MoDiff fused static total | {entry['modiff_fused_static']['int8']['total_median_ms']:.3f} | {entry['modiff_fused_static']['int4']['total_median_ms']:.3f} | {entry['modiff_fused_static']['total_speedup']:.2f}x |",
            f"| MoDiff fused dynamic total | {entry['modiff_fused_dynamic']['int8']['total_median_ms']:.3f} | {entry['modiff_fused_dynamic']['int4']['total_median_ms']:.3f} | {entry['modiff_fused_dynamic']['total_speedup']:.2f}x |",
            "",
            "| MoDiff breakdown | INT8 (ms) | INT4 (ms) | INT4 / INT8 speedup |",
            "| --- | --- | --- | --- |",
            f"| Static step1 | {entry['modiff_fused_static']['int8']['step1']['median_ms']:.3f} | {entry['modiff_fused_static']['int4']['step1']['median_ms']:.3f} | {entry['modiff_fused_static']['step1_speedup']:.2f}x |",
            f"| Static conv | {entry['modiff_fused_static']['int8']['conv']['median_ms']:.3f} | {entry['modiff_fused_static']['int4']['conv']['median_ms']:.3f} | {entry['modiff_fused_static']['conv_speedup']:.2f}x |",
            f"| Dynamic step1 | {entry['modiff_fused_dynamic']['int8']['step1']['median_ms']:.3f} | {entry['modiff_fused_dynamic']['int4']['step1']['median_ms']:.3f} | {entry['modiff_fused_dynamic']['step1_speedup']:.2f}x |",
            f"| Dynamic conv | {entry['modiff_fused_dynamic']['int8']['conv']['median_ms']:.3f} | {entry['modiff_fused_dynamic']['int4']['conv']['median_ms']:.3f} | {entry['modiff_fused_dynamic']['conv_speedup']:.2f}x |",
            "",
        ])

    lines.extend([
        "## Interpretation",
        "",
        "- If INT4 were delivering a clean 2x benefit, the **raw conv-only** line would already be close to 2x.",
        "- In practice, the speedup typically shrinks from raw conv to fused baseline to fused MoDiff because the extra work is increasingly **memory-traffic heavy** and less sensitive to the nominal tensor-core throughput ratio.",
        "- The MoDiff `step1` path is especially important: it includes residual handling, quantization, and cache maintenance. That work is much less likely to scale as 2x when moving from INT8 to INT4.",
        "- So if the report shows only a modest end-to-end INT4 advantage, that is consistent with a pipeline where **raw compute is not the only bottleneck**.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layerwise A40 INT8 vs INT4 speedup benchmark")
    parser.add_argument("--iters", type=int, default=100, help="timed CUDA iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=20, help="warmup CUDA iterations per benchmark")
    parser.add_argument(
        "--output-json",
        type=str,
        default="analysis_int4_vs_int8/layerwise_speedup_a40.json",
        help="path to write JSON results",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="analysis_int4_vs_int8/LAYERWISE_A40_REPORT.md",
        help="path to write Markdown summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    shapes: List[Shape] = [
        (32, 192, 32, 32),
        (32, 384, 16, 16),
        (32, 768, 8, 8),
    ]

    all_results = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "warmup": args.warmup,
            "iters": args.iters,
            "shapes": [{"n": n, "c": c, "h": h, "w": w} for n, c, h, w in shapes],
        },
        "results": [],
    }

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Warmup={args.warmup}, iters={args.iters}")

    for shape in shapes:
        print(f"\nBenchmarking {format_shape(shape)}")
        entry = measure_shape(shape, warmup=args.warmup, iters=args.iters)
        all_results["results"].append(entry)

        print(
            "  Raw conv-only:          "
            f"INT8={entry['raw_conv_only']['int8']['median_ms']:.3f}ms, "
            f"INT4={entry['raw_conv_only']['int4']['median_ms']:.3f}ms, "
            f"speedup={entry['raw_conv_only']['int4_over_int8_speedup']:.2f}x"
        )
        print(
            "  Baseline fused static:  "
            f"INT8={entry['baseline_fused_static']['int8']['median_ms']:.3f}ms, "
            f"INT4={entry['baseline_fused_static']['int4']['median_ms']:.3f}ms, "
            f"speedup={entry['baseline_fused_static']['int4_over_int8_speedup']:.2f}x"
        )
        print(
            "  MoDiff fused static:    "
            f"INT8={entry['modiff_fused_static']['int8']['total_median_ms']:.3f}ms, "
            f"INT4={entry['modiff_fused_static']['int4']['total_median_ms']:.3f}ms, "
            f"speedup={entry['modiff_fused_static']['total_speedup']:.2f}x"
        )
        print(
            "  MoDiff fused dynamic:   "
            f"INT8={entry['modiff_fused_dynamic']['int8']['total_median_ms']:.3f}ms, "
            f"INT4={entry['modiff_fused_dynamic']['int4']['total_median_ms']:.3f}ms, "
            f"speedup={entry['modiff_fused_dynamic']['total_speedup']:.2f}x"
        )

    output_json = os.path.join(REPO_ROOT, args.output_json)
    output_md = os.path.join(REPO_ROOT, args.output_md)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_md), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    render_markdown_report(all_results, output_md)

    print(f"\nSaved JSON results to {output_json}")
    print(f"Saved Markdown report to {output_md}")


if __name__ == "__main__":
    main()