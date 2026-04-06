#!/usr/bin/env python3
"""
Enumerate every unique Conv2d shape used by the LSUN-Churches LDM UNet,
benchmark FP32 normal generation against INT8/INT4 quantized paths, and
generate a report + plots in a dedicated output folder.

Compared paths:
1. FP32 normal generation:
   standard PyTorch/CuDNN `nn.Conv2d` using the exact layer shape seen in the
   UNet forward pass.

2. INT8 / INT4 fused baseline (dynamic-scale):
    fused dynamic scale discovery -> quantize(+pack) without a_hat update ->
    CUTLASS conv -> dequantized write into a preallocated output buffer
    (no o_hat update)

3. INT8 / INT4 fused baseline (static-scale):
   quantize(+pack) -> CUTLASS conv -> dequant

4. INT8 / INT4 fused MoDiff (static-scale):
   step1_static_quantize_* -> conv2d_*_fprop_o_hat

5. INT8 / INT4 fused MoDiff (dynamic-scale):
   step1_quantize_* -> conv2d_*_fprop_o_hat

The script also reports a weighted aggregate over one UNet forward pass using
the actual per-shape invocation counts observed from the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, UNetModel

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


def benchmark_cuda(
    fn: Callable[[], None],
    warmup: int,
    iters: int,
    repeats: int,
    prepare: Callable[[], None] | None = None,
) -> Dict[str, float]:
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")
    if repeats <= 0:
        raise ValueError(f"repeats must be > 0, got {repeats}")

    torch.cuda.synchronize()

    for _ in range(warmup):
        if prepare is not None:
            prepare()
        fn()

    torch.cuda.synchronize()

    times_ms: List[float] = []
    for _ in range(repeats):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for idx in range(iters):
            if prepare is not None:
                prepare()
            start_events[idx].record()
            fn()
            end_events[idx].record()

        torch.cuda.synchronize()
        elapsed_ms = sum(start.elapsed_time(end) for start, end in zip(start_events, end_events))
        times_ms.append(float(elapsed_ms / iters))

    times_ms.sort()

    return {
        "median_ms": float(statistics.median(times_ms)),
        "mean_ms": float(sum(times_ms) / len(times_ms)),
        "min_ms": float(times_ms[0]),
        "max_ms": float(times_ms[-1]),
        "stddev_ms": float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "timing_mode": "synchronized_per_call_cuda_event_average",
        "reset_before_each_call": prepare is not None,
        "warmup": float(warmup),
        "iters_per_repeat": float(iters),
        "timed_repeats": float(repeats),
        "total_timed_calls": float(iters * repeats),
    }


def ms(stats: Dict[str, float]) -> float:
    return float(stats["mean_ms"])


def speedup(reference_ms: float, candidate_ms: float) -> float:
    return float(reference_ms) / float(candidate_ms) if candidate_ms > 0.0 else math.inf


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    last_dim = tensor.shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(f"Last dimension must be even for INT4 packing, got {last_dim}")
    reshaped = tensor.view(*tensor.shape[:-1], last_dim // 2, 2)
    low = reshaped[..., 0] & 0x0F
    high = (reshaped[..., 1] & 0x0F) << 4
    return (low | high).to(torch.int8)


def repo_quantized_exclusion_reasons(layer_name: str, spec: "ConvShapeSpec") -> List[str]:
    reasons: List[str] = []
    if spec.in_channels < 32:
        reasons.append("in_channels<32")
    if "skip" in layer_name:
        reasons.append("skip_connection")
    if layer_name.startswith("out."):
        reasons.append("final_out")
    if spec.kernel == (1, 1):
        reasons.append("pointwise_1x1")
    if spec.groups != 1:
        reasons.append("grouped_conv")
    return reasons


def conv_output_hw(h: int, w: int, kernel: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int]) -> Tuple[int, int]:
    h_out = (h + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    w_out = (w + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
    return h_out, w_out


@dataclass
class ConvShapeSpec:
    batch_size: int
    in_channels: int
    out_channels: int
    in_h: int
    in_w: int
    kernel: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int
    bias: bool
    count: int = 0
    layer_names: List[str] = field(default_factory=list)
    repo_supported_count: int = 0
    repo_unsupported_count: int = 0
    repo_unsupported_reasons: List[str] = field(default_factory=list)
    repo_supported_layer_names: List[str] = field(default_factory=list)
    repo_unsupported_layer_names: List[str] = field(default_factory=list)

    @property
    def out_h(self) -> int:
        return conv_output_hw(self.in_h, self.in_w, self.kernel, self.stride, self.padding, self.dilation)[0]

    @property
    def out_w(self) -> int:
        return conv_output_hw(self.in_h, self.in_w, self.kernel, self.stride, self.padding, self.dilation)[1]

    @property
    def key(self) -> Tuple[object, ...]:
        return (
            self.batch_size,
            self.in_channels,
            self.out_channels,
            self.in_h,
            self.in_w,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias,
        )

    @property
    def label(self) -> str:
        kh, kw = self.kernel
        return (
            f"{self.in_h}x{self.in_w} | {self.in_channels}->{self.out_channels} | "
            f"k{kh}x{kw}"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "input_h": self.in_h,
            "input_w": self.in_w,
            "output_h": self.out_h,
            "output_w": self.out_w,
            "kernel": list(self.kernel),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "groups": self.groups,
            "bias": self.bias,
            "count": self.count,
            "label": self.label,
            "layer_names": list(self.layer_names),
            "repo_supported_count": self.repo_supported_count,
            "repo_unsupported_count": self.repo_unsupported_count,
            "repo_quantized_supported": self.repo_supported_count > 0,
            "repo_unsupported_reasons": list(self.repo_unsupported_reasons),
            "repo_supported_layer_names": list(self.repo_supported_layer_names),
            "repo_unsupported_layer_names": list(self.repo_unsupported_layer_names),
        }


def enumerate_conv_shapes(config_path: str, batch_size: int) -> Tuple[List[ConvShapeSpec], Dict[str, object]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    unet_params = cfg["model"]["params"]["unet_config"]["params"]
    model = UNetModel(**unet_params).cuda().eval().to(memory_format=torch.channels_last)

    AttentionBlock.forward = lambda self, x: self._forward(x)

    records: List[Dict[str, object]] = []
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            def make_hook(layer_name: str, layer: nn.Conv2d):
                def hook(mod, inputs, output):
                    x = inputs[0]
                    records.append(
                        {
                            "name": layer_name,
                            "in_shape": tuple(int(v) for v in x.shape),
                            "out_shape": tuple(int(v) for v in output.shape),
                            "in_channels": int(layer.in_channels),
                            "out_channels": int(layer.out_channels),
                            "kernel": tuple(int(v) for v in layer.kernel_size),
                            "stride": tuple(int(v) for v in layer.stride),
                            "padding": tuple(int(v) for v in layer.padding),
                            "dilation": tuple(int(v) for v in layer.dilation),
                            "groups": int(layer.groups),
                            "bias": layer.bias is not None,
                        }
                    )
                return hook

            handles.append(module.register_forward_hook(make_hook(name, module)))

    x = torch.randn(
        batch_size,
        int(unet_params["in_channels"]),
        int(unet_params["image_size"]),
        int(unet_params["image_size"]),
        device="cuda",
        dtype=torch.float32,
    ).contiguous(memory_format=torch.channels_last)
    timesteps = torch.randint(0, 1000, (batch_size,), device="cuda")

    with torch.inference_mode():
        _ = model(x, timesteps=timesteps, context=None)

    for handle in handles:
        handle.remove()

    grouped: "OrderedDict[Tuple[object, ...], ConvShapeSpec]" = OrderedDict()
    for record in records:
        spec = ConvShapeSpec(
            batch_size=int(record["in_shape"][0]),
            in_channels=int(record["in_channels"]),
            out_channels=int(record["out_channels"]),
            in_h=int(record["in_shape"][2]),
            in_w=int(record["in_shape"][3]),
            kernel=tuple(record["kernel"]),
            stride=tuple(record["stride"]),
            padding=tuple(record["padding"]),
            dilation=tuple(record["dilation"]),
            groups=int(record["groups"]),
            bias=bool(record["bias"]),
        )
        if spec.key not in grouped:
            grouped[spec.key] = spec
        grouped_spec = grouped[spec.key]
        grouped_spec.count += 1
        if len(grouped_spec.layer_names) < 8:
            grouped_spec.layer_names.append(str(record["name"]))

        exclusion_reasons = repo_quantized_exclusion_reasons(str(record["name"]), spec)
        if exclusion_reasons:
            grouped_spec.repo_unsupported_count += 1
            for reason in exclusion_reasons:
                if reason not in grouped_spec.repo_unsupported_reasons:
                    grouped_spec.repo_unsupported_reasons.append(reason)
            if len(grouped_spec.repo_unsupported_layer_names) < 8:
                grouped_spec.repo_unsupported_layer_names.append(str(record["name"]))
        else:
            grouped_spec.repo_supported_count += 1
            if len(grouped_spec.repo_supported_layer_names) < 8:
                grouped_spec.repo_supported_layer_names.append(str(record["name"]))

    specs = sorted(
        grouped.values(),
        key=lambda s: (s.in_h, s.in_channels, s.out_channels, s.kernel, s.stride, s.padding),
    )

    inventory = {
        "config_path": config_path,
        "batch_size": batch_size,
        "num_conv_calls": len(records),
        "num_unique_conv_shapes": len(specs),
        "num_unique_3x3": sum(1 for s in specs if s.kernel == (3, 3)),
        "num_unique_1x1": sum(1 for s in specs if s.kernel == (1, 1)),
        "num_repo_supported_conv_calls": sum(s.repo_supported_count for s in specs),
        "num_repo_unsupported_conv_calls": sum(s.repo_unsupported_count for s in specs),
        "num_unique_repo_supported_shapes": sum(1 for s in specs if s.repo_supported_count > 0),
        "num_unique_repo_unsupported_shapes": sum(1 for s in specs if s.repo_unsupported_count > 0),
        "num_unique_repo_mixed_shapes": sum(1 for s in specs if s.repo_supported_count > 0 and s.repo_unsupported_count > 0),
    }

    del model, x, timesteps
    torch.cuda.empty_cache()
    return specs, inventory


def prepare_quantized_weights(conv: nn.Conv2d) -> Dict[str, object]:
    weight = conv.weight.detach().float()
    out_channels = int(weight.shape[0])

    weight_flat = weight.reshape(out_channels, -1)

    scale8 = torch.clamp(weight_flat.abs().max(dim=1).values / 127.0, min=1e-8)
    weight8 = (weight_flat / scale8.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    weight8 = weight8.reshape_as(weight).permute(0, 2, 3, 1).contiguous()

    scale4 = torch.clamp(weight_flat.abs().max(dim=1).values / 7.0, min=1e-8)
    weight4 = (weight_flat / scale4.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
    weight4 = weight4.reshape_as(weight).permute(0, 2, 3, 1).contiguous()
    weight4_packed = pack_int4(weight4).contiguous()

    bias = conv.bias.detach().float().view(1, -1, 1, 1).contiguous() if conv.bias is not None else None

    return {
        "int8_weight": weight8,
        "int8_weight_scale": scale8.view(1, out_channels, 1, 1).contiguous(),
        "int4_weight": weight4_packed,
        "int4_weight_scale": scale4.view(1, out_channels, 1, 1).contiguous(),
        "bias": bias,
        "empty_bias": torch.empty(0, device=weight.device),
    }


def format_optional(value: float | None, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}{suffix}"


def markdown_cell(text: object) -> str:
    return str(text).replace("|", "\\|")


def safe_median_ms(stats: Dict[str, float] | None) -> float | None:
    return ms(stats) if stats is not None else None


def benchmark_single_shape(spec: ConvShapeSpec, warmup: int, iters: int, repeats: int) -> Dict[str, object]:
    if spec.groups != 1:
        raise ValueError(f"CUTLASS benchmark currently expects groups=1, got {spec.groups} for {spec.label}")
    if spec.in_channels % 2 != 0:
        raise ValueError(f"INT4 path expects even input channels, got {spec.in_channels} for {spec.label}")

    conv = nn.Conv2d(
        spec.in_channels,
        spec.out_channels,
        kernel_size=spec.kernel,
        stride=spec.stride,
        padding=spec.padding,
        dilation=spec.dilation,
        groups=spec.groups,
        bias=spec.bias,
    ).cuda().eval().to(memory_format=torch.channels_last)

    x = torch.randn(
        spec.batch_size,
        spec.in_channels,
        spec.in_h,
        spec.in_w,
        device="cuda",
        dtype=torch.float32,
    ).contiguous(memory_format=torch.channels_last)

    fp32_stats = benchmark_cuda(lambda: conv(x), warmup=warmup, iters=iters, repeats=repeats)

    result: Dict[str, object] = {
        "shape": spec.to_dict(),
        "fp32_normal": fp32_stats,
        "quantized_status": {
            "state": "repo_excluded" if spec.repo_supported_count == 0 else "benchmarked",
            "repo_supported_count": spec.repo_supported_count,
            "repo_unsupported_count": spec.repo_unsupported_count,
            "repo_unsupported_reasons": list(spec.repo_unsupported_reasons),
        },
    }

    if spec.repo_supported_count == 0:
        del conv, x
        torch.cuda.empty_cache()
        return result

    out_h, out_w = spec.out_h, spec.out_w
    q = prepare_quantized_weights(conv)

    # Static activation scales for baseline-static / MoDiff-static paths.
    static_scale8 = torch.tensor(
        [127.0 / max(float(x.abs().amax().item()), 1e-6)],
        device="cuda",
        dtype=torch.float32,
    )
    static_inv8 = 1.0 / static_scale8

    static_scale4 = torch.tensor(
        [7.0 / max(float(x.abs().amax().item()), 1e-6)],
        device="cuda",
        dtype=torch.float32,
    )
    static_inv4 = 1.0 / static_scale4

    x8_raw = modiff_cutlass.scale_quantize_int8(x, static_scale8)
    x4_raw = modiff_cutlass.scale_quantize_and_pack(x, static_scale4)

    raw8_stats = benchmark_cuda(
        lambda: modiff_cutlass.conv2d_int8_fprop(
            x8_raw,
            q["int8_weight"],
            static_inv8.view(1),
            q["empty_bias"],
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        ),
        warmup=warmup,
        iters=iters,
        repeats=repeats,
    )

    raw4_stats = benchmark_cuda(
        lambda: modiff_cutlass.conv2d_int4_fprop(
            x4_raw,
            q["int4_weight"],
            static_inv4.view(1),
            q["empty_bias"],
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        ),
        warmup=warmup,
        iters=iters,
        repeats=repeats,
    )

    baseline_cache_d8 = torch.zeros_like(x)
    baseline_cache_d4 = torch.zeros_like(x)
    baseline_residual_d8 = torch.empty_like(x)
    baseline_residual_d4 = torch.empty_like(x)
    baseline_absmax_d8 = torch.zeros(1, device="cuda", dtype=torch.float32)
    baseline_absmax_d4 = torch.zeros(1, device="cuda", dtype=torch.float32)
    baseline_scale_d8 = torch.empty(1, device="cuda", dtype=torch.float32)
    baseline_scale_d4 = torch.empty(1, device="cuda", dtype=torch.float32)
    baseline_inv_d8 = torch.empty(1, device="cuda", dtype=torch.float32)
    baseline_inv_d4 = torch.empty(1, device="cuda", dtype=torch.float32)
    baseline_retire_d8 = torch.zeros(1, device="cuda", dtype=torch.int32)
    baseline_retire_d4 = torch.zeros(1, device="cuda", dtype=torch.int32)
    baseline_out_d8 = torch.empty(spec.batch_size, spec.out_channels, out_h, out_w, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    baseline_out_d4 = torch.empty(spec.batch_size, spec.out_channels, out_h, out_w, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    smooth_inv = torch.empty(0, device="cuda", dtype=torch.float32)

    def baseline_dynamic_int8():
        baseline_absmax_d8.zero_()
        baseline_retire_d8.zero_()
        xq = modiff_cutlass.step1_quantize_no_ahat_fprop(
            x,
            baseline_cache_d8,
            baseline_residual_d8,
            baseline_absmax_d8,
            baseline_scale_d8,
            baseline_inv_d8,
            baseline_retire_d8,
            127.0,
            smooth_inv,
        )
        out = modiff_cutlass.conv2d_int8_fprop_no_ohat_prealloc(
            xq,
            q["int8_weight"],
            baseline_inv_d8.view(1),
            q["int8_weight_scale"].view(-1),
            baseline_out_d8,
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        )
        return out

    def baseline_dynamic_int4():
        baseline_absmax_d4.zero_()
        baseline_retire_d4.zero_()
        xq = modiff_cutlass.step1_quantize_pack_int4_no_ahat_fprop(
            x,
            baseline_cache_d4,
            baseline_residual_d4,
            baseline_absmax_d4,
            baseline_scale_d4,
            baseline_inv_d4,
            baseline_retire_d4,
            7.0,
            smooth_inv,
        )
        out = modiff_cutlass.conv2d_int4_fprop_no_ohat_prealloc(
            xq,
            q["int4_weight"],
            baseline_inv_d4.view(1),
            q["int4_weight_scale"].view(-1),
            baseline_out_d4,
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        )
        return out

    def baseline_static_int8():
        xq = modiff_cutlass.scale_quantize_int8(x, static_scale8)
        out_raw = modiff_cutlass.conv2d_int8_fprop(
            xq,
            q["int8_weight"],
            static_inv8.view(1),
            q["empty_bias"],
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        )
        return out_raw * q["int8_weight_scale"]

    def baseline_static_int4():
        xq = modiff_cutlass.scale_quantize_and_pack(x, static_scale4)
        out_raw = modiff_cutlass.conv2d_int4_fprop(
            xq,
            q["int4_weight"],
            static_inv4.view(1),
            q["empty_bias"],
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        )
        return out_raw * q["int4_weight_scale"]

    baseline_dynamic8_stats = benchmark_cuda(baseline_dynamic_int8, warmup=warmup, iters=iters, repeats=repeats)
    baseline_dynamic4_stats = benchmark_cuda(baseline_dynamic_int4, warmup=warmup, iters=iters, repeats=repeats)
    baseline_static8_stats = benchmark_cuda(baseline_static_int8, warmup=warmup, iters=iters, repeats=repeats)
    baseline_static4_stats = benchmark_cuda(baseline_static_int4, warmup=warmup, iters=iters, repeats=repeats)

    # Separate cache/output buffers per mode to keep the benchmark paths isolated.
    # Reset them before every timed call so repeated benchmark iterations do not
    # benefit from cache/o_hat convergence across calls.
    cache_s8 = torch.zeros_like(x)
    cache_d8 = torch.zeros_like(x)
    cache_s4 = torch.zeros_like(x)
    cache_d4 = torch.zeros_like(x)

    o_hat_s8 = torch.zeros(spec.batch_size, spec.out_channels, out_h, out_w, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    o_hat_d8 = torch.zeros(spec.batch_size, spec.out_channels, out_h, out_w, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    o_hat_s4 = torch.zeros(spec.batch_size, spec.out_channels, out_h, out_w, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    o_hat_d4 = torch.zeros(spec.batch_size, spec.out_channels, out_h, out_w, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)

    residual_s8 = torch.empty_like(x)
    residual_d8 = torch.empty_like(x)
    residual_s4 = torch.empty_like(x)
    residual_d4 = torch.empty_like(x)

    absmax_s8 = torch.zeros(1, device="cuda", dtype=torch.float32)
    absmax_d8 = torch.zeros(1, device="cuda", dtype=torch.float32)
    absmax_s4 = torch.zeros(1, device="cuda", dtype=torch.float32)
    absmax_d4 = torch.zeros(1, device="cuda", dtype=torch.float32)

    scale_s8 = torch.empty(1, device="cuda", dtype=torch.float32)
    scale_d8 = torch.empty(1, device="cuda", dtype=torch.float32)
    scale_s4 = torch.empty(1, device="cuda", dtype=torch.float32)
    scale_d4 = torch.empty(1, device="cuda", dtype=torch.float32)

    inv_s8 = torch.empty(1, device="cuda", dtype=torch.float32)
    inv_d8 = torch.empty(1, device="cuda", dtype=torch.float32)
    inv_s4 = torch.empty(1, device="cuda", dtype=torch.float32)
    inv_d4 = torch.empty(1, device="cuda", dtype=torch.float32)

    retire_s8 = torch.zeros(1, device="cuda", dtype=torch.int32)
    retire_d8 = torch.zeros(1, device="cuda", dtype=torch.int32)
    retire_s4 = torch.zeros(1, device="cuda", dtype=torch.int32)
    retire_d4 = torch.zeros(1, device="cuda", dtype=torch.int32)

    def reset_cache_s8():
        cache_s8.zero_()

    def reset_cache_d8():
        cache_d8.zero_()

    def reset_cache_s4():
        cache_s4.zero_()

    def reset_cache_d4():
        cache_d4.zero_()

    def reset_ohat_s8():
        o_hat_s8.zero_()

    def reset_ohat_d8():
        o_hat_d8.zero_()

    def reset_ohat_s4():
        o_hat_s4.zero_()

    def reset_ohat_d4():
        o_hat_d4.zero_()

    def modiff_static_int8_step1():
        return modiff_cutlass.step1_static_quantize_fprop(x, cache_s8, static_scale8.view(1), smooth_inv)

    modiff_static8_step1_stats = benchmark_cuda(
        modiff_static_int8_step1,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_cache_s8,
    )
    reset_cache_s8()
    x8_static = modiff_cutlass.step1_static_quantize_fprop(x, cache_s8, static_scale8.view(1), smooth_inv)

    def modiff_static_int8_conv():
        return modiff_cutlass.conv2d_int8_fprop_o_hat(
            x8_static,
            q["int8_weight"],
            static_inv8.view(1),
            q["int8_weight_scale"].view(-1),
            o_hat_s8,
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        )

    modiff_static8_conv_stats = benchmark_cuda(
        modiff_static_int8_conv,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_ohat_s8,
    )

    def modiff_static_int4_step1():
        return modiff_cutlass.step1_static_quantize_pack_int4_fprop(x, cache_s4, static_scale4.view(1), smooth_inv)

    modiff_static4_step1_stats = benchmark_cuda(
        modiff_static_int4_step1,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_cache_s4,
    )
    reset_cache_s4()
    x4_static = modiff_cutlass.step1_static_quantize_pack_int4_fprop(x, cache_s4, static_scale4.view(1), smooth_inv)

    def modiff_static_int4_conv():
        return modiff_cutlass.conv2d_int4_fprop_o_hat(
            x4_static,
            q["int4_weight"],
            static_inv4.view(1),
            q["int4_weight_scale"].view(-1),
            o_hat_s4,
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        )

    modiff_static4_conv_stats = benchmark_cuda(
        modiff_static_int4_conv,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_ohat_s4,
    )

    def modiff_dynamic_int8_step1():
        absmax_d8.zero_()
        retire_d8.zero_()
        return modiff_cutlass.step1_quantize_fprop(
            x,
            cache_d8,
            residual_d8,
            absmax_d8,
            scale_d8,
            inv_d8,
            retire_d8,
            127.0,
            smooth_inv,
        )

    modiff_dynamic8_step1_stats = benchmark_cuda(
        modiff_dynamic_int8_step1,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_cache_d8,
    )
    reset_cache_d8()
    absmax_d8.zero_()
    retire_d8.zero_()
    x8_dynamic = modiff_cutlass.step1_quantize_fprop(
        x,
        cache_d8,
        residual_d8,
        absmax_d8,
        scale_d8,
        inv_d8,
        retire_d8,
        127.0,
        smooth_inv,
    )

    def modiff_dynamic_int8_conv():
        return modiff_cutlass.conv2d_int8_fprop_o_hat(
            x8_dynamic,
            q["int8_weight"],
            inv_d8.view(1),
            q["int8_weight_scale"].view(-1),
            o_hat_d8,
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        )

    modiff_dynamic8_conv_stats = benchmark_cuda(
        modiff_dynamic_int8_conv,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_ohat_d8,
    )

    def modiff_dynamic_int4_step1():
        absmax_d4.zero_()
        retire_d4.zero_()
        return modiff_cutlass.step1_quantize_pack_int4_fprop(
            x,
            cache_d4,
            residual_d4,
            absmax_d4,
            scale_d4,
            inv_d4,
            retire_d4,
            7.0,
            smooth_inv,
        )

    modiff_dynamic4_step1_stats = benchmark_cuda(
        modiff_dynamic_int4_step1,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_cache_d4,
    )
    reset_cache_d4()
    absmax_d4.zero_()
    retire_d4.zero_()
    x4_dynamic = modiff_cutlass.step1_quantize_pack_int4_fprop(
        x,
        cache_d4,
        residual_d4,
        absmax_d4,
        scale_d4,
        inv_d4,
        retire_d4,
        7.0,
        smooth_inv,
    )

    def modiff_dynamic_int4_conv():
        return modiff_cutlass.conv2d_int4_fprop_o_hat(
            x4_dynamic,
            q["int4_weight"],
            inv_d4.view(1),
            q["int4_weight_scale"].view(-1),
            o_hat_d4,
            spec.stride[0],
            spec.stride[1],
            spec.padding[0],
            spec.padding[1],
            spec.dilation[0],
            spec.dilation[1],
        )

    modiff_dynamic4_conv_stats = benchmark_cuda(
        modiff_dynamic_int4_conv,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_ohat_d4,
    )

    fp32_ms = ms(fp32_stats)
    baseline_dynamic8_ms = ms(baseline_dynamic8_stats)
    baseline_dynamic4_ms = ms(baseline_dynamic4_stats)
    baseline_static8_ms = ms(baseline_static8_stats)
    baseline_static4_ms = ms(baseline_static4_stats)
    modiff_static8_total = ms(modiff_static8_step1_stats) + ms(modiff_static8_conv_stats)
    modiff_static4_total = ms(modiff_static4_step1_stats) + ms(modiff_static4_conv_stats)
    modiff_dynamic8_total = ms(modiff_dynamic8_step1_stats) + ms(modiff_dynamic8_conv_stats)
    modiff_dynamic4_total = ms(modiff_dynamic4_step1_stats) + ms(modiff_dynamic4_conv_stats)

    result.update({
        "raw_conv_only": {
            "int8": raw8_stats,
            "int4": raw4_stats,
            "int4_over_int8_speedup": speedup(ms(raw8_stats), ms(raw4_stats)),
        },
        "baseline_fused_dynamic": {
            "int8": baseline_dynamic8_stats,
            "int4": baseline_dynamic4_stats,
            "int8_speedup_vs_fp32": speedup(fp32_ms, baseline_dynamic8_ms),
            "int4_speedup_vs_fp32": speedup(fp32_ms, baseline_dynamic4_ms),
            "int4_over_int8_speedup": speedup(baseline_dynamic8_ms, baseline_dynamic4_ms),
        },
        "baseline_fused_static": {
            "int8": baseline_static8_stats,
            "int4": baseline_static4_stats,
            "int8_speedup_vs_fp32": speedup(fp32_ms, baseline_static8_ms),
            "int4_speedup_vs_fp32": speedup(fp32_ms, baseline_static4_ms),
            "int4_over_int8_speedup": speedup(baseline_static8_ms, baseline_static4_ms),
        },
        "modiff_fused_static": {
            "int8": {
                "step1": modiff_static8_step1_stats,
                "conv": modiff_static8_conv_stats,
                "total_ms": modiff_static8_total,
                "total_mean_ms": modiff_static8_total,
            },
            "int4": {
                "step1": modiff_static4_step1_stats,
                "conv": modiff_static4_conv_stats,
                "total_ms": modiff_static4_total,
                "total_mean_ms": modiff_static4_total,
            },
            "int8_speedup_vs_fp32": speedup(fp32_ms, modiff_static8_total),
            "int4_speedup_vs_fp32": speedup(fp32_ms, modiff_static4_total),
            "int4_over_int8_speedup": speedup(modiff_static8_total, modiff_static4_total),
        },
        "modiff_fused_dynamic": {
            "int8": {
                "step1": modiff_dynamic8_step1_stats,
                "conv": modiff_dynamic8_conv_stats,
                "total_ms": modiff_dynamic8_total,
                "total_mean_ms": modiff_dynamic8_total,
            },
            "int4": {
                "step1": modiff_dynamic4_step1_stats,
                "conv": modiff_dynamic4_conv_stats,
                "total_ms": modiff_dynamic4_total,
                "total_mean_ms": modiff_dynamic4_total,
            },
            "int8_speedup_vs_fp32": speedup(fp32_ms, modiff_dynamic8_total),
            "int4_speedup_vs_fp32": speedup(fp32_ms, modiff_dynamic4_total),
            "int4_over_int8_speedup": speedup(modiff_dynamic8_total, modiff_dynamic4_total),
        },
    })

    del conv, x, cache_s8, cache_d8, cache_s4, cache_d4
    del baseline_cache_d8, baseline_cache_d4, baseline_residual_d8, baseline_residual_d4
    del baseline_out_d8, baseline_out_d4
    del o_hat_s8, o_hat_d8, o_hat_s4, o_hat_d4
    del residual_s8, residual_d8, residual_s4, residual_d4
    torch.cuda.empty_cache()
    return result


def compute_aggregates(results: List[Dict[str, object]]) -> Dict[str, object]:
    weighted_ms = {
        "fp32_all": 0.0,
        "fp32_repo_supported": 0.0,
        "fp32_repo_excluded": 0.0,
        "int8_baseline_dynamic": 0.0,
        "int4_baseline_dynamic": 0.0,
        "int8_baseline_static": 0.0,
        "int4_baseline_static": 0.0,
        "int8_modiff_static": 0.0,
        "int4_modiff_static": 0.0,
        "int8_modiff_dynamic": 0.0,
        "int4_modiff_dynamic": 0.0,
        "int8_raw_conv_only": 0.0,
        "int4_raw_conv_only": 0.0,
    }

    for entry in results:
        count = int(entry["shape"]["count"])
        repo_supported_count = int(entry["shape"]["repo_supported_count"])
        repo_unsupported_count = int(entry["shape"]["repo_unsupported_count"])
        fp32_ms = ms(entry["fp32_normal"])
        weighted_ms["fp32_all"] += count * fp32_ms
        weighted_ms["fp32_repo_supported"] += repo_supported_count * fp32_ms
        weighted_ms["fp32_repo_excluded"] += repo_unsupported_count * fp32_ms

        if entry["quantized_status"]["state"] != "benchmarked":
            continue

        weighted_ms["int8_baseline_dynamic"] += repo_supported_count * ms(entry["baseline_fused_dynamic"]["int8"])
        weighted_ms["int4_baseline_dynamic"] += repo_supported_count * ms(entry["baseline_fused_dynamic"]["int4"])
        weighted_ms["int8_baseline_static"] += repo_supported_count * ms(entry["baseline_fused_static"]["int8"])
        weighted_ms["int4_baseline_static"] += repo_supported_count * ms(entry["baseline_fused_static"]["int4"])
        weighted_ms["int8_modiff_static"] += repo_supported_count * float(entry["modiff_fused_static"]["int8"]["total_ms"])
        weighted_ms["int4_modiff_static"] += repo_supported_count * float(entry["modiff_fused_static"]["int4"]["total_ms"])
        weighted_ms["int8_modiff_dynamic"] += repo_supported_count * float(entry["modiff_fused_dynamic"]["int8"]["total_ms"])
        weighted_ms["int4_modiff_dynamic"] += repo_supported_count * float(entry["modiff_fused_dynamic"]["int4"]["total_ms"])
        weighted_ms["int8_raw_conv_only"] += repo_supported_count * ms(entry["raw_conv_only"]["int8"])
        weighted_ms["int4_raw_conv_only"] += repo_supported_count * ms(entry["raw_conv_only"]["int4"])

    fp32_supported_total = weighted_ms["fp32_repo_supported"]
    repo_supported_call_count = sum(int(entry["shape"]["repo_supported_count"]) for entry in results)
    repo_unsupported_call_count = sum(int(entry["shape"]["repo_unsupported_count"]) for entry in results)

    speedups_vs_fp32 = {
        mode: speedup(fp32_supported_total, value)
        for mode, value in weighted_ms.items()
        if mode not in {"fp32_all", "fp32_repo_supported", "fp32_repo_excluded"}
    }

    return {
        "weighted_ms_per_unet_forward": weighted_ms,
        "weighted_speedup_vs_repo_supported_fp32": speedups_vs_fp32,
        "weighted_raw_int4_over_int8_speedup": speedup(
            weighted_ms["int8_raw_conv_only"],
            weighted_ms["int4_raw_conv_only"],
        ),
        "repo_quantized_coverage": {
            "supported_call_count": repo_supported_call_count,
            "unsupported_call_count": repo_unsupported_call_count,
            "supported_fp32_share_pct": 100.0 * weighted_ms["fp32_repo_supported"] / weighted_ms["fp32_all"] if weighted_ms["fp32_all"] > 0.0 else 0.0,
            "unsupported_fp32_share_pct": 100.0 * weighted_ms["fp32_repo_excluded"] / weighted_ms["fp32_all"] if weighted_ms["fp32_all"] > 0.0 else 0.0,
        },
    }


def enrich_results_with_weighted_contribution(
    results: List[Dict[str, object]],
    fp32_total_ms: float,
    fp32_supported_total_ms: float,
) -> None:
    for entry in results:
        count = int(entry["shape"]["count"])
        repo_supported_count = int(entry["shape"]["repo_supported_count"])
        repo_unsupported_count = int(entry["shape"]["repo_unsupported_count"])
        weighted_fp32_all_ms = count * ms(entry["fp32_normal"])
        weighted_fp32_supported_ms = repo_supported_count * ms(entry["fp32_normal"])
        weighted_fp32_excluded_ms = repo_unsupported_count * ms(entry["fp32_normal"])
        entry["weighted_fp32_all_ms_per_unet_forward"] = weighted_fp32_all_ms
        entry["weighted_fp32_all_pct"] = 100.0 * weighted_fp32_all_ms / fp32_total_ms if fp32_total_ms > 0.0 else 0.0
        entry["weighted_fp32_supported_ms_per_unet_forward"] = weighted_fp32_supported_ms
        entry["weighted_fp32_supported_pct"] = 100.0 * weighted_fp32_supported_ms / fp32_supported_total_ms if fp32_supported_total_ms > 0.0 else 0.0
        entry["weighted_fp32_excluded_ms_per_unet_forward"] = weighted_fp32_excluded_ms


def write_json(output_dir: str, payload: Dict[str, object]) -> str:
    path = os.path.join(output_dir, "all_conv_sizes_fp32_int8_int4_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def per_shape_speedups(entry: Dict[str, object], mode: str) -> Tuple[float | None, float | None, float | None]:
    if entry["quantized_status"]["state"] != "benchmarked":
        return None, None, None

    fp32_value = ms(entry["fp32_normal"])

    if mode == "raw":
        raw = entry["raw_conv_only"]
        return (
            speedup(fp32_value, ms(raw["int8"])),
            speedup(fp32_value, ms(raw["int4"])),
            float(raw["int4_over_int8_speedup"]),
        )

    if mode in {"baseline", "baseline_dynamic"}:
        baseline = entry["baseline_fused_dynamic"]
        return (
            float(baseline["int8_speedup_vs_fp32"]),
            float(baseline["int4_speedup_vs_fp32"]),
            float(baseline["int4_over_int8_speedup"]),
        )

    if mode == "baseline_static":
        baseline = entry["baseline_fused_static"]
        return (
            float(baseline["int8_speedup_vs_fp32"]),
            float(baseline["int4_speedup_vs_fp32"]),
            float(baseline["int4_over_int8_speedup"]),
        )

    if mode == "modiff_static":
        modiff_static = entry["modiff_fused_static"]
        return (
            float(modiff_static["int8_speedup_vs_fp32"]),
            float(modiff_static["int4_speedup_vs_fp32"]),
            float(modiff_static["int4_over_int8_speedup"]),
        )

    if mode == "modiff_dynamic":
        modiff_dynamic = entry["modiff_fused_dynamic"]
        return (
            float(modiff_dynamic["int8_speedup_vs_fp32"]),
            float(modiff_dynamic["int4_speedup_vs_fp32"]),
            float(modiff_dynamic["int4_over_int8_speedup"]),
        )

    raise ValueError(f"Unknown mode: {mode}")


def report_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    results = sorted(payload["results"], key=lambda e: e["weighted_fp32_all_ms_per_unet_forward"], reverse=True)
    return [entry for entry in results if entry["quantized_status"]["state"] == "benchmarked"]


def write_report(output_dir: str, payload: Dict[str, object]) -> str:
    path = os.path.join(output_dir, "ALL_CONV_SIZES_FP32_INT8_INT4_REPORT.md")
    results = report_rows(payload)
    metadata = payload["metadata"]
    lines: List[str] = [
        "# Full per-shape benchmark table",
        (
            "- **Timing**: each number is the synchronized per-call average over "
            f"{int(metadata['timed_repeats'])} timed repeats × {int(metadata['iters'])} iterations, "
            f"after {int(metadata['warmup'])} warm-up iterations."
        ),
        "- **Timing mode**: synchronized per-call CUDA-event timing.",
        "- **State reset fairness**: MoDiff `a_hat` and `o_hat` buffers are reset to a fixed zero state before every timed call, outside the timed region, so repeated iterations cannot benefit from cache convergence/drift.",
        "- **Raw**: quantized input is precomputed outside the timed region; timed work is the CUTLASS conv wrapper only, including its output/workspace allocation.",
        "- **Baseline dynamic**: no a_hat/o_hat updates; timed work includes fused dynamic-scale discovery + quantization and the no-o_hat conv/dequant path into a preallocated output buffer. Standalone bias add is excluded.",
        "- **Baseline static**: no MoDiff; a precomputed activation scale is reused in the timed region. Standalone bias add is excluded.",
        "- **MoDiff static/dynamic**: timed work includes the MoDiff step1 path plus the fused convolution/update path, with each timed call starting from the same zeroed state.",
        "| Shape | Raw INT8 | Raw INT4 | Raw INT4/INT8 | Baseline dynamic INT8 | Baseline dynamic INT4 | Baseline dynamic INT4/INT8 | Baseline static INT8 | Baseline static INT4 | Baseline static INT4/INT8 | MoDiff static INT8 | MoDiff static INT4 | MoDiff static INT4/INT8 | MoDiff dynamic INT8 | MoDiff dynamic INT4 | MoDiff dynamic INT4/INT8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for entry in results:
        raw_int8, raw_int4, raw_ratio = per_shape_speedups(entry, "raw")
        baseline_dynamic_int8, baseline_dynamic_int4, baseline_dynamic_ratio = per_shape_speedups(entry, "baseline_dynamic")
        baseline_static_int8, baseline_static_int4, baseline_static_ratio = per_shape_speedups(entry, "baseline_static")
        static_int8, static_int4, static_ratio = per_shape_speedups(entry, "modiff_static")
        dynamic_int8, dynamic_int4, dynamic_ratio = per_shape_speedups(entry, "modiff_dynamic")

        lines.append(
            f"| {markdown_cell(entry['shape']['label'])} | "
            f"{format_optional(raw_int8, digits=2, suffix='x')} | {format_optional(raw_int4, digits=2, suffix='x')} | {format_optional(raw_ratio, digits=2, suffix='x')} | "
            f"{format_optional(baseline_dynamic_int8, digits=2, suffix='x')} | {format_optional(baseline_dynamic_int4, digits=2, suffix='x')} | {format_optional(baseline_dynamic_ratio, digits=2, suffix='x')} | "
            f"{format_optional(baseline_static_int8, digits=2, suffix='x')} | {format_optional(baseline_static_int4, digits=2, suffix='x')} | {format_optional(baseline_static_ratio, digits=2, suffix='x')} | "
            f"{format_optional(static_int8, digits=2, suffix='x')} | {format_optional(static_int4, digits=2, suffix='x')} | {format_optional(static_ratio, digits=2, suffix='x')} | "
            f"{format_optional(dynamic_int8, digits=2, suffix='x')} | {format_optional(dynamic_int4, digits=2, suffix='x')} | {format_optional(dynamic_ratio, digits=2, suffix='x')} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_csv(output_dir: str, payload: Dict[str, object]) -> str:
    path = os.path.join(output_dir, "ALL_CONV_SIZES_FP32_INT8_INT4_REPORT.csv")
    results = report_rows(payload)
    headers = [
        "Shape",
        "Raw INT8",
        "Raw INT4",
        "Raw INT4/INT8",
        "Baseline dynamic INT8",
        "Baseline dynamic INT4",
        "Baseline dynamic INT4/INT8",
        "Baseline static INT8",
        "Baseline static INT4",
        "Baseline static INT4/INT8",
        "MoDiff static INT8",
        "MoDiff static INT4",
        "MoDiff static INT4/INT8",
        "MoDiff dynamic INT8",
        "MoDiff dynamic INT4",
        "MoDiff dynamic INT4/INT8",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for entry in results:
            raw_int8, raw_int4, raw_ratio = per_shape_speedups(entry, "raw")
            baseline_dynamic_int8, baseline_dynamic_int4, baseline_dynamic_ratio = per_shape_speedups(entry, "baseline_dynamic")
            baseline_static_int8, baseline_static_int4, baseline_static_ratio = per_shape_speedups(entry, "baseline_static")
            static_int8, static_int4, static_ratio = per_shape_speedups(entry, "modiff_static")
            dynamic_int8, dynamic_int4, dynamic_ratio = per_shape_speedups(entry, "modiff_dynamic")
            writer.writerow([
                entry["shape"]["label"],
                format_optional(raw_int8, digits=2, suffix="x"),
                format_optional(raw_int4, digits=2, suffix="x"),
                format_optional(raw_ratio, digits=2, suffix="x"),
                format_optional(baseline_dynamic_int8, digits=2, suffix="x"),
                format_optional(baseline_dynamic_int4, digits=2, suffix="x"),
                format_optional(baseline_dynamic_ratio, digits=2, suffix="x"),
                format_optional(baseline_static_int8, digits=2, suffix="x"),
                format_optional(baseline_static_int4, digits=2, suffix="x"),
                format_optional(baseline_static_ratio, digits=2, suffix="x"),
                format_optional(static_int8, digits=2, suffix="x"),
                format_optional(static_int4, digits=2, suffix="x"),
                format_optional(static_ratio, digits=2, suffix="x"),
                format_optional(dynamic_int8, digits=2, suffix="x"),
                format_optional(dynamic_int4, digits=2, suffix="x"),
                format_optional(dynamic_ratio, digits=2, suffix="x"),
            ])
    return path


def plot_weighted_totals(output_dir: str, payload: Dict[str, object]) -> str:
    weighted = payload["aggregates"]["weighted_ms_per_unet_forward"]
    order = [
        "fp32_all",
        "fp32_repo_supported",
        "int8_baseline_dynamic",
        "int4_baseline_dynamic",
        "int8_baseline_static",
        "int4_baseline_static",
        "int8_modiff_static",
        "int4_modiff_static",
        "int8_modiff_dynamic",
        "int4_modiff_dynamic",
    ]

    labels = [label.replace("_", "\n") for label in order]
    values = [weighted[key] for key in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(
        range(len(order)),
        values,
        color=["#808080", "#A0A0A0", "#4F81BD", "#C0504D", "#6D9EEB", "#E06666", "#7FBA00", "#F28E2B", "#8064A2", "#E15759"],
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Weighted time per UNet forward (ms)")
    ax.set_title("All LDM Conv Shapes: weighted conv time")
    ax.grid(axis="y", alpha=0.25)

    fp32_supported = weighted["fp32_repo_supported"]
    for bar, key in zip(bars, order):
        value = weighted[key]
        if key == "fp32_all":
            text = "all"
        elif key == "fp32_repo_supported":
            text = "1.00x"
        else:
            text = f"{fp32_supported / value:.2f}x"
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, text, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "01_weighted_total_times.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_top_shapes(output_dir: str, payload: Dict[str, object], mode_key: str, title: str, filename: str) -> str:
    results = [entry for entry in payload["results"] if entry["quantized_status"]["state"] == "benchmarked"]
    results = sorted(results, key=lambda e: e["weighted_fp32_supported_ms_per_unet_forward"], reverse=True)[:15]
    labels = [f"{entry['shape']['label']} ×{entry['shape']['repo_supported_count']}" for entry in results][::-1]

    if mode_key == "baseline_dynamic":
        int8_values = [entry["baseline_fused_dynamic"]["int8_speedup_vs_fp32"] for entry in results][::-1]
        int4_values = [entry["baseline_fused_dynamic"]["int4_speedup_vs_fp32"] for entry in results][::-1]
    elif mode_key == "baseline_static":
        int8_values = [entry["baseline_fused_static"]["int8_speedup_vs_fp32"] for entry in results][::-1]
        int4_values = [entry["baseline_fused_static"]["int4_speedup_vs_fp32"] for entry in results][::-1]
    elif mode_key == "modiff_dynamic":
        int8_values = [entry["modiff_fused_dynamic"]["int8_speedup_vs_fp32"] for entry in results][::-1]
        int4_values = [entry["modiff_fused_dynamic"]["int4_speedup_vs_fp32"] for entry in results][::-1]
    else:
        int8_values = [entry["modiff_fused_static"]["int8_speedup_vs_fp32"] for entry in results][::-1]
        int4_values = [entry["modiff_fused_static"]["int4_speedup_vs_fp32"] for entry in results][::-1]

    y = list(range(len(labels)))
    height = 0.38

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh([v - height / 2 for v in y], int8_values, height=height, label="INT8", color="#4F81BD")
    ax.barh([v + height / 2 for v in y], int4_values, height=height, label="INT4", color="#C0504D")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Speedup vs FP32 normal generation")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_speedup_heatmap(output_dir: str, payload: Dict[str, object]) -> str:
    results = [entry for entry in payload["results"] if entry["quantized_status"]["state"] == "benchmarked"]
    results = sorted(results, key=lambda e: e["weighted_fp32_supported_ms_per_unet_forward"], reverse=True)

    shape_labels = [entry["shape"]["label"] for entry in results]
    mode_labels = ["baseline\ndynamic", "baseline\nstatic", "MoDiff\nstatic", "MoDiff\ndynamic"]

    int8_matrix = []
    int4_matrix = []
    for entry in results:
        int8_matrix.append([
            entry["baseline_fused_dynamic"]["int8_speedup_vs_fp32"],
            entry["baseline_fused_static"]["int8_speedup_vs_fp32"],
            entry["modiff_fused_static"]["int8_speedup_vs_fp32"],
            entry["modiff_fused_dynamic"]["int8_speedup_vs_fp32"],
        ])
        int4_matrix.append([
            entry["baseline_fused_dynamic"]["int4_speedup_vs_fp32"],
            entry["baseline_fused_static"]["int4_speedup_vs_fp32"],
            entry["modiff_fused_static"]["int4_speedup_vs_fp32"],
            entry["modiff_fused_dynamic"]["int4_speedup_vs_fp32"],
        ])

    int8_matrix = np.array(int8_matrix, dtype=np.float32)
    int4_matrix = np.array(int4_matrix, dtype=np.float32)

    vmin = float(min(int8_matrix.min(), int4_matrix.min()))
    vmax = float(max(int8_matrix.max(), int4_matrix.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, 0.28 * len(shape_labels))))
    for ax, matrix, title in zip(
        axes,
        [int8_matrix, int4_matrix],
        ["INT8 speedup vs FP32", "INT4 speedup vs FP32"],
    ):
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(mode_labels)))
        ax.set_xticklabels(mode_labels)
        ax.set_yticks(range(len(shape_labels)))
        ax.set_yticklabels(shape_labels, fontsize=7)
        ax.tick_params(axis="x", labelrotation=0)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                text_color = "white" if value < (vmin + vmax) / 2.0 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}x", ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("Speedup vs FP32")
    fig.suptitle("Per-shape speedup heatmap across modes", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, "05_speedup_heatmap.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_excluded_shapes(output_dir: str, payload: Dict[str, object]) -> str:
    results = [entry for entry in payload["results"] if entry["shape"]["repo_unsupported_count"] > 0]
    results = sorted(results, key=lambda e: e["weighted_fp32_excluded_ms_per_unet_forward"], reverse=True)[:15]

    labels = [f"{entry['shape']['label']} ×{entry['shape']['repo_unsupported_count']}" for entry in results][::-1]
    values = [entry["weighted_fp32_excluded_ms_per_unet_forward"] for entry in results][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(labels)), values, color="#808080")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Weighted FP32 time per UNet forward (ms)")
    ax.set_title("Repo-excluded conv shapes: FP32 contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "04_repo_excluded_shapes_fp32_contribution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all LDM Conv2d shapes against FP32 / INT8 / INT4")
    parser.add_argument(
        "--config",
        type=str,
        default="models/ldm/lsun_churches256/config.yaml",
        help="path to the LDM config used to instantiate the UNet",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size used for shape enumeration and benchmarking")
    parser.add_argument("--warmup", type=int, default=100, help="warmup iterations before each timed microbenchmark")
    parser.add_argument("--iters", type=int, default=1000, help="timed iterations per repeat")
    parser.add_argument("--timed-repeats", type=int, default=10, help="number of synchronized timed repeats used to average each microbenchmark")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_int4_vs_int8/ldm_all_conv_fp32_compare_a40",
        help="dedicated output folder for JSON, report, and plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    config_path = os.path.join(REPO_ROOT, args.config)
    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {config_path}")
    print(
        f"Batch size: {args.batch_size} | Warmup: {args.warmup} | "
        f"Iterations/repeat: {args.iters} | Timed repeats: {args.timed_repeats}"
    )

    specs, inventory = enumerate_conv_shapes(config_path, args.batch_size)
    print(f"Enumerated {inventory['num_unique_conv_shapes']} unique Conv2d shapes from {inventory['num_conv_calls']} calls.")

    results: List[Dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        if spec.repo_supported_count > 0:
            status = f"quantized_count={spec.repo_supported_count}"
        else:
            status = f"repo-excluded ({', '.join(spec.repo_unsupported_reasons)})"
        print(f"[{index:02d}/{len(specs)}] Benchmarking {spec.label} (count={spec.count}; {status})")
        results.append(benchmark_single_shape(spec, warmup=args.warmup, iters=args.iters, repeats=args.timed_repeats))

    aggregates = compute_aggregates(results)
    enrich_results_with_weighted_contribution(
        results,
        aggregates["weighted_ms_per_unet_forward"]["fp32_all"],
        aggregates["weighted_ms_per_unet_forward"]["fp32_repo_supported"],
    )
    results.sort(key=lambda e: e["weighted_fp32_all_ms_per_unet_forward"], reverse=True)

    payload = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "warmup": args.warmup,
            "iters": args.iters,
            "timed_repeats": args.timed_repeats,
            "timing_mode": "synchronized_per_call_cuda_event_average",
        },
        "inventory": inventory,
        "aggregates": aggregates,
        "results": results,
    }

    json_path = write_json(output_dir, payload)
    report_path = write_report(output_dir, payload)
    csv_path = write_csv(output_dir, payload)
    plot1 = plot_weighted_totals(output_dir, payload)
    plot2 = plot_top_shapes(
        output_dir,
        payload,
        mode_key="baseline_dynamic",
        title="Top repo-supported conv contributors: INT8/INT4 dynamic baseline speedup vs FP32",
        filename="02_top_shapes_baseline_dynamic_speedup_vs_fp32.png",
    )
    plot3 = plot_top_shapes(
        output_dir,
        payload,
        mode_key="baseline_static",
        title="Top repo-supported conv contributors: INT8/INT4 static baseline speedup vs FP32",
        filename="03_top_shapes_baseline_static_speedup_vs_fp32.png",
    )
    plot4 = plot_top_shapes(
        output_dir,
        payload,
        mode_key="modiff_dynamic",
        title="Top repo-supported conv contributors: INT8/INT4 dynamic MoDiff speedup vs FP32",
        filename="04_top_shapes_modiff_dynamic_speedup_vs_fp32.png",
    )
    plot5 = plot_speedup_heatmap(output_dir, payload)
    plot6 = plot_excluded_shapes(output_dir, payload)

    print("\nWeighted aggregate per UNet forward:")
    weighted = aggregates["weighted_ms_per_unet_forward"]
    speedups = aggregates["weighted_speedup_vs_repo_supported_fp32"]
    coverage = aggregates["repo_quantized_coverage"]
    print(f"  FP32 all convs:            {weighted['fp32_all']:.3f} ms")
    print(f"  FP32 repo-supported:       {weighted['fp32_repo_supported']:.3f} ms  ({coverage['supported_fp32_share_pct']:.1f}% of all FP32 conv time)")
    print(f"  FP32 repo-excluded:        {weighted['fp32_repo_excluded']:.3f} ms  ({coverage['unsupported_fp32_share_pct']:.1f}% of all FP32 conv time)")
    print(f"  INT8 baseline dynamic:     {weighted['int8_baseline_dynamic']:.3f} ms  ({speedups['int8_baseline_dynamic']:.2f}x vs supported FP32)")
    print(f"  INT4 baseline dynamic:     {weighted['int4_baseline_dynamic']:.3f} ms  ({speedups['int4_baseline_dynamic']:.2f}x vs supported FP32)")
    print(f"  INT8 baseline static:      {weighted['int8_baseline_static']:.3f} ms  ({speedups['int8_baseline_static']:.2f}x vs supported FP32)")
    print(f"  INT4 baseline static:      {weighted['int4_baseline_static']:.3f} ms  ({speedups['int4_baseline_static']:.2f}x vs supported FP32)")
    print(f"  INT8 MoDiff static:        {weighted['int8_modiff_static']:.3f} ms  ({speedups['int8_modiff_static']:.2f}x vs supported FP32)")
    print(f"  INT4 MoDiff static:        {weighted['int4_modiff_static']:.3f} ms  ({speedups['int4_modiff_static']:.2f}x vs supported FP32)")
    print(f"  INT8 MoDiff dynamic:       {weighted['int8_modiff_dynamic']:.3f} ms  ({speedups['int8_modiff_dynamic']:.2f}x vs supported FP32)")
    print(f"  INT4 MoDiff dynamic:       {weighted['int4_modiff_dynamic']:.3f} ms  ({speedups['int4_modiff_dynamic']:.2f}x vs supported FP32)")
    print(f"  Raw conv-only INT4/INT8:   {aggregates['weighted_raw_int4_over_int8_speedup']:.2f}x")

    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved Markdown report to {report_path}")
    print(f"Saved CSV report to {csv_path}")
    print(f"Saved plots to: {plot1}, {plot2}, {plot3}, {plot4}, {plot5}, {plot6}")


if __name__ == "__main__":
    main()