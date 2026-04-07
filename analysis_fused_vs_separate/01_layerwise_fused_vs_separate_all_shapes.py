#!/usr/bin/env python3
"""Layerwise fused-vs-separate MoDiff benchmark across all unique LDM conv shapes.

This script enumerates every unique ``Conv2d`` shape exercised by the
LSUN-Churches LDM UNet and benchmarks a single **modulated MoDiff hot-path
update** for both:

- the current fused CUTLASS implementation
- the separate-kernel MoDiff implementation

It reports INT8 and INT4 timings for:
- Step1 (residual / scale / quantize / a_hat update)
- Conv side (conv / dequant / o_hat update)
- Combined hot-path total

The benchmark uses warmup iterations plus timed iterations × repeats, and it
resets ``a_hat`` / ``o_hat`` state before every timed call so repeated runs do
not benefit from cache convergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, UNetModel
from integration.kernels.int4_optimized import pack_int4

try:
    import modiff_cutlass
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Failed to import modiff_cutlass. Build the extension first and ensure the "
        "Torch shared libraries are visible to the Python environment."
    ) from exc


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def ensure_working_conv_backend() -> str:
    """Return the active convolution backend, falling back if cuDNN is broken.

    On some CUDA 12.4 / PyTorch 2.6 environments we observed that CUDA itself
    is available and the custom CUTLASS extension loads correctly, but cuDNN
    fails at first use with `CUDNN_STATUS_NOT_INITIALIZED`. For this benchmark
    we can safely fall back to PyTorch's non-cuDNN CUDA conv path because the
    measurements of interest are the CUTLASS fused-vs-separate quantized paths.
    """
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
        return "cuda-no-cudnn"

    try:
        x = torch.randn(1, 8, 8, 8, device="cuda", dtype=torch.float32)
        conv = nn.Conv2d(8, 8, 3, padding=1).cuda().eval()
        with torch.inference_mode():
            _ = conv(x)
        del x, conv
        torch.cuda.synchronize()
        return "cudnn"
    except RuntimeError as exc:
        message = str(exc)
        if "CUDNN_STATUS_NOT_INITIALIZED" not in message:
            raise
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print("[warn] cuDNN initialization failed; falling back to non-cuDNN CUDA convolutions.")
        return "cuda-no-cudnn"


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
    def out_h(self) -> int:
        return (
            (self.in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel[0] - 1) - 1)
            // self.stride[0]
            + 1
        )

    @property
    def out_w(self) -> int:
        return (
            (self.in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel[1] - 1) - 1)
            // self.stride[1]
            + 1
        )

    @property
    def label(self) -> str:
        kh, kw = self.kernel
        sh, sw = self.stride
        ph, pw = self.padding
        return (
            f"{self.in_h}x{self.in_w} | {self.in_channels}->{self.out_channels} | "
            f"k{kh}x{kw} s{sh}x{sw} p{ph}x{pw}"
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
            "repo_unsupported_reasons": list(self.repo_unsupported_reasons),
            "repo_supported_layer_names": list(self.repo_supported_layer_names),
            "repo_unsupported_layer_names": list(self.repo_unsupported_layer_names),
        }


def benchmark_cuda(
    fn: Callable[[], None],
    *,
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

    per_repeat_ms: List[float] = []
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
        elapsed = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events))
        per_repeat_ms.append(float(elapsed / iters))

    per_repeat_ms.sort()
    return {
        "mean_ms": float(sum(per_repeat_ms) / len(per_repeat_ms)),
        "median_ms": float(statistics.median(per_repeat_ms)),
        "min_ms": float(per_repeat_ms[0]),
        "max_ms": float(per_repeat_ms[-1]),
        "stddev_ms": float(statistics.pstdev(per_repeat_ms)) if len(per_repeat_ms) > 1 else 0.0,
        "warmup": float(warmup),
        "iters_per_repeat": float(iters),
        "timed_repeats": float(repeats),
        "total_timed_calls": float(iters * repeats),
        "timing_mode": "synchronized_per_call_cuda_event_average",
        "reset_before_each_call": prepare is not None,
    }


def ms(stats: Dict[str, float]) -> float:
    return float(stats["mean_ms"])


def speedup(reference_ms: float, candidate_ms: float) -> float:
    return float(reference_ms) / float(candidate_ms) if candidate_ms > 0.0 else float("inf")


def repo_quantized_exclusion_reasons(layer_name: str, spec: ConvShapeSpec) -> List[str]:
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


def enumerate_conv_shapes(config_path: str, batch_size: int, max_shapes: int | None = None) -> List[ConvShapeSpec]:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    unet_params = cfg["model"]["params"]["unet_config"]["params"]
    model = UNetModel(**unet_params).cuda().eval().to(memory_format=torch.channels_last)
    AttentionBlock.forward = lambda self, x: self._forward(x)

    records: List[Dict[str, object]] = []
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            def make_hook(layer_name: str, layer: nn.Conv2d):
                def hook(mod, inputs, output):
                    x = inputs[0]
                    records.append(
                        {
                            "name": layer_name,
                            "in_shape": tuple(int(v) for v in x.shape),
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

            hooks.append(module.register_forward_hook(make_hook(name, module)))

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

    for hook in hooks:
        hook.remove()

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
        bucket = grouped.setdefault(spec.key, spec)
        bucket.count += 1
        if len(bucket.layer_names) < 8:
            bucket.layer_names.append(str(record["name"]))

        exclusion_reasons = repo_quantized_exclusion_reasons(str(record["name"]), spec)
        if exclusion_reasons:
            bucket.repo_unsupported_count += 1
            for reason in exclusion_reasons:
                if reason not in bucket.repo_unsupported_reasons:
                    bucket.repo_unsupported_reasons.append(reason)
            if len(bucket.repo_unsupported_layer_names) < 8:
                bucket.repo_unsupported_layer_names.append(str(record["name"]))
        else:
            bucket.repo_supported_count += 1
            if len(bucket.repo_supported_layer_names) < 8:
                bucket.repo_supported_layer_names.append(str(record["name"]))

    specs = sorted(
        grouped.values(),
        key=lambda item: (
            -item.count,
            -item.in_h,
            -item.in_channels,
            -item.out_channels,
            item.kernel,
            item.stride,
            item.padding,
        ),
    )

    if max_shapes is not None:
        specs = specs[:max_shapes]

    del model, x, timesteps
    torch.cuda.empty_cache()
    return specs


def prepare_quantized_weights(conv: nn.Conv2d) -> Dict[str, torch.Tensor]:
    weight = conv.weight.detach().float()
    out_channels = int(weight.shape[0])
    weight_flat = weight.reshape(out_channels, -1)

    scale8 = torch.clamp(weight_flat.abs().max(dim=1).values / 127.0, min=1e-8)
    weight8 = (weight_flat / scale8.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    weight8 = weight8.reshape_as(weight).permute(0, 2, 3, 1).contiguous()

    scale4 = torch.clamp(weight_flat.abs().max(dim=1).values / 7.0, min=1e-8)
    weight4 = (weight_flat / scale4.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
    weight4 = weight4.reshape_as(weight).permute(0, 2, 3, 1).contiguous()
    packed4 = pack_int4(weight4).contiguous()

    return {
        "int8_weight": weight8,
        "int8_weight_scale": scale8.view(1, out_channels, 1, 1).contiguous(),
        "int4_weight": packed4,
        "int4_weight_scale": scale4.view(1, out_channels, 1, 1).contiguous(),
        "empty_bias": torch.empty(0, device=weight.device),
    }


def make_static_scale(x: torch.Tensor, qmax: float) -> Tuple[float, torch.Tensor, torch.Tensor]:
    scale = float((qmax / torch.clamp(x.abs().amax(), min=1e-6)).item())
    scale_tensor = torch.tensor([scale], device=x.device, dtype=torch.float32)
    inv_tensor = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)
    return scale, scale_tensor, inv_tensor


def benchmark_int8_shape(spec: ConvShapeSpec, warmup: int, iters: int, repeats: int, quant_mode: str) -> Dict[str, object]:
    if spec.groups != 1:
        raise RuntimeError(f"INT8 CUTLASS path expects groups=1, got {spec.groups}")

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

    weights = prepare_quantized_weights(conv)
    smooth_inv = torch.empty(0, device="cuda", dtype=torch.float32)
    static_scale_float = None
    static_scale_tensor = None
    static_inv_tensor = None
    static_inv_float = None

    if quant_mode == "static":
        static_scale_float, static_scale_tensor, static_inv_tensor = make_static_scale(x, 127.0)
        static_inv_float = 1.0 / static_scale_float

    stride_h, stride_w = spec.stride
    pad_h, pad_w = spec.padding
    dil_h, dil_w = spec.dilation
    out_shape = (spec.batch_size, spec.out_channels, spec.out_h, spec.out_w)

    # Fused buffers
    fused_cache_step1 = torch.zeros_like(x)
    fused_cache_total = torch.zeros_like(x)
    fused_ohat_conv = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    fused_ohat_total = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    fused_residual_step1 = torch.empty_like(x)
    fused_residual_total = torch.empty_like(x)
    fused_absmax_step1 = torch.zeros(1, device="cuda", dtype=torch.float32)
    fused_absmax_total = torch.zeros(1, device="cuda", dtype=torch.float32)
    fused_scale_step1 = torch.empty(1, device="cuda", dtype=torch.float32)
    fused_scale_total = torch.empty(1, device="cuda", dtype=torch.float32)
    fused_inv_step1 = torch.empty(1, device="cuda", dtype=torch.float32)
    fused_inv_total = torch.empty(1, device="cuda", dtype=torch.float32)
    fused_retire_step1 = torch.zeros(1, device="cuda", dtype=torch.int32)
    fused_retire_total = torch.zeros(1, device="cuda", dtype=torch.int32)

    def reset_fused_step1():
        fused_cache_step1.zero_()
        fused_absmax_step1.zero_()
        fused_retire_step1.zero_()

    def fused_step1():
        if quant_mode == "static":
            return modiff_cutlass.step1_static_quantize_fprop(
                x,
                fused_cache_step1,
                static_scale_tensor.view(1),
                smooth_inv,
            )
        return modiff_cutlass.step1_quantize_fprop(
            x,
            fused_cache_step1,
            fused_residual_step1,
            fused_absmax_step1,
            fused_scale_step1,
            fused_inv_step1,
            fused_retire_step1,
            127.0,
            smooth_inv,
        )

    fused_step1_stats = benchmark_cuda(
        fused_step1,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_fused_step1,
    )

    reset_fused_step1()
    fused_x_int8 = fused_step1()
    fused_inv_for_conv = static_inv_tensor.clone() if quant_mode == "static" else fused_inv_step1.clone()

    def reset_fused_conv():
        fused_ohat_conv.zero_()

    def fused_conv():
        modiff_cutlass.conv2d_int8_fprop_o_hat(
            fused_x_int8,
            weights["int8_weight"],
            fused_inv_for_conv.view(1),
            weights["int8_weight_scale"].view(-1),
            fused_ohat_conv,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        )

    fused_conv_stats = benchmark_cuda(
        fused_conv,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_fused_conv,
    )

    def reset_fused_total():
        fused_cache_total.zero_()
        fused_ohat_total.zero_()
        fused_absmax_total.zero_()
        fused_retire_total.zero_()

    def fused_total():
        if quant_mode == "static":
            x_int8 = modiff_cutlass.step1_static_quantize_fprop(
                x,
                fused_cache_total,
                static_scale_tensor.view(1),
                smooth_inv,
            )
            fused_inv = static_inv_tensor.view(1)
        else:
            x_int8 = modiff_cutlass.step1_quantize_fprop(
                x,
                fused_cache_total,
                fused_residual_total,
                fused_absmax_total,
                fused_scale_total,
                fused_inv_total,
                fused_retire_total,
                127.0,
                smooth_inv,
            )
            fused_inv = fused_inv_total.view(1)
        modiff_cutlass.conv2d_int8_fprop_o_hat(
            x_int8,
            weights["int8_weight"],
            fused_inv,
            weights["int8_weight_scale"].view(-1),
            fused_ohat_total,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        )

    fused_total_stats = benchmark_cuda(
        fused_total,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_fused_total,
    )

    # Separate buffers
    sep_cache_step1 = torch.zeros_like(x)
    sep_cache_total = torch.zeros_like(x)
    sep_ohat_conv = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    sep_ohat_total = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)

    def reset_sep_step1():
        sep_cache_step1.zero_()

    def sep_step1():
        residual = x - sep_cache_step1
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        if quant_mode == "static":
            scale = static_scale_float
            inv_scale = static_inv_float
        else:
            scale = 127.0 / torch.clamp(residual.abs().amax(), min=1e-6)
            inv_scale = 1.0 / scale
        x_int8 = (residual * scale).round().clamp(-127, 127).to(torch.int8)
        sep_cache_step1.add_(x_int8.float() * inv_scale)
        return x_int8

    sep_step1_stats = benchmark_cuda(
        sep_step1,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_sep_step1,
    )

    sep_cache_step1.zero_()
    residual_for_conv = x - sep_cache_step1
    if not residual_for_conv.is_contiguous(memory_format=torch.channels_last):
        residual_for_conv = residual_for_conv.contiguous(memory_format=torch.channels_last)
    if quant_mode == "static":
        sep_scale_for_conv = static_scale_float
        sep_inv_for_conv = static_inv_tensor
    else:
        sep_scale_for_conv = 127.0 / torch.clamp(residual_for_conv.abs().amax(), min=1e-6)
        sep_inv_for_conv = 1.0 / sep_scale_for_conv
    sep_x_int8 = (residual_for_conv * sep_scale_for_conv).round().clamp(-127, 127).to(torch.int8)

    def reset_sep_conv():
        sep_ohat_conv.zero_()

    def sep_conv():
        out_raw = modiff_cutlass.conv2d_int8_fprop(
            sep_x_int8,
            weights["int8_weight"],
            sep_inv_for_conv.view(1),
            weights["empty_bias"],
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        )
        sep_ohat_conv.add_(out_raw * weights["int8_weight_scale"])

    sep_conv_stats = benchmark_cuda(
        sep_conv,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_sep_conv,
    )

    def reset_sep_total():
        sep_cache_total.zero_()
        sep_ohat_total.zero_()

    def sep_total():
        residual = x - sep_cache_total
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        if quant_mode == "static":
            scale = static_scale_float
            inv_scale = static_inv_float
            inv_scale_tensor = static_inv_tensor
        else:
            scale = 127.0 / torch.clamp(residual.abs().amax(), min=1e-6)
            inv_scale = 1.0 / scale
            inv_scale_tensor = inv_scale.view(1)
        x_int8 = (residual * scale).round().clamp(-127, 127).to(torch.int8)
        sep_cache_total.add_(x_int8.float() * inv_scale)
        out_raw = modiff_cutlass.conv2d_int8_fprop(
            x_int8,
            weights["int8_weight"],
            inv_scale_tensor,
            weights["empty_bias"],
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        )
        sep_ohat_total.add_(out_raw * weights["int8_weight_scale"])

    sep_total_stats = benchmark_cuda(
        sep_total,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_sep_total,
    )

    result = {
        "status": "ok",
        "fused_step1": fused_step1_stats,
        "separate_step1": sep_step1_stats,
        "step1_speedup": speedup(ms(sep_step1_stats), ms(fused_step1_stats)),
        "fused_conv": fused_conv_stats,
        "separate_conv": sep_conv_stats,
        "conv_speedup": speedup(ms(sep_conv_stats), ms(fused_conv_stats)),
        "fused_total": fused_total_stats,
        "separate_total": sep_total_stats,
        "total_speedup": speedup(ms(sep_total_stats), ms(fused_total_stats)),
    }

    del conv, x
    torch.cuda.empty_cache()
    return result


def benchmark_int4_shape(spec: ConvShapeSpec, warmup: int, iters: int, repeats: int, quant_mode: str) -> Dict[str, object]:
    if spec.groups != 1:
        raise RuntimeError(f"INT4 CUTLASS path expects groups=1, got {spec.groups}")
    if spec.in_channels % 2 != 0:
        raise RuntimeError(f"INT4 path expects even input channels, got {spec.in_channels}")

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

    weights = prepare_quantized_weights(conv)
    smooth_inv = torch.empty(0, device="cuda", dtype=torch.float32)
    static_scale_float = None
    static_scale_tensor = None
    static_inv_tensor = None
    static_inv_float = None

    if quant_mode == "static":
        static_scale_float, static_scale_tensor, static_inv_tensor = make_static_scale(x, 7.0)
        static_inv_float = 1.0 / static_scale_float

    stride_h, stride_w = spec.stride
    pad_h, pad_w = spec.padding
    dil_h, dil_w = spec.dilation
    out_shape = (spec.batch_size, spec.out_channels, spec.out_h, spec.out_w)

    fused_cache_step1 = torch.zeros_like(x)
    fused_cache_total = torch.zeros_like(x)
    fused_ohat_conv = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    fused_ohat_total = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    fused_residual_step1 = torch.empty_like(x)
    fused_residual_total = torch.empty_like(x)
    fused_absmax_step1 = torch.zeros(1, device="cuda", dtype=torch.float32)
    fused_absmax_total = torch.zeros(1, device="cuda", dtype=torch.float32)
    fused_scale_step1 = torch.empty(1, device="cuda", dtype=torch.float32)
    fused_scale_total = torch.empty(1, device="cuda", dtype=torch.float32)
    fused_inv_step1 = torch.empty(1, device="cuda", dtype=torch.float32)
    fused_inv_total = torch.empty(1, device="cuda", dtype=torch.float32)
    fused_retire_step1 = torch.zeros(1, device="cuda", dtype=torch.int32)
    fused_retire_total = torch.zeros(1, device="cuda", dtype=torch.int32)

    def reset_fused_step1():
        fused_cache_step1.zero_()
        fused_absmax_step1.zero_()
        fused_retire_step1.zero_()

    def fused_step1():
        if quant_mode == "static":
            return modiff_cutlass.step1_static_quantize_pack_int4_fprop(
                x,
                fused_cache_step1,
                static_scale_tensor.view(1),
                smooth_inv,
            )
        return modiff_cutlass.step1_quantize_pack_int4_fprop(
            x,
            fused_cache_step1,
            fused_residual_step1,
            fused_absmax_step1,
            fused_scale_step1,
            fused_inv_step1,
            fused_retire_step1,
            7.0,
            smooth_inv,
        )

    fused_step1_stats = benchmark_cuda(
        fused_step1,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_fused_step1,
    )

    reset_fused_step1()
    fused_x_packed = fused_step1()
    fused_inv_for_conv = static_inv_tensor.clone() if quant_mode == "static" else fused_inv_step1.clone()

    def reset_fused_conv():
        fused_ohat_conv.zero_()

    def fused_conv():
        modiff_cutlass.conv2d_int4_fprop_o_hat(
            fused_x_packed,
            weights["int4_weight"],
            fused_inv_for_conv.view(1),
            weights["int4_weight_scale"].view(-1),
            fused_ohat_conv,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        )

    fused_conv_stats = benchmark_cuda(
        fused_conv,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_fused_conv,
    )

    def reset_fused_total():
        fused_cache_total.zero_()
        fused_ohat_total.zero_()
        fused_absmax_total.zero_()
        fused_retire_total.zero_()

    def fused_total():
        if quant_mode == "static":
            x_packed = modiff_cutlass.step1_static_quantize_pack_int4_fprop(
                x,
                fused_cache_total,
                static_scale_tensor.view(1),
                smooth_inv,
            )
            fused_inv = static_inv_tensor.view(1)
        else:
            x_packed = modiff_cutlass.step1_quantize_pack_int4_fprop(
                x,
                fused_cache_total,
                fused_residual_total,
                fused_absmax_total,
                fused_scale_total,
                fused_inv_total,
                fused_retire_total,
                7.0,
                smooth_inv,
            )
            fused_inv = fused_inv_total.view(1)
        modiff_cutlass.conv2d_int4_fprop_o_hat(
            x_packed,
            weights["int4_weight"],
            fused_inv,
            weights["int4_weight_scale"].view(-1),
            fused_ohat_total,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        )

    fused_total_stats = benchmark_cuda(
        fused_total,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_fused_total,
    )

    sep_cache_step1 = torch.zeros_like(x)
    sep_cache_total = torch.zeros_like(x)
    sep_ohat_conv = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    sep_ohat_total = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)

    def reset_sep_step1():
        sep_cache_step1.zero_()

    def sep_step1():
        residual = x - sep_cache_step1
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        if quant_mode == "static":
            scale = static_scale_float
        else:
            scale = 7.0 / torch.clamp(residual.abs().amax(), min=1e-6)
        r_clamped = (residual * scale).round().clamp(-7, 7)
        x_packed = modiff_cutlass.quantize_and_pack(r_clamped)
        sep_cache_step1.add_(r_clamped / scale)
        return x_packed

    sep_step1_stats = benchmark_cuda(
        sep_step1,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_sep_step1,
    )

    sep_cache_step1.zero_()
    residual_for_conv = x - sep_cache_step1
    if not residual_for_conv.is_contiguous(memory_format=torch.channels_last):
        residual_for_conv = residual_for_conv.contiguous(memory_format=torch.channels_last)
    if quant_mode == "static":
        sep_scale_for_conv = static_scale_float
        sep_inv_for_conv = static_inv_tensor
    else:
        sep_scale_for_conv = 7.0 / torch.clamp(residual_for_conv.abs().amax(), min=1e-6)
        sep_inv_for_conv = 1.0 / sep_scale_for_conv
    sep_r_clamped = (residual_for_conv * sep_scale_for_conv).round().clamp(-7, 7)
    sep_x_packed = modiff_cutlass.quantize_and_pack(sep_r_clamped)

    def reset_sep_conv():
        sep_ohat_conv.zero_()

    def sep_conv():
        out_raw = modiff_cutlass.conv2d_int4_fprop(
            sep_x_packed,
            weights["int4_weight"],
            sep_inv_for_conv.view(1),
            weights["empty_bias"],
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        )
        sep_ohat_conv.add_(out_raw * weights["int4_weight_scale"])

    sep_conv_stats = benchmark_cuda(
        sep_conv,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_sep_conv,
    )

    def reset_sep_total():
        sep_cache_total.zero_()
        sep_ohat_total.zero_()

    def sep_total():
        residual = x - sep_cache_total
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        if quant_mode == "static":
            scale = static_scale_float
            inv_scale = static_inv_float
            inv_scale_tensor = static_inv_tensor
        else:
            scale = 7.0 / torch.clamp(residual.abs().amax(), min=1e-6)
            inv_scale = 1.0 / scale
            inv_scale_tensor = inv_scale.view(1)
        r_clamped = (residual * scale).round().clamp(-7, 7)
        x_packed = modiff_cutlass.quantize_and_pack(r_clamped)
        sep_cache_total.add_(r_clamped / scale)
        out_raw = modiff_cutlass.conv2d_int4_fprop(
            x_packed,
            weights["int4_weight"],
            inv_scale_tensor,
            weights["empty_bias"],
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
        )
        sep_ohat_total.add_(out_raw * weights["int4_weight_scale"])

    sep_total_stats = benchmark_cuda(
        sep_total,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        prepare=reset_sep_total,
    )

    result = {
        "status": "ok",
        "fused_step1": fused_step1_stats,
        "separate_step1": sep_step1_stats,
        "step1_speedup": speedup(ms(sep_step1_stats), ms(fused_step1_stats)),
        "fused_conv": fused_conv_stats,
        "separate_conv": sep_conv_stats,
        "conv_speedup": speedup(ms(sep_conv_stats), ms(fused_conv_stats)),
        "fused_total": fused_total_stats,
        "separate_total": sep_total_stats,
        "total_speedup": speedup(ms(sep_total_stats), ms(fused_total_stats)),
    }

    del conv, x
    torch.cuda.empty_cache()
    return result


def benchmark_single_shape(spec: ConvShapeSpec, warmup: int, iters: int, repeats: int, quant_mode: str) -> Dict[str, object]:
    result: Dict[str, object] = {
        "shape": spec.to_dict(),
        "int8": {"status": "not-run"},
        "int4": {"status": "not-run"},
    }

    if spec.repo_supported_count == 0:
        unsupported_message = ", ".join(spec.repo_unsupported_reasons) if spec.repo_unsupported_reasons else "unsupported"
        result["int8"] = {"status": "skipped_unsupported", "error": unsupported_message}
        result["int4"] = {"status": "skipped_unsupported", "error": unsupported_message}
        return result

    try:
        result["int8"] = benchmark_int8_shape(spec, warmup, iters, repeats, quant_mode)
    except Exception as exc:  # pragma: no cover - runtime path
        result["int8"] = {"status": "error", "error": str(exc)}

    try:
        result["int4"] = benchmark_int4_shape(spec, warmup, iters, repeats, quant_mode)
    except Exception as exc:  # pragma: no cover - runtime path
        result["int4"] = {"status": "error", "error": str(exc)}

    return result


def compute_aggregates(results: List[Dict[str, object]]) -> Dict[str, object]:
    aggregates = {
        "int8": {
            "weighted_fused_step1_ms": 0.0,
            "weighted_separate_step1_ms": 0.0,
            "weighted_fused_conv_ms": 0.0,
            "weighted_separate_conv_ms": 0.0,
            "weighted_fused_total_ms": 0.0,
            "weighted_separate_total_ms": 0.0,
            "benchmarked_calls": 0,
            "benchmarked_shapes": 0,
        },
        "int4": {
            "weighted_fused_step1_ms": 0.0,
            "weighted_separate_step1_ms": 0.0,
            "weighted_fused_conv_ms": 0.0,
            "weighted_separate_conv_ms": 0.0,
            "weighted_fused_total_ms": 0.0,
            "weighted_separate_total_ms": 0.0,
            "benchmarked_calls": 0,
            "benchmarked_shapes": 0,
        },
    }

    for entry in results:
        count = int(entry["shape"]["repo_supported_count"])
        for precision in ("int8", "int4"):
            payload = entry[precision]
            if payload.get("status") != "ok":
                continue
            bucket = aggregates[precision]
            bucket["weighted_fused_step1_ms"] += count * ms(payload["fused_step1"])
            bucket["weighted_separate_step1_ms"] += count * ms(payload["separate_step1"])
            bucket["weighted_fused_conv_ms"] += count * ms(payload["fused_conv"])
            bucket["weighted_separate_conv_ms"] += count * ms(payload["separate_conv"])
            bucket["weighted_fused_total_ms"] += count * ms(payload["fused_total"])
            bucket["weighted_separate_total_ms"] += count * ms(payload["separate_total"])
            bucket["benchmarked_calls"] += count
            bucket["benchmarked_shapes"] += 1

    for precision in ("int8", "int4"):
        bucket = aggregates[precision]
        bucket["weighted_step1_speedup"] = speedup(
            bucket["weighted_separate_step1_ms"],
            bucket["weighted_fused_step1_ms"],
        )
        bucket["weighted_conv_speedup"] = speedup(
            bucket["weighted_separate_conv_ms"],
            bucket["weighted_fused_conv_ms"],
        )
        bucket["weighted_total_speedup"] = speedup(
            bucket["weighted_separate_total_ms"],
            bucket["weighted_fused_total_ms"],
        )

    return aggregates


def sorted_rows(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def score(entry: Dict[str, object]) -> float:
        count = float(entry["shape"]["repo_supported_count"])
        best_total = 0.0
        for precision in ("int8", "int4"):
            payload = entry[precision]
            if payload.get("status") == "ok":
                best_total = max(best_total, ms(payload["separate_total"]))
        return count * best_total

    return sorted(results, key=score, reverse=True)


def write_json(output_dir: str, payload: Dict[str, object]) -> str:
    path = os.path.join(output_dir, "layerwise_fused_vs_separate_results.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def write_csv(output_dir: str, rows: List[Dict[str, object]]) -> str:
    path = os.path.join(output_dir, "LAYERWISE_FUSED_VS_SEPARATE.csv")
    headers = [
        "Shape",
        "Total Count",
        "Supported Count",
        "Unsupported Count",
        "INT8 fused step1 (ms)",
        "INT8 separate step1 (ms)",
        "INT8 step1 speedup",
        "INT8 fused conv (ms)",
        "INT8 separate conv (ms)",
        "INT8 conv speedup",
        "INT8 fused total (ms)",
        "INT8 separate total (ms)",
        "INT8 total speedup",
        "INT4 fused step1 (ms)",
        "INT4 separate step1 (ms)",
        "INT4 step1 speedup",
        "INT4 fused conv (ms)",
        "INT4 separate conv (ms)",
        "INT4 conv speedup",
        "INT4 fused total (ms)",
        "INT4 separate total (ms)",
        "INT4 total speedup",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for entry in rows:
            row = [
                entry["shape"]["label"],
                entry["shape"]["count"],
                entry["shape"]["repo_supported_count"],
                entry["shape"]["repo_unsupported_count"],
            ]
            for precision in ("int8", "int4"):
                payload = entry[precision]
                if payload.get("status") == "ok":
                    row.extend([
                        f"{ms(payload['fused_step1']):.6f}",
                        f"{ms(payload['separate_step1']):.6f}",
                        f"{payload['step1_speedup']:.4f}x",
                        f"{ms(payload['fused_conv']):.6f}",
                        f"{ms(payload['separate_conv']):.6f}",
                        f"{payload['conv_speedup']:.4f}x",
                        f"{ms(payload['fused_total']):.6f}",
                        f"{ms(payload['separate_total']):.6f}",
                        f"{payload['total_speedup']:.4f}x",
                    ])
                else:
                    row.extend([payload.get("status", "error")] * 9)
            writer.writerow(row)
    return path


def write_report(output_dir: str, payload: Dict[str, object]) -> str:
    report_path = os.path.join(output_dir, "LAYERWISE_FUSED_VS_SEPARATE_REPORT.md")
    rows = sorted_rows(payload["results"])
    aggregates = payload["aggregates"]
    metadata = payload["metadata"]

    lines: List[str] = [
        f"# Layerwise fused-vs-separate MoDiff benchmark ({metadata['quant_mode']} quantization)",
        "",
        f"**Date**: {metadata['generated_at']}",
        f"**GPU**: {metadata['gpu_name']}",
        f"**Config**: `{metadata['config_path']}`",
        f"**Batch Size**: {metadata['batch_size']}",
        f"**Quant Mode**: {metadata['quant_mode']}",
        "",
        "This benchmark isolates one **modulated MoDiff update** per unique Conv2d shape observed in the LSUN-Churches LDM UNet.",
        "",
        "Timing notes:",
        f"- Each value is the synchronized per-call average over {int(metadata['timed_repeats'])} timed repeats × {int(metadata['iters'])} iterations, after {int(metadata['warmup'])} warm-up iterations.",
        "- `a_hat` and `o_hat` buffers are reset to a fixed zero state before every timed call, outside the timed region.",
        "- The layerwise benchmark isolates the MoDiff hot path where fusion matters most: residual update + quantization + conv-side dequant/accumulate.",
        "- All unique Conv2d shapes are enumerated, but only shapes that match the repository's quantized-conversion rules are benchmarked; excluded shapes are still reported separately.",
        "- First-step warmup behavior is intentionally left to the whole-model benchmark.",
        f"- Activation quantization mode: **{metadata['quant_mode']}** ({metadata['activation_scale_policy']}).",
        "",
        "## Weighted aggregate over one UNet forward",
        "",
        "| Precision | Fused Step1 (ms) | Separate Step1 (ms) | Step1 speedup | Fused Conv (ms) | Separate Conv (ms) | Conv speedup | Fused Total (ms) | Separate Total (ms) | Fusion speedup | Benchmarked calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for precision in ("int8", "int4"):
        bucket = aggregates[precision]
        lines.append(
            f"| {precision.upper()} | {bucket['weighted_fused_step1_ms']:.3f} | {bucket['weighted_separate_step1_ms']:.3f} | {bucket['weighted_step1_speedup']:.2f}x | "
            f"{bucket['weighted_fused_conv_ms']:.3f} | {bucket['weighted_separate_conv_ms']:.3f} | {bucket['weighted_conv_speedup']:.2f}x | "
            f"{bucket['weighted_fused_total_ms']:.3f} | {bucket['weighted_separate_total_ms']:.3f} | {bucket['weighted_total_speedup']:.2f}x | {bucket['benchmarked_calls']} |"
        )

    for precision in ("int8", "int4"):
        lines.extend([
            "",
            f"## {precision.upper()} per-shape results",
            "",
            "| Shape | Total Count | Supported Count | Unsupported Count | Fused Step1 (ms) | Separate Step1 (ms) | Step1 speedup | Fused Conv (ms) | Separate Conv (ms) | Conv speedup | Fused Total (ms) | Separate Total (ms) | Fusion speedup |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for entry in rows:
            result = entry[precision]
            escaped_label = entry["shape"]["label"].replace("|", "\\|")
            if result.get("status") == "ok":
                lines.append(
                    f"| {escaped_label} | {entry['shape']['count']} | {entry['shape']['repo_supported_count']} | {entry['shape']['repo_unsupported_count']} | "
                    f"{ms(result['fused_step1']):.4f} | {ms(result['separate_step1']):.4f} | {result['step1_speedup']:.2f}x | "
                    f"{ms(result['fused_conv']):.4f} | {ms(result['separate_conv']):.4f} | {result['conv_speedup']:.2f}x | "
                    f"{ms(result['fused_total']):.4f} | {ms(result['separate_total']):.4f} | {result['total_speedup']:.2f}x |"
                )
            else:
                reason = result.get('error', result.get('status', 'error'))
                lines.append(
                    f"| {escaped_label} | {entry['shape']['count']} | {entry['shape']['repo_supported_count']} | {entry['shape']['repo_unsupported_count']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | {reason} |"
                )

    failed_rows = [
        entry for entry in rows
        if entry["int8"].get("status") != "ok" or entry["int4"].get("status") != "ok"
    ]
    if failed_rows:
        lines.extend([
            "",
            "## Shapes with runtime issues or exclusions",
            "",
            "| Shape | INT8 status | INT4 status |",
            "| --- | --- | --- |",
        ])
        for entry in failed_rows:
            int8_status = entry["int8"].get("error", entry["int8"].get("status", "ok"))
            int4_status = entry["int4"].get("error", entry["int4"].get("status", "ok"))
            escaped_label = entry["shape"]["label"].replace("|", "\\|")
            lines.append(
                f"| {escaped_label} | {int8_status} | {int4_status} |"
            )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layerwise fused-vs-separate MoDiff benchmark across all LDM conv shapes")
    parser.add_argument(
        "--config",
        type=str,
        default="models/ldm/lsun_churches256/config.yaml",
        help="UNet config used for conv-shape enumeration",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size used for shape enumeration and synthetic tensors")
    parser.add_argument("--warmup", type=int, default=100, help="warmup iterations before timed runs")
    parser.add_argument("--iters", type=int, default=1000, help="timed iterations per repeat")
    parser.add_argument("--timed-repeats", type=int, default=10, help="number of timed repeats")
    parser.add_argument("--max-shapes", type=int, default=None, help="optional cap for quick smoke tests")
    parser.add_argument("--seed", type=int, default=20260407, help="random seed for synthetic tensors")
    parser.add_argument(
        "--quant-mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="activation quantization mode: dynamic recomputes per-call scales, static calibrates one fixed scale per shape",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_fused_vs_separate/layerwise_results",
        help="directory for JSON/CSV/Markdown outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config_path = os.path.join(REPO_ROOT, args.config)
    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {config_path}")
    backend = ensure_working_conv_backend()
    print(f"Conv backend: {backend}")
    print(
        f"Batch size: {args.batch_size} | Warmup: {args.warmup} | "
        f"Iterations/repeat: {args.iters} | Timed repeats: {args.timed_repeats} | Quant mode: {args.quant_mode}"
    )

    specs = enumerate_conv_shapes(config_path, args.batch_size, args.max_shapes)
    print(f"Enumerated {len(specs)} unique Conv2d shapes.")

    results: List[Dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        if spec.repo_supported_count > 0:
            status = f"supported={spec.repo_supported_count}"
        else:
            status = f"excluded ({', '.join(spec.repo_unsupported_reasons)})"
        print(f"[{index:02d}/{len(specs)}] Benchmarking {spec.label} (count={spec.count}; {status})")
        results.append(benchmark_single_shape(spec, args.warmup, args.iters, args.timed_repeats, args.quant_mode))

    payload = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "conv_backend": backend,
            "config_path": config_path,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "timed_repeats": args.timed_repeats,
            "quant_mode": args.quant_mode,
            "activation_scale_policy": (
                "per-shape static scale calibrated once from the synthetic activation tensor"
                if args.quant_mode == "static"
                else "per-call dynamic scale recomputed from the current activation tensor"
            ),
        },
        "results": results,
        "aggregates": compute_aggregates(results),
    }

    json_path = write_json(output_dir, payload)
    csv_path = write_csv(output_dir, sorted_rows(results))
    report_path = write_report(output_dir, payload)

    print("\nLayerwise weighted totals:")
    for precision in ("int8", "int4"):
        bucket = payload["aggregates"][precision]
        print(
            f"  {precision.upper()}: fused={bucket['weighted_fused_total_ms']:.3f} ms | "
            f"separate={bucket['weighted_separate_total_ms']:.3f} ms | "
            f"speedup={bucket['weighted_total_speedup']:.2f}x"
        )

    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved CSV report to {csv_path}")
    print(f"Saved Markdown report to {report_path}")


if __name__ == "__main__":
    main()
