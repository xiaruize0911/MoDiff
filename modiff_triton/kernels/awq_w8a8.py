"""
Optional AWQ W8A8 GEMM wrapper.

This module adapts MoDiff-style tensors to the llm-awq CUDA extension:
  - activations are FP16/FP32 [M, K] and quantized with AWQ invoke_quant
  - weights are INT8 [K, N] in MoDiff layout and transposed to AWQ [N, K]
  - scales are FP16 per token/per output channel
  - output is FP16 [M, N]
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Optional

import torch


def _add_common_awq_paths() -> None:
    for path in (
        os.environ.get("AWQ_KERNEL_PATH"),
        "/workspace/llm-awq/awq/kernels",
        "/workspace/llm-awq",
    ):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


@lru_cache(maxsize=1)
def get_awq_engine():
    _add_common_awq_paths()
    try:
        import awq_inference_engine  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "awq_inference_engine is not available. Build it with "
            "`cd /workspace/llm-awq/awq/kernels && python setup.py build_ext --inplace`, "
            "or set AWQ_KERNEL_PATH to the directory containing the extension."
        ) from exc
    return awq_inference_engine


def is_awq_available() -> bool:
    try:
        get_awq_engine()
        return True
    except ImportError:
        return False


def _as_awq_weight(weight_int8: torch.Tensor) -> torch.Tensor:
    """Return AWQ's [N, K] contiguous layout from MoDiff's [K, N] layout."""
    if weight_int8.dim() != 2:
        raise ValueError("weight_int8 must be 2D [K, N]")
    return weight_int8.t().contiguous()


def _as_awq_scales(scale_w: torch.Tensor, out_features: int) -> torch.Tensor:
    if scale_w.numel() == 1:
        scale_w = scale_w.expand(out_features)
    if scale_w.numel() != out_features:
        raise ValueError(f"scale_w must be scalar or have {out_features} elements")
    return scale_w.contiguous().to(torch.float16)


def quantize_awq_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize x with AWQ's per-token FP16->INT8 kernel.

    Returns:
        x_int8: [M, K] int8
        scale_a: [M] fp16, dequant scale written by AWQ
    """
    if not x.is_cuda:
        raise ValueError("AWQ kernels require CUDA tensors")
    if x.dim() != 2:
        raise ValueError("x must be 2D [M, K]")
    engine = get_awq_engine()
    x_contig = x.contiguous()
    x_int8 = torch.empty_like(x_contig, dtype=torch.int8)
    scale_a = torch.empty((x_contig.shape[0],), device=x.device, dtype=torch.float16)
    engine.invoke_quant(x_int8, x_contig, scale_a)
    return x_int8, scale_a


def awq_gemm_w8a8(
    x_int8: torch.Tensor,
    weight_int8: torch.Tensor,
    scale_w: torch.Tensor,
    scale_a: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weight_is_awq_layout: bool = False,
    allow_unsafe_small_m: bool = False,
) -> torch.Tensor:
    """
    Run AWQ W8A8 GEMM.

    Args:
        x_int8: [M, K] int8 activations.
        weight_int8: [K, N] MoDiff layout by default, or [N, K] when
            weight_is_awq_layout=True.
        scale_w: scalar or [N] dequant scales.
        scale_a: [M] dequant scales from AWQ quantization.
        bias: optional [N] bias.
    """
    if x_int8.dtype != torch.int8 or weight_int8.dtype != torch.int8:
        raise TypeError("x_int8 and weight_int8 must be torch.int8")
    if x_int8.dim() != 2 or weight_int8.dim() != 2:
        raise ValueError("x_int8 and weight_int8 must be 2D")

    weight_awq = weight_int8.contiguous() if weight_is_awq_layout else _as_awq_weight(weight_int8)
    m, k = x_int8.shape
    n, k_weight = weight_awq.shape
    if k != k_weight:
        raise ValueError(f"shape mismatch: x K={k}, weight K={k_weight}")
    if n % 2 != 0:
        raise ValueError("AWQ W8A8 GEMM requires an even output feature dimension")

    # The upstream AWQ kernel's M <= 128 branch produced incorrect results on
    # the A40/CUDA 12.4 test host. Keep the integrated baseline correct by
    # falling back unless a benchmark explicitly asks for the raw kernel.
    if m <= 128 and not allow_unsafe_small_m:
        weight_modiff = weight_awq.t().contiguous()
        if hasattr(torch, "_int_mm") and m > 16:
            out_i32 = torch._int_mm(x_int8.contiguous(), weight_modiff)
        else:
            out_i32 = x_int8.float().matmul(weight_modiff.float()).to(torch.int32)
        w_scales = _as_awq_scales(scale_w.to(x_int8.device), n).float()
        a_scales = scale_a.contiguous().to(torch.float32)
        out = out_i32.float() * a_scales[:, None] * w_scales[None, :]
        if bias is not None:
            out = out + bias.to(out.device, dtype=torch.float32)[None, :]
        return out.to(torch.float16)

    engine = get_awq_engine()
    out = torch.empty((m, n), device=x_int8.device, dtype=torch.float16)
    w_scales = _as_awq_scales(scale_w.to(x_int8.device), n)
    a_scales = scale_a.contiguous().to(torch.float16)
    if a_scales.numel() != m:
        raise ValueError(f"scale_a must have {m} elements")

    if bias is None:
        engine.w8a8_gemm_forward_cuda(x_int8.contiguous(), weight_awq, w_scales, a_scales, out)
    else:
        engine.w8a8_gemm_fuse_bias_forward_cuda(
            x_int8.contiguous(),
            weight_awq,
            w_scales,
            a_scales,
            out,
            bias.contiguous().to(torch.float16),
        )
    return out


def awq_fused_quant_gemm_w8a8(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale_w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weight_is_awq_layout: bool = False,
    allow_unsafe_small_m: bool = False,
) -> torch.Tensor:
    """Quantize activations with AWQ and run AWQ W8A8 GEMM."""
    x_int8, scale_a = quantize_awq_per_token(x)
    return awq_gemm_w8a8(
        x_int8,
        weight_int8,
        scale_w,
        scale_a,
        bias=bias,
        weight_is_awq_layout=weight_is_awq_layout,
        allow_unsafe_small_m=allow_unsafe_small_m,
    )


def awq_fused_quant_gemm_w8a8_prealloc(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale_w: torch.Tensor,
    x_int8: torch.Tensor,
    scale_a: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weight_is_awq_layout: bool = False,
) -> torch.Tensor:
    """
    Quantize activations and run AWQ W8A8 GEMM into caller-owned buffers.

    This is intended for hot Conv1x1 paths where allocating x_int8, per-token
    scales, and output on every layer call shows up in whole-pipeline timing.
    """
    if not x.is_cuda:
        raise ValueError("AWQ kernels require CUDA tensors")
    if x.dim() != 2:
        raise ValueError("x must be 2D [M, K]")
    if x_int8.shape != x.shape or x_int8.dtype != torch.int8:
        raise ValueError("x_int8 must be int8 with the same shape as x")

    weight_awq = weight_int8.contiguous() if weight_is_awq_layout else _as_awq_weight(weight_int8)
    m, k = x.shape
    n, k_weight = weight_awq.shape
    if k != k_weight:
        raise ValueError(f"shape mismatch: x K={k}, weight K={k_weight}")
    if scale_a.shape != (m,) or scale_a.dtype != torch.float16:
        raise ValueError("scale_a must be fp16 with shape [M]")
    if out.shape != (m, n) or out.dtype != torch.float16:
        raise ValueError("out must be fp16 with shape [M, N]")
    if n % 2 != 0:
        raise ValueError("AWQ W8A8 GEMM requires an even output feature dimension")

    engine = get_awq_engine()
    x_contig = x.contiguous()
    engine.invoke_quant(x_int8, x_contig, scale_a)

    w_scales = _as_awq_scales(scale_w.to(x.device), n)
    if bias is None:
        engine.w8a8_gemm_forward_cuda(x_int8, weight_awq, w_scales, scale_a, out)
    else:
        engine.w8a8_gemm_fuse_bias_forward_cuda(
            x_int8,
            weight_awq,
            w_scales,
            scale_a,
            out,
            bias.contiguous().to(torch.float16),
        )
    return out


__all__ = [
    "awq_fused_quant_gemm_w8a8",
    "awq_fused_quant_gemm_w8a8_prealloc",
    "awq_gemm_w8a8",
    "get_awq_engine",
    "is_awq_available",
    "quantize_awq_per_token",
]
