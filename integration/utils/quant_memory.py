"""Memory accounting helpers for quantized MoDiff modules."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import torch


def tensor_mib(tensor: Any) -> float:
    """Return tensor storage size in MiB, or 0 for missing/non-tensor values."""
    if not isinstance(tensor, torch.Tensor):
        return 0.0
    return tensor.numel() * tensor.element_size() / 1024 / 1024


def _add_attr(bucket: Dict[str, float], module: torch.nn.Module, key: str, attr: str) -> None:
    bucket[key] += tensor_mib(getattr(module, attr, None))


def report_quant_memory(model: torch.nn.Module) -> Dict[str, Any]:
    """Bucket resident tensors held by INT8/INT4/MoDiff modules.

    Run this after a warmup pass. Many MoDiff caches are lazy-created on the
    first sampled pass, so setup-only memory snapshots miss the important part.
    """
    buckets: Dict[str, float] = defaultdict(float)
    module_counts: Dict[str, int] = defaultdict(int)

    for module in model.modules():
        name = type(module).__name__

        if name in ("OptimizedInt8Conv2d", "OptimizedInt4Conv2d"):
            module_counts[name] += 1
            _add_attr(buckets, module, "conv_a_hat_cache_mib", "a_hat_cache")
            _add_attr(buckets, module, "conv_o_hat_cache_mib", "o_hat_cache")
            _add_attr(buckets, module, "conv_residual_buf_mib", "_residual_buf")
            _add_attr(buckets, module, "conv_quant_weights_mib", "weight_int8")
            _add_attr(buckets, module, "conv_quant_weights_mib", "weight_packed")
            _add_attr(buckets, module, "conv_bias_mib", "bias")
            _add_attr(buckets, module, "conv_scale_state_mib", "weight_scale_channel")
            _add_attr(buckets, module, "conv_scale_state_mib", "smooth_scale")
            _add_attr(buckets, module, "conv_scale_state_mib", "_smooth_inv")
            _add_attr(buckets, module, "conv_scale_state_mib", "static_input_scale")
            _add_attr(buckets, module, "conv_scale_state_mib", "_cached_alpha_tensor")
            _add_attr(buckets, module, "conv_scale_state_mib", "_cached_scale_tensor")

        elif name in ("OptimizedInt8Linear", "OptimizedInt4Linear"):
            module_counts[name] += 1
            _add_attr(buckets, module, "linear_a_hat_cache_mib", "a_hat_cache")
            _add_attr(buckets, module, "linear_o_hat_cache_mib", "o_hat_cache")
            _add_attr(buckets, module, "linear_residual_buf_mib", "_residual_buf")
            _add_attr(buckets, module, "linear_fp16_weights_mib", "weight_fp16")
            # weight_int8_t / weight_dequant_scale exist on OptimizedInt8Linear, which still has
            # a live int_gemm path. OptimizedInt4Linear used to carry a packed-int4
            # `weight_packed_t` for its W4A4 path; that path and both of its buffers were removed
            # on 2026-08-01 (no crossover at any K -- see the class docstring), so INT4 linears now
            # contribute only weight_fp16 here. The line accounting weight_packed_t was dropped
            # rather than left to silently sum to zero.
            _add_attr(buckets, module, "linear_quant_weights_mib", "weight_int8_t")
            _add_attr(buckets, module, "linear_scale_state_mib", "weight_dequant_scale")
            _add_attr(buckets, module, "linear_scale_state_mib", "static_input_scale")
            _add_attr(buckets, module, "linear_bias_mib", "bias")

        elif name == "MoDiffConv1dCUTLASS":
            module_counts[name] += 1
            _add_attr(buckets, module, "attention_a_hat_cache_mib", "a_hat_cache")
            _add_attr(buckets, module, "attention_o_hat_cache_mib", "o_hat_cache")
            _add_attr(buckets, module, "attention_fp16_weights_mib", "weight_fp16")
            _add_attr(buckets, module, "attention_quant_weights_mib", "w_int8_t")
            _add_attr(buckets, module, "attention_scale_state_mib", "w_scale")
            _add_attr(buckets, module, "attention_bias_mib", "bias_fp32")

    total_mib = sum(buckets.values())
    sorted_buckets = {k: round(v, 3) for k, v in sorted(buckets.items()) if v > 0}
    cache_mib = sum(v for k, v in buckets.items() if "cache" in k or "residual_buf" in k)
    weight_mib = sum(v for k, v in buckets.items() if "weight" in k)
    scale_mib = sum(v for k, v in buckets.items() if "scale" in k or "bias" in k)

    return {
        "total_tracked_mib": round(total_mib, 3),
        "cache_and_residual_mib": round(cache_mib, 3),
        "weight_mib": round(weight_mib, 3),
        "scale_and_bias_mib": round(scale_mib, 3),
        "buckets_mib": sorted_buckets,
        "module_counts": dict(sorted(module_counts.items())),
    }


def format_quant_memory_report(report: Dict[str, Any], max_buckets: int = 8) -> str:
    """Format the largest memory buckets for console logs."""
    buckets = report.get("buckets_mib", {})
    top = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:max_buckets]
    parts = [
        f"tracked={report.get('total_tracked_mib', 0):.1f}MiB",
        f"cache/residual={report.get('cache_and_residual_mib', 0):.1f}MiB",
        f"weights={report.get('weight_mib', 0):.1f}MiB",
    ]
    if top:
        parts.append("top=[" + ", ".join(f"{k}:{v:.1f}" for k, v in top) + "]")
    return " | ".join(parts)
