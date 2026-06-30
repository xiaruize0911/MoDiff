"""AWQ W8A8 wrapper for Conv1d(kernel_size=1) attention projections."""

from __future__ import annotations

import torch
import torch.nn as nn


class AWQW8A8Conv1d1x1(nn.Module):
    """Run a Conv1d-1x1 as AWQ W8A8 GEMM over flattened sequence tokens."""

    def __init__(self, conv: nn.Conv1d, layer_name: str = ""):
        super().__init__()
        if conv.kernel_size != (1,) or conv.stride != (1,) or conv.padding != (0,) or conv.dilation != (1,) or conv.groups != 1:
            raise ValueError("AWQW8A8Conv1d1x1 only supports ungrouped Conv1d with kernel_size=1")
        if conv.out_channels % 2 != 0:
            raise ValueError("AWQ W8A8 GEMM requires an even output channel count")

        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels

        weight_fp32 = conv.weight.data.reshape(conv.out_channels, conv.in_channels).float()
        weight_absmax = weight_fp32.abs().max()
        weight_scale = torch.clamp(weight_absmax / 127.0, min=1e-8)
        weight_int8_awq = torch.round(weight_fp32 / weight_scale).clamp(-128, 127).to(torch.int8).contiguous()
        self.register_buffer("weight_int8_awq", weight_int8_awq)
        self.register_buffer("weight_dequant_scale", weight_scale.float().reshape(1))

        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.half())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Conv1d input must be [B, C, L]")
        if not x.is_cuda:
            return torch.nn.functional.conv1d(
                x,
                (self.weight_int8_awq.float() * self.weight_dequant_scale).reshape(self.out_channels, self.in_channels, 1),
                self.bias.float() if self.bias is not None else None,
            )

        from modiff_triton.kernels.awq_w8a8 import awq_fused_quant_gemm_w8a8

        b, c, length = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {c}")

        x_2d = x.transpose(1, 2).reshape(-1, c)
        out_2d = awq_fused_quant_gemm_w8a8(
            x_2d,
            self.weight_int8_awq,
            self.weight_dequant_scale,
            self.bias,
            weight_is_awq_layout=True,
        )
        return out_2d.reshape(b, length, self.out_channels).transpose(1, 2).contiguous()


def convert_model_conv1d_1x1_to_awq(model: nn.Module, prefix: str = "") -> int:
    """Replace eligible Conv1d-1x1 layers with AWQ W8A8 GEMM wrappers."""
    converted = 0
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv1d) and not isinstance(child, AWQW8A8Conv1d1x1):
            if (
                child.kernel_size == (1,)
                and child.stride == (1,)
                and child.padding == (0,)
                and child.dilation == (1,)
                and child.groups == 1
                and child.out_channels % 2 == 0
            ):
                optimized = AWQW8A8Conv1d1x1(child, layer_name=full_name)
                target_device = child.weight.device
                if target_device.type != "cpu":
                    optimized = optimized.to(target_device)
                setattr(model, name, optimized)
                converted += 1
            else:
                converted += convert_model_conv1d_1x1_to_awq(child, prefix=full_name)
        else:
            converted += convert_model_conv1d_1x1_to_awq(child, prefix=full_name)
    return converted


__all__ = ["AWQW8A8Conv1d1x1", "convert_model_conv1d_1x1_to_awq"]
