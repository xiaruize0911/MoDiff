"""
AWQ-backed INT8 baseline linear layer.

This is intentionally baseline-only: it does not implement MoDiff temporal
modulation or output-cache accumulation. It provides a comparable W8A8 linear
path using llm-awq's CUDA extension.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import MoDiffConfig
from ..kernels.awq_w8a8 import awq_fused_quant_gemm_w8a8


class AWQW8A8BaselineLinear(nn.Module):
    """W8A8 linear baseline using AWQ's quantization + GEMM kernels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: MoDiffConfig | None = None,
    ):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError("AWQ W8A8 GEMM requires even out_features")

        self.in_features = in_features
        self.out_features = out_features
        self.config = config or MoDiffConfig(
            weight_bits=8,
            act_bits=8,
            modulation_enabled=False,
            weight_channel_wise=True,
        )

        # AWQ GEMM expects [N, K] weights.
        self.register_buffer("weight_int8_awq", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: MoDiffConfig | None = None,
    ) -> "AWQW8A8BaselineLinear":
        config = config or MoDiffConfig(
            weight_bits=8,
            act_bits=8,
            modulation_enabled=False,
            weight_channel_wise=True,
        )
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            config=config,
        ).to(linear.weight.device)

        weight = linear.weight.detach().float()
        if config.weight_channel_wise:
            weight_max = weight.abs().max(dim=1).values
            weight_scale = torch.clamp(weight_max / 127.0, min=config.eps)
            weight_int = torch.round(weight / weight_scale.unsqueeze(1)).clamp(-128, 127)
        else:
            weight_max = weight.abs().max()
            weight_scale = torch.clamp(weight_max / 127.0, min=config.eps)
            weight_int = torch.round(weight / weight_scale).clamp(-128, 127)
            weight_scale = weight_scale.expand(linear.out_features)

        q_linear.weight_int8_awq.copy_(weight_int.contiguous().to(torch.int8))
        q_linear.weight_scale.copy_(weight_scale.contiguous().float())
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.detach().float())
        return q_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        out = awq_fused_quant_gemm_w8a8(
            x_2d,
            self.weight_int8_awq,
            self.weight_scale,
            self.bias,
            weight_is_awq_layout=True,
        )
        return out.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits=W8A8, backend=awq"
        )


__all__ = ["AWQW8A8BaselineLinear"]
