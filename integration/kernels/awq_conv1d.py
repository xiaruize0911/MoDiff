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
        # The AWQ W8A8 GEMM tiles its N (output-feature) dimension in blocks of 128
        # (CTA_N). For an N that is not a multiple of 128 (e.g. this UNet's 576/192-wide
        # attention projections at the finest resolution) the kernel's fused-bias
        # variant reads a full CTA_N-wide bias slice for the last tile, running past the
        # end of a length-N bias array. Padding N (and weight/bias) up to a full tile
        # keeps that read in-bounds and avoids partial-tile handling entirely.
        # NOTE: this is a robustness fix, not a cure-all -- it does NOT resolve the
        # separate batch>=16 illegal-memory-access in dense_kernel0_fuse_bias (see the
        # batch-aware gate in benchmark_ldm.py); that is a distinct, memory-layout-
        # dependent defect in the vendored kernel that only manifests with the INT8
        # Conv2d path also active.
        self._padded_out_channels = ((conv.out_channels + 127) // 128) * 128
        pad = self._padded_out_channels - conv.out_channels

        weight_fp32 = conv.weight.data.reshape(conv.out_channels, conv.in_channels).float()
        weight_absmax = weight_fp32.abs().max()
        weight_scale = torch.clamp(weight_absmax / 127.0, min=1e-8)
        weight_int8_awq = torch.round(weight_fp32 / weight_scale).clamp(-128, 127).to(torch.int8).contiguous()
        if pad:
            weight_int8_awq = torch.cat(
                [weight_int8_awq, weight_int8_awq.new_zeros(pad, conv.in_channels)], dim=0
            )
        self.register_buffer("weight_int8_awq", weight_int8_awq)
        self.register_buffer("weight_dequant_scale", weight_scale.float().reshape(1))

        if conv.bias is not None:
            bias = conv.bias.data.half()
            if pad:
                bias = torch.cat([bias, bias.new_zeros(pad)])
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.register_buffer("_weight_scale_awq", weight_scale.half().reshape(1), persistent=False)
        self._x_int8_buf = None
        self._scale_a_buf = None
        self._out_2d_buf = None

    def _ensure_awq_buffers(self, x_2d: torch.Tensor) -> None:
        m = x_2d.shape[0]
        if (
            self._x_int8_buf is None
            or self._x_int8_buf.shape != x_2d.shape
            or self._x_int8_buf.device != x_2d.device
        ):
            self._x_int8_buf = torch.empty_like(x_2d, dtype=torch.int8)
        if (
            self._scale_a_buf is None
            or self._scale_a_buf.shape != (m,)
            or self._scale_a_buf.device != x_2d.device
        ):
            self._scale_a_buf = torch.empty((m,), device=x_2d.device, dtype=torch.float16)
        out_shape = (m, self._padded_out_channels)
        if (
            self._out_2d_buf is None
            or self._out_2d_buf.shape != out_shape
            or self._out_2d_buf.device != x_2d.device
        ):
            self._out_2d_buf = torch.empty(out_shape, device=x_2d.device, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Conv1d input must be [B, C, L]")
        if not x.is_cuda:
            weight = self.weight_int8_awq[: self.out_channels]
            bias = self.bias[: self.out_channels] if self.bias is not None else None
            return torch.nn.functional.conv1d(
                x,
                (weight.float() * self.weight_dequant_scale).reshape(self.out_channels, self.in_channels, 1),
                bias.float() if bias is not None else None,
            )

        from modiff_triton.kernels.awq_w8a8 import awq_fused_quant_gemm_w8a8_prealloc

        b, c, length = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {c}")

        x_2d = x.transpose(1, 2).reshape(-1, c)
        self._ensure_awq_buffers(x_2d)
        out_2d = awq_fused_quant_gemm_w8a8_prealloc(
            x_2d,
            self.weight_int8_awq,
            self._weight_scale_awq,
            self._x_int8_buf,
            self._scale_a_buf,
            self._out_2d_buf,
            self.bias,
            weight_is_awq_layout=True,
        )
        if self._padded_out_channels != self.out_channels:
            out_2d = out_2d[:, : self.out_channels]
        return out_2d.reshape(b, length, self.out_channels).transpose(1, 2).contiguous()


def convert_model_conv1d_1x1_to_awq(
    model: nn.Module,
    prefix: str = "",
    min_in_channels: int = 0,
) -> int:
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
                and child.in_channels >= min_in_channels
            ):
                optimized = AWQW8A8Conv1d1x1(child, layer_name=full_name)
                target_device = child.weight.device
                if target_device.type != "cpu":
                    optimized = optimized.to(target_device)
                setattr(model, name, optimized)
                converted += 1
            else:
                converted += convert_model_conv1d_1x1_to_awq(
                    child,
                    prefix=full_name,
                    min_in_channels=min_in_channels,
                )
        else:
            converted += convert_model_conv1d_1x1_to_awq(
                child,
                prefix=full_name,
                min_in_channels=min_in_channels,
            )
    return converted


__all__ = ["AWQW8A8Conv1d1x1", "convert_model_conv1d_1x1_to_awq"]
