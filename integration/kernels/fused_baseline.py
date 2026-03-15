"""
Fused Kernel Baseline: Separate Q + Computation + DQ for INT8 and INT4.

This module provides a baseline implementation where quantization (Q),
convolution (computation), and dequantization (DQ) are done as separate
kernels. This is used to compare against the current MoDiff fused kernel
implementation to measure the benefit of kernel fusion.

Modes:
- int8_fused_baseline: INT8 with separate Q → Conv → DQ kernels
- int4_fused_baseline: INT4 with separate Q → Conv → DQ kernels

The "fused" in the current MoDiff implementation combines:
  sub_absmax_scale + scale_quantize + conv + scale_accumulate
into fewer kernel launches. This baseline splits them apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import time

try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False


class SeparateKernelInt8Conv2d(nn.Module):
    """
    INT8 Conv2d with separate Q, Computation, and DQ kernels.
    
    This uses the same CUTLASS backend but calls each operation separately:
    1. Compute absmax (separate kernel)
    2. Compute scale = 127/absmax (separate kernel)  
    3. Quantize to INT8 (separate kernel)
    4. CUTLASS INT8 conv (separate kernel)
    5. Dequantize with weight scale (separate kernel)
    6. Add bias (separate kernel)
    
    vs the fused version which combines steps 1-3 into step1_quantize_fprop
    and steps 4-6 into conv2d_int8_fprop_o_hat.
    """

    def __init__(self, conv: nn.Conv2d, layer_name: str = ""):
        super().__init__()
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        K = self.out_channels

        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self.groups = conv.groups

        w_data = conv.weight.data

        # Per-channel INT8 weight quantization
        w_flat = w_data.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)
        self.register_buffer('weight_scale_channel', ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        w_quant = w_quant.reshape_as(w_data)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        self.register_buffer('weight_int8', w_nhwc)

        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.view(1, -1, 1, 1))
        else:
            self.bias = None

        self._empty_bias = None
        self.use_cutlass = HAS_CUTLASS and self.groups == 1

        # MoDiff state
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        self.warmup_steps = 3

        # Calibration
        self.is_calibrated = False
        self.register_buffer('static_input_scale', torch.tensor(1.0))
        self._cached_scale_float: Optional[float] = None
        self._cached_alpha_tensor: Optional[torch.Tensor] = None

        # Kernel timing
        self.kernel_times: Dict[str, float] = {}

    def _separate_quantize(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Step 1: Separate quantization kernel."""
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        # Scale
        x_scaled = x * scale
        # Round + clamp
        x_int8 = x_scaled.round().clamp(-127, 127).to(torch.int8)
        return x_int8

    def _separate_conv(self, x_int8: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Step 2: Separate CUTLASS convolution kernel."""
        if self._empty_bias is None or self._empty_bias.device != x_int8.device:
            self._empty_bias = torch.empty(0, device=x_int8.device)

        return modiff_cutlass.conv2d_int8_fprop(
            x_int8, self.weight_int8, alpha, self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

    def _separate_dequantize(self, out_raw: torch.Tensor) -> torch.Tensor:
        """Step 3: Separate dequantization + bias kernel."""
        out = out_raw * self.weight_scale_channel
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        if not self.modiff_enabled:
            return self._forward_standard(x)
        elif self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
            return output
        else:
            return self._forward_modulated(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward with separate kernels."""
        if self.is_calibrated and self.use_cutlass:
            if self._cached_scale_float is None:
                self._cached_scale_float = float(self.static_input_scale.item())
            if self._cached_alpha_tensor is None:
                self._cached_alpha_tensor = torch.tensor(
                    [1.0 / self._cached_scale_float], device=x.device, dtype=torch.float32)

            # Separate kernel 1: Quantize
            x_int8 = self._separate_quantize(x, self._cached_scale_float)
            # Separate kernel 2: Conv
            out_raw = self._separate_conv(x_int8, self._cached_alpha_tensor)
            # Separate kernel 3: Dequantize + bias
            return self._separate_dequantize(out_raw)

        # Fallback
        abs_max = x.abs().max().item()
        scale = 127.0 / max(abs_max, 1e-6)
        x_int8 = self._separate_quantize(x, scale)
        alpha = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)
        out_raw = self._separate_conv(x_int8, alpha)
        return self._separate_dequantize(out_raw)

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """First step with separate kernels."""
        abs_max = x.abs().max().item()
        scale = 127.0 / max(abs_max, 1e-6)

        a_hat = (x * scale).round().clamp(-127, 127) / scale
        x_int8 = self._separate_quantize(x, scale)
        alpha = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)
        out_raw = self._separate_conv(x_int8, alpha)
        o_hat = self._separate_dequantize(out_raw)

        for _ in range(self.warmup_steps - 1):
            residual = x - a_hat
            r_abs = residual.abs().max().item()
            r_scale = 127.0 / max(r_abs, 1e-6)
            r_dq = (residual * r_scale).round().clamp(-127, 127) / r_scale
            r_int8 = self._separate_quantize(residual, r_scale)
            r_alpha = torch.tensor([1.0 / r_scale], device=x.device, dtype=torch.float32)
            r_out = self._separate_conv(r_int8, r_alpha)
            r_out = r_out * self.weight_scale_channel  # dequant only, no bias
            a_hat = a_hat + r_dq
            o_hat = o_hat + r_out

        self.a_hat_cache = a_hat
        self.o_hat_cache = o_hat
        return o_hat.clone()

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """Modulated step with SEPARATE kernels (no fusion)."""
        if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
            self.is_first_step = True
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

        # Separate kernel 1: Compute residual
        residual = x - self.a_hat_cache

        # Separate kernel 2: Compute absmax
        abs_max = residual.abs().amax()

        # Separate kernel 3: Compute scale
        scale = 127.0 / torch.clamp(abs_max, min=1e-6)
        inv_scale = 1.0 / scale

        # Separate kernel 4: Quantize
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        x_int8 = (residual * scale).round().clamp(-127, 127).to(torch.int8)

        # Separate kernel 5: Dequantize for cache update
        r_dq = x_int8.float() * inv_scale

        # Separate kernel 6: Update a_hat cache
        self.a_hat_cache.add_(r_dq)

        # Separate kernel 7: CUTLASS conv
        if self._empty_bias is None or self._empty_bias.device != x.device:
            self._empty_bias = torch.empty(0, device=x.device)
        out_raw = modiff_cutlass.conv2d_int8_fprop(
            x_int8, self.weight_int8, inv_scale.view(1), self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

        # Separate kernel 8: Dequant weight scale
        out_scaled = out_raw * self.weight_scale_channel

        # Separate kernel 9: Accumulate o_hat
        self.o_hat_cache.add_(out_scaled)

        return self.o_hat_cache

    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()

    def reset_state(self):
        self.is_first_step = True
        self.a_hat_cache = None
        self.o_hat_cache = None

    def set_static_scale(self, scale: float):
        self.static_input_scale.fill_(scale)
        self.is_calibrated = True
        self._cached_scale_float = float(scale)
        self._cached_alpha_tensor = None


class SeparateKernelInt4Conv2d(nn.Module):
    """
    INT4 Conv2d with separate Q, Computation, and DQ kernels.
    
    Same as SeparateKernelInt8Conv2d but for INT4 precision.
    """

    def __init__(self, conv: nn.Conv2d, layer_name: str = ""):
        super().__init__()
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        K = self.out_channels

        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self.groups = conv.groups

        w_data = conv.weight.data
        w_flat = w_data.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values
        ch_scale = torch.clamp(ch_max / 7.0, min=1e-8)
        self.register_buffer('weight_scale_channel', ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
        w_quant = w_quant.reshape_as(w_data)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()

        from integration.kernels.int4_optimized import pack_int4
        if self.in_channels % 2 == 0:
            self.register_buffer('weight_packed', pack_int4(w_nhwc))
        else:
            self.register_buffer('weight_packed', torch.empty(0, dtype=torch.int8))

        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.view(1, -1, 1, 1))
        else:
            self.bias = None

        self._empty_bias = None
        self.use_cutlass = HAS_CUTLASS and self.groups == 1 and self.in_channels % 2 == 0

        # MoDiff state
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        self.warmup_steps = 3

        self.is_calibrated = False
        self.register_buffer('static_input_scale', torch.tensor(1.0))
        self._cached_scale_float: Optional[float] = None
        self._cached_alpha_tensor: Optional[torch.Tensor] = None

    def _separate_quantize_pack(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Separate quantize + pack to INT4."""
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        x_scaled = x * scale
        x_clamped = x_scaled.round().clamp(-7, 7)
        # Pack using CUTLASS kernel
        return modiff_cutlass.quantize_and_pack(x_clamped)

    def _separate_conv(self, x_packed: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Separate CUTLASS INT4 convolution."""
        if self._empty_bias is None or self._empty_bias.device != x_packed.device:
            self._empty_bias = torch.empty(0, device=x_packed.device)
        return modiff_cutlass.conv2d_int4_fprop(
            x_packed, self.weight_packed, alpha, self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

    def _separate_dequantize(self, out_raw: torch.Tensor) -> torch.Tensor:
        """Separate dequant + bias."""
        out = out_raw * self.weight_scale_channel
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        if not self.modiff_enabled:
            return self._forward_standard(x)
        elif self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
            return output
        else:
            return self._forward_modulated(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_calibrated and self.use_cutlass:
            if self._cached_scale_float is None:
                self._cached_scale_float = float(self.static_input_scale.item())
            if self._cached_alpha_tensor is None:
                self._cached_alpha_tensor = torch.tensor(
                    [1.0 / self._cached_scale_float], device=x.device, dtype=torch.float32)
            x_packed = self._separate_quantize_pack(x, self._cached_scale_float)
            out_raw = self._separate_conv(x_packed, self._cached_alpha_tensor)
            return self._separate_dequantize(out_raw)
        abs_max = x.abs().max().item()
        scale = 7.0 / max(abs_max, 1e-6)
        x_packed = self._separate_quantize_pack(x, scale)
        alpha = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)
        out_raw = self._separate_conv(x_packed, alpha)
        return self._separate_dequantize(out_raw)

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        abs_max = x.abs().max().item()
        scale = 7.0 / max(abs_max, 1e-6)
        a_hat = (x * scale).round().clamp(-7, 7) / scale
        x_packed = self._separate_quantize_pack(x, scale)
        alpha = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)
        out_raw = self._separate_conv(x_packed, alpha)
        o_hat = self._separate_dequantize(out_raw)

        for _ in range(self.warmup_steps - 1):
            residual = x - a_hat
            r_abs = residual.abs().max().item()
            r_scale = 7.0 / max(r_abs, 1e-6)
            r_dq = (residual * r_scale).round().clamp(-7, 7) / r_scale
            r_packed = self._separate_quantize_pack(residual, r_scale)
            r_alpha = torch.tensor([1.0 / r_scale], device=x.device, dtype=torch.float32)
            r_out = self._separate_conv(r_packed, r_alpha)
            r_out = r_out * self.weight_scale_channel
            a_hat = a_hat + r_dq
            o_hat = o_hat + r_out
        self.a_hat_cache = a_hat
        self.o_hat_cache = o_hat
        return o_hat.clone()

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
            self.is_first_step = True
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

        # Separate kernel 1: Compute residual
        residual = x - self.a_hat_cache

        # Separate kernel 2: Compute absmax
        abs_max = residual.abs().amax()

        # Separate kernel 3: Compute scale
        scale = 7.0 / torch.clamp(abs_max, min=1e-6)
        inv_scale = 1.0 / scale

        # Separate kernel 4: Quantize + pack
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        r_clamped = (residual * scale).round().clamp(-7, 7)
        x_packed = modiff_cutlass.quantize_and_pack(r_clamped)

        # Separate kernel 5: Dequantize for cache
        r_dq = r_clamped / scale

        # Separate kernel 6: Update a_hat cache
        self.a_hat_cache.add_(r_dq)

        # Separate kernel 7: CUTLASS conv
        if self._empty_bias is None or self._empty_bias.device != x.device:
            self._empty_bias = torch.empty(0, device=x.device)
        out_raw = modiff_cutlass.conv2d_int4_fprop(
            x_packed, self.weight_packed, inv_scale.view(1), self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

        # Separate kernel 8: Dequant weight scale
        out_scaled = out_raw * self.weight_scale_channel

        # Separate kernel 9: Accumulate o_hat
        self.o_hat_cache.add_(out_scaled)

        return self.o_hat_cache

    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()

    def reset_state(self):
        self.is_first_step = True
        self.a_hat_cache = None
        self.o_hat_cache = None

    def set_static_scale(self, scale: float):
        self.static_input_scale.fill_(scale)
        self.is_calibrated = True
        self._cached_scale_float = float(scale)
        self._cached_alpha_tensor = None


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def convert_model_to_separate_int8(model: nn.Module, prefix: str = "") -> nn.Module:
    """Convert Conv2d layers to SeparateKernelInt8Conv2d."""
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, SeparateKernelInt8Conv2d):
            if child.in_channels < 32:
                continue
            if 'skip' in name or full_name.startswith('out.') or child.kernel_size == (1, 1) or child.groups != 1:
                continue
            optimized = SeparateKernelInt8Conv2d(child, layer_name=full_name)
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized = optimized.to(target_device)
            setattr(model, name, optimized)
        else:
            convert_model_to_separate_int8(child, prefix=full_name)
    model = model.to(memory_format=torch.channels_last)
    for m in model.modules():
        if isinstance(m, SeparateKernelInt8Conv2d):
            m.weight_int8.data = m.weight_int8.data.contiguous()
    return model


def convert_model_to_separate_int4(model: nn.Module, prefix: str = "") -> nn.Module:
    """Convert Conv2d layers to SeparateKernelInt4Conv2d."""
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, SeparateKernelInt4Conv2d):
            if child.in_channels < 32:
                continue
            if 'skip' in name or full_name.startswith('out.') or child.kernel_size == (1, 1) or child.groups != 1:
                continue
            optimized = SeparateKernelInt4Conv2d(child, layer_name=full_name)
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized = optimized.to(target_device)
            setattr(model, name, optimized)
        else:
            convert_model_to_separate_int4(child, prefix=full_name)
    model = model.to(memory_format=torch.channels_last)
    for m in model.modules():
        if isinstance(m, SeparateKernelInt4Conv2d):
            m.weight_packed.data = m.weight_packed.data.contiguous()
    return model


def enable_modiff_mode_separate_int8(model: nn.Module, enabled: bool = True):
    for m in model.modules():
        if isinstance(m, SeparateKernelInt8Conv2d):
            m.enable_modiff(enabled)


def enable_modiff_mode_separate_int4(model: nn.Module, enabled: bool = True):
    for m in model.modules():
        if isinstance(m, SeparateKernelInt4Conv2d):
            m.enable_modiff(enabled)


def reset_modiff_state_separate_int8(model: nn.Module):
    for m in model.modules():
        if isinstance(m, SeparateKernelInt8Conv2d):
            m.reset_state()


def reset_modiff_state_separate_int4(model: nn.Module):
    for m in model.modules():
        if isinstance(m, SeparateKernelInt4Conv2d):
            m.reset_state()


def apply_separate_int8_scales(model: nn.Module, scales: Dict[str, float]) -> int:
    loaded = 0
    for m in model.modules():
        if isinstance(m, SeparateKernelInt8Conv2d) and m.layer_name in scales:
            m.set_static_scale(scales[m.layer_name])
            loaded += 1
    return loaded


def apply_separate_int4_scales(model: nn.Module, scales: Dict[str, float]) -> int:
    loaded = 0
    for m in model.modules():
        if isinstance(m, SeparateKernelInt4Conv2d) and m.layer_name in scales:
            m.set_static_scale(scales[m.layer_name])
            loaded += 1
    return loaded
