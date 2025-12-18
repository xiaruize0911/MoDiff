"""
MoDiff Quantized Conv2d Layers

This module implements MoDiff for convolutional layers.
The paper states in Section 3.1:
    "We denote the input and output activations for A^(l)(·) at time step t 
     as a_t^(l) and o_t^(l)=A^(l)(a_t^(l)), respectively. Note that we focus 
     on accelerating the computation of linear operators, such as linear and 
     convolutional layers, since they are the most costly operations in 
     neural networks."

Conv2d is treated as a linear operator with the same MoDiff framework:
    â_t = Q(a_t - â_{t+1}) + â_{t+1}          -- Eq. (ec5)
    ô_t = Conv(Q(a_t - â_{t+1})) + ô_{t+1}    -- Eq. (ec6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from .config import MoDiffConfig
from ..kernels.quantize import (
    quantize_symmetric_int8,
    quantize_symmetric_int4,
    dequantize_int8,
    dequantize_int4,
    compute_dynamic_scale_int8,
    compute_dynamic_scale_int4,
)
from ..kernels.modulated_quantize import (
    modulated_quantize_int8,
    modulated_quantize_int4,
    modulated_quantize_first_step_int8,
    modulated_quantize_first_step_int4,
)


class W8A8MoDiffConv2d(nn.Module):
    """
    W8A8 Conv2d layer with MoDiff error-compensated modulation.
    
    Implements the MoDiff framework for 2D convolutions:
        - INT8 weights (pre-quantized, per-channel)
        - INT8 activations (dynamic quantization)
        - Error-compensated modulation across timesteps
        
    For Conv2d, we:
        1. Quantize the input activation (or residual)
        2. Perform INT8 convolution
        3. Dequantize and accumulate with cache
        
    Note: This uses PyTorch's conv2d with quantized simulation.
    For maximum performance, a custom Triton conv kernel would be needed.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        config: MoDiffConfig = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.config = config or MoDiffConfig(weight_bits=8, act_bits=8)
        
        # Weight buffer (INT8)
        self.register_buffer(
            "weight_int8",
            torch.empty(out_channels, in_channels // groups, *self.kernel_size, dtype=torch.int8)
        )
        
        # Per-channel weight scales
        self.register_buffer(
            "weight_scale",
            torch.empty(out_channels, dtype=torch.float32)
        )
        
        # Bias
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(out_channels, dtype=torch.float32)
            )
        else:
            self.register_buffer("bias", None)
        
        # MoDiff caches
        self.a_hat_cache: Optional[torch.Tensor] = None  # [B, C, H, W]
        self.o_hat_cache: Optional[torch.Tensor] = None  # [B, C_out, H_out, W_out]
        
        self.is_first_step = True
        self.modulation_enabled = self.config.modulation_enabled
    
    @classmethod
    def from_conv2d(
        cls,
        conv: nn.Conv2d,
        config: MoDiffConfig = None,
    ) -> "W8A8MoDiffConv2d":
        """Create W8A8MoDiffConv2d from a pretrained nn.Conv2d."""
        config = config or MoDiffConfig(weight_bits=8, act_bits=8)
        
        q_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            config=config,
        )
        
        # Quantize weights (per-channel along out_channels)
        weight = conv.weight.data.float()  # [out_c, in_c/groups, kH, kW]
        
        # Reshape for per-channel quantization
        weight_flat = weight.view(conv.out_channels, -1)  # [out_c, in_c*kH*kW]
        weight_max = weight_flat.abs().max(dim=1).values  # [out_c]
        weight_scale = weight_max / 127.0
        weight_scale = torch.clamp(weight_scale, min=1e-8)
        
        # Quantize
        weight_int = torch.round(weight / weight_scale.view(-1, 1, 1, 1)).clamp(-128, 127)
        
        q_conv.weight_int8.copy_(weight_int.to(torch.int8))
        q_conv.weight_scale.copy_(weight_scale)
        
        if conv.bias is not None:
            q_conv.bias.copy_(conv.bias.data.float())
        
        return q_conv
    
    def reset_cache(self):
        """Reset MoDiff caches."""
        self.a_hat_cache = None
        self.o_hat_cache = None
        self.is_first_step = True
    
    def set_modulation(self, enabled: bool):
        """Enable/disable modulation."""
        self.modulation_enabled = enabled
        if not enabled:
            self.reset_cache()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MoDiff.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C_out, H_out, W_out]
        """
        if not self.modulation_enabled:
            return self._forward_standard(x)
        
        if self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
        else:
            output = self._forward_modulated(x)
        
        return output
    
    def _quantize_activation_int8(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activation tensor for Conv2d."""
        # Flatten for scale computation
        x_flat = x.flatten()
        scale, _ = compute_dynamic_scale_int8(x_flat, symmetric=True)
        
        # Quantize
        x_int = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
        
        return x_int, scale
    
    def _conv2d_int8(
        self,
        x_int: torch.Tensor,
        scale_a: torch.Tensor,
        add_bias: bool = True,
    ) -> torch.Tensor:
        """
        INT8 Conv2d with dequantization.
        
        Performs: output = dequant(conv(x_int, weight_int)) + bias
        """
        # Convert to float for conv (simulated quantization)
        # In a true INT8 kernel, this would be int8 throughout
        x_float = x_int.float()
        weight_float = self.weight_int8.float()
        
        # Convolution
        output = F.conv2d(
            x_float,
            weight_float,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        
        # Dequantize: output * scale_a * scale_w
        # Per-channel weight scale: need to broadcast [out_c] -> [1, out_c, 1, 1]
        weight_scale_bc = self.weight_scale.view(1, -1, 1, 1)
        output = output * scale_a * weight_scale_bc
        
        # Add bias
        if add_bias and self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard W8A8 forward without modulation."""
        x_int, scale_a = self._quantize_activation_int8(x)
        return self._conv2d_int8(x_int, scale_a, add_bias=True)
    
    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        First timestep (t=T).
        
        Implements:
            â_T = Q(a_T)
            ô_T = Conv(â_T) + bias
        """
        # Quantize
        x_int, scale_a = self._quantize_activation_int8(x)
        
        # Store dequantized activation as cache (â_T)
        a_hat = x_int.float() * scale_a
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat.half()
        else:
            self.a_hat_cache = a_hat
        
        # Conv with bias
        output = self._conv2d_int8(x_int, scale_a, add_bias=True)
        
        # Store output cache
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """
        Subsequent timesteps (t<T) with error compensation.
        
        Implements:
            â_t = Q(a_t - â_{t+1}) + â_{t+1}
            ô_t = Conv(Q(a_t - â_{t+1})) + ô_{t+1}
        """
        # Get cached values
        a_hat_prev = self.a_hat_cache.float() if self.a_hat_cache.dtype == torch.float16 else self.a_hat_cache
        o_hat_prev = self.o_hat_cache.float() if self.o_hat_cache.dtype == torch.float16 else self.o_hat_cache
        
        # Handle shape mismatches
        if a_hat_prev.shape != x.shape:
            return self._forward_first_step(x)
        
        # Compute residual
        residual = x - a_hat_prev
        
        # Quantize residual
        residual_flat = residual.flatten()
        scale_a, _ = compute_dynamic_scale_int8(residual_flat, symmetric=True)
        residual_int = torch.round(residual / scale_a).clamp(-128, 127).to(torch.int8)
        
        # Dequantize for cache update
        residual_dequant = residual_int.float() * scale_a
        
        # Update activation cache: â_t = Q(residual) + â_{t+1}
        a_hat_new = residual_dequant + a_hat_prev
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat_new.half()
        else:
            self.a_hat_cache = a_hat_new
        
        # Conv WITHOUT bias
        conv_output = self._conv2d_int8(residual_int, scale_a, add_bias=False)
        
        # Accumulate with cached output
        output = conv_output + o_hat_prev
        
        # Update output cache
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bits=W8A8, modulation={self.modulation_enabled}"
        )


class W4A4MoDiffConv2d(nn.Module):
    """
    W4A4 Conv2d layer with MoDiff error-compensated modulation.
    
    Uses INT4 quantization for maximum compression.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        config: MoDiffConfig = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.config = config or MoDiffConfig(weight_bits=4, act_bits=4)
        
        # For W4A4, we store unpacked weights for simplicity
        # A production implementation would pack them
        self.register_buffer(
            "weight_int4",
            torch.empty(out_channels, in_channels // groups, *self.kernel_size, dtype=torch.int8)
        )
        
        self.register_buffer(
            "weight_scale",
            torch.empty(out_channels, dtype=torch.float32)
        )
        
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(out_channels, dtype=torch.float32)
            )
        else:
            self.register_buffer("bias", None)
        
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        
        self.is_first_step = True
        self.modulation_enabled = self.config.modulation_enabled
    
    @classmethod
    def from_conv2d(
        cls,
        conv: nn.Conv2d,
        config: MoDiffConfig = None,
    ) -> "W4A4MoDiffConv2d":
        """Create W4A4MoDiffConv2d from pretrained Conv2d."""
        config = config or MoDiffConfig(weight_bits=4, act_bits=4)
        
        q_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            config=config,
        )
        
        # Quantize weights to INT4 (per-channel)
        weight = conv.weight.data.float()
        weight_flat = weight.view(conv.out_channels, -1)
        weight_max = weight_flat.abs().max(dim=1).values
        weight_scale = weight_max / 7.0  # INT4 symmetric: [-8, 7]
        weight_scale = torch.clamp(weight_scale, min=1e-8)
        
        weight_int = torch.round(weight / weight_scale.view(-1, 1, 1, 1)).clamp(-8, 7)
        
        q_conv.weight_int4.copy_(weight_int.to(torch.int8))
        q_conv.weight_scale.copy_(weight_scale)
        
        if conv.bias is not None:
            q_conv.bias.copy_(conv.bias.data.float())
        
        return q_conv
    
    def reset_cache(self):
        self.a_hat_cache = None
        self.o_hat_cache = None
        self.is_first_step = True
    
    def set_modulation(self, enabled: bool):
        self.modulation_enabled = enabled
        if not enabled:
            self.reset_cache()
    
    def _quantize_activation_int4(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activation to INT4."""
        x_flat = x.flatten()
        scale, _ = compute_dynamic_scale_int4(x_flat, symmetric=True)
        x_int = torch.round(x / scale).clamp(-8, 7).to(torch.int8)
        return x_int, scale
    
    def _conv2d_int4(
        self,
        x_int: torch.Tensor,
        scale_a: torch.Tensor,
        add_bias: bool = True,
    ) -> torch.Tensor:
        """INT4 Conv2d (simulated)."""
        x_float = x_int.float()
        weight_float = self.weight_int4.float()
        
        output = F.conv2d(
            x_float,
            weight_float,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        
        weight_scale_bc = self.weight_scale.view(1, -1, 1, 1)
        output = output * scale_a * weight_scale_bc
        
        if add_bias and self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.modulation_enabled:
            x_int, scale_a = self._quantize_activation_int4(x)
            return self._conv2d_int4(x_int, scale_a, add_bias=True)
        
        if self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
        else:
            output = self._forward_modulated(x)
        
        return output
    
    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        x_int, scale_a = self._quantize_activation_int4(x)
        
        a_hat = x_int.float() * scale_a
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat.half()
        else:
            self.a_hat_cache = a_hat
        
        output = self._conv2d_int4(x_int, scale_a, add_bias=True)
        
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        a_hat_prev = self.a_hat_cache.float() if self.a_hat_cache.dtype == torch.float16 else self.a_hat_cache
        o_hat_prev = self.o_hat_cache.float() if self.o_hat_cache.dtype == torch.float16 else self.o_hat_cache
        
        if a_hat_prev.shape != x.shape:
            return self._forward_first_step(x)
        
        residual = x - a_hat_prev
        
        residual_flat = residual.flatten()
        scale_a, _ = compute_dynamic_scale_int4(residual_flat, symmetric=True)
        residual_int = torch.round(residual / scale_a).clamp(-8, 7).to(torch.int8)
        
        residual_dequant = residual_int.float() * scale_a
        a_hat_new = residual_dequant + a_hat_prev
        
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat_new.half()
        else:
            self.a_hat_cache = a_hat_new
        
        conv_output = self._conv2d_int4(residual_int, scale_a, add_bias=False)
        output = conv_output + o_hat_prev
        
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, bits=W4A4, modulation={self.modulation_enabled}"
        )
