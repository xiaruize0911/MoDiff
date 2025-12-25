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
from ..kernels.conv_w8a8 import (
    conv2d_int8_triton_direct,
    conv2d_int8_triton_accumulate,
)
from ..kernels.conv_w8a8_fused import (
    conv2d_w8a8_3x3_fused,
    conv2d_w8a8_3x3_standard,
)
from ..kernels.gemm_w8a8 import gemm_w8a8, gemm_w8a8_accum
from ..kernels.gemm_w4a4 import gemm_w4a4, gemm_w4a4_accum, pack_int4_weight


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
        assert self.groups == 1, "W4A4MoDiffConv2d currently supports groups=1 for packed INT4 path"
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
        self._last_act_scale = None  # cached activation scale for reuse
    
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
        self._last_act_scale = None
    
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

    def _can_use_triton_conv(self):
        return (
            self.groups == 1 and
            self.kernel_size == (3, 3) and
            self.stride == (1, 1) and
            self.padding == (1, 1) and
            self.dilation == (1, 1)
        )
    
    def _can_use_fused_conv(self):
        """Check if we can use the fused direct conv kernel (fastest path)."""
        return (
            torch.cuda.is_available() and
            self.groups == 1 and
            self.kernel_size == (3, 3) and
            self.stride == (1, 1) and
            self.padding == (1, 1) and
            self.dilation == (1, 1)
        )
    
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

    def _conv2d_int8_im2col_gemm(
        self,
        x_int: torch.Tensor,
        scale_a: torch.Tensor,
        add_bias: bool,
    ) -> torch.Tensor:
        """Integer conv via unfold + int8 GEMM (generic shapes)."""
        N, C, H, W = x_int.shape
        kH, kW = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation

        # Dequant then requant columns to keep scale_a
        x_fp = x_int.float() * scale_a
        cols = torch.nn.functional.unfold(
            x_fp,
            kernel_size=(kH, kW),
            dilation=(dil_h, dil_w),
            padding=(pad_h, pad_w),
            stride=(stride_h, stride_w),
        )  # [N, C*kH*kW, L]
        cols_int = torch.round(cols / scale_a).clamp(-128, 127).to(torch.int8)
        cols_int = cols_int.transpose(1, 2).contiguous().view(-1, C * kH * kW)  # [N*L, K]

        # Prepare weights: [out, in/groups, kH, kW] -> [K, out]
        weight_int8 = self.weight_int8.view(self.out_channels, -1).t().contiguous()
        out_mat = gemm_w8a8(
            cols_int,
            weight_int8,
            scale_a,
            self.weight_scale if self.weight_scale.numel() == 1 else self.weight_scale,
            bias=self.bias if add_bias else None,
        )  # [N*L, O]

        # Reshape back
        L = cols.shape[-1]
        out = out_mat.view(N, L, self.out_channels).transpose(1, 2)
        # Output spatial dims
        H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
        out = out.view(N, self.out_channels, H_out, W_out)
        return out
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard W8A8 forward without modulation."""
        # Use fused direct conv kernel (no quantization overhead, no im2col)
        if self._can_use_fused_conv():
            return conv2d_w8a8_3x3_standard(x, self.weight_int8, self.weight_scale, self.bias)
        
        # Fallback: quantize then conv
        x_int, scale_a = self._quantize_activation_int8(x)
        self._last_act_scale = scale_a

        if torch.cuda.is_available() and self._can_use_triton_conv():
            return conv2d_int8_triton_direct(
                x_int, self.weight_int8, scale_a, self.weight_scale, self.bias
            )
        # integer im2col + GEMM fallback
        return self._conv2d_int8_im2col_gemm(x_int, scale_a, add_bias=True)
    
    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        First timestep (t=T).
        
        Implements:
            â_T = Q(a_T)
            ô_T = Conv(â_T) + bias
        """
        # Quantize
        x_int, scale_a = self._quantize_activation_int8(x)
        self._last_act_scale = scale_a
        
        # Store dequantized activation as cache (â_T)
        a_hat = x_int.float() * scale_a
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat.half()
        else:
            self.a_hat_cache = a_hat
        
        # Conv with bias
        if torch.cuda.is_available() and self._can_use_triton_conv():
            output = conv2d_int8_triton_direct(
                x_int, self.weight_int8, scale_a, self.weight_scale, self.bias
            )
        else:
            output = self._conv2d_int8_im2col_gemm(x_int, scale_a, add_bias=True)
        
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
            self.reset_cache()
            return self._forward_first_step(x)

        # Use fused direct conv kernel (eliminates im2col, fuses all ops)
        if self._can_use_fused_conv():
            conv_output = conv2d_w8a8_3x3_fused(
                x, a_hat_prev, self.weight_int8, o_hat_prev,
                self.weight_scale, bias=None
            )
            # Update caches (fused kernel doesn't return a_hat, so compute it)
            residual = x - a_hat_prev
            scale_a = residual.abs().max() / 127.0
            scale_a = torch.clamp(scale_a, min=1e-8)
            residual_int = torch.round(residual / scale_a).clamp(-128, 127)
            a_hat_new = residual_int * scale_a + a_hat_prev
            self._last_act_scale = scale_a
        else:
            # Fallback: separate quantization + conv
            # Eq. (ec5): quantize residual with optional scale reuse
            reuse_scale = self.config.reuse_act_scale and (self._last_act_scale is not None)
            residual_int, a_hat_new, scale_a = modulated_quantize_int8(
                x, a_hat_prev, scale=self._last_act_scale if reuse_scale else None
            )
            self._last_act_scale = scale_a
            
            # Conv WITHOUT bias
            if torch.cuda.is_available() and self._can_use_triton_conv():
                conv_output = conv2d_int8_triton_accumulate(
                    residual_int, self.weight_int8, o_hat_prev,
                    scale_a, self.weight_scale, bias=None
                )
            else:
                conv_output = self._conv2d_int8_im2col_gemm(residual_int, scale_a, add_bias=False)
                conv_output = conv_output + o_hat_prev

        # Update activation cache
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat_new.half()
        else:
            self.a_hat_cache = a_hat_new
        
        # Update output cache
        if self.config.store_cache_fp16:
            self.o_hat_cache = conv_output.half()
        else:
            self.o_hat_cache = conv_output.clone()
        
        return conv_output
    
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
        
        # Store packed weights (INT4 two per byte) and scale per channel
        k_total = (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1]
        self.register_buffer(
            "weight_packed",
            torch.empty(out_channels, (k_total + 1) // 2, dtype=torch.int8)
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
        self._last_act_scale: Optional[torch.Tensor] = None
    
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
        
        # Quantize and pack weights to INT4 (per-channel)
        weight = conv.weight.data.float()  # [out, in/groups, kH, kW]
        oc, icg, kH, kW = weight.shape
        weight_flat = weight.view(oc, -1)  # [out, icg*kH*kW]
        weight_max = weight_flat.abs().max(dim=1).values
        weight_scale = torch.clamp(weight_max / 7.0, min=1e-8)

        weight_int = torch.round(weight_flat / weight_scale.unsqueeze(1)).clamp(-8, 7).to(torch.int8)  # [out, K]
        # Pack along K dimension (two int4 per byte)
        if weight_int.shape[1] % 2 != 0:
            weight_int = torch.nn.functional.pad(weight_int, (0, 1), value=0)
        lo = (weight_int[:, 0::2] + 8).to(torch.int8)
        hi = (weight_int[:, 1::2] + 8).to(torch.int8)
        packed = (lo & 0xF) | ((hi & 0xF) << 4)  # [out, K//2]

        q_conv.weight_packed.copy_(packed.contiguous())
        q_conv.weight_scale.copy_(weight_scale)
        
        if conv.bias is not None:
            q_conv.bias.copy_(conv.bias.data.float())
        
        return q_conv
    
    def reset_cache(self):
        self.a_hat_cache = None
        self.o_hat_cache = None
        self.is_first_step = True
        self._last_act_scale = None
    
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

    def _pack_int4_pairs(self, mat_int: torch.Tensor) -> torch.Tensor:
        """Pack 2D int8 matrix with values in [-8,7] into int4-packed int8."""
        if mat_int.numel() % 2 != 0:
            mat_int = torch.nn.functional.pad(mat_int, (0, 1), value=0)
        lo = (mat_int[:, 0::2] + 8).to(torch.int8)
        hi = (mat_int[:, 1::2] + 8).to(torch.int8)
        packed = (lo & 0xF) | ((hi & 0xF) << 4)
        return packed

    def _conv2d_int4_im2col_gemm(
        self,
        x_int: torch.Tensor,
        scale_a: torch.Tensor,
        add_bias: bool,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """INT4 conv via unfold + packed INT4 GEMM (integer matmul)."""
        N, C, H, W = x_int.shape
        kH, kW = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation

        # Unfold integer activations
        cols_int = torch.nn.functional.unfold(
            x_int,
            kernel_size=(kH, kW),
            dilation=(dil_h, dil_w),
            padding=(pad_h, pad_w),
            stride=(stride_h, stride_w),
        )  # [N, K, L] with int8 values in [-8,7]

        # Pack activations: [N*L, K] -> [N*L, K/2]
        N_, K_, L_ = cols_int.shape
        cols_2d = cols_int.transpose(1, 2).contiguous().view(-1, K_)  # [N*L, K]
        # If K is odd, pad activation columns to match packed width and weight packing
        if K_ % 2 != 0:
            cols_2d = torch.nn.functional.pad(cols_2d, (0, 1), value=0)
            K_even = K_ + 1
        else:
            K_even = K_
        cols_packed = self._pack_int4_pairs(cols_2d)

        # Prepare packed weights: stored as [out, K/2], need [K/2, out]
        weight_packed = self.weight_packed.t().contiguous()

        if cache is not None:
            out_mat = gemm_w4a4_accum(
                cols_packed,
                weight_packed,
                scale_a,
                self.weight_scale,
                in_features=K_even,
                cache=cache.view(-1, self.out_channels),
            )
        else:
            out_mat = gemm_w4a4(
                cols_packed,
                weight_packed,
                scale_a,
                self.weight_scale,
                in_features=K_even,
                bias=self.bias if add_bias else None,
            )  # [N*L, out]

        # Reshape back
        H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
        out = out_mat.view(N, L_, self.out_channels).transpose(1, 2).contiguous()
        out = out.view(N, self.out_channels, H_out, W_out)
        return out
    
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
        self._last_act_scale = scale_a

        # Cache dequantized activation
        a_hat = x_int.float() * scale_a
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat.half()
        else:
            self.a_hat_cache = a_hat
        
        output = self._conv2d_int4_im2col_gemm(x_int, scale_a, add_bias=True)
        
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        a_hat_prev = self.a_hat_cache.float() if self.a_hat_cache.dtype == torch.float16 else self.a_hat_cache
        o_hat_prev = self.o_hat_cache.float() if self.o_hat_cache.dtype == torch.float16 else self.o_hat_cache
        
        if a_hat_prev.shape != x.shape:
            self.reset_cache()
            return self._forward_first_step(x)
        
        # Residual quantization (modulated) with optional scale reuse
        reuse_scale = self.config.reuse_act_scale and (self._last_act_scale is not None)
        scale_a, _ = compute_dynamic_scale_int4((x - a_hat_prev).flatten(), symmetric=True) if not reuse_scale else (self._last_act_scale, None)
        residual_int = torch.round((x - a_hat_prev) / scale_a).clamp(-8, 7).to(torch.int8)
        self._last_act_scale = scale_a
        
        residual_dequant = residual_int.float() * scale_a
        a_hat_new = residual_dequant + a_hat_prev
        
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat_new.half()
        else:
            self.a_hat_cache = a_hat_new
        
        conv_output = self._conv2d_int4_im2col_gemm(residual_int, scale_a, add_bias=False, cache=o_hat_prev)
        output = conv_output
        
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
