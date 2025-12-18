"""
MoDiff Quantized Linear Layers

This module implements the core MoDiff linear layers with error-compensated
modulation following the paper's equations (ec1-ec8).

Layer-level MoDiff Algorithm:
    
    At timestep T (first step):
        â_T = Q(a_T)                              -- Eq. (ec1)
        ô_T = A(â_T) + bias                       -- Eq. (ec2)
        
    At timestep t < T:
        â_t = Q(a_t - â_{t+1}) + â_{t+1}          -- Eq. (ec5)
        ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}       -- Eq. (ec6)
        
Key insight from paper Section 3.3:
    "The residual should be computed based on â_{t} instead of a_{t},
     which will compensate the errors and avoid error accumulation."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .config import MoDiffConfig
from ..kernels.quantize import (
    quantize_symmetric_int8,
    quantize_symmetric_int4,
    compute_dynamic_scale_int8,
    compute_dynamic_scale_int4,
)
from ..kernels.modulated_quantize import (
    modulated_quantize_int8,
    modulated_quantize_int4,
    modulated_quantize_first_step_int8,
    modulated_quantize_first_step_int4,
)
from ..kernels.gemm_w8a8 import gemm_w8a8, gemm_w8a8_accum
from ..kernels.gemm_w4a4 import gemm_w4a4, gemm_w4a4_accum, pack_int4_weight


class W8A8MoDiffLinear(nn.Module):
    """
    W8A8 Linear layer with MoDiff error-compensated modulation.
    
    This implements layer-level MoDiff following the paper's framework:
        - INT8 weights (pre-quantized)
        - INT8 activations (dynamic quantization)
        - Error-compensated modulation across timesteps
        
    Buffers:
        weight_int8: Pre-quantized INT8 weights [out_features, in_features]
        weight_scale: Weight quantization scale [out_features] or scalar
        bias: FP16/FP32 bias [out_features]
        
    Caches (created at runtime):
        a_hat_cache: â_{t+1} - previous quantized activation [batch, in_features]
        o_hat_cache: ô_{t+1} - previous output [batch, out_features]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: MoDiffConfig = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or MoDiffConfig(weight_bits=8, act_bits=8)
        
        # Weight buffer (INT8)
        self.register_buffer(
            "weight_int8",
            torch.empty(out_features, in_features, dtype=torch.int8)
        )
        
        # Weight scale (per-channel or per-tensor)
        if self.config.weight_channel_wise:
            self.register_buffer(
                "weight_scale",
                torch.empty(out_features, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "weight_scale",
                torch.empty(1, dtype=torch.float32)
            )
        
        # Bias
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(out_features, dtype=torch.float32)
            )
        else:
            self.register_buffer("bias", None)
        
        # MoDiff caches (will be initialized on first forward)
        self.a_hat_cache: Optional[torch.Tensor] = None  # â_{t+1}
        self.o_hat_cache: Optional[torch.Tensor] = None  # ô_{t+1}
        
        # State tracking
        self.is_first_step = True
        self.modulation_enabled = self.config.modulation_enabled
        
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: MoDiffConfig = None,
    ) -> "W8A8MoDiffLinear":
        """
        Create W8A8MoDiffLinear from a pretrained nn.Linear.
        
        This performs weight quantization and stores the result.
        """
        config = config or MoDiffConfig(weight_bits=8, act_bits=8)
        
        q_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )
        
        # Quantize weights
        weight = linear.weight.data.float()
        
        if config.weight_channel_wise:
            # Per-channel quantization
            weight_max = weight.abs().max(dim=1).values
            weight_scale = weight_max / 127.0
            weight_scale = torch.clamp(weight_scale, min=1e-8)
            weight_int = torch.round(weight / weight_scale.unsqueeze(1)).clamp(-128, 127)
        else:
            # Per-tensor quantization
            weight_max = weight.abs().max()
            weight_scale = weight_max / 127.0
            weight_scale = torch.clamp(weight_scale, min=1e-8)
            weight_int = torch.round(weight / weight_scale).clamp(-128, 127)
        
        q_linear.weight_int8.copy_(weight_int.to(torch.int8))
        q_linear.weight_scale.copy_(weight_scale)
        
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data.float())
        
        return q_linear
    
    def reset_cache(self):
        """Reset MoDiff caches for new diffusion sequence."""
        self.a_hat_cache = None
        self.o_hat_cache = None
        self.is_first_step = True
    
    def set_modulation(self, enabled: bool):
        """Enable/disable MoDiff modulation."""
        self.modulation_enabled = enabled
        if not enabled:
            self.reset_cache()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MoDiff error-compensated modulation.
        
        Args:
            x: Input activation tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        x = x.view(-1, self.in_features)  # [M, K]
        M = x.shape[0]
        
        if not self.modulation_enabled:
            # Standard W8A8 forward (no modulation)
            return self._forward_standard(x).view(*batch_dims, self.out_features)
        
        if self.is_first_step:
            # First timestep (t=T): Eq. (ec1) and (ec2)
            output = self._forward_first_step(x)
            self.is_first_step = False
        else:
            # Subsequent timesteps (t<T): Eq. (ec5) and (ec6)
            output = self._forward_modulated(x)
        
        return output.view(*batch_dims, self.out_features)
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard W8A8 forward without modulation."""
        # Quantize activation
        x_int, scale_a = quantize_symmetric_int8(x)
        
        # Get weight scale (handle per-channel)
        if self.config.weight_channel_wise:
            # For per-channel weights, we need to handle differently
            # Use average scale for simplicity in GEMM
            scale_w = self.weight_scale.mean()
        else:
            scale_w = self.weight_scale
        
        # GEMM
        output = gemm_w8a8(
            x_int,
            self.weight_int8.t().contiguous(),
            scale_a,
            scale_w,
            self.bias,
        )
        
        return output
    
    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        First timestep (t=T) forward.
        
        Implements:
            â_T = Q(a_T)                  -- Eq. (ec1)
            ô_T = A(â_T) + bias           -- Eq. (ec2)
        """
        # Eq. (ec1): â_T = Q(a_T)
        x_int, a_hat, scale_a = modulated_quantize_first_step_int8(x)
        
        # Store cache for next timestep
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat.half()
        else:
            self.a_hat_cache = a_hat
        
        # Get weight scale
        if self.config.weight_channel_wise:
            scale_w = self.weight_scale.mean()
        else:
            scale_w = self.weight_scale
        
        # Eq. (ec2): ô_T = A(â_T) + bias
        output = gemm_w8a8(
            x_int,
            self.weight_int8.t().contiguous(),
            scale_a,
            scale_w,
            self.bias,
        )
        
        # Store output cache
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """
        Subsequent timesteps (t<T) forward with error compensation.
        
        Implements:
            â_t = Q(a_t - â_{t+1}) + â_{t+1}          -- Eq. (ec5)
            ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}       -- Eq. (ec6)
        """
        # Get cached values (convert back to float32 if stored as fp16)
        a_hat_prev = self.a_hat_cache.float() if self.a_hat_cache.dtype == torch.float16 else self.a_hat_cache
        o_hat_prev = self.o_hat_cache.float() if self.o_hat_cache.dtype == torch.float16 else self.o_hat_cache
        
        # Handle batch size changes
        if a_hat_prev.shape[0] != x.shape[0]:
            # Batch size changed, need to reset or broadcast
            # For simplicity, treat as first step if batch size differs
            return self._forward_first_step(x)
        
        # Eq. (ec5): â_t = Q(a_t - â_{t+1}) + â_{t+1}
        residual_int, a_hat_new, scale_a = modulated_quantize_int8(x, a_hat_prev)
        
        # Update cache
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat_new.half()
        else:
            self.a_hat_cache = a_hat_new
        
        # Get weight scale
        if self.config.weight_channel_wise:
            scale_w = self.weight_scale.mean()
        else:
            scale_w = self.weight_scale
        
        # Eq. (ec6): ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}
        # Note: NO bias added here (only at first timestep)
        output = gemm_w8a8_accum(
            residual_int,
            self.weight_int8.t().contiguous(),
            scale_a,
            scale_w,
            o_hat_prev,
        )
        
        # Update output cache
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits=W8A8, "
            f"modulation={self.modulation_enabled}"
        )


class W4A4MoDiffLinear(nn.Module):
    """
    W4A4 Linear layer with MoDiff error-compensated modulation.
    
    Uses INT4 quantization with packing (2 values per byte) for
    maximum compression. The paper shows MoDiff enables W4A4 to
    achieve near-W8A8 accuracy because residuals have smaller range.
    
    Buffers:
        weight_packed: Packed INT4 weights [out_features, in_features//2]
        weight_scale: Weight quantization scale
        bias: FP32 bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: MoDiffConfig = None,
    ):
        super().__init__()
        
        assert in_features % 2 == 0, "in_features must be even for INT4 packing"
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or MoDiffConfig(weight_bits=4, act_bits=4)
        
        # Packed weight buffer (INT4 packed into INT8)
        self.register_buffer(
            "weight_packed",
            torch.empty(out_features, in_features // 2, dtype=torch.int8)
        )
        
        # Weight scale
        self.register_buffer(
            "weight_scale",
            torch.empty(1, dtype=torch.float32)
        )
        
        # Bias
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(out_features, dtype=torch.float32)
            )
        else:
            self.register_buffer("bias", None)
        
        # MoDiff caches
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        
        self.is_first_step = True
        self.modulation_enabled = self.config.modulation_enabled
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: MoDiffConfig = None,
    ) -> "W4A4MoDiffLinear":
        """Create W4A4MoDiffLinear from a pretrained nn.Linear."""
        config = config or MoDiffConfig(weight_bits=4, act_bits=4)
        
        q_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )
        
        # Quantize and pack weights
        weight = linear.weight.data.float()  # [out, in]
        
        # Need to pack along in_features dimension
        # Transpose to [in, out], pack, then transpose back
        weight_t = weight.t().contiguous()  # [in, out]
        weight_packed, weight_scale = pack_int4_weight(weight_t)  # [in//2, out]
        weight_packed = weight_packed.t().contiguous()  # [out, in//2]
        
        q_linear.weight_packed.copy_(weight_packed)
        q_linear.weight_scale.copy_(weight_scale)
        
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data.float())
        
        return q_linear
    
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
        """Forward pass with W4A4 MoDiff."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        x = x.view(-1, self.in_features)
        
        if not self.modulation_enabled:
            return self._forward_standard(x).view(*batch_dims, self.out_features)
        
        if self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
        else:
            output = self._forward_modulated(x)
        
        return output.view(*batch_dims, self.out_features)
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard W4A4 forward."""
        # Quantize and pack activation
        x_packed, scale_a, _ = quantize_symmetric_int4(x)
        
        output = gemm_w4a4(
            x_packed.view(x.shape[0], -1),
            self.weight_packed.t().contiguous(),
            scale_a,
            self.weight_scale,
            self.in_features,
            self.bias,
        )
        
        return output
    
    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """First timestep forward for W4A4."""
        # Quantize activation
        x_packed, a_hat, scale_a, orig_shape = modulated_quantize_first_step_int4(x)
        
        # Store cache
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat.half()
        else:
            self.a_hat_cache = a_hat
        
        # GEMM
        output = gemm_w4a4(
            x_packed.view(x.shape[0], -1),
            self.weight_packed.t().contiguous(),
            scale_a,
            self.weight_scale,
            self.in_features,
            self.bias,
        )
        
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """Modulated forward for W4A4."""
        a_hat_prev = self.a_hat_cache.float() if self.a_hat_cache.dtype == torch.float16 else self.a_hat_cache
        o_hat_prev = self.o_hat_cache.float() if self.o_hat_cache.dtype == torch.float16 else self.o_hat_cache
        
        if a_hat_prev.shape[0] != x.shape[0]:
            return self._forward_first_step(x)
        
        # Modulated quantization
        residual_packed, a_hat_new, scale_a, _ = modulated_quantize_int4(x, a_hat_prev)
        
        if self.config.store_cache_fp16:
            self.a_hat_cache = a_hat_new.half()
        else:
            self.a_hat_cache = a_hat_new
        
        # GEMM with accumulation
        output = gemm_w4a4_accum(
            residual_packed.view(x.shape[0], -1),
            self.weight_packed.t().contiguous(),
            scale_a,
            self.weight_scale,
            self.in_features,
            o_hat_prev,
        )
        
        if self.config.store_cache_fp16:
            self.o_hat_cache = output.half()
        else:
            self.o_hat_cache = output.clone()
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits=W4A4, "
            f"modulation={self.modulation_enabled}"
        )
