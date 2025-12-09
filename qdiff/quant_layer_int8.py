"""
INT8 Quantization Layer for MoDiff - Clean Implementation Based on INT4

This module implements INT8 (8-bit) quantization for the MoDiff framework,
following the paper's MSE-based calibration methodology. This is a cleaner
implementation derived from the INT4 code but with native TensorRT INT8 support.

Key Features:
1. 256-level symmetric/asymmetric quantization
2. MSE-based scale calibration (matching paper methodology)
3. Native TensorRT INT8 support (no proxy hack needed)
4. MoDiff modulation support (quantize residuals)

Paper: "Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization"
"""

import logging
from typing import Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class UniformAffineQuantizerINT8(nn.Module):
    """
    INT8 Uniform Affine Quantizer following MoDiff paper methodology.
    
    Supports:
    - 8-bit quantization (256 levels: -128 to 127 for signed, 0-255 for unsigned)
    - MSE-based scale search (matching paper's calibration)
    - Dynamic per-tensor and static per-channel quantization
    - MoDiff modulation (quantize residuals)
    
    Advantage over INT4:
    - 256 quantization levels vs 16
    - Native TensorRT INT8 support
    - Better accuracy with faster calibration
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = True,
        channel_wise: bool = False,
        scale_method: str = 'mse',
        leaf_param: bool = False,
        prob: float = 1.0,
        dynamic: bool = False,
    ):
        super().__init__()
        assert n_bits == 8, "INT8 quantizer only supports 8 bits"
        
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits  # 256 for 8-bit
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.leaf_param = leaf_param
        self.prob = prob
        self.dynamic = dynamic
        
        # Quantization bounds
        if symmetric:
            self.q_min = -(2 ** (n_bits - 1))  # -128 for 8-bit
            self.q_max = 2 ** (n_bits - 1) - 1  # 127 for 8-bit
        else:
            self.q_min = 0
            self.q_max = 2 ** n_bits - 1  # 255 for 8-bit
        
        # Scale and zero point parameters
        self.register_buffer('delta', None)
        self.register_buffer('zero_point', None)
        self.register_buffer('min_val', None)
        self.register_buffer('max_val', None)
        
        # For running statistics
        self.running_stat = False
        self.register_buffer('running_min', None)
        self.register_buffer('running_max', None)
        
        self.inited = False
        
    def set_inited(self, inited: bool = True):
        self.inited = inited
        
    def update_quantize_range(self, x: torch.Tensor) -> None:
        """Update running min/max statistics for calibration."""
        if self.running_min is None:
            self.running_min = x.min().detach()
            self.running_max = x.max().detach()
        else:
            self.running_min = 0.9 * self.running_min + 0.1 * x.min().detach()
            self.running_max = 0.9 * self.running_max + 0.1 * x.max().detach()
            
    def init_quantization_params(self, x: torch.Tensor, channel_wise: bool = False) -> None:
        """
        Initialize quantization parameters using MSE-based scale search.
        
        This follows the paper's methodology of using MSE (not entropy) for calibration.
        """
        if self.scale_method == 'mse':
            self.delta, self.zero_point = self._mse_scale_search(x, channel_wise)
        elif self.scale_method == 'max':
            self.delta, self.zero_point = self._max_scale_init(x, channel_wise)
        elif self.scale_method == 'minmax':
            self.delta, self.zero_point = self._minmax_scale_init(x, channel_wise)
        else:
            raise ValueError(f"Unknown scale method: {self.scale_method}")
        
        self.inited = True
        
    def _mse_scale_search(
        self, 
        x: torch.Tensor, 
        channel_wise: bool = False,
        num_candidates: int = 80
    ) -> tuple:
        """
        MSE-based scale search following MoDiff paper methodology.
        
        For INT8, we can use fewer candidates than INT4 since the
        256 quantization levels provide more tolerance.
        
        Args:
            x: Input tensor to quantize
            channel_wise: Whether to compute per-channel scales
            num_candidates: Number of scale candidates to search
            
        Returns:
            (delta, zero_point) tuple
        """
        x_flat = x.detach().clone()
        
        if channel_wise:
            # Per-channel: reshape to (channels, -1)
            n_channels = x.shape[0] if len(x.shape) > 0 else 1
            x_flat = x.reshape(n_channels, -1)
            
            best_delta = torch.zeros(n_channels, device=x.device, dtype=x.dtype)
            best_zp = torch.zeros(n_channels, device=x.device, dtype=x.dtype)
            
            for c in range(n_channels):
                delta, zp = self._mse_search_1d(x_flat[c], num_candidates)
                best_delta[c] = delta
                best_zp[c] = zp
                
            # Reshape for broadcasting
            shape = [1] * len(x.shape)
            shape[0] = n_channels
            best_delta = best_delta.reshape(shape)
            best_zp = best_zp.reshape(shape)
        else:
            # Per-tensor
            best_delta, best_zp = self._mse_search_1d(x_flat.flatten(), num_candidates)
            
        return best_delta, best_zp
    
    def _mse_search_1d(self, x: torch.Tensor, num_candidates: int = 80) -> tuple:
        """
        1D MSE search for optimal scale and zero point.
        
        INT8's 256 levels allow for faster convergence than INT4.
        """
        x_min, x_max = x.min(), x.max()
        
        if self.symmetric:
            # Symmetric: zero_point = 0, search for optimal scale
            x_absmax = torch.max(x_min.abs(), x_max.abs())
            
            if x_absmax < 1e-8:
                return torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device)
            
            # Search from 0.8x to 1.1x of the max-based scale (tighter for INT8)
            base_delta = x_absmax / self.q_max
            candidates = torch.linspace(0.8, 1.1, num_candidates, device=x.device) * base_delta
            
            best_mse = float('inf')
            best_delta = base_delta
            
            for delta in candidates:
                x_q = torch.clamp(torch.round(x / delta), self.q_min, self.q_max)
                x_dq = x_q * delta
                mse = ((x - x_dq) ** 2).mean().item()
                
                if mse < best_mse:
                    best_mse = mse
                    best_delta = delta
                    
            return best_delta, torch.tensor(0.0, device=x.device)
        else:
            # Asymmetric: search for both scale and zero point
            if x_max - x_min < 1e-8:
                return torch.tensor(1.0, device=x.device), x_min
                
            base_delta = (x_max - x_min) / (self.n_levels - 1)
            
            best_mse = float('inf')
            best_delta = base_delta
            best_zp = x_min
            
            # Grid search over delta and zero point
            for d_mult in torch.linspace(0.9, 1.1, 15, device=x.device):
                delta = base_delta * d_mult
                for zp_shift in torch.linspace(-0.05, 0.05, 8, device=x.device):
                    zp = x_min + zp_shift * (x_max - x_min)
                    
                    x_q = torch.clamp(torch.round((x - zp) / delta), 0, self.n_levels - 1)
                    x_dq = x_q * delta + zp
                    mse = ((x - x_dq) ** 2).mean().item()
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_delta = delta
                        best_zp = zp
                        
            return best_delta, best_zp
    
    def _max_scale_init(self, x: torch.Tensor, channel_wise: bool = False) -> tuple:
        """Max-based scale initialization."""
        if channel_wise:
            n_channels = x.shape[0]
            x_flat = x.reshape(n_channels, -1)
            x_max = x_flat.abs().max(dim=1)[0]
            
            shape = [1] * len(x.shape)
            shape[0] = n_channels
            
            delta = x_max / self.q_max
            delta = delta.reshape(shape)
            zero_point = torch.zeros_like(delta)
        else:
            x_max = x.abs().max()
            delta = x_max / self.q_max
            zero_point = torch.tensor(0.0, device=x.device)
            
        return delta, zero_point
    
    def _minmax_scale_init(self, x: torch.Tensor, channel_wise: bool = False) -> tuple:
        """Min-max scale initialization."""
        if channel_wise:
            n_channels = x.shape[0]
            x_flat = x.reshape(n_channels, -1)
            x_min = x_flat.min(dim=1)[0]
            x_max = x_flat.max(dim=1)[0]
            
            shape = [1] * len(x.shape)
            shape[0] = n_channels
            
            delta = (x_max - x_min) / (self.n_levels - 1)
            delta = delta.reshape(shape)
            zero_point = x_min.reshape(shape)
        else:
            x_min, x_max = x.min(), x.max()
            delta = (x_max - x_min) / (self.n_levels - 1)
            zero_point = x_min
            
        return delta, zero_point
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to INT8 range."""
        if self.delta is None:
            self.init_quantization_params(x, self.channel_wise)
            
        if self.symmetric:
            x_int = torch.round(x / self.delta)
        else:
            x_int = torch.round((x - self.zero_point) / self.delta)
            
        x_int = torch.clamp(x_int, self.q_min, self.q_max)
        return x_int.to(torch.int8)
    
    def dequantize(self, x_int: torch.Tensor, delta=None, zero_point=None) -> torch.Tensor:
        """Dequantize INT8 tensor back to float."""
        delta = delta if delta is not None else self.delta
        zero_point = zero_point if zero_point is not None else self.zero_point
        
        if self.symmetric:
            return x_int.float() * delta
        else:
            return x_int.float() * delta + zero_point
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fake quantization.
        
        Uses STE (Straight-Through Estimator) for gradient computation.
        """
        if self.running_stat:
            self.update_quantize_range(x)
            return x
            
        if not self.inited:
            if self.leaf_param:
                self.delta = nn.Parameter(torch.tensor(1.0, device=x.device))
                self.zero_point = nn.Parameter(torch.tensor(0.0, device=x.device))
            self.init_quantization_params(x, self.channel_wise)
            
        if self.dynamic:
            # Recompute scale on each forward pass
            self.init_quantization_params(x, self.channel_wise)
            
        # Fake quantization: quantize and immediately dequantize
        x_int = self.quantize(x)
        x_dq = self.dequantize(x_int)
        
        # STE: use dequantized for forward, but gradient flows through original x
        return x + (x_dq - x).detach()
    
    def get_dynamic_range(self) -> float:
        """
        Get dynamic range for TensorRT.
        
        TensorRT uses dynamic range = scale * q_max for INT8 calibration.
        """
        if self.delta is None:
            return 1.0
        
        if isinstance(self.delta, torch.Tensor):
            scale = self.delta.item() if self.delta.numel() == 1 else self.delta.max().item()
        else:
            scale = self.delta
            
        return scale * self.q_max


class QuantModuleINT8(nn.Module):
    """
    INT8 Quantized Module for MoDiff following paper methodology.
    
    Implements:
    1. INT8 weight quantization
    2. INT8 activation quantization (fake-quant)
    3. MoDiff modulation (quantize residuals)
    4. Error compensation mechanism
    
    The module wraps Conv2d/Linear layers with INT8 quantization.
    """
    
    def __init__(
        self,
        org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d],
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_act_quant: bool = False,
        act_quant_mode: str = 'qdiff',
        modulate: bool = False,
    ):
        super().__init__()
        
        # Determine layer type
        if isinstance(org_module, nn.Conv2d):
            self.fwd_func = F.conv2d
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_func = F.conv1d
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
        elif isinstance(org_module, nn.Linear):
            self.fwd_func = F.linear
            self.fwd_kwargs = dict()
        else:
            raise ValueError(f"Unsupported module type: {type(org_module)}")
        
        # Store original weights
        self.register_buffer('org_weight', org_module.weight.data.clone())
        if org_module.bias is not None:
            self.register_buffer('org_bias', org_module.bias.data.clone())
        else:
            self.org_bias = None
            
        # Working weight/bias (can be quantized)
        self.weight = nn.Parameter(org_module.weight.data.clone())
        self.bias = nn.Parameter(org_module.bias.data.clone()) if org_module.bias is not None else None
        
        # Quantization parameters - default to 8-bit
        weight_quant_params = weight_quant_params.copy()
        weight_quant_params.setdefault('n_bits', 8)
        weight_quant_params.setdefault('symmetric', True)
        weight_quant_params.setdefault('channel_wise', True)
        weight_quant_params.setdefault('scale_method', 'mse')
        
        act_quant_params = act_quant_params.copy()
        act_quant_params.setdefault('n_bits', 8)
        act_quant_params.setdefault('symmetric', True)
        act_quant_params.setdefault('channel_wise', False)
        act_quant_params.setdefault('scale_method', 'mse')
        
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        
        # Create quantizers
        self.weight_quantizer = UniformAffineQuantizerINT8(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizerINT8(**act_quant_params)
        
        # Split quantization support (for skip connections)
        self.split = 0
        self.weight_quantizer_0 = None
        self.act_quantizer_0 = None
        
        # Quantization state
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        
        # INT8 storage
        self.register_buffer('weight_int8', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        self.weight_original_shape = None
        
        # Activation function passthrough
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.extra_repr = org_module.extra_repr
        
        # MoDiff modulation support
        self.modulate = modulate
        self.a_hat = None  # Cached activation for modulation
        self.o_hat = None  # Cached output for modulation
        
    def quantize_weights_to_int8(self) -> None:
        """Quantize weights to INT8 format."""
        if self.weight_int8 is not None:
            return  # Already quantized
            
        # Initialize quantizer if needed
        if not self.weight_quantizer.inited:
            self.weight_quantizer.init_quantization_params(
                self.weight.data, 
                self.weight_quant_params.get('channel_wise', True)
            )
            
        # Quantize to INT8
        self.weight_int8 = self.weight_quantizer.quantize(self.weight.data)
        
        # Store scale and zero point
        self.weight_scale = self.weight_quantizer.delta.clone()
        self.weight_zero_point = (
            self.weight_quantizer.zero_point.clone() 
            if self.weight_quantizer.zero_point is not None 
            else torch.tensor(0.0)
        )
        self.weight_original_shape = self.weight.shape
        
        logger.debug(f"Quantized weights to INT8: {self.weight_original_shape}")
        
    def dequantize_weights_from_int8(self) -> torch.Tensor:
        """Dequantize INT8 weights to float."""
        if self.weight_int8 is None:
            return self.weight if self.weight is not None else self.org_weight
            
        # Dequantize
        weight_fp = self.weight_quantizer.dequantize(
            self.weight_int8, 
            self.weight_scale, 
            self.weight_zero_point
        )
        
        return weight_fp
        
    def forward(self, input: torch.Tensor, split: int = 0) -> torch.Tensor:
        """
        Forward pass with INT8 quantization and MoDiff modulation.
        """
        # Fast path: no quantization
        if not self.use_weight_quant and not self.use_act_quant:
            weight = self.org_weight if self.weight is None else self.weight
            bias = self.org_bias if self.org_bias is not None else self.bias
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
            return self.activation_function(out)
        
        # Handle split quantization
        if split != 0:
            if self.split != 0:
                assert split == self.split
            else:
                logger.info(f"Split at {split}!")
                self.split = split
                self.set_split()
        
        # === Activation quantization with MoDiff modulation ===
        if not self.disable_act_quant and self.use_act_quant:
            if self.modulate:
                # MoDiff: quantize residual activations (key paper innovation!)
                if self.a_hat is None or self.a_hat.shape != input.shape:
                    # First step or shape change: initialize cache
                    self.a_hat = input.clone().detach()
                else:
                    # Quantize the residual (delta) instead of absolute value
                    residual = input - self.a_hat
                    
                    if self.split != 0:
                        res_0 = self.act_quantizer(residual[:, :self.split, ...])
                        res_1 = self.act_quantizer_0(residual[:, self.split:, ...])
                        residual_q = torch.cat([res_0, res_1], dim=1)
                    else:
                        residual_q = self.act_quantizer(residual)
                    
                    # Update cache with quantized residual (error compensation)
                    self.a_hat = (self.a_hat + residual_q).clone().detach()
                    input = residual_q
            else:
                # Standard activation quantization
                if self.split != 0:
                    input_0 = self.act_quantizer(input[:, :self.split, ...])
                    input_1 = self.act_quantizer_0(input[:, self.split:, ...])
                    input = torch.cat([input_0, input_1], dim=1)
                else:
                    input = self.act_quantizer(input)
        
        # === Weight quantization ===
        if self.use_weight_quant:
            if self.weight_int8 is None:
                self.quantize_weights_to_int8()
            weight = self.dequantize_weights_from_int8()
        else:
            weight = self.org_weight if self.weight is None else self.weight
        
        # === Forward computation ===
        bias = self.org_bias if self.org_bias is not None else self.bias
        
        if self.modulate and self.use_act_quant:
            # MoDiff output modulation
            if self.o_hat is None:
                out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
            else:
                # Error compensation: add cached output
                out = self.fwd_func(input, weight, None, **self.fwd_kwargs)
                out = self.o_hat + out
            self.o_hat = out.clone().detach()
        else:
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        return self.activation_function(out)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """Enable/disable quantization."""
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        
        if weight_quant and self.weight_int8 is None:
            self.quantize_weights_to_int8()
            
    def set_dynamic_state(self, dynamic: bool) -> None:
        """Enable/disable dynamic activation quantization."""
        self.act_quantizer.dynamic = dynamic
        if self.act_quantizer_0 is not None:
            self.act_quantizer_0.dynamic = dynamic
            
    def set_modulation(self, modulation: bool) -> None:
        """Enable/disable MoDiff modulation."""
        self.modulate = modulation
        
    def reset_cache(self) -> None:
        """Reset cached activations and outputs."""
        self.a_hat = None
        self.o_hat = None
        
    def set_split(self) -> None:
        """Initialize split quantizers for skip connections."""
        self.weight_quantizer_0 = UniformAffineQuantizerINT8(**self.weight_quant_params)
        self.act_quantizer_0 = UniformAffineQuantizerINT8(**self.act_quant_params)
        
    def set_running_stat(self, running_stat: bool) -> None:
        """Enable/disable running statistics for calibration."""
        self.act_quantizer.running_stat = running_stat
        if self.act_quantizer_0 is not None:
            self.act_quantizer_0.running_stat = running_stat
            
    def get_scales(self) -> dict:
        """Get quantization scales for TensorRT export."""
        scales = {
            'weight_scale': self.weight_scale.cpu().numpy() if self.weight_scale is not None else None,
            'weight_zero_point': self.weight_zero_point.cpu().numpy() if self.weight_zero_point is not None else None,
            'act_scale': self.act_quantizer.delta.cpu().numpy() if self.act_quantizer.delta is not None else None,
            'act_zero_point': self.act_quantizer.zero_point.cpu().numpy() if self.act_quantizer.zero_point is not None else None,
            'act_dynamic_range': self.act_quantizer.get_dynamic_range(),
            'n_bits': 8,
        }
        return scales


class StraightThrough(nn.Module):
    """Identity activation for quantized layers."""
    def forward(self, x):
        return x


def convert_to_int8_module(
    module: nn.Module,
    weight_quant_params: dict = {},
    act_quant_params: dict = {},
    modulate: bool = False,
    name: str = "",
) -> nn.Module:
    """
    Recursively convert nn.Conv2d/Linear layers to QuantModuleINT8.
    
    Args:
        module: Module to convert
        weight_quant_params: Weight quantization parameters
        act_quant_params: Activation quantization parameters
        modulate: Enable MoDiff modulation
        name: Module name for logging
        
    Returns:
        Converted module with INT8 quantization
    """
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        
        if isinstance(child, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            # Convert to quantized module
            quant_module = QuantModuleINT8(
                child,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params,
                modulate=modulate,
            )
            setattr(module, child_name, quant_module)
            logger.debug(f"Converted {full_name} to INT8 quantized layer")
        else:
            # Recurse
            convert_to_int8_module(child, weight_quant_params, act_quant_params, modulate, full_name)
            
    return module
