"""
Optimized INT4 MoDiff with:
1. Per-channel weight quantization (4-bit signed: -8 to 7)
2. Dynamic activation quantization with calibration support
3. MoDiff temporal caching for error-compensated modulation

INT4 Challenges:
- Only 16 discrete values (-8 to 7), requiring careful calibration
- 2x memory packing overhead (2 values per byte)
- Higher quantization noise than INT8

Expected performance:
- 2x theoretical speedup vs INT8 for large models (>1280 channels)
- Quality degradation without MoDiff compensation
- Critical to use dynamic quantization for residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modiff_cuda'))

try:
    import modiff_int4 as cutlass_int4
    HAS_CUTLASS = True
    print("Warning: INT4 kernels are loaded but not fully implemented.")
    print("INT4 convolution uses FP16 fallback. Full CUTLASS INT4 Tensor Core support is TODO.")
except ImportError:
    HAS_CUTLASS = False
    print("Warning: CUTLASS INT4 not available")


class CalibrationConfig:
    """
    Stores pre-calibrated quantization scales for static quantization.
    
    INT4 requires more careful calibration than INT8 due to limited range.
    Uses percentile-based clipping to handle outliers.
    """
    
    def __init__(self):
        self.scales: Dict[str, float] = {}
        self.running_max: Dict[str, float] = {}
        self.calibration_count = 0
        self.is_calibrated = False
        self.momentum = 0.99
        self.percentile = 99.99  # Clip outliers for better INT4 range usage
    
    def update(self, name: str, tensor: torch.Tensor):
        """Update running max using percentile for robustness."""
        with torch.no_grad():
            # Use 99.99 percentile instead of max to handle outliers
            abs_vals = tensor.abs().flatten()
            if abs_vals.numel() > 100:
                percentile_val = torch.quantile(abs_vals, self.percentile / 100.0).item()
            else:
                percentile_val = abs_vals.max().item()
            
            if name in self.running_max:
                old_max = self.running_max[name]
                self.running_max[name] = max(
                    self.momentum * old_max + (1 - self.momentum) * percentile_val,
                    percentile_val
                )
            else:
                self.running_max[name] = percentile_val
    
    def finalize(self):
        """Convert running max to scales (max/7 for INT4 range -8 to 7)."""
        for name, max_val in self.running_max.items():
            if max_val < 1e-8:
                max_val = 1.0
            self.scales[name] = max_val / 7.0  # INT4 symmetric range
        self.is_calibrated = True
        print(f"INT4 Calibration complete: {len(self.scales)} layers")
    
    def get_scale(self, name: str) -> float:
        return self.scales.get(name, 1.0 / 7.0)
    
    def get_inv_scale(self, name: str) -> float:
        scale = self.get_scale(name)
        return 1.0 / (scale + 1e-8)
    
    def save(self, path: str):
        torch.save({
            'scales': self.scales,
            'is_calibrated': self.is_calibrated
        }, path)
        print(f"Saved INT4 calibration to {path}")
    
    def load(self, path: str):
        data = torch.load(path)
        self.scales = data['scales']
        self.is_calibrated = data['is_calibrated']
        print(f"Loaded INT4 calibration from {path}: {len(self.scales)} layers")


# Global calibration config
_calib_config = CalibrationConfig()


def get_calibration_config() -> CalibrationConfig:
    return _calib_config


def reset_calibration():
    global _calib_config
    _calib_config = CalibrationConfig()


class OptimizedInt4Conv2d(nn.Module):
    """
    Optimized INT4 Conv2d with:
    1. Per-channel weight quantization (4-bit)
    2. Dynamic/static activation quantization
    3. MoDiff temporal caching for quality preservation
    
    INT4 quantization:
    - Weights: Per-channel, symmetric, range [-8, 7]
    - Activations: Per-tensor, dynamic (with optional calibration)
    - MoDiff residuals: Always dynamic to preserve precision
    """
    
    def __init__(self, conv: nn.Conv2d, layer_name: str = ""):
        super().__init__()
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        self.stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
        self.padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
        self.dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation
        self.groups = conv.groups
        
        # Quantize weights to INT4
        if HAS_CUTLASS and self.groups == 1:
            w_fp32 = conv.weight.data.float()
            if w_fp32.device.type != 'cuda':
                w_fp32 = w_fp32.cuda()
            weight_int4, weight_scales = cutlass_int4.quantize_weight(w_fp32)
            self.register_buffer('weight_int4', weight_int4)
            self.register_buffer('weight_scales', weight_scales)
        else:
            self.register_buffer('weight_int4', None)
            self.register_buffer('weight_scales', None)
        
        # FP16 weight for fallback
        self.register_buffer('weight_fp16', conv.weight.data.half())
        
        # Bias
        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.float())
        else:
            self.register_buffer('bias', torch.empty(0))
        
        # MoDiff state
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        
        # Calibration state
        self.calibrating = False
    
    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()
    
    def reset_state(self):
        self.is_first_step = True
        if self.o_hat_cache is not None:
            self.o_hat_cache.zero_()
    
    def set_calibrating(self, calibrating: bool):
        self.calibrating = calibrating
    
    def _should_use_int4(self, C: int, H: int, W: int) -> bool:
        """
        INT4 has even higher overhead than INT8 due to packing.
        Only beneficial for very large layers.
        
        INT4 now uses INT8 CUTLASS path via unpacking.
        Conservative heuristics to avoid overhead on small layers.
        """
        if not HAS_CUTLASS or self.weight_int4 is None:
            return False
        if self.groups != 1:
            return False
        
        max_ch = max(C, self.out_channels)
        min_spatial = min(H, W)
        
        # INT4 for medium-large layers (384+ channels)
        # More conservative than INT8 due to unpacking overhead
        return (max_ch >= 512 and min_spatial >= 8) or (max_ch >= 256 and min_spatial >= 16)
    
    def _forward_int4_static(self, x: torch.Tensor) -> torch.Tensor:
        """INT4 forward with static quantization scale."""
        config = get_calibration_config()
        scale = config.get_scale(self.layer_name)
        
        bias = self.bias if self.bias.numel() > 0 else torch.empty(0, device=x.device)
        out = cutlass_int4.conv2d_int4_static(
            x, self.weight_int4, self.weight_scales, bias,
            scale,
            self.stride, self.stride,
            self.padding, self.padding,
            self.dilation, self.dilation
        )
        return out
    
    def _forward_int4_dynamic(self, x: torch.Tensor) -> torch.Tensor:
        """INT4 forward with dynamic quantization."""
        if self.calibrating:
            get_calibration_config().update(self.layer_name, x)
        
        bias = self.bias if self.bias.numel() > 0 else torch.empty(0, device=x.device)
        return cutlass_int4.conv2d_int4(
            x, self.weight_int4, self.weight_scales, bias,
            self.stride, self.stride,
            self.padding, self.padding,
            self.dilation, self.dilation
        )
    
    def _forward_fp16(self, x: torch.Tensor) -> torch.Tensor:
        """FP16 forward (fallback)."""
        x_fp16 = x.half()
        bias = self.bias.half() if self.bias.numel() > 0 else None
        out = F.conv2d(x_fp16, self.weight_fp16, bias,
                      self.stride, self.padding, self.dilation, self.groups)
        return out.float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic precision selection."""
        N, C, H, W = x.shape
        
        use_int4 = self._should_use_int4(C, H, W)
        
        if not self.modiff_enabled:
            # Standard forward
            if use_int4:
                config = get_calibration_config()
                if config.is_calibrated and not self.calibrating:
                    return self._forward_int4_static(x)
                else:
                    return self._forward_int4_dynamic(x)
            else:
                return self._forward_fp16(x)
        
        # MoDiff forward with temporal caching
        if self.is_first_step:
            # First step: full computation
            if use_int4:
                config = get_calibration_config()
                if config.is_calibrated and not self.calibrating:
                    out = self._forward_int4_static(x)
                else:
                    out = self._forward_int4_dynamic(x)
            else:
                out = self._forward_fp16(x)
            
            # Initialize caches
            if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
                self.a_hat_cache = x.clone()
                self.o_hat_cache = out.clone()
            else:
                self.a_hat_cache.copy_(x)
                self.o_hat_cache.copy_(out)
            
            self.is_first_step = False
            return out
        else:
            # Subsequent steps: incremental update
            residual = x - self.a_hat_cache
            self.a_hat_cache.copy_(x)
            
            # CRITICAL: Always use dynamic quantization for residuals in INT4
            # Residual has much smaller range, static scale would destroy it
            if use_int4:
                bias_empty = torch.empty(0, device=x.device)
                # Force dynamic quantization for residual
                if self.calibrating:
                    get_calibration_config().update(self.layer_name, residual)
                conv_residual = cutlass_int4.conv2d_int4(
                    residual, self.weight_int4, self.weight_scales, bias_empty,
                    self.stride, self.stride,
                    self.padding, self.padding,
                    self.dilation, self.dilation
                )
            else:
                residual_fp16 = residual.half()
                conv_residual = F.conv2d(residual_fp16, self.weight_fp16, None,
                                        self.stride, self.padding, self.dilation, self.groups).float()
            
            # Update cache
            self.o_hat_cache.add_(conv_residual)
            return self.o_hat_cache.clone()


def convert_model_to_optimized_int4(model: nn.Module, prefix: str = "") -> nn.Module:
    """Convert all Conv2d layers to OptimizedInt4Conv2d."""
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Conv2d):
            optimized_conv = OptimizedInt4Conv2d(child, layer_name=full_name)
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int4(child, prefix=full_name)
    
    return model


def enable_modiff_mode(model: nn.Module, enabled: bool = True):
    """Enable/disable MoDiff mode for all OptimizedInt4Conv2d layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            module.enable_modiff(enabled)


def reset_modiff_state(model: nn.Module):
    """Reset MoDiff state for all OptimizedInt4Conv2d layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            module.reset_state()


def set_calibrating(model: nn.Module, calibrating: bool):
    """Set calibration mode for all OptimizedInt4Conv2d layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            module.set_calibrating(calibrating)
