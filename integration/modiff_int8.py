"""
W8A8 MoDiff implementation using CUTLASS INT8 kernels.
True INT8 weights and activations with per-channel weight scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add modiff_cuda to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modiff_cuda'))

try:
    import modiff_int8 as cutlass_int8
    HAS_CUTLASS_INT8 = True
    print("CUTLASS INT8 kernel loaded successfully")
except ImportError as e:
    print(f"Warning: CUTLASS INT8 kernel not available: {e}")
    print("Falling back to PyTorch FP16")
    HAS_CUTLASS_INT8 = False


class CutlassInt8Conv2d(nn.Module):
    """
    INT8 Convolution using CUTLASS Tensor Core kernels.
    
    Uses:
    - INT8 weights (per-channel quantization)
    - INT8 activations (per-tensor dynamic quantization)
    - INT32 accumulation
    - FP32 output after dequantization
    """
    
    def __init__(self, original_conv):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size[0] if isinstance(original_conv.kernel_size, tuple) else original_conv.kernel_size
        self.stride = original_conv.stride[0] if isinstance(original_conv.stride, tuple) else original_conv.stride
        self.padding = original_conv.padding[0] if isinstance(original_conv.padding, tuple) else original_conv.padding
        self.dilation = original_conv.dilation[0] if isinstance(original_conv.dilation, tuple) else original_conv.dilation
        self.groups = original_conv.groups
        
        # Quantize weights using CUTLASS utility
        w = original_conv.weight.data.float()
        if HAS_CUTLASS_INT8 and self.groups == 1:
            weight_int8, scales = cutlass_int8.quantize_weight(w.cuda())
            self.register_buffer('weight_int8', weight_int8)
            self.register_buffer('weight_scale', scales)
        else:
            # Fallback: keep FP32 weights
            self.register_buffer('weight_int8', None)
            self.register_buffer('weight_scale', None)
        
        # Keep FP32/FP16 weight for fallback
        self.register_buffer('weight_fp32', w)
        self.register_buffer('weight_fp16', w.half())
        
        # Bias
        if original_conv.bias is not None:
            self.register_buffer('bias', original_conv.bias.data.clone())
        else:
            self.register_buffer('bias', None)
        
        # MoDiff state
        self.use_int8 = HAS_CUTLASS_INT8 and self.groups == 1
        self.use_fp16 = True  # Always available
        self.enabled = False
        
    def enable_modiff(self, enabled=True):
        self.enabled = enabled
        
    def reset_state(self):
        pass  # No state to reset in this version
    
    def forward(self, x):
        if self.use_int8 and self.enabled:
            # Use CUTLASS INT8 kernel
            bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
            return cutlass_int8.conv2d_int8(
                x, self.weight_int8, self.weight_scale, bias,
                self.stride, self.stride,
                self.padding, self.padding,
                self.dilation, self.dilation
            )
        elif self.use_fp16:
            # FP16 with Tensor Cores
            x_fp16 = x.half()
            out = F.conv2d(x_fp16, self.weight_fp16, None,
                          self.stride, self.padding, self.dilation, self.groups)
            if self.bias is not None:
                out = out + self.bias.half().view(1, -1, 1, 1)
            return out.float()
        else:
            # FP32 fallback
            return F.conv2d(x, self.weight_fp32, self.bias,
                          self.stride, self.padding, self.dilation, self.groups)


class FP16Conv2d(nn.Module):
    """FP16 Convolution using cuDNN Tensor Cores."""
    
    def __init__(self, original_conv):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        
        # Store weights in FP16
        self.register_buffer('weight', original_conv.weight.data.half())
        if original_conv.bias is not None:
            self.register_buffer('bias', original_conv.bias.data.half())
        else:
            self.register_buffer('bias', None)
        
        self.enabled = False
        
    def enable_modiff(self, enabled=True):
        self.enabled = enabled
        
    def reset_state(self):
        pass
    
    def forward(self, x):
        if self.enabled:
            x_fp16 = x.half() if x.dtype != torch.float16 else x
            out = F.conv2d(x_fp16, self.weight, self.bias,
                          self.stride, self.padding, self.dilation, self.groups)
            return out.float() if x.dtype == torch.float32 else out
        else:
            # FP32 fallback
            return F.conv2d(x, self.weight.float(), 
                          self.bias.float() if self.bias is not None else None,
                          self.stride, self.padding, self.dilation, self.groups)


def convert_model_to_int8(model):
    """Convert all Conv2d layers to CUTLASS INT8."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, CutlassInt8Conv2d(module))
        else:
            convert_model_to_int8(module)
    return model


def convert_model_to_fp16(model):
    """Convert all Conv2d layers to FP16."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, FP16Conv2d(module))
        else:
            convert_model_to_fp16(module)
    return model


def enable_modiff_mode(model, enabled=True):
    """Enable/disable MoDiff mode for all converted layers."""
    for module in model.modules():
        if hasattr(module, 'enable_modiff'):
            module.enable_modiff(enabled)


def reset_modiff_state(model):
    """Reset MoDiff state for all converted layers."""
    for module in model.modules():
        if hasattr(module, 'reset_state'):
            module.reset_state()


# Aliases for backward compatibility
convert_model_to_w8a8 = convert_model_to_int8
convert_model_to_modiff_int8 = convert_model_to_int8
convert_model_to_modiff_fp16 = convert_model_to_fp16
