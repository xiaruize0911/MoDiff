"""
INT8-Only Inference Path for MoDiff

Key optimizations:
1. Keep activations in INT8 format between convolutions
2. Only dequantize for non-linear operations (SiLU, GroupNorm, Attention)
3. Fuse quantize-conv-dequantize into single kernels where possible
4. Use persistent scales to avoid recomputation

Data flow:
  Standard: FP32 → Q → INT8 Conv → DQ → FP32 → Q → INT8 Conv → DQ → FP32
  Optimized: FP32 → Q → [INT8 Conv → INT8 Conv] → DQ → NonLinear → Q → ...
  
This reduces Q/DQ operations by grouping consecutive convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modiff_cuda'))

try:
    import modiff_int8 as cutlass_int8
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("CUTLASS INT8 not available")


class INT8ConvBlock(nn.Module):
    """
    Fused INT8 convolution block that keeps intermediate activations in INT8.
    
    For a ResBlock pattern:
        x → Conv1 → SiLU → Conv2 → + → out
        └────────────────────────────┘
        
    Optimized to:
        x(FP32) → Q → Conv1(INT8) → DQ → SiLU → Q → Conv2(INT8) → DQ → + → out
        
    Further fused:
        x(FP32) → [Q+Conv1+DQ+SiLU+Q+Conv2+DQ] → + → out
        
    Where internal Q/DQ can use cached scales from MoDiff.
    """
    
    def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d, activation=F.silu):
        super().__init__()
        
        # Store original conv configs
        self.conv1_config = {
            'in_channels': conv1.in_channels,
            'out_channels': conv1.out_channels,
            'kernel_size': conv1.kernel_size,
            'stride': conv1.stride,
            'padding': conv1.padding,
        }
        self.conv2_config = {
            'in_channels': conv2.in_channels,
            'out_channels': conv2.out_channels,
            'kernel_size': conv2.kernel_size,
            'stride': conv2.stride,
            'padding': conv2.padding,
        }
        
        # Quantize weights once
        if HAS_CUTLASS:
            self.weight1_int8, self.scales1 = cutlass_int8.quantize_weight(conv1.weight.data.cuda())
            self.weight2_int8, self.scales2 = cutlass_int8.quantize_weight(conv2.weight.data.cuda())
        else:
            self.register_buffer('weight1_fp32', conv1.weight.data)
            self.register_buffer('weight2_fp32', conv2.weight.data)
            
        self.register_buffer('bias1', conv1.bias.data if conv1.bias is not None else None)
        self.register_buffer('bias2', conv2.bias.data if conv2.bias is not None else None)
        
        self.activation = activation
        
        # MoDiff caches for the block
        self.input_scale_cache: Optional[float] = None
        self.mid_scale_cache: Optional[float] = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with reduced Q/DQ operations.
        """
        if not HAS_CUTLASS:
            # Fallback to FP32
            h = F.conv2d(x, self.weight1_fp32, self.bias1, 
                        self.conv1_config['stride'], self.conv1_config['padding'])
            h = self.activation(h)
            h = F.conv2d(h, self.weight2_fp32, self.bias2,
                        self.conv2_config['stride'], self.conv2_config['padding'])
            return h
        
        # INT8 path with cached scales
        # Conv1: x → INT8 conv → FP32
        out1 = cutlass_int8.conv2d_int8(
            x, self.weight1_int8, self.scales1, self.bias1,
            self.conv1_config['stride'][0], self.conv1_config['padding'][0], 1, 1
        )
        
        # Activation (must be FP32)
        h = self.activation(out1)
        
        # Conv2: h → INT8 conv → FP32
        out2 = cutlass_int8.conv2d_int8(
            h, self.weight2_int8, self.scales2, self.bias2,
            self.conv2_config['stride'][0], self.conv2_config['padding'][0], 1, 1
        )
        
        return out2


class INT8OnlyConv2d(nn.Module):
    """
    INT8 Conv2d that accepts INT8 input and produces INT8 output.
    
    This avoids the Q/DQ overhead when chaining multiple convolutions.
    
    Key insight: 
    - INT8 × INT8 → INT32 (accumulator)
    - We can requantize INT32 → INT8 using a single fused operation
    - Scale propagation: output_scale = input_scale * weight_scale / 127
    """
    
    def __init__(self, original_conv: nn.Conv2d):
        super().__init__()
        
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        
        # Quantize weights
        if HAS_CUTLASS:
            w = original_conv.weight.data
            self.weight_int8, self.weight_scales = cutlass_int8.quantize_weight(w.cuda())
        else:
            self.register_buffer('weight_fp32', original_conv.weight.data)
            self.weight_scales = None
            
        self.register_buffer('bias', original_conv.bias.data if original_conv.bias is not None else None)
        
        # For INT8-in/INT8-out mode
        self.int8_mode = False
        self.input_scale: Optional[torch.Tensor] = None
        self.output_scale: Optional[torch.Tensor] = None
        
    def set_int8_mode(self, enabled: bool, input_scale: Optional[torch.Tensor] = None):
        """Enable INT8-in/INT8-out mode with provided input scale."""
        self.int8_mode = enabled
        self.input_scale = input_scale
        
    def forward_int8_to_int8(self, x_int8: torch.Tensor, input_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        INT8 input → INT8 output (no intermediate FP32).
        
        Returns: (output_int8, output_scale)
        """
        # This would require a custom CUDA kernel that:
        # 1. Does INT8 conv → INT32
        # 2. Applies requantization: INT32 * (input_scale * weight_scale) / new_scale → INT8
        # 
        # For now, we simulate with Q/DQ
        x_fp32 = x_int8.float() * input_scale
        
        if HAS_CUTLASS:
            out_fp32 = cutlass_int8.conv2d_int8(
                x_fp32, self.weight_int8, self.weight_scales, self.bias,
                self.stride[0], self.padding[0], 1, 1
            )
        else:
            out_fp32 = F.conv2d(x_fp32, self.weight_fp32, self.bias, self.stride, self.padding)
        
        # Requantize output
        out_max = out_fp32.abs().max()
        output_scale = out_max / 127.0
        out_int8 = (out_fp32 / output_scale).round().clamp(-127, 127).to(torch.int8)
        
        return out_int8, output_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward (FP32 in/out)."""
        if HAS_CUTLASS:
            return cutlass_int8.conv2d_int8(
                x, self.weight_int8, self.weight_scales, self.bias,
                self.stride[0], self.padding[0], 1, 1
            )
        else:
            return F.conv2d(x, self.weight_fp32, self.bias, self.stride, self.padding)


class FusedQuantizeConvDequantize(torch.autograd.Function):
    """
    Custom autograd function that fuses Q+Conv+DQ into a single operation.
    
    This reduces kernel launches from 3 to 1 for the common pattern:
        x(FP32) → Quantize → Conv(INT8) → Dequantize → y(FP32)
    """
    
    @staticmethod
    def forward(ctx, x, weight_int8, weight_scales, bias, stride, padding):
        # In a real implementation, this would call a single fused CUDA kernel
        # For now, we call the existing CUTLASS kernel which already does Q+Conv+DQ
        if HAS_CUTLASS:
            return cutlass_int8.conv2d_int8(x, weight_int8, weight_scales, bias, stride, padding, 1, 1)
        else:
            raise RuntimeError("CUTLASS not available")
    
    @staticmethod
    def backward(ctx, grad_output):
        # INT8 training not supported
        raise NotImplementedError("INT8 backward not implemented")


def analyze_layer_fusion_opportunities(model: nn.Module) -> Dict[str, List[str]]:
    """
    Analyze model to find consecutive Conv2d layers that can be fused.
    
    Returns dict of fuseable layer groups.
    """
    fusion_groups = {}
    current_group = []
    group_id = 0
    
    prev_was_conv = False
    for name, module in model.named_modules():
        is_conv = isinstance(module, nn.Conv2d)
        
        if is_conv:
            current_group.append(name)
            prev_was_conv = True
        else:
            if prev_was_conv and len(current_group) > 1:
                fusion_groups[f'group_{group_id}'] = current_group.copy()
                group_id += 1
            current_group = []
            prev_was_conv = False
    
    return fusion_groups


def optimize_model_int8_only(model: nn.Module) -> nn.Module:
    """
    Convert model to use INT8-only paths where possible.
    
    Strategy:
    1. Find consecutive Conv2d layers
    2. Replace with fused INT8 blocks
    3. Only dequantize at non-linear operations
    """
    # For now, just use the existing INT8 conv replacement
    # Full INT8-only would require custom kernels
    from integration.modiff_layers import CutlassInt8Conv2d
    
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, CutlassInt8Conv2d(module))
        else:
            optimize_model_int8_only(module)
    
    return model
