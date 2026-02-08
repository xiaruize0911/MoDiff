
import torch
import torch.nn as nn
from typing import Optional
from integration.profiler import profiler
print(f"DEBUG: int4_optimized loaded. Profiler ID: {id(profiler)}")

# Try to import the compiled extension
try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: modiff_cutlass extension not found. Please compile it using setup.py.")

def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack int8 tensor to packed int4 (2 elements per byte).
    Input: Int8 Tensor usually [K, C, R, S] or [N, H, W, C]
    Output: Int8 Tensor (uint8 view) with last dim halved.
    """
    # Assume input is Int8 range [-8, 7]
    shape = list(tensor.shape)
    last_dim = shape[-1]
    
    if last_dim % 2 != 0:
        raise ValueError(f"Last dimension {last_dim} must be divisible by 2 for INT4 packing")
    
    # Reshape to separate adjacent pairs
    new_shape = shape[:-1] + [last_dim // 2, 2]
    reshaped = tensor.view(new_shape)
    
    low = reshaped[..., 0] & 0x0F
    high = (reshaped[..., 1] & 0x0F) << 4
    
    packed = (low | high).to(torch.int8) 
    return packed

class OptimizedInt4Conv2d(nn.Module):
    """
    CUTLASS-based INT4 Conv2d.
    Implements real INT4 packing for weights.
    """
    def __init__(self, conv: nn.Conv2d, layer_name: str = "", use_compile: bool = False):
        super().__init__()
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        
        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self.groups = conv.groups

        # 1. Weight quantization and packing
        # Conv weight is (K, C, R, S).
        w_data = conv.weight.data
        
        # Per-tensor weight quantization using robust max (mean + 3*std).
        # This preserves more weight information than max-based:
        # With max=0.38, max scaling zeros 63% of weights (step=0.054).
        # With 3sigma=0.085, robust scaling zeros only 26% (step=0.012).
        # A few extreme weights (>3sigma) are clipped to 7, which is an
        # acceptable trade-off for much better bulk representation.
        w_abs = w_data.abs()
        robust_max = w_abs.mean() + 3.0 * w_abs.std()
        robust_max = torch.clamp(robust_max, min=1e-6)
        self.weight_scale = 7.0 / robust_max.item()

        w_quant = (w_data * self.weight_scale).round().clamp(-7, 7).to(torch.int8)
        
        # Compute output scale correction: quantization introduces systematic
        # attenuation because the dead zone around 0 zeros out small weights.
        # Measure ratio of original vs dequantized weight std to compensate.
        w_dequant = w_quant.float() / self.weight_scale
        dequant_std = w_dequant.std().item()
        orig_std = w_data.std().item()
        self.output_correction = (orig_std / dequant_std) if dequant_std > 1e-8 else 1.0
        
        # Permute to Physical NHWC (K, R, S, C)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        
        # Pack
        if self.in_channels % 2 == 0:
            self.weight_packed = pack_int4(w_nhwc)
        else:
            raise ValueError(f"Input channels {self.in_channels} not divisible by 2")

        if conv.bias is not None:
             self.register_buffer('bias', conv.bias.data)
        else:
             self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_start = profiler.start("Layer: OptimizedInt4Conv2d.forward")

        if not HAS_CUTLASS:
             raise RuntimeError("modiff_cutlass backend missing.")
        
        # Ensure channels_last for CUTLASS NHWC path
        if not x.is_contiguous(memory_format=torch.channels_last):
             x = x.contiguous(memory_format=torch.channels_last)

        # Dynamic per-tensor activation scaling
        # Use mean + 3*std (99.87th percentile): clips top 0.13% of values
        # but provides better resolution for the bulk of the distribution.
        # 3-sigma preserves output std better than 5-sigma over 50 DDIM steps
        # (94% vs 80% preservation) because tighter bins give better resolution.
        abs_x = x.abs()
        target_max = abs_x.mean() + 3.0 * abs_x.std()
        target_max = torch.max(target_max, torch.tensor(1e-6, device=x.device))
        input_scale = 7.0 / target_max.item()

        # Combined dequant scale: output = dequant * int_accum + bias
        # Include output_correction to compensate for quantization attenuation
        dequant_scale = self.output_correction / (input_scale * self.weight_scale)
        scale_tensor = torch.tensor([dequant_scale], device=x.device, dtype=torch.float32)

        pack_t = profiler.start("Bandwidth: Packing (quantize_and_pack)")
        x_scaled = x * input_scale
        x_packed = modiff_cutlass.quantize_and_pack(x_scaled)
        profiler.stop("Bandwidth: Packing (quantize_and_pack)", pack_t)
            
        comp_t = profiler.start("Compute: Conv Kernel")
        
        out = modiff_cutlass.conv2d_int4_fprop(
             x_packed, 
             self.weight_packed, 
             scale_tensor,
             self.bias if self.bias is not None else torch.empty(0, device=x.device),
             self.stride[0], self.stride[1], 
             self.padding[0], self.padding[1], 
             self.dilation[0], self.dilation[1]
        )
        profiler.stop("Compute: Conv Kernel", comp_t)

        profiler.stop("Layer: OptimizedInt4Conv2d.forward", fwd_start)
        return out

def convert_model_to_optimized_int4(model: nn.Module, prefix: str = "", use_compile: bool = False) -> nn.Module:
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, OptimizedInt4Conv2d):
            if child.in_channels < 32:
                 continue
            # Skip sensitive layers that destroy signal quality when quantized:
            # 1. Skip connections (residual path - critical for signal preservation)
            # 2. Final output projection (directly affects output quality)
            # 3. First input projection (sets the initial signal range)
            is_skip = 'skip' in name
            is_output = (prefix == '' and name == 'out') or name == 'out'
            # Check if this is the top-level output conv (e.g. model.out.2)
            is_final_out = full_name.startswith('out.')
            
            if is_skip or is_final_out:
                continue
                
            optimized_conv = OptimizedInt4Conv2d(child, layer_name=full_name, use_compile=use_compile)
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int4(child, prefix=full_name, use_compile=use_compile)
    return model.to(memory_format=torch.channels_last)

# Stubbed functions
def enable_modiff_mode(model, enabled=True): pass
def reset_modiff_state(model): pass
def set_calibrating_int4(model, calibrating): pass
