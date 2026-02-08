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
    # We want to pack along the last dimension.
    # Check shape
    shape = list(tensor.shape)
    last_dim = shape[-1]
    
    if last_dim % 2 != 0:
        raise ValueError(f"Last dimension {last_dim} must be divisible by 2 for INT4 packing")
    
    # Reshape to separate adjacent pairs
    # [..., dim] -> [..., dim/2, 2]
    new_shape = shape[:-1] + [last_dim // 2, 2]
    reshaped = tensor.view(new_shape)
    
    # Extract
    low = reshaped[..., 0] & 0x0F
    high = (reshaped[..., 1] & 0x0F) << 4
    
    # Combine
    packed = (low | high).to(torch.int8) # Store as int8 container
    
    return packed

class OptimizedInt4Conv2d(nn.Module):
    """
    CUTLASS-based INT4 Conv2d.
    Implements real INT4 packing for weights.
    Activations are packed on the fly (overhead warning) or should be pre-packed.
    """
    def __init__(self, conv: nn.Conv2d, layer_name: str = "", use_compile: bool = False):
        super().__init__()
        if not hasattr(OptimizedInt4Conv2d, '_init_printed'):
             print(f"DEBUG: OptimizedInt4Conv2d init for {layer_name}")
             OptimizedInt4Conv2d._init_printed = True
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        
        # Handle tuple vs int properties robustly
        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self.groups = conv.groups

        # 1. Weight quantization and packing
        # Real implementation would calibrate/quantize.
        # Here we cast to Int8 then pack.
        # Ensure NHWC (K, R, S, C) for packing along C
        # Conv weight is (K, C, R, S).
        # Permute to (K, R, S, C)
        w_nhwc = conv.weight.data.permute(0, 2, 3, 1).contiguous()
        
        # Fake quantization (cast)
        w_int8 = w_nhwc.to(torch.int8)
        
        # Pack
        if self.in_channels % 2 == 0:
            self.weight_packed = pack_int4(w_int8)
        else:
            # Fallback (should not happen in Standard UNet ResBlocks)
            # Pad to even? For now just use int8 and fail?
            # Actually, CUTLASS INT4 kernel requires packing.
            raise ValueError(f"Input channels {self.in_channels} not divisible by 2")

        if conv.bias is not None:
             self.bias = conv.bias.data
        else:
             self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Measure total layer time
        # print(f"DEBUG: OptimizedInt4Conv2d.forward called for {self.layer_name}")
        fwd_start = profiler.start("Layer: OptimizedInt4Conv2d.forward")

        # Overhead: Setup
        ov_t = profiler.start("Overhead: Checks & Contiguous")
        if not HAS_CUTLASS:
             # Fail loud if user requested CUTLASS rebuild
             raise RuntimeError("modiff_cutlass backend missing. Run `python setup.py install`")
        
        # 2. Input Packing using Fused Kernel
        # x is [N, C, H, W]. 
        # Convert to NHWC [N, H, W, C]
        if not x.is_contiguous(memory_format=torch.channels_last):
             x = x.contiguous(memory_format=torch.channels_last)
        profiler.stop("Overhead: Checks & Contiguous", ov_t)

        # Bandwidth: Pack (Fused Quantize + Pack)
        # This calls the fast CUDA kernel we just wrote.
        # Skips intermediate int8 tensor!
        pack_t = profiler.start("Bandwidth: Packing (quantize_and_pack)")
        x_packed = modiff_cutlass.quantize_and_pack(x)
        profiler.stop("Bandwidth: Packing (quantize_and_pack)", pack_t)
            
        # Call CUTLASS extension
        # Note: In a real Scenario, weight packing happen offline or in init.
        # Here we pass metadata.
        
        # Compute: Conv
        comp_t = profiler.start("Compute: Conv Kernel")
        out = modiff_cutlass.conv2d_int4_fprop(
             x_packed, 
             self.weight_packed, 
             torch.ones(1, device=x.device), # Scale placeholder
             self.bias if self.bias is not None else torch.empty(0, device=x.device),
             self.stride[0], self.stride[1], 
             self.padding[0], self.padding[1], 
             self.dilation[0], self.dilation[1]
        )
        profiler.stop("Compute: Conv Kernel", comp_t)

        profiler.stop("Layer: OptimizedInt4Conv2d.forward", fwd_start)
        return out

# Factory function
def convert_model_to_optimized_int4(model: nn.Module, prefix: str = "", use_compile: bool = False) -> nn.Module:
    # Use named_children to robustly traverse Sequential and ModuleList
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Conv2d) and not isinstance(child, OptimizedInt4Conv2d):
            # Skip very small channel counts (e.g. input layer C=4) as they break Tensor Core alignment
            # and provide negligible speedup.
            if child.in_channels < 32:
                 print(f"Skipping INT4 conversion for {full_name} (in_channels={child.in_channels})")
                 continue
                 
            print(f"Initializing INT4 conversion for {full_name} (in_channels={child.in_channels})")
            optimized_conv = OptimizedInt4Conv2d(child, layer_name=full_name, use_compile=use_compile)
            
            # Special handling for Sequential/ModuleList/ModuleDict if needed?
            # Module.__setattr__ handles updating _modules for a given name properly.
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int4(child, prefix=full_name, use_compile=use_compile)
            
    return model.to(memory_format=torch.channels_last)

# Stubbed functions for compatibility with existing scripts
def enable_modiff_mode(model: nn.Module, enabled: bool = True):
    pass

def reset_modiff_state(model: nn.Module):
    pass

def set_calibrating_int4(model: nn.Module, calibrating: bool):
    pass
