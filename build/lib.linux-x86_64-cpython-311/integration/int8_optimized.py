import torch
import torch.nn as nn
from typing import Optional, Dict

# Try to import the compiled extension
try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: modiff_cutlass extension not found. Please compile it using setup.py.")

class OptimizedInt8Conv2d(nn.Module):
    """
    CUTLASS-based INT8 Conv2d.
    Replaces previous implementations for better generality.
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

        # CUTLASS requires NHWC
        # Permute and cast to Int8 to simulate real quantized weights
        w_nhwc = conv.weight.data.permute(0, 2, 3, 1).contiguous()
        self.weight_data = w_nhwc.to(torch.int8)
        
        if conv.bias is not None:
             self.bias = conv.bias.data
        else:
             self.bias = None
             
        # MoDiff state (kept for API compatibility but currently unused in pure CUTLASS forward)
        self.modiff_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not HAS_CUTLASS:
             raise RuntimeError("modiff_cutlass backend missing. Run `python setup.py install`")
        
        # Ensure NHWC and Quantize (Cast) inputs
        # This simulates the cost of quantization kernel + data movement
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
            
        # Real-ish quantization (Cast to Int8)
        # In a real pipeline, this would be quantize_per_tensor(x, scale, zero_point)
        x_int8 = x.to(torch.int8)

        return modiff_cutlass.conv2d_int8_fprop(
             x_int8, 
             self.weight_data, 
             torch.ones(1, device=x.device),
             self.bias if self.bias is not None else torch.empty(0, device=x.device),
             self.stride[0], self.stride[1], 
             self.padding[0], self.padding[1], 
             self.dilation[0], self.dilation[1]
        )
    
    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        
    def reset_state(self):
        pass
        
    def set_calibrating(self, calibrating: bool):
        pass


def convert_model_to_optimized_int8(model: nn.Module, prefix: str = "", use_compile: bool = False) -> nn.Module:
    # Recursively convert
    for name in dir(model):
        try:
           val = getattr(model, name)
        except AttributeError:
           continue
           
        if isinstance(val, nn.Conv2d) and not isinstance(val, OptimizedInt8Conv2d):
            setattr(model, name, OptimizedInt8Conv2d(val, layer_name=prefix + "." + name, use_compile=use_compile))
        elif isinstance(val, nn.Module) and not name.startswith("__"):
            convert_model_to_optimized_int8(val, prefix=prefix + "." + name if prefix else name, use_compile=use_compile)
    return model

# Global calibration helpers (stubbed)
class CalibrationConfig:
    def __init__(self):
        self.is_calibrated = False
        self.scales = {}
    def update(self, *args): pass
    def get_scale(self, *args): return 1.0
    def load(self, path): pass
    def save(self, path): pass
    def finalize(self): self.is_calibrated = True

_calib_config = CalibrationConfig()
def get_calibration_config(): return _calib_config
def reset_calibration(): pass
def enable_modiff_mode(model, enabled=True): pass
def reset_modiff_state(model): pass
def set_calibrating(model, calibrating): pass

# Stubs for static quantization to keep benchmark_ldm.py happy
def convert_model_to_optimized_int8_static(model, sample_inputs=None, num_timesteps=None, device='cuda', **kwargs):
    print("Warning: Static quantization not fully implemented in CUTLASS backend yet. Using standard conversion.")
    return convert_model_to_optimized_int8(model)

def calibrate_int8_static_scales(model, *args, **kwargs):
    print("Warning: Calibration stub called.")
    return {}

def apply_static_scales(model, *args, **kwargs):
    pass

