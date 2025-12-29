
import torch
import torch.nn as nn
import sys
import os

# Ensure MoDiff is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from modiff_cuda.nn.conv import W8A8MoDiffConv2dCUDA
except ImportError:
    print("Warning: Could not import W8A8MoDiffConv2dCUDA. Make sure modiff_cuda is installed.")
    W8A8MoDiffConv2dCUDA = None

class MoDiffConv2dWrapper(nn.Module):
    def __init__(self, original_conv, use_cuda_kernel=True):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        
        self.use_cuda_kernel = use_cuda_kernel and (W8A8MoDiffConv2dCUDA is not None)
        
        if self.use_cuda_kernel:
            # Create the CUDA kernel layer
            # We assume original_conv weights are float. 
            # The from_float method we added handles quantization.
            self.conv_layer = W8A8MoDiffConv2dCUDA.from_float(original_conv)
        else:
            self.conv_layer = original_conv
            
        # Handle bias
        if original_conv.bias is not None:
            self.register_buffer('bias', original_conv.bias)
        else:
            self.register_buffer('bias', None)
            
        # State for MoDiff
        self.last_input = None
        self.last_output = None
        self.enabled = False # If False, acts like normal Conv2d
        
    def reset_state(self):
        self.last_input = None
        self.last_output = None
        
    def enable_modiff(self, enabled=True):
        self.enabled = enabled
        if not enabled:
            self.reset_state()

    def forward(self, x):
        if not self.enabled:
            # Standard forward pass
            if self.use_cuda_kernel:
                out = self.conv_layer(x)
                if self.bias is not None:
                    # Add bias if kernel doesn't support it
                    # Reshape bias to [1, C, 1, 1]
                    out += self.bias.view(1, -1, 1, 1)
                return out
            else:
                return self.conv_layer(x)
        
        # MoDiff Logic
        if self.last_input is None:
            # First step (T)
            # Compute full output
            if self.use_cuda_kernel:
                out = self.conv_layer(x)
                if self.bias is not None:
                    out += self.bias.view(1, -1, 1, 1)
            else:
                out = self.conv_layer(x)
            
            # Save state
            self.last_input = x.detach() # Save float input
            self.last_output = out.detach()
            return out
        else:
            # Subsequent steps (T-1, ...)
            # Calculate delta
            delta = x - self.last_input
            
            if self.use_cuda_kernel:
                # Use the specialized kernel with accumulation
                # The kernel is bias-free, which is exactly what we want for delta
                out = self.conv_layer(delta, prev_output=self.last_output)
            else:
                # Simulation in FP32
                # We need bias-free convolution for delta
                delta_out = nn.functional.conv2d(
                    delta, 
                    self.conv_layer.weight, 
                    bias=None, 
                    stride=self.conv_layer.stride, 
                    padding=self.conv_layer.padding, 
                    dilation=self.conv_layer.dilation, 
                    groups=self.conv_layer.groups
                )
                out = delta_out + self.last_output
            
            # Update state
            self.last_input = x.detach()
            self.last_output = out.detach()
            
            return out

def convert_model_to_modiff(model, use_cuda_kernel=True):
    """
    Recursively replace nn.Conv2d with MoDiffConv2dWrapper.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Replace
            wrapper = MoDiffConv2dWrapper(module, use_cuda_kernel=use_cuda_kernel)
            setattr(model, name, wrapper)
        else:
            convert_model_to_modiff(module, use_cuda_kernel=use_cuda_kernel)
    return model

def enable_modiff_mode(model, enabled=True):
    for module in model.modules():
        if isinstance(module, MoDiffConv2dWrapper):
            module.enable_modiff(enabled)

def reset_modiff_state(model):
    for module in model.modules():
        if isinstance(module, MoDiffConv2dWrapper):
            module.reset_state()
