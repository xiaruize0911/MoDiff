import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import modiff_cuda_backend
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import modiff_cuda_backend

print(f"Backend loaded from: {modiff_cuda_backend.__file__}")

class W8A8MoDiffConv2dCUDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        
        # Weight is [out_channels, in_channels * k * k] for Implicit GEMM
        # Pad K to 16 bytes for alignment
        self.K = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.padded_K = (self.K + 15) // 16 * 16
        
        self.register_buffer('weight', torch.zeros((out_channels, self.padded_K), dtype=torch.int8))
        self.register_buffer('weight_scales', torch.zeros((out_channels,), dtype=torch.float32))
        
    @classmethod
    def from_float(cls, module, weight_scales):
        instance = cls(module.in_channels, module.out_channels, module.kernel_size, 
                      module.stride, module.padding, module.dilation, module.groups, 
                      module.bias is not None)
        
        w = module.weight.data
        w_flat = w.view(module.out_channels, -1)
        
        # Pad weights
        K = w_flat.size(1)
        padded_K = (K + 15) // 16 * 16
        if padded_K > K:
            w_padded = torch.zeros(module.out_channels, padded_K, device=w.device, dtype=w.dtype)
            w_padded[:, :K] = w_flat
            w_flat = w_padded
            
        w_q = torch.round(w_flat / weight_scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
        
        instance.weight.copy_(w_q)
        instance.weight_scales.copy_(weight_scales)
        
        return instance

    def forward(self, x, prev_output=None, input_scale=None, output_layout='NCHW'):
        # x: [N, C, H, W] or [N, H, W, C] if _layout='NHWC'
        
        # Check input layout
        is_nhwc = getattr(x, '_layout', 'NCHW') == 'NHWC'
        
        # Debug print for shapes (only once per shape to avoid spam)
        if not hasattr(self, 'logged_shape'):
            print(f"Conv Input: {x.shape}, Layout: {'NHWC' if is_nhwc else 'NCHW'}")
            self.logged_shape = True
        
        # Get activation scale
        if input_scale is not None:
            act_scale = input_scale
        elif hasattr(x, 'next_scale'):
            act_scale = x.next_scale
        else:
            act_scale = modiff_cuda_backend.find_max_abs(x) / 127.0
        
        if x.dtype == torch.int8:
            # Pre-quantized input
            if is_nhwc:
                input_int8 = x
            else:
                # Permute NCHW -> NHWC
                input_int8 = x.permute(0, 2, 3, 1).contiguous()
        elif is_nhwc:
            # Input is NHWC. Just quantize.
            input_int8 = modiff_cuda_backend.quantize_tensor(x, act_scale)
        else:
            # Input is NCHW. Permute + Quantize.
            input_int8 = modiff_cuda_backend.quantize_permute(x, act_scale)
        
        compute_max = (output_layout == 'NHWC')
        
        # Debug print types
        print(f"Call args types: input={input_int8.dtype}, weight={self.weight.dtype}, act_scale={act_scale.dtype}, w_scales={self.weight_scales.dtype}")
        print(f"Call args shapes: input={input_int8.shape}, weight={self.weight.shape}, act_scale={act_scale.shape}, w_scales={self.weight_scales.shape}")
        print(f"Int args: k={self.kernel_size[0]}, s={self.stride[0]}, p={self.padding[0]}, max={compute_max}, sk={self.padded_K}")

        if prev_output is None:
            out_nhwc, out_max = modiff_cuda_backend.conv2d_fast_w8a8(
                input_int8,
                self.weight,
                act_scale,
                self.weight_scales,
                int(self.kernel_size[0]),
                stride=int(self.stride[0]),
                padding=int(self.padding[0]),
                compute_max=compute_max
                # stride_k=int(self.padded_K)
            )
        else:
            # prev_output handling
            if getattr(prev_output, '_layout', 'NCHW') == 'NHWC':
                prev_output_nhwc = prev_output
            else:
                prev_output_nhwc = prev_output.permute(0, 2, 3, 1).contiguous()
                
            out_nhwc, out_max = modiff_cuda_backend.conv2d_fast_w8a8_accum(
                input_int8,
                self.weight,
                prev_output_nhwc,
                act_scale,
                self.weight_scales,
                int(self.kernel_size[0]),
                stride=int(self.stride[0]),
                padding=int(self.padding[0]),
                compute_max=compute_max
                # stride_k=int(self.padded_K)
            )
            
        if output_layout == 'NHWC':
            # Max is already computed in conv kernel
            out_nhwc.next_scale = out_max / 127.0
            out_nhwc._layout = 'NHWC'
            return out_nhwc
        else:
            # Convert back to NCHW using fast kernel
            # Compute max for next layer
            out_nchw, out_max = modiff_cuda_backend.permute_half_nhwc_nchw(out_nhwc, True)
            
            # Attach next scale to output
            out_nchw.next_scale = out_max / 127.0
            out_nchw._layout = 'NCHW'
            return out_nchw
