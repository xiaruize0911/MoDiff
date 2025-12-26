
import torch
import torch.nn as nn
from .kernels.fused_modiff import fused_modiff_gemm

class FusedMoDiffLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Quantization params
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_int8', torch.zeros(in_features, out_features, dtype=torch.int8)) # Transposed
        
        # Cache
        self.prev_activation = None
        self.prev_output = None
        
    def forward(self, x):
        if self.prev_activation is None:
            # First step (Standard GEMM)
            # For simplicity in this demo, we just use FP32 fallback or standard GEMM
            # In real MoDiff, this is Quant -> GEMM -> Cache
            out = nn.functional.linear(x, self.weight, self.bias)
            self.prev_activation = x.clone() # Should be quantized version
            self.prev_output = out.clone()
            return out
        else:
            # Fused Step
            # x: [M, K]
            # prev_activation: [M, K]
            # weight_int8: [K, N]
            # prev_output: [M, N]
            
            # Ensure shapes match
            if x.shape != self.prev_activation.shape:
                self.prev_activation = x.clone()
                out = nn.functional.linear(x, self.weight, self.bias)
                self.prev_output = out.clone()
                return out

            out = fused_modiff_gemm(
                x, self.prev_activation, 
                self.weight_int8, 
                self.prev_output, 
                self.weight_scale
            )
            
            # Update output cache (already done in kernel? No, kernel writes to 'out', we need to update 'prev_output')
            # Actually, the kernel adds 'prev_output' to result.
            # The result 'out' IS the new 'prev_output' for the next step.
            self.prev_output = out
            
            return out
