
import torch
import torch.nn as nn
import time
import numpy as np
from ddim.models.diffusion import Model
from modiff_cuda.nn.conv import W8A8MoDiffConv2dCUDA

def replace_layers(model):
    """Replace nn.Conv2d with W8A8MoDiffConv2dCUDA."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Create weight scales (per-channel)
            weight_scales = torch.ones(module.out_channels, device='cuda', dtype=torch.float32) * 0.01
            
            # Create our CUDA module
            new_module = W8A8MoDiffConv2dCUDA.from_float(module, weight_scales)
            setattr(model, name, new_module)
        else:
            replace_layers(module)

def benchmark_model(model, input_shape, num_runs=50, warmup=10):
    # Determine dtype from model
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float32
    
    print(f"Benchmark Input Dtype: {dtype}")
    x = torch.randn(input_shape, dtype=dtype).cuda()
    t = torch.tensor([10]).cuda()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x, t)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x, t)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_runs * 1000
    return avg_time

def main():
    # Mock config for CIFAR-10 model
    class Config:
        def __init__(self):
            self.model = type('obj', (object,), {
                'type': 'simple',
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 2, 2, 2],
                'num_res_blocks': 2,
                'attn_resolutions': [16],
                'dropout': 0.1,
                'resamp_with_conv': True,
            })
            self.data = type('obj', (object,), {
                'image_size': 32
            })
            self.diffusion = type('obj', (object,), {
                'num_diffusion_timesteps': 1000
            })
            self.split_shortcut = False
    
    config = Config()
    
    print("Creating FP32 Model...")
    model_fp32 = Model(config).cuda().eval()
    
    print("Creating INT8 CUDA Model...")
    model_int8 = Model(config).cuda().eval().half()
    replace_layers(model_int8)
    
    input_shape = (64, 3, 32, 32)
    
    print(f"Benchmarking FP32 Model (Input: {input_shape})...")
    fp32_time = benchmark_model(model_fp32, input_shape)
    print(f"FP32 Average Time: {fp32_time:.4f} ms")
    
    print(f"Benchmarking INT8 CUDA Model (Input: {input_shape})...")
    int8_time = benchmark_model(model_int8, input_shape)
    print(f"INT8 CUDA Average Time: {int8_time:.4f} ms")
    
    print(f"Speedup: {fp32_time / int8_time:.2f}x")

if __name__ == "__main__":
    main()
