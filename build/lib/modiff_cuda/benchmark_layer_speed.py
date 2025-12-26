import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os

# Add current directory to path to find modiff_cuda_backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nn.conv import W8A8MoDiffConv2dCUDA

def benchmark_layer(name, layer, x, num_iters=100):
    # Warmup
    for _ in range(10):
        _ = layer(x)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = layer(x)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / num_iters * 1000

def run_benchmark():
    device = torch.device("cuda")
    print(f"Benchmarking on {torch.cuda.get_device_name(0)}")
    print("-" * 80)
    print(f"{'Shape (N, C, H, W)':<30} {'PyTorch FP32':<15} {'PyTorch FP16':<15} {'MoDiff CUDA':<15} {'Speedup':<10}")
    print("-" * 80)

    configs = [
        (1, 128, 32, 32),
        (1, 256, 16, 16),
        (1, 512, 8, 8),
        (8, 128, 32, 32),
        (8, 256, 16, 16),
        (8, 512, 8, 8),
        (32, 128, 32, 32),
        (32, 256, 16, 16),
        (32, 512, 8, 8),
    ]

    for N, C, H, W in configs:
        C_in = C
        C_out = C
        K = 3
        
        # PyTorch FP32
        conv_fp32 = nn.Conv2d(C_in, C_out, K, padding=1).to(device)
        x_fp32 = torch.randn(N, C_in, H, W, device=device)
        time_fp32 = benchmark_layer("FP32", conv_fp32, x_fp32)
        
        # PyTorch FP16
        conv_fp16 = nn.Conv2d(C_in, C_out, K, padding=1).half().to(device)
        x_fp16 = torch.randn(N, C_in, H, W, device=device).half()
        time_fp16 = benchmark_layer("FP16", conv_fp16, x_fp16)
        
        # MoDiff CUDA
        conv_modiff = W8A8MoDiffConv2dCUDA(C_in, C_out, K, padding=1).to(device)
        # Initialize weights
        conv_modiff.weight.data = torch.randint(-127, 127, (C_out, C_in * K * K), dtype=torch.int8, device=device)
        conv_modiff.weight_scales.data = torch.ones(C_out, device=device)
        
        # MoDiff expects FP32 input and handles quantization internally
        x_modiff = x_fp32.clone()
        time_modiff = benchmark_layer("MoDiff", conv_modiff, x_modiff)
        
        speedup = time_fp16 / time_modiff
        
        shape_str = f"({N}, {C}, {H}, {W})"
        print(f"{shape_str:<30} {time_fp32:>10.3f} ms   {time_fp16:>10.3f} ms   {time_modiff:>10.3f} ms   {speedup:>9.2f}x")

if __name__ == "__main__":
    run_benchmark()
