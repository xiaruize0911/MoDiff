import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os

# Add current directory to path to find modiff_cuda_backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import modiff_cuda_backend

def benchmark_layer(name, func, args, num_iters=100):
    # Warmup
    for _ in range(10):
        _ = func(*args)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = func(*args)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / num_iters * 1000

def run_benchmark():
    device = torch.device("cuda")
    print(f"Benchmarking on {torch.cuda.get_device_name(0)}")
    print("-" * 100)
    print(f"{'Shape (N, C, H, W)':<30} {'PyTorch FP16':<15} {'MoDiff CUDA':<15} {'Speedup':<10}")
    print("-" * 100)

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
        stride = 1
        padding = 1
        
        # PyTorch FP16
        input_fp16 = torch.randn(N, C_in, H, W, device=device, dtype=torch.float16)
        weight_fp16 = torch.randn(C_out, C_in, K, K, device=device, dtype=torch.float16)
        
        time_fp16 = benchmark_layer("FP16", torch.nn.functional.conv2d, (input_fp16, weight_fp16, None, stride, padding))
        
        # MoDiff CUDA
        # Input: NHWC, int8
        input_int8 = torch.randint(-127, 127, (N, H, W, C_in), dtype=torch.int8, device=device)
        # Weight: [C_out, C_in * K * K], int8 (Implicit GEMM layout)
        weight_int8 = torch.randint(-127, 127, (C_out, C_in * K * K), dtype=torch.int8, device=device)
        
        act_scale = torch.tensor([0.01], device=device, dtype=torch.float32)
        weight_scales = torch.ones(C_out, device=device, dtype=torch.float32) * 0.01
        
        # Args: input, weight, act_scale, weight_scales, kernel_size, stride, padding, compute_max
        modiff_args = (input_int8, weight_int8, act_scale, weight_scales, K, stride, padding, False)
        
        time_modiff = benchmark_layer("MoDiff", modiff_cuda_backend.conv2d_fast_w8a8, modiff_args)
        
        speedup = time_fp16 / time_modiff
        
        shape_str = f"({N}, {C}, {H}, {W})"
        print(f"{shape_str:<30} {time_fp16:>10.3f} ms   {time_modiff:>10.3f} ms   {speedup:>9.2f}x")

if __name__ == "__main__":
    run_benchmark()
