
import torch
import time
import numpy as np
import triton
from tqdm import tqdm
import sys
import os

# Add workspace to path for imports
sys.path.insert(0, os.getcwd())

from modiff_triton.kernels.gemm_w8a8 import gemm_w8a8
from modiff_triton.kernels.gemm_w4a4 import gemm_w4a4

def pack_int4(tensor):
    """Pack two INT8 values into one byte as INT4 x 2."""
    # Assume input is in range [-8, 7]
    tensor = tensor.clamp(-8, 7).to(torch.int8) + 8
    lo = (tensor[:, 0::2] & 0xF)
    hi = (tensor[:, 1::2] & 0xF) << 4
    return (lo | hi).to(torch.int8)

def benchmark_raw_gemm(M=1024, N=1024, K=1024, num_runs=100):
    print(f"\nBenchmarking Matrix Size: M={M}, N={N}, K={K}")
    print("-" * 60)
    
    # Setup FP32/FP16
    a_fp32 = torch.randn(M, K).cuda()
    b_fp32 = torch.randn(K, N).cuda()
    
    # PyTorch FP32
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = torch.mm(a_fp32, b_fp32)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) * 1000 / num_runs
    
    # PyTorch FP16
    a_fp16 = a_fp32.half()
    b_fp16 = b_fp32.half()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = torch.mm(a_fp16, b_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) * 1000 / num_runs
    
    # Triton INT8
    a_int8 = (a_fp32 * 127).to(torch.int8)
    b_int8 = (b_fp32 * 127).to(torch.int8)
    scale_a = torch.tensor([1.0/127.0]).cuda()
    scale_b = torch.tensor([1.0/127.0]).cuda()
    
    # Warmup Triton INT8
    gemm_w8a8(a_int8, b_int8, scale_a, scale_b)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = gemm_w8a8(a_int8, b_int8, scale_a, scale_b)
    torch.cuda.synchronize()
    int8_time = (time.time() - start) * 1000 / num_runs
    
    # Triton INT4
    # Note: K must be even for packed INT4
    K_even = K if K % 2 == 0 else K + 1
    if K % 2 != 0:
        a_int4_raw = torch.randn(M, K_even).cuda()
        b_int4_raw = torch.randn(K_even, N).cuda()
    else:
        a_int4_raw = a_fp32
        b_int4_raw = b_fp32
        
    a_packed = pack_int4((a_int4_raw * 7).to(torch.int8))
    # We need weight in [K//2, N]
    b_packed = pack_int4((b_int4_raw.t() * 7).to(torch.int8)).t().contiguous()
    
    # Warmup Triton INT4
    gemm_w4a4(a_packed, b_packed, scale_a, scale_b, K_even)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = gemm_w4a4(a_packed, b_packed, scale_a, scale_b, K_even)
    torch.cuda.synchronize()
    int4_time = (time.time() - start) * 1000 / num_runs
    
    # Results
    print(f"FP32: {fp32_time:.4f} ms")
    print(f"FP16: {fp16_time:.4f} ms (Speedup vs FP32: {fp32_time/fp16_time:.2f}x)")
    print(f"INT8: {int8_time:.4f} ms (Speedup vs FP32: {fp32_time/int8_time:.2f}x, vs FP16: {fp16_time/int8_time:.2f}x)")
    print(f"INT4: {int4_time:.4f} ms (Speedup vs FP32: {fp32_time/int4_time:.2f}x, vs INT8: {int8_time/int4_time:.2f}x)")

if __name__ == "__main__":
    # Common sizes in UNet
    # Intermediate layers often have large K, N
    benchmark_raw_gemm(1024, 1024, 1024)
    benchmark_raw_gemm(2048, 2048, 2048)
    benchmark_raw_gemm(4096, 4096, 4096)
