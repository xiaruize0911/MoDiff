
import torch
import time
import numpy as np
import sys
import os

# Add workspace to path for imports
sys.path.insert(0, os.getcwd())

from modiff_triton.kernels.gemm_w8a8 import gemm_w8a8, gemm_w8a8_accum
from modiff_triton.kernels.gemm_w4a4 import gemm_w4a4, gemm_w4a4_accum

def pack_int4(tensor):
    tensor = tensor.clamp(-8, 7).to(torch.int8) + 8
    lo = (tensor[:, 0::2] & 0xF)
    hi = (tensor[:, 1::2] & 0xF) << 4
    return (lo | hi).to(torch.int8)

def benchmark_modulated_gemm(M=1024, N=1024, K=1024, num_runs=100):
    print(f"\nBenchmarking Modulated vs Standard GEMM: M={M}, N={N}, K={K}")
    print("-" * 60)
    
    # Setup
    scale_a = torch.tensor([1.0/127.0]).cuda()
    scale_b = torch.tensor([1.0/127.0]).cuda()
    
    # --- INT8 ---
    a_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8).cuda()
    b_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8).cuda()
    cache = torch.randn(M, N).cuda()
    
    # Warmup INT8
    for _ in range(10):
        _ = gemm_w8a8(a_int8, b_int8, scale_a, scale_b)
        _ = gemm_w8a8_accum(a_int8, b_int8, scale_a, scale_b, cache)
    
    # Standard INT8
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = gemm_w8a8(a_int8, b_int8, scale_a, scale_b)
    torch.cuda.synchronize()
    int8_std_time = (time.time() - start) * 1000 / num_runs
    
    # Modulated INT8 (MoDiff)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = gemm_w8a8_accum(a_int8, b_int8, scale_a, scale_b, cache)
    torch.cuda.synchronize()
    int8_modiff_time = (time.time() - start) * 1000 / num_runs
    
    # --- INT4 ---
    K_even = K if K % 2 == 0 else K + 1
    a_packed = torch.randint(-128, 127, (M, K_even // 2), dtype=torch.int8).cuda()
    b_packed = torch.randint(-128, 127, (K_even // 2, N), dtype=torch.int8).cuda()
    
    # Warmup INT4
    for _ in range(10):
        _ = gemm_w4a4(a_packed, b_packed, scale_a, scale_b, K_even)
        _ = gemm_w4a4_accum(a_packed, b_packed, scale_a, scale_b, K_even, cache)
    
    # Standard INT4
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = gemm_w4a4(a_packed, b_packed, scale_a, scale_b, K_even)
    torch.cuda.synchronize()
    int4_std_time = (time.time() - start) * 1000 / num_runs
    
    # Modulated INT4 (MoDiff)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = gemm_w4a4_accum(a_packed, b_packed, scale_a, scale_b, K_even, cache)
    torch.cuda.synchronize()
    int4_modiff_time = (time.time() - start) * 1000 / num_runs
    
    print(f"INT8 Standard: {int8_std_time:.4f} ms")
    print(f"INT8 MoDiff  : {int8_modiff_time:.4f} ms (Overhead: {(int8_modiff_time/int8_std_time - 1)*100:.2f}%)")
    print(f"INT4 Standard: {int4_std_time:.4f} ms")
    print(f"INT4 MoDiff  : {int4_modiff_time:.4f} ms (Overhead: {(int4_modiff_time/int4_std_time - 1)*100:.2f}%)")

if __name__ == "__main__":
    benchmark_modulated_gemm(1024, 1024, 1024)
    benchmark_modulated_gemm(1024, 512, 512)
