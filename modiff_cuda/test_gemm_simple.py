import torch
import modiff_cuda
import time

print("="*60)
print("Testing GEMM W8A8 CUDA Kernel")
print("="*60)

M, N, K = 1024, 128, 128

input_int8 = torch.randint(-127, 127, (M, K), dtype=torch.int8, device='cuda')
weight_int8 = torch.randint(-127, 127, (N, K), dtype=torch.int8, device='cuda')

scale_A = torch.ones((M), dtype=torch.float16, device='cuda') * 0.01
scale_B = torch.ones((N), dtype=torch.float16, device='cuda') * 0.01

print(f"Input: {M}x{K}")
print(f"Weight: {N}x{K}")
print(f"Expected output: {M}x{N}")

print("\nRunning GEMM CUDA kernel...")
try:
    out_cuda = modiff_cuda.gemm_w8a8(input_int8, weight_int8, scale_A, scale_B)
    torch.cuda.synchronize()
    print(f"✓ Kernel executed successfully!")
    print(f"Output shape: {out_cuda.shape}")
    print(f"Output dtype: {out_cuda.dtype}")
    print(f"Output range: [{out_cuda.min().item():.3f}, {out_cuda.max().item():.3f}]")
    
    # Benchmark
    warmup = 10
    for _ in range(warmup):
        _ = modiff_cuda.gemm_w8a8(input_int8, weight_int8, scale_A, scale_B)
    
    torch.cuda.synchronize()
    iters = 100
    start = time.time()
    for _ in range(iters):
        _ = modiff_cuda.gemm_w8a8(input_int8, weight_int8, scale_A, scale_B)
    torch.cuda.synchronize()
    end = time.time()
    custom_time = (end - start) / iters * 1000
    print(f"\n✓ Custom CUDA time: {custom_time:.3f} ms")
    
except RuntimeError as e:
    print(f"✗ Kernel failed: {e}")
