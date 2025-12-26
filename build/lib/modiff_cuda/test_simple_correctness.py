#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import conv_simple
import time

def test_simple_conv():
    print("=" * 60)
    print("Testing Simple INT8 Convolution Kernel")
    print("=" * 60)
    
    # Test parameters
    N, C_in, H, W = 1, 64, 32, 32
    C_out = 64
    K = 3
    stride = 1
    padding = 1
    
    # Create test data
    input_int8 = torch.randint(-128, 127, (N, H, W, C_in), dtype=torch.int8, device='cuda')
    weight_int8 = torch.randint(-128, 127, (C_out, K, K, C_in), dtype=torch.int8, device='cuda')
    
    print(f"\nInput shape: {input_int8.shape} (NHWC)")
    print(f"Weight shape: {weight_int8.shape} (OIHW)")
    
    # Run CUDA kernel
    print("\nRunning simple CUDA INT8 kernel...")
    start = time.time()
    output_cuda = conv_simple.conv2d_simple(input_int8, weight_int8, stride=stride, padding=padding)
    torch.cuda.synchronize()
    elapsed_cuda = time.time() - start
    
    print(f"✓ CUDA kernel completed")
    print(f"  Output shape: {output_cuda.shape}")
    print(f"  Output dtype: {output_cuda.dtype}")
    print(f"  Time: {elapsed_cuda*1000:.2f} ms")
    
    # Verify against PyTorch
    print("\nVerifying against PyTorch FP32...")
    
    # Convert to NCHW for PyTorch
    input_fp32 = input_int8.float().permute(0, 3, 1, 2)  # NHWC -> NCHW
    weight_fp32 = weight_int8.float().permute(0, 3, 1, 2)  # OIHW -> OIHW already correct
    
    output_torch = F.conv2d(input_fp32, weight_fp32, stride=stride, padding=padding)
    output_torch = output_torch.permute(0, 2, 3, 1).to(torch.int32)  # NCHW -> NHWC
    
    # Compare
    max_diff = torch.abs(output_cuda.cpu() - output_torch.cpu()).max().item()
    mean_diff = torch.abs(output_cuda.cpu() - output_torch.cpu()).float().mean().item()
    
    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff:.4f}")
    
    if max_diff == 0:
        print("✓ Perfect match! Kernel is correct.")
    elif max_diff < 10:
        print("✓ Close match (acceptable rounding differences)")
    else:
        print(f"✗ Verification failed - difference too large: {max_diff}")
        
        # Debug: show some values
        print("\nDebug: First few values")
        print("CUDA output:", output_cuda[0, 0, 0, :10].cpu().numpy())
        print("Torch output:", output_torch[0, 0, 0, :10].cpu().numpy())
        return False
    
    # Benchmark
    print("\n" + "=" * 60)
    print("Benchmarking")
    print("=" * 60)
    
    # Warmup
    for _ in range(10):
        _ = conv_simple.conv2d_simple(input_int8, weight_int8, stride, padding)
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        output = conv_simple.conv2d_simple(input_int8, weight_int8, stride, padding)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iterations
    
    print(f"Simple INT8 Conv: {elapsed*1000:.3f} ms/iter")
    
    # Compare with FP32
    for _ in range(10):
        _ = F.conv2d(input_fp32, weight_fp32, stride=stride, padding=padding)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = F.conv2d(input_fp32, weight_fp32, stride=stride, padding=padding)
    torch.cuda.synchronize()
    elapsed_fp32 = (time.time() - start) / iterations
    
    print(f"PyTorch FP32 Conv: {elapsed_fp32*1000:.3f} ms/iter")
    print(f"Speedup: {elapsed_fp32/elapsed:.2f}x {'FASTER' if elapsed < elapsed_fp32 else 'SLOWER'}")
    
    return True

if __name__ == "__main__":
    success = test_simple_conv()
    exit(0 if success else 1)
