#!/usr/bin/env python3
"""
Use PyTorch's native quantized convolution (backed by cuDNN/oneDNN INT8 kernels)
This is the most pragmatic approach - leveraging highly optimized libraries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.quantized import Conv2d as QConv2d
import time

def test_quantized_conv():
    print("Testing PyTorch Quantized INT8 Convolution")
    print("=" * 60)
    
    # Test parameters
    N, C_in, H, W = 4, 128, 32, 32
    C_out = 128
    K = 3
    stride = 1
    padding = 1
    
    # Create FP32 convolution for comparison
    conv_fp32 = nn.Conv2d(C_in, C_out, K, stride=stride, padding=padding).cuda()
    
    # Create quantized convolution
    # PyTorch's quantized conv uses INT8 weights and activations
    conv_int8 = torch.quantization.quantize_dynamic(
        conv_fp32, {nn.Conv2d}, dtype=torch.qint8
    )
    
    # Create test input
    input_fp32 = torch.randn(N, C_in, H, W).cuda()
    
    print(f"\nInput shape: {input_fp32.shape}")
    print(f"Conv FP32: {conv_fp32}")
    print(f"Conv INT8: {conv_int8}")
    
    # Benchmark FP32
    print("\n" + "=" * 60)
    print("Benchmarking FP32 Convolution")
    print("=" * 60)
    
    # Warmup
    for _ in range(10):
        _ = conv_fp32(input_fp32)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        output_fp32 = conv_fp32(input_fp32)
    torch.cuda.synchronize()
    elapsed_fp32 = (time.time() - start) / iterations
    
    print(f"FP32 Conv time: {elapsed_fp32*1000:.3f} ms")
    print(f"Output shape: {output_fp32.shape}")
    
    # Benchmark INT8 (on CPU - PyTorch's quantized ops are CPU-only by default)
    print("\n" + "=" * 60)
    print("Note: PyTorch quantized Conv2d is CPU-only")
    print("For GPU INT8, need cuDNN or TensorRT integration")
    print("=" * 60)
    
    # Let's try TorchScript quantization for GPU
    print("\nTrying TorchScript quantization...")
    
    # Trace the model
    example_input = torch.randn(1, C_in, H, W).cuda()
    traced_model = torch.jit.trace(conv_fp32, example_input)
    
    # Benchmark traced model
    for _ in range(10):
        _ = traced_model(input_fp32)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        output_traced = traced_model(input_fp32)
    torch.cuda.synchronize()
    elapsed_traced = (time.time() - start) / iterations
    
    print(f"TorchScript FP32 time: {elapsed_traced*1000:.3f} ms")
    
    # Try using torch.compile for optimization (PyTorch 2.0+)
    try:
        print("\nTrying torch.compile optimization...")
        compiled_conv = torch.compile(conv_fp32, mode="max-autotune")
        
        # Warmup
        for _ in range(10):
            _ = compiled_conv(input_fp32)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            output_compiled = compiled_conv(input_fp32)
        torch.cuda.synchronize()
        elapsed_compiled = (time.time() - start) / iterations
        
        print(f"torch.compile FP32 time: {elapsed_compiled*1000:.3f} ms")
        speedup = elapsed_fp32 / elapsed_compiled
        print(f"Speedup vs baseline: {speedup:.2f}x")
        
    except Exception as e:
        print(f"torch.compile not available: {e}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline FP32: {elapsed_fp32*1000:.3f} ms")
    if 'elapsed_compiled' in locals():
        print(f"Optimized FP32: {elapsed_compiled*1000:.3f} ms ({elapsed_fp32/elapsed_compiled:.2f}x faster)")
    
    print("\nFor true INT8 performance on GPU, consider:")
    print("1. TensorRT - NVIDIA's inference optimizer with INT8 support")
    print("2. cuDNN with explicit INT8 APIs")
    print("3. Custom CUTLASS-based kernels (like ViDiT-Q)")

if __name__ == "__main__":
    test_quantized_conv()
