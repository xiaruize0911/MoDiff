"""
Benchmarks for MoDiff Triton Kernels

Compares performance of:
    - FP16 baseline
    - W8A8 MoDiff
    - W4A4 MoDiff
"""

import torch
import torch.nn as nn
import time
from typing import Callable, Dict, List
import sys
import os

# Add parent directory to path
modiff_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, modiff_dir)

from modiff_triton.nn.linear import W8A8MoDiffLinear, W4A4MoDiffLinear
from modiff_triton.nn.conv import W8A8MoDiffConv2d, W4A4MoDiffConv2d


def benchmark_fn(fn: Callable, warmup: int = 10, iterations: int = 100) -> float:
    """Benchmark a function and return average time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # Convert to ms


def benchmark_linear(
    batch_size: int = 32,
    seq_len: int = 512,
    in_features: int = 1024,
    out_features: int = 1024,
    num_timesteps: int = 20,
) -> Dict[str, float]:
    """Benchmark linear layers."""
    print(f"\n--- Linear Benchmark ---")
    print(f"Shape: [{batch_size}, {seq_len}, {in_features}] -> [{batch_size}, {seq_len}, {out_features}]")
    print(f"Timesteps: {num_timesteps}")
    
    # Create layers
    fp16_linear = nn.Linear(in_features, out_features).cuda().half()
    w8a8_linear = W8A8MoDiffLinear.from_linear(fp16_linear.float()).cuda()
    w4a4_linear = W4A4MoDiffLinear.from_linear(fp16_linear.float()).cuda()
    
    # Input
    x = torch.randn(batch_size, seq_len, in_features, device='cuda', dtype=torch.float16)
    
    results = {}
    
    # FP16 baseline
    def fp16_forward():
        with torch.no_grad():
            return fp16_linear(x)
    
    results['FP16'] = benchmark_fn(fp16_forward)
    print(f"FP16:     {results['FP16']:.3f} ms")
    
    # W8A8 single forward
    def w8a8_forward():
        w8a8_linear.reset_cache()
        w8a8_linear.set_modulation(False)
        return w8a8_linear(x.float())
    
    results['W8A8 (no mod)'] = benchmark_fn(w8a8_forward)
    print(f"W8A8 (no mod): {results['W8A8 (no mod)']:.3f} ms")
    
    # W8A8 with modulation (simulating diffusion)
    def w8a8_modulated():
        w8a8_linear.reset_cache()
        w8a8_linear.set_modulation(True)
        for t in range(num_timesteps):
            x_t = x.float() + torch.randn_like(x, dtype=torch.float32) * 0.01
            _ = w8a8_linear(x_t)
    
    results['W8A8 MoDiff'] = benchmark_fn(w8a8_modulated, warmup=5, iterations=20) / num_timesteps
    print(f"W8A8 MoDiff:  {results['W8A8 MoDiff']:.3f} ms/step")
    
    # W4A4 with modulation
    def w4a4_modulated():
        w4a4_linear.reset_cache()
        w4a4_linear.set_modulation(True)
        for t in range(num_timesteps):
            x_t = x.float() + torch.randn_like(x, dtype=torch.float32) * 0.01
            _ = w4a4_linear(x_t)
    
    results['W4A4 MoDiff'] = benchmark_fn(w4a4_modulated, warmup=5, iterations=20) / num_timesteps
    print(f"W4A4 MoDiff:  {results['W4A4 MoDiff']:.3f} ms/step")
    
    # Speedup
    print(f"\nSpeedup vs FP16:")
    print(f"  W8A8 MoDiff: {results['FP16'] / results['W8A8 MoDiff']:.2f}x")
    print(f"  W4A4 MoDiff: {results['FP16'] / results['W4A4 MoDiff']:.2f}x")
    
    return results


def benchmark_conv2d(
    batch_size: int = 8,
    in_channels: int = 256,
    out_channels: int = 256,
    height: int = 32,
    width: int = 32,
    kernel_size: int = 3,
    num_timesteps: int = 20,
) -> Dict[str, float]:
    """Benchmark Conv2d layers."""
    print(f"\n--- Conv2d Benchmark ---")
    print(f"Shape: [{batch_size}, {in_channels}, {height}, {width}] -> [{batch_size}, {out_channels}, {height}, {width}]")
    print(f"Kernel: {kernel_size}x{kernel_size}")
    print(f"Timesteps: {num_timesteps}")
    
    # Create layers
    fp16_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1).cuda().half()
    w8a8_conv = W8A8MoDiffConv2d.from_conv2d(fp16_conv.float()).cuda()
    w4a4_conv = W4A4MoDiffConv2d.from_conv2d(fp16_conv.float()).cuda()
    
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float16)
    
    results = {}
    
    # FP16
    def fp16_forward():
        with torch.no_grad():
            return fp16_conv(x)
    
    results['FP16'] = benchmark_fn(fp16_forward)
    print(f"FP16:     {results['FP16']:.3f} ms")
    
    # W8A8 with modulation
    def w8a8_modulated():
        w8a8_conv.reset_cache()
        w8a8_conv.set_modulation(True)
        for t in range(num_timesteps):
            x_t = x.float() + torch.randn_like(x, dtype=torch.float32) * 0.01
            _ = w8a8_conv(x_t)
    
    results['W8A8 MoDiff'] = benchmark_fn(w8a8_modulated, warmup=5, iterations=20) / num_timesteps
    print(f"W8A8 MoDiff:  {results['W8A8 MoDiff']:.3f} ms/step")
    
    # W4A4 with modulation
    def w4a4_modulated():
        w4a4_conv.reset_cache()
        w4a4_conv.set_modulation(True)
        for t in range(num_timesteps):
            x_t = x.float() + torch.randn_like(x, dtype=torch.float32) * 0.01
            _ = w4a4_conv(x_t)
    
    results['W4A4 MoDiff'] = benchmark_fn(w4a4_modulated, warmup=5, iterations=20) / num_timesteps
    print(f"W4A4 MoDiff:  {results['W4A4 MoDiff']:.3f} ms/step")
    
    print(f"\nSpeedup vs FP16:")
    print(f"  W8A8 MoDiff: {results['FP16'] / results['W8A8 MoDiff']:.2f}x")
    print(f"  W4A4 MoDiff: {results['FP16'] / results['W4A4 MoDiff']:.2f}x")
    
    return results


def benchmark_memory():
    """Benchmark memory usage."""
    print(f"\n--- Memory Benchmark ---")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    in_features = 4096
    out_features = 4096
    batch_size = 32
    seq_len = 512
    
    # FP16
    fp16_linear = nn.Linear(in_features, out_features).cuda().half()
    fp16_mem = fp16_linear.weight.numel() * 2  # 2 bytes per FP16
    
    # W8A8
    w8a8_linear = W8A8MoDiffLinear.from_linear(fp16_linear.float())
    w8a8_mem = w8a8_linear.weight_int8.numel() * 1  # 1 byte per INT8
    
    # W4A4 (packed)
    w4a4_linear = W4A4MoDiffLinear.from_linear(fp16_linear.float())
    w4a4_mem = w4a4_linear.weight_packed.numel() * 1  # 0.5 byte per INT4 (packed)
    
    print(f"Weight memory:")
    print(f"  FP16: {fp16_mem / 1024 / 1024:.2f} MB")
    print(f"  W8A8: {w8a8_mem / 1024 / 1024:.2f} MB ({fp16_mem / w8a8_mem:.1f}x compression)")
    print(f"  W4A4: {w4a4_mem / 1024 / 1024:.2f} MB ({fp16_mem / w4a4_mem:.1f}x compression)")
    
    # Cache memory (MoDiff overhead)
    x = torch.randn(batch_size, seq_len, in_features, device='cuda')
    a_cache_size = x.numel() * 2  # FP16 cache
    o_cache_size = batch_size * seq_len * out_features * 2
    
    print(f"\nMoDiff cache overhead (per layer):")
    print(f"  â cache: {a_cache_size / 1024 / 1024:.2f} MB")
    print(f"  ô cache: {o_cache_size / 1024 / 1024:.2f} MB")
    print(f"  Total:   {(a_cache_size + o_cache_size) / 1024 / 1024:.2f} MB")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("MoDiff Triton Kernel Benchmarks")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return
    
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Linear benchmarks
    benchmark_linear(
        batch_size=32,
        seq_len=512,
        in_features=1024,
        out_features=1024,
    )
    
    # Conv2d benchmarks
    benchmark_conv2d(
        batch_size=8,
        in_channels=256,
        out_channels=256,
        height=32,
        width=32,
    )
    
    # Memory benchmarks
    benchmark_memory()
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
