"""
Benchmark script for MoDiff INT4 convolution kernels.
Compares custom INT4 convolution against PyTorch FP32/FP16 baseline and INT8 kernels.
"""

import torch
import argparse
from icecream import ic

try:
    import modiff_int4
    import modiff_int8
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import modiff_int4
    import modiff_int8


def test_int4_conv_correctness(batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding):
    """Test correctness of INT4 convolution."""
    print(f"\nTesting correctness: B={batch_size}, C_in={in_channels}, C_out={out_channels}, H={height}, W={width}, K={kernel_size}, S={stride}, P={padding}")
    
    # Create random FP32 input and weight
    x_fp32 = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    weight_fp32 = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32, device="cuda")
    
    # Compute ground truth with FP32
    output_gt = torch.nn.functional.conv2d(x_fp32, weight_fp32, stride=stride, padding=padding)
    
    # Quantize input: per-tensor quantization to INT4 range [-8, 7]
    x_scale = x_fp32.abs().max() / 7.0
    x_int4_packed = modiff_int4.quantize_to_int4_packed(x_fp32, x_scale)
    
    # Quantize weight: per-channel quantization to INT4 range
    weight_scale = weight_fp32.abs().view(out_channels, -1).max(dim=1)[0] / 7.0
    weight_int4_packed = modiff_int4.quantize_weight(weight_fp32, weight_scale)
    
    # Compute INT4 convolution
    output_int4 = modiff_int4.conv2d_int4(x_int4_packed, weight_int4_packed, x_scale, weight_scale, kernel_size, stride, padding)
    
    # Compute error
    max_error = torch.max(torch.abs(output_int4 - output_gt))
    mean_error = torch.mean(torch.abs(output_int4 - output_gt))
    relative_error = mean_error / (output_gt.abs().mean() + 1e-8)
    
    ic(max_error)
    ic(mean_error)
    ic(relative_error)
    
    print(f"Max error: {max_error:.6f}, Mean error: {mean_error:.6f}, Relative error: {relative_error:.6f}")


def benchmark_int4_conv(batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, num_iter=100, num_warmup_iter=20):
    """Benchmark INT4 convolution against PyTorch baselines and INT8."""
    
    print(f"\n{'='*80}")
    print(f"Benchmarking: B={batch_size}, C_in={in_channels}, C_out={out_channels}, H={height}, W={width}, K={kernel_size}, S={stride}, P={padding}")
    print(f"{'='*80}")
    
    # Prepare test data
    input_fp32_list = []
    input_fp16_list = []
    input_int4_list = []
    input_int8_list = []
    input_scale_int4_list = []
    input_scale_int8_list = []
    weight_fp32_list = []
    weight_fp16_list = []
    weight_int4_list = []
    weight_int8_list = []
    weight_scale_int4_list = []
    weight_scale_int8_list = []
    
    for _ in range(num_iter + 1):
        x_fp32 = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
        x_fp16 = x_fp32.to(torch.float16)
        
        weight_fp32 = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32, device="cuda")
        weight_fp16 = weight_fp32.to(torch.float16)
        
        # Quantize to INT4
        x_scale_int4 = x_fp32.abs().max() / 7.0
        x_int4 = modiff_int4.quantize_to_int4_packed(x_fp32, x_scale_int4)
        
        weight_scale_int4 = weight_fp32.abs().view(out_channels, -1).max(dim=1)[0] / 7.0
        weight_int4 = modiff_int4.quantize_weight(weight_fp32, weight_scale_int4)
        
        # Quantize to INT8
        x_scale_int8 = x_fp32.abs().max() / 127.0
        x_int8 = modiff_int8.quantize_tensor(x_fp32, x_scale_int8)
        
        weight_scale_int8 = weight_fp32.abs().view(out_channels, -1).max(dim=1)[0] / 127.0
        weight_int8 = modiff_int8.quantize_weight(weight_fp32, weight_scale_int8)
        
        input_fp32_list.append(x_fp32)
        input_fp16_list.append(x_fp16)
        input_int4_list.append(x_int4)
        input_int8_list.append(x_int8)
        input_scale_int4_list.append(x_scale_int4)
        input_scale_int8_list.append(x_scale_int8)
        weight_fp32_list.append(weight_fp32)
        weight_fp16_list.append(weight_fp16)
        weight_int4_list.append(weight_int4)
        weight_int8_list.append(weight_int8)
        weight_scale_int4_list.append(weight_scale_int4)
        weight_scale_int8_list.append(weight_scale_int8)
    
    # Warmup
    for _ in range(num_warmup_iter):
        torch.nn.functional.conv2d(input_fp32_list[-1], weight_fp32_list[-1], stride=stride, padding=padding)
        torch.nn.functional.conv2d(input_fp16_list[-1], weight_fp16_list[-1], stride=stride, padding=padding)
        modiff_int8.conv2d_int8(input_int8_list[-1], weight_int8_list[-1], input_scale_int8_list[-1], weight_scale_int8_list[-1], kernel_size, stride, padding)
        modiff_int4.conv2d_int4(input_int4_list[-1], weight_int4_list[-1], input_scale_int4_list[-1], weight_scale_int4_list[-1], kernel_size, stride, padding)
        modiff_int4.conv2d_int4_static(input_int4_list[-1], weight_int4_list[-1], input_scale_int4_list[-1], weight_scale_int4_list[-1], kernel_size, stride, padding)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Benchmark FP32
    start.record()
    for i in range(num_iter):
        torch.nn.functional.conv2d(input_fp32_list[i], weight_fp32_list[i], stride=stride, padding=padding)
    end.record()
    torch.cuda.synchronize()
    avg_time_fp32 = start.elapsed_time(end) / num_iter
    
    # Benchmark FP16
    start.record()
    for i in range(num_iter):
        torch.nn.functional.conv2d(input_fp16_list[i], weight_fp16_list[i], stride=stride, padding=padding)
    end.record()
    torch.cuda.synchronize()
    avg_time_fp16 = start.elapsed_time(end) / num_iter
    
    # Benchmark INT8
    start.record()
    for i in range(num_iter):
        modiff_int8.conv2d_int8(input_int8_list[i], weight_int8_list[i], input_scale_int8_list[i], weight_scale_int8_list[i], kernel_size, stride, padding)
    end.record()
    torch.cuda.synchronize()
    avg_time_int8 = start.elapsed_time(end) / num_iter
    
    # Benchmark INT4 dynamic
    start.record()
    for i in range(num_iter):
        modiff_int4.conv2d_int4(input_int4_list[i], weight_int4_list[i], input_scale_int4_list[i], weight_scale_int4_list[i], kernel_size, stride, padding)
    end.record()
    torch.cuda.synchronize()
    avg_time_int4 = start.elapsed_time(end) / num_iter
    
    # Benchmark INT4 static
    start.record()
    for i in range(num_iter):
        modiff_int4.conv2d_int4_static(input_int4_list[i], weight_int4_list[i], input_scale_int4_list[i], weight_scale_int4_list[i], kernel_size, stride, padding)
    end.record()
    torch.cuda.synchronize()
    avg_time_int4_static = start.elapsed_time(end) / num_iter
    
    # Calculate speedup
    speedup_int4_vs_fp32 = avg_time_fp32 / avg_time_int4
    speedup_int4_vs_fp16 = avg_time_fp16 / avg_time_int4
    speedup_int4_vs_int8 = avg_time_int8 / avg_time_int4
    speedup_static_vs_dynamic = avg_time_int4 / avg_time_int4_static
    
    print(f"\nResults:")
    print(f"  FP32 Conv2d:              {avg_time_fp32:.4f} ms")
    print(f"  FP16 Conv2d:              {avg_time_fp16:.4f} ms")
    print(f"  INT8 Conv2d:              {avg_time_int8:.4f} ms (speedup vs FP32: {avg_time_fp32/avg_time_int8:.2f}x)")
    print(f"  INT4 Conv2d (dynamic):    {avg_time_int4:.4f} ms (speedup vs FP32: {speedup_int4_vs_fp32:.2f}x, vs FP16: {speedup_int4_vs_fp16:.2f}x, vs INT8: {speedup_int4_vs_int8:.2f}x)")
    print(f"  INT4 Conv2d (static):     {avg_time_int4_static:.4f} ms (speedup vs dynamic: {speedup_static_vs_dynamic:.2f}x)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MoDiff INT4 Conv2d kernels")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--in_channels", type=int, default=256, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=256, help="Output channels")
    parser.add_argument("--height", type=int, default=64, help="Input height")
    parser.add_argument("--width", type=int, default=64, help="Input width")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--padding", type=int, default=1, help="Padding")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--num_warmup_iter", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--test_correctness", action="store_true", help="Test correctness only")
    
    args = parser.parse_args()
    
    if args.test_correctness:
        test_int4_conv_correctness(
            args.batch_size, args.in_channels, args.out_channels,
            args.height, args.width, args.kernel_size, args.stride, args.padding
        )
    else:
        # Test correctness first
        test_int4_conv_correctness(
            args.batch_size, args.in_channels, args.out_channels,
            args.height, args.width, args.kernel_size, args.stride, args.padding
        )
        
        # Run benchmark
        benchmark_int4_conv(
            args.batch_size, args.in_channels, args.out_channels,
            args.height, args.width, args.kernel_size, args.stride, args.padding,
            args.num_iter, args.num_warmup_iter
        )
        
        # Additional common configurations
        print("\n\n" + "="*80)
        print("Benchmarking common configurations...")
        print("="*80)
        
        # Configuration 1: Large spatial size
        benchmark_int4_conv(4, 128, 128, 128, 128, 3, 1, 1, args.num_iter, args.num_warmup_iter)
        
        # Configuration 2: Small spatial size, more channels
        benchmark_int4_conv(4, 512, 512, 32, 32, 3, 1, 1, args.num_iter, args.num_warmup_iter)
        
        # Configuration 3: Stride 2 downsampling
        benchmark_int4_conv(4, 256, 512, 64, 64, 3, 2, 1, args.num_iter, args.num_warmup_iter)
