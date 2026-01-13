"""
Benchmark script for MoDiff fused kernels (GroupNorm + SiLU, etc.)
Compares custom fused operations against sequential PyTorch operations.
"""

import torch
import argparse
from icecream import ic

try:
    import fused_conv_norm_act
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import fused_conv_norm_act


def test_fused_groupnorm_silu_correctness(batch_size, channels, height, width, num_groups):
    """Test correctness of fused GroupNorm + SiLU."""
    print(f"\nTesting fused_groupnorm_silu correctness: B={batch_size}, C={channels}, H={height}, W={width}, G={num_groups}")
    
    # Create random input
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device="cuda")
    
    # Create GroupNorm parameters
    weight = torch.randn(channels, dtype=torch.float32, device="cuda")
    bias = torch.randn(channels, dtype=torch.float32, device="cuda")
    
    # Ground truth: PyTorch GroupNorm + SiLU
    groupnorm = torch.nn.GroupNorm(num_groups, channels).to("cuda")
    groupnorm.weight.data = weight.clone()
    groupnorm.bias.data = bias.clone()
    
    output_gt = torch.nn.functional.silu(groupnorm(x))
    
    # Fused kernel
    output_fused = fused_conv_norm_act.fused_groupnorm_silu(x, weight, bias, num_groups, 1e-5)
    
    # Compute error
    max_error = torch.max(torch.abs(output_fused - output_gt))
    mean_error = torch.mean(torch.abs(output_fused - output_gt))
    relative_error = mean_error / (output_gt.abs().mean() + 1e-8)
    
    ic(max_error)
    ic(mean_error)
    ic(relative_error)
    
    print(f"Max error: {max_error:.6f}, Mean error: {mean_error:.6f}, Relative error: {relative_error:.6f}")


def benchmark_fused_groupnorm_silu(batch_size, channels, height, width, num_groups, num_iter=100, num_warmup_iter=20):
    """Benchmark fused GroupNorm + SiLU."""
    
    print(f"\n{'='*80}")
    print(f"Benchmarking fused_groupnorm_silu: B={batch_size}, C={channels}, H={height}, W={width}, G={num_groups}")
    print(f"{'='*80}")
    
    # Prepare test data
    input_list = []
    weight_list = []
    bias_list = []
    groupnorm_list = []
    
    for _ in range(num_iter + 1):
        x = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device="cuda")
        weight = torch.randn(channels, dtype=torch.float32, device="cuda")
        bias = torch.randn(channels, dtype=torch.float32, device="cuda")
        
        groupnorm = torch.nn.GroupNorm(num_groups, channels).to("cuda")
        groupnorm.weight.data = weight.clone()
        groupnorm.bias.data = bias.clone()
        
        input_list.append(x)
        weight_list.append(weight)
        bias_list.append(bias)
        groupnorm_list.append(groupnorm)
    
    # Warmup
    for _ in range(num_warmup_iter):
        torch.nn.functional.silu(groupnorm_list[-1](input_list[-1]))
        fused_conv_norm_act.fused_groupnorm_silu(input_list[-1], weight_list[-1], bias_list[-1], num_groups, 1e-5)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Benchmark PyTorch sequential
    start.record()
    for i in range(num_iter):
        torch.nn.functional.silu(groupnorm_list[i](input_list[i]))
    end.record()
    torch.cuda.synchronize()
    avg_time_pytorch = start.elapsed_time(end) / num_iter
    
    # Benchmark fused kernel
    start.record()
    for i in range(num_iter):
        fused_conv_norm_act.fused_groupnorm_silu(input_list[i], weight_list[i], bias_list[i], num_groups, 1e-5)
    end.record()
    torch.cuda.synchronize()
    avg_time_fused = start.elapsed_time(end) / num_iter
    
    # Calculate speedup
    speedup = avg_time_pytorch / avg_time_fused
    
    print(f"\nResults:")
    print(f"  PyTorch (GroupNorm + SiLU):   {avg_time_pytorch:.4f} ms")
    print(f"  Fused GroupNorm+SiLU:         {avg_time_fused:.4f} ms (speedup: {speedup:.2f}x)")


def test_fused_conv_groupnorm_silu_correctness(batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, num_groups):
    """Test correctness of fused Conv + GroupNorm + SiLU."""
    print(f"\nTesting fused_conv_groupnorm_silu correctness: B={batch_size}, C_in={in_channels}, C_out={out_channels}, H={height}, W={width}, K={kernel_size}, G={num_groups}")
    
    # Create random input
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    
    # Create Conv parameters
    conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32, device="cuda")
    conv_bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    # Create GroupNorm parameters
    norm_weight = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    norm_bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    # Ground truth: PyTorch Conv + GroupNorm + SiLU
    conv_out = torch.nn.functional.conv2d(x, conv_weight, conv_bias, stride=stride, padding=padding)
    
    groupnorm = torch.nn.GroupNorm(num_groups, out_channels).to("cuda")
    groupnorm.weight.data = norm_weight.clone()
    groupnorm.bias.data = norm_bias.clone()
    
    output_gt = torch.nn.functional.silu(groupnorm(conv_out))
    
    # Fused kernel (two-pass version)
    output_fused = fused_conv_norm_act.fused_conv_groupnorm_silu_two_pass(
        x, conv_weight, conv_bias, norm_weight, norm_bias, num_groups, kernel_size, stride, padding, 1e-5
    )
    
    # Compute error
    max_error = torch.max(torch.abs(output_fused - output_gt))
    mean_error = torch.mean(torch.abs(output_fused - output_gt))
    relative_error = mean_error / (output_gt.abs().mean() + 1e-8)
    
    ic(max_error)
    ic(mean_error)
    ic(relative_error)
    
    print(f"Max error: {max_error:.6f}, Mean error: {mean_error:.6f}, Relative error: {relative_error:.6f}")


def benchmark_fused_conv_groupnorm_silu(batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, num_groups, num_iter=100, num_warmup_iter=20):
    """Benchmark fused Conv + GroupNorm + SiLU."""
    
    print(f"\n{'='*80}")
    print(f"Benchmarking fused_conv_groupnorm_silu: B={batch_size}, C_in={in_channels}, C_out={out_channels}, H={height}, W={width}, K={kernel_size}, G={num_groups}")
    print(f"{'='*80}")
    
    # Prepare test data
    input_list = []
    conv_weight_list = []
    conv_bias_list = []
    norm_weight_list = []
    norm_bias_list = []
    groupnorm_list = []
    
    for _ in range(num_iter + 1):
        x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
        conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32, device="cuda")
        conv_bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
        norm_weight = torch.randn(out_channels, dtype=torch.float32, device="cuda")
        norm_bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
        
        groupnorm = torch.nn.GroupNorm(num_groups, out_channels).to("cuda")
        groupnorm.weight.data = norm_weight.clone()
        groupnorm.bias.data = norm_bias.clone()
        
        input_list.append(x)
        conv_weight_list.append(conv_weight)
        conv_bias_list.append(conv_bias)
        norm_weight_list.append(norm_weight)
        norm_bias_list.append(norm_bias)
        groupnorm_list.append(groupnorm)
    
    # Warmup
    for _ in range(num_warmup_iter):
        # PyTorch sequential
        conv_out = torch.nn.functional.conv2d(input_list[-1], conv_weight_list[-1], conv_bias_list[-1], stride=stride, padding=padding)
        torch.nn.functional.silu(groupnorm_list[-1](conv_out))
        
        # Fused
        fused_conv_norm_act.fused_conv_groupnorm_silu_two_pass(
            input_list[-1], conv_weight_list[-1], conv_bias_list[-1], 
            norm_weight_list[-1], norm_bias_list[-1], num_groups, kernel_size, stride, padding, 1e-5
        )
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Benchmark PyTorch sequential
    start.record()
    for i in range(num_iter):
        conv_out = torch.nn.functional.conv2d(input_list[i], conv_weight_list[i], conv_bias_list[i], stride=stride, padding=padding)
        torch.nn.functional.silu(groupnorm_list[i](conv_out))
    end.record()
    torch.cuda.synchronize()
    avg_time_pytorch = start.elapsed_time(end) / num_iter
    
    # Benchmark fused kernel
    start.record()
    for i in range(num_iter):
        fused_conv_norm_act.fused_conv_groupnorm_silu_two_pass(
            input_list[i], conv_weight_list[i], conv_bias_list[i], 
            norm_weight_list[i], norm_bias_list[i], num_groups, kernel_size, stride, padding, 1e-5
        )
    end.record()
    torch.cuda.synchronize()
    avg_time_fused = start.elapsed_time(end) / num_iter
    
    # Calculate speedup
    speedup = avg_time_pytorch / avg_time_fused
    
    print(f"\nResults:")
    print(f"  PyTorch (Conv + GroupNorm + SiLU):   {avg_time_pytorch:.4f} ms")
    print(f"  Fused Conv+GroupNorm+SiLU:           {avg_time_fused:.4f} ms (speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MoDiff fused kernels")
    parser.add_argument("--kernel", type=str, default="all", choices=["all", "groupnorm_silu", "conv_groupnorm_silu"], help="Which kernel to benchmark")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--channels", type=int, default=256, help="Number of channels (for groupnorm_silu)")
    parser.add_argument("--in_channels", type=int, default=256, help="Input channels (for conv_groupnorm_silu)")
    parser.add_argument("--out_channels", type=int, default=256, help="Output channels (for conv_groupnorm_silu)")
    parser.add_argument("--height", type=int, default=64, help="Input height")
    parser.add_argument("--width", type=int, default=64, help="Input width")
    parser.add_argument("--num_groups", type=int, default=32, help="Number of groups for GroupNorm")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size (for conv_groupnorm_silu)")
    parser.add_argument("--stride", type=int, default=1, help="Stride (for conv_groupnorm_silu)")
    parser.add_argument("--padding", type=int, default=1, help="Padding (for conv_groupnorm_silu)")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--num_warmup_iter", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--test_correctness", action="store_true", help="Test correctness only")
    
    args = parser.parse_args()
    
    if args.kernel == "all" or args.kernel == "groupnorm_silu":
        print("\n" + "="*80)
        print("FUSED GROUPNORM + SILU")
        print("="*80)
        
        if args.test_correctness:
            test_fused_groupnorm_silu_correctness(args.batch_size, args.channels, args.height, args.width, args.num_groups)
        else:
            test_fused_groupnorm_silu_correctness(args.batch_size, args.channels, args.height, args.width, args.num_groups)
            benchmark_fused_groupnorm_silu(args.batch_size, args.channels, args.height, args.width, args.num_groups, args.num_iter, args.num_warmup_iter)
            
            # Additional configurations
            print("\n\nAdditional configurations:")
            benchmark_fused_groupnorm_silu(4, 512, 32, 32, 32, args.num_iter, args.num_warmup_iter)
            benchmark_fused_groupnorm_silu(4, 128, 128, 128, 32, args.num_iter, args.num_warmup_iter)
    
    if args.kernel == "all" or args.kernel == "conv_groupnorm_silu":
        print("\n" + "="*80)
        print("FUSED CONV + GROUPNORM + SILU")
        print("="*80)
        
        if args.test_correctness:
            test_fused_conv_groupnorm_silu_correctness(
                args.batch_size, args.in_channels, args.out_channels,
                args.height, args.width, args.kernel_size, args.stride, args.padding, args.num_groups
            )
        else:
            test_fused_conv_groupnorm_silu_correctness(
                args.batch_size, args.in_channels, args.out_channels,
                args.height, args.width, args.kernel_size, args.stride, args.padding, args.num_groups
            )
            benchmark_fused_conv_groupnorm_silu(
                args.batch_size, args.in_channels, args.out_channels,
                args.height, args.width, args.kernel_size, args.stride, args.padding, args.num_groups,
                args.num_iter, args.num_warmup_iter
            )
            
            # Additional configurations
            print("\n\nAdditional configurations:")
            benchmark_fused_conv_groupnorm_silu(4, 128, 256, 64, 64, 3, 1, 1, 32, args.num_iter, args.num_warmup_iter)
            benchmark_fused_conv_groupnorm_silu(4, 256, 512, 32, 32, 3, 2, 1, 32, args.num_iter, args.num_warmup_iter)
