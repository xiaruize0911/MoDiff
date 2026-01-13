"""
Benchmark script for MoDiff quantization kernels.
Tests quantization/dequantization operations and fused residual operations.
"""

import torch
import argparse
from icecream import ic

try:
    import modiff_fused_ops
    import modiff_int8
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import modiff_fused_ops
    import modiff_int8


def test_quantize_tensor_correctness(batch_size, channels, height, width):
    """Test correctness of tensor quantization."""
    print(f"\nTesting quantize_tensor correctness: B={batch_size}, C={channels}, H={height}, W={width}")
    
    # Create random FP32 input
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device="cuda")
    
    # Compute scale
    scale = x.abs().max() / 127.0
    
    # Ground truth quantization
    x_int8_gt = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    
    # Custom quantization kernel
    x_int8 = modiff_int8.quantize_tensor(x, scale)
    
    # Compute error
    max_error = torch.max(torch.abs(x_int8.float() - x_int8_gt.float()))
    
    ic(max_error)
    print(f"Max error: {max_error:.6f}")
    
    assert max_error < 1.0, "Quantization error too large!"


def test_fused_residual_quantize_correctness(batch_size, channels, height, width):
    """Test correctness of fused residual + quantization."""
    print(f"\nTesting fused_residual_quantize correctness: B={batch_size}, C={channels}, H={height}, W={width}")
    
    # Create random inputs
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float16, device="cuda")
    residual = torch.randn(batch_size, channels, height, width, dtype=torch.float16, device="cuda")
    
    # Ground truth: add residual then quantize
    sum_gt = (x + residual).float()
    scale_gt = sum_gt.abs().max() / 127.0
    quantized_gt = torch.round(sum_gt / scale_gt).clamp(-128, 127).to(torch.int8)
    
    # Fused kernel
    quantized_fused, scale_fused = modiff_fused_ops.fused_residual_quantize(x, residual)
    
    # Compute error
    max_error_quant = torch.max(torch.abs(quantized_fused.float() - quantized_gt.float()))
    error_scale = torch.abs(scale_fused - scale_gt)
    
    ic(max_error_quant)
    ic(error_scale)
    
    print(f"Max quantization error: {max_error_quant:.6f}, Scale error: {error_scale:.6f}")


def test_fused_dequantize_accumulate_correctness(batch_size, channels, height, width):
    """Test correctness of fused dequantization + accumulation."""
    print(f"\nTesting fused_dequantize_accumulate correctness: B={batch_size}, C={channels}, H={height}, W={width}")
    
    # Create random inputs
    x_int8 = torch.randint(-128, 127, (batch_size, channels, height, width), dtype=torch.int8, device="cuda")
    scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
    accumulator = torch.randn(batch_size, channels, height, width, dtype=torch.float16, device="cuda")
    
    # Ground truth: dequantize then accumulate
    x_dequant = x_int8.float() * scale
    result_gt = (x_dequant + accumulator.float()).half()
    
    # Fused kernel
    result_fused = modiff_fused_ops.fused_dequantize_accumulate(x_int8, scale, accumulator)
    
    # Compute error
    max_error = torch.max(torch.abs(result_fused - result_gt))
    mean_error = torch.mean(torch.abs(result_fused - result_gt))
    
    ic(max_error)
    ic(mean_error)
    
    print(f"Max error: {max_error:.6f}, Mean error: {mean_error:.6f}")


def benchmark_quantization(batch_size, channels, height, width, num_iter=100, num_warmup_iter=20):
    """Benchmark quantization operations."""
    
    print(f"\n{'='*80}")
    print(f"Benchmarking quantization: B={batch_size}, C={channels}, H={height}, W={width}")
    print(f"{'='*80}")
    
    # Prepare test data
    input_fp32_list = []
    input_fp16_list = []
    scale_list = []
    
    for _ in range(num_iter + 1):
        x_fp32 = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device="cuda")
        x_fp16 = x_fp32.to(torch.float16)
        scale = x_fp32.abs().max() / 127.0
        
        input_fp32_list.append(x_fp32)
        input_fp16_list.append(x_fp16)
        scale_list.append(scale)
    
    # Warmup
    for _ in range(num_warmup_iter):
        # PyTorch baseline
        torch.round(input_fp32_list[-1] / scale_list[-1]).clamp(-128, 127).to(torch.int8)
        
        # Custom kernels
        modiff_int8.quantize_tensor(input_fp32_list[-1], scale_list[-1])
        modiff_int8.quantize_tensor_fast(input_fp32_list[-1], scale_list[-1])
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Benchmark PyTorch baseline
    start.record()
    for i in range(num_iter):
        torch.round(input_fp32_list[i] / scale_list[i]).clamp(-128, 127).to(torch.int8)
    end.record()
    torch.cuda.synchronize()
    avg_time_pytorch = start.elapsed_time(end) / num_iter
    
    # Benchmark quantize_tensor
    start.record()
    for i in range(num_iter):
        modiff_int8.quantize_tensor(input_fp32_list[i], scale_list[i])
    end.record()
    torch.cuda.synchronize()
    avg_time_custom = start.elapsed_time(end) / num_iter
    
    # Benchmark quantize_tensor_fast
    start.record()
    for i in range(num_iter):
        modiff_int8.quantize_tensor_fast(input_fp32_list[i], scale_list[i])
    end.record()
    torch.cuda.synchronize()
    avg_time_fast = start.elapsed_time(end) / num_iter
    
    # Calculate speedup
    speedup_custom = avg_time_pytorch / avg_time_custom
    speedup_fast = avg_time_pytorch / avg_time_fast
    
    print(f"\nQuantization Results:")
    print(f"  PyTorch baseline:         {avg_time_pytorch:.4f} ms")
    print(f"  quantize_tensor:          {avg_time_custom:.4f} ms (speedup: {speedup_custom:.2f}x)")
    print(f"  quantize_tensor_fast:     {avg_time_fast:.4f} ms (speedup: {speedup_fast:.2f}x)")


def benchmark_fused_residual_quantize(batch_size, channels, height, width, num_iter=100, num_warmup_iter=20):
    """Benchmark fused residual + quantization."""
    
    print(f"\n{'='*80}")
    print(f"Benchmarking fused_residual_quantize: B={batch_size}, C={channels}, H={height}, W={width}")
    print(f"{'='*80}")
    
    # Prepare test data
    input_list = []
    residual_list = []
    
    for _ in range(num_iter + 1):
        x = torch.randn(batch_size, channels, height, width, dtype=torch.float16, device="cuda")
        residual = torch.randn(batch_size, channels, height, width, dtype=torch.float16, device="cuda")
        
        input_list.append(x)
        residual_list.append(residual)
    
    # Warmup
    for _ in range(num_warmup_iter):
        # PyTorch baseline
        sum_result = (input_list[-1] + residual_list[-1]).float()
        scale = sum_result.abs().max() / 127.0
        torch.round(sum_result / scale).clamp(-128, 127).to(torch.int8)
        
        # Fused kernel
        modiff_fused_ops.fused_residual_quantize(input_list[-1], residual_list[-1])
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Benchmark PyTorch baseline
    start.record()
    for i in range(num_iter):
        sum_result = (input_list[i] + residual_list[i]).float()
        scale = sum_result.abs().max() / 127.0
        torch.round(sum_result / scale).clamp(-128, 127).to(torch.int8)
    end.record()
    torch.cuda.synchronize()
    avg_time_pytorch = start.elapsed_time(end) / num_iter
    
    # Benchmark fused kernel
    start.record()
    for i in range(num_iter):
        modiff_fused_ops.fused_residual_quantize(input_list[i], residual_list[i])
    end.record()
    torch.cuda.synchronize()
    avg_time_fused = start.elapsed_time(end) / num_iter
    
    # Calculate speedup
    speedup = avg_time_pytorch / avg_time_fused
    
    print(f"\nFused Residual+Quantize Results:")
    print(f"  PyTorch (add + quantize):     {avg_time_pytorch:.4f} ms")
    print(f"  Fused residual+quantize:      {avg_time_fused:.4f} ms (speedup: {speedup:.2f}x)")


def benchmark_fused_dequantize_accumulate(batch_size, channels, height, width, num_iter=100, num_warmup_iter=20):
    """Benchmark fused dequantization + accumulation."""
    
    print(f"\n{'='*80}")
    print(f"Benchmarking fused_dequantize_accumulate: B={batch_size}, C={channels}, H={height}, W={width}")
    print(f"{'='*80}")
    
    # Prepare test data
    input_int8_list = []
    scale_list = []
    accumulator_list = []
    
    for _ in range(num_iter + 1):
        x_int8 = torch.randint(-128, 127, (batch_size, channels, height, width), dtype=torch.int8, device="cuda")
        scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        accumulator = torch.randn(batch_size, channels, height, width, dtype=torch.float16, device="cuda")
        
        input_int8_list.append(x_int8)
        scale_list.append(scale)
        accumulator_list.append(accumulator)
    
    # Warmup
    for _ in range(num_warmup_iter):
        # PyTorch baseline
        x_dequant = input_int8_list[-1].float() * scale_list[-1]
        (x_dequant + accumulator_list[-1].float()).half()
        
        # Fused kernel
        modiff_fused_ops.fused_dequantize_accumulate(input_int8_list[-1], scale_list[-1], accumulator_list[-1])
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Benchmark PyTorch baseline
    start.record()
    for i in range(num_iter):
        x_dequant = input_int8_list[i].float() * scale_list[i]
        (x_dequant + accumulator_list[i].float()).half()
    end.record()
    torch.cuda.synchronize()
    avg_time_pytorch = start.elapsed_time(end) / num_iter
    
    # Benchmark fused kernel
    start.record()
    for i in range(num_iter):
        modiff_fused_ops.fused_dequantize_accumulate(input_int8_list[i], scale_list[i], accumulator_list[i])
    end.record()
    torch.cuda.synchronize()
    avg_time_fused = start.elapsed_time(end) / num_iter
    
    # Calculate speedup
    speedup = avg_time_pytorch / avg_time_fused
    
    print(f"\nFused Dequantize+Accumulate Results:")
    print(f"  PyTorch (dequant + add):      {avg_time_pytorch:.4f} ms")
    print(f"  Fused dequantize+accumulate:  {avg_time_fused:.4f} ms (speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MoDiff quantization kernels")
    parser.add_argument("--operation", type=str, default="all", 
                        choices=["all", "quantize", "fused_residual_quantize", "fused_dequantize_accumulate"],
                        help="Which operation to benchmark")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--channels", type=int, default=256, help="Number of channels")
    parser.add_argument("--height", type=int, default=64, help="Input height")
    parser.add_argument("--width", type=int, default=64, help="Input width")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--num_warmup_iter", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--test_correctness", action="store_true", help="Test correctness only")
    
    args = parser.parse_args()
    
    if args.operation == "all" or args.operation == "quantize":
        print("\n" + "="*80)
        print("QUANTIZATION OPERATIONS")
        print("="*80)
        
        if args.test_correctness:
            test_quantize_tensor_correctness(args.batch_size, args.channels, args.height, args.width)
        else:
            test_quantize_tensor_correctness(args.batch_size, args.channels, args.height, args.width)
            benchmark_quantization(args.batch_size, args.channels, args.height, args.width, args.num_iter, args.num_warmup_iter)
            
            # Additional configurations
            print("\n\nAdditional configurations:")
            benchmark_quantization(4, 512, 32, 32, args.num_iter, args.num_warmup_iter)
            benchmark_quantization(4, 128, 128, 128, args.num_iter, args.num_warmup_iter)
    
    if args.operation == "all" or args.operation == "fused_residual_quantize":
        print("\n" + "="*80)
        print("FUSED RESIDUAL + QUANTIZE")
        print("="*80)
        
        if args.test_correctness:
            test_fused_residual_quantize_correctness(args.batch_size, args.channels, args.height, args.width)
        else:
            test_fused_residual_quantize_correctness(args.batch_size, args.channels, args.height, args.width)
            benchmark_fused_residual_quantize(args.batch_size, args.channels, args.height, args.width, args.num_iter, args.num_warmup_iter)
            
            # Additional configurations
            print("\n\nAdditional configurations:")
            benchmark_fused_residual_quantize(4, 512, 32, 32, args.num_iter, args.num_warmup_iter)
            benchmark_fused_residual_quantize(4, 128, 128, 128, args.num_iter, args.num_warmup_iter)
    
    if args.operation == "all" or args.operation == "fused_dequantize_accumulate":
        print("\n" + "="*80)
        print("FUSED DEQUANTIZE + ACCUMULATE")
        print("="*80)
        
        if args.test_correctness:
            test_fused_dequantize_accumulate_correctness(args.batch_size, args.channels, args.height, args.width)
        else:
            test_fused_dequantize_accumulate_correctness(args.batch_size, args.channels, args.height, args.width)
            benchmark_fused_dequantize_accumulate(args.batch_size, args.channels, args.height, args.width, args.num_iter, args.num_warmup_iter)
            
            # Additional configurations
            print("\n\nAdditional configurations:")
            benchmark_fused_dequantize_accumulate(4, 512, 32, 32, args.num_iter, args.num_warmup_iter)
            benchmark_fused_dequantize_accumulate(4, 128, 128, 128, args.num_iter, args.num_warmup_iter)
