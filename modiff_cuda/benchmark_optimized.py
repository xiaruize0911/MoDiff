import torch
import modiff_cuda
import time

def benchmark_conv(name, func, input, weight, stride=1, padding=1):
    # Warmup
    for _ in range(10):
        _ = func(input, weight, stride, padding)
    
    torch.cuda.synchronize()
    start = time.time()
    iters = 100
    for _ in range(iters):
        _ = func(input, weight, stride, padding)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters * 1000
    print(f"{name:20}: {avg_time:.4f} ms")
    return avg_time

def test_correctness(name, func, input, weight, ref_output, stride=1, padding=1):
    out = func(input, weight, stride, padding)
    max_diff = (out - ref_output).abs().max().item()
    if max_diff == 0:
        print(f"{name:20}: ✓ Correct")
    else:
        print(f"{name:20}: ✗ Incorrect (max_diff={max_diff})")

def run_benchmarks():
    N, H, W, C_in = 1, 64, 64, 128
    C_out, K = 128, 3
    stride, padding = 1, 1
    
    print(f"Config: N={N}, H={H}, W={W}, C_in={C_in}, C_out={C_out}, K={K}")
    print("-" * 40)
    
    input = torch.randint(-127, 127, (N, H, W, C_in), dtype=torch.int8, device='cuda')
    weight = torch.randint(-127, 127, (C_out, K, K, C_in), dtype=torch.int8, device='cuda')
    
    # Reference (Simple)
    ref_output = modiff_cuda.conv2d_simple(input, weight, stride, padding)
    
    # Correctness
    test_correctness("Simple", modiff_cuda.conv2d_simple, input, weight, ref_output, stride, padding)
    test_correctness("Tiled", modiff_cuda.conv2d_tiled, input, weight, ref_output, stride, padding)
    test_correctness("Fast", modiff_cuda.conv2d_fast, input, weight, ref_output, stride, padding)
    test_correctness("Opt V1", modiff_cuda.conv2d_opt_v1, input, weight, ref_output, stride, padding)
    
    print("-" * 40)
    
    # Performance
    benchmark_conv("Simple", modiff_cuda.conv2d_simple, input, weight, stride, padding)
    benchmark_conv("Tiled", modiff_cuda.conv2d_tiled, input, weight, stride, padding)
    benchmark_conv("Fast", modiff_cuda.conv2d_fast, input, weight, stride, padding)
    benchmark_conv("Opt V1", modiff_cuda.conv2d_opt_v1, input, weight, stride, padding)
    
    # PyTorch FP16 for comparison
    input_h = input.half()
    weight_h = weight.half().permute(0, 3, 1, 2).contiguous() # [C_out, C_in, K, K]
    input_h_nchw = input_h.permute(0, 3, 1, 2).contiguous()
    
    # Warmup
    for _ in range(10):
        _ = torch.nn.functional.conv2d(input_h_nchw, weight_h, stride=stride, padding=padding)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = torch.nn.functional.conv2d(input_h_nchw, weight_h, stride=stride, padding=padding)
    torch.cuda.synchronize()
    end = time.time()
    print(f"{'PyTorch FP16':20}: {(end - start) / 100 * 1000:.4f} ms")

if __name__ == "__main__":
    run_benchmarks()
