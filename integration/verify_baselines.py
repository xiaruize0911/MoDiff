
import torch
import time
import numpy as np
import pandas as pd
from modiff_triton.kernels.conv_w8a8_fused import conv2d_w8a8_3x3_standard
from modiff_triton.kernels.gemm_w4a4 import gemm_w4a4
import torch.nn.functional as F

def benchmark_conv(func, name, batch=32, c=128, h=32, w=32, out_c=128, num_runs=100):
    # Setup inputs
    x = torch.randn(batch, c, h, w).cuda().to(memory_format=torch.channels_last)
    weight = torch.randn(out_c, c, 3, 3).cuda().to(memory_format=torch.channels_last)
    bias = torch.randn(out_c).cuda()
    
    # Pre-quantize weights for integer kernels
    weight_int8 = (weight * 127).to(torch.int8)
    scale_w = torch.ones(out_c).cuda() * (1.0/127.0)
    
    # Warmup
    for _ in range(10):
        func(x, weight, bias)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        func(x, weight, bias)
    torch.cuda.synchronize()
    
    avg_ms = (time.time() - start) * 1000 / num_runs
    print(f"{name:20s}: {avg_ms:.4f} ms")
    return avg_ms

def run_fp32(x, w, b):
    return F.conv2d(x, w, b, padding=1)

def run_fp16(x, w, b):
    with torch.amp.autocast('cuda', dtype=torch.float16):
        return F.conv2d(x, w, b, padding=1)

def run_int8_triton(x, w, b):
    # Quantize once and cache (just for this benchmark script)
    if not hasattr(run_int8_triton, "_weight_int8"):
        run_int8_triton._weight_int8 = (w * 127).to(torch.int8)
        run_int8_triton._scale_w = torch.ones(w.shape[0]).cuda() * (1.0/127.0)
    
    return conv2d_w8a8_3x3_standard(x.to(memory_format=torch.channels_last), 
                                   run_int8_triton._weight_int8, 
                                   run_int8_triton._scale_w, 
                                   bias=b, static_scale=0.1)

print("Starting Baseline Verification (Batch=32, Res=32x32, Channels=128)")
print("-" * 60)

results = {}
results['FP32'] = benchmark_conv(run_fp32, "FP32 (Base)")
results['FP16'] = benchmark_conv(run_fp16, "FP16 (Autocast)")
results['INT8'] = benchmark_conv(run_int8_triton, "INT8 (Triton Std)")

# Calculate Speedups
print("-" * 60)
print(f"INT8 vs FP32 Speedup: {results['FP32']/results['INT8']:.2f}x")
print(f"INT8 vs FP16 Speedup: {results['FP16']/results['INT8']:.2f}x")

def verify_parity():
    print("\nVerifying Accuracy Parity (Simulation vs Real Triton)...")
    batch, c, h, w, out_c = 1, 128, 32, 32, 128
    x = torch.randn(batch, c, h, w).cuda()
    weight = torch.randn(out_c, c, 3, 3).cuda()
    
    # Simulation (Fake Quant)
    x_q_sim = torch.fake_quantize_per_tensor_affine(x, 0.1, 0, -128, 127)
    w_q_sim = torch.fake_quantize_per_channel_affine(weight, torch.ones(out_c).cuda()*0.01, torch.zeros(out_c).int().cuda(), 0, -128, 127)
    out_sim = F.conv2d(x_q_sim, w_q_sim, padding=1)
    
    # Real Triton Kernel
    weight_int8 = (weight / 0.01).round().clamp(-128, 127).to(torch.int8)
    scale_w = torch.ones(out_c).cuda() * 0.01
    x_nhwc = x.to(memory_format=torch.channels_last)
    out_triton = conv2d_w8a8_3x3_standard(x_nhwc, weight_int8, scale_w, static_scale=0.1)
    
    # Check MSE
    diff = (out_sim - out_triton).abs().mean()
    print(f"Mean Absolute Error: {diff.item():.2e}")
    if diff < 1e-4:
        print("✓ Parity Check PASSED")
    else:
        print("✗ Parity Check FAILED")

verify_parity()

