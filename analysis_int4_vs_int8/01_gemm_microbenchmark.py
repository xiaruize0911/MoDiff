#!/usr/bin/env python3
"""
GEMM Microbenchmark: INT4 vs INT8 vs FP16 vs FP32
===================================================

Validates the raw compute throughput of different precision levels
on simple matrix multiplications (no neural network overhead).

According to NVIDIA's blog (https://developer.nvidia.com/blog/int4-for-ai-inference/),
INT4 should be ~50% faster than INT8 on tensor cores.

This script measures:
1. Pure GEMM throughput at various matrix sizes
2. CUTLASS INT4/INT8 convolution throughput (as used in the actual pipeline)
3. Quantization + packing overhead for INT4 vs INT8
4. End-to-end (quantize + compute) comparison

Results are saved to JSON and CSV for plotting.
"""

import os
import sys
import json
import time
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict

import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("WARNING: modiff_cutlass not available. CUTLASS benchmarks will be skipped.")


# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark_fn(fn, warmup=20, iters=100, sync=True):
    """Benchmark a function with CUDA event timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    if sync:
        torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    # Use median to avoid outliers
    median_ms = times_ms[len(times_ms) // 2]
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = times_ms[0]
    max_ms = times_ms[-1]
    
    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
    }


def pack_int4_tensor(tensor):
    """Pack int8 tensor (values in [-8,7]) to packed int4 (2 per byte), last dim halved."""
    shape = list(tensor.shape)
    last_dim = shape[-1]
    assert last_dim % 2 == 0
    new_shape = shape[:-1] + [last_dim // 2, 2]
    reshaped = tensor.view(new_shape)
    low = reshaped[..., 0] & 0x0F
    high = (reshaped[..., 1] & 0x0F) << 4
    packed = (low | high).to(torch.int8)
    return packed


# ============================================================================
# 1. Pure GEMM Benchmark (using torch matmul for FP32/FP16, Triton for INT8/4)
# ============================================================================

def benchmark_torch_matmul(M, N, K, dtype, device='cuda'):
    """Benchmark torch.matmul at given precision."""
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    
    def fn():
        return torch.matmul(A, B)
    
    result = benchmark_fn(fn)
    # Compute TFLOPS
    flops = 2 * M * N * K
    tflops = flops / (result["median_ms"] * 1e-3) / 1e12
    result["tflops"] = tflops
    result["M"] = M
    result["N"] = N
    result["K"] = K
    result["dtype"] = str(dtype)
    return result


# ============================================================================
# 2. CUTLASS Convolution Benchmark (INT8 vs INT4)
# ============================================================================

def benchmark_cutlass_conv(N, C, H, W, K, R, S, stride, padding, precision='int8', device='cuda'):
    """Benchmark CUTLASS convolution kernels directly."""
    if not HAS_CUTLASS:
        return None
    
    H_out = (H + 2 * padding - R) // stride + 1
    W_out = (W + 2 * padding - S) // stride + 1
    
    empty_bias = torch.empty(0, device=device)
    
    if precision == 'int8':
        # Create INT8 input in NCHW channels_last format
        inp = torch.randint(-127, 127, (N, C, H, W), dtype=torch.int8, device=device)
        inp = inp.contiguous(memory_format=torch.channels_last)
        # Weight: [K, R, S, C] for CUTLASS (NHWC-like)
        wt = torch.randint(-127, 127, (K, R, S, C), dtype=torch.int8, device=device).contiguous()
        scale = torch.tensor([1.0], device=device, dtype=torch.float32)
        
        def fn():
            return modiff_cutlass.conv2d_int8_fprop(
                inp, wt, scale, empty_bias,
                stride, stride, padding, padding, 1, 1
            )
    
    elif precision == 'int4':
        assert C % 2 == 0
        # Create packed INT4 input: (N, H, W, C//2)
        inp_int8 = torch.randint(-7, 7, (N, H, W, C), dtype=torch.int8, device=device)
        inp_packed = pack_int4_tensor(inp_int8).contiguous()
        # Weight: (K, R, S, C//2) packed
        wt_int8 = torch.randint(-7, 7, (K, R, S, C), dtype=torch.int8, device=device)
        wt_packed = pack_int4_tensor(wt_int8).contiguous()
        scale = torch.tensor([1.0], device=device, dtype=torch.float32)
        
        def fn():
            return modiff_cutlass.conv2d_int4_fprop(
                inp_packed, wt_packed, scale, empty_bias,
                stride, stride, padding, padding, 1, 1
            )
    else:
        raise ValueError(f"Unknown precision: {precision}")
    
    result = benchmark_fn(fn)
    
    # Compute effective TOPS
    flops = 2 * N * K * H_out * W_out * C * R * S
    tops = flops / (result["median_ms"] * 1e-3) / 1e12
    result["tops"] = tops
    result["precision"] = precision
    result["shape"] = f"N={N},C={C},H={H},W={W},K={K},R={R},S={S}"
    return result


# ============================================================================
# 3. Quantization Overhead Benchmark
# ============================================================================

def benchmark_quantization_overhead(N, C, H, W, device='cuda'):
    """Measure quantization + packing overhead for INT8 vs INT4."""
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    x_nhwc = x.contiguous(memory_format=torch.channels_last)
    scale = torch.tensor([0.5], device=device, dtype=torch.float32)
    
    results = {}
    
    if HAS_CUTLASS:
        # INT8 quantization
        def quant_int8():
            return modiff_cutlass.scale_quantize_int8(x_nhwc, scale)
        results['int8_quantize'] = benchmark_fn(quant_int8)
        
        # INT4 quantize + pack
        def quant_int4():
            return modiff_cutlass.scale_quantize_and_pack(x_nhwc, scale)
        results['int4_quantize_pack'] = benchmark_fn(quant_int4)
    
    # PyTorch native quantization for comparison
    def quant_pytorch_int8():
        return (x_nhwc * 0.5).round().clamp(-127, 127).to(torch.int8)
    results['pytorch_int8_quantize'] = benchmark_fn(quant_pytorch_int8)
    
    return results


# ============================================================================
# 4. End-to-End (Quant + Conv) Benchmark
# ============================================================================

def benchmark_end_to_end_conv(N, C, H, W, K, R, S, stride, padding, device='cuda'):
    """Measure full pipeline: quantize → conv → dequant for INT8 vs INT4."""
    if not HAS_CUTLASS:
        return None
    
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    x_nhwc = x.contiguous(memory_format=torch.channels_last)
    
    H_out = (H + 2 * padding - R) // stride + 1
    W_out = (W + 2 * padding - S) // stride + 1
    
    empty_bias = torch.empty(0, device=device)
    
    results = {}
    
    # --- FP32 baseline (PyTorch native) ---
    conv_fp32 = torch.nn.Conv2d(C, K, (R, S), stride=stride, padding=padding, bias=False).cuda()
    conv_fp32 = conv_fp32.to(memory_format=torch.channels_last)
    x_fp32 = x_nhwc.clone()
    
    def fp32_fn():
        return conv_fp32(x_fp32)
    results['fp32'] = benchmark_fn(fp32_fn)
    
    # --- FP16 ---
    conv_fp16 = conv_fp32.half()
    x_fp16 = x_nhwc.half()
    
    def fp16_fn():
        return conv_fp16(x_fp16)
    results['fp16'] = benchmark_fn(fp16_fn)
    
    # --- INT8: quantize + conv ---
    # Pre-quantize weights
    w_int8 = (conv_fp32.weight.data.permute(0, 2, 3, 1).contiguous() * 127).round().clamp(-127, 127).to(torch.int8)
    scale_int8 = torch.tensor([127.0 / max(x_nhwc.abs().max().item(), 1e-6)], device=device)
    inv_scale_int8 = torch.tensor([1.0 / scale_int8.item()], device=device)
    
    def int8_fn():
        q = modiff_cutlass.scale_quantize_int8(x_nhwc, scale_int8)
        out = modiff_cutlass.conv2d_int8_fprop(
            q, w_int8, inv_scale_int8, empty_bias,
            stride, stride, padding, padding, 1, 1
        )
        return out
    results['int8'] = benchmark_fn(int8_fn)
    
    # --- INT4: quantize + pack + conv ---
    assert C % 2 == 0
    w_int4 = (conv_fp32.weight.data.permute(0, 2, 3, 1).contiguous() * 7).round().clamp(-7, 7).to(torch.int8)
    w_int4_packed = pack_int4_tensor(w_int4).contiguous()
    scale_int4 = torch.tensor([7.0 / max(x_nhwc.abs().max().item(), 1e-6)], device=device)
    inv_scale_int4 = torch.tensor([1.0 / scale_int4.item()], device=device)
    
    def int4_fn():
        q = modiff_cutlass.scale_quantize_and_pack(x_nhwc, scale_int4)
        out = modiff_cutlass.conv2d_int4_fprop(
            q, w_int4_packed, inv_scale_int4, empty_bias,
            stride, stride, padding, padding, 1, 1
        )
        return out
    results['int4'] = benchmark_fn(int4_fn)
    
    # --- measure individual components ---
    # INT8 quantize only
    def int8_quant_only():
        return modiff_cutlass.scale_quantize_int8(x_nhwc, scale_int8)
    results['int8_quant_only'] = benchmark_fn(int8_quant_only)
    
    # INT4 quantize+pack only
    def int4_quant_only():
        return modiff_cutlass.scale_quantize_and_pack(x_nhwc, scale_int4)
    results['int4_quant_only'] = benchmark_fn(int4_quant_only)
    
    # INT8 conv only (pre-quantized input)
    q_int8 = modiff_cutlass.scale_quantize_int8(x_nhwc, scale_int8)
    def int8_conv_only():
        return modiff_cutlass.conv2d_int8_fprop(
            q_int8, w_int8, inv_scale_int8, empty_bias,
            stride, stride, padding, padding, 1, 1
        )
    results['int8_conv_only'] = benchmark_fn(int8_conv_only)
    
    # INT4 conv only (pre-quantized and packed input)
    q_int4 = modiff_cutlass.scale_quantize_and_pack(x_nhwc, scale_int4)
    def int4_conv_only():
        return modiff_cutlass.conv2d_int4_fprop(
            q_int4, w_int4_packed, inv_scale_int4, empty_bias,
            stride, stride, padding, padding, 1, 1
        )
    results['int4_conv_only'] = benchmark_fn(int4_conv_only)
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    sm_cap = torch.cuda.get_device_capability()
    
    print(f"GPU: {gpu_name} (SM {sm_cap[0]}.{sm_cap[1]})")
    print("=" * 80)
    
    all_results = {
        "metadata": {
            "gpu": gpu_name,
            "sm_capability": f"{sm_cap[0]}.{sm_cap[1]}",
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    }
    
    # ----------------------------------------------------------------
    # Test 1: Pure torch.matmul at different sizes and precisions
    # ----------------------------------------------------------------
    print("\n[Test 1] Pure torch.matmul (FP32 vs FP16)")
    print("-" * 60)
    
    matmul_sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # Sizes typical for diffusion model layers
        (4096, 512, 512),   # Large batch, moderate hidden
        (16384, 256, 256),  # Very large spatial dim, small channels
        (1024, 320, 320),   # LDM-like first block
        (1024, 640, 640),   # LDM-like mid block
    ]
    
    matmul_results = []
    for M, N, K in matmul_sizes:
        for dtype in [torch.float32, torch.float16]:
            r = benchmark_torch_matmul(M, N, K, dtype, device)
            matmul_results.append(r)
            dtype_str = "FP32" if dtype == torch.float32 else "FP16"
            print(f"  {dtype_str} [{M}x{K}] @ [{K}x{N}]: {r['median_ms']:.3f} ms  ({r['tflops']:.2f} TFLOPS)")
    
    all_results["matmul"] = matmul_results
    
    # ----------------------------------------------------------------
    # Test 2: CUTLASS Conv2d INT8 vs INT4
    # ----------------------------------------------------------------
    if HAS_CUTLASS:
        print("\n[Test 2] CUTLASS Conv2d: INT8 vs INT4 (pure kernel)")
        print("-" * 60)
        
        # Typical shapes from LDM (LSUN Churches 256x256)
        conv_shapes = [
            # (N, C, H, W, K, R, S, stride, padding) — representative of LDM
            (32, 128, 64, 64, 128, 3, 3, 1, 1),   # Early blocks
            (32, 256, 32, 32, 256, 3, 3, 1, 1),   # Mid blocks  
            (32, 512, 16, 16, 512, 3, 3, 1, 1),   # Deep blocks
            (32, 512, 8, 8, 512, 3, 3, 1, 1),     # Deepest
            # Larger test
            (8, 128, 128, 128, 128, 3, 3, 1, 1),   # Large spatial
            (8, 256, 64, 64, 256, 3, 3, 1, 1),    # Large spatial mid
            # Downsampling conv
            (32, 128, 64, 64, 256, 3, 3, 2, 1),   # Stride-2 downsample
            (32, 256, 32, 32, 512, 3, 3, 2, 1),   # Stride-2 downsample
        ]
        
        conv_results = []
        for shape in conv_shapes:
            N, C, H, W, K, R, S, stride, padding = shape
            for prec in ['int8', 'int4']:
                r = benchmark_cutlass_conv(N, C, H, W, K, R, S, stride, padding, prec, device)
                if r:
                    conv_results.append(r)
                    print(f"  {prec.upper():5s} N={N} C={C} H={H} W={W} K={K}: "
                          f"{r['median_ms']:.3f} ms  ({r['tops']:.2f} TOPS)")
            print()
        
        all_results["cutlass_conv"] = conv_results
    
    # ----------------------------------------------------------------
    # Test 3: Quantization overhead
    # ----------------------------------------------------------------
    if HAS_CUTLASS:
        print("\n[Test 3] Quantization Overhead: INT8 vs INT4")
        print("-" * 60)
        
        quant_shapes = [
            (32, 128, 64, 64),
            (32, 256, 32, 32),
            (32, 512, 16, 16),
            (32, 512, 8, 8),
        ]
        
        quant_results = []
        for N, C, H, W in quant_shapes:
            r = benchmark_quantization_overhead(N, C, H, W, device)
            quant_results.append({"shape": f"N={N},C={C},H={H},W={W}", **{k: v for k, v in r.items()}})
            print(f"  Shape N={N} C={C} H={H} W={W}:")
            for name, timing in r.items():
                print(f"    {name:30s}: {timing['median_ms']:.4f} ms")
        
        all_results["quantization_overhead"] = quant_results
    
    # ----------------------------------------------------------------
    # Test 4: End-to-End Conv (quant + compute)
    # ----------------------------------------------------------------
    if HAS_CUTLASS:
        print("\n[Test 4] End-to-End Conv: FP32 vs FP16 vs INT8 vs INT4")
        print("-" * 60)
        
        e2e_shapes = [
            (32, 128, 64, 64, 128, 3, 3, 1, 1),
            (32, 256, 32, 32, 256, 3, 3, 1, 1),
            (32, 512, 16, 16, 512, 3, 3, 1, 1),
            (32, 512, 8, 8, 512, 3, 3, 1, 1),
        ]
        
        e2e_results = []
        for shape in e2e_shapes:
            N, C, H, W, K, R, S, stride, padding = shape
            r = benchmark_end_to_end_conv(N, C, H, W, K, R, S, stride, padding, device)
            if r:
                r_entry = {"shape": f"N={N},C={C},H={H},W={W},K={K},R={R},S={S}"}
                for name, timing in r.items():
                    r_entry[name] = timing
                e2e_results.append(r_entry)
                
                print(f"\n  Shape: N={N} C={C} H={H} W={W} K={K} {R}x{S}:")
                fp32_ms = r['fp32']['median_ms']
                for name in ['fp32', 'fp16', 'int8', 'int4',
                             'int8_quant_only', 'int4_quant_only',
                             'int8_conv_only', 'int4_conv_only']:
                    if name in r:
                        ms = r[name]['median_ms']
                        speedup = fp32_ms / ms if ms > 0 else 0
                        print(f"    {name:20s}: {ms:.4f} ms  (speedup: {speedup:.2f}x vs FP32)")
                
                # Breakdown analysis
                print(f"\n    --- TIME BREAKDOWN ---")
                int8_total = r['int8']['median_ms']
                int4_total = r['int4']['median_ms']
                int8_quant = r['int8_quant_only']['median_ms']
                int4_quant = r['int4_quant_only']['median_ms']
                int8_conv = r['int8_conv_only']['median_ms']
                int4_conv = r['int4_conv_only']['median_ms']
                
                print(f"    INT8 total = {int8_total:.4f} ms (quant: {int8_quant:.4f} ms [{100*int8_quant/int8_total:.1f}%] + conv: {int8_conv:.4f} ms [{100*int8_conv/int8_total:.1f}%])")
                print(f"    INT4 total = {int4_total:.4f} ms (quant: {int4_quant:.4f} ms [{100*int4_quant/int4_total:.1f}%] + conv: {int4_conv:.4f} ms [{100*int4_conv/int4_total:.1f}%])")
                
                if int8_conv > 0:
                    conv_speedup = int8_conv / int4_conv
                    print(f"    INT4 conv speedup over INT8 conv: {conv_speedup:.2f}x")
                if int8_quant > 0:
                    quant_ratio = int4_quant / int8_quant
                    print(f"    INT4 quant+pack cost / INT8 quant cost: {quant_ratio:.2f}x")
        
        all_results["end_to_end_conv"] = e2e_results
    
    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    results_path = os.path.join(output_dir, "gemm_benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    # Save CSV summary for easy plotting
    csv_path = os.path.join(output_dir, "gemm_benchmark_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test', 'shape', 'precision', 'median_ms', 'speedup_vs_fp32', 'notes'])
        
        # E2E results
        if 'end_to_end_conv' in all_results:
            for entry in all_results['end_to_end_conv']:
                shape = entry['shape']
                fp32_ms = entry.get('fp32', {}).get('median_ms', 0)
                for prec in ['fp32', 'fp16', 'int8', 'int4',
                             'int8_quant_only', 'int4_quant_only',
                             'int8_conv_only', 'int4_conv_only']:
                    if prec in entry:
                        ms = entry[prec]['median_ms']
                        spd = fp32_ms / ms if ms > 0 else 0
                        writer.writerow(['e2e_conv', shape, prec, f"{ms:.4f}", f"{spd:.2f}", ''])
    
    print(f"CSV summary saved to {csv_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Key findings about INT4 vs INT8 speedup")
    print("=" * 80)
    
    if 'end_to_end_conv' in all_results:
        print("\nPer-shape analysis:")
        for entry in all_results['end_to_end_conv']:
            shape = entry['shape']
            int8_conv = entry.get('int8_conv_only', {}).get('median_ms', 0)
            int4_conv = entry.get('int4_conv_only', {}).get('median_ms', 0)
            int8_quant = entry.get('int8_quant_only', {}).get('median_ms', 0)
            int4_quant = entry.get('int4_quant_only', {}).get('median_ms', 0)
            int8_total = entry.get('int8', {}).get('median_ms', 0)
            int4_total = entry.get('int4', {}).get('median_ms', 0)
            
            conv_ratio = int8_conv / int4_conv if int4_conv > 0 else 0
            total_ratio = int8_total / int4_total if int4_total > 0 else 0
            quant_overhead_int4 = 100 * int4_quant / int4_total if int4_total > 0 else 0
            quant_overhead_int8 = 100 * int8_quant / int8_total if int8_total > 0 else 0
            
            print(f"\n  {shape}:")
            print(f"    Pure conv: INT4 is {conv_ratio:.2f}x of INT8")
            print(f"    E2E:       INT4 is {total_ratio:.2f}x of INT8")
            print(f"    Quant overhead: INT8={quant_overhead_int8:.1f}%, INT4={quant_overhead_int4:.1f}%")


if __name__ == "__main__":
    main()
