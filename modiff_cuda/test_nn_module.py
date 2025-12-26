import torch
import torch.nn as nn
import modiff_cuda
import time
try:
    import modiff_cuda_backend
except ImportError:
    # Try to find it in the installed package
    import sys
    import os
    # This is a hack, but might work if installed in site-packages
    pass

def test_layer(batch, in_c, out_c, h, w, kernel_size=3, stride=1, padding=1):
    print(f"Testing {kernel_size}x{kernel_size} Conv, Stride={stride}, Padding={padding}, Batch={batch}")
    
    input = torch.randn(batch, in_c, h, w).cuda()
    layer_fp32 = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False).cuda()
    
    # Quantize weights
    w_fp32 = layer_fp32.weight.data # [out_c, in_c, k, k]
    w_scales = w_fp32.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] / 127.0
    
    # Permute weight to [out_c, k, k, in_c] for the kernel
    w_permuted = w_fp32.permute(0, 2, 3, 1).contiguous()
    w_int8 = (w_permuted / w_scales).round().clamp(-128, 127).to(torch.int8)
    
    # Dequantized weights for reference (must use original layout for nn.Conv2d)
    w_deq = (w_int8.float() * w_scales).permute(0, 3, 1, 2).contiguous()
    layer_deq = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False).cuda()
    layer_deq.weight.data = w_deq
    
    # MoDiff Layer
    layer_int8 = modiff_cuda.W8A8MoDiffConv2dCUDA(in_c, out_c, kernel_size, stride=stride, padding=padding).cuda()
    # Flatten the permuted weight
    layer_int8.weight.data = w_int8.view(out_c, -1)
    layer_int8.weight_scales.data = w_scales.view(-1)
    
    # Warmup
    for _ in range(10):
        out_fp32 = layer_fp32(input)
        out_int8 = layer_int8(input)
        out_deq = layer_deq(input)
        
    # Measure full layer time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out_int8 = layer_int8(input)
    torch.cuda.synchronize()
    int8_time = (time.time() - start) * 1000 / 100
    
    # Measure Chained Performance (Simulate Layer N -> Layer N+1)
    # We run two layers. The first layer computes the scale for the second.
    # We measure the time of the second layer when it receives the scale.
    
    # Setup Layer 2 (same config)
    layer_int8_2 = modiff_cuda.W8A8MoDiffConv2dCUDA(out_c, out_c, kernel_size, stride=stride, padding=padding).cuda()
    # Just copy weights for simplicity
    layer_int8_2.weight.data = layer_int8.weight.data.clone()
    layer_int8_2.weight_scales.data = layer_int8.weight_scales.data.clone()
    
    # Run Layer 1 once to get output with scale
    out_1 = layer_int8(input)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        # This simulates the optimized path where out_1 has .next_scale
        out_2 = layer_int8_2(out_1)
    torch.cuda.synchronize()
    chained_time = (time.time() - start) * 1000 / 100
    
    # Measure with CUDA Graphs
    # Capture graph
    g = torch.cuda.CUDAGraph()
    # Run once to allocate
    out_2 = layer_int8_2(out_1)
    
    with torch.cuda.graph(g):
        out_2 = layer_int8_2(out_1)
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    graph_time = (time.time() - start) * 1000 / 100
    
    # Measure NHWC Flow Performance
    # Layer 1 outputs NHWC -> Layer 2 takes NHWC
    
    # Run Layer 1 with output_layout='NHWC'
    out_1_nhwc = layer_int8(input, output_layout='NHWC')
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        # Layer 2 should detect NHWC input and skip permutation
        out_2_nhwc = layer_int8_2(out_1_nhwc, output_layout='NHWC')
    torch.cuda.synchronize()
    nhwc_time = (time.time() - start) * 1000 / 100
    
    # Measure NHWC Flow + Graph
    g_nhwc = torch.cuda.CUDAGraph()
    # Run once to allocate
    out_2_nhwc = layer_int8_2(out_1_nhwc, output_layout='NHWC')
    
    with torch.cuda.graph(g_nhwc):
        out_2_nhwc = layer_int8_2(out_1_nhwc, output_layout='NHWC')
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        g_nhwc.replay()
    torch.cuda.synchronize()
    nhwc_graph_time = (time.time() - start) * 1000 / 100
    
    # Measure kernel only time (bypass quantization)
    # Pre-quantize input
    input_nhwc = input.permute(0, 2, 3, 1).contiguous()
    act_scale = input.abs().max() / 127.0
    input_int8_t = (input_nhwc / act_scale).round().clamp(-128, 127).to(torch.int8)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        modiff_cuda_backend.conv2d_fast_w8a8(
            input_int8_t,
            layer_int8.weight,
            act_scale,
            layer_int8.weight_scales,
            kernel_size,
            stride,
            padding
        )
    torch.cuda.synchronize()
    kernel_time = (time.time() - start) * 1000 / 100

    # Measure Fused Max Kernel Time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        modiff_cuda_backend.conv2d_fast_w8a8(
            input_int8_t,
            layer_int8.weight,
            act_scale,
            layer_int8.weight_scales,
            kernel_size,
            stride,
            padding,
            True # compute_max
        )
    torch.cuda.synchronize()
    fused_kernel_time = (time.time() - start) * 1000 / 100

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out_fp32 = layer_fp32(input)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) * 1000 / 100
    
    # Measure scale calculation time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        act_scale = input.abs().max() / 127.0
        act_scale_item = act_scale.item()
    torch.cuda.synchronize()
    scale_time = (time.time() - start) * 1000 / 100
    
    # Measure quantization kernel time
    dummy_scale = torch.tensor(0.5).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        modiff_cuda_backend.quantize_tensor(input_nhwc, dummy_scale)
    torch.cuda.synchronize()
    quant_time = (time.time() - start) * 1000 / 100

    print(f"INT8 Layer Time: {int8_time:.4f} ms")
    print(f"INT8 Chained Time (Fused Scale): {chained_time:.4f} ms")
    print(f"INT8 Graph Time: {graph_time:.4f} ms")
    print(f"INT8 NHWC Flow Time: {nhwc_time:.4f} ms")
    print(f"INT8 NHWC Graph Time: {nhwc_graph_time:.4f} ms")
    print(f"  - Scale Calc Time: {scale_time:.4f} ms")
    print(f"  - Quant Kernel Time: {quant_time:.4f} ms")
    print(f"INT8 Kernel Time: {kernel_time:.4f} ms")
    print(f"INT8 Fused Kernel Time: {fused_kernel_time:.4f} ms")
    print(f"FP32 Time: {fp32_time:.4f} ms")
    print(f"Layer Speedup: {fp32_time / int8_time:.2f}x")
    print(f"Kernel Speedup: {fp32_time / kernel_time:.2f}x")
    
    diff_deq = (out_int8.float() - out_deq).abs().max().item()
    diff_fp32 = (out_int8.float() - out_fp32).abs().max().item()
    print(f"Max diff (against deq ref): {diff_deq:.4f}")
    print(f"Max diff (against fp32 ref): {diff_fp32:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    # Test 1x1
    print("Testing 1x1 Conv, Stride=1, Padding=0, Batch=64")
    test_layer(64, 64, 64, 32, 32, kernel_size=1, stride=1, padding=0)
    # Test 3x3
    print("Testing 3x3 Conv, Stride=1, Padding=1, Batch=64")
    test_layer(64, 64, 64, 32, 32, kernel_size=3, stride=1, padding=1)
    # Test 3x3 Larger
    print("Testing 3x3 Conv, Stride=1, Padding=1, Batch=64, C=128")
    test_layer(64, 128, 128, 32, 32, kernel_size=3, stride=1, padding=1)
