import torch
import modiff_cuda
import time

def test_conv_simple():
    print("="*60)
    print("Testing Conv2d W8A8 CUDA Kernel (GEMM mode)")
    print("="*60)
    
    # Use smaller dimensions for initial testing
    N, H, W, C = 1, 32, 32, 128  # Input (unpadded)
    K = 128  # Output channels
    
    M = N * H * W  # 1024 output positions
    
    print(f"Input: {N}x{H}x{W}x{C} (flattened to {M}x{C})")
    print(f"Weight: {K}x{C} (treating as linear layer for now)")
    print(f"Expected output: {N}x{H}x{W}x{K}")
    
    # Create input and weight (flattened, no spatial structure for now)
    input_int8 = torch.randint(-127, 127, (M, C), dtype=torch.int8, device='cuda')
    weight_int8 = torch.randint(-127, 127, (1, 1, K, C), dtype=torch.int8, device='cuda')  # 1x1 kernel
    
    # Reshape input to NHWC for interface
    input_nhwc = input_int8.reshape(N, H, W, C)
    
    scale_A = torch.ones((M), dtype=torch.float16, device='cuda') * 0.01
    scale_B = torch.ones((K), dtype=torch.float16, device='cuda') * 0.01
    
    print("\nRunning CUDA kernel (linear mode, no convolution)...")
    try:
        out_cuda = modiff_cuda.conv2d_w8a8(input_nhwc, weight_int8, scale_A, scale_B, padding_h=0, padding_w=0)
        torch.cuda.synchronize()
        print(f"✓ Kernel executed successfully!")
        print(f"Output shape: {out_cuda.shape}")
        print(f"Output dtype: {out_cuda.dtype}")
        print(f"Output range: [{out_cuda.min().item():.3f}, {out_cuda.max().item():.3f}]")
        
        # Benchmark
        warmup = 10
        for _ in range(warmup):
            _ = modiff_cuda.conv2d_w8a8(input_nhwc, weight_int8, scale_A, scale_B, 0, 0)
        
        torch.cuda.synchronize()
        iters = 100
        start = time.time()
        for _ in range(iters):
            _ = modiff_cuda.conv2d_w8a8(input_nhwc, weight_int8, scale_A, scale_B, 0, 0)
        torch.cuda.synchronize()
        end = time.time()
        custom_time = (end - start) / iters * 1000
        print(f"\n✓ Custom CUDA time: {custom_time:.3f} ms")
        
    except RuntimeError as e:
        print(f"✗ Kernel failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conv_simple()
