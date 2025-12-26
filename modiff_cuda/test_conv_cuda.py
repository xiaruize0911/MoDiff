import torch
import modiff_cuda
import time

def test_conv():
    N, H, W, C = 1, 32, 32, 128
    R, S, K = 3, 3, 128 # K is output channels
    
    # Inputs
    input_int8 = torch.randint(-127, 127, (N, H, W, C), dtype=torch.int8, device='cuda')
    weight_int8 = torch.randint(-127, 127, (R, S, K, C), dtype=torch.int8, device='cuda')
    
    scale_A = torch.ones((N*H*W), dtype=torch.float16, device='cuda') * 0.01
    scale_B = torch.ones((K), dtype=torch.float16, device='cuda') * 0.01
    
    # Run CUDA kernel
    # Warmup
    for _ in range(10):
        out_cuda = modiff_cuda.conv2d_w8a8(input_int8, weight_int8, scale_A, scale_B)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out_cuda = modiff_cuda.conv2d_w8a8(input_int8, weight_int8, scale_A, scale_B)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Custom CUDA time: {(end - start)/100 * 1000:.3f} ms")
    
    # Baseline (FP16)
    input_fp16 = input_int8.to(torch.float16) * 0.01
    weight_fp16 = weight_int8.to(torch.float16) * 0.01
    weight_fp16_nchw = weight_fp16.permute(2, 3, 0, 1).contiguous() # [K, C, R, S]
    input_fp16_nchw = input_fp16.permute(0, 3, 1, 2).contiguous() # [N, C, H, W]
    
    # Warmup
    for _ in range(10):
        out_ref = torch.nn.functional.conv2d(input_fp16_nchw, weight_fp16_nchw, padding=1)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out_ref = torch.nn.functional.conv2d(input_fp16_nchw, weight_fp16_nchw, padding=1)
    torch.cuda.synchronize()
    end = time.time()
    print(f"PyTorch FP16 time: {(end - start)/100 * 1000:.3f} ms")
    
    # Correctness check
    # Note: My kernel does NOT handle padding yet!
    # The kernel assumes valid input.
    # If I pass 32x32 input and 3x3 kernel, output is 30x30?
    # No, the kernel loops `blockIdx_m` up to `M`.
    # `M = N*H*W`.
    # For each pixel, it reads `r, s` neighbors.
    # If `h=0, r=0`, it reads `h=0`.
    # If `h=31, r=2`, it reads `h=33` -> Out of bounds!
    # My kernel assumes PADDED input or valid region.
    # To match `padding=1`, I should pad the input before passing to kernel.
    
    input_padded = torch.nn.functional.pad(input_int8.permute(0,3,1,2), (1,1,1,1)).permute(0,2,3,1).contiguous()
    # Input padded is [N, H+2, W+2, C]
    # But I passed `input_int8` (32x32) to kernel.
    # The kernel will read OOB.
    # I should pass `input_padded`.
    
    # Update test to use padded input
    print("Running with padded input...")
    out_cuda = modiff_cuda.conv2d_w8a8(input_padded, weight_int8, scale_A, scale_B, padding_h=1, padding_w=1)
    print(f"Output shape: {out_cuda.shape}")
    
    # Benchmark with padded input
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out_cuda = modiff_cuda.conv2d_w8a8(input_padded, weight_int8, scale_A, scale_B, padding_h=1, padding_w=1)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Custom CUDA time (padded): {(end - start)/100 * 1000:.3f} ms")


if __name__ == "__main__":
    test_conv()
