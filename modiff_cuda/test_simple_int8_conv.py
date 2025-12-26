#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import time

# Simple Python-based im2col + GEMM for INT8 convolution
def conv2d_int8_simple(input, weight, stride=1, padding=0):
    """
    Simple INT8 convolution using PyTorch's unfold + mm
    Input: [N, H, W, C_in] INT8
    Weight: [C_out, C_in, K, K] INT8
    Output: [N, H_out, W_out, C_out] INT32
    """
    N, H, W, C_in = input.shape
    C_out, _, K, _ = weight.shape
    
    # Convert to NCHW format for unfold
    input_nchw = input.permute(0, 3, 1, 2).contiguous()  # [N, C_in, H, W]
    
    # Use unfold to perform im2col
    # unfold extracts sliding local blocks
    unfolded = F.unfold(input_nchw.float(), kernel_size=K, stride=stride, padding=padding)
    # unfolded shape: [N, C_in*K*K, H_out*W_out]
    
    # Convert back to INT8 (unfold only works with float)
    unfolded = unfolded.to(torch.int8)
    
    # Reshape weight to [C_out, C_in*K*K]
    weight_matrix = weight.view(C_out, -1)  # [C_out, C_in*K*K]
    
    # Perform batch matrix multiply
    # For each sample: [C_out, C_in*K*K] @ [C_in*K*K, H_out*W_out] = [C_out, H_out*W_out]
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    
    outputs = []
    for i in range(N):
        # PyTorch doesn't support INT8/INT32 mm on CUDA, use FP32 for now
        # In real implementation, this would call cuBLAS cublasGemmEx
        out = torch.mm(weight_matrix.float(), unfolded[i].float())  # [C_out, H_out*W_out]
        out = out.to(torch.int32)  # Convert result to INT32
        out = out.view(C_out, H_out, W_out).permute(1, 2, 0)  # [H_out, W_out, C_out]
        outputs.append(out)
    
    output = torch.stack(outputs, dim=0)  # [N, H_out, W_out, C_out]
    return output


if __name__ == "__main__":
    print("Testing simple INT8 convolution")
    print("=" * 60)
    
    # Test parameters
    N, C_in, H, W = 1, 128, 32, 32
    C_out = 128
    K = 3
    stride = 1
    padding = 1
    
    # Create test data
    input_data = torch.randint(-128, 127, (N, H, W, C_in), dtype=torch.int8, device='cuda')
    weight_data = torch.randint(-128, 127, (C_out, C_in, K, K), dtype=torch.int8, device='cuda')
    
    print(f"Input shape: {input_data.shape}")
    print(f"Weight shape: {weight_data.shape}")
    
    # Run convolution
    print("\nRunning INT8 convolution...")
    start = time.time()
    output = conv2d_int8_simple(input_data, weight_data, stride=stride, padding=padding)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"✓ Convolution completed successfully")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Time: {elapsed*1000:.2f} ms")
    
    # Verify against FP32 convolution
    print("\nVerifying against FP32...")
    input_fp32 = input_data.float().permute(0, 3, 1, 2)  # NCHW
    weight_fp32 = weight_data.float()  #.permute(0, 2, 3, 1)  # Keep as NCHW
    
    output_fp32 = F.conv2d(input_fp32, weight_fp32, stride=stride, padding=padding)
    output_fp32 = output_fp32.permute(0, 2, 3, 1).to(torch.int32)  # Convert to NHWC
    
    max_diff = torch.abs(output.cpu() - output_fp32.cpu()).max().item()
    print(f"Max difference vs FP32: {max_diff}")
    
    if max_diff < 10:  # Allow some rounding differences
        print("✓ Verification passed!")
    else:
        print(f"✗ Verification failed - difference too large: {max_diff}")
