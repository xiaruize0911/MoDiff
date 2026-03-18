"""
Triton INT8 Conv2d kernel (3x3, stride=1, padding=1, dilation=1, groups=1).

This implements a direct convolution without unfold/im2col to reduce overhead.
Assumptions:
  - Input: NCHW int8
  - Weight: OIHW int8 with kH=kW=3
  - Groups=1
  - Stride=1, Padding=1, Dilation=1
Output is FP32.
"""

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        # INT8 conv optimized - larger BLOCK_K for better tensor core usage
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=6, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
    ],
    key=['C', 'O'],
)
@triton.jit
def _conv3x3_w8a8_implicit_gemm_kernel(
    x_ptr,          # int8 [N, C, H, W]
    w_ptr,          # int8 [O, C, 3, 3]
    b_ptr,          # fp32 [O] or nullptr
    y_ptr,          # fp32 [N, O, H, W]
    prev_y_ptr,     # fp32 [N, O, H, W] or nullptr
    act_scale_ptr,  # fp32 scalar
    weight_scale_ptr,  # fp32 scalar or vector
    N, C, H, W, O,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wk1, stride_wk2,
    stride_yn, stride_yo, stride_yh, stride_yw,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_W_IS_VECTOR: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """
    Implicit GEMM Convolution Kernel.
    M = N*H*W (Output pixels)
    N = O (Output channels)
    K = C (Input channels) - we iterate over 3x3 spatial locations in outer loop
    """
    # -----------------------------------------------------------
    # 1. Swizzling for better L2 Cache Locality
    # -----------------------------------------------------------
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N * H * W, BLOCK_M)
    num_pid_n = tl.cdiv(O, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # 2. Pre-compute Coordinates (Hoisted out of loop)
    # -----------------------------------------------------------
    # Output pixel indices (M dimension)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (N * H * W)
    
    # Output channel indices (N dimension)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < O

    # Vectorized coordinate computation (Done ONCE)
    # Note: Integer division is expensive, doing it once is key.
    batch_idx = offs_m // (H * W)
    hw_idx = offs_m % (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    # Pre-calculate base pointers
    # X base: batch offset
    x_base_ptr = x_ptr + batch_idx[:, None] * stride_xn
    
    # W base: output channel offset
    w_base_ptr = w_ptr + offs_n[None, :] * stride_wo

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # -----------------------------------------------------------
    # 3. Main Loop
    # -----------------------------------------------------------
    # Iterate over 3x3 kernel spatial locations
    # We keep r, s outer to maximize spatial locality of A loads
    for r in range(3):
        for s in range(3):
            # Pre-compute spatial offsets for this kernel position
            h_in = h_idx + r - 1
            w_in = w_idx + s - 1
            
            # Check bounds (vectorized)
            in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            
            # Base pointer for A at this (r, s)
            # Note: We add h_in, w_in offsets. 
            # If out of bounds, we mask later, but pointer arithmetic is safe (just garbage address)
            # To be safe, we can clamp or just rely on mask.
            a_ptr_base = x_base_ptr + h_in[:, None] * stride_xh + w_in[:, None] * stride_xw
            
            # Base pointer for B at this (r, s)
            b_ptr_base = w_base_ptr + (r * stride_wk1 + s * stride_wk2)

            # Iterate over C (K dimension)
            for k in range(0, C, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < C
                
                # --- Load Weights (B matrix) ---
                # Shape: [BLOCK_K, BLOCK_N]
                b_ptrs = b_ptr_base + offs_k[:, None] * stride_wc
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
                
                # --- Load Input (A matrix) ---
                # Shape: [BLOCK_M, BLOCK_K]
                a_ptrs = a_ptr_base + offs_k[None, :] * stride_xc
                
                # Mask: valid M, valid K, and spatial bounds
                # Note: in_bounds is [BLOCK_M, 1]
                mask_a = mask_m[:, None] & mask_k[None, :] & in_bounds[:, None]
                
                a = tl.load(a_ptrs, mask=mask_a, other=0)
                
                # --- Matrix Multiply ---
                acc = tl.dot(a, b, acc)

    # -----------------------------------------------------------
    # 4. Epilogue
    # -----------------------------------------------------------
    scale_a = tl.load(act_scale_ptr)
    if SCALE_W_IS_VECTOR:
        scale_w = tl.load(weight_scale_ptr + offs_n, mask=mask_n, other=0.0)
    else:
        scale_w = tl.load(weight_scale_ptr)

    # Convert to float and scale
    out = acc.to(tl.float32) * scale_a * scale_w
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        out = out + bias[None, :]
        
    # Add previous output if provided
    # Re-use pre-computed coordinates
    y_base_ptrs = y_ptr + \
                  batch_idx[:, None] * stride_yn + \
                  offs_n[None, :] * stride_yo + \
                  h_idx[:, None] * stride_yh + \
                  w_idx[:, None] * stride_yw
                  
    if prev_y_ptr is not None:
        prev_y_ptrs = prev_y_ptr + \
                      batch_idx[:, None] * stride_yn + \
                      offs_n[None, :] * stride_yo + \
                      h_idx[:, None] * stride_yh + \
                      w_idx[:, None] * stride_yw
        prev_y = tl.load(prev_y_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        out = out + prev_y

    tl.store(y_base_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def conv2d_int8_triton_accumulate(
    x_int8: torch.Tensor, 
    weight_int8: torch.Tensor, 
    prev_output: torch.Tensor,
    act_scale: torch.Tensor, 
    weight_scale: torch.Tensor, 
    bias: torch.Tensor = None
):
    """Direct INT8 conv (3x3 s1 p1) with accumulation using Implicit GEMM."""
    assert x_int8.ndim == 4 and weight_int8.ndim == 4
    N, C, H, W = x_int8.shape
    O = weight_int8.shape[0]
    assert weight_int8.shape[2:] == (3, 3)
    
    if prev_output is not None:
        assert prev_output.shape == (N, O, H, W)

    y = torch.empty((N, O, H, W), device=x_int8.device, dtype=torch.float32)
    
    is_vector_scale_w = (weight_scale.numel() > 1)

    # Grid dimensions
    # M = N*H*W
    # N = O
    # We launch a 1D grid because we handle swizzling manually inside the kernel
    grid = lambda META: (
        triton.cdiv(N * H * W, META['BLOCK_M']) * triton.cdiv(O, META['BLOCK_N']),
    )
    
    _conv3x3_w8a8_implicit_gemm_kernel[grid](
        x_int8, weight_int8, bias, y, prev_output,
        act_scale, weight_scale,
        N, C, H, W, O,
        x_int8.stride(0), x_int8.stride(1), x_int8.stride(2), x_int8.stride(3),
        weight_int8.stride(0), weight_int8.stride(1), weight_int8.stride(2), weight_int8.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        SCALE_W_IS_VECTOR=is_vector_scale_w,
    )

    return y


def conv2d_int8_triton_direct(x_int8: torch.Tensor, weight_int8: torch.Tensor, act_scale: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor):
    """Direct INT8 conv (3x3 s1 p1) using Implicit GEMM."""
    return conv2d_int8_triton_accumulate(x_int8, weight_int8, None, act_scale, weight_scale, bias)
