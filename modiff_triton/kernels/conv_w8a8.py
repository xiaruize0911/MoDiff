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
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4),
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
    weight_scale_ptr,  # fp32 scalar
    N, C, H, W, O,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wk1, stride_wk2,
    stride_yn, stride_yo, stride_yh, stride_yw,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implicit GEMM Convolution Kernel.
    M = N*H*W (Output pixels)
    N = O (Output channels)
    K = C (Input channels) - we iterate over 3x3 spatial locations in outer loop
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Output pixel indices (M dimension)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (N * H * W)
    
    # Output channel indices (N dimension)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < O
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Iterate over 3x3 kernel spatial locations
    for r in range(3):
        for s in range(3):
            # Iterate over C (K dimension)
            for k in range(0, C, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < C
                
                # --- Load Weights (B matrix) ---
                # Shape: [O, C, 3, 3] -> [O, C] slice for current (r, s)
                # We want [K, N] for dot product, so [C, O]
                # w_ptr offset: o*stride_wo + c*stride_wc + r*stride_wk1 + s*stride_wk2
                
                # Load B tile: [BLOCK_K, BLOCK_N]
                # offs_k[:, None] is C dimension
                # offs_n[None, :] is O dimension
                b_ptrs = w_ptr + \
                         offs_n[None, :] * stride_wo + \
                         offs_k[:, None] * stride_wc + \
                         r * stride_wk1 + \
                         s * stride_wk2
                
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
                
                # --- Load Input (A matrix) ---
                # Shape: [N, C, H, W]
                # We want [M, K] -> [N*H*W, C]
                
                # Vectorized coordinate computation
                batch_idx = offs_m // (H * W)
                hw_idx = offs_m % (H * W)
                h_idx = hw_idx // W
                w_idx = hw_idx % W
                
                # Apply kernel offset
                h_in = h_idx + r - 1
                w_in = w_idx + s - 1
                
                # Check bounds
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                
                # Input pointers: n*stride_xn + c*stride_xc + h*stride_xh + w*stride_xw
                a_ptrs = x_ptr + \
                         batch_idx[:, None] * stride_xn + \
                         offs_k[None, :] * stride_xc + \
                         h_in[:, None] * stride_xh + \
                         w_in[:, None] * stride_xw
                
                # Mask: valid M, valid K, and spatial bounds
                mask_a = mask_m[:, None] & mask_k[None, :] & in_bounds[:, None]
                
                a = tl.load(a_ptrs, mask=mask_a, other=0)
                
                # --- Matrix Multiply ---
                # a: [BLOCK_M, BLOCK_K]
                # b: [BLOCK_K, BLOCK_N] (transposed load effectively)
                acc += tl.dot(a, b)

    # --- Epilogue ---
    scale_a = tl.load(act_scale_ptr)
    scale_w = tl.load(weight_scale_ptr)
    
    # Convert to float and scale
    out = acc.to(tl.float32) * scale_a * scale_w
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        out = out + bias[None, :]
        
    # Add previous output if provided
    # Output pointer: n*stride_yn + o*stride_yo + h*stride_yh + w*stride_yw
    # Map M back to (n, h, w)
    batch_idx = offs_m // (H * W)
    hw_idx = offs_m % (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
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
    
    # Grid dimensions
    # M = N*H*W
    # N = O
    grid = lambda META: (
        triton.cdiv(N * H * W, META['BLOCK_M']),
        triton.cdiv(O, META['BLOCK_N'])
    )
    
    _conv3x3_w8a8_implicit_gemm_kernel[grid](
        x_int8, weight_int8, bias, y, prev_output,
        act_scale, weight_scale,
        N, C, H, W, O,
        x_int8.stride(0), x_int8.stride(1), x_int8.stride(2), x_int8.stride(3),
        weight_int8.stride(0), weight_int8.stride(1), weight_int8.stride(2), weight_int8.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )

    return y


def conv2d_int8_triton_direct(x_int8: torch.Tensor, weight_int8: torch.Tensor, act_scale: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor):
    """Direct INT8 conv (3x3 s1 p1) using Implicit GEMM."""
    return conv2d_int8_triton_accumulate(x_int8, weight_int8, None, act_scale, weight_scale, bias)
