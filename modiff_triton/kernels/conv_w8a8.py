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
        triton.Config({'BLOCK_K': 32, 'BLOCK_O': 32}, num_warps=4),
        triton.Config({'BLOCK_K': 64, 'BLOCK_O': 32}, num_warps=4),
        triton.Config({'BLOCK_K': 64, 'BLOCK_O': 64}, num_warps=8),
    ],
    key=['C', 'O'],
)
@triton.jit
def _conv3x3_w8a8_kernel(
    x_ptr,          # int8 [N, C, H, W]
    w_ptr,          # int8 [O, C, 3, 3]
    b_ptr,          # fp32 [O] or nullptr
    y_ptr,          # fp32 [N, O, H, W]
    act_scale_ptr,  # fp32 scalar
    weight_scale_ptr,  # fp32 scalar
    N, C, H, W, O,
    stride_n, stride_c, stride_h, stride_w,
    stride_wo, stride_wc, stride_wk1, stride_wk2,
    stride_yn, stride_yo, stride_yh, stride_yw,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_co = tl.program_id(1)

    hw = pid_hw % (H * W)
    n = pid_hw // (H * W)
    h = hw // W
    w = hw % W

    co_start = pid_co * BLOCK_O
    offs_co = co_start + tl.arange(0, BLOCK_O)
    mask_co = offs_co < O

    acc = tl.zeros((BLOCK_O,), dtype=tl.int32)

    K_total = C * 9  # 3x3 kernel

    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total

        k_c = offs_k // 9
        k_hw = offs_k % 9
        k_dh = k_hw // 3 - 1  # -1,0,1
        k_dw = k_hw % 3 - 1   # -1,0,1
        k_h_idx = k_dh + 1    # 0,1,2
        k_w_idx = k_dw + 1    # 0,1,2

        h_in = h + k_dh
        w_in = w + k_dw
        in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
        mask = k_mask & in_bounds

        x_ptrs = x_ptr + n * stride_n + k_c * stride_c + h_in * stride_h + w_in * stride_w
        a = tl.load(x_ptrs, mask=mask, other=0).to(tl.int32)  # [BLOCK_K]

        w_ptrs = w_ptr + offs_co[None, :] * stride_wo + k_c[:, None] * stride_wc + k_h_idx[:, None] * stride_wk1 + k_w_idx[:, None] * stride_wk2
        b = tl.load(w_ptrs, mask=mask[:, None] & mask_co[None, :], other=0).to(tl.int32)  # [BLOCK_K, BLOCK_O]

        acc += tl.sum(a[:, None] * b, axis=0)

    scale_a = tl.load(act_scale_ptr)
    scale_w = tl.load(weight_scale_ptr)
    out = acc.to(tl.float32) * scale_a * scale_w

    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0)
        out = out + bias

    y_ptrs = y_ptr + n * stride_yn + offs_co * stride_yo + h * stride_yh + w * stride_yw
    tl.store(y_ptrs, out, mask=mask_co)


def conv2d_int8_triton_direct(x_int8: torch.Tensor, weight_int8: torch.Tensor, act_scale: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor):
    """Direct INT8 conv (3x3 s1 p1) using Triton, output FP32.

    Args:
        x_int8: [N, C, H, W] int8
        weight_int8: [O, C, 3, 3] int8
        act_scale: scalar tensor
        weight_scale: scalar tensor
        bias: [O] float or None
    """
    assert x_int8.ndim == 4 and weight_int8.ndim == 4
    N, C, H, W = x_int8.shape
    O = weight_int8.shape[0]
    assert weight_int8.shape[2:] == (3, 3)

    y = torch.empty((N, O, H, W), device=x_int8.device, dtype=torch.float32)

    grid = (N * H * W, triton.cdiv(O, 32))
    _conv3x3_w8a8_kernel[grid](
        x_int8, weight_int8, bias, y,
        act_scale, weight_scale,
        N, C, H, W, O,
        x_int8.stride(0), x_int8.stride(1), x_int8.stride(2), x_int8.stride(3),
        weight_int8.stride(0), weight_int8.stride(1), weight_int8.stride(2), weight_int8.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )

    return y
