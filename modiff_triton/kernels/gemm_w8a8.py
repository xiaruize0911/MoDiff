"""
W8A8 GEMM Kernels for MoDiff

This module implements INT8 matrix multiplication kernels for MoDiff,
following paper Eq. (ec6):
    ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}

Two variants:
    1. gemm_w8a8: Standard INT8 GEMM with dequantization
    2. gemm_w8a8_accum: GEMM with output accumulation (adds ô_{t+1})

The kernel performs:
    Output = dequant(INT8_GEMM(activation_int8, weight_int8)) + bias + cache
    
Where:
    - activation_int8: Quantized residual Q(a_t - â_{t+1})
    - weight_int8: Pre-quantized weights
    - cache: ô_{t+1} from previous timestep (for t < T)
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# W8A8 GEMM Kernel (Standard)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_w8a8_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    # Scales
    scale_a_ptr, scale_b_ptr,
    # Bias (optional)
    bias_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Flags
    HAS_BIAS: tl.constexpr,
    SCALE_B_IS_VECTOR: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    INT8 GEMM: C = dequant(A_int8 @ B_int8) + bias
    
    A: [M, K] INT8 (activation)
    B: [K, N] INT8 (weight)
    C: [M, N] FP16/FP32
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator in INT32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        
        # Load A and B blocks
        a = tl.load(a_ptrs, mask=k_mask[None, :] & (offs_m[:, None] < M), other=0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
        
        # INT8 dot product
        acc += tl.dot(a.to(tl.int8), b.to(tl.int8), out_dtype=tl.int32)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Load scales
    scale_a = tl.load(scale_a_ptr)
    if SCALE_B_IS_VECTOR:
        scale_b = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    else:
        scale_b = tl.load(scale_b_ptr)
    
    # Dequantize: FP = INT32 * scale_a * scale_b
    c = acc.to(tl.float32) * scale_a * scale_b
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        c += bias[None, :]
    
    # Store output
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# W8A8 GEMM with Output Accumulation (MoDiff Eq. ec6)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_w8a8_accum_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    # Scales
    scale_a_ptr, scale_b_ptr,
    # Cache for accumulation (ô_{t+1})
    cache_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_cache_m, stride_cache_n,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SCALE_B_IS_VECTOR: tl.constexpr,
):
    """
    MoDiff GEMM with output accumulation.
    
    Implements Eq. (ec6): ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}
    
    C = dequant(A_int8 @ B_int8) + cache
    
    Where:
        - A: Quantized residual Q(a_t - â_{t+1})
        - B: Quantized weights
        - cache: Previous output ô_{t+1}
        
    Note: Bias is NOT added here (only at first timestep T)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Main GEMM loop
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        
        a = tl.load(a_ptrs, mask=k_mask[None, :] & (offs_m[:, None] < M), other=0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
        
        acc += tl.dot(a.to(tl.int8), b.to(tl.int8), out_dtype=tl.int32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Load scales and dequantize
    scale_a = tl.load(scale_a_ptr)
    if SCALE_B_IS_VECTOR:
        scale_b = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    else:
        scale_b = tl.load(scale_b_ptr)
    c = acc.to(tl.float32) * scale_a * scale_b
    
    # Load and add cached output ô_{t+1} (MoDiff accumulation)
    cache_ptrs = cache_ptr + (offs_m[:, None] * stride_cache_m + offs_n[None, :] * stride_cache_n)
    cache_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    cache = tl.load(cache_ptrs, mask=cache_mask, other=0.0)
    
    c = c + cache
    
    # Store output
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=cache_mask)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def gemm_w8a8(
    A_int8: torch.Tensor,       # [M, K] INT8
    B_int8: torch.Tensor,       # [K, N] INT8
    scale_a: torch.Tensor,      # Scalar
    scale_b: torch.Tensor,      # Scalar or [N]
    bias: torch.Tensor = None,  # [N] or None
) -> torch.Tensor:
    """
    W8A8 GEMM: C = dequant(A @ B) + bias
    
    Used for first timestep (t=T) in MoDiff: ô_T = A(â_T) + bias
    
    Args:
        A_int8: INT8 activation tensor [M, K]
        B_int8: INT8 weight tensor [K, N]
        scale_a: Activation scale
        scale_b: Weight scale
        bias: Optional bias
        
    Returns:
        C: Output tensor [M, N] in FP32
    """
    assert A_int8.dtype == torch.int8
    assert B_int8.dtype == torch.int8
    
    M, K = A_int8.shape
    K2, N = B_int8.shape
    assert K == K2, f"Shape mismatch: A[{M}, {K}] @ B[{K2}, {N}]"
    
    # Ensure scales are tensors
    if not isinstance(scale_a, torch.Tensor):
        scale_a = torch.tensor(scale_a, dtype=torch.float32, device=A_int8.device)
    if not isinstance(scale_b, torch.Tensor):
        scale_b = torch.tensor(scale_b, dtype=torch.float32, device=A_int8.device)
    
    # Allocate output
    C = torch.empty((M, N), device=A_int8.device, dtype=torch.float32)
    
    is_vector_scale_b = (scale_b.numel() > 1)

    # Grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    _gemm_w8a8_kernel[grid](
        A_int8, B_int8, C,
        scale_a, scale_b,
        bias if bias is not None else A_int8, # Dummy if None
        M, N, K,
        A_int8.stride(0), A_int8.stride(1),
        B_int8.stride(0), B_int8.stride(1),
        C.stride(0), C.stride(1),
        HAS_BIAS=(bias is not None),
        SCALE_B_IS_VECTOR=is_vector_scale_b,
    )
    
    return C


def gemm_w8a8_accum(
    A_int8: torch.Tensor,       # [M, K] INT8 - quantized residual
    B_int8: torch.Tensor,       # [K, N] INT8 - quantized weight
    scale_a: torch.Tensor,      # Activation scale
    scale_b: torch.Tensor,      # Weight scale
    cache: torch.Tensor,        # [M, N] - ô_{t+1}
) -> torch.Tensor:
    """
    MoDiff W8A8 GEMM with output accumulation.
    
    Implements Eq. (ec6): ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}
    
    Args:
        A_int8: Quantized residual Q(a_t - â_{t+1}) [M, K]
        B_int8: Quantized weights [K, N]
        scale_a: Activation (residual) scale
        scale_b: Weight scale
        cache: Previous output ô_{t+1} [M, N]
        
    Returns:
        C: ô_t [M, N] in FP32
    """
    assert A_int8.dtype == torch.int8
    assert B_int8.dtype == torch.int8
    
    M, K = A_int8.shape
    K2, N = B_int8.shape
    assert K == K2
    assert cache.shape == (M, N), f"Cache shape mismatch: expected ({M}, {N}), got {cache.shape}"
    
    # Ensure scales are tensors
    if not isinstance(scale_a, torch.Tensor):
        scale_a = torch.tensor(scale_a, dtype=torch.float32, device=A_int8.device)
    if not isinstance(scale_b, torch.Tensor):
        scale_b = torch.tensor(scale_b, dtype=torch.float32, device=A_int8.device)
    
    # Allocate output
    C = torch.empty((M, N), device=A_int8.device, dtype=torch.float32)
    
    is_vector_scale_b = (scale_b.numel() > 1)

    # Grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    _gemm_w8a8_accum_kernel[grid](
        A_int8, B_int8, C,
        scale_a, scale_b,
        cache,
        M, N, K,
        A_int8.stride(0), A_int8.stride(1),
        B_int8.stride(0), B_int8.stride(1),
        C.stride(0), C.stride(1),
        cache.stride(0), cache.stride(1),
        SCALE_B_IS_VECTOR=is_vector_scale_b,
    )
    
    return C


def gemm_w8a8_modiff_full(
    A_int8: torch.Tensor,       # [M, K] INT8
    B_int8: torch.Tensor,       # [K, N] INT8
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    cache: torch.Tensor = None,
    is_first_step: bool = False,
) -> torch.Tensor:
    """
    Complete MoDiff W8A8 GEMM.
    
    At t=T (first step):
        ô_T = A(â_T) + bias
        
    At t<T:
        ô_t = A(Q(residual)) + ô_{t+1}  (no bias)
    """
    if is_first_step or cache is None:
        return gemm_w8a8(A_int8, B_int8, scale_a, scale_b, bias)
    else:
        return gemm_w8a8_accum(A_int8, B_int8, scale_a, scale_b, cache)
