"""
W4A4 GEMM Kernels for MoDiff

This module implements INT4 matrix multiplication kernels for MoDiff.
INT4 provides maximum compression but requires careful handling:
    - Two INT4 values packed per INT8 byte
    - Unpacking during computation
    - Higher numerical sensitivity (only 16 levels)

The paper shows MoDiff enables W4A4 to achieve near-W8A8 accuracy because:
    "The residual (a_t - â_{t+1}) has a much smaller range, more than 10x smaller,
     which suggests that activation bit precision can be lowered by at least 3 bits
     while maintaining comparable quantization error."
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Helper Functions for INT4 Packing
# ============================================================================

@triton.jit
def unpack_int4_to_int8(packed):
    """
    Unpack two INT4 values from one INT8.
    Returns two int8 values in range [-8, 7].
    """
    lo = ((packed & 0xF).to(tl.int8) - 8)
    hi = (((packed >> 4) & 0xF).to(tl.int8) - 8)
    return lo, hi


# ============================================================================
# W4A4 GEMM Kernel
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_w4a4_kernel(
    # Pointers
    A_packed_ptr,   # [M, K//2] packed INT4 activations
    B_packed_ptr,   # [K//2, N] packed INT4 weights
    C_ptr,          # [M, N] output
    # Scales
    scale_a_ptr, scale_b_ptr,
    # Bias
    bias_ptr,
    # Dimensions (K is the unpacked dimension)
    M, N, K,
    # Strides (for packed tensors)
    stride_am, stride_ak,  # stride_ak is for K//2
    stride_bk, stride_bn,  # stride_bk is for K//2
    stride_cm, stride_cn,
    # Flags
    HAS_BIAS: tl.constexpr,
    # Block sizes (K is for unpacked)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    W4A4 GEMM with packed INT4 inputs.
    
    A: [M, K//2] packed (2 INT4 per byte)
    B: [K//2, N] packed (2 INT4 per byte)
    C: [M, N] FP32
    
    The kernel unpacks INT4 on-the-fly during computation.
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
    
    # Offsets for output
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # For packed K dimension (K//2)
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    offs_k_packed = tl.arange(0, BLOCK_K_PACKED)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    K_packed = K // 2
    
    # Main loop over K (packed)
    for k_packed in range(0, K_packed, BLOCK_K_PACKED):
        k_mask = (k_packed + offs_k_packed) < K_packed
        
        # Load packed A: [BLOCK_M, BLOCK_K//2]
        a_ptrs = A_packed_ptr + (offs_m[:, None] * stride_am + (k_packed + offs_k_packed[None, :]) * stride_ak)
        a_packed = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        
        # Load packed B: [BLOCK_K//2, BLOCK_N]
        b_ptrs = B_packed_ptr + ((k_packed + offs_k_packed[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_packed = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
        
        # Unpack INT4 to INT8
        # A: each element contains 2 values for consecutive K indices
        a_lo = ((a_packed & 0xF).to(tl.int8) - 8)  # [BLOCK_M, BLOCK_K//2]
        a_hi = (((a_packed >> 4) & 0xF).to(tl.int8) - 8)
        
        # B: each element contains 2 values for consecutive K indices  
        b_lo = ((b_packed & 0xF).to(tl.int8) - 8)  # [BLOCK_K//2, BLOCK_N]
        b_hi = (((b_packed >> 4) & 0xF).to(tl.int8) - 8)
        
        # Compute: need to handle the interleaved structure
        # a_lo[i, j] corresponds to K index 2*j
        # a_hi[i, j] corresponds to K index 2*j+1
        # Similarly for B
        
        # Method: compute separately and sum
        # acc += a_lo @ b_lo + a_hi @ b_hi
        acc += tl.dot(a_lo, b_lo, out_dtype=tl.int32)
        acc += tl.dot(a_hi, b_hi, out_dtype=tl.int32)
    
    # Dequantize
    scale_a = tl.load(scale_a_ptr)
    scale_b = tl.load(scale_b_ptr)
    c = acc.to(tl.float32) * scale_a * scale_b
    
    # Add bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        c += bias[None, :]
    
    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# W4A4 GEMM with Output Accumulation
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_w4a4_accum_kernel(
    # Pointers
    A_packed_ptr, B_packed_ptr, C_ptr,
    # Scales
    scale_a_ptr, scale_b_ptr,
    # Cache (ô_{t+1})
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
):
    """
    MoDiff W4A4 GEMM with output accumulation.
    
    Implements Eq. (ec6): ô_t = A(Q_4bit(a_t - â_{t+1})) + ô_{t+1}
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
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    offs_k_packed = tl.arange(0, BLOCK_K_PACKED)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    K_packed = K // 2
    
    for k_packed in range(0, K_packed, BLOCK_K_PACKED):
        k_mask = (k_packed + offs_k_packed) < K_packed
        
        a_ptrs = A_packed_ptr + (offs_m[:, None] * stride_am + (k_packed + offs_k_packed[None, :]) * stride_ak)
        a_packed = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        
        b_ptrs = B_packed_ptr + ((k_packed + offs_k_packed[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_packed = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
        
        a_lo = ((a_packed & 0xF).to(tl.int8) - 8)
        a_hi = (((a_packed >> 4) & 0xF).to(tl.int8) - 8)
        b_lo = ((b_packed & 0xF).to(tl.int8) - 8)
        b_hi = (((b_packed >> 4) & 0xF).to(tl.int8) - 8)
        
        acc += tl.dot(a_lo, b_lo, out_dtype=tl.int32)
        acc += tl.dot(a_hi, b_hi, out_dtype=tl.int32)
    
    # Dequantize
    scale_a = tl.load(scale_a_ptr)
    scale_b = tl.load(scale_b_ptr)
    c = acc.to(tl.float32) * scale_a * scale_b
    
    # Add cached output (MoDiff accumulation)
    cache_ptrs = cache_ptr + (offs_m[:, None] * stride_cache_m + offs_n[None, :] * stride_cache_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    cache = tl.load(cache_ptrs, mask=c_mask, other=0.0)
    c = c + cache
    
    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def pack_int4_weight(weight_fp: torch.Tensor) -> tuple:
    """
    Quantize and pack FP weight to INT4.
    
    Args:
        weight_fp: FP16/FP32 weight tensor [K, N]
        
    Returns:
        weight_packed: Packed INT4 weight [K//2, N]
        scale: Weight scale
    """
    K, N = weight_fp.shape
    assert K % 2 == 0, "K must be even for INT4 packing"
    
    # Compute scale
    weight_max = weight_fp.abs().max()
    scale = weight_max / 7.0
    scale = torch.clamp(scale, min=1e-8)
    
    # Quantize to [-8, 7]
    weight_int = torch.round(weight_fp / scale).clamp(-8, 7)
    
    # Convert to unsigned [0, 15]
    weight_uint = (weight_int + 8).to(torch.uint8)
    
    # Pack: pairs along K dimension
    weight_lo = weight_uint[0::2, :]  # [K//2, N]
    weight_hi = weight_uint[1::2, :]  # [K//2, N]
    weight_packed = (weight_lo & 0xF) | ((weight_hi & 0xF) << 4)
    
    return weight_packed.to(torch.int8), scale


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack INT4 from INT8 container."""
    lo = (packed & 0xF).to(torch.int8) - 8  # Convert unsigned to signed
    hi = ((packed >> 4) & 0xF).to(torch.int8) - 8
    
    # Interleave
    unpacked = torch.zeros(packed.shape[0] * 2, packed.shape[1], dtype=torch.float32, device=packed.device)
    unpacked[0::2, :] = lo.float()
    unpacked[1::2, :] = hi.float()
    
    return unpacked


def gemm_w4a4(
    A_packed: torch.Tensor,     # [M, K//2] packed INT4
    B_packed: torch.Tensor,     # [K//2, N] packed INT4
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    K: int,                     # Unpacked K dimension
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    W4A4 GEMM: C = dequant(A @ B) + bias
    
    Args:
        A_packed: Packed INT4 activation [M, K//2]
        B_packed: Packed INT4 weight [K//2, N]
        scale_a: Activation scale
        scale_b: Weight scale
        K: Original (unpacked) K dimension
        bias: Optional bias
        
    Returns:
        C: Output [M, N] in FP32
    """
    M, K_packed = A_packed.shape
    K_packed2, N = B_packed.shape
    assert K_packed == K_packed2
    assert K_packed == K // 2
    
    if not isinstance(scale_a, torch.Tensor):
        scale_a = torch.tensor(scale_a, dtype=torch.float32, device=A_packed.device)
    if not isinstance(scale_b, torch.Tensor):
        scale_b = torch.tensor(scale_b, dtype=torch.float32, device=A_packed.device)
    
    # Unpack and dequantize
    # A_packed: [M, K//2] -> A_unpacked: [M, K]
    A_lo = ((A_packed & 0xF).float() - 8) * scale_a  # [M, K//2]
    A_hi = (((A_packed >> 4) & 0xF).float() - 8) * scale_a  # [M, K//2]
    A_fp = torch.zeros(M, K, dtype=torch.float32, device=A_packed.device)
    A_fp[:, 0::2] = A_lo
    A_fp[:, 1::2] = A_hi
    
    # B_packed: [K//2, N] -> B_unpacked: [K, N]
    B_lo = ((B_packed & 0xF).float() - 8) * scale_b  # [K//2, N]
    B_hi = (((B_packed >> 4) & 0xF).float() - 8) * scale_b  # [K//2, N]
    B_fp = torch.zeros(K, N, dtype=torch.float32, device=B_packed.device)
    B_fp[0::2, :] = B_lo
    B_fp[1::2, :] = B_hi
    
    # Matrix multiplication
    C = torch.mm(A_fp, B_fp)
    
    if bias is not None:
        C = C + bias
    
    return C


def gemm_w4a4_accum(
    A_packed: torch.Tensor,     # [M, K//2] packed INT4
    B_packed: torch.Tensor,     # [K//2, N] packed INT4
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    K: int,
    cache: torch.Tensor,        # [M, N] - ô_{t+1}
) -> torch.Tensor:
    """
    MoDiff W4A4 GEMM with output accumulation.
    
    Implements Eq. (ec6) with INT4.
    """
    M, K_packed = A_packed.shape
    K_packed2, N = B_packed.shape
    assert K_packed == K_packed2
    assert cache.shape == (M, N)
    
    if not isinstance(scale_a, torch.Tensor):
        scale_a = torch.tensor(scale_a, dtype=torch.float32, device=A_packed.device)
    if not isinstance(scale_b, torch.Tensor):
        scale_b = torch.tensor(scale_b, dtype=torch.float32, device=A_packed.device)
    
    # Unpack and dequantize A
    A_lo = ((A_packed & 0xF).float() - 8) * scale_a
    A_hi = (((A_packed >> 4) & 0xF).float() - 8) * scale_a
    A_fp = torch.zeros(M, K, dtype=torch.float32, device=A_packed.device)
    A_fp[:, 0::2] = A_lo
    A_fp[:, 1::2] = A_hi
    
    # Unpack and dequantize B
    B_lo = ((B_packed & 0xF).float() - 8) * scale_b
    B_hi = (((B_packed >> 4) & 0xF).float() - 8) * scale_b
    B_fp = torch.zeros(K, N, dtype=torch.float32, device=B_packed.device)
    B_fp[0::2, :] = B_lo
    B_fp[1::2, :] = B_hi
    
    # Matrix multiplication with accumulation
    C = torch.mm(A_fp, B_fp) + cache
    
    return C
