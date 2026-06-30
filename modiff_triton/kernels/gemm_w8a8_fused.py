"""
Fused Quantization + GEMM Kernel for W8A8

This kernel fuses the quantization and GEMM operations to eliminate:
1. Intermediate memory traffic (quantized activations)
2. Separate kernel launch overhead
3. Memory bandwidth bottleneck

Performance target: Match cuBLAS FP32 for small matrices (512×512)
"""

import torch
import triton
import triton.language as tl

_AWQ_WEIGHT_CACHE: dict[tuple[int, tuple[int, int], tuple[int, int], str], torch.Tensor] = {}


def _cached_awq_weight(weight_int8: torch.Tensor) -> torch.Tensor:
    """Return AWQ's [N, K] layout without paying transpose cost every call."""
    key = (
        weight_int8.data_ptr(),
        tuple(weight_int8.shape),
        tuple(weight_int8.stride()),
        str(weight_int8.device),
    )
    cached = _AWQ_WEIGHT_CACHE.get(key)
    if cached is None or cached.device != weight_int8.device:
        cached = weight_int8.t().contiguous()
        _AWQ_WEIGHT_CACHE[key] = cached
    return cached


def _should_use_awq_fused(x: torch.Tensor, weight_int8: torch.Tensor) -> bool:
    if not x.is_cuda or x.dim() != 2 or weight_int8.dim() != 2:
        return False
    m, k = x.shape
    k_weight, n = weight_int8.shape
    return (
        k == k_weight
        and m > 128
        and n == k
        and n % 2 == 0
        and n in (512, 2048, 4096)
    )


@triton.autotune(
    configs=[
        # INT8-optimized configs with larger BLOCK_K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_quantize_gemm_w8a8_kernel(
    # Input activation (FP32/FP16)
    X_ptr,
    # Weight (INT8, pre-quantized)
    W_ptr,
    # Output
    Out_ptr,
    # Weight scale
    scale_w_ptr,
    # Bias (optional)
    bias_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    # Flags
    HAS_BIAS: tl.constexpr,
    SCALE_W_IS_VECTOR: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused quantize + GEMM kernel.
    
    Computes: Out = (quant(X) @ W_int8) * scale_x * scale_w + bias
    
    Where quantization is done on-the-fly in shared memory without
    writing intermediate results to global memory.
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compiler hints for optimization
    
    # Block offsets
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    # ========================================================================
    # Step 1: Compute quantization scale for X
    # ========================================================================
    # We need to find max(|X[m, :]|) for each row to compute per-row scale
    # For simplicity, we'll compute global scale for the block
    
    # Load first block to compute scale
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    mask_m = offs_m[:, None] < M
    
    # Compute max absolute value across all K for this M block
    abs_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        x_block = tl.load(x_ptrs + k * BLOCK_K * stride_xk, mask=mask_m & k_mask[None, :], other=0.0)
        x_abs = tl.abs(x_block)
        block_max = tl.max(x_abs, axis=1)
        abs_max = tl.maximum(abs_max, block_max)
    
    # Compute scale: scale = max(|x|) / 127
    # Add small epsilon to avoid division by zero
    scale_x = abs_max / 127.0
    scale_x = tl.where(scale_x < 1e-8, 1e-8, scale_x)
    
    # ========================================================================
    # Step 2: Fused quantization + GEMM
    # ========================================================================
    
    # Reset pointers
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    
    # Accumulator in INT32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Main GEMM loop with fused quantization
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        
        # Load FP32/FP16 activation
        x_fp = tl.load(x_ptrs, mask=mask_m & k_mask[None, :], other=0.0)
        
        # Quantize on-the-fly: x_int8 = clamp(round(x / scale), -128, 127)
        x_scaled = x_fp / scale_x[:, None]
        x_int = tl.floor(x_scaled + 0.5)  # Round
        x_int = tl.maximum(tl.minimum(x_int, 127.0), -128.0)  # Clamp
        
        # Load INT8 weight
        w_int = tl.load(w_ptrs, mask=k_mask[:, None], other=0)
        
        # INT8 GEMM with accumulator
        acc = tl.dot(x_int.to(tl.int8), w_int.to(tl.int8), acc, out_dtype=tl.int32)
        
        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # ========================================================================
    # Step 3: Dequantize and write output
    # ========================================================================
    
    # Convert to FP32 for dequantization
    acc_fp = acc.to(tl.float32)
    
    # Load weight scale
    if SCALE_W_IS_VECTOR:
        scale_w = tl.load(scale_w_ptr + offs_n, mask=offs_n < N, other=1.0)
        scale_combined = scale_x[:, None] * scale_w[None, :]
    else:
        scale_w = tl.load(scale_w_ptr)
        scale_combined = scale_x[:, None] * scale_w
    
    # Dequantize
    out = acc_fp * scale_combined
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        out += bias[None, :]
    
    # Store output
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out, mask=out_mask)


def gemm_w8a8_fused(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale_w: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused quantization + GEMM for W8A8.
    
    This kernel quantizes the activation on-the-fly and performs GEMM
    without storing intermediate quantized activations.
    
    Args:
        x: Input activation [M, K] (FP32/FP16)
        weight_int8: Pre-quantized weight [K, N] (INT8)
        scale_w: Weight quantization scale (scalar or [N])
        bias: Optional bias [N]
        
    Returns:
        output: [M, N] (FP32/FP16)
    """
    # Input validation
    assert x.dim() == 2, "Input must be 2D"
    assert weight_int8.dim() == 2, "Weight must be 2D"
    assert x.shape[1] == weight_int8.shape[0], "Dimension mismatch"
    
    M, K = x.shape
    K, N = weight_int8.shape

    # AWQ-style fused path for verified square projection GEMMs. The original
    # Triton kernel below computes per-row activation scales inside every N tile,
    # which repeats the full K scan for each output-column block. AWQ quantizes
    # once per token, then runs a tuned dense INT8 kernel with fused dequant/bias.
    if _should_use_awq_fused(x, weight_int8):
        try:
            from .awq_w8a8 import awq_fused_quant_gemm_w8a8

            return awq_fused_quant_gemm_w8a8(
                x,
                _cached_awq_weight(weight_int8),
                scale_w,
                bias,
                weight_is_awq_layout=True,
            )
        except ImportError:
            pass
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Check if we should use PyTorch fast path
    # For very small matrices, PyTorch might be faster due to kernel launch overhead
    if M <= 16 and N <= 16:
        # Use standard quantize + torch._int_mm path for tiny matrices
        from .quantize import quantize_symmetric_int8
        from .gemm_w8a8 import gemm_w8a8
        x_int, scale_x = quantize_symmetric_int8(x)
        return gemm_w8a8(x_int, weight_int8, scale_x, scale_w, bias)
    
    # Determine if scale_w is vector
    scale_w_is_vector = scale_w.numel() > 1
    has_bias = bias is not None
    
    # Grid configuration
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    _fused_quantize_gemm_w8a8_kernel[grid](
        x, weight_int8, output,
        scale_w,
        bias if has_bias else x,  # Dummy pointer if no bias
        M, N, K,
        x.stride(0), x.stride(1),
        weight_int8.stride(0), weight_int8.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=has_bias,
        SCALE_W_IS_VECTOR=scale_w_is_vector,
    )
    
    return output
