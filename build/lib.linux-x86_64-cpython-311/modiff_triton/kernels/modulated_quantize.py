"""
Modulated Quantization Kernels for MoDiff

This module implements the core MoDiff modulated quantization operation
following the paper's Error-Compensated Modulation (Section 3.3):

    At timestep t < T:
        â_t = Q(a_t - â_{t+1}) + â_{t+1}          -- Eq. (ec5)
        
    Where the quantization error is:
        e_t = (a_t - â_{t+1}) - Q(a_t - â_{t+1})
            = a_t - â_t

The key insight from the paper:
    "The residual should be computed based on â_{t} instead of a_{t}, 
     which will compensate the errors and avoid error accumulation."

This implements the fused operation: compute residual -> quantize -> update cache
"""

import torch
import triton
import triton.language as tl
from .quantize import compute_dynamic_scale_int8, compute_dynamic_scale_int4


# Helper function for rounding (compatible with different Triton versions)
@triton.jit
def _round(x):
    """Round to nearest integer (compatible with all Triton versions)."""
    return tl.floor(x + 0.5)


@triton.jit
def _modulated_max_abs_kernel(
    a_ptr, b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes max(|a - b|) for each block.
    Used to compute dynamic scale without materializing the residual tensor.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    diff = tl.abs(a - b)
    max_val = tl.max(diff)
    
    tl.store(out_ptr + pid, max_val)


# ============================================================================
# INT8 Modulated Quantization Kernel
# ============================================================================

@triton.jit
def _modulated_quantize_int8_kernel(
    a_t_ptr,            # Current activation a_t [M, K]
    a_hat_prev_ptr,     # Cached â_{t+1} [M, K]
    residual_int_ptr,   # Output: Q(a_t - â_{t+1}) as INT8
    a_hat_new_ptr,      # Output: â_t = Q(a_t - â_{t+1}) + â_{t+1}
    scale_ptr,          # Output: quantization scale
    M, K,               # Dimensions
    stride_am, stride_ak,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused modulated quantization kernel (INT8).
    
    Implements Eq. (ec5): â_t = Q(a_t - â_{t+1}) + â_{t+1}
    
    Operations:
        1. residual = a_t - â_{t+1}
        2. scale = max(|residual|) / 127
        3. residual_int = round(residual / scale)
        4. residual_dequant = residual_int * scale
        5. â_t = residual_dequant + â_{t+1}
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Create masks
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]
    
    # Compute memory offsets
    offs = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    
    # Load a_t and â_{t+1}
    a_t = tl.load(a_t_ptr + offs, mask=mask, other=0.0)
    a_hat_prev = tl.load(a_hat_prev_ptr + offs, mask=mask, other=0.0)
    
    # Step 1: Compute residual (a_t - â_{t+1})
    residual = a_t - a_hat_prev
    
    # Step 2: Compute dynamic scale for this block
    # For simplicity, we compute per-block max; in practice, you'd want global scale
    residual_abs = tl.abs(residual)
    block_max = tl.max(residual_abs)
    scale = block_max / 127.0
    scale = tl.where(scale < 1e-8, 1e-8, scale)
    
    # Step 3: Quantize residual
    residual_scaled = residual / scale
    residual_int = _round(residual_scaled)
    residual_int = tl.maximum(tl.minimum(residual_int, 127.0), -128.0)
    
    # Step 4: Dequantize for error compensation
    residual_dequant = residual_int * scale
    
    # Step 5: Compute â_t = Q(residual) + â_{t+1}
    a_hat_new = residual_dequant + a_hat_prev
    
    # Store outputs
    tl.store(residual_int_ptr + offs, residual_int.to(tl.int8), mask=mask)
    tl.store(a_hat_new_ptr + offs, a_hat_new, mask=mask)
    
    # Store scale (one per block row for simplicity)
    if pid_k == 0:
        tl.store(scale_ptr + pid_m, scale)


@triton.jit
def _modulated_quantize_int8_global_scale_kernel(
    a_t_ptr,            # Current activation a_t [M, K]
    a_hat_prev_ptr,     # Cached â_{t+1} [M, K]
    residual_int_ptr,   # Output: Q(a_t - â_{t+1}) as INT8
    a_hat_new_ptr,      # Output: â_t
    scale_ptr,          # Pre-computed global scale (pointer)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Modulated quantization with pre-computed global scale.
    More efficient when scale is computed separately.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    a_t = tl.load(a_t_ptr + offsets, mask=mask, other=0.0)
    a_hat_prev = tl.load(a_hat_prev_ptr + offsets, mask=mask, other=0.0)
    
    # Load scale
    scale = tl.load(scale_ptr)
    
    # Compute residual
    residual = a_t - a_hat_prev
    
    # Quantize with global scale
    residual_scaled = residual / scale
    residual_int = _round(residual_scaled)
    residual_int = tl.maximum(tl.minimum(residual_int, 127.0), -128.0)
    
    # Dequantize and compute new â_t
    residual_dequant = residual_int * scale
    a_hat_new = residual_dequant + a_hat_prev
    
    # Store
    tl.store(residual_int_ptr + offsets, residual_int.to(tl.int8), mask=mask)
    tl.store(a_hat_new_ptr + offsets, a_hat_new, mask=mask)


# ============================================================================
# INT4 Modulated Quantization Kernel
# ============================================================================

@triton.jit
def _modulated_quantize_int4_kernel(
    a_t_ptr,            # Current activation a_t
    a_hat_prev_ptr,     # Cached â_{t+1}
    residual_packed_ptr,# Output: packed INT4 residual
    a_hat_new_ptr,      # Output: â_t
    scale,              # Pre-computed global scale
    n_elements,         # Must be even
    BLOCK_SIZE: tl.constexpr,
):
    """
    Modulated quantization for INT4 with packing.
    
    Same as INT8 but:
        - Quantizes to [-8, 7] range
        - Packs two INT4 values into one INT8
    """
    pid = tl.program_id(0)
    # Process pairs of elements
    block_start = pid * BLOCK_SIZE * 2
    offs_lo = block_start + tl.arange(0, BLOCK_SIZE) * 2
    offs_hi = offs_lo + 1
    mask_lo = offs_lo < n_elements
    mask_hi = offs_hi < n_elements
    
    # Load inputs (pairs)
    a_t_lo = tl.load(a_t_ptr + offs_lo, mask=mask_lo, other=0.0)
    a_t_hi = tl.load(a_t_ptr + offs_hi, mask=mask_hi, other=0.0)
    a_hat_prev_lo = tl.load(a_hat_prev_ptr + offs_lo, mask=mask_lo, other=0.0)
    a_hat_prev_hi = tl.load(a_hat_prev_ptr + offs_hi, mask=mask_hi, other=0.0)
    
    # Compute residuals
    residual_lo = a_t_lo - a_hat_prev_lo
    residual_hi = a_t_hi - a_hat_prev_hi
    
    # Quantize to INT4 range [-8, 7]
    res_scaled_lo = residual_lo / scale
    res_scaled_hi = residual_hi / scale
    res_int_lo = _round(res_scaled_lo)
    res_int_hi = _round(res_scaled_hi)
    res_int_lo = tl.maximum(tl.minimum(res_int_lo, 7.0), -8.0)
    res_int_hi = tl.maximum(tl.minimum(res_int_hi, 7.0), -8.0)
    
    # Dequantize and update cache
    res_dequant_lo = res_int_lo * scale
    res_dequant_hi = res_int_hi * scale
    a_hat_new_lo = res_dequant_lo + a_hat_prev_lo
    a_hat_new_hi = res_dequant_hi + a_hat_prev_hi
    
    # Pack INT4: convert to unsigned [0, 15], then pack
    res_uint4_lo = (res_int_lo + 8.0).to(tl.int8)
    res_uint4_hi = (res_int_hi + 8.0).to(tl.int8)
    packed = (res_uint4_lo & 0xF) | ((res_uint4_hi & 0xF) << 4)
    
    # Store packed residual
    out_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offs < (n_elements // 2)
    tl.store(residual_packed_ptr + out_offs, packed, mask=out_mask)
    
    # Store updated cache
    tl.store(a_hat_new_ptr + offs_lo, a_hat_new_lo, mask=mask_lo)
    tl.store(a_hat_new_ptr + offs_hi, a_hat_new_hi, mask=mask_hi)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def modulated_quantize_int8(
    a_t: torch.Tensor,
    a_hat_prev: torch.Tensor,
    scale: torch.Tensor = None,
    out_a_hat: torch.Tensor = None,
) -> tuple:
    """
    MoDiff modulated quantization (INT8).
    
    Implements paper Eq. (ec5): â_t = Q(a_t - â_{t+1}) + â_{t+1}
    
    Args:
        a_t: Current activation tensor [*, K]
        a_hat_prev: Cached quantized activation â_{t+1} from previous timestep
        scale: Pre-computed scale (if None, compute dynamically from residual)
        out_a_hat: Optional output tensor for â_t (can be a_hat_prev for in-place)
        
    Returns:
        residual_int: Quantized residual Q(a_t - â_{t+1}) as INT8
        a_hat_new: Updated cache â_t = Q(a_t - â_{t+1}) + â_{t+1}
        scale: Quantization scale used
    """
    assert a_t.shape == a_hat_prev.shape, "Shape mismatch between a_t and a_hat_prev"
    
    original_shape = a_t.shape
    a_t_flat = a_t.contiguous().flatten()
    a_hat_prev_flat = a_hat_prev.contiguous().flatten()
    n_elements = a_t_flat.numel()
    
    if scale is None:
        # Dynamic scale from residual (paper: residual has smaller range)
        # Fused max(|a - b|) computation to avoid materializing residual
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        block_maxes = torch.empty(grid[0], dtype=torch.float32, device=a_t.device)
        
        _modulated_max_abs_kernel[grid](
            a_t_flat, a_hat_prev_flat,
            block_maxes,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        max_val = block_maxes.max()
        scale = max_val / 127.0
        scale = torch.clamp(scale, min=1e-8)
    
    # Allocate outputs
    residual_int = torch.empty(n_elements, dtype=torch.int8, device=a_t.device)
    
    if out_a_hat is not None:
        assert out_a_hat.shape == original_shape
        a_hat_new_flat = out_a_hat.view(-1)
    else:
        a_hat_new_flat = torch.empty(n_elements, dtype=a_t.dtype, device=a_t.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Pass scale tensor directly (or wrap float in tensor)
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=a_t.device, dtype=torch.float32)
    
    _modulated_quantize_int8_global_scale_kernel[grid](
        a_t_flat, a_hat_prev_flat,
        residual_int, a_hat_new_flat,
        scale,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape outputs
    residual_int = residual_int.view(original_shape)
    a_hat_new = a_hat_new_flat.view(original_shape)
    
    return residual_int, a_hat_new, scale


def modulated_quantize_int4(
    a_t: torch.Tensor,
    a_hat_prev: torch.Tensor,
    scale: torch.Tensor = None,
) -> tuple:
    """
    MoDiff modulated quantization (INT4).
    
    Implements paper Eq. (ec5) with INT4 quantization and packing.
    
    Args:
        a_t: Current activation tensor
        a_hat_prev: Cached quantized activation â_{t+1}
        scale: Pre-computed scale (if None, compute dynamically)
        
    Returns:
        residual_packed: Packed INT4 residual (2 values per byte)
        a_hat_new: Updated cache â_t
        scale: Quantization scale used
    """
    assert a_t.shape == a_hat_prev.shape, "Shape mismatch"
    
    original_shape = a_t.shape
    a_t_flat = a_t.contiguous().flatten()
    a_hat_prev_flat = a_hat_prev.contiguous().flatten()
    n_elements = a_t_flat.numel()
    
    # Pad to even number if needed
    if n_elements % 2 != 0:
        a_t_flat = torch.nn.functional.pad(a_t_flat, (0, 1), value=0)
        a_hat_prev_flat = torch.nn.functional.pad(a_hat_prev_flat, (0, 1), value=0)
        n_elements = a_t_flat.numel()
    
    if scale is None:
        # Fused max(|a - b|) computation
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        block_maxes = torch.empty(grid[0], dtype=torch.float32, device=a_t.device)
        
        _modulated_max_abs_kernel[grid](
            a_t_flat, a_hat_prev_flat,
            block_maxes,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        max_val = block_maxes.max()
        scale = max_val / 7.0  # INT4 symmetric range
        scale = torch.clamp(scale, min=1e-8)
    
    # Allocate outputs
    n_packed = n_elements // 2
    residual_packed = torch.empty(n_packed, dtype=torch.int8, device=a_t.device)
    a_hat_new_flat = torch.empty(n_elements, dtype=a_t.dtype, device=a_t.device)
    
    # Launch kernel
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n_packed, BLOCK_SIZE),)
    
    # Extract scalar value from tensor for Triton
    scale_value = scale.item() if isinstance(scale, torch.Tensor) else float(scale)
    
    _modulated_quantize_int4_kernel[grid](
        a_t_flat, a_hat_prev_flat,
        residual_packed, a_hat_new_flat,
        scale_value,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape (trim padding if added)
    original_numel = torch.tensor(original_shape).prod().item()
    a_hat_new = a_hat_new_flat[:original_numel].view(original_shape)
    
    return residual_packed, a_hat_new, scale, original_shape


def modulated_quantize_first_step_int8(a_T: torch.Tensor) -> tuple:
    """
    MoDiff quantization for the first timestep (t=T).
    
    Implements paper Eq. (ec1): â_T = Q(a_T)
    
    At the first timestep, there's no previous cache, so we just quantize directly.
    
    Args:
        a_T: Activation at first timestep
        
    Returns:
        a_T_int: Quantized activation (INT8)
        a_hat_T: Dequantized activation (becomes the cache for next step)
        scale: Quantization scale
    """
    scale, _ = compute_dynamic_scale_int8(a_T, symmetric=True)
    
    # Quantize
    a_T_scaled = a_T / scale
    a_T_int = torch.round(a_T_scaled).clamp(-128, 127).to(torch.int8)
    
    # Dequantize to get â_T (this will be cached for error compensation)
    a_hat_T = a_T_int.float() * scale
    
    return a_T_int, a_hat_T, scale


def modulated_quantize_first_step_int4(a_T: torch.Tensor) -> tuple:
    """
    MoDiff quantization for the first timestep (t=T) with INT4.
    
    Implements paper Eq. (ec1): â_T = Q(a_T)
    """
    original_shape = a_T.shape
    a_T_flat = a_T.flatten()
    n_elements = a_T_flat.numel()
    
    # Pad if needed
    if n_elements % 2 != 0:
        a_T_flat = torch.nn.functional.pad(a_T_flat, (0, 1), value=0)
        n_elements = a_T_flat.numel()
    
    scale, _ = compute_dynamic_scale_int4(a_T, symmetric=True)
    
    # Quantize to [-8, 7]
    a_T_scaled = a_T_flat / scale
    a_T_int = torch.round(a_T_scaled).clamp(-8, 7)
    
    # Pack
    a_T_uint = (a_T_int + 8).to(torch.uint8)
    a_T_lo = a_T_uint[0::2]
    a_T_hi = a_T_uint[1::2]
    a_T_packed = (a_T_lo & 0xF) | ((a_T_hi & 0xF) << 4)
    
    # Dequantize to get â_T
    a_hat_T_flat = a_T_int * scale
    original_numel = torch.tensor(original_shape).prod().item()
    a_hat_T = a_hat_T_flat[:original_numel].view(original_shape)
    
    return a_T_packed.to(torch.int8), a_hat_T, scale, original_shape
