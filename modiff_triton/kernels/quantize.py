"""
Quantization Kernels for MoDiff

Following the paper's Theorem 1 (Quantization Error):
    Given x ∈ R^d and bandwidth b, the max-min dynamic quantizer:
        s = (max(x) - min(x)) / (2^b - 1)
        z = floor(-min(x) / s)
        x_int = clamp(floor(x/s) + z, 0, 2^b - 1)
        Q(x) = s * (x_int - z)
    
    Error bound: ||x - Q(x)||_2^2 <= s^2 * d

For MoDiff, we apply quantization to the residual (a_t - â_{t+1}), which has
a much smaller range, resulting in lower quantization error.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# INT8 Quantization Kernels
# ============================================================================

@triton.jit
def _quantize_symmetric_int8_kernel(
    x_ptr,          # Input tensor
    out_ptr,        # Output quantized tensor (int8)
    scale_ptr,      # Output scale
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Symmetric INT8 quantization: x_int = round(x / scale), scale = max(|x|) / 127
    
    Following paper Theorem 1 with symmetric quantization.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute max absolute value (for symmetric quantization)
    x_abs = tl.abs(x)
    x_max = tl.max(x_abs)
    
    # Compute scale: scale = max(|x|) / 127
    scale = x_max / 127.0
    scale = tl.where(scale < 1e-8, 1e-8, scale)  # Avoid division by zero
    
    # Quantize: x_int = clamp(round(x / scale), -128, 127)
    x_scaled = x / scale
    x_int = tl.floor(x_scaled + 0.5)
    x_int = tl.maximum(tl.minimum(x_int, 127.0), -128.0)
    
    # Store results
    tl.store(out_ptr + offsets, x_int.to(tl.int8), mask=mask)
    
    # Store scale (only first thread)
    if pid == 0:
        tl.store(scale_ptr, scale)


@triton.jit
def _quantize_symmetric_int8_global_scale_kernel(
    x_ptr,          # Input tensor
    out_ptr,        # Output quantized tensor (int8)
    scale,          # Global scale (scalar)
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Symmetric INT8 quantization with pre-computed global scale.
    x_int = clamp(round(x / scale), -128, 127)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Quantize
    x_scaled = x / scale
    x_int = tl.floor(x_scaled + 0.5)
    x_int = tl.maximum(tl.minimum(x_int, 127.0), -128.0)
    
    # Store results
    tl.store(out_ptr + offsets, x_int.to(tl.int8), mask=mask)


@triton.jit
def _quantize_asymmetric_int8_kernel(
    x_ptr,          # Input tensor
    out_ptr,        # Output quantized tensor (int8)
    scale_ptr,      # Output scale
    zp_ptr,         # Output zero point
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Asymmetric INT8 quantization following paper Theorem 1:
        s = (max(x) - min(x)) / 255
        z = round(-min(x) / s)
        x_int = clamp(round(x/s) + z, 0, 255)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute min and max
    x_min = tl.min(x)
    x_max = tl.max(x)
    
    # Compute scale and zero point
    scale = (x_max - x_min) / 255.0
    scale = tl.where(scale < 1e-8, 1e-8, scale)
    zero_point = tl.floor(-x_min / scale + 0.5)
    
    # Quantize
    x_scaled = x / scale + zero_point
    x_int = tl.floor(x_scaled + 0.5)
    x_int = tl.maximum(tl.minimum(x_int, 255.0), 0.0)
    
    # Convert to signed int8 for storage (subtract 128)
    x_int_signed = x_int - 128.0
    
    # Store results
    tl.store(out_ptr + offsets, x_int_signed.to(tl.int8), mask=mask)
    
    if pid == 0:
        tl.store(scale_ptr, scale)
        tl.store(zp_ptr, zero_point.to(tl.int32))


@triton.jit
def _dequantize_int8_kernel(
    x_int_ptr,      # Input quantized tensor (int8)
    scale_ptr,      # Scale
    out_ptr,        # Output dequantized tensor
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize INT8 to FP16/FP32"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load scale
    scale = tl.load(scale_ptr)
    
    # Load quantized values
    x_int = tl.load(x_int_ptr + offsets, mask=mask, other=0).to(tl.float32)
    
    # Dequantize
    x = x_int * scale
    
    tl.store(out_ptr + offsets, x, mask=mask)


# ============================================================================
# INT4 Quantization Kernels  
# ============================================================================

@triton.jit
def _quantize_symmetric_int4_kernel(
    x_ptr,          # Input tensor
    out_ptr,        # Output quantized tensor (packed int8, 2 int4 per int8)
    scale_ptr,      # Output scale
    n_elements,     # Total number of elements (must be even)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Symmetric INT4 quantization: x_int = round(x / scale), scale = max(|x|) / 7
    
    Two INT4 values packed into one INT8:
        packed = (low & 0xF) | ((high & 0xF) << 4)
    """
    pid = tl.program_id(0)
    # Process 2 elements at a time (for packing)
    block_start = pid * BLOCK_SIZE * 2
    offsets_lo = block_start + tl.arange(0, BLOCK_SIZE) * 2
    offsets_hi = offsets_lo + 1
    mask_lo = offsets_lo < n_elements
    mask_hi = offsets_hi < n_elements
    
    # Load input pairs
    x_lo = tl.load(x_ptr + offsets_lo, mask=mask_lo, other=0.0)
    x_hi = tl.load(x_ptr + offsets_hi, mask=mask_hi, other=0.0)
    
    # Compute max absolute value
    x_abs_lo = tl.abs(x_lo)
    x_abs_hi = tl.abs(x_hi)
    x_max = tl.maximum(tl.max(x_abs_lo), tl.max(x_abs_hi))
    
    # Compute scale: scale = max(|x|) / 7 (INT4 symmetric range: -8 to 7)
    scale = x_max / 7.0
    scale = tl.where(scale < 1e-8, 1e-8, scale)
    
    # Quantize to INT4 range [-8, 7]
    x_lo_int = tl.floor(x_lo / scale + 0.5)
    x_hi_int = tl.floor(x_hi / scale + 0.5)
    x_lo_int = tl.maximum(tl.minimum(x_lo_int, 7.0), -8.0)
    x_hi_int = tl.maximum(tl.minimum(x_hi_int, 7.0), -8.0)
    
    # Pack two INT4 into INT8: low nibble and high nibble
    # Convert to unsigned 4-bit (add 8): range [0, 15]
    x_lo_uint4 = (x_lo_int + 8.0).to(tl.int8)
    x_hi_uint4 = (x_hi_int + 8.0).to(tl.int8)
    
    # Pack: (low & 0xF) | ((high & 0xF) << 4)
    packed = (x_lo_uint4 & 0xF) | ((x_hi_uint4 & 0xF) << 4)
    
    # Store packed values
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < (n_elements // 2)
    tl.store(out_ptr + out_offsets, packed, mask=out_mask)
    
    if pid == 0:
        tl.store(scale_ptr, scale)


@triton.jit
def _dequantize_int4_kernel(
    x_packed_ptr,   # Input packed tensor (int8, 2 int4 per int8)
    scale_ptr,      # Scale
    out_ptr,        # Output dequantized tensor
    n_packed,       # Number of packed elements
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize packed INT4 to FP16/FP32"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_packed
    
    # Load scale
    scale = tl.load(scale_ptr)
    
    # Load packed values
    packed = tl.load(x_packed_ptr + offsets, mask=mask, other=0)
    
    # Unpack: low nibble and high nibble
    x_lo_uint4 = (packed & 0xF).to(tl.float32)
    x_hi_uint4 = ((packed >> 4) & 0xF).to(tl.float32)
    
    # Convert back to signed: subtract 8
    x_lo_int4 = x_lo_uint4 - 8.0
    x_hi_int4 = x_hi_uint4 - 8.0
    
    # Dequantize
    x_lo = x_lo_int4 * scale
    x_hi = x_hi_int4 * scale
    
    # Store unpacked values
    out_offsets_lo = block_start * 2 + tl.arange(0, BLOCK_SIZE) * 2
    out_offsets_hi = out_offsets_lo + 1
    out_mask_lo = out_offsets_lo < (n_packed * 2)
    out_mask_hi = out_offsets_hi < (n_packed * 2)
    
    tl.store(out_ptr + out_offsets_lo, x_lo, mask=out_mask_lo)
    tl.store(out_ptr + out_offsets_hi, x_hi, mask=out_mask_hi)


# ============================================================================
# Dynamic Scale Computation Kernels
# ============================================================================

@triton.jit
def _compute_scale_reduction_kernel(
    x_ptr,
    partial_max_ptr,
    partial_min_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """First stage: compute partial min/max per block"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    block_max = tl.max(x)
    block_min = tl.min(x)
    
    tl.store(partial_max_ptr + pid, block_max)
    tl.store(partial_min_ptr + pid, block_min)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def compute_dynamic_scale_int8(x: torch.Tensor, symmetric: bool = True) -> tuple:
    """
    Compute dynamic quantization scale for INT8.
    
    Args:
        x: Input tensor
        symmetric: If True, use symmetric quantization
        
    Returns:
        scale: Quantization scale
        zero_point: Zero point (0 for symmetric)
    """
    x_flat = x.flatten()
    
    if symmetric:
        x_abs_max = x_flat.abs().max()
        scale = x_abs_max / 127.0
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.tensor(0, dtype=torch.int32, device=x.device)
    else:
        x_min = x_flat.min()
        x_max = x_flat.max()
        scale = (x_max - x_min) / 255.0
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-x_min / scale).to(torch.int32)
    
    return scale, zero_point


def compute_dynamic_scale_int4(x: torch.Tensor, symmetric: bool = True) -> tuple:
    """
    Compute dynamic quantization scale for INT4.
    
    Args:
        x: Input tensor
        symmetric: If True, use symmetric quantization
        
    Returns:
        scale: Quantization scale
        zero_point: Zero point (0 for symmetric)
    """
    x_flat = x.flatten()
    
    if symmetric:
        x_abs_max = x_flat.abs().max()
        scale = x_abs_max / 7.0  # INT4 symmetric: -8 to 7
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.tensor(0, dtype=torch.int32, device=x.device)
    else:
        x_min = x_flat.min()
        x_max = x_flat.max()
        scale = (x_max - x_min) / 15.0  # INT4 unsigned: 0 to 15
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-x_min / scale).to(torch.int32)
    
    return scale, zero_point


def quantize_symmetric_int8(x: torch.Tensor, scale: torch.Tensor = None) -> tuple:
    """
    Symmetric INT8 quantization.
    
    Args:
        x: Input tensor (FP16/FP32)
        scale: Pre-computed scale (if None, compute dynamically)
        
    Returns:
        x_int: Quantized tensor (INT8)
        scale: Quantization scale
    """
    if scale is None:
        scale, _ = compute_dynamic_scale_int8(x, symmetric=True)
    
    # Use Triton kernel for quantization
    x_flat = x.flatten()
    n_elements = x_flat.numel()
    x_int = torch.empty(n_elements, dtype=torch.int8, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    scale_val = scale.item() if isinstance(scale, torch.Tensor) else float(scale)
    
    _quantize_symmetric_int8_global_scale_kernel[grid](
        x_flat, x_int,
        scale_val,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x_int.view(x.shape), scale


def quantize_symmetric_int4(x: torch.Tensor, scale: torch.Tensor = None) -> tuple:
    """
    Symmetric INT4 quantization with packing.
    
    Args:
        x: Input tensor (FP16/FP32), must have even number of elements
        scale: Pre-computed scale (if None, compute dynamically)
        
    Returns:
        x_packed: Packed tensor (INT8, 2 INT4 per INT8)
        scale: Quantization scale
    """
    original_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()
    
    # Ensure even number of elements
    if n_elements % 2 != 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, 1), value=0)
        n_elements = x_flat.numel()
    
    if scale is None:
        scale, _ = compute_dynamic_scale_int4(x, symmetric=True)
    
    # Quantize to INT4 range [-8, 7]
    x_scaled = x_flat / scale
    x_int = torch.round(x_scaled).clamp(-8, 7)
    
    # Convert to unsigned [0, 15] for packing
    x_uint = (x_int + 8).to(torch.uint8)
    
    # Pack pairs: low nibble | (high nibble << 4)
    x_lo = x_uint[0::2]
    x_hi = x_uint[1::2]
    x_packed = (x_lo & 0xF) | ((x_hi & 0xF) << 4)
    
    return x_packed.to(torch.int8), scale, original_shape


def quantize_asymmetric_int8(x: torch.Tensor) -> tuple:
    """
    Asymmetric INT8 quantization following paper Theorem 1.
    
    Returns:
        x_int: Quantized tensor (INT8)
        scale: Quantization scale
        zero_point: Zero point
    """
    scale, zero_point = compute_dynamic_scale_int8(x, symmetric=False)
    
    x_scaled = x / scale + zero_point.float()
    x_int = torch.round(x_scaled).clamp(0, 255)
    # Store as signed int8 (subtract 128)
    x_int = (x_int - 128).to(torch.int8)
    
    return x_int, scale, zero_point


def quantize_asymmetric_int4(x: torch.Tensor) -> tuple:
    """
    Asymmetric INT4 quantization with packing.
    
    Returns:
        x_packed: Packed tensor (INT8, 2 INT4 per INT8)
        scale: Quantization scale
        zero_point: Zero point
    """
    original_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()
    
    if n_elements % 2 != 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, 1), value=0)
        n_elements = x_flat.numel()
    
    scale, zero_point = compute_dynamic_scale_int4(x, symmetric=False)
    
    # Quantize to [0, 15]
    x_scaled = x_flat / scale + zero_point.float()
    x_uint = torch.round(x_scaled).clamp(0, 15).to(torch.uint8)
    
    # Pack pairs
    x_lo = x_uint[0::2]
    x_hi = x_uint[1::2]
    x_packed = (x_lo & 0xF) | ((x_hi & 0xF) << 4)
    
    return x_packed.to(torch.int8), scale, zero_point, original_shape


def dequantize_int8(x_int: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor = None) -> torch.Tensor:
    """
    Dequantize INT8 tensor.
    
    Args:
        x_int: Quantized tensor (INT8)
        scale: Quantization scale
        zero_point: Zero point (optional, for asymmetric)
        
    Returns:
        x: Dequantized tensor (FP32)
    """
    x = x_int.float() * scale
    if zero_point is not None and zero_point != 0:
        # For asymmetric, x_int was stored as (q - 128), so add back
        x = (x_int.float() + 128 - zero_point.float()) * scale
    return x


def dequantize_int4(x_packed: torch.Tensor, scale: torch.Tensor, original_shape: tuple,
                    zero_point: torch.Tensor = None) -> torch.Tensor:
    """
    Dequantize packed INT4 tensor.
    
    Args:
        x_packed: Packed tensor (INT8, 2 INT4 per INT8)
        scale: Quantization scale
        original_shape: Original tensor shape
        zero_point: Zero point (optional, for asymmetric)
        
    Returns:
        x: Dequantized tensor (FP32)
    """
    # Unpack
    x_lo = (x_packed & 0xF).to(torch.float32)
    x_hi = ((x_packed >> 4) & 0xF).to(torch.float32)
    
    # Interleave
    x_flat = torch.zeros(x_packed.numel() * 2, dtype=torch.float32, device=x_packed.device)
    x_flat[0::2] = x_lo
    x_flat[1::2] = x_hi
    
    if zero_point is None or zero_point == 0:
        # Symmetric: stored as unsigned [0, 15], convert to signed [-8, 7]
        x_flat = (x_flat - 8) * scale
    else:
        # Asymmetric
        x_flat = (x_flat - zero_point.float()) * scale
    
    # Reshape to original
    n_original = torch.tensor(original_shape).prod().item()
    x_flat = x_flat[:n_original]
    
    return x_flat.view(original_shape)
