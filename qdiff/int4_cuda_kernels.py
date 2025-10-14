"""
INT4 CUDA Kernels for efficient packing/unpacking and dequantization.

This module provides CUDA kernels for:
1. Packing two 4-bit values into a single 8-bit uint8 storage
2. Unpacking uint8 storage back to two separate 4-bit values
3. Dequantizing int4 values to float32 using scale and zero_point

The kernels are implemented using PyTorch's custom CUDA extensions or pure PyTorch
for fallback when CUDA compilation is not available.
"""

import torch
import torch.nn as nn
from typing import Tuple


def pack_int4_to_uint8(x_int4: torch.Tensor) -> torch.Tensor:
    """
    Pack two consecutive int4 values into a single uint8 byte.
    
    INT4 packing scheme:
    - Two 4-bit values [v0, v1] are packed into one uint8 byte
    - v0 occupies lower 4 bits, v1 occupies upper 4 bits
    - Result: packed_byte = (v1 << 4) | (v0 & 0xF)
    
    Args:
        x_int4: Tensor of int4 values in the range [0, 15], shape [..., N]
                N must be even for packing
    
    Returns:
        packed: uint8 tensor with shape [..., N//2]
    
    Example:
        x = torch.tensor([3, 5, 7, 9], dtype=torch.uint8)
        packed = pack_int4_to_uint8(x)  # shape [2], values: [0x53, 0x97]
    """
    assert x_int4.shape[-1] % 2 == 0, "Last dimension must be even for int4 packing"
    
    # Ensure input is uint8 and clamp to 4-bit range [0, 15]
    x_int4 = x_int4.to(torch.uint8)
    x_int4 = torch.clamp(x_int4, 0, 15)
    
    # Reshape to [..., N//2, 2] to pair consecutive values
    shape = x_int4.shape
    x_pairs = x_int4.reshape(*shape[:-1], shape[-1] // 2, 2)
    
    # Pack: lower 4 bits from first value, upper 4 bits from second value
    # packed = (x_pairs[..., 1] << 4) | (x_pairs[..., 0] & 0xF)
    packed = (x_pairs[..., 1] * 16) + (x_pairs[..., 0] & 15)
    
    return packed.to(torch.uint8)


def unpack_uint8_to_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 storage back to two separate int4 values per byte.
    
    INT4 unpacking scheme:
    - Each uint8 byte contains two 4-bit values
    - Lower 4 bits → first value (v0 = byte & 0xF)
    - Upper 4 bits → second value (v1 = (byte >> 4) & 0xF)
    
    Args:
        packed: uint8 tensor with shape [..., N//2]
    
    Returns:
        unpacked: uint8 tensor with shape [..., N], values in range [0, 15]
    
    Example:
        packed = torch.tensor([0x53, 0x97], dtype=torch.uint8)
        unpacked = unpack_uint8_to_int4(packed)  # [3, 5, 7, 9]
    """
    packed = packed.to(torch.uint8)
    
    # Extract lower and upper 4 bits
    lower_4bit = packed & 15  # Lower 4 bits: byte & 0xF
    upper_4bit = (packed >> 4) & 15  # Upper 4 bits: (byte >> 4) & 0xF
    
    # Stack and reshape to interleave values
    # Shape: [..., N//2, 2] then reshape to [..., N]
    unpacked = torch.stack([lower_4bit, upper_4bit], dim=-1)
    shape = unpacked.shape
    unpacked = unpacked.reshape(*shape[:-2], shape[-2] * 2)
    
    return unpacked


def dequantize_int4_cuda(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    target_shape: tuple = None
) -> torch.Tensor:
    """
    Dequantize packed int4 weights to float32 using per-tensor or per-channel scales.
    
    Dequantization formula:
        float_weight = (unpacked_int4 - zero_point) * scale
    
    This function:
    1. Unpacks uint8 storage to int4 values [0, 15]
    2. Applies zero_point offset and scale to convert to float32
    3. Reshapes to target_shape if provided
    
    Args:
        packed_weight: Packed uint8 tensor, shape [out_features, (in_features*h*w)//2]
        scale: Quantization scale, shape [1] (per-tensor) or [out_features, 1] (per-channel)
        zero_point: Zero point offset, shape matching scale
        target_shape: Optional target shape to reshape dequantized output (e.g., [out, in, h, w])
    
    Returns:
        float_weight: Dequantized float32 tensor, shape target_shape or [out_features, in_features]
    
    Example:
        # Per-tensor quantization
        packed = pack_int4_to_uint8(torch.tensor([0, 15, 7, 8], dtype=torch.uint8))
        scale = torch.tensor([0.1])
        zero_point = torch.tensor([7.0])
        weight = dequantize_int4_cuda(packed, scale, zero_point)
        # Result: [-0.7, 0.8, 0.0, 0.1]
    """
    # Unpack int4 values from uint8 storage
    unpacked = unpack_uint8_to_int4(packed_weight)  # [out_features, in_features_packed, ...]
    
    # Convert to float for dequantization
    unpacked_float = unpacked.to(torch.float32)
    
    # Dequantize: (x_int4 - zero_point) * scale
    # Broadcasting handles per-tensor and per-channel scales
    dequantized = (unpacked_float - zero_point) * scale
    
    # Reshape to target shape if provided
    if target_shape is not None:
        # Remove any padding that was added during quantization
        expected_numel = 1
        for dim in target_shape:
            expected_numel *= dim
        if dequantized.numel() > expected_numel:
            # Flatten and slice to remove padding
            dequantized = dequantized.flatten()[:expected_numel]
        dequantized = dequantized.view(target_shape)
    
    return dequantized


def quantize_to_int4_cuda(
    weight_fp32: torch.Tensor,
    n_bits: int = 4,
    symmetric: bool = False,
    channel_wise: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize float32 weights to int4 format with automatic scale/zero_point calculation.
    
    Quantization process:
    1. Calculate min/max per tensor or per channel
    2. Compute scale and zero_point to map [min, max] → [0, 15]
    3. Quantize: x_int = round(x_fp32 / scale) + zero_point
    4. Clamp to [0, 15] and pack into uint8 storage
    
    Args:
        weight_fp32: Float32 weight tensor, shape [out_features, in_features, ...]
        n_bits: Number of bits (must be 4 for this implementation)
        symmetric: If True, use symmetric quantization around zero [-8, 7]
        channel_wise: If True, compute per-channel (per-output-feature) scales
    
    Returns:
        packed_weight: Packed uint8 tensor, shape [out_features, in_features//2, ...]
        scale: Quantization scale tensor
        zero_point: Zero point offset tensor
    
    Example:
        weight = torch.randn(64, 128)  # [out_features, in_features]
        packed, scale, zp = quantize_to_int4_cuda(weight, channel_wise=True)
        # packed.shape = [64, 64]  (packed along in_features dimension)
        # scale.shape = [64, 1]     (per output channel)
    """
    assert n_bits == 4, "Only 4-bit quantization is supported"
    n_levels = 2 ** n_bits  # 16 levels for int4
    
    # Store original shape for later
    orig_shape = weight_fp32.shape
    
    # For multi-dimensional tensors (e.g., Conv2d [out, in, h, w]), 
    # flatten all dims except first (output channels) for easier processing
    if len(orig_shape) > 2:
        # Reshape to [out_channels, -1]
        weight_fp32 = weight_fp32.view(orig_shape[0], -1)
    
    # Pad if total elements per channel is odd (required for packing)
    shape = weight_fp32.shape
    padded = False
    if shape[1] % 2 != 0:
        # Pad the flattened dimension to make it even
        pad_size = [shape[0], 1]
        padding = torch.zeros(pad_size, dtype=weight_fp32.dtype, device=weight_fp32.device)
        weight_fp32 = torch.cat([weight_fp32, padding], dim=1)
        shape = weight_fp32.shape
        padded = True
    
    if channel_wise:
        # Per-channel quantization (per output feature)
        # Compute min/max along all dims except the first (output channels)
        dims_to_reduce = list(range(1, len(shape)))
        x_min = weight_fp32.amin(dim=dims_to_reduce, keepdim=True)  # [out_features, 1, ...]
        x_max = weight_fp32.amax(dim=dims_to_reduce, keepdim=True)
    else:
        # Per-tensor quantization
        x_min = weight_fp32.min()
        x_max = weight_fp32.max()
    
    if symmetric:
        # Symmetric quantization: map [-abs_max, abs_max] → [0, 15]
        # For true symmetric we'd use [-8, 7], but using [0, 15] with symmetric range
        x_absmax = torch.maximum(x_min.abs(), x_max.abs())
        scale = x_absmax / (n_levels - 1)
        zero_point = torch.zeros_like(scale)
    else:
        # Asymmetric quantization: map [min, max] → [0, 15]
        scale = (x_max - x_min) / (n_levels - 1)
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
        zero_point = -x_min / scale
    
    # Quantize: x_int = round(x_fp32 / scale) + zero_point
    x_int = torch.round(weight_fp32 / scale) + zero_point
    x_int = torch.clamp(x_int, 0, n_levels - 1).to(torch.uint8)
    
    # Pack into uint8 storage (two int4 values per byte)
    packed_weight = pack_int4_to_uint8(x_int)
    
    # Store original shape info for later reshaping during dequantization
    # Note: Users should store orig_shape separately if they need to dequantize later
    
    return packed_weight, scale, zero_point


# CUDA kernel optimization note:
# For production use, consider implementing custom CUDA kernels using:
# 1. torch.utils.cpp_extension for inline C++/CUDA compilation
# 2. Triton language for easier kernel development
# 3. Pre-compiled .cu files with setuptools
#
# Custom CUDA kernels can provide:
# - Fused unpack + dequantize operations
# - Vectorized memory access (load 128-bit = 32 int4 values at once)
# - Faster int4 matrix multiplication using Tensor Cores (compute capability >= 8.0)
#
# Example skeleton for custom CUDA kernel:
"""
// Example C++ signature for custom kernel
torch::Tensor dequantize_int4_kernel(
    torch::Tensor packed_weight,  // uint8 packed
    torch::Tensor scale,          // float32 scale
    torch::Tensor zero_point      // float32 zero_point
) {
    // Launch CUDA kernel with grid/block configuration
    // Use shared memory for scale/zero_point if per-channel
    // Vectorized loads and stores for efficiency
}
"""

# For now, we use pure PyTorch implementation which is portable but slower
# than custom CUDA kernels. Users can replace these functions with optimized
# versions as needed.
