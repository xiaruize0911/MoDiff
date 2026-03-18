# MoDiff Triton Kernels
# Implementation of "Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization"

from .kernels import (
    quantize_symmetric_int8,
    quantize_symmetric_int4,
    dequantize_int8,
    dequantize_int4,
    compute_dynamic_scale_int8,
    compute_dynamic_scale_int4,
    modulated_quantize_int8,
    modulated_quantize_int4,
)

from .nn import (
    MoDiffConfig,
    W8A8MoDiffLinear,
    W4A4MoDiffLinear,
    W8A8MoDiffConv2d,
    W4A4MoDiffConv2d,
)

__all__ = [
    # Kernels
    "quantize_symmetric_int8",
    "quantize_symmetric_int4",
    "dequantize_int8",
    "dequantize_int4",
    "compute_dynamic_scale_int8",
    "compute_dynamic_scale_int4",
    "modulated_quantize_int8",
    "modulated_quantize_int4",
    # Modules
    "MoDiffConfig",
    "W8A8MoDiffLinear",
    "W4A4MoDiffLinear",
    "W8A8MoDiffConv2d",
    "W4A4MoDiffConv2d",
]
