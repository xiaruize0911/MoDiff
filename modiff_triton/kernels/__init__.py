# MoDiff Triton Kernels
from .quantize import (
    quantize_symmetric_int8,
    quantize_symmetric_int4,
    quantize_asymmetric_int8,
    quantize_asymmetric_int4,
    dequantize_int8,
    dequantize_int4,
    compute_dynamic_scale_int8,
    compute_dynamic_scale_int4,
)

from .modulated_quantize import (
    modulated_quantize_int8,
    modulated_quantize_int4,
    modulated_quantize_first_step_int8,
    modulated_quantize_first_step_int4,
)

from .gemm_w8a8 import (
    gemm_w8a8,
    gemm_w8a8_accum,
)

from .gemm_w4a4 import (
    gemm_w4a4,
    gemm_w4a4_accum,
    pack_int4_weight,
)

from .conv_w8a8_fused import (
    conv2d_w8a8_3x3_fused,
    conv2d_w8a8_3x3_standard,
)

from .awq_w8a8 import (
    awq_fused_quant_gemm_w8a8,
    awq_gemm_w8a8,
    is_awq_available,
    quantize_awq_per_token,
)

__all__ = [
    # Quantization
    "quantize_symmetric_int8",
    "quantize_symmetric_int4",
    "quantize_asymmetric_int8",
    "quantize_asymmetric_int4",
    "dequantize_int8",
    "dequantize_int4",
    "compute_dynamic_scale_int8",
    "compute_dynamic_scale_int4",
    # Modulated quantization
    "modulated_quantize_int8",
    "modulated_quantize_int4",
    "modulated_quantize_first_step_int8",
    "modulated_quantize_first_step_int4",
    # GEMM
    "gemm_w8a8",
    "gemm_w8a8_accum",
    "gemm_w4a4",
    "gemm_w4a4_accum",
    "pack_int4_weight",
    # Fused Conv2d
    "conv2d_w8a8_3x3_fused",
    "conv2d_w8a8_3x3_standard",
    # Optional AWQ baseline
    "awq_fused_quant_gemm_w8a8",
    "awq_gemm_w8a8",
    "is_awq_available",
    "quantize_awq_per_token",
]
