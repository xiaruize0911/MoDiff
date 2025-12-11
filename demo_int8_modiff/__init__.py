"""
MoDiff INT8 Demo - Custom INT8 Kernels with Error-Compensated Modulation

This package implements the Q-Diff + MoDiff methodology from the paper:
"Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization"

Key Features:
1. True INT8 computation using PyTorch native quantized operations
2. MoDiff error-compensated modulation (quantize residuals, not raw activations)
3. 2-2.5x speedup with paper-level FID (~4.1 on CIFAR-10)

Usage:
    from demo_int8_modiff import (
        NativeINT8Quantizer,
        QuantLayerMoDiff,
        QuantModelMoDiff,
        DDIMSamplerMoDiff,
        calibrate_model_qdiff,
        compute_fid,
    )
"""

from demo_int8_modiff.quant_int8_native import (
    NativeINT8Quantizer,
    mse_scale_search,
    quantize_tensor_int8,
    dequantize_tensor_int8,
)

from demo_int8_modiff.quant_layer_modiff import (
    QuantLayerMoDiff,
    StraightThrough,
)

from demo_int8_modiff.quant_model_modiff import (
    QuantModelMoDiff,
)

from demo_int8_modiff.ddim_sampler import (
    DDIMSamplerMoDiff,
    compute_alpha,
)

from demo_int8_modiff.calibration import (
    calibrate_model_qdiff,
    calibrate_model_modiff,
    generate_calibration_data,
    save_calibrated_model,
    load_calibrated_model,
)

from demo_int8_modiff.fid_evaluation import (
    compute_fid,
    compute_reference_stats,
    get_inception_model,
)

from demo_int8_modiff.utils import (
    set_seed,
    load_config,
    dict2namespace,
    load_model_checkpoint,
    images_to_uint8,
    save_samples_npz,
)

__version__ = "1.0.0"
__all__ = [
    # Core quantization
    "NativeINT8Quantizer",
    "QuantLayerMoDiff", 
    "QuantModelMoDiff",
    # Sampling
    "DDIMSamplerMoDiff",
    # Calibration
    "calibrate_model_qdiff",
    "calibrate_model_modiff",
    "generate_calibration_data",
    "save_calibrated_model",
    "load_calibrated_model",
    # Evaluation
    "compute_fid",
    "compute_reference_stats",
    # Utilities
    "set_seed",
    "load_config",
    "dict2namespace",
    "load_model_checkpoint",
    "images_to_uint8",
    "save_samples_npz",
]
