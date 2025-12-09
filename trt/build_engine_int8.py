#!/usr/bin/env python3
"""
Build TensorRT INT8 Engine for MoDiff - Native INT8 Support

This script builds a TensorRT engine with native INT8 quantization using
the MoDiff paper's MSE-based scale calibration methodology.

Unlike INT4 which requires a proxy, TensorRT has native INT8 support,
providing better accuracy and more reliable performance.

Usage:
    python build_engine_int8.py --onnx modiff_unet.onnx --calib-dir calibration/ --engine modiff_int8.plan

Paper: "Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np

# Suppress TensorRT verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("ERROR: TensorRT not found. Please install TensorRT.")
    sys.exit(1)

from int8_calibrator import (
    MoDiffINT8Calibrator,
    INT8ScaleComputer,
    inject_int8_scales_to_network,
    create_int8_calibration_cache,
)


def set_all_tensor_dynamic_ranges(network, default_dynamic_range: float = 6.0):
    """
    Set dynamic range for ALL tensors in the network.
    
    TensorRT INT8 requires dynamic ranges for every tensor. For tensors without
    explicit calibration data, we set a default dynamic range. This ensures
    the engine builds without "Missing scale and zero-point" warnings.
    
    The default value of 6.0 is chosen because:
    - Most activations in diffusion models are in [-6, 6] range after normalization
    - This is conservative enough to avoid clipping
    - Matches typical ReLU6 / tanh output ranges
    
    Args:
        network: TensorRT network
        default_dynamic_range: Default range for uncalibrated tensors
        
    Returns:
        Number of tensors with dynamic range set
    """
    set_count = 0
    
    # Set dynamic range for all layer outputs (intermediate tensors)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        
        # Set for all outputs
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            if tensor is not None:
                try:
                    # Check if already has a dynamic range set
                    # If not, set the default
                    tensor.set_dynamic_range(-default_dynamic_range, default_dynamic_range)
                    set_count += 1
                except Exception:
                    pass  # Some tensors don't support dynamic range
        
        # Set for all inputs (that aren't network inputs)
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            if tensor is not None and not tensor.is_network_input:
                try:
                    tensor.set_dynamic_range(-default_dynamic_range, default_dynamic_range)
                    set_count += 1
                except Exception:
                    pass
    
    # Set for network inputs
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if tensor is not None:
            try:
                tensor.set_dynamic_range(-default_dynamic_range, default_dynamic_range)
                set_count += 1
            except Exception:
                pass
    
    # Set for network outputs
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        if tensor is not None:
            try:
                tensor.set_dynamic_range(-default_dynamic_range, default_dynamic_range)
                set_count += 1
            except Exception:
                pass
    
    return set_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce TensorRT logging
trt_logger = logging.getLogger('tensorrt')
trt_logger.setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TensorRT INT8 engine for MoDiff UNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx",
        default="modiff_unet_cifar10.onnx",
        help="Path to the ONNX file to convert",
    )
    parser.add_argument(
        "--calib-dir",
        default="../calibration",
        help="Directory containing calibration .npz files",
    )
    parser.add_argument(
        "--scales-file",
        default=None,
        help="Pre-computed INT8 scales JSON file (optional)",
    )
    parser.add_argument(
        "--engine",
        default="modiff_unet_int8.plan",
        help="Output path for the TensorRT engine",
    )
    parser.add_argument(
        "--workspace-gb",
        type=int,
        default=8,
        help="Workspace size in GB",
    )
    parser.add_argument(
        "--scale-method",
        default="mse",
        choices=["mse", "max", "minmax"],
        help="Scale computation method (mse recommended)",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=True,
        help="Use symmetric quantization",
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Use asymmetric quantization (default is symmetric)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 for non-quantized operations",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Optimization batch size",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for dynamic shapes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose TensorRT logging",
    )
    return parser.parse_args()


def load_or_compute_scales(args: argparse.Namespace) -> Dict:
    """Load pre-computed scales or compute new ones from calibration data."""
    
    if args.scales_file and Path(args.scales_file).exists():
        logger.info(f"Loading pre-computed scales from {args.scales_file}")
        with open(args.scales_file) as f:
            return json.load(f)
    
    # Compute scales from calibration data
    calib_dir = Path(args.calib_dir)
    if not calib_dir.exists():
        logger.warning(f"Calibration directory not found: {calib_dir}")
        return {}
    
    cache_path = calib_dir / "int8_scales.json"
    
    logger.info("Computing INT8 scales from calibration data...")
    scales = create_int8_calibration_cache(
        calib_dir=calib_dir,
        output_path=cache_path,
        symmetric=not args.asymmetric,
        scale_method=args.scale_method,
    )
    
    return scales


def build_int8_engine(args: argparse.Namespace) -> None:
    """Build TensorRT engine with native INT8 quantization."""
    
    onnx_path = Path(args.onnx).resolve()
    engine_path = Path(args.engine).resolve()
    calib_dir = Path(args.calib_dir).resolve()
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    # TensorRT setup
    trt_log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
    trt_logger = trt.Logger(trt_log_level)
    
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    
    # Parse ONNX
    logger.info(f"Parsing ONNX from {onnx_path}")
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            logger.error("ONNX parser errors:")
            for i in range(parser.num_errors):
                logger.error(f"  {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")
    
    logger.info(f"Parsed ONNX with {network.num_inputs} inputs, {network.num_layers} layers")
    
    # Set dynamic ranges for ALL tensors BEFORE configuring the builder
    # This prevents "Missing scale and zero-point" warnings
    logger.info("Setting INT8 dynamic ranges for all network tensors...")
    default_dr = 6.0  # Conservative default for normalized activations
    num_set = set_all_tensor_dynamic_ranges(network, default_dynamic_range=default_dr)
    logger.info(f"Set default dynamic range ({default_dr}) for {num_set} tensors")
    
    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
    
    # Enable native INT8 mode
    config.set_flag(trt.BuilderFlag.INT8)
    logger.info("Native INT8 mode enabled")
    
    # Enable FP16 for non-quantized ops
    use_fp16 = args.fp16 and not args.no_fp16
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled for non-quantized operations")
    
    # Setup calibrator
    if calib_dir.exists():
        logger.info(f"Setting up INT8 calibrator with {args.scale_method} scale method")
        calibrator = MoDiffINT8Calibrator(
            calib_dir=calib_dir,
            cache_path=calib_dir / "modiff_int8.cache",
            symmetric=not args.asymmetric,
            scale_method=args.scale_method,
        )
        config.int8_calibrator = calibrator
    else:
        logger.warning("No calibration data - using default calibration")
    
    # Load/compute scales and refine dynamic ranges for specific layers
    scales = load_or_compute_scales(args)
    if scales:
        logger.info("Refining INT8 scales for calibrated layers...")
        if 'latent' in scales:
            latent_scale = scales['latent'].get('scale', 1.0)
            dynamic_range = scales['latent'].get('dynamic_range', 127.0)
            logger.info(f"Input latent scale: {latent_scale:.6f}, dynamic_range: {dynamic_range:.6f}")
            
            # Set the refined dynamic range for the latent input
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                if inp.name == 'latent':
                    inp.set_dynamic_range(-dynamic_range, dynamic_range)
                    logger.info(f"Applied calibrated dynamic range to 'latent' input")
        
        # Inject per-layer scales if available
        if len(scales) > 3:  # More than just latent, timesteps, metadata
            num_injected = inject_int8_scales_to_network(network, scales)
            logger.info(f"Injected calibrated scales for {num_injected} layers")
    
    # Setup optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # CIFAR10 model: latent is (batch, 3, 32, 32), timesteps is (batch,)
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        
        logger.info(f"Input '{input_name}': shape={input_shape}")
        
        if input_name == "latent":
            # Dynamic batch, fixed spatial dims
            min_shape = (1, 3, 32, 32)
            opt_shape = (args.batch_size, 3, 32, 32)
            max_shape = (args.max_batch_size, 3, 32, 32)
        elif input_name == "timesteps":
            min_shape = (1,)
            opt_shape = (args.batch_size,)
            max_shape = (args.max_batch_size,)
        elif input_name == "context":
            # Optional context for conditional models
            min_shape = (1, 1, 512)
            opt_shape = (args.batch_size, 1, 512)
            max_shape = (args.max_batch_size, 1, 512)
        else:
            # Default: assume batch dimension is dynamic
            if len(input_shape) > 0:
                min_shape = tuple([1] + list(input_shape[1:]))
                opt_shape = tuple([args.batch_size] + list(input_shape[1:]))
                max_shape = tuple([args.max_batch_size] + list(input_shape[1:]))
            else:
                continue
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        logger.info(f"  Profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
    
    config.add_optimization_profile(profile)
    
    # Build engine
    logger.info("Building TensorRT engine (this may take several minutes)...")
    logger.info(f"  Quantization: Native INT8")
    logger.info(f"  Scale method: {args.scale_method}")
    logger.info(f"  FP16: {'enabled' if use_fp16 else 'disabled'}")
    logger.info(f"  Workspace: {args.workspace_gb} GB")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine - TRT 10.x uses IHostMemory object
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle TRT 10.x IHostMemory object
    if hasattr(serialized_engine, 'tobytes'):
        # TRT 10.x: IHostMemory has tobytes()
        engine_bytes = serialized_engine.tobytes()
    elif hasattr(serialized_engine, '__bytes__'):
        engine_bytes = bytes(serialized_engine)
    else:
        # Fallback: try direct conversion
        engine_bytes = memoryview(serialized_engine).tobytes()
    
    with engine_path.open("wb") as f:
        f.write(engine_bytes)
    
    engine_size_mb = len(engine_bytes) / (1024 * 1024)
    logger.info(f"✓ Saved INT8 engine to {engine_path} ({engine_size_mb:.1f} MB)")
    
    # Save metadata
    metadata = {
        'onnx_path': str(onnx_path),
        'n_bits': 8,
        'scale_method': args.scale_method,
        'symmetric': not args.asymmetric,
        'fp16': use_fp16,
        'engine_size_mb': engine_size_mb,
    }
    
    metadata_path = engine_path.with_suffix('.int8_meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MoDiff INT8 TensorRT Engine Builder")
    print("Following paper: Modulated Diffusion (ICML 2025)")
    print("Native TensorRT INT8 Support")
    print("=" * 60)
    
    try:
        build_int8_engine(args)
        print("\n✓ Engine build complete!")
    except Exception as e:
        logger.error(f"Engine build failed: {e}")
        raise


if __name__ == "__main__":
    main()
