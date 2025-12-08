import argparse
import logging
from pathlib import Path
import sys
import os

import tensorrt as trt

# Suppress TensorRT verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
trt_logger = logging.getLogger('tensorrt')
trt_logger.setLevel(logging.WARNING)

from entropy_calibrator import MoDiffEntropyCalibrator, MoDiffScaleExtractorCalibrator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT INT8 engine for MoDiff UNet")
    parser.add_argument(
        "--onnx",
        default=str(Path.home() / "modiff_trt" / "export" / "modiff_unet_fp32_simplified.onnx"),
        help="Path to the ONNX file to convert",
    )
    parser.add_argument(
        "--calib-dir",
        default=str(Path.home() / "modiff_trt" / "calib"),
        help="Directory containing calibration .npz files",
    )
    parser.add_argument(
        "--scales-dir",
        default=str(Path.home() / "modiff_trt" / "export" / "extracted_scales"),
        help="Directory containing extracted model scales (for MoDiffScaleExtractorCalibrator)",
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
        "--no-int8",
        action="store_true",
        help="Disable INT8 calibration (build FP32 engine)",
    )
    parser.add_argument(
        "--use-entropy",
        action="store_true",
        help="Use entropy-based calibration (default is scale extraction if available)",
    )
    parser.add_argument(
        "--min-shape",
        nargs=3,
        type=int,
        default=(1, 4, 32),
        help="Min spatial shape for latent tensor: batch, channels, height",
    )
    parser.add_argument(
        "--opt-shape",
        nargs=3,
        type=int,
        default=(1, 4, 32),
        help="Opt spatial shape for latent tensor",
    )
    parser.add_argument(
        "--max-shape",
        nargs=3,
        type=int,
        default=(8, 4, 32),
        help="Max spatial shape for latent tensor: batch, channels, height",
    )
    return parser.parse_args()


def inject_scales_to_network(network, scales_file: Path) -> None:
    """
    Inject pre-computed quantization scales into the TensorRT network.
    This overrides the calibrator and ensures we use the exact scales from PyTorch.
    """
    import numpy as np
    if not scales_file.exists():
        return
        
    print(f"[INFO] Loading scales from {scales_file} for injection")
    try:
        data = np.load(scales_file, allow_pickle=True)
        # Convert npz to dict
        scales_dict = {k: data[k].item() for k in data.files}
    except Exception as e:
        print(f"[ERROR] Failed to load scales: {e}")
        return

    # Build a map from module name to scale
    # keys are like "layer_123_module.name"
    module_scales = {}
    for key, val in scales_dict.items():
        # Extract module name: layer_123_name -> name
        parts = key.split('_', 2)
        if len(parts) > 2:
            module_name = parts[2]
            if 'act_scale' in val:
                module_scales[module_name] = float(val['act_scale'])

    print(f"[INFO] Found {len(module_scales)} layers with activation scales")

    # Iterate over network layers
    matched = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        
        # Normalize ONNX name
        onnx_name = layer.name.replace('/', '.')
        if onnx_name.startswith('.'): onnx_name = onnx_name[1:]
        onnx_name = onnx_name.replace('model.diffusion_model.', '')
        
        # Find best matching PyTorch module
        best_match = None
        for mod_name in module_scales:
            # Check if ONNX name ends with the module name (e.g. ...conv1 matches conv1)
            # or if it contains it clearly
            if mod_name in onnx_name:
                if best_match is None or len(mod_name) > len(best_match):
                    best_match = mod_name
        
        if best_match:
            scale = module_scales[best_match]
            # CRITICAL FIX: scale = maxabs/127, so dynamic_range = scale * 127 = maxabs
            # This matches how INT4 does it: dynamic_range = scale * 7.5 (for q_max=7)
            # For INT8, q_max = 127, so we multiply by 127
            dynamic_range = scale * 127.0
            
            # Set dynamic range for INPUTS of this layer (since act_scale is input quantization)
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                # Skip weights/constants
                if not inp.is_network_input and layer.type != trt.LayerType.CONSTANT:
                     # Assuming symmetric INT8
                     inp.set_dynamic_range(-dynamic_range, dynamic_range)
            matched += 1

    print(f"[INFO] Injected scales for {matched} layers (Direct Injection, dynamic_range = scale * 127)")


def build_engine(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()
    calib_dir = Path(args.calib_dir).expanduser().resolve()
    scales_dir = Path(args.scales_dir).expanduser().resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    trt_logger = trt.Logger(trt.Logger.WARNING)  # Reduced from INFO
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info(f"Parsing ONNX from {onnx_path}")
    with onnx_path.open("rb") as handle:
        if not parser.parse(handle.read()):
            logger.error("ONNX parser errors:")
            for idx in range(parser.num_errors):
                logger.error(f"  {parser.get_error(idx)}")
            raise RuntimeError(f"Failed to parse ONNX graph at {onnx_path}")

    logger.info(f"Parsed ONNX graph with {network.num_inputs} inputs")

    config = builder.create_builder_config()
    # TensorRT 10.x uses set_memory_pool_limit instead of max_workspace_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
    
    if not args.no_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        
        # Choose calibrator based on available scales
        scales_file = scales_dir / "model_scales.npz" if scales_dir.is_dir() else None
        
        if scales_file and scales_file.exists() and not args.use_entropy:
            logger.info("✓ Using MoDiffScaleExtractorCalibrator (scales from trained model)")
            config.int8_calibrator = MoDiffScaleExtractorCalibrator(
                calib_data_dir=calib_dir,
                scales_file=scales_file,
                cache_path=calib_dir / "modiff_int8_scales.cache"
            )
            inject_scales_to_network(network, scales_file)
        else:
            if scales_file and not scales_file.exists():
                logger.warning("Extracted scales not found, falling back to entropy calibration")
            logger.info("Using MoDiffEntropyCalibrator (entropy-based calibration)")
            config.int8_calibrator = MoDiffEntropyCalibrator(calib_dir)
        
        logger.info("INT8 calibration enabled")
    else:
        logger.info("Building FP32 engine (INT8 disabled)")

    profile = builder.create_optimization_profile()
    logger.info(f"Setting optimization profile for {network.num_inputs} inputs")
    
    # For each input, replace dynamic dimensions with concrete values
    # TensorRT requires min/opt/max to match for static inputs
    for idx in range(network.num_inputs):
        tensor = network.get_input(idx)
        shape = list(tensor.shape)
        
        # Replace dynamic batch dimension with fixed size
        if shape[0] == -1:
            shape[0] = 1  # Use batch size 1 for optimization
        
        static_shape = tuple(shape)
        profile.set_shape(tensor.name, static_shape, static_shape, static_shape)
    config.add_optimization_profile(profile)

    logger.info("Building engine...")
    try:
        # TensorRT 10.x uses build_serialized_network instead of build_engine
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        logger.error(f"Engine build failed: {e}")
        raise
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine (returned None)")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with engine_path.open("wb") as handle:
        handle.write(serialized_engine)
    logger.info(f"✓ Saved engine to {engine_path}")
    logger.info(f"Engine size: {engine_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    args = parse_args()
    build_engine(args)
    
    logger.info(f"✓ Engine build complete!")
    logger.info(f"Engine: {Path(args.engine).resolve()}")
    logger.info(f"Size: {Path(args.engine).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()