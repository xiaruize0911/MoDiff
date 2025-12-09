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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT FP32 engine for MoDiff UNet")
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
        "--engine",
        default="modiff_unet_fp32.plan",
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
        default=True,
        help="Build FP32 engine (default, INT8 removed)",
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


def build_engine(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()
    calib_dir = Path(args.calib_dir).expanduser().resolve()

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
    
    # Build FP32 engine (INT8 support removed)
    logger.info("Building FP32 engine")

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

    # Handle TRT 10.x IHostMemory object
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(serialized_engine, 'tobytes'):
        # TRT 10.x: IHostMemory has tobytes()
        engine_bytes = serialized_engine.tobytes()
    elif hasattr(serialized_engine, '__bytes__'):
        engine_bytes = bytes(serialized_engine)
    else:
        # Fallback: try direct write (older TRT versions)
        engine_bytes = serialized_engine
    
    with engine_path.open("wb") as handle:
        handle.write(engine_bytes)
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