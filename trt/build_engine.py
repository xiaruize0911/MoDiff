import argparse
from pathlib import Path

import tensorrt as trt

from entropy_calibrator import MoDiffEntropyCalibrator


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

    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    print(f"[INFO] Parsing ONNX from {onnx_path}")
    with onnx_path.open("rb") as handle:
        if not parser.parse(handle.read()):
            print("[ERROR] ONNX parser errors:")
            for idx in range(parser.num_errors):
                print(f"  {parser.get_error(idx)}")
            raise RuntimeError(f"Failed to parse ONNX graph at {onnx_path}")

    print("[INFO] Parsed ONNX graph with the following inputs:")
    for idx in range(network.num_inputs):
        tensor = network.get_input(idx)
        print(f"  - {tensor.name}: shape={tensor.shape}, dtype={tensor.dtype}")

    config = builder.create_builder_config()
    # TensorRT 10.x uses set_memory_pool_limit instead of max_workspace_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
    if not args.no_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = MoDiffEntropyCalibrator(calib_dir)
        print("[INFO] INT8 calibration enabled")
    else:
        print("[INFO] Building FP32 engine (INT8 disabled)")

    profile = builder.create_optimization_profile()
    print(f"[INFO] Setting optimization profile from ONNX model shapes:")
    
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
        print(f"  {tensor.name}: {static_shape}")
    config.add_optimization_profile(profile)

    print("[INFO] Building engine...")
    try:
        # TensorRT 10.x uses build_serialized_network instead of build_engine
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"[ERROR] Engine build failed with exception: {e}")
        raise
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine (returned None)")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with engine_path.open("wb") as handle:
        handle.write(serialized_engine)
    print(f"[INFO] Saved engine to {engine_path}")


def main() -> None:
    args = parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()