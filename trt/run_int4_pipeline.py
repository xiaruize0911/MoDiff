#!/usr/bin/env python3
"""
MoDiff INT4 TensorRT Pipeline - Complete End-to-End

This script runs the complete INT4 quantization pipeline:
1. Generate calibration data
2. Extract INT4 scales using MSE-based search
3. Build TensorRT engine with INT4-style quantization
4. (Optional) Run inference benchmark

Following paper: "Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization"
Key innovation: Reduces activation quantization from 8 bits to 3-4 bits using modulated quantization.

Usage:
    cd MoDiff/trt
    python run_int4_pipeline.py --config ../configs/cifar10.yml

Requirements:
    - TensorRT 8.x or 10.x
    - PyCUDA
    - PyTorch with CUDA
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import subprocess
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoDiff INT4 TensorRT Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="../configs/cifar10.yml",
        help="Path to model config",
    )
    parser.add_argument(
        "--onnx",
        default="modiff_unet_cifar10.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--calib-dir",
        default="../calibration",
        help="Calibration data directory",
    )
    parser.add_argument(
        "--output-dir",
        default="int4_output",
        help="Output directory for INT4 artifacts",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=4,
        choices=[3, 4],
        help="Quantization bits",
    )
    parser.add_argument(
        "--scale-method",
        default="mse",
        choices=["mse", "max", "minmax"],
        help="Scale computation method",
    )
    parser.add_argument(
        "--skip-calib",
        action="store_true",
        help="Skip calibration data generation",
    )
    parser.add_argument(
        "--skip-scales",
        action="store_true",
        help="Skip scale extraction",
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="Skip engine building",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark after building",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use pretrained CIFAR10 model",
    )
    return parser.parse_args()


def step_generate_calibration(args: argparse.Namespace) -> bool:
    """Step 1: Generate calibration data."""
    logger.info("=" * 60)
    logger.info("Step 1: Generate Calibration Data")
    logger.info("=" * 60)
    
    calib_dir = Path(args.calib_dir)
    if calib_dir.exists() and len(list(calib_dir.glob("*.npz"))) > 0:
        logger.info(f"✓ Calibration data already exists in {calib_dir}")
        return True
    
    cmd = [
        sys.executable, "create_calib_data.py",
        "--config", args.config,
        "--num-samples", "64",
        "--output-dir", str(calib_dir),
    ]
    if args.use_pretrained:
        cmd.append("--use-pretrained")
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logger.error("Calibration data generation failed")
        return False
    
    logger.info("✓ Calibration data generated")
    return True


def step_extract_scales(args: argparse.Namespace) -> bool:
    """Step 2: Extract INT4 scales using MSE search."""
    logger.info("=" * 60)
    logger.info(f"Step 2: Extract INT{args.n_bits} Scales (MSE-based)")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir) / "scales"
    scales_file = output_dir / f"model_scales_int{args.n_bits}.json"
    
    if scales_file.exists():
        logger.info(f"✓ Scales already exist: {scales_file}")
        return True
    
    cmd = [
        sys.executable, "extract_scales_int4.py",
        "--config", args.config,
        "--calib-dir", args.calib_dir,
        "--output", str(output_dir),
        "--n-bits", str(args.n_bits),
        "--scale-method", args.scale_method,
    ]
    if args.use_pretrained:
        cmd.append("--use-pretrained")
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logger.error("Scale extraction failed")
        return False
    
    logger.info("✓ INT4 scales extracted")
    return True


def step_build_engine(args: argparse.Namespace) -> bool:
    """Step 3: Build TensorRT INT4 engine."""
    logger.info("=" * 60)
    logger.info(f"Step 3: Build TensorRT INT{args.n_bits} Engine")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir)
    engine_path = output_dir / f"modiff_unet_int{args.n_bits}.plan"
    
    if engine_path.exists():
        logger.info(f"✓ Engine already exists: {engine_path}")
        return True
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "build_engine_int4.py",
        "--onnx", args.onnx,
        "--calib-dir", args.calib_dir,
        "--engine", str(engine_path),
        "--n-bits", str(args.n_bits),
        "--scale-method", args.scale_method,
        "--fp16",
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logger.error("Engine build failed")
        return False
    
    logger.info("✓ TensorRT INT4 engine built")
    return True


def step_benchmark(args: argparse.Namespace) -> bool:
    """Step 4: Run inference benchmark."""
    logger.info("=" * 60)
    logger.info("Step 4: Inference Benchmark")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir)
    engine_path = output_dir / f"modiff_unet_int{args.n_bits}.plan"
    
    if not engine_path.exists():
        logger.error(f"Engine not found: {engine_path}")
        return False
    
    # Simple benchmark using trtexec if available
    try:
        result = subprocess.run(
            ["trtexec", "--loadEngine", str(engine_path), "--warmUp=1000", "--iterations=100"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode == 0:
            # Parse output for latency
            for line in result.stdout.split('\n'):
                if 'mean' in line.lower() or 'latency' in line.lower():
                    logger.info(line.strip())
        else:
            logger.warning("trtexec benchmark failed, trying Python benchmark...")
            _python_benchmark(engine_path)
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.info("trtexec not available, running Python benchmark...")
        _python_benchmark(engine_path)
    
    return True


def _python_benchmark(engine_path: Path, num_warmup: int = 10, num_iters: int = 100):
    """Run Python-based benchmark."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np
        
        # Load engine
        logger.info(f"Loading engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs = []
        outputs = []
        bindings = []
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            
            # Replace -1 with 1 for dynamic dims
            shape = tuple(1 if s == -1 else s for s in shape)
            size = int(np.prod(shape))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
        
        stream = cuda.Stream()
        
        # Warmup
        logger.info(f"Warming up ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            for inp in inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            context.execute_async_v2(bindings, stream.handle)
            stream.synchronize()
        
        # Benchmark
        logger.info(f"Benchmarking ({num_iters} iterations)...")
        start = time.perf_counter()
        for _ in range(num_iters):
            for inp in inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            context.execute_async_v2(bindings, stream.handle)
            stream.synchronize()
        end = time.perf_counter()
        
        total_time = end - start
        avg_latency = (total_time / num_iters) * 1000  # ms
        throughput = num_iters / total_time  # samples/sec
        
        logger.info(f"Results:")
        logger.info(f"  Average latency: {avg_latency:.2f} ms")
        logger.info(f"  Throughput: {throughput:.1f} samples/sec")
        
    except Exception as e:
        logger.error(f"Python benchmark failed: {e}")


def print_summary(args: argparse.Namespace, success: bool):
    """Print pipeline summary."""
    print("\n" + "=" * 60)
    print("MoDiff INT4 Pipeline Summary")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    
    print(f"\nConfiguration:")
    print(f"  Quantization bits: {args.n_bits}")
    print(f"  Scale method: {args.scale_method}")
    print(f"  Output directory: {output_dir}")
    
    print(f"\nGenerated files:")
    
    # Check calibration
    calib_dir = Path(args.calib_dir)
    calib_files = list(calib_dir.glob("*.npz")) if calib_dir.exists() else []
    print(f"  Calibration samples: {len(calib_files)}")
    
    # Check scales
    scales_file = output_dir / "scales" / f"model_scales_int{args.n_bits}.json"
    print(f"  Scales file: {'✓' if scales_file.exists() else '✗'} {scales_file}")
    
    # Check engine
    engine_path = output_dir / f"modiff_unet_int{args.n_bits}.plan"
    if engine_path.exists():
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"  TensorRT engine: ✓ {engine_path} ({size_mb:.1f} MB)")
    else:
        print(f"  TensorRT engine: ✗ {engine_path}")
    
    print(f"\nStatus: {'✓ SUCCESS' if success else '✗ FAILED'}")
    
    if success:
        print(f"\nNext steps:")
        print(f"  1. Use the engine for INT4 inference:")
        print(f"     from trt.int4_inference import TRTInt4Wrapper")
        print(f"     engine = TRTInt4Wrapper('{engine_path}')")
        print(f"     output = engine(latent, timesteps)")
        print(f"")
        print(f"  2. Or use with QuantModelINT4:")
        print(f"     from qdiff import QuantModelINT4")
        print(f"     qmodel = QuantModelINT4(model, trt_engine_path='{engine_path}')")


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"MoDiff INT{args.n_bits} TensorRT Pipeline")
    print("Following paper: Modulated Diffusion (ICML 2025)")
    print("=" * 60)
    print()
    
    # Change to trt directory
    trt_dir = Path(__file__).parent
    os.chdir(trt_dir)
    logger.info(f"Working directory: {trt_dir}")
    
    success = True
    
    # Step 1: Generate calibration data
    if not args.skip_calib:
        if not step_generate_calibration(args):
            success = False
    else:
        logger.info("Skipping calibration data generation")
    
    # Step 2: Extract INT4 scales
    if success and not args.skip_scales:
        if not step_extract_scales(args):
            success = False
    else:
        logger.info("Skipping scale extraction")
    
    # Step 3: Build TensorRT engine
    if success and not args.skip_engine:
        if not step_build_engine(args):
            success = False
    else:
        logger.info("Skipping engine build")
    
    # Step 4: Benchmark (optional)
    if success and args.benchmark:
        step_benchmark(args)
    
    # Print summary
    print_summary(args, success)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
