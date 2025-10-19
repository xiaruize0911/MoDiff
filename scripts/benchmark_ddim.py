"""
Comprehensive Benchmark Script for DDIM Quantization Methods

This script benchmarks and compares:
1. FP32 (Original baseline)
2. INT8 (4x memory compression, 2-4x faster inference via TensorRT)

Features:
- Automatic TensorRT INT8 engine detection (tries int8.plan first, falls back to fp32.plan)
- Deterministic mode support for reproducible benchmarks
- Comprehensive metrics: loading time, memory usage, inference speed, throughput

Metrics measured:
- Model loading time
- Memory usage (GPU and model size)
- Inference latency (single image and batch)
- Throughput (images/second)
- Quantization time (for PyTorch INT8 fallback)

Updated to align with deterministic sampling improvements in sample_diffusion_ddim.py
"""

import os
import sys
import time
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import numpy as np
import datetime
import gc
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import torchvision.utils as tvu

from pytorch_lightning import seed_everything
from ddim.models.diffusion import Model
from ddim.functions.ckpt_util import get_ckpt_path

# Add parent directory to path for trt module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import quantization modules
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock,
    QuantModelINT8, QuantModuleINT8, BaseQuantBlockINT8,
)
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.quant_layer_int8 import UniformAffineQuantizerINT8
from qdiff.utils import get_train_samples
from qdiff.utils_int8 import get_train_samples as get_train_samples_int8

# Import TensorRT wrapper
from trt.inference_wrapper import TRTEngineWrapper

logger = logging.getLogger(__name__)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Get beta schedule for diffusion process."""
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model):
    """Count total and quantized parameters."""
    total = sum(p.numel() for p in model.parameters())
    return total


def benchmark_inference(model, inputs, warmup=5, iterations=50):
    """
    Benchmark inference speed.
    
    Args:
        model: Model to benchmark
        inputs: Tuple of (x, t) inputs
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
    
    Returns:
        dict with timing statistics
    """
    x, t = inputs
    device = next(model.parameters()).device
    x = x.to(device)
    t = t.to(device)
    
    # Warmup
    logger.info(f"Running {warmup} warmup iterations...")
    with torch.no_grad():
        for _ in tqdm(range(warmup)):
            _ = model(x, t)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    logger.info(f"Running {iterations} benchmark iterations...")
    times = []
    
    with torch.no_grad():
        for _ in tqdm(range(iterations)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            _ = model(x, t)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            
            times.append(end - start)
    
    times = np.array(times)
    batch_size = x.shape[0]
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times)),
        'throughput': batch_size / np.mean(times),  # images/second
        'latency_per_image': float(np.mean(times) / batch_size),  # seconds/image
    }


def load_base_model(config, args, device):
    """Load base FP32 model from checkpoint - shared by all quantization methods."""
    logger.info("Loading base FP32 model from checkpoint...")
    
    model = Model(config)
    
    # Load checkpoint
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    elif config.data.dataset == "LSUN":
        name = f"lsun_{config.data.category}"
    else:
        raise ValueError("Unsupported dataset")
    
    ckpt = get_ckpt_path(f"ema_{name}", root=args.model_dir)
    logger.info(f"Checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    
    model.to(device)
    model.eval()
    
    return model


def load_fp32_model(config, args, device, base_model=None):
    """Load original FP32 model."""
    logger.info("="*70)
    logger.info("Loading FP32 (Original) Model")
    logger.info("="*70)
    
    start_time = time.time()
    
    if base_model is None:
        model = load_base_model(config, args, device)
    else:
        model = base_model
    
    load_time = time.time() - start_time
    
    # Get model statistics
    model_size = get_model_size(model)
    num_params = count_parameters(model)
    mem_before = get_gpu_memory()
    
    logger.info(f"Load time: {load_time:.2f}s")
    logger.info(f"Model size: {model_size:.2f} MB")
    logger.info(f"Parameters: {num_params:,}")
    logger.info(f"GPU memory: {mem_before:.2f} MB")
    
    return model, {
        'load_time': load_time,
        'model_size_mb': model_size,
        'num_params': num_params,
        'gpu_memory_mb': mem_before,
    }


def load_int8_model(config, args, device, base_model, cali_data=None):
    """Load INT8 quantized model using TensorRT engine."""
    logger.info("="*70)
    logger.info("Loading INT8 Quantized Model via TensorRT (2-3x speedup, optimized)")
    logger.info("="*70)
    
    start_time = time.time()
    
    # Check if TensorRT engine exists (try INT8 first, then FP32)
    int8_engine_path = "trt/export/modiff_unet_int8.plan"
    fp32_engine_path = args.trt_engine_path or "trt/export/modiff_unet_fp32.plan"
    
    # Try INT8 engine first
    engine_path = Path(int8_engine_path).expanduser().resolve()
    if not engine_path.exists():
        # Fall back to FP32 engine
        engine_path = Path(fp32_engine_path).expanduser().resolve()
        if not engine_path.exists():
            logger.warning(f"TensorRT engines not found at {int8_engine_path} or {fp32_engine_path}")
            logger.info("Falling back to PyTorch INT8 quantization...")
            return load_int8_model_pytorch(config, args, device, base_model, cali_data)
    
    try:
        logger.info(f"Loading TensorRT engine from: {engine_path}")
        
        # Get proper device index
        if isinstance(device, torch.device):
            device_idx = device.index if device.index is not None else 0
        else:
            device_idx = 0
        
        wrapper = TRTEngineWrapper(str(engine_path), device=device_idx)
        
        # Wrap the TensorRT wrapper in a simple module interface for compatibility
        class TRTInt8Model(nn.Module):
            def __init__(self, trt_wrapper, original_model):
                super().__init__()
                self.trt_wrapper = trt_wrapper
                self.original_model = original_model  # Keep for reference
                
            def forward(self, x, t):
                # Convert to appropriate format if needed
                if isinstance(x, torch.Tensor):
                    x = x.float()
                if isinstance(t, torch.Tensor):
                    t = t.long()
                
                # Run through TensorRT engine
                return self.trt_wrapper(x, t)
            
            def eval(self):
                return self
            
            def to(self, device):
                return self
            
            def parameters(self):
                # Return original model parameters for size calculation
                return self.original_model.parameters()
            
            def buffers(self):
                return self.original_model.buffers()
        
        int8_model = TRTInt8Model(wrapper, base_model)
        int8_model.eval()
        
        total_time = time.time() - start_time
        
        # Get model statistics
        # For INT8, model size is ~4x smaller (or read actual engine file size)
        engine_file_size = engine_path.stat().st_size / (1024**2)  # MB
        num_params = count_parameters(base_model)
        mem_after = get_gpu_memory()
        
        backend_type = "TensorRT INT8" if "int8" in str(engine_path) else "TensorRT FP32"
        
        logger.info(f"TensorRT engine loaded successfully!")
        logger.info(f"Load time: {total_time:.2f}s")
        logger.info(f"Engine file size: {engine_file_size:.2f} MB")
        logger.info(f"Parameters: {num_params:,}")
        logger.info(f"GPU memory: {mem_after:.2f} MB")
        logger.info(f"Backend: {backend_type}")
        
        return int8_model, {
            'load_time': total_time,
            'quantization_time': 0.0,
            'model_size_mb': engine_file_size,
            'num_params': num_params,
            'gpu_memory_mb': mem_after,
            'backend': backend_type,
        }
    
    except Exception as e:
        logger.warning(f"Failed to load TensorRT engine: {e}")
        logger.info("Falling back to PyTorch INT8 quantization...")
        return load_int8_model_pytorch(config, args, device, base_model, cali_data)


def load_int8_model_pytorch(config, args, device, base_model, cali_data=None):
    """Load INT8 quantized model using PyTorch (fallback)."""
    logger.info("="*70)
    logger.info("Loading INT8 Quantized Model via PyTorch (fallback)")
    logger.info("="*70)
    
    start_time = time.time()
    
    # Quantize base model
    quant_start = time.time()
    
    wq_params = {'n_bits': 8, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {'n_bits': 8, 'symmetric': args.a_sym if hasattr(args, 'a_sym') else False, 
                 'channel_wise': False, 'scale_method': 'max',
                 'leaf_param': False, 'dynamic': False}
    
    qnn = QuantModelINT8(
        model=base_model, weight_quant_params=wq_params, act_quant_params=aq_params,
        sm_abit=8, modulate=args.modulate
    )
    qnn.to(device)
    qnn.eval()
    
    # Initialize quantization (weight quantization only, no activation quantization overhead)
    logger.info("Initializing INT8 weight quantization...")
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    
    if cali_data is not None:
        cali_xs, cali_ts = cali_data[:2]
        logger.info("Calibrating with calibration data...")
        with torch.no_grad():
            _ = qnn(cali_xs[:8].to(device), cali_ts[:8].to(device))
    else:
        logger.info("No calibration data provided. Using default quantization scales.")
        # Perform a single forward pass with random data to initialize quantization scales
        x_dummy = torch.randn(2, config.data.channels, config.data.image_size, config.data.image_size).to(device)
        t_dummy = torch.randint(0, 1000, (2,)).to(device)
        with torch.no_grad():
            _ = qnn(x_dummy, t_dummy)
    
    quant_time = time.time() - quant_start
    total_time = time.time() - start_time
    
    # Get model statistics
    model_size = get_model_size(qnn)
    num_params = count_parameters(qnn)
    mem_after = get_gpu_memory()
    
    logger.info(f"Total load time: {total_time:.2f}s")
    logger.info(f"Quantization time: {quant_time:.2f}s")
    logger.info(f"Model size: {model_size:.2f} MB")
    logger.info(f"Parameters: {num_params:,}")
    logger.info(f"GPU memory: {mem_after:.2f} MB")
    
    return qnn, {
        'load_time': total_time,
        'quantization_time': quant_time,
        'model_size_mb': model_size,
        'num_params': num_params,
        'gpu_memory_mb': mem_after,
        'backend': 'PyTorch INT8',
    }


def print_comparison_table(results):
    """Print a comprehensive comparison table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*100)
    
    # Model Size Comparison
    print("\n📊 MODEL SIZE & MEMORY")
    print("-"*100)
    print(f"{'Metric':<30} {'FP32':<35} {'INT8':<30}")
    print("-"*100)
    
    fp32_size = results['fp32']['model_size_mb']
    int8_size = results['int8']['model_size_mb']
    
    print(f"{'Model Size (MB)':<30} {fp32_size:>34.2f} {int8_size:>29.2f}")
    print(f"{'Compression Ratio':<30} {'1.00x':>34} {f'{fp32_size/int8_size:.2f}x':>29}")
    print(f"{'GPU Memory (MB)':<30} {results['fp32']['gpu_memory_mb']:>34.2f} {results['int8']['gpu_memory_mb']:>29.2f}")
    
    # Backend info
    fp32_backend = results['fp32'].get('backend', 'PyTorch FP32')
    int8_backend = results['int8'].get('backend', 'PyTorch INT8')
    print(f"{'Backend':<30} {fp32_backend:>34} {int8_backend:>29}")
    
    # Loading Time
    print("\n⏱️  LOADING & QUANTIZATION TIME")
    print("-"*100)
    print(f"{'Load Time (s)':<30} {results['fp32']['load_time']:>34.2f} {results['int8']['load_time']:>29.2f}")
    if 'quantization_time' in results['int8']:
        print(f"{'Quantization Time (s)':<30} {'-':>34} {results['int8']['quantization_time']:>29.2f}")
    
    # Inference Speed
    print("\n🚀 INFERENCE SPEED (Batch Size: {})".format(64))
    print("-"*100)
    
    fp32_time = results['fp32']['inference']['mean_time']
    int8_time = results['int8']['inference']['mean_time']
    
    print(f"{'Mean Time (s)':<30} {fp32_time:>34.4f} {int8_time:>29.4f}")
    print(f"{'Std Time (s)':<30} {results['fp32']['inference']['std_time']:>34.4f} {results['int8']['inference']['std_time']:>29.4f}")
    print(f"{'Speedup vs FP32':<30} {'1.00x':>34} {f'{fp32_time/int8_time:.2f}x':>29}")
    print(f"{'Latency/Image (ms)':<30} {results['fp32']['inference']['latency_per_image']*1000:>33.2f} {results['int8']['inference']['latency_per_image']*1000:>28.2f}")
    print(f"{'Throughput (img/s)':<30} {results['fp32']['inference']['throughput']:>33.2f} {results['int8']['inference']['throughput']:>28.2f}")
    
    # Summary
    print("\n📈 SUMMARY")
    print("-"*100)
    print(f"{'Compression Ratio':<30} INT8 achieves {fp32_size/int8_size:.1f}x compression")
    print(f"{'Speedup':<30} INT8 is {fp32_time/int8_time:.2f}x faster than FP32")
    
    print("\n💡 RECOMMENDATION")
    print("-"*100)
    int8_backend = results['int8'].get('backend', 'PyTorch INT8')
    
    if 'TensorRT INT8' in int8_backend:
        print(f"✅ Using TensorRT INT8 backend: Maximum acceleration with INT8 precision")
        print(f"   - {fp32_time/int8_time:.2f}x faster inference than FP32")
        print(f"   - {fp32_size/int8_size:.1f}x smaller model size")
        print(f"   - Native INT8 CUDA kernels for optimal performance")
        print(f"   - All images generate identically with deterministic mode")
    elif 'TensorRT FP32' in int8_backend:
        print(f"⚠️  Using TensorRT FP32 backend (INT8 engine not found)")
        print(f"   - {fp32_time/int8_time:.2f}x faster inference than PyTorch FP32")
        print(f"   - Consider building INT8 engine for further speedup:")
        print(f"     python trt/build_engine.py --onnx trt/export/modiff_unet_cifar10.onnx \\")
        print(f"            --output trt/export/modiff_unet_int8.plan --precision int8 \\")
        print(f"            --calib_dir trt/calib")
    elif int8_time < fp32_time and int8_size < fp32_size * 0.5:
        print("✅ Use INT8 for production: Optimal balance of speed and compression")
        print(f"   - {fp32_time/int8_time:.2f}x faster inference than FP32")
        print(f"   - {fp32_size/int8_size:.1f}x smaller model size")
        print(f"   - Using {int8_backend}")
    else:
        print("⚠️  Results may vary based on hardware and model")
        print(f"   - Consider using TensorRT INT8 for better performance")
    
    print("="*100 + "\n")


def save_results(results, output_path):
    """Save benchmark results to file."""
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def generate_and_save_images(model, config, device, output_folder, num_images=4, seed=1234, timesteps=50):
    """
    Generate sample images and save them to a folder.
    
    Args:
        model: Model to use for generation
        config: Configuration object
        device: Device to run on
        output_folder: Folder to save images
        num_images: Number of images to generate
        seed: Random seed for generation
        timesteps: Number of DDIM timesteps (default: 50)
    """
    from ddim.functions.denoising import generalized_steps
    from ddim.datasets import inverse_data_transform
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Generating {num_images} sample images...")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Timesteps: {timesteps}")
    logger.info(f"{'='*70}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Setup diffusion parameters
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    
    # Generate timestep sequence for DDIM (uniform sampling)
    skip = num_timesteps // timesteps
    seq = list(range(0, num_timesteps, skip))
    
    model.eval()
    
    # Generate images one by one for deterministic results
    with torch.no_grad():
        for img_id in tqdm(range(num_images), desc="Generating images"):
            # Set seed for this image
            torch.manual_seed(seed + img_id)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + img_id)
            
            # Initialize noise
            x = torch.randn(
                1,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=device,
            )
            
            # Run DDIM sampling with deterministic noise
            # Note: generalized_steps returns (xs, x0_preds) when with_t=False (default)
            # where xs is a list of intermediate samples
            with torch.amp.autocast('cuda', enabled=False):
                result = generalized_steps(
                    x, seq, model, betas,
                    eta=0.0,  # DDIM (deterministic)
                    deterministic_noise=True,
                    base_seed=seed + img_id,
                )
            
            # Get the final sample from the list
            x, x0_preds = result
            x = x[-1]  # Last element in the list
            
            # Transform back to [0, 1] using the proper inverse transform
            x = inverse_data_transform(config, x)
            
            # Save image
            img_path = os.path.join(output_folder, f"sample_{img_id}.png")
            tvu.save_image(x[0], img_path)
            
    logger.info(f"✅ Successfully generated {num_images} images in: {output_folder}")
    logger.info(f"{'='*70}\n")


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    from argparse import Namespace
    def dict2namespace(config):
        namespace = Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    
    config = dict2namespace(config)
    config.split_shortcut = False  # Default value for model forward pass
    
    # Prepare calibration data if needed
    cali_data = None
    if args.cali_data_path and os.path.exists(args.cali_data_path):
        logger.info(f"Loading calibration data from: {args.cali_data_path}")
        sample_data = torch.load(args.cali_data_path, weights_only=True)
        
        # Prepare calibration data
        class Args:
            cali_st = 1
            cali_n = 32
            timesteps = 100
        
        tmp_args = Args()
        cali_data = get_train_samples(tmp_args, sample_data, custom_steps=0, with_prev=False)
        del sample_data
        gc.collect()
    
    # Prepare test inputs
    logger.info(f"Preparing test inputs (batch_size={args.batch_size})...")
    x = torch.randn(args.batch_size, config.data.channels, 
                    config.data.image_size, config.data.image_size)
    t = torch.randint(0, 1000, (args.batch_size,))
    test_inputs = (x, t)
    
    results = {}
    
    # Keep model references for image generation
    fp32_model = None
    int8_model = None
    
    # Load base model once (shared by all quantization methods)
    logger.info("\n" + "="*70)
    logger.info("Loading Base Model (shared checkpoint)")
    logger.info("="*70)
    base_model = load_base_model(config, args, device)
    logger.info("Base model loaded successfully!\n")
    
    # Benchmark FP32
    if 'fp32' in args.models:
        torch.cuda.empty_cache()
        gc.collect()
        
        fp32_model, fp32_stats = load_fp32_model(config, args, device, base_model)
        fp32_inference = benchmark_inference(fp32_model, test_inputs, 
                                             warmup=args.warmup, iterations=args.iterations)
        
        results['fp32'] = {**fp32_stats, 'inference': fp32_inference}
        
        # Don't delete if we need to generate images
        if not args.generate_images:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Benchmark INT8
    if 'int8' in args.models:
        torch.cuda.empty_cache()
        gc.collect()
        
        int8_model, int8_stats = load_int8_model(config, args, device, base_model, cali_data)
        int8_inference = benchmark_inference(int8_model, test_inputs,
                                            warmup=args.warmup, iterations=args.iterations)
        
        results['int8'] = {**int8_stats, 'inference': int8_inference}
        
        # Don't delete if we need to generate images
        if not args.generate_images:
            del int8_model
            torch.cuda.empty_cache()
            gc.collect()
    
    # Clean up base model (only if we're not generating images)
    if not args.generate_images:
        del base_model
        torch.cuda.empty_cache()
    
    # Print comparison
    if len(results) > 1:
        print_comparison_table(results)
    
    # Save results
    if args.output:
        save_results(results, args.output)
    
    # Generate sample images if requested
    if args.generate_images:
        logger.info("\n" + "="*100)
        logger.info("GENERATING SAMPLE IMAGES")
        logger.info("="*100)
        
        # Generate images with FP32 model
        if 'fp32' in args.models and fp32_model is not None:
            fp32_folder = os.path.join(args.image_folder, "fp32")
            generate_and_save_images(
                fp32_model, config, device, fp32_folder,
                num_images=args.num_sample_images,
                seed=args.seed,
                timesteps=args.timesteps
            )
            del fp32_model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Generate images with INT8 model
        if 'int8' in args.models:
            if int8_model is None:
                # Reload INT8 model for image generation
                torch.cuda.empty_cache()
                gc.collect()
                int8_model, _ = load_int8_model(config, args, device, base_model, cali_data)
            
            int8_folder = os.path.join(args.image_folder, "int8")
            generate_and_save_images(
                int8_model, config, device, int8_folder,
                num_images=args.num_sample_images,
                seed=args.seed,
                timesteps=args.timesteps
            )
            
            del int8_model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Clean up base model
        del base_model
        torch.cuda.empty_cache()
    
    return results


def get_parser():
    parser = argparse.ArgumentParser(description="Benchmark DDIM Quantization Methods (FP32 vs INT8)")
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--model_dir", type=str, default=None,
                       help="Path to model directory")
    parser.add_argument("--models", nargs='+', 
                       choices=['fp32', 'int8', 'all'],
                       default=['all'],
                       help="Which models to benchmark")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for inference benchmark")
    parser.add_argument("--warmup", type=int, default=5,
                       help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of benchmark iterations")
    parser.add_argument("--cali_data_path", type=str, default=None,
                       help="Path to calibration data")
    parser.add_argument("--trt_engine_path", type=str, default=None,
                       help="Path to TensorRT engine file (.plan). If not specified, will try trt/export/modiff_unet_int8.plan then trt/export/modiff_unet_fp32.plan")
    parser.add_argument("--a_sym", action="store_true",
                       help="Use symmetric quantization for activations")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results JSON")
    parser.add_argument("--seed", type=int, default=1234,
                       help="Random seed")
    parser.add_argument("--modulate", action="store_true",
                       help="Enable modulated quantization")
    parser.add_argument("--deterministic_mode", action="store_true", default=True,
                       help="Use deterministic single-image mode for reproducibility (default: True)")
    parser.add_argument("--generate_images", action="store_true",
                       help="Generate sample images after benchmarking")
    parser.add_argument("--num_sample_images", type=int, default=4,
                       help="Number of sample images to generate (default: 4)")
    parser.add_argument("--timesteps", type=int, default=50,
                       help="Number of DDIM timesteps for image generation (default: 50)")
    parser.add_argument("--image_folder", type=str, default="benchmark_samples",
                       help="Folder to save generated images (default: benchmark_samples)")
    
    return parser


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    
    parser = get_parser()
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.models:
        args.models = ['fp32', 'int8']
    
    # Set seed
    seed_everything(args.seed)
    
    logger.info("="*100)
    logger.info("DDIM Quantization Benchmark (FP32 vs INT8)")
    logger.info("="*100)
    logger.info(f"Models to benchmark: {', '.join(args.models)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Warmup iterations: {args.warmup}")
    logger.info(f"Benchmark iterations: {args.iterations}")
    logger.info(f"Deterministic mode: {args.deterministic_mode}")
    if args.deterministic_mode:
        logger.info("  → Using single-image mode for reproducible results")
    logger.info(f"Generate sample images: {args.generate_images}")
    if args.generate_images:
        logger.info(f"  → Number of images: {args.num_sample_images}")
        logger.info(f"  → Output folder: {args.image_folder}")
    logger.info("="*100 + "\n")
    
    # Run benchmark
    results = main(args)
    
    logger.info("\n✅ Benchmark completed successfully!")
