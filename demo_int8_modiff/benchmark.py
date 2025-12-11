"""
Benchmark Module

Compares performance and quality between FP32, INT8, and INT8+MoDiff models.

Usage:
    python -m demo_int8_modiff.benchmark \
        --config configs/cifar10.yml \
        --ckpt models/ema_diffusion_cifar10_model/model-790000.ckpt \
        --num_samples 100 \
        --num_images 1000

Measures:
- Inference latency
- Memory usage  
- FID score (optional)
- Sample quality metrics
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ddim.models.diffusion import Model
from demo_int8_modiff.quant_model_modiff import QuantModelMoDiff
from demo_int8_modiff.ddim_sampler import DDIMSamplerMoDiff
from demo_int8_modiff.calibration import calibrate_model_qdiff

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)


def dict2namespace(config: dict):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def load_model(config_path: str, ckpt_path: str, device: str = 'cuda') -> Tuple[nn.Module, object]:
    """Load model and config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    model = Model(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if 'ema' in ckpt:
        state_dict = ckpt['ema']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    
    return model, config


def measure_latency(
    model: nn.Module,
    sampler: DDIMSamplerMoDiff,
    num_samples: int,
    image_size: int,
    channels: int,
    ddim_steps: int,
    device: str,
    warmup: int = 5,
) -> Dict[str, float]:
    """
    Measure inference latency.
    
    Returns dict with:
    - total_time: Total time for all samples
    - time_per_sample: Average time per sample
    - time_per_step: Average time per denoising step
    """
    # Warmup
    logger.info(f"Warming up with {warmup} samples...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = sampler.sample(
                num_samples=1,
                image_size=image_size,
                channels=channels,
                ddim_steps=ddim_steps,
            )
    
    torch.cuda.synchronize() if device == 'cuda' else None
    
    # Benchmark
    logger.info(f"Benchmarking {num_samples} samples...")
    
    times = []
    for i in tqdm(range(num_samples), desc="Measuring latency"):
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = sampler.sample(
                num_samples=1,
                image_size=image_size,
                channels=channels,
                ddim_steps=ddim_steps,
            )
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.perf_counter()
        times.append(end - start)
    
    total_time = sum(times)
    time_per_sample = np.mean(times)
    time_per_step = time_per_sample / ddim_steps
    
    return {
        'total_time': total_time,
        'time_per_sample': time_per_sample,
        'time_per_step': time_per_step,
        'std_dev': np.std(times),
        'min_time': min(times),
        'max_time': max(times),
    }


def measure_memory(
    model: nn.Module,
    image_size: int,
    channels: int,
    device: str,
) -> Dict[str, float]:
    """Measure GPU memory usage."""
    if device != 'cuda':
        return {'peak_memory_mb': 0, 'allocated_mb': 0}
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    # Run a forward pass
    x = torch.randn(1, channels, image_size, image_size, device=device)
    t = torch.randint(0, 1000, (1,), device=device)
    
    with torch.no_grad():
        _ = model(x, t)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    
    return {
        'peak_memory_mb': peak_memory,
        'allocated_mb': allocated,
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total,
        'trainable_params': trainable,
        'total_params_m': total / 1e6,
    }


def benchmark_model(
    model: nn.Module,
    config: object,
    model_name: str,
    num_samples: int,
    ddim_steps: int,
    device: str,
    use_modiff: bool = False,
) -> Dict:
    """Benchmark a single model configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'='*60}")
    
    # Create sampler
    sampler = DDIMSamplerMoDiff(
        model=model,
        num_timesteps=config.diffusion.num_diffusion_timesteps,
        beta_schedule='linear',
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )
    
    # Set MoDiff state if applicable
    if hasattr(model, 'set_modulation'):
        model.set_modulation(use_modiff)
    
    image_size = config.data.image_size
    channels = config.data.channels
    
    results = {
        'model_name': model_name,
        'use_modiff': use_modiff,
    }
    
    # Parameter count
    param_info = count_parameters(model)
    results.update(param_info)
    logger.info(f"Parameters: {param_info['total_params_m']:.2f}M")
    
    # Memory usage
    mem_info = measure_memory(model, image_size, channels, device)
    results.update(mem_info)
    logger.info(f"Peak memory: {mem_info['peak_memory_mb']:.1f} MB")
    
    # Latency
    latency_info = measure_latency(
        model=model,
        sampler=sampler,
        num_samples=num_samples,
        image_size=image_size,
        channels=channels,
        ddim_steps=ddim_steps,
        device=device,
    )
    results.update(latency_info)
    logger.info(f"Time per sample: {latency_info['time_per_sample']:.3f}s ± {latency_info['std_dev']:.3f}s")
    logger.info(f"Time per step: {latency_info['time_per_step']*1000:.2f}ms")
    
    return results


def generate_samples_for_fid(
    model: nn.Module,
    config: object,
    output_dir: str,
    num_images: int,
    ddim_steps: int,
    batch_size: int,
    device: str,
    use_modiff: bool = False,
):
    """Generate samples for FID evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    
    sampler = DDIMSamplerMoDiff(
        model=model,
        num_timesteps=config.diffusion.num_diffusion_timesteps,
        beta_schedule='linear',
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )
    
    if hasattr(model, 'set_modulation'):
        model.set_modulation(use_modiff)
    
    image_size = config.data.image_size
    channels = config.data.channels
    
    all_samples = []
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
        
        samples = sampler.sample(
            num_samples=current_batch_size,
            image_size=image_size,
            channels=channels,
            ddim_steps=ddim_steps,
        )
        
        # Convert to uint8
        samples = (samples.clamp(-1, 1) + 1) / 2 * 255
        samples = samples.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        all_samples.append(samples)
    
    all_samples = np.concatenate(all_samples, axis=0)[:num_images]
    
    # Save as npz
    output_path = os.path.join(output_dir, 'samples.npz')
    np.savez(output_path, samples=all_samples)
    logger.info(f"Saved {len(all_samples)} samples to {output_path}")
    
    return output_path


def print_comparison(results: List[Dict]):
    """Print comparison table."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    headers = ['Model', 'MoDiff', 'Params (M)', 'Memory (MB)', 'Time/Sample (s)', 'Time/Step (ms)']
    
    # Header
    print(f"\n{headers[0]:<20} {headers[1]:<8} {headers[2]:<12} {headers[3]:<12} {headers[4]:<16} {headers[5]:<12}")
    print("-"*80)
    
    # Rows
    for r in results:
        modiff_str = "Yes" if r.get('use_modiff', False) else "No"
        print(f"{r['model_name']:<20} {modiff_str:<8} {r['total_params_m']:<12.2f} "
              f"{r['peak_memory_mb']:<12.1f} {r['time_per_sample']:<16.3f} "
              f"{r['time_per_step']*1000:<12.2f}")
    
    print("="*80)
    
    # Speedup comparison
    if len(results) >= 2:
        baseline = results[0]
        print("\nSpeedup vs baseline:")
        for r in results[1:]:
            speedup = baseline['time_per_sample'] / r['time_per_sample']
            print(f"  {r['model_name']}: {speedup:.2f}x")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark INT8 MoDiff")
    
    parser.add_argument("--config", type=str, default="configs/cifar10.yml")
    parser.add_argument("--ckpt", type=str,
                        default="models/ema_diffusion_cifar10_model/model-790000.ckpt")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples for latency measurement")
    parser.add_argument("--num_images", type=int, default=1000,
                        help="Number of images for FID evaluation")
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    
    parser.add_argument("--skip_fp32", action="store_true")
    parser.add_argument("--skip_int8", action="store_true")
    parser.add_argument("--skip_modiff", action="store_true")
    parser.add_argument("--compute_fid", action="store_true")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    
    # Load base model
    logger.info("Loading base model...")
    base_model, config = load_model(args.config, args.ckpt, device)
    
    # Benchmark FP32
    if not args.skip_fp32:
        fp32_results = benchmark_model(
            model=base_model,
            config=config,
            model_name="FP32 (Baseline)",
            num_samples=args.num_samples,
            ddim_steps=args.ddim_steps,
            device=device,
            use_modiff=False,
        )
        results.append(fp32_results)
        
        if args.compute_fid:
            logger.info("Generating FP32 samples for FID...")
            generate_samples_for_fid(
                model=base_model,
                config=config,
                output_dir=os.path.join(args.output_dir, "fp32_samples"),
                num_images=args.num_images,
                ddim_steps=args.ddim_steps,
                batch_size=args.batch_size,
                device=device,
            )
    
    # Create quantized model
    logger.info("\nCreating INT8 quantized model...")
    quant_model = QuantModelMoDiff(
        model=base_model,
        weight_quant_params={'n_bits': 8, 'symmetric': True, 'channel_wise': True},
        act_quant_params={'n_bits': 8, 'symmetric': True, 'channel_wise': False},
    )
    quant_model.to(device)
    
    # Simple calibration
    logger.info("Running calibration...")
    quant_model.set_quant_state(weight_quant=True, act_quant=False)
    
    # Quick calibration with random data
    calib_data = torch.randn(64, config.data.channels, config.data.image_size, 
                             config.data.image_size, device=device)
    calib_t = torch.randint(0, config.diffusion.num_diffusion_timesteps, 
                            (64,), device=device)
    
    with torch.no_grad():
        for i in range(0, 64, 8):
            _ = quant_model(calib_data[i:i+8], calib_t[i:i+8])
    
    quant_model.set_quant_state(weight_quant=True, act_quant=True)
    
    # Benchmark INT8 without MoDiff
    if not args.skip_int8:
        int8_results = benchmark_model(
            model=quant_model,
            config=config,
            model_name="INT8 (No MoDiff)",
            num_samples=args.num_samples,
            ddim_steps=args.ddim_steps,
            device=device,
            use_modiff=False,
        )
        results.append(int8_results)
        
        if args.compute_fid:
            logger.info("Generating INT8 samples for FID...")
            generate_samples_for_fid(
                model=quant_model,
                config=config,
                output_dir=os.path.join(args.output_dir, "int8_samples"),
                num_images=args.num_images,
                ddim_steps=args.ddim_steps,
                batch_size=args.batch_size,
                device=device,
                use_modiff=False,
            )
    
    # Benchmark INT8 with MoDiff
    if not args.skip_modiff:
        modiff_results = benchmark_model(
            model=quant_model,
            config=config,
            model_name="INT8 + MoDiff",
            num_samples=args.num_samples,
            ddim_steps=args.ddim_steps,
            device=device,
            use_modiff=True,
        )
        results.append(modiff_results)
        
        if args.compute_fid:
            logger.info("Generating INT8+MoDiff samples for FID...")
            generate_samples_for_fid(
                model=quant_model,
                config=config,
                output_dir=os.path.join(args.output_dir, "int8_modiff_samples"),
                num_images=args.num_images,
                ddim_steps=args.ddim_steps,
                batch_size=args.batch_size,
                device=device,
                use_modiff=True,
            )
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    import json
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Compute FID if requested
    if args.compute_fid:
        logger.info("\nComputing FID scores...")
        from demo_int8_modiff.fid_evaluation import compute_fid
        
        fid_results = {}
        for result in results:
            model_name = result['model_name'].replace(' ', '_').replace('(', '').replace(')', '')
            sample_dir = os.path.join(args.output_dir, f"{model_name.lower()}_samples")
            
            if os.path.exists(sample_dir):
                fid = compute_fid(sample_dir, ref_dataset='cifar10', device=device)
                fid_results[result['model_name']] = fid
                logger.info(f"{result['model_name']}: FID = {fid:.2f}")
        
        print("\n" + "="*50)
        print("FID SCORES")
        print("="*50)
        for name, fid in fid_results.items():
            print(f"{name}: {fid:.2f}")
        print("="*50)


if __name__ == "__main__":
    main()
