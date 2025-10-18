"""
Benchmark script to compare INT4, simulated quantization, and FP32 models.
Measures inference speed and FID score for diffusion models.
"""

import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pytorch_lightning import seed_everything

from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import generalized_steps

# Import both quantization versions
from qdiff import QuantModel, QuantModule, BaseQuantBlock
from qdiff import QuantModelINT4, QuantModuleINT4, BaseQuantBlockINT4
from qdiff.utils import resume_cali_model
from qdiff.utils_int4 import resume_cali_model_int4

import torchvision.utils as tvu


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Get noise schedule for diffusion process"""
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, 
                           num_diffusion_timesteps, dtype=np.float64) ** 2)
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def load_fp32_model(config, model_dir, device):
    """Load original FP32 model without quantization"""
    model = Model(config)
    
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    elif config.data.dataset == "LSUN":
        name = f"lsun_{config.data.category}"
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")
    
    ckpt = get_ckpt_path(f"ema_{name}", root=model_dir)
    print(f"Loading FP32 model from {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_simulated_quant_model(config, model_dir, cali_ckpt, cali_data, device, modulate=False):
    """Load model with simulated quantization (fake quantization)"""
    # Load base model
    model = Model(config)
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    elif config.data.dataset == "LSUN":
        name = f"lsun_{config.data.category}"
    else:
        raise ValueError
    
    ckpt = get_ckpt_path(f"ema_{name}", root=model_dir)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    
    # Apply simulated quantization (stays in float32)
    wq_params = {'n_bits': 4, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {
        'n_bits': 4, 
        'symmetric': False, 
        'channel_wise': False, 
        'scale_method': 'max',
        'leaf_param': True,  # Required parameter
        'dynamic': False
    }
    
    qnn = QuantModel(
        model=model, 
        weight_quant_params=wq_params, 
        act_quant_params=aq_params,
        sm_abit=8,  # Softmax activation bit
        modulate=modulate
    )
    qnn.to(device)
    qnn.eval()
    
    # Load calibrated weights
    if cali_ckpt and os.path.exists(cali_ckpt):
        print(f"Loading simulated quantization from {cali_ckpt}")
        resume_cali_model(qnn, cali_ckpt, cali_data, quant_act=True, cond=False)
    else:
        print("Warning: No calibration checkpoint provided for simulated quantization")
        # Initialize quantization (weight only for speed)
        print("Enabling weight quantization only (faster inference)...")
        qnn.set_quant_state(weight_quant=True, act_quant=False)
        dummy_x, dummy_t = cali_data[0][:1].to(device), cali_data[1][:1].to(device)
        _ = qnn(dummy_x, dummy_t)
    
    # Free original weights to make fair memory comparison
    print("Freeing original weights to measure true quantized model size...")
    free_original_weights(qnn)
    
    return qnn


def load_int4_model(config, model_dir, cali_ckpt, cali_data, device, modulate=False):
    """Load model with true INT4 quantization (packed uint8 storage)"""
    # Load base model
    model = Model(config)
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    elif config.data.dataset == "LSUN":
        name = f"lsun_{config.data.category}"
    else:
        raise ValueError
    
    ckpt = get_ckpt_path(f"ema_{name}", root=model_dir)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    
    # Apply INT4 quantization (true 4-bit storage)
    wq_params = {'n_bits': 4, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {
        'n_bits': 4, 
        'symmetric': False, 
        'channel_wise': False, 
        'scale_method': 'max',
        'leaf_param': True,  # Required parameter
        'dynamic': False
    }
    
    qnn = QuantModelINT4(
        model=model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        sm_abit=4,  # Softmax activation bit (must be 4 for INT4)
        modulate=modulate
    )
    qnn.to(device)
    qnn.eval()
    
    # Load calibrated weights
    if cali_ckpt and os.path.exists(cali_ckpt):
        print(f"Loading INT4 quantization from {cali_ckpt}")
        resume_cali_model_int4(qnn, cali_ckpt, cali_data, quant_act=True, cond=False)
    else:
        print("Warning: No calibration checkpoint provided for INT4")
        # Initialize quantization (weight only for speed)
        print("Enabling weight quantization only (faster inference)...")
        qnn.set_quant_state(weight_quant=True, act_quant=False)
        dummy_x, dummy_t = cali_data[0][:1].to(device), cali_data[1][:1].to(device)
        _ = qnn(dummy_x, dummy_t)
    
    # Free original weights to make fair memory comparison
    print("Freeing original weights to measure true INT4 model size...")
    free_original_weights(qnn)
    
    return qnn


def free_original_weights(qnn):
    """
    Free original float32 weights from quantized model to save memory.
    After quantization, the original weights are no longer needed.
    This makes memory comparison fair between FP32 and quantized models.
    
    For INT4 models, we keep only the shape information needed for dequantization.
    """
    for name, module in qnn.named_modules():
        if hasattr(module, 'org_weight'):
            # For INT4: Save shape, then delete the actual tensor data
            if hasattr(module, 'weight_int4_packed'):
                # INT4 quantized layer - save shape for dequantization
                weight_shape = module.org_weight.shape
                del module.org_weight
                # Create a dummy tensor with just the shape info (no data)
                module.org_weight_shape = weight_shape
            else:
                # Simulated quantization - can delete completely
                delattr(module, 'org_weight')
                
        if hasattr(module, 'org_bias') and module.org_bias is not None:
            # Bias can be deleted for both (small size anyway)
            delattr(module, 'org_bias')
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_memory(model, device):
    """Calculate model memory usage in MB"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # Trigger memory allocation
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        dummy_t = torch.tensor([0], device=device)
        with torch.no_grad():
            _ = model(dummy_input, dummy_t)
        
        memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        return memory_mb
    else:
        # For CPU, estimate from parameter size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        return param_size / 1024 / 1024


def benchmark_speed(model, device, config, num_samples=100, warmup=10):
    """Benchmark inference speed"""
    print(f"Warming up with {warmup} samples...")
    
    # Get betas for diffusion
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            x = torch.randn(1, config.data.channels, config.data.image_size, 
                          config.data.image_size, device=device)
            timesteps = 100  # Use 100 timesteps for benchmarking
            skip = len(betas) // timesteps
            seq = range(0, len(betas), skip)
            _ = generalized_steps(x, seq, model, betas, eta=0.0)
    
    # Benchmark
    print(f"Benchmarking with {num_samples} samples...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            x = torch.randn(1, config.data.channels, config.data.image_size,
                          config.data.image_size, device=device)
            _ = generalized_steps(x, seq, model, betas, eta=0.0)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    
    return avg_time, total_time


def generate_samples_for_fid(model, device, config, num_samples, output_dir, modulate=False):
    """Generate samples for FID calculation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get betas
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    
    timesteps = 100
    skip = len(betas) // timesteps
    seq = range(0, len(betas), skip)
    
    print(f"Generating {num_samples} samples for FID calculation...")
    img_id = 0
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating FID samples"):
            x = torch.randn(1, config.data.channels, config.data.image_size,
                          config.data.image_size, device=device)
            
            # Reset modulation cache if needed
            if modulate and hasattr(model, 'reset_cache'):
                model.reset_cache()
            
            # Generate sample
            x_gen = generalized_steps(x, seq, model, betas, eta=0.0)
            x_gen = x_gen[0][-1]  # Get final denoised image
            
            # Inverse transform
            x_gen = inverse_data_transform(config, x_gen)
            
            # Save image
            tvu.save_image(x_gen, os.path.join(output_dir, f"{img_id}.png"))
            img_id += 1


def calculate_fid(real_path, fake_path):
    """Calculate FID score using torch-fidelity"""
    try:
        from torch_fidelity import calculate_metrics
        
        print(f"Calculating FID between {real_path} and {fake_path}")
        metrics = calculate_metrics(
            input1=fake_path,
            input2=real_path,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
        )
        return metrics['frechet_inception_distance']
    except ImportError:
        print("torch-fidelity not installed. Install with: pip install torch-fidelity")
        return None
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None


def dict2namespace(config):
    """Convert dict to namespace"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    parser = argparse.ArgumentParser(description="Benchmark INT4 vs Simulated vs FP32 quantization")
    
    # Model and data
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_dir", type=str, default="models/", help="Path to model directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    # Checkpoints
    parser.add_argument("--sim_ckpt", type=str, default=None, 
                       help="Checkpoint for simulated quantization")
    parser.add_argument("--int4_ckpt", type=str, default=None,
                       help="Checkpoint for INT4 quantization")
    
    # Calibration data
    parser.add_argument("--cali_data_path", type=str, default=None,
                       help="Path to calibration data")
    
    # Benchmark settings
    parser.add_argument("--speed_samples", type=int, default=100,
                       help="Number of samples for speed benchmark")
    parser.add_argument("--fid_samples", type=int, default=1000,
                       help="Number of samples for FID calculation")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Warmup iterations for speed benchmark")
    
    # FID settings
    parser.add_argument("--real_data_path", type=str, default=None,
                       help="Path to real data for FID calculation")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--skip_fid", action="store_true",
                       help="Skip FID calculation (only speed benchmark)")
    
    # Options
    parser.add_argument("--modulate", action="store_true",
                       help="Use modulation for quantized models")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--skip_fp32", action="store_true",
                       help="Skip FP32 baseline benchmark")
    parser.add_argument("--skip_sim", action="store_true",
                       help="Skip simulated quantization benchmark")
    parser.add_argument("--skip_int4", action="store_true",
                       help="Skip INT4 quantization benchmark")
    
    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config.split_shortcut = False  # Disable split for benchmarking
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load calibration data if provided
    cali_data = None
    if args.cali_data_path and os.path.exists(args.cali_data_path):
        print(f"Loading calibration data from {args.cali_data_path}")
        sample_data = torch.load(args.cali_data_path, weights_only=True)
        # Use small subset for initialization
        if isinstance(sample_data, dict):
            cali_xs = sample_data['xs'][0][:64] if 'xs' in sample_data else None
            cali_ts = sample_data['ts'][0][:64] if 'ts' in sample_data else None
            if cali_xs is not None:
                cali_data = (cali_xs, cali_ts)
    
    if cali_data is None:
        print("Creating dummy calibration data...")
        cali_data = (
            torch.randn(64, config.data.channels, config.data.image_size, config.data.image_size),
            torch.randint(0, 1000, (64,))
        )
    
    # Results dictionary
    results = {
        'fp32': {'speed': None, 'memory': None, 'fid': None},
        'simulated': {'speed': None, 'memory': None, 'fid': None},
        'int4': {'speed': None, 'memory': None, 'fid': None},
    }
    
    # ============= FP32 Baseline (Run First) =============
    if not args.skip_fp32:
        print("\n" + "="*80)
        print("BENCHMARK 1/3: FP32 (Original Model)")
        print("="*80)
        
        model_fp32 = load_fp32_model(config, args.model_dir, device)
        
        # Memory
        print("\nMeasuring memory usage...")
        memory = get_model_memory(model_fp32, device)
        results['fp32']['memory'] = memory
        print(f"Memory usage: {memory:.2f} MB")
        
        # Speed
        print("\nMeasuring inference speed...")
        avg_time, total_time = benchmark_speed(model_fp32, device, config, 
                                               args.speed_samples, args.warmup)
        results['fp32']['speed'] = avg_time
        print(f"Average time per sample: {avg_time:.4f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        
        # FID
        if not args.skip_fid and args.real_data_path:
            print("\nGenerating samples for FID...")
            fp32_dir = os.path.join(args.output_dir, "fp32_samples")
            generate_samples_for_fid(model_fp32, device, config, args.fid_samples, 
                                    fp32_dir, modulate=False)
            fid = calculate_fid(args.real_data_path, fp32_dir)
            results['fp32']['fid'] = fid
            if fid:
                print(f"FID score: {fid:.4f}")
        
        del model_fp32
        torch.cuda.empty_cache()
    
    # ============= INT4 Quantization (Run Second - True 4-bit) =============
    if not args.skip_int4:
        print("\n" + "="*80)
        print("BENCHMARK 2/3: INT4 Quantization (True 4-bit Storage)")
        print("NOTE: Current implementation optimizes for MEMORY, not SPEED")
        print("      Weights are dequantized to float32 on each forward pass")
        print("      Expect slower inference but ~70% memory reduction")
        print("="*80)
        
        model_int4 = load_int4_model(config, args.model_dir, args.int4_ckpt,
                                     cali_data, device, args.modulate)
        
        # Memory
        print("\nMeasuring memory usage...")
        memory = get_model_memory(model_int4, device)
        results['int4']['memory'] = memory
        print(f"Memory usage: {memory:.2f} MB")
        
        # Speed
        print("\nMeasuring inference speed...")
        avg_time, total_time = benchmark_speed(model_int4, device, config,
                                               args.speed_samples, args.warmup)
        results['int4']['speed'] = avg_time
        print(f"Average time per sample: {avg_time:.4f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        
        # FID
        if not args.skip_fid and args.real_data_path:
            print("\nGenerating samples for FID...")
            int4_dir = os.path.join(args.output_dir, "int4_samples")
            generate_samples_for_fid(model_int4, device, config, args.fid_samples,
                                    int4_dir, modulate=args.modulate)
            fid = calculate_fid(args.real_data_path, int4_dir)
            results['int4']['fid'] = fid
            if fid:
                print(f"FID score: {fid:.4f}")
        
        del model_int4
        torch.cuda.empty_cache()
    
    # ============= Simulated Quantization (Run Third - Fake Quantization) =============
    if not args.skip_sim:
        print("\n" + "="*80)
        print("BENCHMARK 3/3: Simulated Quantization (Fake Quantization, Float32)")
        print("="*80)
        
        model_sim = load_simulated_quant_model(config, args.model_dir, args.sim_ckpt,
                                               cali_data, device, args.modulate)
        
        # Memory
        print("\nMeasuring memory usage...")
        memory = get_model_memory(model_sim, device)
        results['simulated']['memory'] = memory
        print(f"Memory usage: {memory:.2f} MB")
        
        # Speed
        print("\nMeasuring inference speed...")
        avg_time, total_time = benchmark_speed(model_sim, device, config,
                                               args.speed_samples, args.warmup)
        results['simulated']['speed'] = avg_time
        print(f"Average time per sample: {avg_time:.4f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        
        # FID
        if not args.skip_fid and args.real_data_path:
            print("\nGenerating samples for FID...")
            sim_dir = os.path.join(args.output_dir, "simulated_samples")
            generate_samples_for_fid(model_sim, device, config, args.fid_samples,
                                    sim_dir, modulate=args.modulate)
            fid = calculate_fid(args.real_data_path, sim_dir)
            results['simulated']['fid'] = fid
            if fid:
                print(f"FID score: {fid:.4f}")
        
        del model_sim
        torch.cuda.empty_cache()
    
    # ============= Summary =============
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print("\nInterpretation Guide:")
    print("  - Speedup >1.0x = Faster than FP32 (lower is better for 's/img')")
    print("  - Speedup <1.0x = Slower than FP32 (but may save memory)")
    print("  - Memory Ratio: Lower is better (e.g., 0.33x = 67% reduction)")
    
    # Print table header
    print(f"\n{'Model':<20} {'Speed (s/img)':<15} {'Speedup':<12} {'Memory (MB)':<15} {'Memory Ratio':<15} {'FID':<10}")
    print("-" * 97)
    
    # FP32 baseline
    baseline_speed = results['fp32']['speed'] if results['fp32']['speed'] else 1.0
    baseline_memory = results['fp32']['memory'] if results['fp32']['memory'] else 1.0
    
    for model_name in ['fp32', 'simulated', 'int4']:
        if results[model_name]['speed'] is None:
            continue
        
        speed = results[model_name]['speed']
        speedup = baseline_speed / speed if speed else 0
        memory = results[model_name]['memory']
        mem_ratio = memory / baseline_memory if baseline_memory else 0
        fid = results[model_name]['fid']
        
        fid_str = f"{fid:.4f}" if fid else "N/A"
        print(f"{model_name.upper():<20} {speed:<15.4f} {speedup:<12.2f}x {memory:<15.2f} {mem_ratio:<15.2f}x {fid_str:<10}")
    
    print("-" * 97)
    
    # Calculate improvements with proper interpretation
    print("\nKEY RESULTS:")
    print("-" * 50)
    
    if results['int4']['speed'] and results['fp32']['speed']:
        speedup = results['fp32']['speed'] / results['int4']['speed']
        if speedup < 1.0:
            print(f"INT4 vs FP32: {1/speedup:.2f}x SLOWER (expected due to dequantization overhead)")
        else:
            print(f"INT4 vs FP32: {speedup:.2f}x speedup")
    
    if results['int4']['memory'] and results['fp32']['memory']:
        mem_reduction = (1 - results['int4']['memory'] / results['fp32']['memory']) * 100
        print(f"INT4 vs FP32: {mem_reduction:.1f}% memory reduction ← PRIMARY BENEFIT")
    
    if results['simulated']['speed'] and results['int4']['speed']:
        speedup = results['simulated']['speed'] / results['int4']['speed']
        if speedup < 1.0:
            print(f"INT4 vs Simulated: {1/speedup:.2f}x slower")
        else:
            print(f"INT4 vs Simulated: {speedup:.2f}x speedup")
    
    print("\nNOTE: INT4 trades speed for memory. Current implementation:")
    print("  - Weights stored in true 4-bit format (memory savings)")
    print("  - Dequantized to float32 on each forward pass (speed overhead)")
    print("  - For speed improvements, need optimized CUDA INT4 kernels")
    print("-" * 50)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "benchmark_results.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Speed samples: {args.speed_samples}\n")
        f.write(f"FID samples: {args.fid_samples if not args.skip_fid else 'Skipped'}\n")
        f.write(f"Modulation: {args.modulate}\n\n")
        
        f.write(f"{'Model':<20} {'Speed (s/img)':<15} {'Speedup':<12} {'Memory (MB)':<15} {'Memory Ratio':<15} {'FID':<10}\n")
        f.write("-" * 97 + "\n")
        
        for model_name in ['fp32', 'simulated', 'int4']:
            if results[model_name]['speed'] is None:
                continue
            speed = results[model_name]['speed']
            speedup = baseline_speed / speed if speed else 0
            memory = results[model_name]['memory']
            mem_ratio = memory / baseline_memory if baseline_memory else 0
            fid = results[model_name]['fid']
            fid_str = f"{fid:.4f}" if fid else "N/A"
            f.write(f"{model_name.upper():<20} {speed:<15.4f} {speedup:<12.2f}x {memory:<15.2f} {mem_ratio:<15.2f}x {fid_str:<10}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
