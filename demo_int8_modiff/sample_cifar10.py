"""
CIFAR-10 Sampling Script with MoDiff INT8 Quantization

This script generates samples from a quantized DDIM model on CIFAR-10
using MoDiff error-compensated modulation for INT8 inference.

Usage:
    # Quick test (100 samples)
    python -m demo_int8_modiff.sample_cifar10 --num_samples 100 --modulate
    
    # Full FID evaluation (50k samples)
    python -m demo_int8_modiff.sample_cifar10 --num_samples 50000 --modulate
    
    # Without MoDiff (for comparison)
    python -m demo_int8_modiff.sample_cifar10 --num_samples 1000 --no_modulate

Paper Settings:
    - 100 DDIM steps
    - eta = 0 (deterministic)
    - 50,000 samples for FID evaluation
    - W8A8 quantization with MoDiff → FID ~4.1
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import time

import torch
import yaml
from tqdm import tqdm
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddim.models.diffusion import Model
from demo_int8_modiff.quant_model_modiff import QuantModelMoDiff
from demo_int8_modiff.ddim_sampler import DDIMSamplerMoDiff, inverse_data_transform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def dict2namespace(config: dict):
    """Convert dictionary to namespace."""
    import argparse
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def load_model(config_path: str, ckpt_path: str, device: str = 'cuda'):
    """
    Load pretrained DDIM model for CIFAR-10.
    
    Args:
        config_path: Path to config YAML
        ckpt_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    config = load_config(config_path)
    config = dict2namespace(config)
    
    # Add missing config attributes that the model might expect
    if not hasattr(config, 'split_shortcut'):
        config.split_shortcut = False
    if not hasattr(config.model, 'split_shortcut'):
        config.model.split_shortcut = False
    
    # Create model
    model = Model(config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'ema' in ckpt:
        state_dict = ckpt['ema']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    return model, config


def save_samples(samples: list, output_dir: str, prefix: str = "sample"):
    """
    Save generated samples as PNG images.
    
    Args:
        samples: List of sample tensors
        output_dir: Output directory
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, sample in enumerate(tqdm(samples, desc="Saving samples")):
        # Transform to [0, 1] range
        img = inverse_data_transform(sample)
        
        # Convert to PIL Image
        if img.dim() == 4:
            img = img[0]
        img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        
        if img.shape[2] == 3:
            pil_img = Image.fromarray(img, mode='RGB')
        else:
            pil_img = Image.fromarray(img[:, :, 0], mode='L')
        
        pil_img.save(os.path.join(output_dir, f"{prefix}_{i:05d}.png"))
    
    logger.info(f"Saved {len(samples)} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 samples with MoDiff INT8")
    
    # Model paths
    parser.add_argument("--config", type=str, default="configs/cifar10.yml",
                        help="Path to config file")
    parser.add_argument("--ckpt", type=str, 
                        default="models/ema_diffusion_cifar10_model/model-790000.ckpt",
                        help="Path to model checkpoint")
    
    # Quantization settings
    parser.add_argument("--weight_bit", type=int, default=8,
                        help="Weight quantization bits")
    parser.add_argument("--act_bit", type=int, default=8,
                        help="Activation quantization bits")
    parser.add_argument("--modulate", action="store_true", default=True,
                        help="Enable MoDiff modulation (default: True)")
    parser.add_argument("--no_modulate", action="store_true",
                        help="Disable MoDiff modulation")
    parser.add_argument("--native_int8", action="store_true", default=True,
                        help="Use native INT8 operations")
    parser.add_argument("--no_quant", action="store_true",
                        help="Disable quantization (FP32 baseline)")
    
    # Sampling settings
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=100,
                        help="Number of DDIM steps (paper: 100)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta (0=deterministic, paper: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="output/int8_modiff_samples",
                        help="Output directory for samples")
    parser.add_argument("--save_samples", action="store_true", default=True,
                        help="Save generated samples as images")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Handle modulate flag
    if args.no_modulate:
        args.modulate = False
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Print settings
    logger.info("="*60)
    logger.info("MoDiff INT8 Sampling for CIFAR-10")
    logger.info("="*60)
    logger.info(f"Weight bits: {args.weight_bit}")
    logger.info(f"Activation bits: {args.act_bit}")
    logger.info(f"MoDiff modulation: {args.modulate}")
    logger.info(f"DDIM steps: {args.ddim_steps}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*60)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, config = load_model(args.config, args.ckpt, device)
    
    # Create quantized model
    if not args.no_quant:
        logger.info("Creating quantized model with MoDiff...")
        qmodel = QuantModelMoDiff(
            model,
            weight_bits=args.weight_bit,
            act_bits=args.act_bit,
            modulate=args.modulate,
            use_native_int8=args.native_int8,
        )
        
        # Calibrate weights
        qmodel.calibrate_weights()
        
        # Enable quantization
        qmodel.set_quant_state(weight_quant=True, act_quant=True)
        
        qmodel.print_quant_summary()
        sample_model = qmodel
    else:
        logger.info("Running FP32 baseline (no quantization)")
        sample_model = model
    
    # Create sampler
    sampler = DDIMSamplerMoDiff(
        sample_model,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_timesteps=config.diffusion.num_diffusion_timesteps,
        device=device,
    )
    
    # Generate samples
    logger.info(f"\nGenerating {args.num_samples} samples...")
    start_time = time.time()
    
    samples = sampler.sample(
        num_samples=args.num_samples,
        image_size=config.data.image_size,
        num_channels=config.data.channels,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        seed=args.seed,
        verbose=True,
    )
    
    total_time = time.time() - start_time
    time_per_sample = total_time / args.num_samples
    
    logger.info(f"\nGeneration complete!")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Time per sample: {time_per_sample*1000:.2f}ms")
    logger.info(f"Throughput: {args.num_samples/total_time:.2f} samples/s")
    
    # Save samples
    if args.save_samples:
        # Create output directory with settings
        quant_str = f"W{args.weight_bit}A{args.act_bit}" if not args.no_quant else "FP32"
        modiff_str = "modiff" if args.modulate else "baseline"
        output_subdir = f"{quant_str}_{modiff_str}_steps{args.ddim_steps}"
        full_output_dir = os.path.join(args.output_dir, output_subdir)
        
        save_samples(samples, full_output_dir)
        
        # Save metadata
        import json
        metadata = {
            'num_samples': args.num_samples,
            'weight_bits': args.weight_bit,
            'act_bits': args.act_bit,
            'modulate': args.modulate,
            'ddim_steps': args.ddim_steps,
            'eta': args.eta,
            'seed': args.seed,
            'total_time_seconds': total_time,
            'time_per_sample_ms': time_per_sample * 1000,
            'throughput_samples_per_sec': args.num_samples / total_time,
        }
        
        with open(os.path.join(full_output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {full_output_dir}/metadata.json")
    
    return samples


if __name__ == "__main__":
    main()
