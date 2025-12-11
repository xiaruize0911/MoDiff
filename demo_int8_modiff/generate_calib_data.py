"""
Generate Calibration Data Script

Generates and saves calibration data for Q-Diff/MoDiff quantization.

Usage:
    python -m demo_int8_modiff.generate_calib_data \
        --config configs/cifar10.yml \
        --ckpt models/ema_diffusion_cifar10_model/model-790000.ckpt \
        --num_samples 1024 \
        --output calibration/cifar10_calib_data.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from ddim.models.diffusion import Model
from demo_int8_modiff.calibration import generate_calibration_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def dict2namespace(config: dict):
    import argparse
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def main():
    parser = argparse.ArgumentParser(description="Generate calibration data")
    
    parser.add_argument("--config", type=str, default="configs/cifar10.yml")
    parser.add_argument("--ckpt", type=str, 
                        default="models/ema_diffusion_cifar10_model/model-790000.ckpt")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_calib_steps", type=int, default=8)
    parser.add_argument("--output", type=str, default="calibration/cifar10_calib_data.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config = dict2namespace(config)
    
    # Load model
    logger.info(f"Loading model from {args.ckpt}")
    model = Model(config)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device)
    
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
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    
    # Generate calibration data
    logger.info(f"Generating {args.num_samples} calibration samples...")
    
    calib_data = generate_calibration_data(
        model=model,
        num_samples=args.num_samples,
        image_size=config.data.image_size,
        num_channels=config.data.channels,
        num_timesteps=config.diffusion.num_diffusion_timesteps,
        num_calib_steps=args.num_calib_steps,
        device=device,
        seed=args.seed,
    )
    
    # Save calibration data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(calib_data, args.output)
    
    logger.info(f"Saved calibration data to {args.output}")
    logger.info(f"Data shapes:")
    logger.info(f"  xs: {calib_data['xs'].shape}")
    logger.info(f"  ts: {calib_data['ts'].shape}")
    logger.info(f"  xs_prev: {calib_data['xs_prev'].shape}")


if __name__ == "__main__":
    main()
