#!/usr/bin/env python3
"""
INT8 Scale Extractor for MoDiff - MSE-based Scale Search (Following INT4's Working Methodology)

This script extracts INT8 quantization scales using the same MSE-based approach that works
for INT4, adapted for INT8's 256 quantization levels.

Key Fix: Use MSE-based scale search (like INT4) instead of simple max-abs/127.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict
import json

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from ddim.models.diffusion import Model
from ddim.functions.ckpt_util import get_ckpt_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class INT8ScaleComputerMSE:
    """
    Compute INT8 quantization scales using MSE-based search.
    
    This follows the same methodology as INT4 but with 256 levels instead of 16.
    """
    
    def __init__(self, symmetric: bool = True, num_candidates: int = 100):
        self.n_bits = 8
        self.n_levels = 256
        self.symmetric = symmetric
        self.num_candidates = num_candidates
        
        if symmetric:
            self.q_min = -128
            self.q_max = 127
        else:
            self.q_min = 0
            self.q_max = 255
            
    def compute_scale(self, x: np.ndarray) -> tuple:
        """
        Compute optimal INT8 scale using MSE search.
        
        Args:
            x: Input tensor to quantize (numpy array)
            
        Returns:
            (scale, zero_point) tuple
        """
        x_flat = x.flatten().astype(np.float32)
        
        if self.symmetric:
            x_absmax = max(abs(x_flat.min()), abs(x_flat.max()))
            
            if x_absmax < 1e-8:
                return 1.0, 0.0
            
            # Base scale from max method
            base_scale = x_absmax / self.q_max
            
            # Search from 0.5x to 1.2x of base scale
            candidates = np.linspace(0.5, 1.2, self.num_candidates) * base_scale
            
            best_mse = float('inf')
            best_scale = base_scale
            
            for scale in candidates:
                # Quantize and dequantize
                x_q = np.clip(np.round(x_flat / scale), self.q_min, self.q_max)
                x_dq = x_q * scale
                mse = np.mean((x_flat - x_dq) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_scale = scale
                    
            return float(best_scale), 0.0
        else:
            # Asymmetric quantization
            x_min, x_max = x_flat.min(), x_flat.max()
            
            if x_max - x_min < 1e-8:
                return 1.0, float(x_min)
            
            base_scale = (x_max - x_min) / (self.n_levels - 1)
            
            best_mse = float('inf')
            best_scale = base_scale
            best_zp = x_min
            
            # Grid search
            for d_mult in np.linspace(0.8, 1.2, 20):
                scale = base_scale * d_mult
                for zp_shift in np.linspace(-0.1, 0.1, 10):
                    zp = x_min + zp_shift * (x_max - x_min)
                    
                    x_q = np.clip(np.round((x_flat - zp) / scale), 0, self.n_levels - 1)
                    x_dq = x_q * scale + zp
                    mse = np.mean((x_flat - x_dq) ** 2)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_scale = scale
                        best_zp = zp
                        
            return float(best_scale), float(best_zp)


def dict_to_namespace(data):
    namespace = argparse.Namespace()
    for key, value in data.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def extract_int8_scales_mse(
    model: nn.Module,
    calib_dir: Path,
    num_samples: int = 16,
    symmetric: bool = True,
) -> Dict[str, Dict]:
    """
    Extract INT8 scales from model using MSE-based approach (like INT4).
    
    Args:
        model: PyTorch model
        calib_dir: Directory with calibration .npz files
        num_samples: Number of calibration samples to use
        symmetric: Use symmetric quantization
        
    Returns:
        Dictionary of layer scales
    """
    scale_computer = INT8ScaleComputerMSE(symmetric=symmetric)
    
    # Storage for collected activations
    activation_stats: Dict[str, Dict] = {}
    hooks = []
    
    def make_hook(name: str):
        def hook(module, inputs, output):
            # Get input activation
            if isinstance(inputs, tuple) and len(inputs) > 0:
                x = inputs[0]
            else:
                x = inputs
            
            if x is None or not isinstance(x, torch.Tensor):
                return
            
            # Convert to numpy
            x_np = x.detach().cpu().float().numpy()
            
            # Update running statistics
            if name not in activation_stats:
                activation_stats[name] = {
                    'min': x_np.min(),
                    'max': x_np.max(),
                    'absmax': np.abs(x_np).max(),
                    'samples': [x_np.flatten()[:5000]],  # Keep subset for MSE search
                    'shape': list(x.shape),
                    'count': 1,
                }
            else:
                stats = activation_stats[name]
                stats['min'] = min(stats['min'], x_np.min())
                stats['max'] = max(stats['max'], x_np.max())
                stats['absmax'] = max(stats['absmax'], np.abs(x_np).max())
                if len(stats['samples']) < 10:  # Keep up to 10 sample batches
                    stats['samples'].append(x_np.flatten()[:5000])
                stats['count'] += 1
                
        return hook
    
    # Register hooks on Conv2d and Linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    logger.info(f"Registered hooks on {len(hooks)} layers")
    
    # Load calibration data
    samples = sorted(calib_dir.glob("sample_*.npz"))[:num_samples]
    logger.info(f"Loading {len(samples)} calibration samples...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    
    # Run calibration
    with torch.no_grad():
        for sample_path in samples:
            data = np.load(sample_path)
            latent = torch.from_numpy(data['latent']).float().to(device)
            timesteps = torch.from_numpy(data['timesteps']).to(device)
            
            try:
                model(latent, timesteps)
            except Exception as e:
                logger.warning(f"Calibration sample {sample_path.name} failed: {e}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    logger.info(f"Collected statistics for {len(activation_stats)} layers")
    
    # Compute scales using MSE search
    scales = {}
    logger.info("Computing INT8 scales using MSE search...")
    
    for name, stats in activation_stats.items():
        # Combine all samples for MSE search
        all_samples = np.concatenate(stats['samples'])
        
        # Compute scale using MSE search
        scale, zero_point = scale_computer.compute_scale(all_samples)
        
        scales[name] = {
            'act_scale': float(scale),
            'act_zero_point': float(zero_point),
            'min': float(stats['min']),
            'max': float(stats['max']),
            'absmax': float(stats['absmax']),
            'shape': stats['shape'],
            'n_bits': 8,
            'symmetric': symmetric,
            'scale_method': 'mse',
        }
    
    logger.info(f"Computed MSE-based INT8 scales for {len(scales)} layers")
    return scales


def main():
    parser = argparse.ArgumentParser(description="Extract INT8 scales using MSE method (like INT4)")
    parser.add_argument(
        "--config",
        default="../configs/cifar10.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--calib-dir",
        default="../trt/calib",
        help="Calibration data directory",
    )
    parser.add_argument(
        "--output-dir",
        default="../trt/export/extracted_scales_int8_mse",
        help="Output directory for scales",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of calibration samples",
    )
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    calib_dir = Path(args.calib_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Load config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    if not hasattr(config, 'split_shortcut'):
        setattr(config, 'split_shortcut', False)
    
    # Load model
    logger.info("Loading model...")
    model = Model(config)
    ckpt_path = get_ckpt_path('ema_cifar10', root=None)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=True)
    
    # Extract scales
    scales = extract_int8_scales_mse(
        model=model,
        calib_dir=calib_dir,
        num_samples=args.num_samples,
        symmetric=True,
    )
    
    # Save scales
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_dir / "model_scales_int8_mse.json"
    with open(json_path, 'w') as f:
        json.dump(scales, f, indent=2)
    logger.info(f"Saved JSON scales to {json_path}")
    
    # Save as NPZ (for build_engine.py compatibility)
    scales_dict = {}
    for idx, (name, scale_info) in enumerate(scales.items()):
        key = f"layer_{idx}_{name}"
        scales_dict[key] = np.array({'act_scale': scale_info['act_scale']}, dtype=object)
    
    npz_path = output_dir / "model_scales.npz"
    np.savez_compressed(npz_path, **scales_dict)
    logger.info(f"Saved NPZ scales to {npz_path}")
    
    # Save metadata
    metadata = {
        'num_layers': len(scales),
        'num_parameters': len(scales_dict),
        'layer_names': list(scales.keys()),
        'source': 'int8_mse_based_extraction',
        'scale_method': 'mse',
        'n_bits': 8,
        'symmetric': True,
    }
    meta_path = output_dir / "scales_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INT8 MSE Scale Extraction Complete!")
    logger.info("="*60)
    logger.info(f"Extracted scales for {len(scales)} layers")
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nSample scales:")
    for i, (name, info) in enumerate(list(scales.items())[:5]):
        logger.info(f"  {name}: scale={info['act_scale']:.6f}, min={info['min']:.4f}, max={info['max']:.4f}")
    
    # Compare with INT4 scales if available
    int4_json = Path("../trt/int4_output/scales/model_scales_int4.json")
    if int4_json.exists():
        with open(int4_json) as f:
            int4_scales = json.load(f)
        
        logger.info("\n" + "="*60)
        logger.info("INT8 vs INT4 Scale Comparison")
        logger.info("="*60)
        
        # Find common layers
        int8_keys = set(scales.keys())
        int4_keys = set(int4_scales.keys())
        common = int8_keys & int4_keys
        
        if common:
            logger.info(f"Common layers: {len(common)}")
            for name in list(common)[:5]:
                int8_scale = scales[name]['act_scale']
                int4_scale = int4_scales[name]['act_scale']
                ratio = int8_scale / int4_scale if int4_scale > 0 else float('inf')
                logger.info(f"  {name}:")
                logger.info(f"    INT8: {int8_scale:.6f}, INT4: {int4_scale:.6f}, Ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
