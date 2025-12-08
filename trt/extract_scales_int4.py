#!/usr/bin/env python3
"""
INT4 Scale Extractor for MoDiff - Extract Scales from Trained Model

This script extracts INT4 quantization scales from a trained MoDiff model
and saves them for use with TensorRT INT4 engine building.

The extraction follows the paper's methodology:
1. MSE-based scale search for optimal quantization
2. Per-layer scale extraction for Conv2d/Linear layers
3. MoDiff modulation-aware scale computation

Usage:
    python extract_scales_int4.py --config configs/cifar10.yml --ckpt model.pth --output scales_int4.npz

Paper: "Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from int4_calibrator import INT4ScaleComputer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract INT4 scales from trained MoDiff model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="../configs/cifar10.yml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--calib-dir",
        default="../calibration",
        help="Directory with calibration data",
    )
    parser.add_argument(
        "--output",
        default="extracted_scales_int4",
        help="Output directory for scales",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=4,
        choices=[3, 4],
        help="Number of quantization bits",
    )
    parser.add_argument(
        "--scale-method",
        default="mse",
        choices=["mse", "max", "minmax"],
        help="Scale computation method",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of calibration samples to use",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use pretrained CIFAR10 model",
    )
    return parser.parse_args()


class INT4ScaleExtractor:
    """
    Extract INT4 quantization scales from a PyTorch model.
    
    This class hooks into the model's forward pass to collect activation
    statistics and compute optimal INT4 scales using MSE search.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_bits: int = 4,
        symmetric: bool = True,
        scale_method: str = 'mse',
    ):
        self.model = model
        self.scale_computer = INT4ScaleComputer(
            n_bits=n_bits,
            symmetric=symmetric,
            scale_method=scale_method,
        )
        
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.scale_method = scale_method
        
        # Storage for collected activations
        self.activation_stats: Dict[str, Dict] = {}
        self.hooks = []
        
    def _make_hook(self, name: str):
        """Create a forward hook to collect activation statistics."""
        def hook(module, input, output):
            # Get input activation
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input
            
            if x is None or not isinstance(x, torch.Tensor):
                return
            
            # Convert to numpy for statistics
            x_np = x.detach().cpu().float().numpy()
            
            # Update running statistics
            if name not in self.activation_stats:
                self.activation_stats[name] = {
                    'min': x_np.min(),
                    'max': x_np.max(),
                    'absmax': np.abs(x_np).max(),
                    'samples': [x_np.flatten()[:1000]],  # Keep subset for MSE search
                    'shape': list(x.shape),
                    'count': 1,
                }
            else:
                stats = self.activation_stats[name]
                stats['min'] = min(stats['min'], x_np.min())
                stats['max'] = max(stats['max'], x_np.max())
                stats['absmax'] = max(stats['absmax'], np.abs(x_np).max())
                if len(stats['samples']) < 10:  # Keep up to 10 sample batches
                    stats['samples'].append(x_np.flatten()[:1000])
                stats['count'] += 1
                
        return hook
    
    def register_hooks(self) -> None:
        """Register forward hooks on Conv2d and Linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
                
        logger.info(f"Registered hooks on {len(self.hooks)} layers")
        
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def run_calibration(
        self,
        calib_data: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int = 8,
    ) -> None:
        """
        Run calibration data through the model to collect statistics.
        
        Args:
            calib_data: Calibration input tensor (N, C, H, W)
            timesteps: Timesteps tensor (N,)
            batch_size: Batch size for processing
        """
        self.model.eval()
        
        num_samples = min(len(calib_data), len(timesteps))
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        logger.info(f"Running calibration with {num_samples} samples...")
        
        with torch.no_grad():
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, num_samples)
                
                batch_x = calib_data[start:end]
                batch_t = timesteps[start:end]
                
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_t = batch_t.cuda()
                
                try:
                    self.model(batch_x, batch_t)
                except Exception as e:
                    logger.warning(f"Calibration batch {i} failed: {e}")
                    
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{num_batches} batches")
                    
        logger.info(f"Collected statistics for {len(self.activation_stats)} layers")
        
    def compute_scales(self) -> Dict[str, Dict]:
        """
        Compute INT4 scales from collected statistics.
        
        Returns:
            Dictionary mapping layer names to scale info
        """
        scales = {}
        
        logger.info(f"Computing INT{self.n_bits} scales using {self.scale_method} method...")
        
        for name, stats in self.activation_stats.items():
            # Combine all samples for MSE search
            all_samples = np.concatenate(stats['samples'])
            
            # Compute scale using MSE search
            scale, zero_point = self.scale_computer.compute_scale(all_samples)
            
            scales[name] = {
                'act_scale': float(scale),
                'act_zero_point': float(zero_point),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'absmax': float(stats['absmax']),
                'shape': stats['shape'],
                'n_bits': self.n_bits,
                'symmetric': self.symmetric,
                'scale_method': self.scale_method,
            }
            
        logger.info(f"Computed scales for {len(scales)} layers")
        return scales
    
    def extract(
        self,
        calib_data: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int = 8,
    ) -> Dict[str, Dict]:
        """
        Full extraction pipeline: register hooks, run calibration, compute scales.
        
        Args:
            calib_data: Calibration input tensor
            timesteps: Timesteps tensor
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of layer scales
        """
        try:
            self.register_hooks()
            self.run_calibration(calib_data, timesteps, batch_size)
            scales = self.compute_scales()
            return scales
        finally:
            self.remove_hooks()


def load_calibration_data(calib_dir: Path, num_samples: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load calibration data from .npz files."""
    samples = sorted(calib_dir.glob("*.npz"))
    
    if not samples:
        raise FileNotFoundError(f"No calibration samples in {calib_dir}")
    
    all_latents = []
    all_timesteps = []
    
    for sample_path in samples[:num_samples]:
        with np.load(sample_path) as data:
            all_latents.append(data['latent'])
            all_timesteps.append(data['timesteps'])
    
    latents = torch.from_numpy(np.concatenate(all_latents, axis=0))
    timesteps = torch.from_numpy(np.concatenate(all_timesteps, axis=0))
    
    return latents, timesteps


def load_model(config_path: Path, ckpt_path: Optional[Path] = None, use_pretrained: bool = False):
    """Load the diffusion model."""
    import yaml
    from types import SimpleNamespace
    
    def dict_to_namespace(d):
        """Recursively convert dict to SimpleNamespace for attribute access."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(i) for i in d]
        return d
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace for attribute access (Model expects config.model.ch etc.)
    config = dict_to_namespace(config_dict)
    
    # Try to import the model
    try:
        from ddim.models.diffusion import Model as DiffusionModel
        
        # Create model with config namespace
        model = DiffusionModel(config)
        
        if use_pretrained:
            logger.info("Loading pretrained weights...")
            # Try to load pretrained weights
            pretrained_paths = [
                Path.home() / "modiff_trt" / "models" / "cifar10_ema.pth",
                Path(__file__).parent.parent / "models" / "cifar10" / "model.pth",
                Path(__file__).parent.parent / "models" / "diffusion_models_converted" / "ema_diffusion_cifar10_model" / "model-790000.ckpt",
            ]
            for p in pretrained_paths:
                if p.exists():
                    state_dict = torch.load(p, map_location='cpu')
                    if isinstance(state_dict, dict) and 'model' in state_dict:
                        state_dict = state_dict['model']
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded pretrained weights from {p}")
                    break
        elif ckpt_path and ckpt_path.exists():
            logger.info(f"Loading checkpoint from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
        
        return model
        
    except ImportError as e:
        logger.error(f"Failed to import model: {e}")
        raise


def save_scales(
    scales: Dict[str, Dict],
    output_dir: Path,
    n_bits: int,
) -> None:
    """Save extracted scales to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON for human readability
    json_path = output_dir / f"model_scales_int{n_bits}.json"
    with open(json_path, 'w') as f:
        json.dump(scales, f, indent=2)
    logger.info(f"Saved scales to {json_path}")
    
    # Save as NPZ for efficient loading
    npz_path = output_dir / f"model_scales_int{n_bits}.npz"
    # Convert nested dict to flat arrays
    flat_scales = {}
    for layer_name, layer_scales in scales.items():
        for key, value in layer_scales.items():
            flat_key = f"{layer_name}_{key}"
            if isinstance(value, (int, float)):
                flat_scales[flat_key] = np.array(value)
            elif isinstance(value, list):
                flat_scales[flat_key] = np.array(value)
    np.savez(npz_path, **flat_scales)
    logger.info(f"Saved scales to {npz_path}")
    
    # Save summary
    summary_path = output_dir / f"extraction_summary_int{n_bits}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"INT{n_bits} Scale Extraction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total layers: {len(scales)}\n")
        f.write(f"Scale method: {list(scales.values())[0].get('scale_method', 'unknown')}\n")
        f.write(f"Symmetric: {list(scales.values())[0].get('symmetric', 'unknown')}\n\n")
        
        f.write("Layer Statistics:\n")
        f.write("-" * 50 + "\n")
        for name, s in list(scales.items())[:20]:  # First 20 layers
            f.write(f"{name}:\n")
            f.write(f"  scale={s.get('act_scale', 'N/A'):.6f}, ")
            f.write(f"min={s.get('min', 'N/A'):.4f}, ")
            f.write(f"max={s.get('max', 'N/A'):.4f}\n")
        if len(scales) > 20:
            f.write(f"... and {len(scales) - 20} more layers\n")
    logger.info(f"Saved summary to {summary_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"MoDiff INT{args.n_bits} Scale Extractor")
    print("Following paper: Modulated Diffusion (ICML 2025)")
    print("=" * 60)
    
    # Resolve paths
    config_path = Path(args.config).resolve()
    calib_dir = Path(args.calib_dir).resolve()
    output_dir = Path(args.output).resolve()
    ckpt_path = Path(args.ckpt).resolve() if args.ckpt else None
    
    # Load calibration data
    logger.info(f"Loading calibration data from {calib_dir}")
    calib_data, timesteps = load_calibration_data(calib_dir, args.num_samples)
    logger.info(f"Loaded {len(calib_data)} calibration samples")
    
    # Load model
    logger.info(f"Loading model from config {config_path}")
    model = load_model(config_path, ckpt_path, args.use_pretrained)
    
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to CUDA")
    
    # Extract scales
    extractor = INT4ScaleExtractor(
        model,
        n_bits=args.n_bits,
        symmetric=True,
        scale_method=args.scale_method,
    )
    
    scales = extractor.extract(calib_data, timesteps)
    
    # Save scales
    save_scales(scales, output_dir, args.n_bits)
    
    print(f"\n✓ Extracted INT{args.n_bits} scales for {len(scales)} layers")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
