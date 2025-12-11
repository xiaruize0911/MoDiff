"""
Utility Functions

Common utilities for INT8 MoDiff demo.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Utilities
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def dict2namespace(config: dict):
    """Convert dict to namespace recursively."""
    import argparse
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def namespace2dict(namespace) -> dict:
    """Convert namespace to dict recursively."""
    result = {}
    for key, value in vars(namespace).items():
        if hasattr(value, '__dict__'):
            result[key] = namespace2dict(value)
        else:
            result[key] = value
    return result


def save_config(config: Union[dict, object], output_path: str):
    """Save config to YAML file."""
    if not isinstance(config, dict):
        config = namespace2dict(config)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# =============================================================================
# Seed & Reproducibility
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'cuda') -> torch.device:
    """Get appropriate device."""
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# =============================================================================
# Model Utilities
# =============================================================================

def load_model_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    device: str = 'cuda',
    strict: bool = True,
) -> nn.Module:
    """
    Load model checkpoint with proper handling of state dict keys.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'ema' in ckpt:
        state_dict = ckpt['ema']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    # Remove 'module.' prefix from DataParallel/DistributedDataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=strict)
    model.eval()
    
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def freeze_model(model: nn.Module):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


# =============================================================================
# Tensor Utilities
# =============================================================================

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def to_tensor(array: np.ndarray, device: str = 'cuda') -> torch.Tensor:
    """Convert numpy array to tensor."""
    return torch.from_numpy(array).to(device)


def normalize_images(images: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Normalize images to [-1, 1]."""
    return (images - mean) / std


def denormalize_images(images: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1]."""
    return images * std + mean


def images_to_uint8(images: torch.Tensor) -> np.ndarray:
    """
    Convert images tensor to uint8 numpy array.
    Expects input in [-1, 1] range.
    """
    images = denormalize_images(images)
    images = images.clamp(0, 1) * 255
    images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
    return to_numpy(images).astype(np.uint8)


# =============================================================================
# DDIM Schedule Utilities
# =============================================================================

def make_ddim_timesteps(
    ddim_discr_method: str,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
    verbose: bool = False,
) -> np.ndarray:
    """
    Create DDIM timestep schedule.
    
    Args:
        ddim_discr_method: 'uniform' or 'quad'
        num_ddim_timesteps: Number of DDIM steps
        num_ddpm_timesteps: Total DDPM timesteps
        verbose: Print schedule info
    
    Returns:
        Array of timesteps
    """
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(f'Unknown method {ddim_discr_method}')
    
    # Add one to get the final alpha values right (the ones from first scale to data)
    steps_out = ddim_timesteps + 1
    
    if verbose:
        logger.info(f'Selected timesteps for DDIM: {steps_out}')
    
    return steps_out


def make_ddim_sampling_parameters(
    alphacums: np.ndarray,
    ddim_timesteps: np.ndarray,
    eta: float = 0.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DDIM sampling parameters.
    
    Returns:
        sigmas, alphas, alphas_prev
    """
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.concatenate([alphacums[:1], alphacums[ddim_timesteps[:-1]]])
    
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    
    if verbose:
        logger.info(f'Selected alphas for DDIM: {alphas}')
        logger.info(f'For the chosen value of eta = {eta}, '
                   f'which corresponds to sigmas = {sigmas}')
    
    return sigmas, alphas, alphas_prev


def get_beta_schedule(
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
) -> np.ndarray:
    """
    Get beta schedule for diffusion process.
    """
    if beta_schedule == 'linear':
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'quad':
        betas = np.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64
        ) ** 2
    elif beta_schedule == 'cosine':
        s = 0.008
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, num_diffusion_timesteps, steps, dtype=np.float64)
        alphas_cumprod = np.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f'Unknown beta schedule: {beta_schedule}')
    
    return betas


# =============================================================================
# I/O Utilities
# =============================================================================

def save_samples_npz(
    samples: Union[torch.Tensor, np.ndarray],
    output_path: str,
    convert_to_uint8: bool = True,
):
    """Save samples to NPZ file."""
    if isinstance(samples, torch.Tensor):
        if convert_to_uint8:
            samples = images_to_uint8(samples)
        else:
            samples = to_numpy(samples)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, samples=samples)
    logger.info(f"Saved {len(samples)} samples to {output_path}")


def load_samples_npz(npz_path: str) -> np.ndarray:
    """Load samples from NPZ file."""
    data = np.load(npz_path)
    if 'samples' in data:
        return data['samples']
    elif 'arr_0' in data:
        return data['arr_0']
    else:
        raise KeyError(f"No 'samples' or 'arr_0' key in {npz_path}")


def save_samples_images(
    samples: Union[torch.Tensor, np.ndarray],
    output_dir: str,
    prefix: str = 'sample',
    format: str = 'png',
):
    """Save samples as individual images."""
    from PIL import Image
    
    if isinstance(samples, torch.Tensor):
        samples = images_to_uint8(samples)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, sample in enumerate(samples):
        img = Image.fromarray(sample)
        img.save(os.path.join(output_dir, f'{prefix}_{i:05d}.{format}'))


# =============================================================================
# Logging Utilities
# =============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers,
    )


class ProgressLogger:
    """Simple progress logger."""
    
    def __init__(self, total: int, desc: str = '', log_interval: int = 10):
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.start_time = None
    
    def start(self):
        import time
        self.start_time = time.perf_counter()
        self.current = 0
    
    def update(self, n: int = 1):
        import time
        self.current += n
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = time.perf_counter() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            logger.info(f'{self.desc}: {self.current}/{self.total} '
                       f'({100*self.current/self.total:.1f}%) '
                       f'[{elapsed:.1f}s elapsed, {eta:.1f}s remaining]')
    
    def close(self):
        import time
        elapsed = time.perf_counter() - self.start_time
        logger.info(f'{self.desc}: Completed in {elapsed:.1f}s')


# =============================================================================
# Memory Utilities
# =============================================================================

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage info."""
    if not torch.cuda.is_available():
        return {'available': False}
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return {
        'available': True,
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'max_allocated_mb': max_allocated,
    }


def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
