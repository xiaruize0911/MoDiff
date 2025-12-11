"""
DDIM Sampler with MoDiff Support

This module implements DDIM (Denoising Diffusion Implicit Models) sampling
integrated with MoDiff quantized models.

Paper Reference:
- Section 4.1: "For CIFAR-10, we use DDIM models with 100 denoising steps"
- eta=0 for deterministic DDIM sampling

Key Feature:
    Calls model.reset_cache() at the start of each sample to reset
    MoDiff's error-compensated modulation state.
"""

import logging
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute alpha_bar (cumulative product of alphas) at timestep t.
    
    Args:
        beta: Beta schedule tensor [T]
        t: Timestep tensor [N]
        
    Returns:
        Alpha values at timestep t [N, 1, 1, 1]
    """
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def get_beta_schedule(
    beta_schedule: str = "linear",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    num_timesteps: int = 1000,
) -> np.ndarray:
    """
    Get noise schedule betas.
    
    Args:
        beta_schedule: Schedule type ("linear", "quad", "const")
        beta_start: Starting beta value
        beta_end: Ending beta value
        num_timesteps: Number of diffusion timesteps
        
    Returns:
        Beta schedule as numpy array
    """
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "const":
        betas = np.full(num_timesteps, beta_end, dtype=np.float64)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
    
    return betas


class DDIMSamplerMoDiff:
    """
    DDIM Sampler with MoDiff Support.
    
    This sampler implements DDIM sampling and properly integrates with
    MoDiff quantized models by calling reset_cache() for each sample.
    
    Args:
        model: Quantized model (QuantModelMoDiff)
        beta_schedule: Noise schedule type
        beta_start: Starting beta
        beta_end: Ending beta
        num_timesteps: Total diffusion timesteps (T)
        device: Computation device
        
    Usage:
        sampler = DDIMSamplerMoDiff(model)
        samples = sampler.sample(
            num_samples=1000,
            image_size=32,
            num_channels=3,
            ddim_steps=100,
            eta=0.0,
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_timesteps: int = 1000,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Compute beta schedule
        betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps)
        self.betas = torch.from_numpy(betas).float().to(device)
        
        # Compute alphas
        alphas = 1.0 - self.betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        
        logger.info(f"DDIMSamplerMoDiff initialized: T={num_timesteps}, device={device}")
    
    def _get_timestep_sequence(
        self,
        ddim_steps: int,
        skip_type: str = "uniform",
    ) -> List[int]:
        """
        Get the sequence of timesteps for DDIM sampling.
        
        Args:
            ddim_steps: Number of DDIM steps
            skip_type: How to skip timesteps ("uniform" or "quad")
            
        Returns:
            List of timesteps in reverse order (T-1, ..., 0)
        """
        if skip_type == "uniform":
            skip = self.num_timesteps // ddim_steps
            seq = list(range(0, self.num_timesteps, skip))
        elif skip_type == "quad":
            seq = np.linspace(0, np.sqrt(self.num_timesteps * 0.8), ddim_steps) ** 2
            seq = [int(s) for s in seq]
        else:
            raise ValueError(f"Unknown skip type: {skip_type}")
        
        return seq
    
    @torch.no_grad()
    def sample_one(
        self,
        image_size: int = 32,
        num_channels: int = 3,
        ddim_steps: int = 100,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a single sample using DDIM.
        
        Args:
            image_size: Output image size
            num_channels: Number of channels
            ddim_steps: Number of DDIM steps (paper uses 100 for CIFAR-10)
            eta: DDIM eta parameter (0=deterministic, 1=DDPM)
            seed: Random seed for this sample
            
        Returns:
            Generated sample tensor [1, C, H, W]
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Reset MoDiff cache for this sample!
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
        
        # Initialize from noise
        x = torch.randn(1, num_channels, image_size, image_size, device=self.device)
        
        # Get timestep sequence
        seq = self._get_timestep_sequence(ddim_steps)
        seq_next = [-1] + list(seq[:-1])
        
        # DDIM sampling loop
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            t_next = torch.full((1,), j, device=self.device, dtype=torch.long)
            
            # Get alphas
            at = compute_alpha(self.betas, t)
            at_next = compute_alpha(self.betas, t_next) if j >= 0 else torch.ones_like(at)
            
            # Predict noise
            et = self.model(x, t)
            
            # Predict x0
            x0_pred = (x - et * (1 - at).sqrt()) / at.sqrt()
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # DDIM update
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            if eta > 0 and j >= 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = at_next.sqrt() * x0_pred + c1 * noise + c2 * et
        
        return x
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        image_size: int = 32,
        num_channels: int = 3,
        ddim_steps: int = 100,
        eta: float = 0.0,
        batch_size: int = 1,
        seed: int = 42,
        verbose: bool = True,
    ) -> List[torch.Tensor]:
        """
        Generate multiple samples using DDIM.
        
        For MoDiff, we generate one sample at a time to properly
        reset the cache between samples. This is required for
        error-compensated modulation to work correctly.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Output image size
            num_channels: Number of channels
            ddim_steps: Number of DDIM steps
            eta: DDIM eta parameter
            batch_size: Batch size (currently only 1 supported for MoDiff)
            seed: Base random seed
            verbose: Show progress bar
            
        Returns:
            List of generated samples
        """
        samples = []
        
        iterator = range(num_samples)
        if verbose:
            iterator = tqdm(iterator, desc="Generating samples")
        
        for i in iterator:
            sample = self.sample_one(
                image_size=image_size,
                num_channels=num_channels,
                ddim_steps=ddim_steps,
                eta=eta,
                seed=seed + i,
            )
            samples.append(sample.cpu())
        
        return samples
    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        image_size: int = 32,
        num_channels: int = 3,
        ddim_steps: int = 100,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a batch of samples (without MoDiff - for speed comparison).
        
        Note: This does NOT use MoDiff modulation properly since all samples
        in the batch share the same cache state. Use sample() for MoDiff.
        
        Args:
            batch_size: Number of samples in batch
            image_size: Output image size
            num_channels: Number of channels  
            ddim_steps: Number of DDIM steps
            eta: DDIM eta parameter
            seed: Random seed
            
        Returns:
            Batch of samples [B, C, H, W]
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Reset cache (only once for entire batch - NOT proper MoDiff!)
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
        
        # Initialize from noise
        x = torch.randn(batch_size, num_channels, image_size, image_size, device=self.device)
        
        # Get timestep sequence
        seq = self._get_timestep_sequence(ddim_steps)
        seq_next = [-1] + list(seq[:-1])
        
        # DDIM sampling loop
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            t_next = torch.full((batch_size,), j, device=self.device, dtype=torch.long)
            
            at = compute_alpha(self.betas, t)
            at_next = compute_alpha(self.betas, t_next) if j >= 0 else torch.ones_like(at)
            
            et = self.model(x, t)
            
            x0_pred = (x - et * (1 - at).sqrt()) / at.sqrt()
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            if eta > 0 and j >= 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = at_next.sqrt() * x0_pred + c1 * noise + c2 * et
        
        return x


def inverse_data_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Transform from model output range [-1, 1] to image range [0, 1].
    """
    return (x + 1.0) / 2.0


def tensor_to_pil(x: torch.Tensor):
    """
    Convert tensor to PIL Image.
    
    Args:
        x: Tensor [C, H, W] or [1, C, H, W] in range [0, 1]
        
    Returns:
        PIL Image
    """
    from PIL import Image
    
    if x.dim() == 4:
        x = x[0]
    
    x = x.clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).cpu().numpy()
    
    if x.shape[2] == 1:
        x = x[:, :, 0]
    
    return Image.fromarray(x)
