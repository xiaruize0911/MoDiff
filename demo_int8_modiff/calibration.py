"""
Calibration for MoDiff INT8 Quantization

This module implements Q-Diff calibration methodology:
1. Generate calibration data at multiple timesteps
2. Calibrate weight quantizers using MSE search
3. Calibrate activation quantizers on RESIDUALS for MoDiff

Paper Reference:
- Section 4.1: "We use 1024 samples for calibration"
- For MoDiff: calibrate on residuals (a_t - a_{t-1}) for lower error
"""

import logging
import os
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from tqdm import tqdm

from demo_int8_modiff.quant_layer_modiff import QuantLayerMoDiff
from demo_int8_modiff.quant_model_modiff import QuantModelMoDiff

logger = logging.getLogger(__name__)


def generate_calibration_data(
    model: nn.Module,
    num_samples: int = 1024,
    image_size: int = 32,
    num_channels: int = 3,
    num_timesteps: int = 1000,
    num_calib_steps: int = 8,
    device: str = 'cuda',
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate calibration data at multiple timesteps.
    
    For proper MoDiff calibration, we need:
    1. Intermediate activations at various timesteps
    2. Pairs of consecutive timesteps for residual computation
    
    Args:
        model: FP32 model (not quantized yet)
        num_samples: Number of calibration samples
        image_size: Image size
        num_channels: Number of channels
        num_timesteps: Total diffusion timesteps
        num_calib_steps: Number of timesteps to sample for calibration
        device: Computation device
        seed: Random seed
        
    Returns:
        Dictionary with calibration data:
        - 'xs': Input tensors at each timestep [num_calib_steps, num_samples, C, H, W]
        - 'ts': Timestep values [num_calib_steps, num_samples]
        - 'xs_prev': Previous timestep inputs (for MoDiff residual calibration)
        - 'ts_prev': Previous timestep values
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model = model.to(device)
    model.eval()
    
    # Timesteps to sample (uniformly spaced)
    timestep_interval = num_timesteps // num_calib_steps
    calib_timesteps = list(range(0, num_timesteps, timestep_interval))[:num_calib_steps]
    
    logger.info(f"Generating calibration data for timesteps: {calib_timesteps}")
    
    xs_list = []
    ts_list = []
    xs_prev_list = []
    ts_prev_list = []
    
    batch_size = min(64, num_samples)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating calibration data"):
            actual_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            # Random starting noise
            x = torch.randn(actual_batch_size, num_channels, image_size, image_size, device=device)
            
            batch_xs = []
            batch_ts = []
            batch_xs_prev = []
            batch_ts_prev = []
            
            prev_x = None
            prev_t = None
            
            for t_idx, t in enumerate(calib_timesteps):
                t_tensor = torch.full((actual_batch_size,), t, device=device, dtype=torch.long)
                
                # Store current
                batch_xs.append(x.cpu())
                batch_ts.append(t_tensor.cpu())
                
                # Store previous (for MoDiff residual)
                if prev_x is not None:
                    batch_xs_prev.append(prev_x)
                    batch_ts_prev.append(prev_t)
                else:
                    # First timestep: use same as current
                    batch_xs_prev.append(x.cpu())
                    batch_ts_prev.append(t_tensor.cpu())
                
                # Forward through model to get next x
                noise_pred = model(x, t_tensor)
                
                # Simple update (approximate)
                x = x + 0.1 * torch.randn_like(x)  # Simplified for calibration
                
                prev_x = batch_xs[-1]
                prev_t = batch_ts[-1]
            
            xs_list.append(torch.stack(batch_xs, dim=0))
            ts_list.append(torch.stack(batch_ts, dim=0))
            xs_prev_list.append(torch.stack(batch_xs_prev, dim=0))
            ts_prev_list.append(torch.stack(batch_ts_prev, dim=0))
    
    # Concatenate all batches
    xs = torch.cat(xs_list, dim=1)  # [num_steps, total_samples, C, H, W]
    ts = torch.cat(ts_list, dim=1)  # [num_steps, total_samples]
    xs_prev = torch.cat(xs_prev_list, dim=1)
    ts_prev = torch.cat(ts_prev_list, dim=1)
    
    logger.info(f"Generated calibration data: xs shape {xs.shape}")
    
    return {
        'xs': xs,
        'ts': ts,
        'xs_prev': xs_prev,
        'ts_prev': ts_prev,
        'calib_timesteps': calib_timesteps,
    }


def calibrate_model_qdiff(
    qmodel: QuantModelMoDiff,
    calib_data: Dict[str, torch.Tensor],
    batch_size: int = 32,
    device: str = 'cuda',
) -> None:
    """
    Calibrate quantized model using Q-Diff methodology.
    
    Steps:
    1. Calibrate weight quantizers (MSE search on weights)
    2. Initialize activation quantizers (forward pass)
    3. Optional: Fine-tune activation scales with reconstruction loss
    
    Args:
        qmodel: Quantized model to calibrate
        calib_data: Calibration data dictionary
        batch_size: Batch size for calibration
        device: Computation device
    """
    logger.info("Starting Q-Diff calibration...")
    
    # Step 1: Calibrate weights
    logger.info("Step 1: Calibrating weight quantizers...")
    qmodel.calibrate_weights()
    
    # Step 2: Enable weight quantization only
    qmodel.set_quant_state(weight_quant=True, act_quant=False)
    
    # Step 3: Run forward passes to initialize activation quantizers
    logger.info("Step 2: Initializing activation quantizers...")
    
    xs = calib_data['xs']  # [num_steps, num_samples, C, H, W]
    ts = calib_data['ts']  # [num_steps, num_samples]
    
    qmodel.model.to(device)
    qmodel.set_quant_state(weight_quant=True, act_quant=True)
    
    with torch.no_grad():
        for step_idx in range(xs.shape[0]):
            for batch_start in range(0, xs.shape[1], batch_size):
                batch_end = min(batch_start + batch_size, xs.shape[1])
                
                batch_x = xs[step_idx, batch_start:batch_end].to(device)
                batch_t = ts[step_idx, batch_start:batch_end].to(device)
                
                qmodel.reset_cache()
                _ = qmodel(batch_x, batch_t)
    
    logger.info("Calibration complete!")


def calibrate_model_modiff(
    qmodel: QuantModelMoDiff,
    calib_data: Dict[str, torch.Tensor],
    batch_size: int = 32,
    device: str = 'cuda',
) -> None:
    """
    Calibrate quantized model using MoDiff methodology.
    
    Key difference from Q-Diff:
    Calibrate activation quantizers on RESIDUALS (a_t - a_{t-1}),
    not raw activations. This gives better scale estimates for MoDiff.
    
    Args:
        qmodel: Quantized model to calibrate
        calib_data: Calibration data with xs_prev
        batch_size: Batch size
        device: Computation device
    """
    logger.info("Starting MoDiff calibration (residual-aware)...")
    
    # Step 1: Calibrate weights
    logger.info("Step 1: Calibrating weight quantizers...")
    qmodel.calibrate_weights()
    
    # Step 2: Enable modulation and run calibration passes
    logger.info("Step 2: Calibrating with MoDiff modulation...")
    
    xs = calib_data['xs']
    ts = calib_data['ts']
    xs_prev = calib_data['xs_prev']
    ts_prev = calib_data['ts_prev']
    
    qmodel.model.to(device)
    qmodel.set_quant_state(weight_quant=True, act_quant=True)
    qmodel.set_modulation(True)
    
    with torch.no_grad():
        for step_idx in range(xs.shape[0]):
            for batch_start in range(0, xs.shape[1], batch_size):
                batch_end = min(batch_start + batch_size, xs.shape[1])
                
                # Reset cache for each batch
                qmodel.reset_cache()
                
                # First pass with previous timestep (to populate cache)
                batch_x_prev = xs_prev[step_idx, batch_start:batch_end].to(device)
                batch_t_prev = ts_prev[step_idx, batch_start:batch_end].to(device)
                _ = qmodel(batch_x_prev, batch_t_prev)
                
                # Second pass with current timestep (uses residual)
                batch_x = xs[step_idx, batch_start:batch_end].to(device)
                batch_t = ts[step_idx, batch_start:batch_end].to(device)
                _ = qmodel(batch_x, batch_t)
    
    logger.info("MoDiff calibration complete!")


def save_calibrated_model(
    qmodel: QuantModelMoDiff,
    save_path: str,
) -> None:
    """Save calibrated model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    state_dict = qmodel.state_dict()
    torch.save({
        'state_dict': state_dict,
        'weight_bits': qmodel.weight_bits,
        'act_bits': qmodel.act_bits,
        'modulate': qmodel.modulate,
    }, save_path)
    
    logger.info(f"Saved calibrated model to {save_path}")


def load_calibrated_model(
    model: nn.Module,
    ckpt_path: str,
    device: str = 'cuda',
) -> QuantModelMoDiff:
    """Load calibrated model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    qmodel = QuantModelMoDiff(
        model,
        weight_bits=ckpt['weight_bits'],
        act_bits=ckpt['act_bits'],
        modulate=ckpt['modulate'],
    )
    
    qmodel.load_state_dict(ckpt['state_dict'])
    qmodel.to(device)
    
    logger.info(f"Loaded calibrated model from {ckpt_path}")
    return qmodel
