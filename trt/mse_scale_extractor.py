"""
Quantization scale extraction from calibration data using MSE minimization.

This module provides utilities to:
1. Compute MSE-optimal quantization scales for tensors
2. Extract scales from model activations on calibration data
3. Save/load scales for TensorRT calibration

The MSE minimization approach reduces quantization error more effectively than
simple min/max scaling, especially for asymmetric distributions.
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def compute_mse_scales_for_tensor(
    tensor: np.ndarray,
    n_bits: int = 8,
    symmetric: bool = False,
    num_candidates: int = 80,
) -> Tuple[float, float]:
    """
    Compute optimal scale and zero_point for a tensor using MSE minimization.
    
    Iterates through candidate scales by progressively reducing max value,
    and selects the scale that minimizes quantization error (L2.4 norm).
    
    Args:
        tensor: Input tensor (numpy array or torch tensor)
        n_bits: Bit width (8 for INT8)
        symmetric: Whether to use symmetric quantization
        num_candidates: Number of scale candidates to try (80 per QDiff paper)
        
    Returns:
        (scale, zero_point) tuple
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    tensor = tensor.astype(np.float32)
    n_levels = 2 ** n_bits
    
    x_min = np.min(tensor)
    x_max = np.max(tensor)
    
    if x_max == x_min:
        return 1.0, 0.0
    
    best_score = float('inf')
    best_scale = 1.0
    best_zero_point = 0.0
    
    # Try different scale candidates by shrinking the range
    for i in range(num_candidates):
        shrink_factor = 1.0 - (i * 0.01)  # 1.0, 0.99, 0.98, ..., 0.21
        new_max = x_max * shrink_factor
        new_min = x_min * shrink_factor
        
        if symmetric:
            # Symmetric: range is [-x_absmax, x_absmax]
            x_absmax = max(abs(new_min), abs(new_max))
            scale = x_absmax / (n_levels // 2 - 1) if x_absmax > 0 else 1.0
            zero_point = 0
        else:
            # Asymmetric: range is [new_min, new_max]
            range_val = new_max - new_min
            scale = range_val / (n_levels - 1) if range_val > 0 else 1.0
            zero_point = int(np.round(-new_min / scale)) if scale > 0 else 0
        
        if scale < 1e-8:
            scale = 1e-8
        
        # Quantize and dequantize
        tensor_int = np.round(tensor / scale) + zero_point
        tensor_int = np.clip(tensor_int, 0, n_levels - 1).astype(np.float32)
        tensor_dequant = (tensor_int - zero_point) * scale
        
        # Compute L2.4 loss (QDiff paper metric)
        diff = np.abs(tensor - tensor_dequant)
        score = np.mean(np.power(diff, 2.4))
        
        if score < best_score:
            best_score = score
            best_scale = scale
            best_zero_point = zero_point
    
    return float(best_scale), int(best_zero_point)


def extract_mse_scales_from_activations(
    model: torch.nn.Module,
    calib_dataloader,
    device: str = 'cuda',
    num_samples: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Extract MSE-optimal scales by running model on calibration data.
    
    Collects activation statistics and computes MSE-optimal quantization scales
    for each layer. Useful for understanding layer-wise quantization effects.
    
    Args:
        model: PyTorch model
        calib_dataloader: DataLoader with calibration samples
        device: Device for computation ('cuda' or 'cpu')
        num_samples: Max samples to process (None = all)
        
    Returns:
        Dict mapping layer names to scale information:
        {
            'layer_name': {
                'scale': float,
                'zero_point': int,
                'min': float,
                'max': float,
                'mean': float,
                'std': float
            },
            ...
        }
    """
    logger.info("[MSE Scale Extraction] Computing scales from activation statistics")
    
    scales_dict = {}
    activation_stats = {}
    hooks = []
    
    def create_hook(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = []
            
            # Extract tensor output
            out_tensor = None
            if isinstance(output, torch.Tensor):
                out_tensor = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    out_tensor = output[0]
            
            if out_tensor is not None:
                activation_stats[name].append(
                    out_tensor.detach().cpu().numpy().astype(np.float32)
                )
        
        return hook
    
    # Register hooks on Conv2d and Linear layers
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)
    
    model.eval()
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(calib_dataloader):
                if num_samples is not None and batch_idx >= num_samples:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    inputs = [
                        x.to(device) if isinstance(x, torch.Tensor) else x
                        for x in batch
                    ]
                    model(*inputs)
                elif isinstance(batch, dict):
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    model(**batch)
                else:
                    batch = batch.to(device) if isinstance(batch, torch.Tensor) else batch
                    model(batch)
                
                if (batch_idx + 1) % 5 == 0:
                    logger.info(
                        f"  Batch {batch_idx + 1}: {len(activation_stats)} layers recorded"
                    )
    
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    # Compute scales
    logger.info("[MSE Scale Extraction] Computing MSE-optimal scales")
    for layer_name, activations in activation_stats.items():
        if not activations:
            continue
        
        # Stack all activations and flatten
        all_acts = np.concatenate(
            [act.flatten() for act in activations],
            axis=0
        )
        
        scale, zero_point = compute_mse_scales_for_tensor(all_acts)
        
        scales_dict[layer_name] = {
            'scale': float(scale),
            'zero_point': int(zero_point),
            'min': float(np.min(all_acts)),
            'max': float(np.max(all_acts)),
            'mean': float(np.mean(all_acts)),
            'std': float(np.std(all_acts)),
        }
        
        logger.debug(
            f"  {layer_name}: scale={scale:.6f}, zp={zero_point}, "
            f"min={scales_dict[layer_name]['min']:.4f}, "
            f"max={scales_dict[layer_name]['max']:.4f}"
        )
    
    logger.info(f"[MSE Scale Extraction] Extracted scales for {len(scales_dict)} layers")
    return scales_dict


def save_scales_to_json(scales_dict: Dict, output_path: Path) -> None:
    """Save scale dictionary to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(scales_dict, f, indent=2)
    
    logger.info(f"[MSE Scale Extraction] Saved scales to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Example: Compute MSE scales for a sample tensor
    logger.info("=" * 60)
    logger.info("MSE Scale Extraction - Sample Code")
    logger.info("=" * 60)
    
    # Create sample tensor with non-uniform distribution
    sample_tensor = np.random.randn(1000) * 0.5 + 0.1
    logger.info(f"\nSample tensor statistics:")
    logger.info(f"  Shape: {sample_tensor.shape}")
    logger.info(f"  Min: {sample_tensor.min():.6f}")
    logger.info(f"  Max: {sample_tensor.max():.6f}")
    logger.info(f"  Mean: {sample_tensor.mean():.6f}")
    logger.info(f"  Std: {sample_tensor.std():.6f}")
    
    # Test 1: Asymmetric quantization
    logger.info("\n[Test 1] Asymmetric INT8 Quantization:")
    scale_asym, zp_asym = compute_mse_scales_for_tensor(
        sample_tensor, 
        n_bits=8, 
        symmetric=False
    )
    logger.info(f"  Scale: {scale_asym:.8f}")
    logger.info(f"  Zero Point: {zp_asym}")
    
    # Verify quantization
    tensor_int = np.round(sample_tensor / scale_asym) + zp_asym
    tensor_int = np.clip(tensor_int, 0, 255)
    tensor_dequant = (tensor_int - zp_asym) * scale_asym
    error = np.mean(np.abs(sample_tensor - tensor_dequant))
    logger.info(f"  Mean Absolute Error: {error:.8f}")
    
    # Test 2: Symmetric quantization
    logger.info("\n[Test 2] Symmetric INT8 Quantization:")
    scale_sym, zp_sym = compute_mse_scales_for_tensor(
        sample_tensor,
        n_bits=8,
        symmetric=True
    )
    logger.info(f"  Scale: {scale_sym:.8f}")
    logger.info(f"  Zero Point: {zp_sym}")
    
    # Verify quantization
    tensor_int = np.round(sample_tensor / scale_sym) + zp_sym
    tensor_int = np.clip(tensor_int, 0, 255)
    tensor_dequant = (tensor_int - zp_sym) * scale_sym
    error = np.mean(np.abs(sample_tensor - tensor_dequant))
    logger.info(f"  Mean Absolute Error: {error:.8f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Sample code completed successfully!")
    logger.info("=" * 60)
