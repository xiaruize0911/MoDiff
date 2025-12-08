"""
Extract quantization scales from trained PyTorch INT8 models.

This module provides utilities to extract quantization parameters (scale, zero_point)
from trained INT8 models and store them in formats compatible with TensorRT calibration.

The key insight: Instead of using TensorRT's entropy calibration (which produces different
scales than training), we extract the actual scales used during training and inject them
into TensorRT's calibration process.

Author: MoDiff INT8 Fix
Date: October 20, 2025
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


def extract_scales_from_quantized_model(
    model: torch.nn.Module,
    save_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract quantization scales and zero_points from a trained QuantModelINT8.
    
    Args:
        model: QuantModelINT8 instance or wrapped model
        save_dir: Optional directory to save extracted scales
        
    Returns:
        Dictionary mapping layer names to their scale/zero_point parameters
        Structure: {
            'layer_name': {
                'weight_scale': np.ndarray,
                'weight_zero_point': np.ndarray,
                'act_scale': np.ndarray,
                'act_zero_point': np.ndarray,
            },
            ...
        }
    """
    logger.info(f"[ScaleExtractor] Extracting scales from model with {sum(p.numel() for p in model.parameters())} parameters")
    
    scales_dict = OrderedDict()
    layer_idx = 0
    
    # Traverse all modules recursively
    for module_name, module in model.named_modules():
        # Skip container modules
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)):
            continue
            
        current_layer_key = None
        
        # Extract from QuantModuleINT8 (weight quantization)
        if hasattr(module, 'weight_quantizer'):
            current_layer_key = f"layer_{layer_idx}_{module_name}"
            scales_dict[current_layer_key] = {}
            
            quantizer = module.weight_quantizer
            
            # Get weight scale and zero_point
            if hasattr(quantizer, 'scale') and quantizer.scale is not None:
                scale_val = quantizer.scale
                if isinstance(scale_val, torch.Tensor):
                    scale_val = scale_val.detach().cpu().numpy()
                scales_dict[current_layer_key]['weight_scale'] = scale_val
                
                logger.debug(f"  {current_layer_key}: weight_scale shape={scale_val.shape if hasattr(scale_val, 'shape') else 'scalar'}")
            
            if hasattr(quantizer, 'zero_point') and quantizer.zero_point is not None:
                zp_val = quantizer.zero_point
                if isinstance(zp_val, torch.Tensor):
                    zp_val = zp_val.detach().cpu().numpy()
                scales_dict[current_layer_key]['weight_zero_point'] = zp_val
            
            layer_idx += 1
        
        # Extract from activation quantizers (if present)
        if hasattr(module, 'act_quantizer'):
            if current_layer_key is None:
                # No weight quantizer found for this module, create new entry
                current_layer_key = f"layer_{layer_idx}_{module_name}"
                scales_dict[current_layer_key] = {}
                layer_idx += 1
            
            quantizer = module.act_quantizer
            
            if hasattr(quantizer, 'scale') and quantizer.scale is not None:
                scale_val = quantizer.scale
                if isinstance(scale_val, torch.Tensor):
                    scale_val = scale_val.detach().cpu().numpy()
                scales_dict[current_layer_key]['act_scale'] = scale_val
                logger.debug(f"  {current_layer_key}: act_scale shape={scale_val.shape if hasattr(scale_val, 'shape') else 'scalar'}")
            
            if hasattr(quantizer, 'zero_point') and quantizer.zero_point is not None:
                zp_val = quantizer.zero_point
                if isinstance(zp_val, torch.Tensor):
                    zp_val = zp_val.detach().cpu().numpy()
                scales_dict[current_layer_key]['act_zero_point'] = zp_val
    
    logger.info(f"[ScaleExtractor] Extracted scales from {len(scales_dict)} quantized layers")
    
    # Save if requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy compressed archive
        scales_to_save = {}
        for layer_name, params in scales_dict.items():
            for param_name, param_val in params.items():
                if param_val is not None:
                    if isinstance(param_val, np.ndarray):
                        scales_to_save[f"{layer_name}_{param_name}"] = param_val
                    elif isinstance(param_val, (int, float)):
                        scales_to_save[f"{layer_name}_{param_name}"] = np.array([param_val])
        
        scales_file = save_dir / "model_scales.npz"
        np.savez_compressed(scales_file, **scales_to_save)
        logger.info(f"[ScaleExtractor] Saved scales to {scales_file} ({len(scales_to_save)} parameters)")
        
        # Also save a metadata file
        metadata = {
            'num_layers': len(scales_dict),
            'num_parameters': len(scales_to_save),
            'layer_names': list(scales_dict.keys()),
        }
        import json
        metadata_file = save_dir / "scales_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[ScaleExtractor] Saved metadata to {metadata_file}")
    
    return scales_dict


def load_scales_from_file(scales_file: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load extracted scales from a saved .npz file.
    
    Args:
        scales_file: Path to model_scales.npz file
        
    Returns:
        Dictionary of scales
    """
    logger.info(f"[ScaleExtractor] Loading scales from {scales_file}")
    
    with np.load(scales_file, allow_pickle=True) as data:
        scales_dict = {}
        for key in data.files:
            # Parse key format: "layer_idx_name_param_type"
            parts = key.rsplit('_', 2)  # Split from right to get param_type
            if len(parts) == 3:
                layer_key = parts[0]
                param_type = f"{parts[1]}_{parts[2]}"
            else:
                layer_key = key
                param_type = 'value'
            
            if layer_key not in scales_dict:
                scales_dict[layer_key] = {}
            
            scales_dict[layer_key][param_type] = data[key]
    
    logger.info(f"[ScaleExtractor] Loaded scales from {len(scales_dict)} layers")
    return scales_dict


def compare_scales(
    trained_scales: Dict[str, Dict[str, np.ndarray]],
    trt_scales: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """
    Compare trained scales with TensorRT-computed scales to identify mismatches.
    
    Args:
        trained_scales: Scales extracted from trained model
        trt_scales: Scales computed by TensorRT entropy calibration
        
    Returns:
        Dictionary of layer names to scale difference metrics
    """
    logger.info("[ScaleExtractor] Comparing trained vs TensorRT scales")
    
    differences = {}
    
    for layer_name in trained_scales.keys():
        if layer_name not in trt_scales:
            differences[layer_name] = float('inf')
            logger.warning(f"  {layer_name}: Not found in TensorRT scales")
            continue
        
        trained_scale = trained_scales[layer_name].get('weight_scale')
        trt_scale = trt_scales[layer_name].get('weight_scale')
        
        if trained_scale is None or trt_scale is None:
            differences[layer_name] = float('nan')
            continue
        
        # Compute relative difference
        diff = np.mean(np.abs(trained_scale - trt_scale) / (np.abs(trained_scale) + 1e-8))
        differences[layer_name] = float(diff)
        
        if diff > 0.1:
            logger.warning(f"  {layer_name}: {diff:.4f} relative difference (HIGH!)")
        else:
            logger.debug(f"  {layer_name}: {diff:.4f} relative difference")
    
    return differences


def apply_scales_to_calibration(
    calib_cache_path: Path,
    trained_scales: Dict[str, Dict[str, np.ndarray]],
) -> bytes:
    """
    Modify TensorRT calibration cache to use trained scales.
    
    This is an advanced approach that directly modifies the TensorRT calibration
    cache to replace entropy-computed scales with trained scales.
    
    Args:
        calib_cache_path: Path to TensorRT calibration cache
        trained_scales: Extracted scales from trained model
        
    Returns:
        Modified calibration cache as bytes
    
    Note:
        This is a complex operation that requires understanding TensorRT's
        cache format. Use with caution.
    """
    logger.warning("[ScaleExtractor] Calibration cache modification not yet implemented")
    logger.info("[ScaleExtractor] Recommend using MoDiffScaleExtractorCalibrator instead")
    return None


def verify_scales(
    model: torch.nn.Module,
    scales_dict: Dict[str, Dict[str, np.ndarray]],
) -> bool:
    """
    Verify that extracted scales are valid and reasonable.
    
    Args:
        model: The model
        scales_dict: Extracted scales
        
    Returns:
        True if scales are valid, False otherwise
    """
    logger.info("[ScaleExtractor] Verifying extracted scales")
    
    all_valid = True
    
    for layer_name, scales in scales_dict.items():
        for scale_type, scale_val in scales.items():
            if scale_val is None:
                logger.warning(f"  {layer_name}: {scale_type} is None")
                all_valid = False
                continue
            
            # Convert to numpy if needed
            if isinstance(scale_val, torch.Tensor):
                scale_val = scale_val.detach().cpu().numpy()
            
            # Check for NaN or Inf
            if np.any(np.isnan(scale_val)):
                logger.error(f"  {layer_name}: {scale_type} contains NaN!")
                all_valid = False
            
            if np.any(np.isinf(scale_val)):
                logger.error(f"  {layer_name}: {scale_type} contains Inf!")
                all_valid = False
            
            # Check for zero or negative values
            if np.any(scale_val <= 0):
                logger.error(f"  {layer_name}: {scale_type} contains non-positive values!")
                all_valid = False
            
            # Log statistics
            logger.debug(f"  {layer_name}: {scale_type} stats: min={np.min(scale_val):.6f}, max={np.max(scale_val):.6f}, mean={np.mean(scale_val):.6f}")
    
    if all_valid:
        logger.info("[ScaleExtractor] All scales are valid ✓")
    else:
        logger.error("[ScaleExtractor] Some scales are invalid! ✗")
    
    return all_valid


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("""
    Scale Extractor Module
    
    Usage:
        from trt.scale_extractor import extract_scales_from_quantized_model
        
        # Load trained INT8 model
        qmodel = QuantModelINT8(model, weight_params, act_params)
        # ... load checkpoint ...
        
        # Extract scales
        scales = extract_scales_from_quantized_model(
            qmodel,
            save_dir='calib/extracted_scales'
        )
        
        # Use in calibrator
        from trt.entropy_calibrator import MoDiffScaleExtractorCalibrator
        cal = MoDiffScaleExtractorCalibrator(
            calib_data_dir='calib/',
            extracted_scales=scales,
            cache_path='calib/modiff_int8.cache'
        )
    """)
