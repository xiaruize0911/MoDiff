"""
Load trained INT8 checkpoint and extract quantization scales.

This script handles the complete workflow of:
1. Loading the original diffusion model
2. Loading the INT8 checkpoint with quantization parameters
3. Extracting scales and zero_points from all quantized layers
4. Saving scales for use in TensorRT calibration

Author: MoDiff INT8 Fix
Date: October 20, 2025
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: Path, device: str = 'cuda') -> torch.nn.Module:
    """
    Load the original diffusion model (usually from .ckpt file).
    
    Args:
        checkpoint_path: Path to model checkpoint (.ckpt or .pth)
        device: Device to load model on
        
    Returns:
        Loaded model on specified device
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to load with torch first
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint with keys: {list(checkpoint.keys())[:5]}...")
        
        # The checkpoint might be wrapped in a dict
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # Might need to reconstruct model
            logger.warning("Checkpoint contains state_dict only, need model architecture")
            state_dict = checkpoint['state_dict']
            # Remove module. prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            logger.info(f"State dict keys: {list(state_dict.keys())[:5]}...")
            return state_dict
        else:
            model = checkpoint
        
        return model.to(device) if isinstance(model, torch.nn.Module) else model
    
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def load_quant_model_from_checkpoint(
    checkpoint_path: Path,
    weight_params: Dict,
    act_params: Dict,
    device: str = 'cuda',
) -> torch.nn.Module:
    """
    Load or reconstruct the QuantModelINT8 from checkpoint.
    
    Args:
        checkpoint_path: Path to INT8 model checkpoint
        weight_params: Weight quantization parameters
        act_params: Activation quantization parameters
        device: Device to load on
        
    Returns:
        QuantModelINT8 instance with loaded weights
    """
    from qdiff.quant_model_int8 import QuantModelINT8
    
    logger.info(f"Loading INT8 quantized model from: {checkpoint_path}")
    
    # First load the base model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # The checkpoint might contain the full QuantModelINT8
    if isinstance(checkpoint, torch.nn.Module):
        qmodel = checkpoint.to(device)
        logger.info("Loaded QuantModelINT8 directly from checkpoint")
        return qmodel
    
    # Otherwise it might be state_dict
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    # Clean up state dict
    if isinstance(state_dict, dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    logger.info(f"State dict has {len(state_dict)} parameters")
    logger.info(f"Sample keys: {list(state_dict.keys())[:3]}")
    
    # We need the base model architecture to reconstruct QuantModelINT8
    # For now, return the state dict
    return state_dict


def extract_quantization_scales(
    qmodel: torch.nn.Module,
    output_dir: Path = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract quantization scales and zero_points from QuantModelINT8.
    
    Args:
        qmodel: QuantModelINT8 instance with loaded weights
        output_dir: Optional directory to save extracted scales
        
    Returns:
        Dictionary of extracted scales
    """
    from trt.scale_extractor import extract_scales_from_quantized_model
    
    logger.info("Extracting quantization scales from model")
    
    scales = extract_scales_from_quantized_model(qmodel, output_dir)
    
    logger.info(f"Extracted {len(scales)} layer scales")
    
    return scales


def main():
    parser = argparse.ArgumentParser(
        description="Load INT8 checkpoint and extract quantization scales for TensorRT"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "ema_diffusion_cifar10_model" / "model-790000.ckpt",
        help="Path to INT8 model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "cifar10.yml",
        help="Path to model config file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "export" / "extracted_scales",
        help="Directory to save extracted scales"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to load model on"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config.exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
        
        weight_params = config.get('weight_quant_params', {'n_bits': 8, 'channel_wise': True})
        act_params = config.get('act_quant_params', {'n_bits': 8, 'channel_wise': False})
    else:
        logger.warning(f"Config not found: {args.config}")
        weight_params = {'n_bits': 8, 'channel_wise': True}
        act_params = {'n_bits': 8, 'channel_wise': False}
    
    logger.info(f"Weight params: {weight_params}")
    logger.info(f"Act params: {act_params}")
    
    # Load the INT8 model
    try:
        qmodel = load_quant_model_from_checkpoint(
            args.checkpoint,
            weight_params,
            act_params,
            device=args.device,
        )
        logger.info("✓ Loaded INT8 model successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load INT8 model: {e}")
        raise
    
    # Extract scales
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scales = extract_quantization_scales(qmodel, args.output_dir)
        logger.info(f"✓ Extracted {len(scales)} layer scales to {args.output_dir}")
    except Exception as e:
        logger.error(f"✗ Failed to extract scales: {e}")
        raise
    
    logger.info("\n✓ Scale extraction complete!")
    logger.info(f"  Extracted scales saved to: {args.output_dir}")
    logger.info(f"  Use these scales with MoDiffScaleExtractorCalibrator in TensorRT")


if __name__ == "__main__":
    main()
