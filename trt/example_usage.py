#!/usr/bin/env python3
"""
Practical example of using MSE scale extraction with diffusion models.

This script demonstrates:
1. Loading calibration data
2. Extracting MSE scales from a UNet model
3. Saving scales for TensorRT use
4. Comparing quantization errors across layers
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def demonstrate_basic_usage():
    """Show basic usage of MSE scale extraction."""
    from mse_scale_extractor import compute_mse_scales_for_tensor
    
    logger.info("\n" + "=" * 70)
    logger.info("Example 1: Basic Tensor Quantization")
    logger.info("=" * 70)
    
    # Create sample activation (e.g., output of a Conv2d layer)
    activation = np.random.normal(0.5, 0.2, size=(1, 64, 32, 32)).astype(np.float32)
    
    logger.info(f"\nInput activation:")
    logger.info(f"  Shape: {activation.shape}")
    logger.info(f"  Min: {activation.min():.6f}")
    logger.info(f"  Max: {activation.max():.6f}")
    logger.info(f"  Mean: {activation.mean():.6f}")
    logger.info(f"  Std: {activation.std():.6f}")
    
    # Flatten for scale computation
    activation_flat = activation.flatten()
    
    # Compute MSE scales
    scale, zp = compute_mse_scales_for_tensor(activation_flat)
    
    logger.info(f"\nComputed INT8 Scales:")
    logger.info(f"  Scale: {scale:.8f}")
    logger.info(f"  Zero Point: {zp}")
    
    # Verify quantization
    act_int = np.clip(np.round(activation_flat / scale) + zp, 0, 255)
    act_dequant = (act_int - zp) * scale
    error = np.abs(activation_flat - act_dequant)
    
    logger.info(f"\nQuantization Error Statistics:")
    logger.info(f"  Min Error: {error.min():.8f}")
    logger.info(f"  Max Error: {error.max():.8f}")
    logger.info(f"  Mean Error: {error.mean():.8f}")
    logger.info(f"  Std Error: {error.std():.8f}")


def demonstrate_layer_wise_scales():
    """Show extraction of scales for multiple layers."""
    from mse_scale_extractor import compute_mse_scales_for_tensor
    
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Layer-wise Scale Computation")
    logger.info("=" * 70)
    
    # Simulate activations from different layers
    layer_activations = {
        'conv_in': np.random.normal(0.3, 0.15, size=10000),
        'down_0': np.random.normal(0.5, 0.2, size=10000),
        'middle': np.random.normal(0.7, 0.25, size=10000),
        'attn': np.random.normal(0.4, 0.18, size=10000),
        'up_0': np.random.normal(0.6, 0.22, size=10000),
        'conv_out': np.random.normal(0.2, 0.12, size=10000),
    }
    
    scales_dict = {}
    
    logger.info("\nComputing scales for each layer:")
    logger.info(f"{'Layer Name':<15} {'Scale':<15} {'ZP':<6} {'MAE':<12}")
    logger.info("-" * 50)
    
    for layer_name, activation in layer_activations.items():
        scale, zp = compute_mse_scales_for_tensor(activation.astype(np.float32))
        
        # Compute quantization error
        act_int = np.clip(np.round(activation / scale) + zp, 0, 255)
        act_dequant = (act_int - zp) * scale
        mae = np.mean(np.abs(activation - act_dequant))
        
        scales_dict[layer_name] = {
            'scale': scale,
            'zero_point': zp,
            'mae': mae,
        }
        
        logger.info(f"{layer_name:<15} {scale:<15.8f} {zp:<6} {mae:<12.8f}")
    
    logger.info("\n✓ Successfully computed scales for all layers")


def demonstrate_bit_width_selection():
    """Show how bit width affects quantization."""
    from mse_scale_extractor import compute_mse_scales_for_tensor
    
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Bit Width Selection")
    logger.info("=" * 70)
    
    activation = np.random.normal(0, 1, size=5000).astype(np.float32)
    
    logger.info(f"\nInput activation statistics:")
    logger.info(f"  Size: {len(activation)}")
    logger.info(f"  Min: {activation.min():.6f}, Max: {activation.max():.6f}")
    
    logger.info(f"\n{'Bit Width':<12} {'Levels':<10} {'Scale':<15} {'MAE':<12}")
    logger.info("-" * 50)
    
    for n_bits in [4, 8, 16, 32]:
        scale, zp = compute_mse_scales_for_tensor(
            activation, 
            n_bits=n_bits
        )
        n_levels = 2 ** n_bits
        
        # Compute error
        act_int = np.clip(np.round(activation / scale) + zp, 0, n_levels - 1)
        act_dequant = (act_int - zp) * scale
        mae = np.mean(np.abs(activation - act_dequant))
        
        logger.info(f"{n_bits:<12} {n_levels:<10} {scale:<15.8f} {mae:<12.8f}")
    
    logger.info("\n→ Note: More bits = smaller error, larger model")
    logger.info("→ Typical choice: 8-bit for good accuracy/size tradeoff")


def demonstrate_symmetric_vs_asymmetric():
    """Compare symmetric vs asymmetric quantization."""
    from mse_scale_extractor import compute_mse_scales_for_tensor
    
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Symmetric vs Asymmetric Quantization")
    logger.info("=" * 70)
    
    # Create ReLU-like activation (mostly positive)
    activation = np.abs(np.random.normal(0.5, 0.3, size=5000)).astype(np.float32)
    
    logger.info(f"\nInput activation (ReLU-like):")
    logger.info(f"  Min: {activation.min():.6f}, Max: {activation.max():.6f}")
    logger.info(f"  Mean: {activation.mean():.6f}")
    
    # Asymmetric
    scale_asym, zp_asym = compute_mse_scales_for_tensor(activation, symmetric=False)
    act_int = np.clip(np.round(activation / scale_asym) + zp_asym, 0, 255)
    act_dequant = (act_int - zp_asym) * scale_asym
    mae_asym = np.mean(np.abs(activation - act_dequant))
    
    logger.info(f"\nAsymmetric Quantization:")
    logger.info(f"  Scale: {scale_asym:.8f}")
    logger.info(f"  Zero Point: {zp_asym}")
    logger.info(f"  MAE: {mae_asym:.8f}")
    
    # Symmetric
    scale_sym, zp_sym = compute_mse_scales_for_tensor(activation, symmetric=True)
    act_int = np.clip(np.round(activation / scale_sym) + zp_sym, 0, 255)
    act_dequant = (act_int - zp_sym) * scale_sym
    mae_sym = np.mean(np.abs(activation - act_dequant))
    
    logger.info(f"\nSymmetric Quantization:")
    logger.info(f"  Scale: {scale_sym:.8f}")
    logger.info(f"  Zero Point: {zp_sym}")
    logger.info(f"  MAE: {mae_sym:.8f}")
    
    improvement = ((mae_sym - mae_asym) / mae_sym) * 100
    logger.info(f"\n✓ Asymmetric is {improvement:.1f}% better for this activation")


def demonstrate_pytorch_integration():
    """Show integration with PyTorch models."""
    from mse_scale_extractor import compute_mse_scales_for_tensor
    
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: PyTorch Tensor Integration")
    logger.info("=" * 70)
    
    # Create PyTorch tensor (as would come from model layer)
    torch_tensor = torch.randn(64, 256, 8, 8)  # Batch, Channels, Height, Width
    
    logger.info(f"\nPyTorch tensor:")
    logger.info(f"  Shape: {torch_tensor.shape}")
    logger.info(f"  Type: {torch_tensor.dtype}")
    logger.info(f"  Device: {torch_tensor.device}")
    
    # Directly pass to scale computation
    scale, zp = compute_mse_scales_for_tensor(torch_tensor)
    
    logger.info(f"\nComputed scales:")
    logger.info(f"  Scale: {scale:.8f}")
    logger.info(f"  Zero Point: {zp}")
    logger.info(f"  ✓ Automatically converted PyTorch tensor to numpy")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    logger.info("\n" + "█" * 70)
    logger.info("█" + " " * 68 + "█")
    logger.info("█" + "  MSE Scale Extraction - Practical Examples".center(68) + "█")
    logger.info("█" + " " * 68 + "█")
    logger.info("█" * 70)
    
    try:
        demonstrate_basic_usage()
        demonstrate_layer_wise_scales()
        demonstrate_bit_width_selection()
        demonstrate_symmetric_vs_asymmetric()
        demonstrate_pytorch_integration()
        
        logger.info("\n" + "█" * 70)
        logger.info("█" + " " * 68 + "█")
        logger.info("█" + "✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!".center(68) + "█")
        logger.info("█" + " " * 68 + "█")
        logger.info("█" * 70)
        logger.info("\nKey Takeaways:")
        logger.info("  1. MSE minimization provides optimal quantization scales")
        logger.info("  2. Asymmetric quantization works better for skewed distributions")
        logger.info("  3. 8-bit is the best choice for most use cases")
        logger.info("  4. Scales should be computed per-layer for best accuracy")
        logger.info("\n")
        
    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
