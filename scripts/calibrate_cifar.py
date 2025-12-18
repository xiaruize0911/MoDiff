"""
Script to calibrate CIFAR-10 diffusion model weights using AdaRound.
This creates the pre-calibrated checkpoint needed for MoDiff evaluation.
"""

import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddim.models.diffusion import Model
from ddim.functions.ckpt_util import get_ckpt_path
from qdiff.quant_model import QuantModel
from qdiff.quant_layer import QuantModule, UniformAffineQuantizer
from qdiff.quant_block import BaseQuantBlock
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.layer_recon import layer_reconstruction
from qdiff.block_recon import block_reconstruction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_train_samples(cali_data, num_samples=1024, num_st=20):
    """Extract training samples from calibration data."""
    nsteps = len(cali_data["ts"])
    timesteps = list(range(0, nsteps, max(1, nsteps // num_st)))
    
    xs_lst = [cali_data["xs"][i][:num_samples] for i in timesteps]
    ts_lst = [cali_data["ts"][i][:num_samples] for i in timesteps]
    
    xs = torch.cat(xs_lst, dim=0)
    ts = torch.cat(ts_lst, dim=0)
    
    return xs, ts


def convert_adaround(model):
    """Convert weight quantizers to AdaRound."""
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            module.weight_quantizer = AdaRoundQuantizer(
                uaq=module.weight_quantizer, 
                round_mode='learned_hard_sigmoid',
                weight_tensor=module.org_weight.data
            )
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                continue
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, QuantModule):
                    if sub_module.split != 0:
                        sub_module.weight_quantizer = AdaRoundQuantizer(
                            uaq=sub_module.weight_quantizer, 
                            round_mode='learned_hard_sigmoid',
                            weight_tensor=sub_module.org_weight.data[:, :sub_module.split, ...]
                        )
                        sub_module.weight_quantizer_0 = AdaRoundQuantizer(
                            uaq=sub_module.weight_quantizer_0, 
                            round_mode='learned_hard_sigmoid',
                            weight_tensor=sub_module.org_weight.data[:, sub_module.split:, ...]
                        )
                    else:
                        sub_module.weight_quantizer = AdaRoundQuantizer(
                            uaq=sub_module.weight_quantizer, 
                            round_mode='learned_hard_sigmoid',
                            weight_tensor=sub_module.org_weight.data
                        )
        else:
            convert_adaround(module)


def set_weight_quantize_params(model, cali_data, batch_size=32):
    """Initialize weight quantization parameters using calibration data."""
    logger.info("Initializing weight quantization parameters...")
    
    cali_xs, cali_ts = cali_data
    device = next(model.parameters()).device
    
    # Run a forward pass to initialize quantizers
    model.set_quant_state(True, False)
    with torch.no_grad():
        _ = model(cali_xs[:batch_size].to(device), cali_ts[:batch_size].to(device))
    
    logger.info("Weight quantization parameters initialized")


def recon_model(model, cali_data, args):
    """
    Block/layer reconstruction using AdaRound.
    This is the core calibration step that optimizes weight rounding.
    """
    device = next(model.parameters()).device
    cali_xs, cali_ts = cali_data
    
    # Move calibration data to device
    cali_xs = cali_xs.to(device)
    cali_ts = cali_ts.to(device)
    cali_data_tuple = (cali_xs, cali_ts)
    
    # Set model to weight quantization mode
    model.set_quant_state(True, False)
    
    # Convert to AdaRound
    logger.info("Converting to AdaRound quantizers...")
    convert_adaround(model)
    
    # Get all quantized modules and blocks
    quant_modules = []
    quant_blocks = []
    
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            if not module.ignore_reconstruction:
                quant_modules.append((name, module))
        elif isinstance(module, BaseQuantBlock):
            if not module.ignore_reconstruction:
                quant_blocks.append((name, module))
    
    logger.info(f"Found {len(quant_modules)} QuantModules and {len(quant_blocks)} QuantBlocks")
    
    # Reconstruct blocks (if any)
    for name, block in tqdm(quant_blocks, desc="Reconstructing blocks"):
        logger.info(f"Reconstructing block: {name}")
        block_reconstruction(
            model, block, cali_data_tuple,
            batch_size=args.cali_batch_size,
            iters=args.cali_iters,
            weight=0.01,
            opt_mode='mse',
            asym=True,
            b_range=(20, 2),
            warmup=0.2,
            act_quant=False,
            lr=args.cali_lr,
            p=args.cali_p
        )
    
    # Reconstruct individual layers
    for name, layer in tqdm(quant_modules, desc="Reconstructing layers"):
        # Skip layers that are part of already-reconstructed blocks
        skip = False
        for block_name, _ in quant_blocks:
            if name.startswith(block_name):
                skip = True
                break
        
        if skip:
            continue
            
        logger.info(f"Reconstructing layer: {name}")
        layer_reconstruction(
            model, layer, cali_data_tuple,
            batch_size=args.cali_batch_size,
            iters=args.cali_iters,
            weight=0.01,
            opt_mode='mse',
            asym=True,
            b_range=(20, 2),
            warmup=0.2,
            act_quant=False,
            lr=args.cali_lr,
            p=args.cali_p
        )
    
    logger.info("Model reconstruction complete")


def save_quantized_checkpoint(model, save_path):
    """Save the calibrated quantized model checkpoint."""
    logger.info(f"Saving calibrated checkpoint to {save_path}")
    
    # Collect state dict
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (AdaRoundQuantizer, UniformAffineQuantizer)):
            prefix = name
            if hasattr(module, 'delta'):
                if torch.is_tensor(module.delta):
                    state_dict[f"{prefix}.delta"] = module.delta.data
                else:
                    state_dict[f"{prefix}.delta"] = torch.tensor(module.delta)
            if hasattr(module, 'zero_point'):
                if torch.is_tensor(module.zero_point):
                    state_dict[f"{prefix}.zero_point"] = module.zero_point.data
                else:
                    state_dict[f"{prefix}.zero_point"] = torch.tensor(float(module.zero_point))
            if hasattr(module, 'alpha'):
                state_dict[f"{prefix}.alpha"] = module.alpha.data
    
    # Save full model state dict instead for compatibility
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved checkpoint with {len(state_dict)} quantizer parameters")


def main():
    parser = argparse.ArgumentParser(description="Calibrate CIFAR-10 diffusion model")
    parser.add_argument("--weight_bit", type=int, default=4, help="Weight bit width")
    parser.add_argument("--cali_data_path", type=str, default="cali_data/cifar10.pt",
                        help="Path to calibration data")
    parser.add_argument("--output_path", type=str, default="quant_models/cifar_w4a8_ckpt.pth",
                        help="Output checkpoint path")
    parser.add_argument("--cali_n", type=int, default=256, help="Number of calibration samples per timestep")
    parser.add_argument("--cali_st", type=int, default=20, help="Number of timesteps for calibration")
    parser.add_argument("--cali_batch_size", type=int, default=32, help="Batch size for calibration")
    parser.add_argument("--cali_iters", type=int, default=20000, help="Number of iterations for reconstruction")
    parser.add_argument("--cali_lr", type=float, default=4e-4, help="Learning rate for reconstruction")
    parser.add_argument("--cali_p", type=float, default=2.4, help="Lp norm for reconstruction")
    parser.add_argument("--split", action="store_true", help="Use split quantization")
    parser.add_argument("--quick", action="store_true", help="Quick calibration with fewer iterations")
    args = parser.parse_args()
    
    if args.quick:
        args.cali_iters = 1000
        args.cali_n = 64
        logger.info("Quick mode: using fewer iterations and samples")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load pretrained model
    logger.info("Loading pretrained CIFAR-10 diffusion model...")
    ckpt_path = get_ckpt_path("cifar10", prefix="models/")
    
    # Model config for CIFAR-10
    model_config = {
        "resolution": 32,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.1,
    }
    
    model = Model(**model_config)
    states = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(states)
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Load calibration data
    logger.info(f"Loading calibration data from {args.cali_data_path}...")
    cali_data_raw = torch.load(args.cali_data_path)
    cali_xs, cali_ts = get_train_samples(cali_data_raw, num_samples=args.cali_n, num_st=args.cali_st)
    logger.info(f"Calibration data shape: xs={cali_xs.shape}, ts={cali_ts.shape}")
    
    # Create quantized model
    logger.info("Creating quantized model wrapper...")
    wq_params = {
        'n_bits': args.weight_bit, 
        'channel_wise': True, 
        'scale_method': 'mse'
    }
    aq_params = {
        'n_bits': 8,  # Default to 8-bit activations for calibration
        'channel_wise': False, 
        'scale_method': 'mse', 
        'leaf_param': True
    }
    
    qnn = QuantModel(
        model=model, 
        weight_quant_params=wq_params, 
        act_quant_params=aq_params
    )
    qnn = qnn.to(device)
    qnn.eval()
    
    # Set split if requested
    if args.split:
        logger.info("Setting split quantization...")
        qnn.set_split()
    
    # Initialize weight quantization
    cali_data_tuple = (cali_xs, cali_ts)
    set_weight_quantize_params(qnn, cali_data_tuple, batch_size=args.cali_batch_size)
    
    # Perform reconstruction (AdaRound calibration)
    logger.info("Starting AdaRound reconstruction...")
    recon_model(qnn, cali_data_tuple, args)
    
    # Save checkpoint
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_quantized_checkpoint(qnn, args.output_path)
    
    logger.info("Calibration complete!")
    

if __name__ == "__main__":
    main()
