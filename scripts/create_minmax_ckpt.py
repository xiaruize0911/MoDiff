"""
Fast checkpoint creation for CIFAR-10 diffusion model using min-max weight calibration.
This creates a checkpoint with weight quantization scales initialized but without
full AdaRound optimization (which takes hours).

For paper-level results, you need the full AdaRound calibration from Q-Diffusion.
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from types import SimpleNamespace

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddim.models.diffusion import Model
from ddim.functions.ckpt_util import get_ckpt_path
from qdiff.quant_model import QuantModel
from qdiff.quant_layer import QuantModule, UniformAffineQuantizer
from qdiff.quant_block import BaseQuantBlock


def dict_to_namespace(d):
    """Recursively convert a dictionary to a namespace."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def add_default_config_values(config):
    """Add default values that might be missing from config."""
    if not hasattr(config, 'split_shortcut'):
        config.split_shortcut = False
    return config


def create_minmax_checkpoint(config_path, weight_bit=4, output_path=None, cali_data_path=None):
    """Create a min-max initialized quantized checkpoint."""
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    config = add_default_config_values(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset}")
    
    ckpt_path = get_ckpt_path(f"ema_{name}")
    print(f"Loading checkpoint from: {ckpt_path}")
    
    model = Model(config)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {param_count:.1f}M parameters")
    
    # Create quantized model
    wq_params = {
        'n_bits': weight_bit, 
        'channel_wise': True, 
        'scale_method': 'mse'
    }
    aq_params = {
        'n_bits': 8, 
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
    print("QuantModel created")
    
    # Load calibration data
    if cali_data_path and os.path.exists(cali_data_path):
        print(f"Loading calibration data from: {cali_data_path}")
        cali_data = torch.load(cali_data_path)
        # Get samples from multiple timesteps
        nsteps = len(cali_data["ts"])
        timesteps = list(range(0, nsteps, max(1, nsteps // 5)))  # 5 timesteps
        xs_lst = [cali_data["xs"][i][:64] for i in timesteps]  # 64 samples per timestep
        ts_lst = [cali_data["ts"][i][:64] for i in timesteps]
        xs = torch.cat(xs_lst, dim=0)
        ts = torch.cat(ts_lst, dim=0)
        print(f"Calibration data: {xs.shape}, {ts.shape}")
    else:
        print("Using random calibration data")
        xs = torch.randn(256, 3, 32, 32)
        ts = torch.randint(0, 1000, (256,)).float()
    
    # Initialize weight quantization parameters
    print("Initializing weight quantization parameters...")
    qnn.set_quant_state(True, False)  # Enable weight quantization only
    
    with torch.no_grad():
        # Run multiple forward passes with different timesteps for better calibration
        batch_size = 32
        for i in range(0, len(xs), batch_size):
            batch_xs = xs[i:i+batch_size].to(device)
            batch_ts = ts[i:i+batch_size].to(device)
            _ = qnn(batch_xs, batch_ts)
    
    print("Weight quantization initialized")
    
    # Count quantized layers
    quant_layers = 0
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule):
            quant_layers += 1
    print(f"Total quantized layers: {quant_layers}")
    
    # Save checkpoint
    if output_path is None:
        output_path = f"quant_models/cifar_w{weight_bit}a8_minmax_ckpt.pth"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(qnn.state_dict(), output_path)
    
    file_size = os.path.getsize(output_path) / 1e6
    print(f"Saved checkpoint to: {output_path} ({file_size:.1f} MB)")
    
    return qnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10.yml")
    parser.add_argument("--weight_bit", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cali_data", type=str, default="cali_data/cifar10.pt")
    args = parser.parse_args()
    
    create_minmax_checkpoint(
        config_path=args.config,
        weight_bit=args.weight_bit,
        output_path=args.output,
        cali_data_path=args.cali_data
    )


if __name__ == "__main__":
    main()
