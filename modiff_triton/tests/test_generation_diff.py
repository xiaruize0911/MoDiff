import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import copy
import numpy as np

# Add path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modiff_triton.benchmark_fid_calibrated import MoDiffModel, load_cifar10_model, get_beta_schedule, compute_alpha

def debug_generation_diff():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print("Loading model...")
    # We use the helper from benchmark script to get a standard model
    # This might download weights if not present
    model_fp32 = load_cifar10_model(device)
    model_fp32.eval()
    
    # Create a copy for MoDiff
    model_modiff_inner = copy.deepcopy(model_fp32)
    model_modiff = MoDiffModel(model_modiff_inner, weight_bit=8, act_bit=8)
    
    # 2. Calibrate MoDiff Model (Dummy Calibration)
    print("Calibrating MoDiff model with dummy data...")
    # Create dummy calibration data
    dummy_xs = torch.randn(32, 3, 32, 32, device=device)
    dummy_ts = torch.randint(0, 1000, (32,), device=device).float()
    
    # We need to manually set scales to something reasonable to avoid zero division or infs
    # if the dummy data is too perfect.
    # But calibrate() should handle it.
    model_modiff.calibrate((dummy_xs, dummy_ts), device)
    
    # Force some reasonable scales if they are zero/inf (just in case)
    for layer in model_modiff.modiff_layers:
        if layer.act_scale is None or layer.act_scale == 0:
            layer.act_scale = torch.tensor(1.0, device=device)
        if layer.weight_scale is None or layer.weight_scale == 0:
            layer.weight_scale = torch.tensor(1.0, device=device)
            
    print("Calibration done.")

    # 3. Setup Generation
    n_samples = 4 # Small batch
    timesteps = 10 # Reduced steps for debugging
    
    # Same initial noise
    torch.manual_seed(42)
    x_init = torch.randn(n_samples, 3, 32, 32, device=device)
    
    x_fp32 = x_init.clone()
    x_modiff = x_init.clone()
    
    # DDIM Schedule
    betas = get_beta_schedule("linear", 0.0001, 0.02, 1000)
    betas = torch.from_numpy(betas).float().to(device)
    skip = 1000 // timesteps
    seq = range(0, 1000, skip)
    seq_next = [-1] + list(seq[:-1])
    
    print(f"Starting generation comparison ({timesteps} steps)...")
    
    # Reset MoDiff cache
    model_modiff.reset_cache()
    
    # Loop
    for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
        t_val = i
        next_t_val = j
        
        t = (torch.ones(n_samples, device=device) * t_val).long()
        next_t = (torch.ones(n_samples, device=device) * next_t_val).long()
        
        at = compute_alpha(betas, t)
        at_next = compute_alpha(betas, next_t) if j >= 0 else torch.ones_like(at)
        
        # --- FP32 Step ---
        with torch.no_grad():
            et_fp32 = model_fp32(x_fp32, t.float())
            
            # DDIM update FP32
            x0_t_fp32 = (x_fp32 - et_fp32 * (1 - at).sqrt()) / at.sqrt()
            x0_t_fp32 = torch.clamp(x0_t_fp32, -1, 1)
            c2 = ((1 - at_next)).sqrt() # eta=0
            x_fp32_next = at_next.sqrt() * x0_t_fp32 + c2 * et_fp32

        # --- MoDiff Step ---
        with torch.no_grad():
            # use_modiff=True is default for MoDiffModel.forward
            et_modiff = model_modiff(x_modiff, t.float(), use_modiff=True)
            
            # DDIM update MoDiff
            x0_t_modiff = (x_modiff - et_modiff * (1 - at).sqrt()) / at.sqrt()
            x0_t_modiff = torch.clamp(x0_t_modiff, -1, 1)
            x_modiff_next = at_next.sqrt() * x0_t_modiff + c2 * et_modiff

        # --- Compare ---
        # Compare noise prediction (et)
        diff_et = (et_fp32 - et_modiff).abs()
        mse_et = (et_fp32 - et_modiff).pow(2).mean()
        max_diff_et = diff_et.max()
        
        # Compare next x
        diff_x = (x_fp32_next - x_modiff_next).abs()
        mse_x = (x_fp32_next - x_modiff_next).pow(2).mean()
        
        print(f"Step {step_idx} (t={t_val}):")
        print(f"  ET Diff: MSE={mse_et:.6f}, Max={max_diff_et:.6f}")
        print(f"  X  Diff: MSE={mse_x:.6f}")
        
        # Update x
        x_fp32 = x_fp32_next
        x_modiff = x_modiff_next
        
        if mse_et > 1.0:
            print("  WARNING: Large divergence detected!")

    print("Generation finished.")

if __name__ == "__main__":
    debug_generation_diff()
