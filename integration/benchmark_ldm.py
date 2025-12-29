
import argparse
import os
import sys
import time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torchvision.utils as tvu
from torch_fidelity import calculate_metrics

# Add MoDiff to path
sys.path.append(os.getcwd())

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from integration.modiff_utils import convert_model_to_modiff, enable_modiff_mode, reset_modiff_state

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt', type=str, default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='integration/results_ldm')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Config
    conf = OmegaConf.load(args.config)
    
    # Load Model
    model = load_model_from_config(conf, args.ckpt)
    
    # Initialize Sampler
    sampler = DDIMSampler(model)
    
    # Shape for LDM (latent shape)
    # Config says image_size: 32, channels: 4 (latent)
    # But wait, image_size in UNet config is 32.
    # The original image size is 256. Downsampling factor 8 (kl-f8).
    # 256 / 8 = 32. Correct.
    shape = (4, 32, 32)
    
    # 1. Benchmark FP32 (Standard)
    print("\n--- Benchmarking FP32 (Standard) ---")
    start_time = time.time()
    # ddim_sampling(self, cond, shape, ...)
    # Unconditional sampling
    samples_fp32, _ = sampler.sample(S=args.steps,
                                     batch_size=args.batch_size,
                                     shape=shape,
                                     eta=0.0,
                                     verbose=False)
    
    # Decode latents
    x_samples_fp32 = model.decode_first_stage(samples_fp32)
    x_samples_fp32 = torch.clamp((x_samples_fp32 + 1.0) / 2.0, min=0.0, max=1.0)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    fp32_time = end_time - start_time
    print(f"FP32 Time: {fp32_time:.4f}s")
    
    # Save FP32 images
    fp32_dir = os.path.join(args.output_dir, 'fp32')
    os.makedirs(fp32_dir, exist_ok=True)
    for i in range(args.batch_size):
        tvu.save_image(x_samples_fp32[i], os.path.join(fp32_dir, f'{i}.png'))
        
    # 2. Benchmark MoDiff (W8A8)
    print("\n--- Benchmarking MoDiff (W8A8) ---")
    # Convert model
    # Note: We only want to convert the UNet (diffusion_model), not the Autoencoder (first_stage_model)
    # Try CUDA kernel first, if fails, fallback? 
    # For now, let's try False to verify logic.
    use_cuda = False
    print(f"Converting model to MoDiff (CUDA={use_cuda})...")
    convert_model_to_modiff(model.model.diffusion_model, use_cuda_kernel=use_cuda)
    enable_modiff_mode(model.model.diffusion_model, True)
    
    # Reset state before sampling
    reset_modiff_state(model.model.diffusion_model)
    
    start_time = time.time()
    
    samples_modiff, _ = sampler.sample(S=args.steps,
                                       batch_size=args.batch_size,
                                       shape=shape,
                                       eta=0.0,
                                       verbose=False)
                                       
    # Decode latents
    x_samples_modiff = model.decode_first_stage(samples_modiff)
    x_samples_modiff = torch.clamp((x_samples_modiff + 1.0) / 2.0, min=0.0, max=1.0)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    modiff_time = end_time - start_time
    print(f"MoDiff Time: {modiff_time:.4f}s")
    
    # Save MoDiff images
    modiff_dir = os.path.join(args.output_dir, 'modiff')
    os.makedirs(modiff_dir, exist_ok=True)
    for i in range(args.batch_size):
        tvu.save_image(x_samples_modiff[i], os.path.join(modiff_dir, f'{i}.png'))
        
    print(f"\nSpeedup: {fp32_time / modiff_time:.2f}x")
    
    # 3. FID Calculation (Relative)
    print("\n--- Calculating Relative FID (FP32 vs MoDiff) ---")
    try:
        metrics = calculate_metrics(input1=modiff_dir, input2=fp32_dir, cuda=True, fid=True, verbose=False)
        print(f"Relative FID (MoDiff vs FP32): {metrics['frechet_inception_distance']:.4f}")
        print("Note: Lower is better (closer to FP32).")
    except Exception as e:
        print(f"FID calculation failed: {e}")

if __name__ == '__main__':
    main()
