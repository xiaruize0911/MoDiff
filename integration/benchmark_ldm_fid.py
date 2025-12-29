
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ViDiT-Q/quant_utils')))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from modiff_utils import convert_model_to_modiff, enable_modiff_mode, reset_modiff_state

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = load_model_from_config(config.model, sd)
    return model

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../MoDiff/models/ldm/lsun_churches256/config.yaml')
    parser.add_argument('--ckpt', type=str, default='../../MoDiff/models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--batch_size', type=int, default=10) 
    parser.add_argument('--n_samples', type=int, default=100) # Total samples
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results_ldm')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.config, args.ckpt)
    
    # Initialize Sampler
    sampler = DDIMSampler(model)
    
    # Shape for LDM (latent shape)
    # Config says image_size: 32, channels: 4 (latent)
    # But wait, config says:
    # unet_config: params: image_size: 32, in_channels: 4
    # This is latent size. Real image size is 256 (downsampling factor 8).
    shape = (4, 32, 32)
    
    # 1. Benchmark FP32
    print("\n--- Benchmarking FP32 (Standard) ---")
    fp32_dir = os.path.join(args.output_dir, 'fp32')
    os.makedirs(fp32_dir, exist_ok=True)
    
    start_time = time.time()
    n_saved = 0
    while n_saved < args.n_samples:
        bs = min(args.batch_size, args.n_samples - n_saved)
        samples, _ = sampler.sample(S=args.steps, batch_size=bs, shape=shape, eta=0.0, verbose=False)
        
        # Decode
        x_samples = model.decode_first_stage(samples)
        
        for x in x_samples:
            img = custom_to_pil(x)
            img.save(os.path.join(fp32_dir, f'{n_saved}.png'))
            n_saved += 1
        print(f"FP32: Generated {n_saved}/{args.n_samples}")
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    fp32_time = end_time - start_time
    print(f"FP32 Time: {fp32_time:.4f}s")
    
    # 2. Benchmark MoDiff (W8A8)
    print("\n--- Benchmarking MoDiff (W8A8) ---")
    modiff_dir = os.path.join(args.output_dir, 'modiff')
    os.makedirs(modiff_dir, exist_ok=True)
    
    # Convert model
    # Note: LDM has UNet in model.model.diffusion_model
    # But my wrapper recursively converts.
    # However, LDM also has First Stage Model (Autoencoder) which we should NOT quantize/modulate usually, 
    # or maybe we should? The paper focuses on Diffusion Model.
    # The diffusion model is `model.model.diffusion_model`.
    # I should only convert that.
    
    print("Converting Diffusion Model to MoDiff...")
    convert_model_to_modiff(model.model.diffusion_model, use_cuda_kernel=True)
    enable_modiff_mode(model.model.diffusion_model, True)
    
    start_time = time.time()
    n_saved = 0
    while n_saved < args.n_samples:
        bs = min(args.batch_size, args.n_samples - n_saved)
        
        # Reset MoDiff state before each batch?
        # No, reset before each SAMPLE process (which is handled inside sampler loop if we modified sampler).
        # But my wrapper handles "first step" logic.
        # However, if we run multiple batches, we need to reset state between batches?
        # Wait, `sampler.sample` runs the full diffusion loop (T -> 0).
        # Inside `sample`, it calls `p_sample_ddim` T times.
        # My wrapper:
        # Step 1 (T): last_input is None -> Compute full -> Save state.
        # Step 2 (T-1): last_input is set -> Compute delta -> Update state.
        # ...
        # Step T (0): ...
        # After `sample` returns, `last_input` is still set (from t=0).
        # If we call `sample` again (next batch), we start at T again.
        # We MUST reset state before next batch!
        # Also, inside `sample`, we start at T.
        # So `reset_modiff_state` should be called before `sampler.sample`.
        
        reset_modiff_state(model.model.diffusion_model)
        
        samples, _ = sampler.sample(S=args.steps, batch_size=bs, shape=shape, eta=0.0, verbose=False)
        
        # Decode (First stage model is NOT quantized/modulated, so it works as is)
        x_samples = model.decode_first_stage(samples)
        
        for x in x_samples:
            img = custom_to_pil(x)
            img.save(os.path.join(modiff_dir, f'{n_saved}.png'))
            n_saved += 1
        print(f"MoDiff: Generated {n_saved}/{args.n_samples}")
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    modiff_time = end_time - start_time
    print(f"MoDiff Time: {modiff_time:.4f}s")
    
    print(f"\nSpeedup: {fp32_time / modiff_time:.2f}x")
    
    # 3. FID Calculation
    print("\n--- Calculating FID ---")
    try:
        # Compare FP32 vs MoDiff
        metrics = calculate_metrics(input1=fp32_dir, input2=modiff_dir, cuda=True, fid=True, verbose=False)
        print(f"FID (FP32 vs MoDiff): {metrics['frechet_inception_distance']:.4f}")
        print("Note: Low FID means MoDiff samples are very similar to FP32 samples (high fidelity preservation).")
        
    except Exception as e:
        print(f"FID calculation failed: {e}")

if __name__ == '__main__':
    main()
