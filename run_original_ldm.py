
import argparse
import os
import sys
import time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torchvision.utils as tvu

# Add current directory to path to find ldm
sys.path.append(os.getcwd())

# Try to import taming-transformers if not in path
try:
    import taming
except ImportError:
    # Assuming taming-transformers is cloned in /workspace/taming-transformers
    sys.path.append('/workspace/taming-transformers')
    try:
        import taming
    except ImportError:
        print("Warning: taming-transformers not found. Please install it or add to path.")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

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
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results/original_ldm')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eta', type=float, default=0.0)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
    # The original image size is 256. Downsampling factor 8 (kl-f8).
    # 256 / 8 = 32.
    shape = (4, 32, 32)
    
    print(f"\n--- Running Original LDM (FP32) ---")
    print(f"Steps: {args.steps}, Batch Size: {args.batch_size}, Eta: {args.eta}")
    
    start_time = time.time()
    
    # Unconditional sampling
    samples_fp32, _ = sampler.sample(S=args.steps,
                                     batch_size=args.batch_size,
                                     shape=shape,
                                     eta=args.eta,
                                     verbose=True)
    
    # Decode latents
    x_samples_fp32 = model.decode_first_stage(samples_fp32)
    x_samples_fp32 = torch.clamp((x_samples_fp32 + 1.0) / 2.0, min=0.0, max=1.0)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    fp32_time = end_time - start_time
    print(f"Execution Time: {fp32_time:.4f}s")
    
    # Save images
    print(f"Saving images to {args.output_dir}")
    for i in range(args.batch_size):
        tvu.save_image(x_samples_fp32[i], os.path.join(args.output_dir, f'sample_{i}.png'))
    
    print("Done.")

if __name__ == "__main__":
    main()
