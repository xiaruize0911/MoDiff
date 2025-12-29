
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

from MoDiff.ddim.models.diffusion import Model
from modiff_utils import convert_model_to_modiff, enable_modiff_mode, reset_modiff_state

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
    return betas

class DDIMSampler:
    def __init__(self, model, beta_schedule="linear", beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000):
        self.model = model
        self.num_timesteps = num_diffusion_timesteps
        betas = get_beta_schedule(beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # Convert to torch
        self.alphas_cumprod = torch.from_numpy(self.alphas_cumprod).float().to(next(model.parameters()).device)
        self.alphas_cumprod_prev = torch.from_numpy(self.alphas_cumprod_prev).float().to(next(model.parameters()).device)

    @torch.no_grad()
    def sample(self, batch_size, shape, steps=50, eta=0.0):
        device = next(self.model.parameters()).device
        b = batch_size
        img = torch.randn((b, *shape), device=device)
        
        # Time steps
        # Simple uniform spacing
        c = self.num_timesteps // steps
        timesteps = np.asarray(list(range(0, self.num_timesteps, c))) + 1
        timesteps = np.flip(timesteps) # [1000, ..., 1] roughly
        
        # Adjust to 0-index
        timesteps = timesteps - 1
        
        print(f"Sampling with {len(timesteps)} steps...")
        
        for i, step in enumerate(timesteps):
            # Reset MoDiff state at the beginning of each step? 
            # No, MoDiff state is maintained ACROSS steps if we are doing the modulation.
            # Wait, the paper says:
            # o_t = A(a_t - a_{t+1}) + o_{t+1}
            # This means we compute t+1 first, then t.
            # Diffusion goes T -> T-1 -> ... -> 0.
            # So we are going "forward" in time index (reverse diffusion).
            # T is noise. 0 is image.
            # Paper Eq 1:
            # o_T = A(a_T)
            # o_{T-1} = A(a_{T-1} - a_T) + o_T
            # So we need to keep state from PREVIOUS iteration of the loop.
            
            # In my wrapper:
            # last_input is a_{t+1} (previous step in loop)
            # x is a_t (current step in loop)
            # So wrapper logic: delta = x - last_input.
            # This matches a_t - a_{t+1}.
            
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            # Model prediction (noise)
            # e_t = model(x_t, t)
            # Here 'img' is x_t.
            e_t = self.model(img, ts)
            
            # DDIM update
            # alpha_t
            alpha_t = self.alphas_cumprod[step]
            alpha_prev = self.alphas_cumprod_prev[step] # Actually we need alpha at next step in sequence
            
            # Find next step (t-1 in diffusion, next in loop)
            if i < len(timesteps) - 1:
                step_next = timesteps[i+1]
                alpha_next = self.alphas_cumprod[step_next]
            else:
                alpha_next = torch.tensor(1.0, device=device) # t=0 -> alpha=1
            
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            
            # Predict x0
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * e_t) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_next - sigma_t**2) * e_t
            
            # Update x_{t-1}
            noise = torch.randn_like(img)
            img_prev = torch.sqrt(alpha_next) * pred_x0 + dir_xt + sigma_t * noise
            
            img = img_prev
            
        return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../MoDiff/configs/cifar10.yml')
    parser.add_argument('--batch_size', type=int, default=16) # Small batch for speed
    parser.add_argument('--steps', type=int, default=20) # Fewer steps for benchmark speed
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Config
    conf = OmegaConf.load(args.config)
    
    # Initialize Model
    # Config structure: model: type: simple, ...
    # Model class expects config object
    print("Initializing model...")
    model = Model(conf)
    model.to(device)
    model.eval()
    
    # Initialize Sampler
    sampler = DDIMSampler(model, 
                          beta_schedule=conf.diffusion.beta_schedule,
                          beta_start=conf.diffusion.beta_start,
                          beta_end=conf.diffusion.beta_end,
                          num_diffusion_timesteps=conf.diffusion.num_diffusion_timesteps)
    
    # 1. Benchmark FP32 (Standard)
    print("\n--- Benchmarking FP32 (Standard) ---")
    start_time = time.time()
    samples_fp32 = sampler.sample(args.batch_size, (3, 32, 32), steps=args.steps)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    fp32_time = end_time - start_time
    print(f"FP32 Time: {fp32_time:.4f}s")
    
    # Save FP32 images
    fp32_dir = os.path.join(args.output_dir, 'fp32')
    os.makedirs(fp32_dir, exist_ok=True)
    for i in range(args.batch_size):
        tvu.save_image(samples_fp32[i], os.path.join(fp32_dir, f'{i}.png'), normalize=True)
        
    # 2. Benchmark MoDiff (W8A8)
    print("\n--- Benchmarking MoDiff (W8A8) ---")
    # Convert model
    convert_model_to_modiff(model, use_cuda_kernel=True)
    enable_modiff_mode(model, True)
    
    # Reset state before sampling
    reset_modiff_state(model)
    
    start_time = time.time()
    # Note: We need to ensure reset_modiff_state is called at the start of sampling.
    # Since my simple sampler doesn't know about MoDiff, I call it here.
    # But wait, MoDiff state needs to be reset ONLY at the very beginning of the diffusion process (T).
    # My wrapper handles "if last_input is None" -> First step.
    # So calling reset_modiff_state() once before sample() is correct.
    
    samples_modiff = sampler.sample(args.batch_size, (3, 32, 32), steps=args.steps)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    modiff_time = end_time - start_time
    print(f"MoDiff Time: {modiff_time:.4f}s")
    
    # Save MoDiff images
    modiff_dir = os.path.join(args.output_dir, 'modiff')
    os.makedirs(modiff_dir, exist_ok=True)
    for i in range(args.batch_size):
        tvu.save_image(samples_modiff[i], os.path.join(modiff_dir, f'{i}.png'), normalize=True)
        
    print(f"\nSpeedup: {fp32_time / modiff_time:.2f}x")
    
    # 3. FID Calculation
    print("\n--- Calculating FID ---")
    # We need a reference. torch-fidelity can download CIFAR-10.
    # We will compare 'fp32_dir' and 'modiff_dir' to 'cifar10-train'.
    # Note: With random weights, FID will be high.
    
    try:
        metrics_fp32 = calculate_metrics(input1=fp32_dir, input2='cifar10-train', cuda=True, fid=True, verbose=False)
        print(f"FP32 FID: {metrics_fp32['frechet_inception_distance']:.4f}")
        
        metrics_modiff = calculate_metrics(input1=modiff_dir, input2='cifar10-train', cuda=True, fid=True, verbose=False)
        print(f"MoDiff FID: {metrics_modiff['frechet_inception_distance']:.4f}")
    except Exception as e:
        print(f"FID calculation failed: {e}")
        print("Note: FID requires internet to download CIFAR-10 and Inception weights.")

if __name__ == '__main__':
    main()
