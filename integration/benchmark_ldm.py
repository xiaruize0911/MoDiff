"""
Benchmark LDM with CUTLASS INT8 and FP16 acceleration.
"""
import argparse
import os
import sys
import time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torchvision.utils as tvu

# Add MoDiff to path
sys.path.append(os.getcwd())

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from integration.modiff_int8 import (
    convert_model_to_int8,
    convert_model_to_fp16,
    enable_modiff_mode,
    reset_modiff_state
)


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
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)
    model.cuda()
    model.eval()
    return model


def benchmark_sampling(sampler, steps, batch_size, shape, warmup=2, iterations=3):
    """Run sampling and return average time."""
    # Warmup
    for _ in range(warmup):
        sampler.sample(S=min(steps, 5), batch_size=batch_size, shape=shape, verbose=False)
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.time()
        samples, _ = sampler.sample(S=steps, batch_size=batch_size, shape=shape,
                                    eta=0.0, verbose=False)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    return np.mean(times), np.std(times), samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt', type=str, default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='integration/results_ldm_benchmark')
    parser.add_argument('--mode', type=str, choices=['all', 'fp32', 'fp16', 'int8'], default='all')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load Config
    conf = OmegaConf.load(args.config)
    
    # Shape for LDM latent space
    shape = (4, 32, 32)  # 256/8 = 32
    
    results = {}
    
    # =========================================================================
    # FP32 Baseline
    # =========================================================================
    if args.mode in ['all', 'fp32']:
        print("\n" + "="*60)
        print("FP32 Baseline")
        print("="*60)
        
        model = load_model_from_config(conf, args.ckpt)
        sampler = DDIMSampler(model)
        
        time_mean, time_std, samples = benchmark_sampling(
            sampler, args.steps, args.batch_size, shape
        )
        results['fp32'] = {'time': time_mean, 'std': time_std}
        print(f"FP32 Time: {time_mean:.3f} ± {time_std:.3f} s")
        
        # Decode and save
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
        
        fp32_dir = os.path.join(args.output_dir, 'fp32')
        os.makedirs(fp32_dir, exist_ok=True)
        for i in range(min(args.batch_size, 4)):
            tvu.save_image(x_samples[i], os.path.join(fp32_dir, f'{i}.png'))
        
        del model, sampler
        torch.cuda.empty_cache()
    
    # =========================================================================
    # FP16 (cuDNN Tensor Cores)
    # =========================================================================
    if args.mode in ['all', 'fp16']:
        print("\n" + "="*60)
        print("FP16 (cuDNN Tensor Cores)")
        print("="*60)
        
        model = load_model_from_config(conf, args.ckpt)
        
        # Convert UNet to FP16
        print("Converting UNet to FP16...")
        convert_model_to_fp16(model.model.diffusion_model)
        enable_modiff_mode(model.model.diffusion_model, True)
        
        sampler = DDIMSampler(model)
        
        time_mean, time_std, samples = benchmark_sampling(
            sampler, args.steps, args.batch_size, shape
        )
        results['fp16'] = {'time': time_mean, 'std': time_std}
        print(f"FP16 Time: {time_mean:.3f} ± {time_std:.3f} s")
        
        if 'fp32' in results:
            speedup = results['fp32']['time'] / time_mean
            print(f"Speedup vs FP32: {speedup:.2f}x")
        
        # Decode and save
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
        
        fp16_dir = os.path.join(args.output_dir, 'fp16')
        os.makedirs(fp16_dir, exist_ok=True)
        for i in range(min(args.batch_size, 4)):
            tvu.save_image(x_samples[i], os.path.join(fp16_dir, f'{i}.png'))
        
        del model, sampler
        torch.cuda.empty_cache()
    
    # =========================================================================
    # INT8 (CUTLASS Tensor Cores)
    # =========================================================================
    if args.mode in ['all', 'int8']:
        print("\n" + "="*60)
        print("INT8 (CUTLASS Tensor Cores)")
        print("="*60)
        
        model = load_model_from_config(conf, args.ckpt)
        
        # Convert UNet to INT8
        print("Converting UNet to CUTLASS INT8...")
        convert_model_to_int8(model.model.diffusion_model)
        enable_modiff_mode(model.model.diffusion_model, True)
        
        sampler = DDIMSampler(model)
        
        time_mean, time_std, samples = benchmark_sampling(
            sampler, args.steps, args.batch_size, shape
        )
        results['int8'] = {'time': time_mean, 'std': time_std}
        print(f"INT8 Time: {time_mean:.3f} ± {time_std:.3f} s")
        
        if 'fp32' in results:
            speedup = results['fp32']['time'] / time_mean
            print(f"Speedup vs FP32: {speedup:.2f}x")
        
        # Decode and save
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
        
        int8_dir = os.path.join(args.output_dir, 'int8')
        os.makedirs(int8_dir, exist_ok=True)
        for i in range(min(args.batch_size, 4)):
            tvu.save_image(x_samples[i], os.path.join(int8_dir, f'{i}.png'))
        
        del model, sampler
        torch.cuda.empty_cache()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if 'fp32' in results:
        print(f"FP32:  {results['fp32']['time']:.3f} ± {results['fp32']['std']:.3f} s (baseline)")
    
    if 'fp16' in results:
        fp16_speedup = results['fp32']['time'] / results['fp16']['time'] if 'fp32' in results else 1.0
        print(f"FP16:  {results['fp16']['time']:.3f} ± {results['fp16']['std']:.3f} s ({fp16_speedup:.2f}x)")
    
    if 'int8' in results:
        int8_speedup = results['fp32']['time'] / results['int8']['time'] if 'fp32' in results else 1.0
        print(f"INT8:  {results['int8']['time']:.3f} ± {results['int8']['std']:.3f} s ({int8_speedup:.2f}x)")
    
    print("="*60)


if __name__ == '__main__':
    main()
