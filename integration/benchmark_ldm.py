"""
Benchmark LDM with different precision modes.

Modes:
- FP32: Standard FP32 baseline

- FP16: Half precision with autocast
- INT8: CUTLASS INT8 with MoDiff modulation
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

# Optional MoDiff imports
try:
    from integration.modiff_layers import (
        convert_model_to_int8_fused,
        enable_modiff_mode,
        reset_modiff_state
    )
    HAS_MODIFF = True
except ImportError:
    HAS_MODIFF = False
    print("MoDiff layers not available")


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
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


def benchmark_sampling(sampler, steps, batch_size, shape, warmup=2, iterations=3, 
                      model=None, use_autocast=False, autocast_dtype=torch.float16):
    """Run sampling and return average time."""
    # Warmup
    for _ in range(warmup):
        if model is not None and HAS_MODIFF:
            reset_modiff_state(model.model.diffusion_model)
        with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
            sampler.sample(S=min(steps, 5), batch_size=batch_size, shape=shape, verbose=False)
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        if model is not None and HAS_MODIFF:
            reset_modiff_state(model.model.diffusion_model)
        torch.cuda.synchronize()
        start = time.time()
        with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
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
    parser.add_argument('--mode', type=str, choices=['all', 'fp32', 'fp16', 'bf16', 'int8'], default='all')
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
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        model = load_model_from_config(conf, args.ckpt)
        sampler = DDIMSampler(model)
        
        time_mean, time_std, samples = benchmark_sampling(
            sampler, args.steps, args.batch_size, shape
        )
        results['fp32'] = {'time': time_mean, 'std': time_std}
        print(f"FP32 Time: {time_mean:.3f} ± {time_std:.3f} s")
        print(f"  Per-step: {time_mean/args.steps*1000:.2f} ms")
        
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
    # FP16 with autocast
    # =========================================================================
    if args.mode in ['all', 'fp16']:
        print("\n" + "="*60)
        print("FP16 (torch.cuda.amp.autocast)")
        print("="*60)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        model = load_model_from_config(conf, args.ckpt)
        sampler = DDIMSampler(model)
        
        time_mean, time_std, samples = benchmark_sampling(
            sampler, args.steps, args.batch_size, shape,
            use_autocast=True, autocast_dtype=torch.float16
        )
        results['fp16'] = {'time': time_mean, 'std': time_std}
        print(f"FP16 Time: {time_mean:.3f} ± {time_std:.3f} s")
        print(f"  Per-step: {time_mean/args.steps*1000:.2f} ms")
        
        if 'fp32' in results:
            speedup = results['fp32']['time'] / time_mean
            print(f"Speedup vs FP32: {speedup:.2f}x")
        
        # Decode and save
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples.float() + 1.0) / 2.0, 0.0, 1.0)
        
        fp16_dir = os.path.join(args.output_dir, 'fp16')
        os.makedirs(fp16_dir, exist_ok=True)
        for i in range(min(args.batch_size, 4)):
            tvu.save_image(x_samples[i], os.path.join(fp16_dir, f'{i}.png'))
        
        del model, sampler
        torch.cuda.empty_cache()
    
    # =========================================================================
    # BF16 with autocast (if supported)
    # =========================================================================
    if args.mode in ['all', 'bf16']:
        print("\n" + "="*60)
        print("BF16 (torch.cuda.amp.autocast)")
        print("="*60)
        
        if not torch.cuda.is_bf16_supported():
            print("BF16 not supported on this GPU, skipping...")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            model = load_model_from_config(conf, args.ckpt)
            sampler = DDIMSampler(model)
            
            time_mean, time_std, samples = benchmark_sampling(
                sampler, args.steps, args.batch_size, shape,
                use_autocast=True, autocast_dtype=torch.bfloat16
            )
            results['bf16'] = {'time': time_mean, 'std': time_std}
            print(f"BF16 Time: {time_mean:.3f} ± {time_std:.3f} s")
            print(f"  Per-step: {time_mean/args.steps*1000:.2f} ms")
            
            if 'fp32' in results:
                speedup = results['fp32']['time'] / time_mean
                print(f"Speedup vs FP32: {speedup:.2f}x")
            
            # Decode and save
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples.float() + 1.0) / 2.0, 0.0, 1.0)
            
            bf16_dir = os.path.join(args.output_dir, 'bf16')
            os.makedirs(bf16_dir, exist_ok=True)
            for i in range(min(args.batch_size, 4)):
                tvu.save_image(x_samples[i], os.path.join(bf16_dir, f'{i}.png'))
            
            del model, sampler
            torch.cuda.empty_cache()
    
    # =========================================================================
    # INT8 MoDiff (CUTLASS + Error Compensation)
    # =========================================================================
    if args.mode in ['all', 'int8'] and HAS_MODIFF:
        print("\n" + "="*60)
        print("INT8 MoDiff (CUTLASS + Error Compensation)")
        print("="*60)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        model = load_model_from_config(conf, args.ckpt)
        
        print("Converting UNet to INT8 MoDiff...")
        convert_model_to_int8_fused(model.model.diffusion_model)
        enable_modiff_mode(model.model.diffusion_model, True)
        
        sampler = DDIMSampler(model)
        
        time_mean, time_std, samples = benchmark_sampling(
            sampler, args.steps, args.batch_size, shape, model=model
        )
        results['int8'] = {'time': time_mean, 'std': time_std}
        print(f"INT8 Time: {time_mean:.3f} ± {time_std:.3f} s")
        print(f"  Per-step: {time_mean/args.steps*1000:.2f} ms")
        
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
    
    baseline = results.get('fp32', {}).get('time', 1.0)
    
    for mode in ['fp32', 'fp16', 'bf16', 'int8']:
        if mode in results:
            t = results[mode]['time']
            std = results[mode]['std']
            speedup = baseline / t if t > 0 else 0
            per_step = t / args.steps * 1000
            label = "(baseline)" if mode == 'fp32' else f"({speedup:.2f}x)"
            print(f"{mode.upper():8s}: {t:.3f} ± {std:.3f} s | {per_step:.2f} ms/step {label}")
    
    print("="*60)
    print(f"\nImages saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
