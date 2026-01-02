"""
Unified LDM Benchmark with Multiple Precision Modes.

Supports:
- FP32: Standard baseline
- FP16: Half precision with autocast
- INT8: Optimized CUTLASS INT8 with static quantization

Usage:
    python integration/benchmark_ldm.py --mode all --steps 50
    python integration/benchmark_ldm.py --mode int8 --num_samples 100 --eval_fid
"""
import argparse
import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import torchvision.utils as tvu
from tqdm import tqdm

# Disable TF32 globally for consistent benchmarking
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.append(os.getcwd())

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# INT8 imports
try:
    from integration.int8_optimized import (
        OptimizedInt8Conv2d,
        convert_model_to_optimized_int8,
        enable_modiff_mode as enable_modiff_mode_int8,
        reset_modiff_state as reset_modiff_state_int8,
        set_calibrating as set_calibrating_int8,
        get_calibration_config as get_calibration_config_int8,
        reset_calibration as reset_calibration_int8,
    )
    HAS_INT8 = True
except ImportError:
    HAS_INT8 = False
    print("Warning: INT8 not available")

# INT4 imports
try:
    from integration.int4_optimized import (
        OptimizedInt4Conv2d,
        convert_model_to_optimized_int4,
        enable_modiff_mode as enable_modiff_mode_int4,
        reset_modiff_state as reset_modiff_state_int4,
        set_calibrating as set_calibrating_int4,
        get_calibration_config as get_calibration_config_int4,
        reset_calibration as reset_calibration_int4,
    )
    HAS_INT4 = True
except ImportError:
    HAS_INT4 = False
    print("Warning: INT4 not available")

# FID (optional)
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False


def load_model(config_path: str, ckpt_path: str, verbose: bool = False):
    """Load LDM model from config and checkpoint."""
    print(f"Loading model from {ckpt_path}")
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    
    model = instantiate_from_config(conf.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose and m:
        print("Missing keys:", m)
    if verbose and u:
        print("Unexpected keys:", u)
    
    return model.cuda().eval(), conf


def count_conv_layers(model: nn.Module) -> int:
    """Count Conv2d layers in model."""
    return sum(1 for m in model.modules() if isinstance(m, (nn.Conv2d, OptimizedInt8Conv2d, OptimizedInt4Conv2d)))


class BenchmarkRunner:
    """Unified benchmark runner for all precision modes."""
    
    def __init__(self, config_path: str, ckpt_path: str, output_dir: str,
                 batch_size: int = 4, steps: int = 50, shape: tuple = (4, 32, 32)):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.steps = steps
        self.shape = shape
        self.results = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _setup_model(self, mode: str):
        """Load and setup model for given mode."""
        model, _ = load_model(self.config_path, self.ckpt_path)
        
        # Configure backends
        # Disable TF32 to ensure pure FP32 baseline and consistent comparisons
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        
        if mode == 'int8' and HAS_INT8:
            print(f"Converting UNet to INT8 ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int8(model.model.diffusion_model)
            enable_modiff_mode_int8(model.model.diffusion_model, True)
        elif mode == 'int4' and HAS_INT4:
            print(f"Converting UNet to INT4 ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int4(model.model.diffusion_model)
            enable_modiff_mode_int4(model.model.diffusion_model, True)
        
        return model, DDIMSampler(model)
    
    def _calibrate_int8(self, model, sampler, num_runs: int = 10):
        """Calibrate INT8 quantization scales."""
        print(f"Calibrating INT8 ({num_runs} runs)...")
        reset_calibration_int8()
        set_calibrating_int8(model.model.diffusion_model, True)
        
        with torch.no_grad():
            for _ in range(num_runs):
                reset_modiff_state_int8(model.model.diffusion_model)
                sampler.sample(S=5, batch_size=2, shape=self.shape, eta=0.0, verbose=False)
        
        get_calibration_config_int8().finalize()
        set_calibrating_int8(model.model.diffusion_model, False)
        print(f"Calibrated {len(get_calibration_config_int8().scales)} layers")
    
    def _calibrate_int4(self, model, sampler, num_runs: int = 10):
        """Calibrate INT4 quantization scales."""
        print(f"Calibrating INT4 ({num_runs} runs)...")
        reset_calibration_int4()
        set_calibrating_int4(model.model.diffusion_model, True)
        
        with torch.no_grad():
            for _ in range(num_runs):
                reset_modiff_state_int4(model.model.diffusion_model)
                sampler.sample(S=5, batch_size=2, shape=self.shape, eta=0.0, verbose=False)
        
        get_calibration_config_int4().finalize()
        set_calibrating_int4(model.model.diffusion_model, False)
        print(f"Calibrated {len(get_calibration_config_int4().scales)} layers")
    
    def _generate_samples(self, model, sampler, mode: str, num_samples: int,
                          use_autocast: bool = False, dtype: torch.dtype = None):
        """Generate samples and measure time."""
        mode_dir = os.path.join(self.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        
        total_time = 0.0
        generated = 0
        
        pbar = tqdm(total=num_samples, desc=f"Generating {mode}")
        while generated < num_samples:
            batch = min(self.batch_size, num_samples - generated)
            
            if mode == 'int8' and HAS_INT8:
                reset_modiff_state_int8(model.model.diffusion_model)
            elif mode == 'int4' and HAS_INT4:
                reset_modiff_state_int4(model.model.diffusion_model)
            
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
                samples, _ = sampler.sample(S=self.steps, batch_size=batch,
                                           shape=self.shape, eta=0.0, verbose=False)
            
            torch.cuda.synchronize()
            total_time += time.time() - start
            
            # Decode and save
            with torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
                x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples.float() + 1.0) / 2.0, 0.0, 1.0)
            
            for i in range(batch):
                tvu.save_image(x_samples[i], os.path.join(mode_dir, f'{generated + i:05d}.png'))
            
            generated += batch
            pbar.update(batch)
        
        pbar.close()
        return total_time, generated
    
    def run_mode(self, mode: str, num_samples: int = 16, calibrate: bool = True):
        """Run benchmark for a specific mode."""
        print(f"\n{'='*60}\n{mode.upper()}\n{'='*60}")
        
        model, sampler = self._setup_model(mode)
        
        # INT8/INT4 calibration
        if mode == 'int8' and HAS_INT8 and calibrate:
            self._calibrate_int8(model, sampler)
        elif mode == 'int4' and HAS_INT4 and calibrate:
            self._calibrate_int4(model, sampler)
        
        # Configure autocast
        use_autocast = mode == 'fp16'
        dtype = torch.float16 if mode == 'fp16' else None
        
        # Generate samples
        total_time, num_gen = self._generate_samples(
            model, sampler, mode, num_samples, use_autocast, dtype
        )
        
        # Record results
        time_per_sample = total_time / num_gen
        time_per_step = total_time / (num_gen * self.steps) * 1000
        
        self.results[mode] = {
            'total_time': total_time,
            'num_samples': num_gen,
            'time_per_sample': time_per_sample,
            'time_per_step_ms': time_per_step,
        }
        
        print(f"\n{mode.upper()} Results:")
        print(f"  Total: {total_time:.2f}s for {num_gen} samples")
        print(f"  Per-sample: {time_per_sample:.3f}s")
        print(f"  Per-step: {time_per_step:.2f}ms")
        
        if 'fp32' in self.results and mode != 'fp32':
            speedup = self.results['fp32']['time_per_sample'] / time_per_sample
            self.results[mode]['speedup'] = speedup
            print(f"  Speedup vs FP32: {speedup:.2f}x")
        
        del model, sampler
        torch.cuda.empty_cache()
    
    def compute_fid(self, reference_mode: str = 'fp32'):
        """Compute FID between modes."""
        if not HAS_FID:
            print("pytorch_fid not available")
            return
        
        ref_dir = os.path.join(self.output_dir, reference_mode)
        if not os.path.exists(ref_dir):
            print(f"Reference directory not found: {ref_dir}")
            return
        
        print(f"\nFID (vs {reference_mode}):")
        for mode in self.results:
            if mode == reference_mode:
                continue
            mode_dir = os.path.join(self.output_dir, mode)
            if os.path.exists(mode_dir):
                try:
                    fid = fid_score.calculate_fid_given_paths(
                        [ref_dir, mode_dir], batch_size=50, device='cuda', dims=2048
                    )
                    self.results[mode]['fid'] = fid
                    print(f"  {mode}: {fid:.2f}")
                except Exception as e:
                    print(f"  {mode}: Error - {e}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(f"\n{'Mode':<15} {'Time/Sample':<15} {'Speedup':<12} {'FID':<10}")
        print("-" * 55)
        
        baseline = self.results.get('fp32', {}).get('time_per_sample', 1.0)
        for mode in ['fp32', 'fp16', 'int8', 'int4']:
            if mode in self.results:
                r = self.results[mode]
                t = f"{r['time_per_sample']:.3f}s"
                speedup = baseline / r['time_per_sample'] if r['time_per_sample'] > 0 else 0
                s = "(baseline)" if mode == 'fp32' else f"{speedup:.2f}x"
                fid = f"{r.get('fid', '-'):.2f}" if 'fid' in r else "-"
                print(f"{mode:<15} {t:<15} {s:<12} {fid:<10}")
        
        # Save results
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="LDM Benchmark")
    parser.add_argument('--config', type=str, default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt', type=str, default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--output_dir', type=str, default='integration/results_ldm_benchmark')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--mode', type=str, choices=['all', 'fp32', 'fp16', 'int8', 'int4'], default='all')
    parser.add_argument('--eval_fid', action='store_true', help='Compute FID between modes')
    parser.add_argument('--skip_calibration', action='store_true')
    args = parser.parse_args()
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: {args.config}")
    print(f"Steps: {args.steps} | Batch: {args.batch_size} | Samples: {args.num_samples}")
    
    runner = BenchmarkRunner(
        args.config, args.ckpt, args.output_dir,
        args.batch_size, args.steps, shape=(4, 32, 32)
    )
    
    modes = ['fp32', 'fp16', 'int8', 'int4'] if args.mode == 'all' else [args.mode]
    
    for mode in modes:
        if mode == 'int8' and not HAS_INT8:
            print(f"Skipping {mode}: not available")
            continue
        if mode == 'int4' and not HAS_INT4:
            print(f"Skipping {mode}: not available")
            continue
        runner.run_mode(mode, args.num_samples, calibrate=not args.skip_calibration)
    
    if args.eval_fid and len(runner.results) > 1:
        runner.compute_fid()
    
    runner.print_summary()


if __name__ == '__main__':
    main()
