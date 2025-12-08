#!/usr/bin/env python3
"""
Generate samples using TensorRT FP32, INT8, and INT4 engines for visual comparison.

This script runs the full DDIM sampling process with each engine variant
and saves the generated images for quality comparison.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DDIMSampler:
    """DDIM sampler for diffusion models."""
    
    def __init__(self, num_timesteps: int = 1000, ddim_steps: int = 50,
                 beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.ddim_steps = ddim_steps
        
        # Linear beta schedule
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        
        # DDIM timesteps (uniformly spaced)
        self.ddim_timesteps = np.asarray(
            list(range(0, num_timesteps, num_timesteps // ddim_steps))
        )
        
        # Precompute alpha values at DDIM timesteps
        self.ddim_alphas = self.alphas_cumprod[self.ddim_timesteps]
        self.ddim_alphas_prev = np.append(1.0, self.ddim_alphas[:-1])
        
        # Compute sigma for DDIM (eta=0 for deterministic)
        self.ddim_sigmas = np.zeros_like(self.ddim_alphas)
    
    def get_schedule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return DDIM schedule parameters."""
        return self.ddim_timesteps, self.ddim_alphas, self.ddim_alphas_prev

class TRTModelRunner:
    """Run TensorRT engine for sampling."""
    
    def __init__(self, engine_path: str, name: str = "TRT"):
        self.name = name
        self.device = 'cuda'
        self.batch_size = 1  # Will be updated based on engine
        
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        logger.info(f"Loading {name} engine from {engine_path}")
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        self.stream = cuda.Stream()
        self._setup_buffers()
        
        logger.info(f"Loaded {name} engine ({len(engine_data) / (1024*1024):.1f} MB, batch={self.batch_size})")
    
    def _setup_buffers(self):
        """Allocate input/output buffers."""
        import tensorrt as trt
        import pycuda.driver as cuda
        
        self.inputs = []
        self.outputs = []
        self.has_dynamic_batch = False
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Check for dynamic dimensions
            if shape[0] == -1:
                self.has_dynamic_batch = True
                shape = tuple(16 if s == -1 else s for s in shape)
            else:
                # Static batch - use engine's batch size
                if name == 'latent':
                    self.batch_size = shape[0]
            
            size = int(np.prod(shape))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            tensor_info = {
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'host': host_mem,
                'device': device_mem,
            }
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
            else:
                self.outputs.append(tensor_info)
    
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run TensorRT inference. Handles batching for static batch engines."""
        import pycuda.driver as cuda
        
        batch_size = x.shape[0]
        
        # If engine has static batch size and input is larger, process in chunks
        if not self.has_dynamic_batch and batch_size > self.batch_size:
            outputs = []
            for i in range(0, batch_size, self.batch_size):
                end = min(i + self.batch_size, batch_size)
                chunk_x = x[i:end]
                chunk_t = t[i:end]
                
                # Pad if needed
                if chunk_x.shape[0] < self.batch_size:
                    pad_size = self.batch_size - chunk_x.shape[0]
                    chunk_x = torch.cat([chunk_x, chunk_x[:pad_size]], dim=0)
                    chunk_t = torch.cat([chunk_t, chunk_t[:pad_size]], dim=0)
                
                out = self._run_inference(chunk_x, chunk_t)
                outputs.append(out[:end - i])
            
            return torch.cat(outputs, dim=0)
        else:
            return self._run_inference(x, t)
    
    def _run_inference(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run single inference pass."""
        import pycuda.driver as cuda
        
        latent = x.cpu().numpy().astype(np.float32)
        timesteps = t.cpu().numpy().astype(np.int32)
        
        # Set input shapes for dynamic dimensions
        if self.has_dynamic_batch:
            for inp in self.inputs:
                if inp['name'] == 'latent':
                    self.context.set_input_shape('latent', latent.shape)
                elif inp['name'] == 'timesteps':
                    self.context.set_input_shape('timesteps', timesteps.shape)
        
        # Copy inputs
        for inp in self.inputs:
            if inp['name'] == 'latent':
                np.copyto(inp['host'][:latent.size], latent.flatten())
            elif inp['name'] == 'timesteps':
                np.copyto(inp['host'][:timesteps.size], timesteps.flatten())
        
        # Transfer to device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        
        # Execute
        self.context.execute_async_v3(self.stream.handle)
        
        # Copy output back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        # Return as tensor
        output_shape = latent.shape
        output = self.outputs[0]['host'][:np.prod(output_shape)].reshape(output_shape)
        return torch.from_numpy(output.copy()).to(x.device)


def ddim_sample(model, sampler: DDIMSampler, shape: Tuple[int, ...],
                device: str = 'cuda') -> torch.Tensor:
    """
    Run DDIM sampling to generate images.
    
    Args:
        model: Model callable (FP32 or TRT)
        sampler: DDIMSampler instance
        shape: Output shape (batch, channels, height, width)
        device: Device to run on
        
    Returns:
        Generated samples tensor
    """
    batch_size, channels, height, width = shape
    
    # Start from random noise
    x = torch.randn(shape, device=device)
    
    timesteps, alphas, alphas_prev = sampler.get_schedule()
    
    # Reverse diffusion process
    for i, step in enumerate(reversed(range(len(timesteps)))):
        t = timesteps[step]
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = model(x, t_tensor)
            if isinstance(noise_pred, torch.Tensor):
                noise_pred = noise_pred.cpu()
            x_cpu = x.cpu()
        
        # DDIM update
        alpha = alphas[step]
        alpha_prev = alphas_prev[step]
        
        # Predict x0
        pred_x0 = (x_cpu - np.sqrt(1 - alpha) * noise_pred) / np.sqrt(alpha)
        
        # Direction pointing to x_t
        dir_xt = np.sqrt(1 - alpha_prev) * noise_pred
        
        # DDIM step (eta=0, deterministic)
        x_prev = np.sqrt(alpha_prev) * pred_x0 + dir_xt
        
        x = x_prev.to(device)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Step {i + 1}/{len(timesteps)}")
    
    return x


def samples_to_images(samples: torch.Tensor) -> List[Image.Image]:
    """Convert sample tensor to PIL images."""
    # Denormalize from [-1, 1] to [0, 255]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    samples = (samples * 255).to(torch.uint8)
    
    images = []
    for i in range(samples.shape[0]):
        img = samples[i].permute(1, 2, 0).cpu().numpy()
        images.append(Image.fromarray(img))
    
    return images


def save_image_grid(images: List[Image.Image], path: str, 
                    grid_size: Optional[Tuple[int, int]] = None):
    """Save images as a grid."""
    n = len(images)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
    
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))
    
    grid.save(path)
    logger.info(f"Saved image grid to {path}")


def generate_samples(args):
    """Generate samples with all available models."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sampler
    sampler = DDIMSampler(
        num_timesteps=1000,
        ddim_steps=args.ddim_steps,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Sample shape
    shape = (args.num_samples, 3, 32, 32)  # CIFAR10 size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate initial noise (same for all models)
    initial_noise = torch.randn(shape, device=device)
    
    results = {}
    all_samples = {}
    
    # ====================
    # FP32 TensorRT Engine (baseline)
    # ====================
    if args.run_fp32 and Path(args.fp32_engine).exists():
        logger.info("\n" + "="*60)
        logger.info("Generating samples with FP32 TensorRT engine")
        logger.info("="*60)

        try:
            fp32_model = TRTModelRunner(args.fp32_engine, "FP32")

            torch.manual_seed(args.seed)
            start_time = time.time()

            x = initial_noise.clone()
            timesteps, alphas, alphas_prev = sampler.get_schedule()

            for i, step in enumerate(reversed(range(len(timesteps)))):
                t = timesteps[step]
                t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

                noise_pred = fp32_model(x, t_tensor).to(device)

                alpha = alphas[step]
                alpha_prev = alphas_prev[step]

                pred_x0 = (x - np.sqrt(1 - alpha) * noise_pred) / np.sqrt(alpha)
                dir_xt = np.sqrt(1 - alpha_prev) * noise_pred
                x = np.sqrt(alpha_prev) * pred_x0 + dir_xt

                if (i + 1) % 10 == 0:
                    logger.info(f"  Step {i + 1}/{len(timesteps)}")

            fp32_time = time.time() - start_time

            fp32_samples = x
            all_samples['fp32'] = fp32_samples.clone()

            # Save FP32 samples
            images = samples_to_images(fp32_samples)
            for i, img in enumerate(images):
                img.save(output_dir / f"fp32_sample_{i:03d}.png")
            save_image_grid(images, str(output_dir / "fp32_grid.png"))

            results['fp32'] = {
                'time': fp32_time,
                'time_per_sample': fp32_time / args.num_samples,
            }
            logger.info(f"FP32 generation time: {fp32_time:.2f}s ({fp32_time/args.num_samples:.2f}s/sample)")

        except Exception as e:
            logger.warning(f"FP32 TensorRT generation failed: {e}")
    else:
        if args.run_fp32:
            logger.warning(f"FP32 engine not found: {args.fp32_engine}")

    # ====================
    # INT8 TensorRT Engine
    # ====================
    if args.run_int8 and Path(args.int8_engine).exists():
        logger.info("\n" + "="*60)
        logger.info("Generating samples with INT8 TensorRT engine")
        logger.info("="*60)
        
        try:
            int8_model = TRTModelRunner(args.int8_engine, "INT8")
            
            torch.manual_seed(args.seed)
            start_time = time.time()
            
            x = initial_noise.clone()
            timesteps, alphas, alphas_prev = sampler.get_schedule()
            
            for i, step in enumerate(reversed(range(len(timesteps)))):
                t = timesteps[step]
                t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
                
                noise_pred = int8_model(x, t_tensor)
                if not isinstance(noise_pred, torch.Tensor):
                    noise_pred = torch.from_numpy(noise_pred)
                noise_pred = noise_pred.to(device)
                
                alpha = alphas[step]
                alpha_prev = alphas_prev[step]
                
                pred_x0 = (x - np.sqrt(1 - alpha) * noise_pred) / np.sqrt(alpha)
                dir_xt = np.sqrt(1 - alpha_prev) * noise_pred
                x = np.sqrt(alpha_prev) * pred_x0 + dir_xt
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Step {i + 1}/{len(timesteps)}")
            
            int8_time = time.time() - start_time
            
            int8_samples = x
            all_samples['int8'] = int8_samples.clone()
            
            # Save INT8 samples
            images = samples_to_images(int8_samples)
            for i, img in enumerate(images):
                img.save(output_dir / f"int8_sample_{i:03d}.png")
            save_image_grid(images, str(output_dir / "int8_grid.png"))
            
            results['int8'] = {
                'time': int8_time,
                'time_per_sample': int8_time / args.num_samples,
            }
            logger.info(f"INT8 generation time: {int8_time:.2f}s ({int8_time/args.num_samples:.2f}s/sample)")
            
        except Exception as e:
            logger.warning(f"INT8 generation failed: {e}")
    else:
        if args.run_int8:
            logger.warning(f"INT8 engine not found: {args.int8_engine}")
    
    # ====================
    # INT4 TensorRT Engine
    # ====================
    if args.run_int4 and Path(args.int4_engine).exists():
        logger.info("\n" + "="*60)
        logger.info("Generating samples with INT4 TensorRT engine")
        logger.info("="*60)
        
        try:
            int4_model = TRTModelRunner(args.int4_engine, "INT4")
            
            torch.manual_seed(args.seed)
            start_time = time.time()
            
            x = initial_noise.clone()
            timesteps, alphas, alphas_prev = sampler.get_schedule()
            
            for i, step in enumerate(reversed(range(len(timesteps)))):
                t = timesteps[step]
                t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
                
                noise_pred = int4_model(x, t_tensor)
                if not isinstance(noise_pred, torch.Tensor):
                    noise_pred = torch.from_numpy(noise_pred)
                noise_pred = noise_pred.to(device)
                
                alpha = alphas[step]
                alpha_prev = alphas_prev[step]
                
                pred_x0 = (x - np.sqrt(1 - alpha) * noise_pred) / np.sqrt(alpha)
                dir_xt = np.sqrt(1 - alpha_prev) * noise_pred
                x = np.sqrt(alpha_prev) * pred_x0 + dir_xt
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Step {i + 1}/{len(timesteps)}")
            
            int4_time = time.time() - start_time
            
            int4_samples = x
            all_samples['int4'] = int4_samples.clone()
            
            # Save INT4 samples
            images = samples_to_images(int4_samples)
            for i, img in enumerate(images):
                img.save(output_dir / f"int4_sample_{i:03d}.png")
            save_image_grid(images, str(output_dir / "int4_grid.png"))
            
            results['int4'] = {
                'time': int4_time,
                'time_per_sample': int4_time / args.num_samples,
            }
            logger.info(f"INT4 generation time: {int4_time:.2f}s ({int4_time/args.num_samples:.2f}s/sample)")
            
        except Exception as e:
            logger.warning(f"INT4 generation failed: {e}")
    else:
        if args.run_int4:
            logger.warning(f"INT4 engine not found: {args.int4_engine}")
    
    # ====================
    # Create comparison grid
    # ====================
    if len(all_samples) > 1:
        logger.info("\n" + "="*60)
        logger.info("Creating comparison grid")
        logger.info("="*60)
        
        # Keep consistent ordering: fp32 first if present
        model_names = list(all_samples.keys())
        if 'fp32' in model_names:
            model_names = ['fp32'] + [m for m in model_names if m != 'fp32']

        comparison_images = []
        for i in range(args.num_samples):
            row_images = []
            for model_name in model_names:
                samples = all_samples[model_name]
                img = samples_to_images(samples[i:i+1])[0]
                row_images.append(img)
            
            # Combine horizontally
            w, h = row_images[0].size
            combined = Image.new('RGB', (w * len(row_images), h))
            for j, img in enumerate(row_images):
                combined.paste(img, (j * w, 0))
            comparison_images.append(combined)
        
        # Save comparison grid
        if comparison_images:
            w, h = comparison_images[0].size
            cols = min(4, len(comparison_images))
            rows = int(np.ceil(len(comparison_images) / cols))
            grid = Image.new('RGB', (cols * w, rows * h))
            
            for i, img in enumerate(comparison_images):
                row = i // cols
                col = i % cols
                grid.paste(img, (col * w, row * h))
            
            grid.save(output_dir / "comparison_grid.png")
            logger.info(f"Saved comparison grid (columns: {', '.join(model_names).upper()})")
    
    # ====================
    # Quality metrics vs FP32 baseline
    # ====================
    if 'fp32' in all_samples and len(all_samples) > 1:
        logger.info("\n" + "="*60)
        logger.info("Quality Metrics (vs FP32 TensorRT baseline)")
        logger.info("="*60)

        fp32_flat = all_samples['fp32'].flatten().cpu().numpy()

        for model_name in ['int8', 'int4']:
            if model_name in all_samples:
                other_flat = all_samples[model_name].flatten().cpu().numpy()

                mse = np.mean((fp32_flat - other_flat) ** 2)
                psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')  # Range [-1, 1]
                mae = np.mean(np.abs(fp32_flat - other_flat))

                logger.info(f"{model_name.upper()}: MSE={mse:.6f}, PSNR={psnr:.2f}dB, MAE={mae:.6f}")

    # ====================
    # Summary
    # ====================
    logger.info("\n" + "="*60)
    logger.info("Generation Summary")
    logger.info("="*60)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"DDIM steps: {args.ddim_steps}")
    print(f"Random seed: {args.seed}")
    print()
    
    if results:
        print("Generation times:")
        baseline_time = results.get('fp32', {}).get('time', None)
        for model_name, stats in results.items():
            speedup = baseline_time / stats['time'] if baseline_time and model_name != 'fp32' else None
            speedup_str = f", {speedup:.1f}x speedup" if speedup else ""
            print(f"  {model_name.upper():6s}: {stats['time']:.2f}s total, "
                  f"{stats['time_per_sample']:.2f}s/sample{speedup_str}")
    
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")
    
    # Save results to JSON
    import json
    results_path = output_dir / "generation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'num_samples': args.num_samples,
                'ddim_steps': args.ddim_steps,
                'seed': args.seed,
            },
            'results': results,
        }, f, indent=2)
    logger.info(f"Saved results to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate samples with TensorRT FP32/INT8/INT4 engines")
    parser.add_argument("--fp32-engine", default="export/modiff_unet_fp32.plan", help="FP32 TensorRT engine path")
    parser.add_argument("--int8-engine", default="modiff_unet_int8.plan", help="INT8 engine path")
    parser.add_argument("--int4-engine", default="int4_output/modiff_unet_int4.plan", help="INT4 engine path")
    parser.add_argument("--output-dir", default="generated_samples", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-fp32", action="store_true", default=True, help="Run FP32 engine")
    parser.add_argument("--run-int8", action="store_true", default=True, help="Run INT8 engine")
    parser.add_argument("--run-int4", action="store_true", default=True, help="Run INT4 engine")
    parser.add_argument("--skip-fp32", action="store_true", help="Skip FP32 engine")
    parser.add_argument("--skip-int8", action="store_true", help="Skip INT8 engine")
    parser.add_argument("--skip-int4", action="store_true", help="Skip INT4 engine")
    args = parser.parse_args()
    
    # Handle skip flags
    if args.skip_fp32:
        args.run_fp32 = False
    if args.skip_int8:
        args.run_int8 = False
    if args.skip_int4:
        args.run_int4 = False
    
    print("="*60)
    print("MoDiff Sample Generation (TensorRT)")
    print("FP32 vs INT8 vs INT4 Comparison")
    print("="*60)
    
    generate_samples(args)


if __name__ == "__main__":
    main()
