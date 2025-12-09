#!/usr/bin/env python3
"""
FID Comparison: FP32 vs INT8 vs INT4 TensorRT Engines

Generates samples from all engines and computes FID against CIFAR-10 statistics.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress TensorRT verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("ERROR: TensorRT or PyCUDA not found")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TRTEngine:
    """Simple TensorRT engine wrapper for inference."""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate buffers
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            # Handle dynamic shapes
            if -1 in shape:
                shape = tuple(1 if s == -1 else s for s in shape)
            
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = {'host': host_mem, 'device': device_mem, 'shape': shape, 'dtype': dtype}
            else:
                self.outputs[name] = {'host': host_mem, 'device': device_mem, 'shape': shape, 'dtype': dtype}
    
    def infer(self, latent: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
        # Set input shapes explicitly for dynamic dimensions
        latent_shape = latent.shape
        timesteps_shape = timesteps.shape
        
        self.context.set_input_shape('latent', latent_shape)
        self.context.set_input_shape('timesteps', timesteps_shape)
        
        # Reallocate buffers if shapes changed
        latent_size = int(np.prod(latent_shape))
        timesteps_size = int(np.prod(timesteps_shape))
        
        # Update latent buffer if needed
        if self.inputs['latent']['host'].size != latent_size:
            dtype = self.inputs['latent']['dtype']
            self.inputs['latent']['host'] = cuda.pagelocked_empty(latent_size, dtype)
            self.inputs['latent']['device'] = cuda.mem_alloc(self.inputs['latent']['host'].nbytes)
        
        # Update timesteps buffer if needed  
        if self.inputs['timesteps']['host'].size != timesteps_size:
            dtype = self.inputs['timesteps']['dtype']
            self.inputs['timesteps']['host'] = cuda.pagelocked_empty(timesteps_size, dtype)
            self.inputs['timesteps']['device'] = cuda.mem_alloc(self.inputs['timesteps']['host'].nbytes)
        
        # Get output shape after setting input shapes
        out_name = list(self.outputs.keys())[0]
        out_shape = self.context.get_tensor_shape(out_name)
        out_size = int(np.prod(out_shape))
        
        # Update output buffer if needed
        if self.outputs[out_name]['host'].size != out_size:
            dtype = self.outputs[out_name]['dtype']
            self.outputs[out_name]['host'] = cuda.pagelocked_empty(out_size, dtype)
            self.outputs[out_name]['device'] = cuda.mem_alloc(self.outputs[out_name]['host'].nbytes)
        self.outputs[out_name]['shape'] = out_shape
        
        # Copy inputs
        np.copyto(self.inputs['latent']['host'], latent.ravel())
        np.copyto(self.inputs['timesteps']['host'], timesteps.ravel())
        
        # Transfer to device
        for name, buf in self.inputs.items():
            cuda.memcpy_htod_async(buf['device'], buf['host'], self.stream)
        
        # Set tensor addresses
        for name, buf in self.inputs.items():
            self.context.set_tensor_address(name, int(buf['device']))
        for name, buf in self.outputs.items():
            self.context.set_tensor_address(name, int(buf['device']))
        
        # Execute
        self.context.execute_async_v3(self.stream.handle)
        
        # Transfer back
        for name, buf in self.outputs.items():
            cuda.memcpy_dtoh_async(buf['host'], buf['device'], self.stream)
        
        self.stream.synchronize()
        
        # Get output
        return self.outputs[out_name]['host'].reshape(out_shape)


def get_beta_schedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """Get linear beta schedule."""
    return np.linspace(beta_start, beta_end, num_timesteps)


def ddim_sample(engine, num_samples, num_steps=50, eta=0.0, seed=None):
    """
    DDIM sampling using TensorRT engine.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # CIFAR-10: 3x32x32
    shape = (1, 3, 32, 32)
    
    # Beta schedule
    betas = get_beta_schedule()
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    # DDIM timesteps (uniform spacing)
    timesteps = np.linspace(0, 999, num_steps, dtype=np.int64)[::-1]
    
    samples = []
    
    for i in tqdm(range(num_samples), desc="Generating samples", leave=False):
        # Start from noise
        x = np.random.randn(*shape).astype(np.float32)
        
        for idx, t in enumerate(timesteps):
            t_batch = np.array([t], dtype=np.int64)
            
            # Predict noise
            noise_pred = engine.infer(x, t_batch)
            
            # DDIM update
            alpha_t = alphas_cumprod[t]
            
            if idx < len(timesteps) - 1:
                t_prev = timesteps[idx + 1]
                alpha_t_prev = alphas_cumprod[t_prev]
            else:
                alpha_t_prev = 1.0
            
            # Predicted x0
            pred_x0 = (x - np.sqrt(1 - alpha_t) * noise_pred) / np.sqrt(alpha_t)
            pred_x0 = np.clip(pred_x0, -1, 1)
            
            # Direction pointing to x_t
            dir_xt = np.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * noise_pred
            
            # Random noise
            if eta > 0 and idx < len(timesteps) - 1:
                noise = np.random.randn(*shape).astype(np.float32)
                sigma = eta * np.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            else:
                noise = 0
                sigma = 0
            
            x = np.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
        
        # Convert to image [0, 255]
        img = ((x[0].transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        samples.append(img)
    
    return samples


def save_samples(samples, output_dir: Path, prefix: str):
    """Save samples as PNG images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(samples):
        img_path = output_dir / f"{prefix}_{i:04d}.png"
        Image.fromarray(img).save(img_path)
    
    logger.info(f"Saved {len(samples)} samples to {output_dir}")


def compute_fid(real_dir: str, fake_dir: str, device='cuda'):
    """Compute FID with numerical stabilization."""
    from pytorch_fid.inception import InceptionV3
    from scipy import linalg
    from torchvision import transforms
    
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    def get_activations(img_dir):
        files = sorted(Path(img_dir).glob('*.png'))
        acts = []
        batch_size = 50
        
        for i in tqdm(range(0, len(files), batch_size), desc="Computing activations", leave=False):
            batch_files = files[i:i+batch_size]
            batch = []
            for f in batch_files:
                img = Image.open(f).convert('RGB')
                img = transform(img)
                batch.append(img)
            
            batch = torch.stack(batch).to(device)
            
            with torch.no_grad():
                pred = model(batch)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                acts.append(pred)
        
        return np.concatenate(acts, axis=0)
    
    def calculate_statistics(acts):
        mu = np.mean(acts, axis=0)
        sigma = np.cov(acts, rowvar=False)
        return mu, sigma
    
    def calculate_fid_stable(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate FID with numerical stabilization."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Add small epsilon to diagonal for numerical stability
        sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
        sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if not np.isfinite(covmean).all():
            logger.warning("FID calculation produced non-finite values, using fallback")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Remove imaginary component (numerical artifact)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                logger.warning(f"Imaginary component in FID: {np.max(np.abs(covmean.imag))}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return fid
    
    # Get activations for both directories
    acts1 = get_activations(real_dir)
    acts2 = get_activations(fake_dir)
    
    mu1, sigma1 = calculate_statistics(acts1)
    mu2, sigma2 = calculate_statistics(acts2)
    
    return calculate_fid_stable(mu1, sigma1, mu2, sigma2)


def compute_fid_from_stats(fake_dir: str, stats_path: str, device='cuda'):
    """Compute FID using pre-computed statistics."""
    from pytorch_fid import fid_score
    from pytorch_fid.fid_score import calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3
    import torch
    
    # Load pre-computed stats
    stats = np.load(stats_path)
    mu_real, sigma_real = stats['mu'], stats['sigma']
    
    # Compute stats for generated images
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    
    # Get activations
    from pytorch_fid.fid_score import get_activations
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    
    # Create dataset
    files = sorted(Path(fake_dir).glob('*.png'))
    
    acts = []
    batch_size = 50
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch = []
        for f in batch_files:
            img = Image.open(f).convert('RGB')
            img = transform(img)
            batch.append(img)
        
        batch = torch.stack(batch).to(device)
        
        with torch.no_grad():
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            acts.append(pred)
    
    acts = np.concatenate(acts, axis=0)
    mu_fake = np.mean(acts, axis=0)
    sigma_fake = np.cov(acts, rowvar=False)
    
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid


def main():
    parser = argparse.ArgumentParser(description="FID comparison between FP32, INT8, and INT4 vs CIFAR-10")
    parser.add_argument("--fp32-engine", default="export/modiff_unet_fp32.plan", help="FP32 engine path")
    parser.add_argument("--int8-engine", default="int8_output/modiff_unet_int8.plan", help="INT8 engine path")
    parser.add_argument("--int4-engine", default="int4_output/modiff_unet_int4.plan", help="INT4 engine path")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples to generate (default: 50k)")
    parser.add_argument("--num-steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--output-dir", default="fid_comparison", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cifar-stats", default=None, help="Pre-computed CIFAR-10 statistics (optional)")
    parser.add_argument("--cifar-dir", default="../data/cifar-10-batches-py", help="CIFAR-10 data directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fp32_samples_dir = output_dir / "fp32_samples"
    int8_samples_dir = output_dir / "int8_samples"
    int4_samples_dir = output_dir / "int4_samples"
    cifar_real_dir = output_dir / "cifar10_real"
    
    # Load engines
    logger.info("="*60)
    logger.info("FID Comparison: FP32 vs INT8 vs INT4 (against CIFAR-10)")
    logger.info("="*60)
    
    logger.info(f"\nGenerating {args.num_samples} samples with {args.num_steps} DDIM steps")
    logger.info(f"This will take a while for 50k samples...")
    
    results = {
        'num_samples': args.num_samples,
        'num_steps': args.num_steps,
        'seed': args.seed,
    }
    
    # First, extract CIFAR-10 images (use all 50k training + 10k test = 60k available)
    logger.info("\n[Step 1] Preparing CIFAR-10 reference images...")
    try:
        import pickle
        cifar_real_dir.mkdir(parents=True, exist_ok=True)
        
        cifar_path = Path(args.cifar_dir)
        all_images = []
        
        # Load all training batches
        for batch_num in range(1, 6):
            batch_file = cifar_path / f"data_batch_{batch_num}"
            if batch_file.exists():
                with open(batch_file, 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                images = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                all_images.append(images)
                logger.info(f"  Loaded data_batch_{batch_num}: {len(images)} images")
        
        # Load test batch
        test_batch = cifar_path / "test_batch"
        if test_batch.exists():
            with open(test_batch, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            images = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            all_images.append(images)
            logger.info(f"  Loaded test_batch: {len(images)} images")
        
        if not all_images:
            raise FileNotFoundError(f"No CIFAR-10 data found in {cifar_path}")
        
        all_images = np.concatenate(all_images, axis=0)
        logger.info(f"  Total CIFAR-10 images available: {len(all_images)}")
        
        # Save images for FID computation (use all available or num_samples, whichever is smaller)
        n_real = min(args.num_samples, len(all_images))
        existing_count = len(list(cifar_real_dir.glob('*.png')))
        
        if existing_count >= n_real:
            logger.info(f"  CIFAR-10 images already extracted ({existing_count} images)")
        else:
            logger.info(f"  Saving {n_real} CIFAR-10 images...")
            for i in tqdm(range(n_real), desc="Saving CIFAR-10"):
                img = Image.fromarray(all_images[i])
                img.save(cifar_real_dir / f"cifar_{i:05d}.png")
            logger.info(f"  Saved {n_real} CIFAR-10 images to {cifar_real_dir}")
        
    except Exception as e:
        logger.error(f"Failed to load CIFAR-10: {e}")
        raise
    
    # Generate FP32 samples
    logger.info("\n[Step 2] Generating FP32 samples...")
    if not Path(args.fp32_engine).exists():
        logger.error(f"FP32 engine not found: {args.fp32_engine}")
        return
    
    fp32_engine = TRTEngine(args.fp32_engine)
    start = time.time()
    fp32_samples = ddim_sample(fp32_engine, args.num_samples, args.num_steps, seed=args.seed)
    fp32_time = time.time() - start
    save_samples(fp32_samples, fp32_samples_dir, "fp32")
    logger.info(f"  FP32 generation: {fp32_time:.1f}s ({args.num_samples/fp32_time:.2f} samples/s)")
    results['fp32_time'] = fp32_time
    del fp32_engine  # Free GPU memory
    
    # Generate INT8 samples (if engine exists)
    int8_time = None
    if Path(args.int8_engine).exists():
        logger.info("\n[Step 3] Generating INT8 samples...")
        int8_engine = TRTEngine(args.int8_engine)
        start = time.time()
        int8_samples = ddim_sample(int8_engine, args.num_samples, args.num_steps, seed=args.seed)
        int8_time = time.time() - start
        save_samples(int8_samples, int8_samples_dir, "int8")
        logger.info(f"  INT8 generation: {int8_time:.1f}s ({args.num_samples/int8_time:.2f} samples/s)")
        results['int8_time'] = int8_time
        del int8_engine
    else:
        logger.warning(f"INT8 engine not found: {args.int8_engine}, skipping...")
    
    # Generate INT4 samples
    int4_time = None
    if Path(args.int4_engine).exists():
        logger.info("\n[Step 4] Generating INT4 samples...")
        int4_engine = TRTEngine(args.int4_engine)
        start = time.time()
        int4_samples = ddim_sample(int4_engine, args.num_samples, args.num_steps, seed=args.seed)
        int4_time = time.time() - start
        save_samples(int4_samples, int4_samples_dir, "int4")
        logger.info(f"  INT4 generation: {int4_time:.1f}s ({args.num_samples/int4_time:.2f} samples/s)")
        results['int4_time'] = int4_time
        del int4_engine
    else:
        logger.warning(f"INT4 engine not found: {args.int4_engine}, skipping...")
    
    # Compute FID scores against CIFAR-10 only
    logger.info("\n" + "="*60)
    logger.info("Computing FID scores against CIFAR-10...")
    logger.info("="*60)
    
    # FID(FP32, CIFAR-10)
    logger.info("\n[Step 5] Computing FID(FP32, CIFAR-10)...")
    fid_fp32_cifar = compute_fid(str(cifar_real_dir), str(fp32_samples_dir))
    logger.info(f"  FID(FP32, CIFAR-10) = {fid_fp32_cifar:.2f}")
    results['fid_fp32_cifar'] = fid_fp32_cifar
    
    # FID(INT8, CIFAR-10)
    fid_int8_cifar = None
    if int8_time is not None:
        logger.info("\n[Step 6] Computing FID(INT8, CIFAR-10)...")
        fid_int8_cifar = compute_fid(str(cifar_real_dir), str(int8_samples_dir))
        logger.info(f"  FID(INT8, CIFAR-10) = {fid_int8_cifar:.2f}")
        results['fid_int8_cifar'] = fid_int8_cifar
    
    # FID(INT4, CIFAR-10)
    fid_int4_cifar = None
    if int4_time is not None:
        logger.info("\n[Step 7] Computing FID(INT4, CIFAR-10)...")
        fid_int4_cifar = compute_fid(str(cifar_real_dir), str(int4_samples_dir))
        logger.info(f"  FID(INT4, CIFAR-10) = {fid_int4_cifar:.2f}")
        results['fid_int4_cifar'] = fid_int4_cifar
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS - FID vs CIFAR-10 (lower = better)")
    logger.info("="*60)
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"DDIM steps: {args.num_steps}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("")
    
    logger.info("Generation Performance:")
    logger.info(f"  FP32: {fp32_time:.1f}s ({args.num_samples/fp32_time:.2f} samples/s)")
    if int8_time is not None:
        speedup_int8 = fp32_time / int8_time
        logger.info(f"  INT8: {int8_time:.1f}s ({args.num_samples/int8_time:.2f} samples/s) - {speedup_int8:.2f}x speedup")
        results['speedup_int8'] = speedup_int8
    if int4_time is not None:
        speedup_int4 = fp32_time / int4_time
        logger.info(f"  INT4: {int4_time:.1f}s ({args.num_samples/int4_time:.2f} samples/s) - {speedup_int4:.2f}x speedup")
        results['speedup_int4'] = speedup_int4
    
    logger.info("")
    logger.info("FID Scores (vs CIFAR-10, lower = better):")
    logger.info(f"  FP32: {fid_fp32_cifar:.2f}")
    if fid_int8_cifar is not None:
        degradation_int8 = fid_int8_cifar - fid_fp32_cifar
        logger.info(f"  INT8: {fid_int8_cifar:.2f} (degradation: {degradation_int8:+.2f})")
    if fid_int4_cifar is not None:
        degradation_int4 = fid_int4_cifar - fid_fp32_cifar
        logger.info(f"  INT4: {fid_int4_cifar:.2f} (degradation: {degradation_int4:+.2f})")
    
    # Save results
    with open(output_dir / "fid_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir / 'fid_results.json'}")


if __name__ == "__main__":
    main()
