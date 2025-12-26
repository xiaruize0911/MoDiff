#!/usr/bin/env python
"""
MoDiff CUDA FID Benchmark - Comparing MoDiff CUDA quantized models against CIFAR-10 dataset.

This script:
1. Loads the pretrained CIFAR-10 diffusion model
2. Applies MoDiff CUDA quantization (W8A8) with proper calibration
3. Generates images and computes FID against real CIFAR-10 dataset
4. Optionally compares with FP32 baseline

Usage:
    python benchmark_fid_cuda.py --num_samples 10000
"""

import os
import sys
import argparse
import json
from pathlib import Path
import weakref
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision
import torchvision.transforms as transforms

# Add paths
# /workspace/MoDiff/modiff_cuda/ -> /workspace/MoDiff/
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddim.models.diffusion import Model

# Import MoDiff CUDA
try:
    from modiff_cuda.nn.conv import W8A8MoDiffConv2dCUDA
    import modiff_cuda_backend
except ImportError:
    print("Error: modiff_cuda not installed. Please run 'pip install .' in modiff_cuda directory.")
    sys.exit(1)

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """Get beta schedule for diffusion."""
    if beta_schedule == "quad":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps)
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def compute_alpha(betas, t):
    """Compute alpha values for diffusion."""
    betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    a = (1 - betas).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


class MoDiffCudaLayer(nn.Module):
    """
    MoDiff quantized layer using CUDA kernels.
    Wraps a Conv2d layer with W8A8MoDiffConv2dCUDA.
    """
    def __init__(self, original_layer, layer_idx=0, parent_model=None):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        
        # Use weakref to avoid circular reference
        object.__setattr__(self, '_parent_model_ref', weakref.ref(parent_model) if parent_model else None)
        
        # Initialize CUDA layer
        if isinstance(original_layer, nn.Conv2d):
            self.cuda_layer = W8A8MoDiffConv2dCUDA(
                in_channels=original_layer.in_channels,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                dilation=original_layer.dilation,
                groups=original_layer.groups,
                bias=False # We handle bias manually
            )
            self.bias = original_layer.bias
        else:
            raise ValueError("MoDiffCudaLayer only supports Conv2d")
        
        # Activation quantization parameters
        self.act_scale = None
        
        # Running statistics for activation calibration
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
        self.calibrating = False
        self.is_calibrated = False
        
        # MoDiff Cache
        self.prev_quantized = None
        self.prev_output = None
        
        # Buffers for CUDA Graphs
        self.res_int8_buffer = None
        self.max_val_buffer = None
        self.res_scale_buffer = None
        
    def calibrate_weights(self):
        """Calibrate weight quantization parameters using min-max."""
        if hasattr(self.original_layer, 'weight'):
            weight = self.original_layer.weight.data # [out, in, k, k]
            
            # Calculate scales per output channel
            # weight is [out_c, in_c, k, k]
            # We want scale per out_c
            w_flat = weight.view(weight.size(0), -1)
            w_max = w_flat.abs().max(dim=1)[0]
            self.weight_scales = w_max / 127.0
            
            # Per-Tensor quantization (match Triton baseline)
            # w_absmax = weight.abs().max()
            # self.weight_scales = (w_absmax / 127.0).repeat(weight.size(0))
            
            # Quantize
            # Broadcast scale: [out_c, 1, 1, 1]
            scale_view = self.weight_scales.view(-1, 1, 1, 1)
            w_int8 = (weight / scale_view).round().clamp(-128, 127).to(torch.int8)
            
            # Set to CUDA layer (flattened as expected by Implicit GEMM kernel)
            self.cuda_layer.weight.data = w_int8.permute(0, 2, 3, 1).contiguous().view(weight.size(0), -1)
            self.cuda_layer.weight_scales.data = self.weight_scales
            
    def update_activation_stats(self, x):
        """Update running statistics for activation calibration."""
        if self.calibrating:
            self.running_min = torch.min(self.running_min, x.min())
            self.running_max = torch.max(self.running_max, x.max())
    
    def calibrate_activations(self):
        """Finalize activation quantization parameters."""
        if self.running_min < float('inf'):
            # Symmetric quantization
            a_absmax = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
            self.act_scale = a_absmax / 127.0
            self.is_calibrated = True
    
    def forward(self, x, use_modiff=True):
        """Forward pass with MoDiff quantization."""
        # Check parent model's flag if available
        parent_model_ref = object.__getattribute__(self, '_parent_model_ref')
        if parent_model_ref is not None:
            parent = parent_model_ref()
            if parent is not None and hasattr(parent, 'use_modiff_flag'):
                use_modiff = parent.use_modiff_flag
        
        # Update activation stats if calibrating
        self.update_activation_stats(x)
        
        if not self.is_calibrated:
            # Not calibrated, run in FP32
            return self.original_layer(x)
        
        # MoDiff Logic
        if self.prev_quantized is not None and use_modiff:
            # Modulated path
            
            # Allocate buffers if needed
            # Note: res_int8_buffer should be NHWC for the new kernel
            if self.res_int8_buffer is None:
                # x is NCHW. We want res_int8_buffer to be NHWC.
                N, C, H, W = x.shape
                self.res_int8_buffer = torch.empty((N, H, W, C), dtype=torch.int8, device=x.device)
                self.max_val_buffer = torch.zeros(1, dtype=torch.float32, device=x.device)
                self.res_scale_buffer = torch.zeros(1, dtype=torch.float32, device=x.device)
            
            # Fused Kernel: Quantize residual and update cache
            # x is NCHW. prev_quantized is NHWC. res_int8_buffer is NHWC.
            modiff_cuda_backend.modiff_quantize_update_permute(
                x, self.prev_quantized, self.res_int8_buffer, self.max_val_buffer
            )
            
            # Compute scale in-place for CUDA Graphs safety
            torch.div(self.max_val_buffer, 127.0, out=self.res_scale_buffer)
            torch.clamp(self.res_scale_buffer, min=1e-8, out=self.res_scale_buffer)
            
            # Run convolution on residual
            # Pass int8 residual (NHWC) to avoid re-quantization and permute
            # Pass prev_output (NHWC) for accumulation
            # We need to tell cuda_layer that input is NHWC
            self.res_int8_buffer._layout = 'NHWC'
            
            out_conv = self.cuda_layer(self.res_int8_buffer, prev_output=self.prev_output, input_scale=self.res_scale_buffer, output_layout='NHWC')
            
            # Update output cache - In-place for CUDA Graphs
            if self.prev_output is None:
                self.prev_output = out_conv.detach().clone()
                self.prev_output._layout = 'NHWC'
            else:
                self.prev_output.copy_(out_conv)
            
            # Convert output back to NCHW for next layer
            # out_conv is NHWC.
            out_nchw, _ = modiff_cuda_backend.permute_half_nhwc_nchw(out_conv, False)
            out = out_nchw
            
        else:
            # Standard path (First step or use_modiff=False)
            # Quantize x (NCHW -> NHWC)
            # We can use the standard quantize_permute
            
            # Update cache
            # We want prev_quantized to be NHWC
            # But we need x_fp (dequantized) to update it?
            # Wait, prev_quantized stores the *reconstructed* activation.
            # x_int8 is NHWC.
            
            # Run convolution
            # cuda_layer handles quantization internally if we pass FP input.
            # But we want to capture the quantized input for cache.
            
            # Let's do it manually to control layout
            if not hasattr(x, 'next_scale'):
                act_scale = modiff_cuda_backend.find_max_abs(x) / 127.0
            else:
                act_scale = x.next_scale
                
            # Quantize + Permute NCHW -> NHWC
            x_int8 = modiff_cuda_backend.quantize_permute(x, act_scale)
            x_int8._layout = 'NHWC'
            
            # Update cache (NHWC)
            # Dequantize x_int8 to FP16/FP32 and store in prev_quantized
            # We don't have a dequantize kernel exposed?
            # We can just store x_int8 and scale? No, MoDiff accumulates residuals in FP.
            # So we need to dequantize.
            x_fp_nhwc = x_int8.float() * act_scale
            
            if self.prev_quantized is None:
                self.prev_quantized = x_fp_nhwc.detach().clone()
            else:
                # Resize if needed (shouldn't happen in fixed graph)
                if self.prev_quantized.shape != x_fp_nhwc.shape:
                     self.prev_quantized = x_fp_nhwc.detach().clone()
                else:
                     self.prev_quantized.copy_(x_fp_nhwc)
            
            # Run convolution
            out_conv = self.cuda_layer(x_int8, prev_output=None, input_scale=act_scale, output_layout='NHWC')
            
            # Update output cache (NHWC)
            if self.prev_output is None:
                self.prev_output = out_conv.detach().clone()
                self.prev_output._layout = 'NHWC'
            else:
                if self.prev_output.shape != out_conv.shape:
                    self.prev_output = out_conv.detach().clone()
                    self.prev_output._layout = 'NHWC'
                else:
                    self.prev_output.copy_(out_conv)
            
            # Convert output back to NCHW
            out_nchw, out_max = modiff_cuda_backend.permute_half_nhwc_nchw(out_conv, True)
            out_nchw.next_scale = out_max / 127.0
            out = out_nchw
            
        # Add bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        else:
            out = out_conv
        
        # Ensure output matches input dtype (likely FP32)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
            
        return out
    
    def reset_cache(self):
        self.prev_quantized = None
        self.prev_output = None
        self.res_int8_buffer = None
        self.max_val_buffer = None
        self.res_scale_buffer = None


class MoDiffModel(nn.Module):
    """
    Wrapper that applies MoDiff quantization to a diffusion model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.modiff_layers = []
        self.use_modiff_flag = True
        
        # Wrap quantizable layers
        self._wrap_layers(model)
    
    def _wrap_layers(self, module, prefix=''):
        """Recursively wrap layers with MoDiff quantization."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Skip first and last layers
            if 'conv_in' in full_name or 'conv_out' in full_name:
                continue
            
            if isinstance(child, nn.Conv2d):
                # Create MoDiff layer
                modiff_layer = MoDiffCudaLayer(
                    child, 
                    layer_idx=len(self.modiff_layers),
                    parent_model=self
                )
                setattr(module, name, modiff_layer)
                self.modiff_layers.append(modiff_layer)
            elif isinstance(child, (nn.Linear, nn.Conv2d)):
                # Skip Linear layers for now (or keep as FP32)
                pass
            else:
                self._wrap_layers(child, full_name)
    
    def calibrate(self, calib_data, device):
        """
        Calibrate quantization parameters using calibration data.
        """
        print(f"Calibrating MoDiff model with {len(self.modiff_layers)} layers...")
        
        # Step 1: Calibrate weights
        for layer in self.modiff_layers:
            layer.calibrate_weights()
        print("  Weight calibration done.")
        
        # Step 2: Calibrate activations using calibration data
        xs, ts = calib_data
        
        # Enable calibration mode
        for layer in self.modiff_layers:
            layer.calibrating = True
        
        # Run forward passes to collect activation statistics
        self.model.eval()
        n_calib = min(256, xs.shape[0])
        batch_size = 32
        
        print(f"  Running {n_calib} calibration samples...")
        with torch.no_grad():
            for i in tqdm(range(0, n_calib, batch_size), desc="  Calibrating"):
                batch_xs = xs[i:i+batch_size].to(device)
                batch_ts = ts[i:i+batch_size].to(device)
                _ = self.model(batch_xs, batch_ts)
        
        # Finalize activation calibration
        for layer in self.modiff_layers:
            layer.calibrating = False
            layer.calibrate_activations()
        
        print("  Activation calibration done.")
    
    def forward(self, x, t, use_modiff=True):
        """Forward pass through the quantized model."""
        self.use_modiff_flag = use_modiff
        return self.model(x, t)
    
    def reset_cache(self):
        """Reset MoDiff cache for all layers."""
        for layer in self.modiff_layers:
            layer.reset_cache()


def load_cifar10_model(device):
    """Load pretrained CIFAR-10 diffusion model."""
    # Model configuration matching cifar10.yml
    class InnerModelConfig:
        def __init__(self):
            self.type = "simple"
            self.in_channels = 3
            self.out_ch = 3
            self.ch = 128
            self.ch_mult = [1, 2, 2, 2]
            self.num_res_blocks = 2
            self.attn_resolutions = [16]
            self.dropout = 0.1
            self.var_type = "fixedlarge"
            self.ema_rate = 0.9999
            self.ema = True
            self.resamp_with_conv = True
            self.split_shortcut = False
    
    class DataConfig:
        def __init__(self):
            self.image_size = 32
    
    class DiffusionConfig:
        def __init__(self):
            self.num_diffusion_timesteps = 1000
    
    class ModelConfig:
        def __init__(self):
            self.model = InnerModelConfig()
            self.data = DataConfig()
            self.diffusion = DiffusionConfig()
            self.split_shortcut = False
    
    config = ModelConfig()
    model = Model(config)
    
    # Download and load pretrained weights
    ckpt_path = os.path.expanduser("~/.cache/diffusion_models/cifar10_ema.pth")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        print("Downloading pretrained CIFAR-10 model...")
        url = "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1"
        torch.hub.download_url_to_file(url, ckpt_path)
    
    states = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(states, strict=True)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded CIFAR-10 model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    return model


def load_calibration_data(device):
    """Load calibration data for quantization."""
    cali_path = Path(__file__).parent.parent / "cali_data" / "cifar10.pt"
    
    if not cali_path.exists():
        print("Calibration data not found. Generating synthetic calibration data...")
        xs = torch.randn(256, 3, 32, 32)
        ts = torch.randint(0, 1000, (256,)).float()
        return xs, ts
    
    print(f"Loading calibration data from {cali_path}")
    cali_data = torch.load(cali_path)
    
    if isinstance(cali_data, dict):
        xs = cali_data['xs']
        ts = cali_data['ts']
        xs = xs.reshape(-1, 3, 32, 32)
        ts = ts.reshape(-1)
    else:
        xs, ts = cali_data
    
    return xs, ts


def generate_samples(model, n_samples, device, timesteps=100, batch_size=64, use_modiff=True, use_cuda_graphs=True):
    """
    Generate samples using DDIM sampling with optional CUDA graphs.
    """
    # Setup diffusion parameters
    betas = get_beta_schedule("linear", 0.0001, 0.02, 1000)
    betas = torch.from_numpy(betas).float().to(device)
    
    skip = 1000 // timesteps
    seq = range(0, 1000, skip)
    
    all_samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    is_modiff = isinstance(model, MoDiffModel)
    
    # CUDA Graph variables
    cuda_graphs = {}
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    for batch_idx in tqdm(range(n_batches), desc="Generating"):
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        x = torch.randn(current_batch_size, 3, 32, 32, device=device)
        
        if is_modiff:
            model.reset_cache()
        
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            
            graph_key = current_batch_size
            if use_cuda_graphs and torch.cuda.is_available() and is_modiff and use_modiff:
                if graph_key not in cuda_graphs:
                    cuda_graphs[graph_key] = {
                        'graph': None,
                        'static_x': None,
                        'static_t': None,
                        'static_et': None,
                        'static_at': None,
                        'static_at_next': None,
                    }
                graph_data = cuda_graphs[graph_key]
            else:
                graph_data = None
            
            for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(n, device=device) * i).long()
                next_t = (torch.ones(n, device=device) * j).long()
                
                at = compute_alpha(betas, t)
                at_next = compute_alpha(betas, next_t) if j >= 0 else torch.ones_like(at)
                
                use_graph = (graph_data is not None and step_idx > 0)
                
                if use_graph:
                    if graph_data['graph'] is None:
                        # Capture graph
                        graph_data['static_x'] = x.clone()
                        graph_data['static_t'] = t.float().clone()
                        graph_data['static_at'] = at.clone()
                        graph_data['static_at_next'] = at_next.clone()
                        
                        # Warmup
                        for _ in range(3):
                            _ = model(graph_data['static_x'], graph_data['static_t'], use_modiff=True)
                        
                        # Capture
                        graph_data['graph'] = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph_data['graph']):
                            graph_data['static_et'] = model(graph_data['static_x'], graph_data['static_t'], use_modiff=True)
                    
                    # Replay graph
                    graph_data['static_x'].copy_(x)
                    graph_data['static_t'].copy_(t.float())
                    graph_data['static_at'].copy_(at)
                    graph_data['static_at_next'].copy_(at_next)
                    graph_data['graph'].replay()
                    et = graph_data['static_et'].clone()
                    
                    at = graph_data['static_at']
                    at_next = graph_data['static_at_next']
                else:
                    if is_modiff:
                        current_use_modiff = use_modiff and (step_idx > 0)
                        et = model(x, t.float(), use_modiff=current_use_modiff)
                    else:
                        et = model(x, t.float())
                
                if torch.isnan(et).any() or torch.isinf(et).any():
                    et = torch.nan_to_num(et, nan=0.0, posinf=1.0, neginf=-1.0)
                
                x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
                x0_t = torch.clamp(x0_t, -1, 1)
                
                c1 = 0
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                x = at_next.sqrt() * x0_t + c2 * et
                
                if torch.isnan(x).any() or torch.isinf(x).any():
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                
                x = torch.clamp(x, -10, 10)
        
        samples = (x + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        if torch.isnan(samples).any() or torch.isinf(samples).any():
            samples = torch.nan_to_num(samples, nan=0.5, posinf=1.0, neginf=0.0)
        
        all_samples.append(samples.cpu())
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    full_samples = torch.cat(all_samples, dim=0)[:n_samples]
    return full_samples, elapsed_time


def compute_fid(generated_samples, real_samples, device):
    """Compute FID between generated and real samples."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception.fc = nn.Identity()
    inception = inception.to(device)
    inception.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    def get_features(samples, batch_size=64, desc="  Extracting features"):
        features = []
        n_batches = (len(samples) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc=desc):
                batch = samples[i*batch_size:(i+1)*batch_size].to(device)
                batch = torch.clamp(batch, 0.0, 1.0)
                batch = F.interpolate(batch, size=(299, 299), mode='bicubic', align_corners=False)
                batch = torch.clamp(batch, 0.0, 1.0)
                batch = (batch - mean) / std
                feat = inception(batch)
                if feat.dim() == 4:
                    feat = feat.squeeze(-1).squeeze(-1)
                features.append(feat.cpu())
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        
        return torch.cat(features, dim=0).numpy()
    
    print("  Computing features for generated samples...")
    gen_features = get_features(generated_samples, desc="  Generated features")
    
    print("  Computing features for real samples...")
    real_features = get_features(real_samples, desc="  Real features")
    
    print("  Computing FID statistics...")
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    print("  Computing FID score...")
    mu1, sigma1 = mu_gen, sigma_gen
    mu2, sigma2 = mu_real, sigma_real
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    from scipy import linalg
    eps = 1e-6
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fid)


def load_cifar10_real_samples(n_samples, device):
    """Load real CIFAR-10 samples for FID computation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    data_root = Path(__file__).parent.parent / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=True, download=True, transform=transform
        )
    except RuntimeError:
        import shutil
        cifar_dir = data_root / "cifar-10-batches-py"
        if cifar_dir.exists():
            shutil.rmtree(cifar_dir)
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=True, download=True, transform=transform
        )
    
    total_available = len(dataset)
    
    if n_samples >= total_available:
        print(f"Loading all {total_available} real CIFAR-10 training samples for FID...")
        samples = torch.stack([dataset[i][0] for i in range(total_available)])
    else:
        print(f"Loading {n_samples} randomly sampled real CIFAR-10 training samples for FID...")
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        indices = torch.randperm(total_available)[:n_samples]
        torch.random.set_rng_state(rng_state)
        samples = torch.stack([dataset[i][0] for i in indices.tolist()])
    
    return samples


def save_sample_grid(samples, path, nrow=8):
    """Save a grid of sample images."""
    from torchvision.utils import make_grid
    
    grid = make_grid(samples[:64], nrow=nrow, padding=2, normalize=False)
    grid = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(grid).save(path)
    print(f"  Saved sample grid to {path}")


def main():
    parser = argparse.ArgumentParser(description="MoDiff CUDA FID Benchmark")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to generate for FID (default: 10000)")
    parser.add_argument("--timesteps", type=int, default=100,
                        help="Number of DDIM timesteps (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation (default: 64)")
    parser.add_argument("--include_fp32", action="store_true",
                        help="Also run FP32 baseline for comparison")
    parser.add_argument("--use_cuda_graphs", action="store_true", default=False,
                        help="Use CUDA graphs for faster generation (default: False)")
    parser.add_argument("--no_cuda_graphs", action="store_false", dest="use_cuda_graphs",
                        help="Disable CUDA graphs")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results_cuda",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip_fid", action="store_true",
                        help="Skip FID calculation to focus on generation speed")
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"MoDiff CUDA FID Benchmark")
    print(f"Samples: {args.num_samples}")
    print(f"Timesteps: {args.timesteps}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading CIFAR-10 diffusion model...")
    fp32_model = load_cifar10_model(device)
    
    # Load real CIFAR-10 samples
    if not args.skip_fid:
        print(f"\nLoading real CIFAR-10 samples...")
        real_samples = load_cifar10_real_samples(args.num_samples, device)
    else:
        real_samples = None
        print("\nSkipping loading real CIFAR-10 samples (FID skipped).")
    
    results = {}

    # FP32 baseline (optional)
    if args.include_fp32:
        print("\n" + "="*60)
        print("FP32 Baseline")
        print("="*60)
        
        torch.manual_seed(args.seed)
        fp32_samples, fp32_time = generate_samples(
            fp32_model, args.num_samples, device, 
            timesteps=args.timesteps, batch_size=args.batch_size,
            use_cuda_graphs=False  # FP32 doesn't benefit from graphs
        )
        
        save_sample_grid(fp32_samples, os.path.join(args.output_dir, "fp32_samples.png"))
        
        if not args.skip_fid:
            print("\nComputing FP32 FID...")
            fp32_fid = compute_fid(fp32_samples, real_samples, device)
            results["FP32"] = {"FID": fp32_fid, "Time": fp32_time}
            print(f"FP32 FID: {fp32_fid:.2f}")
        else:
            results["FP32"] = {"FID": -1.0, "Time": fp32_time}
            print("Skipping FP32 FID computation.")
            
        print(f"Generation time: {fp32_time:.2f}s ({fp32_time/args.num_samples*1000:.1f}ms per sample)")
        
        del fp32_samples
        torch.cuda.empty_cache()
    
    # Load calibration data
    print("\nLoading calibration data...")
    calib_xs, calib_ts = load_calibration_data(device)
    
    # MoDiff CUDA quantized model
    print("\n" + "="*60)
    print(f"MoDiff CUDA W8A8")
    print("="*60)
    
    # Create MoDiff model
    print(f"\nCreating MoDiff CUDA model (W8A8)...")
    modiff_model = MoDiffModel(fp32_model)
    
    # Calibrate
    print("Calibrating quantization parameters...")
    modiff_model.calibrate((calib_xs, calib_ts), device)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with MoDiff CUDA...")
    torch.manual_seed(args.seed)
    modiff_samples, modiff_time = generate_samples(
        modiff_model, args.num_samples, device,
        timesteps=args.timesteps, batch_size=args.batch_size,
        use_modiff=True,
        use_cuda_graphs=args.use_cuda_graphs
    )
    
    save_sample_grid(modiff_samples, os.path.join(args.output_dir, f"modiff_cuda_samples.png"))
    print(f"Generated samples range: {modiff_samples.min().item():.3f} to {modiff_samples.max().item():.3f}")
    
    # Compute FID
    if not args.skip_fid:
        print(f"\nComputing MoDiff CUDA FID...")
        modiff_fid = compute_fid(modiff_samples, real_samples, device)
        results[f"MoDiff_CUDA"] = {"FID": modiff_fid, "Time": modiff_time}
        print(f"MoDiff CUDA FID: {modiff_fid:.2f}")
    else:
        results[f"MoDiff_CUDA"] = {"FID": -1.0, "Time": modiff_time}
        print("Skipping MoDiff CUDA FID computation.")
        
    print(f"Generation time: {modiff_time:.2f}s ({modiff_time/args.num_samples*1000:.1f}ms per sample)")
    
    # Save results
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    for name, metrics in results.items():
        fid = metrics["FID"]
        time_val = metrics["Time"]
        print(f"  {name}:")
        print(f"    FID: {fid:.2f}")
        print(f"    Time: {time_val:.2f}s ({time_val/args.num_samples*1000:.1f}ms/sample)")
        if "FP32" in results and name != "FP32":
            fp32_time = results["FP32"]["Time"]
            speedup = fp32_time / time_val
            print(f"    Speedup: {speedup:.2f}x vs FP32")
    
    results_path = os.path.join(args.output_dir, "fid_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
