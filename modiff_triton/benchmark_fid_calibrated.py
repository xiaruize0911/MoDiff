#!/usr/bin/env python
"""
MoDiff FID Benchmark - Comparing MoDiff quantized models against CIFAR-10 dataset.

This script:
1. Loads the pretrained CIFAR-10 diffusion model
2. Applies MoDiff quantization (W8A8 or W4A4) with proper calibration
3. Generates images and computes FID against real CIFAR-10 dataset
4. Optionally compares with FP32 baseline

Usage:
    python benchmark_fid_calibrated.py --mode w8a8 --num_samples 10000
    python benchmark_fid_calibrated.py --mode w4a4 --num_samples 10000 --include_fp32
"""

import os
import sys
import argparse
import json
from pathlib import Path
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision
import torchvision.transforms as transforms

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ddim.models.diffusion import Model

# Import MoDiff Triton kernels
from kernels.quantize import (
    quantize_symmetric_int8, quantize_symmetric_int4,
    dequantize_int8, dequantize_int4
)
from kernels.modulated_quantize import modulated_quantize_int8, modulated_quantize_int4
from kernels.gemm_w8a8 import gemm_w8a8 as triton_gemm_w8a8
from kernels.gemm_w8a8_fused import gemm_w8a8_fused  # Optimized fused kernel
from kernels.gemm_w4a4 import gemm_w4a4 as triton_gemm_w4a4
from kernels.conv_w8a8 import conv2d_int8_triton_direct, conv2d_int8_triton_accumulate


def conv2d_int8_triton_im2col(
    x_int8: torch.Tensor,
    weight_int8: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
):
    """INT8 Conv2d via unfold + Triton INT8 GEMM.

    Supports only groups=1, kernel=3x3, stride=1, padding=1, dilation=1.
    """
    assert groups == 1, "INT8 conv fast path currently supports groups=1"
    kH = kW = 3
    assert weight_int8.shape[2:] == (kH, kW), "Only 3x3 kernels supported"
    assert stride == (1, 1) and padding == (1, 1) and dilation == (1, 1), "Only stride=1, pad=1, dil=1 supported"

    N, C, H, W = x_int8.shape
    out_channels = weight_int8.shape[0]

    # Unfold expects floating types; dequant -> unfold -> requant to INT8
    x_fp = x_int8.float() * act_scale
    x_cols_fp = F.unfold(x_fp, kernel_size=3, dilation=1, padding=1, stride=1)
    # Re-quantize using same activation scale
    x_cols = torch.round(x_cols_fp / act_scale).clamp(-128, 127).to(torch.int8)
    # Shape: (N, K, L) where K = C*9, L = H*W
    N_batch, K, L = x_cols.shape
    x_mat = x_cols.transpose(1, 2).contiguous().view(-1, K)  # [N*L, K]

    # Prepare weights: [out_channels, C, kH, kW] -> [K, out_channels]
    w_mat = weight_int8.view(out_channels, -1).t().contiguous()

    # GEMM: [N*L, K] @ [K, out_channels] -> [N*L, out_channels]
    out_mat = triton_gemm_w8a8(x_mat, w_mat, act_scale, weight_scale, bias=bias)

    # Reshape back to NCHW
    out = out_mat.view(N, L, out_channels).transpose(1, 2).contiguous()
    out = out.view(N, out_channels, H, W)
    return out


class MoDiffConfig:
    """Configuration for MoDiff quantization."""
    def __init__(self, weight_bit=8, act_bit=8):
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.quant_act = True
        self.a_sym = True  # Symmetric activation quantization
        

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


class MoDiffLayer(nn.Module):
    """
    MoDiff quantized layer using Triton kernels.
    Wraps a linear or conv layer with modulated quantization.
    """
    def __init__(self, original_layer, weight_bit=8, act_bit=8, layer_idx=0, parent_model=None):
        super().__init__()
        self.original_layer = original_layer
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.layer_idx = layer_idx
        
        # Use weakref to avoid circular reference with parent MoDiffModel
        # This prevents PyTorch's module traversal from getting stuck in infinite recursion
        object.__setattr__(self, '_parent_model_ref', weakref.ref(parent_model) if parent_model else None)
        
        # Weight quantization parameters (calibrated)
        self.weight_scale = None
        self.weight_zero_point = None
        self.quantized_weight = None
        
        # Activation quantization parameters
        self.act_scale = None
        self.act_zero_point = None
        
        # MoDiff temporal cache (static buffers for CUDA Graphs)
        self.prev_activation = None
        self.prev_quantized = None
        self.prev_output = None
        
        # Running statistics for activation calibration
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
        self.calibrating = False
        self.has_act_scale = False
        
    def calibrate_weights(self):
        """Calibrate weight quantization parameters using min-max."""
        if hasattr(self.original_layer, 'weight'):
            weight = self.original_layer.weight.data
            w_min = weight.min()
            w_max = weight.max()
            
            # Symmetric quantization for weights
            w_absmax = torch.max(torch.abs(w_min), torch.abs(w_max))
            n_levels = 2 ** self.weight_bit
            self.weight_scale = w_absmax / (n_levels // 2 - 1)
            self.weight_zero_point = torch.zeros(1, device=weight.device)
            
            # Pre-quantize weights
            self.quantized_weight = torch.clamp(
                torch.round(weight / (self.weight_scale + 1e-8)),
                -(n_levels // 2), n_levels // 2 - 1
            ).to(torch.int8)
    
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
            n_levels = 2 ** self.act_bit
            self.act_scale = a_absmax / (n_levels // 2 - 1)
            self.act_zero_point = torch.zeros(1, device=self.running_min.device)
            self.has_act_scale = (self.act_scale > 0).item()
    
    def forward(self, x, use_modiff=True):
        """Forward pass with MoDiff quantization."""
        # Check parent model's flag if available (using weakref)
        parent_model_ref = object.__getattribute__(self, '_parent_model_ref')
        if parent_model_ref is not None:
            parent = parent_model_ref()
            if parent is not None and hasattr(parent, 'use_modiff_flag'):
                use_modiff = parent.use_modiff_flag
        
        # Update activation stats if calibrating
        self.update_activation_stats(x)
        
        if self.weight_scale is None:
            # Not calibrated, run in FP32
            return self.original_layer(x)
        
        # Dequantize weights for fallback path
        weight_fp = self.quantized_weight.float() * self.weight_scale
        
        # Quantization state
        act_scale = None
        x_int8 = None
        x_fp = x
        
        # Quantize activations if calibrated
        if self.act_scale is not None and self.has_act_scale:
            if use_modiff and self.prev_quantized is not None:
                if self.act_bit == 8:
                    # MoDiff modulated quantization returns INT8 residual and updated cache
                    # Use fixed scale to avoid reduction kernel
                    # Note: Using fixed scale might degrade quality if calibration is poor
                    # For now, we use dynamic scale to ensure correctness, as fusion didn't improve speed much
                    x_int8, a_hat_new, act_scale = modulated_quantize_int8(
                        x, self.prev_quantized, scale=None
                    )
                    # In-place update for CUDA Graphs
                    self.prev_quantized.copy_(a_hat_new)
                    x_fp = a_hat_new  # dequantized activation for fallback paths
                else:  # 4-bit modulated
                    res_packed, a_hat_new, act_scale, _ = modulated_quantize_int4(
                        x, self.prev_quantized, scale=None
                    )
                    self.prev_quantized.copy_(a_hat_new)
                    x_fp = a_hat_new
            else:
                if self.act_bit == 8:
                    x_int8, act_scale = quantize_symmetric_int8(x, self.act_scale)
                    x_fp = dequantize_int8(x_int8, act_scale)
                else:
                    x_packed, act_scale, shape = quantize_symmetric_int4(x, self.act_scale)
                    x_fp = dequantize_int4(x_packed, act_scale, shape)
            
            # Update cache for next timestep
            if self.prev_activation is None:
                self.prev_activation = x.detach().clone()
            else:
                self.prev_activation.copy_(x)
            
            if self.prev_quantized is None:
                self.prev_quantized = x_fp.detach().clone()
            elif not use_modiff:
                self.prev_quantized.copy_(x_fp)
        
        # Fast INT8 GEMM path for Linear layers on CUDA (non-MoDiff only)
        if (
            not use_modiff and
            isinstance(self.original_layer, nn.Linear) and
            self.weight_bit == 8 and self.act_bit == 8 and
            x_int8 is not None and torch.cuda.is_available()
        ):
            orig_shape = x_int8.shape
            K = orig_shape[-1]
            x_2d = x_int8.view(-1, K).contiguous()
            # Weight: [out, in] -> [in, out]
            weight_int8 = self.quantized_weight.contiguous().t()
            out_2d = triton_gemm_w8a8(
                x_2d, weight_int8, act_scale, self.weight_scale, bias=self.original_layer.bias
            )
            out = out_2d.view(*orig_shape[:-1], weight_int8.shape[1])
            return out

        # Fast INT8 Conv2d path via direct Triton kernel (MoDiff-safe: uses current quantized activations)
        if (
            isinstance(self.original_layer, nn.Conv2d) and
            self.weight_bit == 8 and self.act_bit == 8 and
            x_int8 is not None and act_scale is not None and torch.cuda.is_available()
        ):
            # Shape guards: only support groups=1, kernel=3x3, stride=1, pad=1, dil=1
            layer = self.original_layer
            if (
                layer.groups == 1 and
                layer.kernel_size == (3, 3) and
                layer.stride == (1, 1) and
                layer.padding == (1, 1) and
                layer.dilation == (1, 1)
            ):
                if use_modiff and self.prev_output is not None:
                    # Fused path: out = prev_output + conv(residual, bias=None)
                    # We use the fused kernel to avoid allocating delta_out and Python add
                    out = conv2d_int8_triton_accumulate(
                        x_int8, self.quantized_weight, self.prev_output, act_scale, self.weight_scale, bias=None
                    )
                else:
                    # Standard path (or first step): out = conv(x, bias)
                    out = conv2d_int8_triton_direct(
                        x_int8, self.quantized_weight, act_scale, self.weight_scale, bias=layer.bias
                    )
                
                if self.prev_output is None:
                    self.prev_output = out.detach().clone()
                else:
                    self.prev_output.copy_(out)
                return out
        
        # Fallback to FP path (still quantized weights)
        if isinstance(self.original_layer, nn.Linear):
            return nn.functional.linear(x_fp, weight_fp, self.original_layer.bias)
        elif isinstance(self.original_layer, nn.Conv2d):
            return nn.functional.conv2d(
                x_fp, weight_fp, self.original_layer.bias,
                self.original_layer.stride, self.original_layer.padding,
                self.original_layer.dilation, self.original_layer.groups
            )
        else:
            return self.original_layer(x_fp)
    
    def reset_cache(self):
        """Reset MoDiff temporal cache."""
        self.prev_activation = None
        self.prev_quantized = None
        self.prev_output = None


class MoDiffModel(nn.Module):
    """
    Wrapper that applies MoDiff quantization to a diffusion model.
    """
    def __init__(self, model, weight_bit=8, act_bit=8):
        super().__init__()
        self.model = model
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.modiff_layers = []
        self.use_modiff_flag = True  # Global flag for all layers
        
        # Wrap quantizable layers
        self._wrap_layers(model)
    
    def _wrap_layers(self, module, prefix=''):
        """Recursively wrap layers with MoDiff quantization."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                # Create MoDiff layer
                modiff_layer = MoDiffLayer(
                    child, 
                    weight_bit=self.weight_bit,
                    act_bit=self.act_bit,
                    layer_idx=len(self.modiff_layers),
                    parent_model=self  # Pass reference to parent
                )
                setattr(module, name, modiff_layer)
                self.modiff_layers.append(modiff_layer)
            else:
                self._wrap_layers(child, full_name)
    
    def calibrate(self, calib_data, device):
        """
        Calibrate quantization parameters using calibration data.
        
        Args:
            calib_data: Tuple of (xs, ts) calibration tensors
            device: Device to run calibration on
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
        n_calib = min(256, xs.shape[0])  # Use up to 256 samples
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
        # Store the flag for layers to access
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
            # Also needed at top level for forward pass
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
        # Generate synthetic calibration data
        xs = torch.randn(256, 3, 32, 32)
        ts = torch.randint(0, 1000, (256,)).float()
        return xs, ts
    
    print(f"Loading calibration data from {cali_path}")
    cali_data = torch.load(cali_path)
    
    # Handle dict format
    if isinstance(cali_data, dict):
        xs = cali_data['xs']  # Shape: [20, 2048, 3, 32, 32]
        ts = cali_data['ts']  # Shape: [20, 2048]
        
        # Flatten across timesteps
        xs = xs.reshape(-1, 3, 32, 32)  # [40960, 3, 32, 32]
        ts = ts.reshape(-1)  # [40960]
    else:
        xs, ts = cali_data
    
    return xs, ts


def generate_samples(model, n_samples, device, timesteps=100, batch_size=64, use_modiff=True, use_cuda_graphs=True):
    """
    Generate samples using DDIM sampling with optional CUDA graphs.
    
    Args:
        model: Diffusion model (FP32 or MoDiff quantized)
        n_samples: Number of samples to generate
        device: Device to run on
        timesteps: Number of DDIM steps
        batch_size: Batch size for generation
        use_modiff: Whether to use MoDiff (only applicable for MoDiffModel)
        use_cuda_graphs: Whether to use CUDA graphs for acceleration
    
    Returns:
        samples: Tensor of generated samples [n_samples, 3, 32, 32]
        elapsed_time: Time taken in seconds
    """
    import time
    start_time = time.time()
    # Setup diffusion parameters
    betas = get_beta_schedule("linear", 0.0001, 0.02, 1000)
    betas = torch.from_numpy(betas).float().to(device)
    
    # DDIM timestep schedule
    skip = 1000 // timesteps
    seq = range(0, 1000, skip)
    
    all_samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    is_modiff = isinstance(model, MoDiffModel)
    
    # CUDA Graph variables (shared across batches of same size)
    cuda_graphs = {}  # Cache graphs per batch size
    
    for batch_idx in tqdm(range(n_batches), desc="Generating"):
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        # Start from random noise
        x = torch.randn(current_batch_size, 3, 32, 32, device=device)
        
        # Reset MoDiff cache for new samples
        if is_modiff:
            model.reset_cache()
        
        # DDIM sampling loop
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            
            # CUDA Graph for this batch size
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
                
                # Model prediction with CUDA graphs
                use_graph = (graph_data is not None and step_idx > 0)  # Skip first step
                
                if use_graph:
                    if graph_data['graph'] is None:
                        # Capture graph on second step (first MoDiff step with caching)
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
                    
                    # Use cached at values
                    at = graph_data['static_at']
                    at_next = graph_data['static_at_next']
                else:
                    # Standard eager execution
                    if is_modiff:
                        et = model(x, t.float(), use_modiff=use_modiff)
                    else:
                        et = model(x, t.float())
                
                # Check model output for NaN/Inf
                if torch.isnan(et).any() or torch.isinf(et).any():
                    et = torch.nan_to_num(et, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # DDIM update
                x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
                x0_t = torch.clamp(x0_t, -1, 1)
                
                c1 = 0  # eta parameter
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                x = at_next.sqrt() * x0_t + c2 * et
                
                # Check for NaN immediately after update
                if torch.isnan(x).any() or torch.isinf(x).any():
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Clamp to prevent NaN propagation
                x = torch.clamp(x, -10, 10)
        
        # Normalize to [0, 1]
        samples = (x + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Check for NaN
        if torch.isnan(samples).any() or torch.isinf(samples).any():
            samples = torch.nan_to_num(samples, nan=0.5, posinf=1.0, neginf=0.0)
        
        all_samples.append(samples.cpu())
    
    elapsed_time = time.time() - start_time
    return torch.cat(all_samples, dim=0)[:n_samples], elapsed_time


def compute_fid(generated_samples, real_samples, device):
    """
    Compute FID between generated and real samples using InceptionV3 features.
    Uses the original implementation from scripts/evaluate.py
    """
    from torch.nn.functional import adaptive_avg_pool2d
    
    # Load InceptionV3
    inception = torchvision.models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
    inception.fc = nn.Identity()  # Remove final FC layer
    inception = inception.to(device)
    inception.eval()
    
    def get_features(samples, batch_size=64, desc="  Extracting features"):
        """Extract InceptionV3 features."""
        features = []
        n_batches = (len(samples) + batch_size - 1) // batch_size
        
        # Resize and normalize for Inception
        resize = transforms.Resize((299, 299), antialias=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc=desc):
                batch = samples[i*batch_size:(i+1)*batch_size].to(device)
                
                # Check for NaN in input
                if torch.isnan(batch).any() or torch.isinf(batch).any():
                    batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=0.0)
                
                batch = resize(batch)
                batch = normalize(batch)
                
                # Check for NaN after normalization
                if torch.isnan(batch).any() or torch.isinf(batch).any():
                    batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=0.0)
                
                feat = inception(batch)
                
                # Check features
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    feat = torch.nan_to_num(feat, nan=0.0)
                
                features.append(feat.cpu())
                
                # Clear cache every 10 batches
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        
        return torch.cat(features, dim=0).numpy()
    
    print("  Computing features for generated samples...")
    gen_features = get_features(generated_samples, desc="  Generated features")
    
    print("  Computing features for real samples...")
    real_features = get_features(real_samples, desc="  Real features")
    
    print("  Computing statistics...")
    # Compute statistics
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    # Check for NaN/Inf
    if np.any(np.isnan(mu_gen)) or np.any(np.isinf(mu_gen)):
        return float('nan')
    
    if np.any(np.isnan(sigma_gen)) or np.any(np.isinf(sigma_gen)):
        return float('nan')
    
    # Compute FID using original method from scripts/evaluate.py
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
    print("  Computing FID score...")
    mu1, sigma1 = mu_gen, sigma_gen
    mu2, sigma2 = mu_real, sigma_real
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    from scipy import linalg
    eps = 1e-6
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f"  Warning: FID calculation produces singular product; adding {eps} to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fid)


def load_cifar10_real_samples(n_samples, device):
    """
    Load real CIFAR-10 samples for FID computation.
    
    For proper FID evaluation, we use the full training set (50k images)
    regardless of how many samples are generated. This matches the standard
    protocol for FID evaluation.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Use absolute path for data directory
    data_root = Path(__file__).parent / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=True, download=True, transform=transform
        )
    except RuntimeError as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        print("Attempting to download dataset...")
        # Force download by removing existing corrupted files
        import shutil
        cifar_dir = data_root / "cifar-10-batches-py"
        if cifar_dir.exists():
            shutil.rmtree(cifar_dir)
        
        # Try downloading again
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=True, download=True, transform=transform
        )
    
    # Use full training set for FID (standard protocol)
    # This gives more stable statistics than random subsampling
    print(f"Loading {len(dataset)} real CIFAR-10 training samples for FID...")
    samples = torch.stack([dataset[i][0] for i in range(len(dataset))])
    
    return samples


def save_sample_grid(samples, path, nrow=8):
    """Save a grid of sample images."""
    from torchvision.utils import make_grid
    
    grid = make_grid(samples[:64], nrow=nrow, padding=2, normalize=False)
    grid = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(grid).save(path)
    print(f"  Saved sample grid to {path}")


def main():
    parser = argparse.ArgumentParser(description="MoDiff FID Benchmark")
    parser.add_argument("--mode", type=str, default="w8a8", choices=["w8a8", "w4a4"],
                        help="Quantization mode (default: w8a8)")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to generate for FID (default: 10000)")
    parser.add_argument("--timesteps", type=int, default=100,
                        help="Number of DDIM timesteps (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation (default: 64)")
    parser.add_argument("--include_fp32", action="store_true",
                        help="Also run FP32 baseline for comparison")
    parser.add_argument("--include_standard_ptq", action="store_true",
                        help="Also run standard PTQ (no MoDiff) for comparison")
    parser.add_argument("--use_cuda_graphs", action="store_true", default=True,
                        help="Use CUDA graphs for faster generation (default: True)")
    parser.add_argument("--no_cuda_graphs", action="store_false", dest="use_cuda_graphs",
                        help="Disable CUDA graphs")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse quantization mode
    if args.mode == "w8a8":
        weight_bit, act_bit = 8, 8
    else:  # w4a4
        weight_bit, act_bit = 4, 4
    
    print(f"\n{'='*60}")
    print(f"MoDiff FID Benchmark")
    print(f"Mode: {args.mode.upper()} (W{weight_bit}A{act_bit})")
    print(f"Samples: {args.num_samples}")
    print(f"Timesteps: {args.timesteps}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading CIFAR-10 diffusion model...")
    fp32_model = load_cifar10_model(device)
    
    # Load real CIFAR-10 samples
    print(f"\nLoading real CIFAR-10 samples...")
    real_samples = load_cifar10_real_samples(args.num_samples, device)
    
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
        
        print("\nComputing FP32 FID...")
        fp32_fid = compute_fid(fp32_samples, real_samples, device)
        results["FP32"] = {"FID": fp32_fid, "Time": fp32_time}
        print(f"FP32 FID: {fp32_fid:.2f}")
        print(f"Generation time: {fp32_time:.2f}s ({fp32_time/args.num_samples*1000:.1f}ms per sample)")
        
        del fp32_samples
        torch.cuda.empty_cache()
    
    # Load calibration data (shared by all quantized models)
    print("\nLoading calibration data...")
    calib_xs, calib_ts = load_calibration_data(device)
    
    # Standard PTQ (no MoDiff temporal caching)
    if args.include_standard_ptq:
        print("\n" + "="*60)
        print(f"Standard PTQ {args.mode.upper()} (No MoDiff)")
        print("="*60)
        
        # Create standard PTQ model (same as MoDiff but without temporal caching)
        print(f"\nCreating standard PTQ model (W{weight_bit}A{act_bit})...")
        fp32_model_ptq = load_cifar10_model(device)
        ptq_model = MoDiffModel(fp32_model_ptq, weight_bit=weight_bit, act_bit=act_bit)
        
        # Calibrate
        print("Calibrating quantization parameters...")
        ptq_model.calibrate((calib_xs, calib_ts), device)
        
        # Generate samples WITHOUT MoDiff (use_modiff=False)
        print(f"\nGenerating {args.num_samples} samples with standard PTQ...")
        torch.manual_seed(args.seed)
        ptq_samples, ptq_time = generate_samples(
            ptq_model, args.num_samples, device,
            timesteps=args.timesteps, batch_size=args.batch_size,
            use_modiff=False,  # Disable MoDiff temporal caching
            use_cuda_graphs=False  # PTQ without MoDiff doesn't benefit
        )
        
        save_sample_grid(ptq_samples, os.path.join(args.output_dir, f"ptq_{args.mode}_samples.png"))
        
        # Compute FID
        print(f"\nComputing Standard PTQ {args.mode.upper()} FID...")
        ptq_fid = compute_fid(ptq_samples, real_samples, device)
        results[f"PTQ_{args.mode.upper()}"] = {"FID": ptq_fid, "Time": ptq_time}
        print(f"Standard PTQ {args.mode.upper()} FID: {ptq_fid:.2f}")
        print(f"Generation time: {ptq_time:.2f}s ({ptq_time/args.num_samples*1000:.1f}ms per sample)")
        
        del ptq_samples, ptq_model, fp32_model_ptq
        torch.cuda.empty_cache()
    
    # MoDiff quantized model
    print("\n" + "="*60)
    print(f"MoDiff {args.mode.upper()}")
    print("="*60)
    
    # Create MoDiff model
    print(f"\nCreating MoDiff model (W{weight_bit}A{act_bit})...")
    modiff_model = MoDiffModel(fp32_model, weight_bit=weight_bit, act_bit=act_bit)
    
    # Calibrate
    print("Calibrating quantization parameters...")
    
    print("Calibrating quantization parameters...")
    modiff_model.calibrate((calib_xs, calib_ts), device)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with MoDiff...")
    torch.manual_seed(args.seed)
    modiff_samples, modiff_time = generate_samples(
        modiff_model, args.num_samples, device,
        timesteps=args.timesteps, batch_size=args.batch_size,
        use_modiff=True,
        use_cuda_graphs=args.use_cuda_graphs  # Use CUDA graphs if enabled
    )
    
    save_sample_grid(modiff_samples, os.path.join(args.output_dir, f"modiff_{args.mode}_samples.png"))
    
    # Compute FID
    print(f"\nComputing MoDiff {args.mode.upper()} FID...")
    modiff_fid = compute_fid(modiff_samples, real_samples, device)
    results[f"MoDiff_{args.mode.upper()}"] = {"FID": modiff_fid, "Time": modiff_time}
    print(f"MoDiff {args.mode.upper()} FID: {modiff_fid:.2f}")
    print(f"Generation time: {modiff_time:.2f}s ({modiff_time/args.num_samples*1000:.1f}ms per sample)")
    
    # Save results
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    for name, metrics in results.items():
        if isinstance(metrics, dict):
            fid = metrics["FID"]
            time_val = metrics["Time"]
            print(f"  {name}:")
            print(f"    FID: {fid:.2f}")
            print(f"    Time: {time_val:.2f}s ({time_val/args.num_samples*1000:.1f}ms/sample)")
            if "FP32" in results and name != "FP32":
                fp32_time = results["FP32"]["Time"]
                speedup = fp32_time / time_val
                print(f"    Speedup: {speedup:.2f}x vs FP32")
        else:
            # Old format compatibility
            print(f"  {name}: FID = {metrics:.2f}")
    
    results_path = os.path.join(args.output_dir, "fid_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
