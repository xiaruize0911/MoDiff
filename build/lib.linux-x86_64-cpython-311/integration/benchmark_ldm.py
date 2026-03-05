"""
Unified LDM Benchmark with Multiple Precision Modes.

Supports:
- FP32: Standard baseline (full precision)
- FP16: Half precision with autocast
- INT8: MoDiff INT8 with optimized kernels + temporal caching
- INT8_baseline: INT8 optimized kernels WITHOUT temporal caching
- INT4: MoDiff INT4 with optimized kernels + temporal caching
- INT4_baseline: INT4 optimized kernels WITHOUT temporal caching

Key differences:
- MoDiff modes (int8/int4): Use temporal caching (reuses cached activations across timesteps)
                           + static pre-calibrated scales (no per-step absmax sync)
- Baseline modes: Same INT8/INT4 kernels + buffer pool + static scales, but no temporal caching
                  After fixing both unfair advantages (buffer pool + static scales), baseline
                  should be marginally FASTER than MoDiff (temporal caching adds subtract/accumulate
                  overhead per step without skipping any convolutions)

Note: All INT8/INT4 modes use the same per-layer quantization approach.
      Baseline modes isolate the performance impact of temporal caching.

Performance hierarchy (expected after fair comparison fix):
  Speed: INT8_baseline ≥ INT8 (MoDiff) > INT4_baseline ≥ INT4 (MoDiff) > FP16 > FP32
  (Baseline is slightly faster than MoDiff because temporal caching adds sub+accumulate overhead
   without skipping any convolutions; the quality benefit of MoDiff is accuracy, not speed)

Usage:
    python integration/benchmark_ldm.py --mode all --steps 50
    python integration/benchmark_ldm.py --mode int8_baseline --num_samples 100 --eval_fid
    python integration/benchmark_ldm.py --mode int8 --steps 50  # With temporal caching
"""
import argparse
import os
import sys

# Set memory management policy BEFORE any torch.cuda calls
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import json
import warnings
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import torchvision.utils as tvu
from tqdm import tqdm

# Suppress NNPACK warnings (not needed for GPU)
warnings.filterwarnings('ignore', message='Could not initialize NNPACK')
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')

# Enable TF32 for faster FP32 operations on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN for optimized convolution kernels
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.getcwd())

from integration.profiler import profiler
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# INT8 imports
try:
    from integration.int8_optimized import (
        OptimizedInt8Conv2d,
        convert_model_to_optimized_int8,
        apply_static_scales,
        enable_modiff_mode as enable_modiff_mode_int8,
        reset_modiff_state as reset_modiff_state_int8,
        set_calibrating as set_calibrating_int8,
        get_calibration_config as get_calibration_config_int8,
        reset_calibration as reset_calibration_int8,
    )
    from integration.profiler import profiler  # Added for profiling
    HAS_INT8 = True
except ImportError:
    HAS_INT8 = False
    print("Warning: INT8 not available")

# INT8 Linear imports
try:
    from integration.int8_linear import (
        OptimizedInt8Linear,
        convert_model_to_int8_linear,
        enable_modiff_mode_linear,
        reset_modiff_state_linear,
        set_calibrating_linear,
        export_linear_static_scales,
        apply_linear_static_scales,
    )
    HAS_INT8_LINEAR = True
except ImportError:
    HAS_INT8_LINEAR = False
    print("Warning: INT8 Linear not available")

# INT4 imports
try:
    from integration.int4_optimized import (
        OptimizedInt4Conv2d,
        convert_model_to_optimized_int4,
        enable_modiff_mode as enable_modiff_mode_int4,
        reset_modiff_state as reset_modiff_state_int4,
        apply_int4_static_scales,
        export_int4_static_scales,
    )
    HAS_INT4 = True
except ImportError:
    HAS_INT4 = False
    print("Warning: INT4 not available")

# INT4 Linear imports
try:
    from integration.int4_linear import (
        OptimizedInt4Linear,
        convert_model_to_int4_linear,
        enable_modiff_mode_int4_linear,
        reset_modiff_state_int4_linear,
        set_calibrating_int4_linear,
        export_int4_linear_static_scales,
        apply_int4_linear_static_scales,
    )
    HAS_INT4_LINEAR = True
except ImportError:
    HAS_INT4_LINEAR = False
    print("Warning: INT4 Linear not available")

# FID (optional)
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False

# Standard PyTorch quantization for baseline
try:
    import torch.quantization as quant
    HAS_TORCH_QUANT = True
except ImportError:
    HAS_TORCH_QUANT = False


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
    conv_types = [nn.Conv2d]
    if HAS_INT8:
        conv_types.append(OptimizedInt8Conv2d)
    if HAS_INT4:
        conv_types.append(OptimizedInt4Conv2d)
    return sum(1 for m in model.modules() if isinstance(m, tuple(conv_types)))


def count_linear_layers(model: nn.Module) -> int:
    """Count Linear layers in model."""
    linear_types = [nn.Linear]
    if HAS_INT8_LINEAR:
        linear_types.append(OptimizedInt8Linear)
    if HAS_INT4_LINEAR:
        linear_types.append(OptimizedInt4Linear)
    return sum(1 for m in model.modules() if isinstance(m, tuple(linear_types)))


def convert_to_int8_baseline(model: nn.Module):
    """Convert model to INT8 using standard PyTorch (no MoDiff)."""
    model.qconfig = torch.quantization.get_default_qconfig('x86')
    torch.quantization.prepare(model, inplace=True)
    # Note: calibration happens during first forward pass
    torch.quantization.convert(model, inplace=True)
    return model


def convert_to_int4_baseline(model: nn.Module):
    """Simulate INT4 with INT8 using standard PyTorch (no MoDiff)."""
    # PyTorch doesn't have native INT4, so we use INT8 as a proxy
    model.qconfig = torch.quantization.get_default_qconfig('x86')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    return model


class FullPipelineInt8Wrapper(nn.Module):
    """Wrapper that quantizes input once and keeps entire pipeline in INT8."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_scale = 127.0 / 6.0  # Assume input range [-6, 6]
        
    def forward(self, x, timesteps, context=None):
        # Quantize input once
        x_int8 = (x * self.input_scale).clamp(-127, 127).to(torch.int8)
        # Forward stays in INT8 (layers handle internally)
        out_int8 = self.model(x_int8.float() / self.input_scale, timesteps, context)
        # Output already dequantized by final layer
        return out_int8


class FullPipelineInt4Wrapper(nn.Module):
    """Wrapper that quantizes input once and keeps entire pipeline in INT4."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_scale = 7.0 / 6.0  # Assume input range [-6, 6], INT4 is [-7, 7]
        
    def forward(self, x, timesteps, context=None):
        # Quantize input once to INT4 range
        x_int4 = (x * self.input_scale).clamp(-7, 7).round()
        # Forward stays in INT4 (layers handle internally)
        out_int4 = self.model(x_int4 / self.input_scale, timesteps, context)
        return out_int4


class FullPipelineInt8Wrapper(nn.Module):
    """Wrapper that quantizes input once and keeps entire pipeline in INT8."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_scale = 127.0 / 6.0  # Assume input range [-6, 6]
        
    def forward(self, x, timesteps, context=None):
        # Quantize input once
        x_int8 = (x * self.input_scale).clamp(-127, 127).to(torch.int8)
        # Forward stays in INT8 (layers handle internally)
        out_int8 = self.model(x_int8.float() / self.input_scale, timesteps, context)
        # Output already dequantized by final layer
        return out_int8


class FullPipelineInt4Wrapper(nn.Module):
    """Wrapper that quantizes input once and keeps entire pipeline in INT4."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_scale = 7.0 / 6.0  # Assume input range [-6, 6], INT4 is [-7, 7]
        
    def forward(self, x, timesteps, context=None):
        # Quantize input once to INT4 range
        x_int4 = (x * self.input_scale).clamp(-7, 7).round()
        # Forward stays in INT4 (layers handle internally)
        out_int4 = self.model(x_int4 / self.input_scale, timesteps, context)
        return out_int4


class BenchmarkRunner:
    """Unified benchmark runner for all precision modes."""
    
    def __init__(self, config_path: str, ckpt_path: str, output_dir: str,
                 batch_size: int = 4, steps: int = 50, shape: tuple = (4, 32, 32),
                 calibration_path: str = None):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.steps = steps
        self.shape = shape
        self.calibration_path = calibration_path
        self.results = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _setup_model(self, mode: str):
        """Load and setup model for given mode."""
        model, _ = load_model(self.config_path, self.ckpt_path)
        
        # Configure backends for maximum speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Use channels_last memory format to eliminate NCHW↔NHWC conversions
        # cuDNN and CUTLASS both prefer NHWC; this avoids costly layout transposes
        model = model.to(memory_format=torch.channels_last)
        
        # Disable gradient checkpointing for inference (eliminates autograd overhead)
        for m in model.modules():
            if hasattr(m, 'use_checkpoint'):
                m.use_checkpoint = False
        # Patch AttentionBlock to bypass checkpoint (it hardcodes flag=True)
        from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
        AttentionBlock.forward = lambda self, x: self._forward(x)
        
        # Apply ResBlock fusion for all modes (8-12% speedup)
        from integration.fused_resblock import fuse_resblocks_in_module, print_fusion_summary
        print("\n" + "="*60)
        print("Applying ResBlock Fusion Optimization")
        print("="*60)
        fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)
        print_fusion_summary(model.model.diffusion_model)
        
        if mode == 'int8' and HAS_INT8:
            print(f"Converting UNet to INT8 ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int8(model.model.diffusion_model)
            
            # Also convert linear layers to INT8 with MoDiff
            if HAS_INT8_LINEAR:
                print(f"Converting UNet linear layers to INT8 ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int8_linear(model.model.diffusion_model)
            
            # Initialize buffer pool for pre-allocated buffers
            from integration.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            
            # Load static calibration if available
            if self.calibration_path and os.path.exists(self.calibration_path):
                print(f"Loading static calibration from {self.calibration_path}")
                scales = torch.load(self.calibration_path, weights_only=True)
                config = get_calibration_config_int8()
                config.scales = scales
                config.is_calibrated = True
                loaded = apply_static_scales(model.model.diffusion_model, scales)
                print(f"✓ Loaded {loaded} INT8 conv layer scales (static quantization enabled)")
                # Load linear scales if present
                if HAS_INT8_LINEAR:
                    linear_scales = {k: v for k, v in scales.items() if k.startswith('linear:')}
                    if linear_scales:
                        clean_scales = {k.replace('linear:', ''): v for k, v in linear_scales.items()}
                        loaded_lin = apply_linear_static_scales(model.model.diffusion_model, clean_scales)
                        print(f"✓ Loaded {loaded_lin} INT8 linear layer scales")
            
            enable_modiff_mode_int8(model.model.diffusion_model, True)
            if HAS_INT8_LINEAR:
                enable_modiff_mode_linear(model.model.diffusion_model, True)
        elif mode == 'int4' and HAS_INT4:
            print(f"Converting UNet to INT4 ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int4(model.model.diffusion_model)
            
            # Also convert linear layers to INT4 with MoDiff
            if HAS_INT4_LINEAR:
                print(f"Converting UNet linear layers to INT4 ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int4_linear(model.model.diffusion_model)
            
            # Initialize buffer pool for pre-allocated buffers
            from integration.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')

            # Load static INT4 activation scales if provided
            if self.calibration_path and os.path.exists(self.calibration_path):
                print(f"Loading INT4 calibration from {self.calibration_path}")
                scales = torch.load(self.calibration_path, weights_only=True)
                loaded = apply_int4_static_scales(model.model.diffusion_model, scales)
                print(f"✓ Loaded static scales for {loaded} INT4 conv layers")
                # Load INT4 linear scales if present
                if HAS_INT4_LINEAR:
                    linear_scales = {k: v for k, v in scales.items() if k.startswith('linear:')}
                    if linear_scales:
                        clean_scales = {k.replace('linear:', ''): v for k, v in linear_scales.items()}
                        loaded_lin = apply_int4_linear_static_scales(model.model.diffusion_model, clean_scales)
                        print(f"✓ Loaded {loaded_lin} INT4 linear layer scales")
            
            enable_modiff_mode_int4(model.model.diffusion_model, True)
            if HAS_INT4_LINEAR:
                enable_modiff_mode_int4_linear(model.model.diffusion_model, True)
        elif mode == 'int8_baseline' and HAS_INT8:
            print(f"Converting UNet to INT8 Baseline (INT8 kernels without MoDiff temporal caching) ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int8(model.model.diffusion_model)
            
            # Also convert linear layers to INT8 (baseline = no temporal caching)
            if HAS_INT8_LINEAR:
                print(f"Converting UNet linear layers to INT8 ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int8_linear(model.model.diffusion_model)
            
            from integration.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            if self.calibration_path and os.path.exists(self.calibration_path):
                print(f"Loading static calibration from {self.calibration_path}")
                scales = torch.load(self.calibration_path, weights_only=True)
                config = get_calibration_config_int8()
                config.scales = scales
                config.is_calibrated = True
                loaded = apply_static_scales(model.model.diffusion_model, scales)
                print(f"✓ Loaded {loaded} INT8 conv layer scales for baseline (static quantization enabled)")
                if HAS_INT8_LINEAR:
                    linear_scales = {k: v for k, v in scales.items() if k.startswith('linear:')}
                    if linear_scales:
                        clean_scales = {k.replace('linear:', ''): v for k, v in linear_scales.items()}
                        loaded_lin = apply_linear_static_scales(model.model.diffusion_model, clean_scales)
                        print(f"✓ Loaded {loaded_lin} INT8 linear layer scales for baseline")
            enable_modiff_mode_int8(model.model.diffusion_model, False)  # Disable temporal caching
            if HAS_INT8_LINEAR:
                enable_modiff_mode_linear(model.model.diffusion_model, False)  # Disable temporal caching
        elif mode == 'int4_baseline' and HAS_INT4:
            print(f"Converting UNet to INT4 Baseline (INT4 kernels without MoDiff temporal caching) ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int4(model.model.diffusion_model)
            
            # Also convert linear layers to INT4 (baseline = no temporal caching)
            if HAS_INT4_LINEAR:
                print(f"Converting UNet linear layers to INT4 ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int4_linear(model.model.diffusion_model)
            
            from integration.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            if self.calibration_path and os.path.exists(self.calibration_path):
                print(f"Loading INT4 calibration from {self.calibration_path}")
                scales = torch.load(self.calibration_path, weights_only=True)
                loaded = apply_int4_static_scales(model.model.diffusion_model, scales)
                print(f"✓ Loaded static scales for {loaded} INT4 baseline conv layers")
                if HAS_INT4_LINEAR:
                    linear_scales = {k: v for k, v in scales.items() if k.startswith('linear:')}
                    if linear_scales:
                        clean_scales = {k.replace('linear:', ''): v for k, v in linear_scales.items()}
                        loaded_lin = apply_int4_linear_static_scales(model.model.diffusion_model, clean_scales)
                        print(f"✓ Loaded {loaded_lin} INT4 linear layer scales for baseline")
            enable_modiff_mode_int4(model.model.diffusion_model, False)  # Disable temporal caching
            if HAS_INT4_LINEAR:
                enable_modiff_mode_int4_linear(model.model.diffusion_model, False)  # Disable temporal caching
        elif mode == 'int8_full_pipeline' and HAS_INT8:
            print(f"Converting UNet to Full INT8 Pipeline (no per-layer Q/DQ) ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            # Full INT8: quantize once at input, stay quantized throughout
            convert_model_to_optimized_int8(model.model.diffusion_model)
            enable_modiff_mode_int8(model.model.diffusion_model, True)
            # Wrap model to handle end-to-end quantization
            model.model.diffusion_model = FullPipelineInt8Wrapper(model.model.diffusion_model)
        elif mode == 'int4_full_pipeline' and HAS_INT4:
            print(f"Converting UNet to Full INT4 Pipeline (no per-layer Q/DQ) ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            # Full INT4: quantize once at input, stay quantized throughout
            convert_model_to_optimized_int4(model.model.diffusion_model)
            enable_modiff_mode_int4(model.model.diffusion_model, True)
            # Wrap model to handle end-to-end quantization
            model.model.diffusion_model = FullPipelineInt4Wrapper(model.model.diffusion_model)
        
        return model, DDIMSampler(model)
    
    def _try_compile_unet(self, model):
        """Try to torch.compile the UNet for fused element-wise kernels."""
        try:
            model.model.diffusion_model = torch.compile(
                model.model.diffusion_model, 
                mode='reduce-overhead',
                fullgraph=False,
            )
            print("✓ torch.compile applied to UNet (reduce-overhead mode)")
        except Exception as e:
            print(f"✗ torch.compile failed: {e}")
    
    def _calibrate_int8(self, model, sampler, num_runs: int = 10):
        """Calibrate INT8 quantization scales (conv + linear)."""
        print(f"Calibrating INT8 ({num_runs} runs)...")
        reset_calibration_int8()
        set_calibrating_int8(model.model.diffusion_model, True)
        if HAS_INT8_LINEAR:
            set_calibrating_linear(model.model.diffusion_model, True)
        
        with torch.no_grad():
            for _ in range(num_runs):
                reset_modiff_state_int8(model.model.diffusion_model)
                if HAS_INT8_LINEAR:
                    reset_modiff_state_linear(model.model.diffusion_model)
                sampler.sample(S=5, batch_size=2, shape=self.shape, eta=0.0, verbose=False)
        
        get_calibration_config_int8().finalize()
        set_calibrating_int8(model.model.diffusion_model, False)
        if HAS_INT8_LINEAR:
            set_calibrating_linear(model.model.diffusion_model, False)
        num_layers = len(get_calibration_config_int8().scales)
        
        # Merge linear scales into the calibration dict
        all_scales = dict(get_calibration_config_int8().scales)
        if HAS_INT8_LINEAR:
            linear_scales = export_linear_static_scales(model.model.diffusion_model)
            for k, v in linear_scales.items():
                all_scales[f'linear:{k}'] = v
            print(f"Calibrated {num_layers} conv layers + {len(linear_scales)} linear layers")
        else:
            print(f"Calibrated {num_layers} layers")
        
        if self.calibration_path:
            torch.save(all_scales, self.calibration_path)
            print(f"✓ Saved INT8 static scales: {self.calibration_path}")
    
    def _calibrate_int4(self, model, sampler, num_runs: int = 5):
        """Calibrate INT4 quantization scales (dynamic to static transition)."""
        print(f"Calibrating INT4 ({num_runs} runs for speed)...")
        from integration.int4_optimized import set_calibrating_int4
        
        set_calibrating_int4(model.model.diffusion_model, True)
        if HAS_INT4_LINEAR:
            set_calibrating_int4_linear(model.model.diffusion_model, True)
        
        with torch.no_grad():
            for _ in range(num_runs):
                reset_modiff_state_int4(model.model.diffusion_model)
                if HAS_INT4_LINEAR:
                    reset_modiff_state_int4_linear(model.model.diffusion_model)
                # Sample a few steps to get representative activations
                sampler.sample(S=5, batch_size=self.batch_size, shape=self.shape, eta=0.0, verbose=False)
        
        set_calibrating_int4(model.model.diffusion_model, False)
        if HAS_INT4_LINEAR:
            set_calibrating_int4_linear(model.model.diffusion_model, False)
        scales = export_int4_static_scales(model.model.diffusion_model)
        
        # Merge linear scales
        if HAS_INT4_LINEAR:
            linear_scales = export_int4_linear_static_scales(model.model.diffusion_model)
            for k, v in linear_scales.items():
                scales[f'linear:{k}'] = v
            print(f"✓ Calibrated INT4 conv layers: {len(scales) - len(linear_scales)}, linear: {len(linear_scales)}")
        
        if len(scales) == 0:
            print("Warning: INT4 calibration exported 0 scales; falling back to dynamic scaling")
        if self.calibration_path:
            torch.save(scales, self.calibration_path)
            print(f"✓ Saved INT4 static scales: {self.calibration_path}")
        print("✓ INT4 Calibration finished (Static scales locked)")
    
    def _generate_samples(self, model, sampler, mode: str, num_samples: int,
                          use_autocast: bool = False, dtype: torch.dtype = None):
        """Generate samples and measure time."""
        mode_dir = os.path.join(self.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        
        # Full warmup at actual batch size + step count to let cuDNN benchmark
        # select optimal kernels. Without this, first timed run includes kernel selection.
        print(f"Warming up cuDNN (full {self.steps}-step pass at batch_size={self.batch_size})...")
        if mode in ('int8', 'int8_baseline') and HAS_INT8:
            reset_modiff_state_int8(model.model.diffusion_model)
        elif mode in ('int4', 'int4_baseline') and HAS_INT4:
            reset_modiff_state_int4(model.model.diffusion_model)
        if mode in ('int8', 'int8_baseline') and HAS_INT8_LINEAR:
            reset_modiff_state_linear(model.model.diffusion_model)
        elif mode in ('int4', 'int4_baseline') and HAS_INT4_LINEAR:
            reset_modiff_state_int4_linear(model.model.diffusion_model)
        with torch.inference_mode():
            sampler.sample(S=self.steps, batch_size=self.batch_size, shape=self.shape, eta=0.0, verbose=False)
        torch.cuda.synchronize()
        print("Warmup complete.")
        
        total_time = 0.0
        generated = 0
        
        pbar = tqdm(total=num_samples, desc=f"Generating {mode}")
        while generated < num_samples:
            batch = min(self.batch_size, num_samples - generated)
            
            if mode in ('int8', 'int8_baseline') and HAS_INT8:
                reset_modiff_state_int8(model.model.diffusion_model)
            elif mode in ('int4', 'int4_baseline') and HAS_INT4:
                reset_modiff_state_int4(model.model.diffusion_model)
            if mode in ('int8', 'int8_baseline') and HAS_INT8_LINEAR:
                reset_modiff_state_linear(model.model.diffusion_model)
            elif mode in ('int4', 'int4_baseline') and HAS_INT4_LINEAR:
                reset_modiff_state_int4_linear(model.model.diffusion_model)
            # Note: baseline modes have state reset but MoDiff optimizations disabled
            
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
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
    
    def run_mode(self, mode: str, num_samples: int = 16, calibrate: bool = True, force_recalibrate: bool = False):
        """Run benchmark for a specific mode."""
        print(f"\n{'='*60}\n{mode.upper()}\n{'='*60}")
        
        # Determine calibration path if not explicitly provided
        original_calib_path = self.calibration_path
        if not self.calibration_path:
            if mode in ('int8', 'int8_baseline'):
                self.calibration_path = 'integration/int8_calibration.pt'
            elif mode in ('int4', 'int4_baseline'):
                self.calibration_path = 'integration/int4_calibration.pt'

        # If forcing recalibration, ignore existing file during setup
        actual_calib_file = self.calibration_path
        if force_recalibrate and self.calibration_path and os.path.exists(self.calibration_path):
            print(f"Force recalibrate: ignoring existing {self.calibration_path}")
            self.calibration_path = None

        # Reset profiler
        if mode in ['int8', 'int4']:
            profiler.reset()
            print("Profiler reset.")

        model, sampler = self._setup_model(mode)
        
        # Restore calibration path for potential saving if we cleared it for force_recalibrate
        if force_recalibrate:
            self.calibration_path = actual_calib_file

        # INT8/INT4 calibration (skip if static scales already loaded or force_recalibrate is False)
        if mode == 'int8' and HAS_INT8:
            config = get_calibration_config_int8()
            if calibrate and (force_recalibrate or not config.is_calibrated):
                self._calibrate_int8(model, sampler)
        elif mode == 'int4' and HAS_INT4:
            # Check if we already have static scales loaded
            # Note: _setup_model already called apply_int4_static_scales if path existed
            # but if we are forcing recalibration, we want to run _calibrate_int4
            if calibrate and (force_recalibrate or not (self.calibration_path and os.path.exists(self.calibration_path))):
                self._calibrate_int4(model, sampler)
        
        # Reset profiler AFTER calibration so generation-only timing is accurate
        if mode in ['int8', 'int4']:
            profiler.reset()
        
        # Initialize Buffer Pool for zero-copy MoDiff
        if mode in ('int8', 'int4'):
            from integration.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size)
        
        # Configure autocast - enable for all modes except fp32 to maximize bandwidth
        use_autocast = mode != 'fp32'
        dtype = torch.float16 if use_autocast else None
        
        # Generate samples
        try:
            total_time, num_gen = self._generate_samples(
                model, sampler, mode, num_samples, use_autocast, dtype
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            if mode in ['int8', 'int4']:
                print(f"\nProfiler Summary ({mode.upper()}) BEFORE CRASH:")
                profiler.print_summary()
            raise e
        
        # Record results
        time_per_sample = total_time / num_gen
        time_per_step = total_time / (num_gen * self.steps) * 1000
        
        self.results[mode] = {
            'total_time': total_time,
            'num_samples': num_gen,
            'time_per_sample': time_per_sample,
            'time_per_step_ms': time_per_step,
        }
        
        # Restore original path for next mode in 'all' loop
        self.calibration_path = original_calib_path
        
        print(f"\n{mode.upper()} Results:")
        print(f"  Total: {total_time:.2f}s for {num_gen} samples")
        print(f"  Per-sample: {time_per_sample:.3f}s")
        print(f"  Per-step: {time_per_step:.2f}ms")
        
        if 'fp32' in self.results and mode != 'fp32':
            speedup = self.results['fp32']['time_per_sample'] / time_per_sample
            self.results[mode]['speedup'] = speedup
            print(f"  Speedup vs FP32: {speedup:.2f}x")
        
        # Print profiler summary
        if mode in ['int8', 'int4']:
            print(f"\nProfiler Summary ({mode.upper()}):")
            profiler.print_summary()
        
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
        for mode in ['fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline']:
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--mode', type=str, choices=['all', 'fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline'], default='all')
    parser.add_argument('--eval_fid', action='store_true', help='Compute FID between modes')
    parser.add_argument('--skip_calibration', action='store_true')
    parser.add_argument('--force_recalibrate', action='store_true', help='Force regeneration of calibration scales')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Path to static calibration file (e.g., integration/int8_calibration.pt)')
    args = parser.parse_args()
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: {args.config}")
    print(f"Steps: {args.steps} | Batch: {args.batch_size} | Samples: {args.num_samples}")
    if args.calibration:
        print(f"Static calibration: {args.calibration}")
    
    runner = BenchmarkRunner(
        args.config, args.ckpt, args.output_dir,
        args.batch_size, args.steps, shape=(4, 32, 32),
        calibration_path=args.calibration
    )
    
    modes = ['fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline'] if args.mode == 'all' else [args.mode]
    
    for mode in modes:
        if mode == 'int8' and not HAS_INT8:
            print(f"Skipping {mode}: not available")
            continue
        if mode == 'int4' and not HAS_INT4:
            print(f"Skipping {mode}: not available")
            continue
        runner.run_mode(mode, args.num_samples, calibrate=not args.skip_calibration, force_recalibrate=args.force_recalibrate)
    
    if args.eval_fid and len(runner.results) > 1:
        runner.compute_fid()
    
    runner.print_summary()


if __name__ == '__main__':
    main()
