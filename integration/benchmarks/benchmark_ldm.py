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

from integration.utils.profiler import profiler
from integration.utils.quant_memory import format_quant_memory_report, report_quant_memory
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# INT8 imports
try:
    from integration.kernels.int8_optimized import (
        OptimizedInt8Conv2d,
        convert_model_to_optimized_int8,
        apply_static_scales,
        enable_modiff_mode as enable_modiff_mode_int8,
        reset_modiff_state as reset_modiff_state_int8,
        set_calibrating as set_calibrating_int8,
        get_calibration_config as get_calibration_config_int8,
        reset_calibration as reset_calibration_int8,
        set_standard_output_fp16 as set_int8_standard_output_fp16,
    )
    from integration.utils.profiler import profiler  # Added for profiling
    HAS_INT8 = True
except ImportError:
    HAS_INT8 = False
    print("Warning: INT8 not available")

# INT8 Linear imports
try:
    from integration.kernels.int8_linear import (
        OptimizedInt8Linear,
        convert_model_to_int8_linear,
        enable_modiff_mode_linear,
        reset_modiff_state_linear,
        set_calibrating_linear,
        export_linear_static_scales,
        apply_linear_static_scales,
        set_standard_output_fp16_linear as set_int8_linear_standard_output_fp16,
    )
    HAS_INT8_LINEAR = True
except ImportError:
    HAS_INT8_LINEAR = False
    print("Warning: INT8 Linear not available")

# INT4 imports
try:
    from integration.kernels.int4_optimized import (
        OptimizedInt4Conv2d,
        convert_model_to_optimized_int4,
        enable_modiff_mode as enable_modiff_mode_int4,
        reset_modiff_state as reset_modiff_state_int4,
        apply_int4_static_scales,
        export_int4_static_scales,
        set_standard_output_fp16 as set_int4_standard_output_fp16,
    )
    HAS_INT4 = True
except ImportError:
    HAS_INT4 = False
    print("Warning: INT4 not available")

# INT4 Linear imports
try:
    from integration.kernels.int4_linear import (
        OptimizedInt4Linear,
        convert_model_to_int4_linear,
        enable_modiff_mode_int4_linear,
        reset_modiff_state_int4_linear,
        set_calibrating_int4_linear,
        export_int4_linear_static_scales,
        apply_int4_linear_static_scales,
        set_standard_output_fp16_linear as set_int4_linear_standard_output_fp16,
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


def count_conv1d_layers(model: nn.Module) -> int:
    """Count Conv1d-style projection layers."""
    return sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))


class _IdentityAttn(nn.Module):
    """Drop-in replacement for an attention module under the `--no_attention`
    ablation: passes its input through unchanged."""
    def forward(self, x):
        return x


def _force_identity_attention(module: nn.Module) -> int:
    """Recursively replace every AttentionBlock / TokenMajorAttentionBlock /
    QuantizedStandardAttentionBlock instance in `module` with `_IdentityAttn`.

    Patching `AttentionBlock.forward` (as the skip_attention setup does) only
    affects instances that are STILL `AttentionBlock` when forward runs -- but
    the attention-conversion step later in `_setup_model` replaces eligible
    AttentionBlocks with TokenMajorAttentionBlock/QuantizedStandardAttentionBlock
    instances, neither of which is an AttentionBlock subclass, so they never
    consult that patched forward. This runs after all conversion and catches
    whichever of the three types actually ended up installed. Returns the
    number of modules replaced.
    """
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
    from integration.fused_ops.quantized_std_attention import QuantizedStandardAttentionBlock
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (AttentionBlock, TokenMajorAttentionBlock, QuantizedStandardAttentionBlock)):
            setattr(module, name, _IdentityAttn())
            n += 1
        else:
            n += _force_identity_attention(child)
    return n


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


def _reset_wxax_modiff_safe(model):
    """Clear the wxax MoDiff temporal caches. No-op when the wxax path is absent or not in MoDiff
    mode, so it is safe to call unconditionally from every per-sample reset point."""
    try:
        from integration.kernels.wxax_linear import reset_wxax_modiff
        reset_wxax_modiff(model)
    except Exception:
        pass


class BenchmarkRunner:
    """Unified benchmark runner for all precision modes."""
    
    def __init__(self, config_path: str, ckpt_path: str, output_dir: str,
                 batch_size: int = 4, steps: int = 50, shape: tuple = (4, 32, 32),
                 calibration_path: str = None, skip_attention: bool = False,
                 skip_resblock: bool = False, skip_groupnorm: bool = False,
                 linear_backend: str = "fp16", linear_int_gemm_min_m: int = 64):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.steps = steps
        self.shape = shape
        self.calibration_path = calibration_path
        self.skip_attention = skip_attention
        self.skip_resblock = skip_resblock
        self.skip_groupnorm = skip_groupnorm
        self.linear_backend = linear_backend
        self.linear_int_gemm_min_m = linear_int_gemm_min_m
        self.results = {}
        
        os.makedirs(output_dir, exist_ok=True)

    def _decode_and_save_samples(self, model, samples, mode_dir: str, start_index: int,
                                 use_autocast: bool = False, dtype: torch.dtype = None,
                                 decode_chunk_size: int = 8):
        """Decode latent samples in smaller chunks and save from CPU tensors."""
        batch = samples.shape[0]

        for chunk_start in range(0, batch, decode_chunk_size):
            chunk_end = min(chunk_start + decode_chunk_size, batch)
            sample_chunk = samples[chunk_start:chunk_end]

            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
                decoded = model.decode_first_stage(sample_chunk)

            decoded = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0).cpu()

            for local_idx, image in enumerate(decoded):
                image_index = start_index + chunk_start + local_idx
                tvu.save_image(image, os.path.join(mode_dir, f'{image_index:05d}.png'))
    
    def _setup_model(self, mode: str):
        """Load and setup model for given mode."""
        # ---- static_* / dynamic_* prefix (static-vs-dynamic quantization study) ----
        # static  = calibrated constants everywhere (conv/linear static scales + static attention
        #           incl. static-c softmax); dynamic = every activation statistic computed at
        #           runtime (uncalibrated conv/linear absmax + dynamic per-token/per-row attention).
        # Implementation: dynamic simply nulls calibration_path so every apply_*_static_scales block
        # is skipped (-> uncalibrated dynamic kernels); attention static-ness is passed explicitly.
        self._attn_static = False
        _BASEMAP = {"int8": "int8_baseline", "int8_modiff": "int8", "int4": "int4_baseline",
                    "int4_modiff": "int4", "fp16": "fp16"}
        _sv = None
        if mode.startswith("static_"):
            _sv, base = True, mode[len("static_"):]
        elif mode.startswith("dynamic_"):
            _sv, base = False, mode[len("dynamic_"):]
        for _k in ("MODIFF_FP16_MATERIALIZED", "MODIFF_STATIC_SOFTMAX"):
            os.environ.pop(_k, None)                     # clear per-mode (one process runs all modes)
        if _sv is not None:
            self._attn_static = _sv
            mode = _BASEMAP.get(base, base)
            # MODIFF_CONVLIN_STATIC=1 keeps conv/linear calibrated (static) even in dynamic_ modes,
            # so ONLY the attention axis toggles -> a clean single-variable comparison. (conv/linear
            # cannot be cleanly toggled otherwise: their fused fp16 GN->int8 quantize path is
            # intrinsically calibration-gated, so full-dynamic pays fp32-GN + separate quantize.)
            if not _sv and os.environ.get("MODIFF_CONVLIN_STATIC") != "1":
                self.calibration_path = None             # -> conv/linear stay uncalibrated (dynamic)
            if mode == "fp16":
                # The static-vs-dynamic softmax study needs the materialized path (both fp16
                # modes must share one code path and differ only in the softmax), so it is
                # still enabled HERE, inside the static_/dynamic_ branch only.
                os.environ["MODIFF_FP16_MATERIALIZED"] = "1"
                if _sv:
                    os.environ["MODIFF_STATIC_SOFTMAX"] = "1"
        model, _ = load_model(self.config_path, self.ckpt_path)
        
        # Configure backends for maximum speed
        torch.backends.cudnn.benchmark = True
        
        # Use channels_last memory format to eliminate NCHW↔NHWC conversions
        # cuDNN and CUTLASS both prefer NHWC; this avoids costly layout transposes
        model = model.to(memory_format=torch.channels_last)
        
        # Disable gradient checkpointing for inference (eliminates autograd overhead)
        for m in model.modules():
            if hasattr(m, 'use_checkpoint'):
                m.use_checkpoint = False
        # Patch AttentionBlock to bypass checkpoint (it hardcodes flag=True)
        from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock
        if self.skip_attention:
            # Ablation: replace AttentionBlock with identity (skip all attention compute)
            AttentionBlock.forward = lambda self, x: x
            print("  → AttentionBlock skipped (identity pass-through, ablation mode)")
        else:
            AttentionBlock.forward = lambda self, x: self._forward(x)
        # Apply ResBlock fusion for all modes (8-12% speedup)
        # NOTE: fusion must happen BEFORE setting skip_resblock lambda,
        # because FusedResBlock is a different class and ignores ResBlock.forward patches.
        from integration.fused_ops.fused_resblock import fuse_resblocks_in_module, print_fusion_summary
        print("\n" + "="*60)
        print("Applying ResBlock Fusion Optimization")
        print("="*60)
        fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)
        print_fusion_summary(model.model.diffusion_model)

        if self.skip_resblock:
            # Ablation: skip the expensive GroupNorm+SiLU+Conv3x3x2 in each ResBlock.
            # CRITICAL: for updown ResBlocks (spatial downsampling/upsampling), we MUST
            # call x_upd(x) first to preserve correct spatial dimensions — otherwise
            # downstream Attention blocks receive wrong-sized (too large) tensors and
            # run O(n^2) attention on full-resolution features, causing ~6× slowdown.
            # Must patch FusedResBlock AFTER fusion, not ResBlock before fusion.
            from integration.fused_ops.fused_resblock import FusedResBlock

            def _fused_resblock_skip(self, x, emb=None, split=0):
                # Apply spatial downsampling/upsampling first (preserves correct resolution)
                if self.updown:
                    x = self.x_upd(x)
                return self.skip_connection(x)

            def _resblock_skip(self, x, emb, split=0):
                if self.updown:
                    x = self.x_upd(x)
                return self.skip_connection(x) if hasattr(self, 'skip_connection') else x

            FusedResBlock.forward = _fused_resblock_skip
            ResBlock.forward = _resblock_skip
            print("  → FusedResBlock + ResBlock skipped (x_upd + skip_connection only, ablation mode)")
        else:
            ResBlock.forward = lambda self, x, emb, split=0: self._forward(x, emb, split)

        if self.skip_groupnorm:
            # Ablation: replace FusedGroupNormSiLU with identity to measure GroupNorm cost.
            # Only affects FusedResBlock internals (fused_in_norm_silu, fused_out_norm_silu).
            # Conv2d layers still run normally → measures "Conv-only ResBlock" timing.
            # T_GN  = T_full - T_skip_gnorm
            # T_Conv = T_skip_gnorm - T_skip_res
            from integration.fused_ops.fused_resblock import FusedGroupNormSiLU
            FusedGroupNormSiLU.forward = lambda self, x: x
            print("  → FusedGroupNormSiLU skipped (identity, conv-only ResBlock ablation mode)")
        
        if mode == 'int8' and HAS_INT8:
            print(f"Converting UNet to INT8 ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int8(model.model.diffusion_model)
            
            # Also convert linear layers to INT8 with MoDiff
            if HAS_INT8_LINEAR:
                print(f"Converting UNet linear layers to INT8 ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int8_linear(
                    model.model.diffusion_model,
                    backend=self.linear_backend,
                    int_gemm_min_m=self.linear_int_gemm_min_m,
                )
            
            
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
                convert_model_to_int4_linear(
                    model.model.diffusion_model,
                    backend=self.linear_backend,
                    int_gemm_min_m=self.linear_int_gemm_min_m,
                )
            

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
            label = "INT8 Baseline"
            print(f"Converting UNet to {label} (INT8 kernels without MoDiff temporal caching) ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int8(
                model.model.diffusion_model,
                skip_pointwise=True,
            )

            # Also convert linear layers to INT8 (baseline = no temporal caching)
            if HAS_INT8_LINEAR:
                print(f"Converting UNet linear layers to INT8 using {self.linear_backend} backend ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int8_linear(
                    model.model.diffusion_model,
                    backend=self.linear_backend,
                    int_gemm_min_m=self.linear_int_gemm_min_m,
                )

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
            set_int8_standard_output_fp16(model.model.diffusion_model, True)
            if HAS_INT8_LINEAR:
                enable_modiff_mode_linear(model.model.diffusion_model, False)  # Disable temporal caching
                set_int8_linear_standard_output_fp16(model.model.diffusion_model, True)
        elif mode == 'int4_baseline' and HAS_INT4:
            print(f"Converting UNet to INT4 Baseline (INT4 kernels without MoDiff temporal caching) ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int4(model.model.diffusion_model)
            
            # Also convert linear layers to INT4 (baseline = no temporal caching)
            if HAS_INT4_LINEAR:
                print(f"Converting UNet linear layers to INT4 ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int4_linear(
                    model.model.diffusion_model,
                    backend=self.linear_backend,
                    int_gemm_min_m=self.linear_int_gemm_min_m,
                )
            
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
            set_int4_standard_output_fp16(model.model.diffusion_model, True)
            if HAS_INT4_LINEAR:
                enable_modiff_mode_int4_linear(model.model.diffusion_model, False)  # Disable temporal caching
                set_int4_linear_standard_output_fp16(model.model.diffusion_model, True)
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
        elif mode == 'attn_modiff':
            # MoDiff applied to attention blocks (QKV + proj_out Conv1d layers).
            # ResBlocks remain in FP16 (no INT8/INT4 quantization).
            # This isolates the benefit of applying MoDiff temporal delta-caching
            # to the linear projections inside each AttentionBlock.
            from integration.kernels.modiff_attention import convert_attention_to_modiff
            n = convert_attention_to_modiff(model.model.diffusion_model, act_bits=8, verbose=True)
            print(f"  → MoDiff attention applied to {n} AttentionBlocks")
        elif mode == 'int8_attn_modiff' and HAS_INT8:
            # Full MoDiff pipeline: INT8 on ResBlock Conv2d  +  MoDiff on AttentionBlock Conv1d.
            # This extends the original MoDiff scope to the attention linear sub-layers.
            print(f"Converting UNet to INT8 + Attention MoDiff ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int8(model.model.diffusion_model)
            if HAS_INT8_LINEAR:
                print(f"Converting UNet linear layers to INT8 ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int8_linear(
                    model.model.diffusion_model,
                    backend=self.linear_backend,
                    int_gemm_min_m=self.linear_int_gemm_min_m,
                )
            if self.calibration_path and os.path.exists(self.calibration_path):
                print(f"Loading static calibration from {self.calibration_path}")
                scales = torch.load(self.calibration_path, weights_only=True)
                config = get_calibration_config_int8()
                config.scales = scales
                config.is_calibrated = True
                loaded = apply_static_scales(model.model.diffusion_model, scales)
                print(f"✓ Loaded {loaded} INT8 conv layer scales (static quantization enabled)")
                if HAS_INT8_LINEAR:
                    linear_scales = {k: v for k, v in scales.items() if k.startswith('linear:')}
                    if linear_scales:
                        clean_scales = {k.replace('linear:', ''): v for k, v in linear_scales.items()}
                        loaded_lin = apply_linear_static_scales(model.model.diffusion_model, clean_scales)
                        print(f"✓ Loaded {loaded_lin} INT8 linear layer scales")
            enable_modiff_mode_int8(model.model.diffusion_model, True)
            if HAS_INT8_LINEAR:
                enable_modiff_mode_linear(model.model.diffusion_model, True)
            # Extend MoDiff to AttentionBlock Conv1d projections
            from integration.kernels.modiff_attention import convert_attention_to_modiff
            n = convert_attention_to_modiff(model.model.diffusion_model, act_bits=8, verbose=True)
            print(f"  → MoDiff attention applied to {n} AttentionBlocks")
        elif mode == 'int4_attn_modiff' and HAS_INT4:
            # Full MoDiff pipeline: INT4 on ResBlock Conv2d  +  MoDiff on AttentionBlock Conv1d.
            print(f"Converting UNet to INT4 + Attention MoDiff ({count_conv_layers(model.model.diffusion_model)} conv layers)...")
            convert_model_to_optimized_int4(model.model.diffusion_model)
            if HAS_INT4_LINEAR:
                print(f"Converting UNet linear layers to INT4 ({count_linear_layers(model.model.diffusion_model)} linear layers)...")
                convert_model_to_int4_linear(
                    model.model.diffusion_model,
                    backend=self.linear_backend,
                    int_gemm_min_m=self.linear_int_gemm_min_m,
                )
            if self.calibration_path and os.path.exists(self.calibration_path):
                print(f"Loading INT4 calibration from {self.calibration_path}")
                scales = torch.load(self.calibration_path, weights_only=True)
                loaded = apply_int4_static_scales(model.model.diffusion_model, scales)
                print(f"✓ Loaded static scales for {loaded} INT4 conv layers")
                if HAS_INT4_LINEAR:
                    linear_scales = {k: v for k, v in scales.items() if k.startswith('linear:')}
                    if linear_scales:
                        clean_scales = {k.replace('linear:', ''): v for k, v in linear_scales.items()}
                        loaded_lin = apply_int4_linear_static_scales(model.model.diffusion_model, clean_scales)
                        print(f"✓ Loaded {loaded_lin} INT4 linear layer scales")
            enable_modiff_mode_int4(model.model.diffusion_model, True)
            if HAS_INT4_LINEAR:
                enable_modiff_mode_int4_linear(model.model.diffusion_model, True)
            # Extend MoDiff to AttentionBlock Conv1d projections
            from integration.kernels.modiff_attention import convert_attention_to_modiff
            n = convert_attention_to_modiff(model.model.diffusion_model, act_bits=8, verbose=True)
            print(f"  → MoDiff attention applied to {n} AttentionBlocks")

        # Attention: token-major layout on the MATH SDPA backend. For int8/int4 modes,
        # MODIFF_STD_ATTN_BITS (8/4, default = mode) swaps in the materialized quantized-attention
        # block (QKᵀ/AV int GEMMs). MODIFF_QUANT_LINEAR=1 additionally
        # quantizes the qkv/proj Linears (W8A8/
        # W4A4 via AWQ/gemm_wxax); disables fused GN->qkv since qkv becomes a quant Linear.
        quant_lin = os.environ.get("MODIFF_QUANT_LINEAR") == "1"
        if quant_lin:
            os.environ["MODIFF_FUSE_GN_QKV"] = "0"
        _sab = os.environ.get("MODIFF_STD_ATTN_BITS")
        std_attn_bits = int(_sab) if _sab is not None else (4 if "int4" in mode else (8 if "int8" in mode else 0))
        def _to_token_major():
            from integration.fused_ops.token_major_attention import convert_attention_to_token_major
            n = convert_attention_to_token_major(model.model.diffusion_model)
            if n > 0:
                print(f"✓ Converted {n} AttentionBlocks to token-major (MATH SDPA)")
        # Quantized attention (W8A8/W4A4, fused FLASH) is ON by default for int8/int4 modes:
        # eligible blocks (head_dim<=48, T%64==0) use the fused flash kernel (attention QKᵀ+softmax+AV
        # in one kernel, scores in SRAM) — faster (int8 fused ~1.5x vs fp16 e2e) and quality-transparent
        # (+~0.004 rel-L2 over fp16 attention); ineligible blocks fall back to fp16 SDPA. Flash is the
        # sole quantized-attention path (the materialized int-GEMM path was removed). MODIFF_QUANT_ATTN=0
        # reverts attention to fp16 SDPA.
        _force_qattn = os.environ.get("MODIFF_QUANT_ATTN", "1") != "0"
        try:
            if std_attn_bits in (4, 8) and (not quant_lin or _force_qattn):
                try:
                    from integration.fused_ops.quantized_std_attention import convert_attention_to_quantized_std
                    # STATIC by default (calibrated Q/K/V scales + static-c softmax, no runtime reductions
                    # — consistent with the static conv/linear quant); MODIFF_QUANT_ATTN_STATIC=0 -> dynamic.
                    _qattn_static = (getattr(self, "_attn_static", False)
                                     or (_force_qattn and os.environ.get("MODIFF_QUANT_ATTN_STATIC", "1") == "1"))
                    n_tm = convert_attention_to_quantized_std(model.model.diffusion_model, bits=std_attn_bits,
                                                              static=_qattn_static)
                    _sv_tag = "STATIC" if _qattn_static else "dynamic"
                    print(f"✓ Converted {n_tm} AttentionBlocks to QUANTIZED standard attention (W{std_attn_bits}A{std_attn_bits} QKᵀ/AV, {_sv_tag})")
                except Exception as _e:      # module/kernels not ready yet -> standard math attention
                    print(f"  (quantized std-attention unavailable: {_e}); using standard math attention")
                    _to_token_major()
            else:
                _to_token_major()
            if quant_lin:
                from integration.kernels.wxax_linear import (
                    convert_linears_to_wxax, set_wxax_calibrating, finalize_wxax_ascale, reset_wxax_modiff)
                lb = 4 if "int4" in mode else 8
                # MoDiff temporal-delta on the LINEAR activations was disabled because rel-err
                # diverged 0.06 -> 3.2 over DDIM steps. That divergence had a single cause --
                # Bug 2, wxax_linear.py passing the already-quantized codes `q` into _gemm(),
                # which re-quantized them and saturated every nonzero delta to +-127, poisoning
                # o_hat while a_hat stayed correct. Bug 2 was fixed 2026-08-03, so the stated
                # reason for this flag no longer holds and it is now measurable rather than
                # assumed: MODIFF_LINEAR=1 turns it on.
                #
                # It stays OFF by default for a different and still-valid reason: the linear
                # MoDiff path has no GEMM o_hat-accumulate epilogue, so it costs three extra
                # full-tensor PyTorch launches per linear per step. That is a speed argument
                # (Stage 3.3 fixes it), not a correctness one -- do not restore the old comment.
                # Gate on the MODE, not just the flag: MODIFF_LINEAR=1 previously turned Linear
                # MoDiff on in int8_baseline / int4_baseline too, which made the "baseline" rows of
                # an A/B carry MoDiff and moved them by +25 ms/step. A baseline must stay a baseline.
                # The *_attn_modiff modes belong here too: they are MoDiff modes (conv + attention
                # projections), not baselines, so MODIFF_LINEAR=1 must reach their Linears as well.
                # Without them in this tuple, asking for "MoDiff everywhere" silently left the 42
                # wxax Linears un-modulated.
                # DEFAULT FLIPPED TO 1 (2026-08-06), by explicit request, accepting the speed
                # cost. What it buys and what it costs, both measured at batch 128 / DDIM 200:
                # A8 latent relL2 0.0607 -> 0.0508 (0.84x, 3 of 3 paired seeds), and 77.4 -> 103.3
                # ms/step, i.e. the int8 speedup drops from 1.38x fp16 to 1.04x. The cost is larger
                # than "three extra launches per linear": MoDiff sets _out_i8 False, which disables
                # the fused int8-output epilogue on all 21 attention blocks (0/21 qout-eligible).
                # Revisit when the linear MoDiff path gains a GEMM o_hat-accumulate epilogue.
                is_modiff = (os.environ.get("MODIFF_LINEAR", "1") == "1"
                             and mode in ("int8", "int4",
                                          "int8_attn_modiff", "int4_attn_modiff"))
                n_lin = convert_linears_to_wxax(model.model.diffusion_model, bits=lb, modiff=is_modiff)
                print(f"✓ Quantized {n_lin} Linear layers to W{lb}A{lb} (modiff={is_modiff})")
                if n_lin > 0:   # static activation-scale calibration (short sample)
                    set_wxax_calibrating(model, True)
                    cb = min(self.batch_size, 4)
                    cond = self._cond_kwargs(model, cb)
                    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(mode != 'fp32'), dtype=torch.float16):
                        DDIMSampler(model).sample(S=5, batch_size=cb, shape=self.shape,
                                                  eta=0.0, verbose=False, **cond)
                    n_cal = finalize_wxax_ascale(model)
                    reset_wxax_modiff(model)   # clear caches populated during calibration
                    print(f"✓ Calibrated {n_cal} W{lb}A{lb} linear activation scales (static)")
        except Exception as e:
            print(f"  (attention/linear conversion skipped: {e})")

        # skip_attention ablation safety net: the patch at line ~365
        # (`AttentionBlock.forward = lambda self, x: x`) only affects instances that
        # are STILL `AttentionBlock` by the time forward runs. But the conversions
        # above (_to_token_major / convert_attention_to_quantized_std) replace every
        # eligible AttentionBlock with a TokenMajorAttentionBlock/
        # QuantizedStandardAttentionBlock instance -- neither is an AttentionBlock
        # subclass, so they never consult that patched .forward, silently undoing
        # the ablation for every mode where conversion succeeds (including fp16).
        # Force-replace whatever attention module ended up installed with an
        # identity pass-through, unconditionally, as the very last attention-related
        # step.
        if self.skip_attention:
            n_id = _force_identity_attention(model.model.diffusion_model)
            print(f"  → Forced {n_id} attention modules to identity pass-through (ablation mode)")

        # Wire SiLU fusion between each FusedResBlock's GroupNorm and its
        # quantized conv (no-op for modes where in_conv/out_conv are still
        # plain nn.Conv2d, e.g. fp32/fp16).
        from integration.fused_ops.fused_resblock import wire_silu_fusion, convert_upsample_to_fused
        n_silu_wired = wire_silu_fusion(model.model.diffusion_model)
        if n_silu_wired > 0:
            print(f"✓ Wired SiLU fusion for {n_silu_wired} quantized conv layers")

        # Fold each Upsample's F.interpolate(nearest,2x) into its conv's quantize prologue
        # (no-op / harmless fallback for modes where Upsample.conv is still plain nn.Conv2d
        # or modiff-enabled, e.g. fp32/fp16/int8_modiff/int4_modiff -- FusedUpsample checks
        # eligibility per-call and falls back to the original two-step path otherwise).
        n_ups_wired = convert_upsample_to_fused(model.model.diffusion_model)
        if n_ups_wired > 0:
            print(f"✓ Wired Upsample->quantize fusion for {n_ups_wired} Upsample layers")

        # Class-conditional models (e.g. cin256) need cross-attention class
        # conditioning and a model-specific latent shape; derive both here so the
        # unconditional path (churches) is untouched.
        if getattr(model, 'cond_stage_key', None) == 'class_label':
            self.shape = (model.channels, model.image_size, model.image_size)
            print(f"Class-conditional model: latent shape -> {self.shape}, "
                  f"sampling class {getattr(self, 'sample_class', 0)}")

        return model, DDIMSampler(model)

    def _cond_kwargs(self, model, batch):
        """Extra sampler.sample kwargs: class-conditioning for class-conditional
        models (cin256), empty for unconditional (churches)."""
        if getattr(model, 'cond_stage_key', None) == 'class_label':
            cls = int(getattr(self, 'sample_class', 0))
            dev = next(model.parameters()).device
            xc = {model.cond_stage_key: torch.full((batch,), cls, dtype=torch.long, device=dev)}
            with torch.no_grad():
                c = model.get_learned_conditioning(xc)
            return {'conditioning': c}
        return {}
    
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
    
    def _calibrate_int8(self, model, sampler, num_runs: int = 10,
                        calib_steps: int = None, calib_batch: int = None,
                        refine_rounds: int = 1):
        """Calibrate INT8 quantization scales (conv + linear).

        `calib_steps`/`calib_batch` default to the runner's own configuration, i.e. the run this
        calibration is FOR. They used to be hardcoded S=5, batch=2 while production runs 20-200 steps
        at batch 4-128, so the observed activation range was systematically short of the real one:
        measured 2026-08-03 with `effective_code_utilisation` (Q=127 is full scale), the conv
        activations ran at 326 (in_conv) / 631 (out_conv) -- i.e. clipping ~2.6-5x -- and matching
        only the horizon already recovered 246 / 580. This affects the BASELINE as much as MoDiff:
        both read the same static scales.

        Capped, because calibration cost is linear in both: a 200-step production run does not need a
        200-step calibration to see the range plateau, and batch is capped where the extreme-value
        gain flattens. Pass explicit values to override.
        """
        calib_steps = min(int(calib_steps or self.steps or 5), 50)
        calib_batch = min(int(calib_batch or self.batch_size or 2), 8)
        print(f"Calibrating INT8 ({num_runs} runs, S={calib_steps}, batch={calib_batch})...")
        reset_calibration_int8()
        set_calibrating_int8(model.model.diffusion_model, True)
        if HAS_INT8_LINEAR:
            set_calibrating_linear(model.model.diffusion_model, True)
        
        from integration.kernels.modiff_attention import reset_attention_modiff
        with torch.no_grad():
            for _ in range(num_runs):
                reset_modiff_state_int8(model.model.diffusion_model)
                if HAS_INT8_LINEAR:
                    reset_modiff_state_linear(model.model.diffusion_model)
                # No-op unless convert_attention_to_modiff has already run (int8_attn_modiff):
                # without this, attention's temporal cache would carry state across what are
                # meant to be independent calibration samples.
                reset_attention_modiff(model.model.diffusion_model)
                # Same reason, for the wxax path: with MODIFF_LINEAR=1 the 21 attention qkv/proj
                # carry a MoDiff a_hat cache too, and it was previously reset ONLY after
                # calibration -- so it survived into the next sample. Measured consequence of
                # leaving one family unreset: a finite latent on run 1 and an all-NaN latent on
                # run 2. No-op unless convert_linears_to_wxax(modiff=True) has run.
                _reset_wxax_modiff_safe(model)
                sampler.sample(S=calib_steps, batch_size=calib_batch, shape=self.shape,
                               eta=0.0, verbose=False,
                               **self._cond_kwargs(model, calib_batch))

        get_calibration_config_int8().finalize()
        set_calibrating_int8(model.model.diffusion_model, False)
        if HAS_INT8_LINEAR:
            set_calibrating_linear(model.model.diffusion_model, False)

        # REFINEMENT ROUNDS. Round 0 necessarily observes the UNCALIBRATED path (is_calibrated is
        # False while calibrating), whose numerics differ from the calibrated path production
        # actually runs -- in MoDiff mode a_hat/o_hat evolve differently, so the activations differ.
        # Calibrating on one path and running the other left the quantizer 4x short even with the
        # horizon and batch matched: measured effective_code_utilisation 170 (in_conv) / 521
        # (out_conv) against Q=127. Re-observing on the now-calibrated path closes that.
        #
        # SmoothQuant is not re-derived: _fold_weights_with_smooth releases _orig_weight, so
        # _apply_smoothquant is skipped from round 1 on and end_calibration takes its
        # "_act_channel_max is not None" branch -- the per-channel smooth scale from round 0 is kept
        # and only the per-tensor scale is refreshed. That is what we want; refolding the weights
        # every round would move the target.
        for _ in range(max(0, int(refine_rounds))):
            set_calibrating_int8(model.model.diffusion_model, True)
            with torch.no_grad():
                for _ in range(num_runs):
                    reset_modiff_state_int8(model.model.diffusion_model)
                    if HAS_INT8_LINEAR:
                        reset_modiff_state_linear(model.model.diffusion_model)
                    reset_attention_modiff(model.model.diffusion_model)
                    # Same reason, for the wxax path: with MODIFF_LINEAR=1 the 21 attention qkv/proj
                    # carry a MoDiff a_hat cache too, and it was previously reset ONLY after
                    # calibration -- so it survived into the next sample. Measured consequence of
                    # leaving one family unreset: a finite latent on run 1 and an all-NaN latent on
                    # run 2. No-op unless convert_linears_to_wxax(modiff=True) has run.
                    _reset_wxax_modiff_safe(model)
                    sampler.sample(S=calib_steps, batch_size=calib_batch, shape=self.shape,
                                   eta=0.0, verbose=False,
                                   **self._cond_kwargs(model, calib_batch))
            get_calibration_config_int8().finalize()
            set_calibrating_int8(model.model.diffusion_model, False)

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
    
    def _calibrate_int4(self, model, sampler, num_runs: int = 5,
                        calib_steps: int = None, calib_batch: int = None):
        """Calibrate INT4 quantization scales (dynamic to static transition).

        Same horizon/batch fix as _calibrate_int8 -- see there. It matters more at 4 bits: with only
        15 levels there is no headroom to absorb a range the calibration never saw."""
        calib_steps = min(int(calib_steps or self.steps or 5), 50)
        calib_batch = min(int(calib_batch or self.batch_size or 2), 8)
        print(f"Calibrating INT4 ({num_runs} runs, S={calib_steps}, batch={calib_batch})...")
        from integration.kernels.int4_optimized import set_calibrating_int4
        from integration.kernels.modiff_attention import reset_attention_modiff

        set_calibrating_int4(model.model.diffusion_model, True)
        if HAS_INT4_LINEAR:
            set_calibrating_int4_linear(model.model.diffusion_model, True)
        # Attention (int4_attn_modiff) always runs its qkv/proj_out projections
        # through MoDiffConv1dCUTLASS, which is INT8-internal regardless of the
        # base conv precision -- so it calibrates via the INT8 calibration API,
        # not set_calibrating_int4. No-op when no attention layers were converted.
        set_calibrating_int8(model.model.diffusion_model, True)

        with torch.no_grad():
            for _ in range(num_runs):
                reset_modiff_state_int4(model.model.diffusion_model)
                if HAS_INT4_LINEAR:
                    reset_modiff_state_int4_linear(model.model.diffusion_model)
                # No-op unless convert_attention_to_modiff has already run: without this,
                # attention's temporal cache would carry state across what are meant to
                # be independent calibration samples.
                reset_attention_modiff(model.model.diffusion_model)
                # Same reason, for the wxax path: with MODIFF_LINEAR=1 the 21 attention qkv/proj
                # carry a MoDiff a_hat cache too, and it was previously reset ONLY after
                # calibration -- so it survived into the next sample. Measured consequence of
                # leaving one family unreset: a finite latent on run 1 and an all-NaN latent on
                # run 2. No-op unless convert_linears_to_wxax(modiff=True) has run.
                _reset_wxax_modiff_safe(model)
                # Sample a few steps to get representative activations.
                # Small fixed batch (matches _calibrate_int8): calibration only
                # needs representative activation statistics, not production-size
                # batches, so using self.batch_size here was doing up to ~84x more
                # compute/memory than calibration requires.
                sampler.sample(S=calib_steps, batch_size=calib_batch, shape=self.shape,
                               eta=0.0, verbose=False,
                               **self._cond_kwargs(model, calib_batch))

        set_calibrating_int4(model.model.diffusion_model, False)
        if HAS_INT4_LINEAR:
            set_calibrating_int4_linear(model.model.diffusion_model, False)
        # Finalizes attention's static_input_scale/is_calibrated in-process (see comment
        # above); its scale isn't persisted into the INT4 calibration file below, so a
        # fresh live calibration is needed each time this mode is loaded from disk.
        set_calibrating_int8(model.model.diffusion_model, False)
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
        if mode in ('int8', 'int8_baseline', 'int8_attn_modiff') and HAS_INT8:
            reset_modiff_state_int8(model.model.diffusion_model)
        elif mode in ('int4', 'int4_baseline', 'int4_attn_modiff') and HAS_INT4:
            reset_modiff_state_int4(model.model.diffusion_model)
        if mode in ('int8', 'int8_baseline', 'int8_attn_modiff') and HAS_INT8_LINEAR:
            reset_modiff_state_linear(model.model.diffusion_model)
        elif mode in ('int4', 'int4_baseline', 'int4_attn_modiff') and HAS_INT4_LINEAR:
            reset_modiff_state_int4_linear(model.model.diffusion_model)
        if mode in ('attn_modiff', 'int8_attn_modiff', 'int4_attn_modiff'):
            from integration.kernels.modiff_attention import reset_attention_modiff
            reset_attention_modiff(model.model.diffusion_model)
            # Same reason, for the wxax path: with MODIFF_LINEAR=1 the 21 attention qkv/proj
            # carry a MoDiff a_hat cache too, and it was previously reset ONLY after
            # calibration -- so it survived into the next sample. Measured consequence of
            # leaving one family unreset: a finite latent on run 1 and an all-NaN latent on
            # run 2. No-op unless convert_linears_to_wxax(modiff=True) has run.
            _reset_wxax_modiff_safe(model)
        # Match the timed run's autocast: quantized convs emit fp16 regardless,
        # and some models (cin256) feed that into plain fp32 nn.Conv2d skips, which
        # errors without autocast to reconcile the dtypes.
        _use_ac = mode != 'fp32'
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=_use_ac, dtype=torch.float16):
            sampler.sample(S=self.steps, batch_size=self.batch_size, shape=self.shape, eta=0.0, verbose=False,
                           **self._cond_kwargs(model, self.batch_size))
        torch.cuda.synchronize()
        quant_memory_after_warmup = report_quant_memory(model.model.diffusion_model)
        if quant_memory_after_warmup["total_tracked_mib"] > 0:
            print(f"Quant memory after warmup: {format_quant_memory_report(quant_memory_after_warmup)}")
        print("Warmup complete.")
        
        total_time = 0.0
        generated = 0

        while generated < num_samples:
            batch = min(self.batch_size, num_samples - generated)
            
            if mode in ('int8', 'int8_baseline', 'int8_attn_modiff') and HAS_INT8:
                reset_modiff_state_int8(model.model.diffusion_model)
            elif mode in ('int4', 'int4_baseline', 'int4_attn_modiff') and HAS_INT4:
                reset_modiff_state_int4(model.model.diffusion_model)
            if mode in ('int8', 'int8_baseline', 'int8_attn_modiff') and HAS_INT8_LINEAR:
                reset_modiff_state_linear(model.model.diffusion_model)
            elif mode in ('int4', 'int4_baseline', 'int4_attn_modiff') and HAS_INT4_LINEAR:
                reset_modiff_state_int4_linear(model.model.diffusion_model)
            if mode in ('attn_modiff', 'int8_attn_modiff', 'int4_attn_modiff'):
                from integration.kernels.modiff_attention import reset_attention_modiff
                reset_attention_modiff(model.model.diffusion_model)
                # Same reason, for the wxax path: with MODIFF_LINEAR=1 the 21 attention qkv/proj
                # carry a MoDiff a_hat cache too, and it was previously reset ONLY after
                # calibration -- so it survived into the next sample. Measured consequence of
                # leaving one family unreset: a finite latent on run 1 and an all-NaN latent on
                # run 2. No-op unless convert_linears_to_wxax(modiff=True) has run.
                _reset_wxax_modiff_safe(model)
            # Note: baseline modes have state reset but MoDiff optimizations disabled
            
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
                samples, _ = sampler.sample(S=self.steps, batch_size=batch,
                                           shape=self.shape, eta=0.0, verbose=False,
                                           **self._cond_kwargs(model, batch))
            
            torch.cuda.synchronize()
            total_time += time.time() - start
            
            self._decode_and_save_samples(
                model,
                samples,
                mode_dir,
                generated,
                use_autocast=use_autocast,
                dtype=dtype,
            )
            
            generated += batch

        return total_time, generated, quant_memory_after_warmup
    
    def run_mode(self, mode: str, num_samples: int = 16, calibrate: bool = True, force_recalibrate: bool = False):
        """Run benchmark for a specific mode."""
        print(f"\n{'='*60}\n{mode.upper()}\n{'='*60}")
        
        # Determine calibration path if not explicitly provided
        original_calib_path = self.calibration_path
        if not self.calibration_path:
            if mode in ('int8', 'int8_baseline', 'int8_attn_modiff'):
                self.calibration_path = 'integration/calibration/int8_calibration.pt'
            elif mode in ('int4', 'int4_baseline', 'int4_attn_modiff'):
                self.calibration_path = 'integration/calibration/int4_calibration.pt'

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
        # attn_modiff modes are included here too: convert_attention_to_modiff (already run by
        # _setup_model above) adds fresh OptimizedInt8Conv2d instances inside the attention
        # MoDiffConv1dCUTLASS wrappers that are never covered by a conv-only calibration file,
        # so without this they'd stay permanently uncalibrated (dynamic) even in "static" runs.
        if mode in ('int8', 'int8_attn_modiff') and HAS_INT8:
            config = get_calibration_config_int8()
            if calibrate and (force_recalibrate or not config.is_calibrated):
                self._calibrate_int8(model, sampler)
        elif mode in ('int4', 'int4_attn_modiff') and HAS_INT4:
            # Check if we already have static scales loaded
            # Note: _setup_model already called apply_int4_static_scales if path existed
            # but if we are forcing recalibration, we want to run _calibrate_int4
            if calibrate and (force_recalibrate or not (self.calibration_path and os.path.exists(self.calibration_path))):
                self._calibrate_int4(model, sampler)
        
        # Reset profiler AFTER calibration so generation-only timing is accurate
        if mode in ['int8', 'int4']:
            profiler.reset()
        
        # Configure autocast - enable for all modes except fp32 to maximize bandwidth
        use_autocast = mode != 'fp32'
        dtype = torch.float16 if use_autocast else None
        
        # Generate samples
        try:
            total_time, num_gen, quant_memory_after_warmup = self._generate_samples(
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
            'quant_memory_after_warmup': quant_memory_after_warmup,
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
        for mode in ['fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline', 'attn_modiff', 'int8_attn_modiff', 'int4_attn_modiff']:
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
    parser.add_argument('--output_dir', type=str, default='integration/results/ldm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--mode', type=str,
                       choices=['all', 'fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline',
                                'attn_modiff', 'int8_attn_modiff', 'int4_attn_modiff'],
                       default='all')
    parser.add_argument('--eval_fid', action='store_true', help='Compute FID between modes')
    parser.add_argument('--skip_calibration', action='store_true')
    parser.add_argument('--force_recalibrate', action='store_true', help='Force regeneration of calibration scales')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Path to static calibration file (e.g., integration/calibration/int8_calibration.pt)')
    parser.add_argument('--linear_backend', type=str, default='fp16',
                       choices=['fp16', 'int_gemm'],
                       help='Linear layer backend for INT8/INT4 modes. fp16 preserves old behavior; int_gemm uses true INT GEMM when large enough.')
    parser.add_argument('--linear_int_gemm_min_m', type=int, default=64,
                       help='Minimum flattened M dimension required before Linear layers use true INT GEMM.')
    parser.add_argument('--no_attention', action='store_true',
                       help='Skip all AttentionBlocks (identity pass-through). Ablation: conv-only baseline.')
    parser.add_argument('--no_resblock', action='store_true',
                       help='Skip all ResBlocks (identity pass-through). Ablation: attention-only baseline.')
    parser.add_argument('--no_groupnorm', action='store_true',
                       help='Skip GroupNorm+SiLU in FusedResBlock (identity). Ablation: conv-only ResBlock.')
    args = parser.parse_args()
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: {args.config}")
    print(f"Steps: {args.steps} | Batch: {args.batch_size} | Samples: {args.num_samples}")
    print(f"Linear backend: {args.linear_backend} | int_gemm_min_m={args.linear_int_gemm_min_m}")
    if args.calibration:
        print(f"Static calibration: {args.calibration}")
    
    if args.no_attention:
        print("NOTE: --no_attention enabled – all AttentionBlocks replaced with identity.")
        print("      Output images will be lower quality (ablation study only).\n")
    if args.no_resblock:
        print("NOTE: --no_resblock enabled – all ResBlocks replaced with identity.")
        print("      Output images will be lower quality (ablation study only).\n")
    if args.no_groupnorm:
        print("NOTE: --no_groupnorm enabled – FusedGroupNormSiLU replaced with identity.")
        print("      Conv2d still runs. Ablation: conv-only ResBlock cost measurement.\n")

    runner = BenchmarkRunner(
        args.config, args.ckpt, args.output_dir,
        args.batch_size, args.steps, shape=(4, 32, 32),
        calibration_path=args.calibration,
        skip_attention=args.no_attention,
        skip_resblock=args.no_resblock,
        skip_groupnorm=args.no_groupnorm,
        linear_backend=args.linear_backend,
        linear_int_gemm_min_m=args.linear_int_gemm_min_m,
    )
    
    modes = ['fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline'] if args.mode == 'all' else [args.mode]
    # attn_modiff modes are not included in 'all' (they extend the existing modes; run explicitly)
    
    for mode in modes:
        if mode in ('int8', 'int8_baseline', 'int8_attn_modiff') and not HAS_INT8:
            print(f"Skipping {mode}: not available")
            continue
        if mode in ('int4', 'int4_baseline', 'int4_attn_modiff') and not HAS_INT4:
            print(f"Skipping {mode}: not available")
            continue
        runner.run_mode(mode, args.num_samples, calibrate=not args.skip_calibration, force_recalibrate=args.force_recalibrate)
    
    if args.eval_fid and len(runner.results) > 1:
        runner.compute_fid()
    
    runner.print_summary()


if __name__ == '__main__':
    main()
