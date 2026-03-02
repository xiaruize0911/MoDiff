#!/usr/bin/env python3
"""
Comprehensive MoDiff Experiment Suite
======================================

Runs all 5 experiments, collects data, generates plots and a full report.

Experiments:
  1. Full pipeline speedup (fp32 / int8 / int4 / int8_baseline / int4_baseline)
  2. Per-component pipeline breakdown (Conv, Linear, SiLU, Attention, GroupNorm, ...)
  3. Per conv-layer-shape speedup analysis (int8 / int4 vs fp32)
  4. Per linear-layer-shape speedup analysis (int8 / int4 vs fp32)
  5. Batch-size ablation study on all of the above
"""

import os
import sys
import json
import time
import warnings
import gc
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings('ignore')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_NAME = torch.cuda.get_device_name(0)
SM_CAP = torch.cuda.get_device_capability()

# ============================================================================
# Utilities
# ============================================================================

def benchmark_fn(fn, warmup=15, iters=80, sync=True):
    """Benchmark with CUDA event timing, returns median ms."""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "max_ms": times[-1],
        "p10_ms": times[int(len(times) * 0.1)],
        "p90_ms": times[int(len(times) * 0.9)],
    }


def pack_int4_tensor(tensor):
    """Pack int8 tensor (values in [-8,7]) to packed int4 (2 per byte)."""
    shape = list(tensor.shape)
    last_dim = shape[-1]
    assert last_dim % 2 == 0
    new_shape = shape[:-1] + [last_dim // 2, 2]
    reshaped = tensor.view(new_shape)
    low = reshaped[..., 0] & 0x0F
    high = (reshaped[..., 1] & 0x0F) << 4
    return (low | high).to(torch.int8)


def load_ldm_model():
    """Load the LSUN Churches LDM model."""
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config_path = 'configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml'
    ckpt_path = 'models/ldm/lsun_churches256/model.ckpt'

    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = pl_sd.get('state_dict', pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    model = model.to(memory_format=torch.channels_last)

    for m in model.modules():
        if hasattr(m, 'use_checkpoint'):
            m.use_checkpoint = False

    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    AttentionBlock.forward = lambda self, x: self._forward(x)

    from integration.fused_resblock import fuse_resblocks_in_module
    fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)

    return model


def cleanup_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# LayerProfiler: hooks to measure time per component type
# ============================================================================

class LayerProfiler:
    """Profile individual layers by type using CUDA-event forward hooks."""

    def __init__(self):
        self.hooks = []
        self.events = {}   # layer_type -> list of (start, end)
        self._active = {}  # module_id -> (layer_type, start_event)
        # Per-instance tracking for per-layer-shape analysis
        self.instance_events = {}  # (layer_type, shape_key) -> list of (start, end)

    def _make_pre_hook(self, layer_type, shape_key=None):
        def hook(module, inp):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._active[id(module)] = (layer_type, ev, shape_key)
        return hook

    def _make_post_hook(self, layer_type, shape_key=None):
        def hook(module, inp, output):
            key = id(module)
            if key in self._active:
                lt, start_ev, sk = self._active.pop(key)
                end_ev = torch.cuda.Event(enable_timing=True)
                end_ev.record()
                self.events.setdefault(lt, []).append((start_ev, end_ev))
                if sk:
                    self.instance_events.setdefault((lt, sk), []).append((start_ev, end_ev))
        return hook

    def register(self, model, track_shapes=False):
        from integration.int8_optimized import OptimizedInt8Conv2d
        from integration.int4_optimized import OptimizedInt4Conv2d
        try:
            from integration.int8_linear import OptimizedInt8Linear
        except ImportError:
            OptimizedInt8Linear = None
        try:
            from integration.int4_linear import OptimizedInt4Linear
        except ImportError:
            OptimizedInt4Linear = None

        for name, module in model.named_modules():
            layer_type = None
            shape_key = None

            if isinstance(module, OptimizedInt8Conv2d):
                layer_type = "Int8Conv2d"
                if track_shapes:
                    shape_key = f"in={module.in_channels},out={module.out_channels},k={module.kernel_size}"
            elif isinstance(module, OptimizedInt4Conv2d):
                layer_type = "Int4Conv2d"
                if track_shapes:
                    shape_key = f"in={module.in_channels},out={module.out_channels},k={module.kernel_size}"
            elif OptimizedInt8Linear and isinstance(module, OptimizedInt8Linear):
                layer_type = "Int8Linear"
                if track_shapes:
                    shape_key = f"in={module.in_features},out={module.out_features}"
            elif OptimizedInt4Linear and isinstance(module, OptimizedInt4Linear):
                layer_type = "Int4Linear"
                if track_shapes:
                    shape_key = f"in={module.in_features},out={module.out_features}"
            elif isinstance(module, nn.GroupNorm):
                layer_type = "GroupNorm"
            elif isinstance(module, nn.SiLU):
                layer_type = "SiLU"
            elif isinstance(module, nn.Conv2d):
                layer_type = "Conv2d(FP32)"
                if track_shapes:
                    shape_key = f"in={module.in_channels},out={module.out_channels},k={module.kernel_size}"
            elif isinstance(module, nn.Linear):
                layer_type = "Linear(FP32)"
                if track_shapes:
                    shape_key = f"in={module.in_features},out={module.out_features}"
            elif 'attention' in name.lower() or 'attn' in type(module).__name__.lower():
                layer_type = "Attention"
            else:
                continue

            h1 = module.register_forward_pre_hook(self._make_pre_hook(layer_type, shape_key))
            h2 = module.register_forward_hook(self._make_post_hook(layer_type, shape_key))
            self.hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def collect(self):
        torch.cuda.synchronize()
        result = {}
        for lt, pairs in self.events.items():
            total_ms = sum(s.elapsed_time(e) for s, e in pairs)
            result[lt] = {
                'total_ms': total_ms,
                'count': len(pairs),
                'avg_ms': total_ms / len(pairs) if pairs else 0,
            }
        return result

    def collect_per_shape(self):
        torch.cuda.synchronize()
        result = {}
        for (lt, sk), pairs in self.instance_events.items():
            total_ms = sum(s.elapsed_time(e) for s, e in pairs)
            result.setdefault(lt, {})[sk] = {
                'total_ms': total_ms,
                'count': len(pairs),
                'avg_ms': total_ms / len(pairs) if pairs else 0,
            }
        return result

    def reset(self):
        self.events.clear()
        self.instance_events.clear()
        self._active.clear()


# ============================================================================
# Experiment 1: Full Pipeline Speedup
# ============================================================================

def _run_calibration_int8(unet, sampler, steps, batch_size, shape, num_cal_runs=3):
    """Calibrate INT8 static activation scales via absmax accumulation.

    Runs ``num_cal_runs`` denoising passes with ``set_calibrating=True`` so every
    OptimizedInt8Conv2d/OptimizedInt8Linear accumulates per-layer absmax statistics.
    Finalising locks those into permanent static scales – on subsequent inference
    steps the expensive per-step absmax reduction is skipped entirely.
    """
    from integration.int8_optimized import (
        reset_calibration,
        set_calibrating,
        get_calibration_config,
        apply_static_scales,
    )
    from integration.int8_linear import (
        set_calibrating_linear,
        export_linear_static_scales,
        apply_linear_static_scales,
    )
    print(f"  Calibrating INT8 ({num_cal_runs} runs × {steps} steps)...")
    reset_calibration()
    set_calibrating(unet, True)
    set_calibrating_linear(unet, True)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for _ in range(num_cal_runs):
            sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    get_calibration_config().finalize()
    set_calibrating(unet, False)
    set_calibrating_linear(unet, False)

    # Collect conv + linear scales into one dict (linear keys prefixed with 'linear:')
    scales = dict(get_calibration_config().scales)
    lin_scales = export_linear_static_scales(unet)
    for k, v in lin_scales.items():
        scales[f'linear:{k}'] = v
    print(f"  Collected {len(scales)} static scales ({len(lin_scales)} linear).")
    return scales


def _apply_int8_scales(unet, scales):
    """Push a pre-computed INT8 scale dict onto a freshly-converted model."""
    from integration.int8_optimized import apply_static_scales, get_calibration_config
    from integration.int8_linear import apply_linear_static_scales
    config = get_calibration_config()
    config.is_calibrated = True
    apply_static_scales(unet, scales)
    lin_scales = {k.replace('linear:', ''): v for k, v in scales.items() if k.startswith('linear:')}
    if lin_scales:
        apply_linear_static_scales(unet, lin_scales)


def _run_calibration_int4(unet, sampler, steps, batch_size, shape, num_cal_runs=3):
    """Calibrate INT4 static activation scales."""
    from integration.int4_optimized import set_calibrating_int4, export_int4_static_scales
    from integration.int4_linear import (
        set_calibrating_int4_linear,
        export_int4_linear_static_scales,
    )
    print(f"  Calibrating INT4 ({num_cal_runs} runs × {steps} steps)...")
    set_calibrating_int4(unet, True)
    set_calibrating_int4_linear(unet, True)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for _ in range(num_cal_runs):
            sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    set_calibrating_int4(unet, False)
    set_calibrating_int4_linear(unet, False)

    scales = export_int4_static_scales(unet)
    lin_scales = export_int4_linear_static_scales(unet)
    for k, v in lin_scales.items():
        scales[f'linear:{k}'] = v
    print(f"  Collected {len(scales)} INT4 static scales ({len(lin_scales)} linear).")
    return scales


def _apply_int4_scales(unet, scales):
    """Push a pre-computed INT4 scale dict onto a freshly-converted model."""
    from integration.int4_optimized import apply_int4_static_scales
    from integration.int4_linear import apply_int4_linear_static_scales
    apply_int4_static_scales(unet, scales)
    lin_scales = {k.replace('linear:', ''): v for k, v in scales.items() if k.startswith('linear:')}
    if lin_scales:
        apply_int4_linear_static_scales(unet, lin_scales)


def experiment_1_pipeline_speedup(steps=50, num_samples=8, batch_size=8):
    """Run benchmark for each mode with forced static calibration.

    Static calibration eliminates the per-step absmax reduction kernel,
    which is the dominant overhead that prevents INT8/INT4 from beating FP32
    in dynamic-quantization mode.

    Calibration strategy
    --------------------
    * INT8/INT8-baseline share calibration (same scale dict, modiff flag differs).
    * INT4/INT4-baseline share calibration similarly.
    * Calibration uses ``CAL_STEPS`` denoising steps × ``CAL_RUNS`` passes —
      enough to collect representative activation statistics quickly.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Full Pipeline Speedup (with forced static calibration)")
    print("=" * 70)

    from ldm.models.diffusion.ddim import DDIMSampler
    shape = (4, 32, 32)
    modes = ['fp32', 'fp16', 'int8_baseline', 'int8', 'int4_baseline', 'int4']
    results = {}

    # Calibration hyper-params: short passes are sufficient to gather absmax stats.
    CAL_STEPS = min(steps, 20)   # max 20 denoising steps per cal run
    CAL_RUNS  = 3                 # 3 passes × CAL_STEPS steps

    # Shared calibration dicts – populated the first time we see an int8/int4 mode
    # and reused for the sibling baseline/MoDiff mode.
    int8_calib_scales: dict | None = None
    int4_calib_scales: dict | None = None

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        model = load_ldm_model()
        unet = model.model.diffusion_model
        sampler = DDIMSampler(model)

        reset_fn = lambda: None

        if mode in ('fp32', 'fp16'):
            pass  # no conversion needed

        elif mode in ('int8', 'int8_baseline'):
            from integration.int8_optimized import (
                convert_model_to_optimized_int8,
                enable_modiff_mode as enable_int8,
                reset_modiff_state as reset_int8,
            )
            from integration.int8_linear import (
                convert_model_to_int8_linear,
                enable_modiff_mode_linear,
                reset_modiff_state_linear,
            )
            from integration.buffer_pool import initialize_buffer_pool

            convert_model_to_optimized_int8(unet)
            convert_model_to_int8_linear(unet)
            initialize_buffer_pool(unet, max_batch_size=batch_size, device='cuda')

            # Calibrate once; reuse cached scales for the paired mode
            if int8_calib_scales is None:
                int8_calib_scales = _run_calibration_int8(unet, sampler, CAL_STEPS, batch_size, shape, CAL_RUNS)
            _apply_int8_scales(unet, int8_calib_scales)

            enable_modiff = (mode == 'int8')
            enable_int8(unet, enable_modiff)
            enable_modiff_mode_linear(unet, enable_modiff)
            reset_fn = lambda: (reset_int8(unet), reset_modiff_state_linear(unet))

        elif mode in ('int4', 'int4_baseline'):
            from integration.int4_optimized import (
                convert_model_to_optimized_int4,
                enable_modiff_mode as enable_int4,
                reset_modiff_state as reset_int4,
            )
            from integration.int4_linear import (
                convert_model_to_int4_linear,
                enable_modiff_mode_int4_linear,
                reset_modiff_state_int4_linear,
            )
            from integration.buffer_pool import initialize_buffer_pool

            convert_model_to_optimized_int4(unet)
            convert_model_to_int4_linear(unet)
            initialize_buffer_pool(unet, max_batch_size=batch_size, device='cuda')

            if int4_calib_scales is None:
                int4_calib_scales = _run_calibration_int4(unet, sampler, CAL_STEPS, batch_size, shape, CAL_RUNS)
            _apply_int4_scales(unet, int4_calib_scales)

            enable_modiff = (mode == 'int4')
            enable_int4(unet, enable_modiff)
            enable_modiff_mode_int4_linear(unet, enable_modiff)
            reset_fn = lambda: (reset_int4(unet), reset_modiff_state_int4_linear(unet))

        # Warmup: full step count so cuDNN benchmark selects optimal kernels.
        # All non-FP32 modes use FP16 autocast so attention/norm also run
        # in FP16 (matches benchmark_ldm.py's use_autocast behaviour).
        use_autocast = (mode != 'fp32')
        reset_fn()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=torch.float16):
            sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
        torch.cuda.synchronize()

        # Timed run
        total_time = 0.0
        generated = 0
        while generated < num_samples:
            bs = min(batch_size, num_samples - generated)
            reset_fn()
            torch.cuda.synchronize()
            start = time.time()
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=torch.float16):
                sampler.sample(S=steps, batch_size=bs, shape=shape, eta=0.0, verbose=False)
            torch.cuda.synchronize()
            total_time += time.time() - start
            generated += bs

        time_per_sample = total_time / num_samples
        time_per_step = time_per_sample / steps * 1000
        results[mode] = {
            'total_time_s': total_time,
            'num_samples': num_samples,
            'time_per_sample': time_per_sample,
            'time_per_step_ms': time_per_step,
            'steps': steps,
            'batch_size': batch_size,
        }
        print(f"  Time/sample: {time_per_sample:.3f}s  |  Time/step: {time_per_step:.2f}ms")

        cleanup_model(model)

    # Compute speedups vs FP32
    fp32_t = results['fp32']['time_per_sample']
    for m, r in results.items():
        r['speedup'] = fp32_t / r['time_per_sample'] if r['time_per_sample'] > 0 else 0

    return results


# ============================================================================
# Experiment 2: Per-component Pipeline Breakdown
# ============================================================================

def experiment_2_breakdown(steps=50, num_batches=2, batch_size=8):
    """Hook-based per-component time breakdown."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Per-Component Pipeline Breakdown")
    print("=" * 70)

    from ldm.models.diffusion.ddim import DDIMSampler
    shape = (4, 32, 32)
    # fp32 = no autocast, int8/int4 = FP16 autocast + calibrated static scales
    modes = ['fp32', 'int8', 'int4']
    results = {}

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        model = load_ldm_model()
        unet = model.model.diffusion_model
        reset_fn = lambda: None

        if mode == 'fp32':
            pass  # plain model, no quantized layers
        elif mode == 'int8':
            from integration.int8_optimized import (
                convert_model_to_optimized_int8,
                enable_modiff_mode as enable_int8,
                reset_modiff_state as reset_int8,
            )
            from integration.int8_linear import (
                convert_model_to_int8_linear,
                enable_modiff_mode_linear,
                reset_modiff_state_linear,
            )
            convert_model_to_optimized_int8(unet)
            convert_model_to_int8_linear(unet)
            from integration.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(unet, max_batch_size=batch_size, device='cuda')
            enable_int8(unet, True)
            enable_modiff_mode_linear(unet, True)
            reset_fn = lambda: (reset_int8(unet), reset_modiff_state_linear(unet))
            sampler = DDIMSampler(model)
            CAL_STEPS = min(steps, 20)
            int8_scales = _run_calibration_int8(unet, sampler, CAL_STEPS, batch_size, shape, num_cal_runs=3)
            _apply_int8_scales(unet, int8_scales)
        elif mode == 'int4':
            from integration.int4_optimized import (
                convert_model_to_optimized_int4,
                enable_modiff_mode as enable_int4,
                reset_modiff_state as reset_int4,
            )
            from integration.int4_linear import (
                convert_model_to_int4_linear,
                enable_modiff_mode_int4_linear,
                reset_modiff_state_int4_linear,
            )
            convert_model_to_optimized_int4(unet)
            convert_model_to_int4_linear(unet)
            from integration.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(unet, max_batch_size=batch_size, device='cuda')
            enable_int4(unet, True)
            enable_modiff_mode_int4_linear(unet, True)
            reset_fn = lambda: (reset_int4(unet), reset_modiff_state_int4_linear(unet))
            sampler = DDIMSampler(model)
            CAL_STEPS = min(steps, 20)
            int4_scales = _run_calibration_int4(unet, sampler, CAL_STEPS, batch_size, shape, num_cal_runs=3)
            _apply_int4_scales(unet, int4_scales)

        if mode == 'fp32':
            sampler = DDIMSampler(model)

        # fp32: no autocast; int8/int4: FP16 autocast (matching exp1 inference conditions)
        use_autocast = (mode != 'fp32')

        # Warmup (no hooks)
        reset_fn()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=torch.float16):
            sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
        torch.cuda.synchronize()

        # Attach hooks
        profiler = LayerProfiler()
        profiler.register(unet, track_shapes=True)

        # Timed runs
        torch.cuda.synchronize()
        wall_start = time.time()
        for _ in range(num_batches):
            reset_fn()
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=torch.float16):
                sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
        torch.cuda.synchronize()
        wall_time = time.time() - wall_start

        layer_stats = profiler.collect()
        per_shape_stats = profiler.collect_per_shape()
        profiler.remove_hooks()

        total_samples = num_batches * batch_size
        results[mode] = {
            'total_time_s': wall_time,
            'time_per_sample': wall_time / total_samples,
            'time_per_step_ms': wall_time / (total_samples * steps) * 1000,
            'layer_stats': layer_stats,
            'per_shape_stats': per_shape_stats,
            'num_batches': num_batches,
            'batch_size': batch_size,
            'steps': steps,
        }

        # Print summary
        print(f"  Wall time: {wall_time:.2f}s  |  Time/step: {results[mode]['time_per_step_ms']:.2f}ms")
        total_ms = sum(v['total_ms'] for v in layer_stats.values())
        print(f"  {'Component':<25} {'Total(ms)':>10} {'Calls':>8} {'Avg(ms)':>10} {'%':>6}")
        for lt, s in sorted(layer_stats.items(), key=lambda x: x[1]['total_ms'], reverse=True):
            pct = s['total_ms'] / total_ms * 100 if total_ms > 0 else 0
            print(f"  {lt:<25} {s['total_ms']:>10.1f} {s['count']:>8} {s['avg_ms']:>10.4f} {pct:>5.1f}%")

        cleanup_model(model)

    return results


# ============================================================================
# Experiment 3: Per Conv-Layer-Shape Speedup (INT8 / INT4 vs FP32 kernel-level)
# ============================================================================

def experiment_3_conv_layer_analysis(batch_sizes=[8]):
    """Benchmark each unique conv shape at FP32, INT8, INT4 level."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Per Conv-Layer-Shape Speedup")
    print("=" * 70)

    if not HAS_CUTLASS:
        print("  CUTLASS not available, skipping.")
        return {}

    # Get spatial dims from model: we need H,W per layer.
    # These depend on the UNet structure. For LSUN Churches (256x256 images,
    # latent 32x32), the spatial dims per channel count are:
    # 192ch -> 32x32, 384ch -> 16x16, 768ch -> 8x8
    # But some skip-connection layers have mismatched in/out channels.
    # We'll enumerate all unique (in_channels, out_channels, kernel_size) and
    # assign spatial dims from the known UNet structure.
    channel_to_spatial = {
        192: 32, 384: 16, 768: 8,
        # For cross-channel convs, use the spatial of the input channel
        4: 32,  # input/output conv
    }

    # Load model to enumerate conv layers
    model = load_ldm_model()
    unet = model.model.diffusion_model

    conv_shapes = {}
    for name, m in unet.named_modules():
        if isinstance(m, nn.Conv2d):
            k = (m.in_channels, m.out_channels, m.kernel_size[0], m.stride[0], m.padding[0])
            if k not in conv_shapes:
                conv_shapes[k] = 0
            conv_shapes[k] += 1

    cleanup_model(model)

    results = {}
    for batch_size in batch_sizes:
        bs_results = {}
        for (C_in, C_out, K, S, P), count in sorted(conv_shapes.items()):
            # Determine spatial size from input channels
            H = W = channel_to_spatial.get(C_in, 16)
            # Skip the 4->192 and 192->4 layers (input/output, tiny)
            if C_in == 4 or C_out == 4:
                continue
            # Some skip connections combine channels (e.g. 1536->768).
            # Use spatial of the larger channel block.
            if C_in > 768:
                H = W = 8
            elif C_in > 384:
                H = W = 8

            shape_key = f"C_in={C_in},C_out={C_out},K={K}x{K},S={S},Count={count}"
            N = batch_size
            empty_bias = torch.empty(0, device='cuda')

            # --- FP32 conv ---
            conv_fp32 = nn.Conv2d(C_in, C_out, K, stride=S, padding=P, bias=False).cuda()
            conv_fp32 = conv_fp32.to(memory_format=torch.channels_last)
            x_fp32 = torch.randn(N, C_in, H, W, device='cuda').to(memory_format=torch.channels_last)
            fp32_res = benchmark_fn(lambda: conv_fp32(x_fp32), warmup=10, iters=50)

            # --- INT8 conv (CUTLASS) ---
            w_int8 = (conv_fp32.weight.data.permute(0, 2, 3, 1).contiguous() * 127).round().clamp(-127, 127).to(torch.int8)
            scale_i8 = torch.tensor([127.0 / max(x_fp32.abs().max().item(), 1e-6)], device='cuda')
            inv_scale_i8 = torch.tensor([1.0 / scale_i8.item()], device='cuda')
            q8 = modiff_cutlass.scale_quantize_int8(x_fp32, scale_i8)

            def int8_fn():
                q = modiff_cutlass.scale_quantize_int8(x_fp32, scale_i8)
                return modiff_cutlass.conv2d_int8_fprop(q, w_int8, inv_scale_i8, empty_bias, S, S, P, P, 1, 1)

            int8_res = benchmark_fn(int8_fn, warmup=10, iters=50)
            int8_kernel = benchmark_fn(
                lambda: modiff_cutlass.conv2d_int8_fprop(q8, w_int8, inv_scale_i8, empty_bias, S, S, P, P, 1, 1),
                warmup=10, iters=50
            )

            # --- INT4 conv (CUTLASS) ---
            if C_in % 2 == 0 and C_out % 2 == 0:
                w_int4 = (conv_fp32.weight.data.permute(0, 2, 3, 1).contiguous() * 7).round().clamp(-7, 7).to(torch.int8)
                w_int4_packed = pack_int4_tensor(w_int4).contiguous()
                scale_i4 = torch.tensor([7.0 / max(x_fp32.abs().max().item(), 1e-6)], device='cuda')
                inv_scale_i4 = torch.tensor([1.0 / scale_i4.item()], device='cuda')
                q4 = modiff_cutlass.scale_quantize_and_pack(x_fp32, scale_i4)

                def int4_fn():
                    q = modiff_cutlass.scale_quantize_and_pack(x_fp32, scale_i4)
                    return modiff_cutlass.conv2d_int4_fprop(q, w_int4_packed, inv_scale_i4, empty_bias, S, S, P, P, 1, 1)

                int4_res = benchmark_fn(int4_fn, warmup=10, iters=50)
                int4_kernel = benchmark_fn(
                    lambda: modiff_cutlass.conv2d_int4_fprop(q4, w_int4_packed, inv_scale_i4, empty_bias, S, S, P, P, 1, 1),
                    warmup=10, iters=50
                )
            else:
                int4_res = None
                int4_kernel = None

            entry = {
                'C_in': C_in, 'C_out': C_out, 'K': K, 'S': S, 'P': P,
                'H': H, 'W': W, 'N': N, 'count': count,
                'fp32_ms': fp32_res['median_ms'],
                'int8_e2e_ms': int8_res['median_ms'],
                'int8_kernel_ms': int8_kernel['median_ms'],
                'int4_e2e_ms': int4_res['median_ms'] if int4_res else None,
                'int4_kernel_ms': int4_kernel['median_ms'] if int4_kernel else None,
                'int8_speedup_vs_fp32': fp32_res['median_ms'] / int8_res['median_ms'] if int8_res['median_ms'] > 0 else 0,
                'int4_speedup_vs_fp32': fp32_res['median_ms'] / int4_res['median_ms'] if (int4_res and int4_res['median_ms'] > 0) else 0,
            }
            bs_results[shape_key] = entry
            i4_str = f"{int4_res['median_ms']:.4f}" if int4_res else "N/A"
            print(f"  {shape_key}: FP32={fp32_res['median_ms']:.4f}ms  INT8={int8_res['median_ms']:.4f}ms  INT4={i4_str}ms")

            del conv_fp32, x_fp32
            torch.cuda.empty_cache()

        results[f"bs={batch_size}"] = bs_results

    return results


# ============================================================================
# Experiment 4: Per Linear-Layer-Shape Speedup (INT8 / INT4 vs FP32)
# ============================================================================

def experiment_4_linear_layer_analysis(batch_sizes=[8]):
    """Benchmark each unique linear shape at FP32, INT8, INT4 level."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Per Linear-Layer-Shape Speedup")
    print("=" * 70)

    # Linear layer shapes from the UNet:
    # in=192,out=768 (1 layer)
    # in=768,out=768 (15 layers)
    # in=768,out=384 (6 layers)
    # in=768,out=1536 (15 layers)
    linear_shapes = [
        (192, 768, 1),
        (768, 768, 15),
        (768, 384, 6),
        (768, 1536, 15),
    ]

    results = {}
    for batch_size in batch_sizes:
        bs_results = {}
        for (in_f, out_f, count) in linear_shapes:
            shape_key = f"in={in_f},out={out_f},count={count}"
            x = torch.randn(batch_size, in_f, device='cuda')

            # --- FP32 ---
            linear_fp32 = nn.Linear(in_f, out_f).cuda()
            fp32_res = benchmark_fn(lambda: linear_fp32(x), warmup=15, iters=80)

            # --- FP16 ---
            linear_fp16 = linear_fp32.half()
            x_fp16 = x.half()
            fp16_res = benchmark_fn(lambda: linear_fp16(x_fp16), warmup=15, iters=80)

            # --- INT8 quantized linear (FP16 GEMM with quant overhead) ---
            from integration.int8_linear import OptimizedInt8Linear, convert_model_to_int8_linear, enable_modiff_mode_linear, reset_modiff_state_linear
            wrapper_int8 = nn.Sequential(nn.Linear(in_f, out_f)).cuda()
            wrapper_int8[0].load_state_dict(linear_fp32.state_dict())
            convert_model_to_int8_linear(wrapper_int8)
            enable_modiff_mode_linear(wrapper_int8, True)

            # First step
            reset_modiff_state_linear(wrapper_int8)
            with torch.no_grad():
                wrapper_int8(x)  # prime first step
            # Measure modulated step (the hot path)
            int8_mod_res = benchmark_fn(lambda: wrapper_int8(x + 0.001 * torch.randn_like(x)), warmup=10, iters=50)

            # Baseline (no caching)
            enable_modiff_mode_linear(wrapper_int8, False)
            int8_base_res = benchmark_fn(lambda: wrapper_int8(x), warmup=15, iters=80)

            # --- INT4 quantized linear ---
            from integration.int4_linear import OptimizedInt4Linear, convert_model_to_int4_linear, enable_modiff_mode_int4_linear, reset_modiff_state_int4_linear
            wrapper_int4 = nn.Sequential(nn.Linear(in_f, out_f)).cuda()
            wrapper_int4[0].load_state_dict(linear_fp32.state_dict())
            convert_model_to_int4_linear(wrapper_int4)
            enable_modiff_mode_int4_linear(wrapper_int4, True)

            reset_modiff_state_int4_linear(wrapper_int4)
            with torch.no_grad():
                wrapper_int4(x)
            int4_mod_res = benchmark_fn(lambda: wrapper_int4(x + 0.001 * torch.randn_like(x)), warmup=10, iters=50)

            enable_modiff_mode_int4_linear(wrapper_int4, False)
            int4_base_res = benchmark_fn(lambda: wrapper_int4(x), warmup=15, iters=80)

            entry = {
                'in_features': in_f, 'out_features': out_f, 'count': count,
                'batch_size': batch_size,
                'fp32_ms': fp32_res['median_ms'],
                'fp16_ms': fp16_res['median_ms'],
                'int8_baseline_ms': int8_base_res['median_ms'],
                'int8_modiff_ms': int8_mod_res['median_ms'],
                'int4_baseline_ms': int4_base_res['median_ms'],
                'int4_modiff_ms': int4_mod_res['median_ms'],
                'int8_base_speedup': fp32_res['median_ms'] / int8_base_res['median_ms'] if int8_base_res['median_ms'] > 0 else 0,
                'int4_base_speedup': fp32_res['median_ms'] / int4_base_res['median_ms'] if int4_base_res['median_ms'] > 0 else 0,
            }
            bs_results[shape_key] = entry
            print(f"  {shape_key} (bs={batch_size}): FP32={fp32_res['median_ms']:.4f}  FP16={fp16_res['median_ms']:.4f}  "
                  f"INT8_base={int8_base_res['median_ms']:.4f}  INT8_mod={int8_mod_res['median_ms']:.4f}  "
                  f"INT4_base={int4_base_res['median_ms']:.4f}  INT4_mod={int4_mod_res['median_ms']:.4f}ms")

            del linear_fp32, linear_fp16, wrapper_int8, wrapper_int4
            torch.cuda.empty_cache()

        results[f"bs={batch_size}"] = bs_results

    return results


# ============================================================================
# Experiment 5: Batch Size Ablation
# ============================================================================

def experiment_5_batch_ablation(batch_sizes=[1, 2, 4, 8, 16], steps=50):
    """Measure full-pipeline time/sample at various batch sizes."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Batch Size Ablation")
    print("=" * 70)

    from ldm.models.diffusion.ddim import DDIMSampler
    shape = (4, 32, 32)
    modes = ['fp32', 'int8', 'int4']
    results = {}

    for mode in modes:
        mode_results = {}
        for batch_size in batch_sizes:
            print(f"\n--- Mode: {mode}, batch_size={batch_size} ---")
            try:
                model = load_ldm_model()
                unet = model.model.diffusion_model
                reset_fn = lambda: None

                if mode == 'fp32':
                    pass
                elif mode == 'int8':
                    from integration.int8_optimized import (
                        convert_model_to_optimized_int8,
                        enable_modiff_mode as enable_int8,
                        reset_modiff_state as reset_int8,
                    )
                    from integration.int8_linear import (
                        convert_model_to_int8_linear,
                        enable_modiff_mode_linear,
                        reset_modiff_state_linear,
                    )
                    convert_model_to_optimized_int8(unet)
                    convert_model_to_int8_linear(unet)
                    from integration.buffer_pool import initialize_buffer_pool
                    initialize_buffer_pool(unet, max_batch_size=batch_size, device='cuda')
                    enable_int8(unet, True)
                    enable_modiff_mode_linear(unet, True)
                    reset_fn = lambda: (reset_int8(unet), reset_modiff_state_linear(unet))
                elif mode == 'int4':
                    from integration.int4_optimized import (
                        convert_model_to_optimized_int4,
                        enable_modiff_mode as enable_int4,
                        reset_modiff_state as reset_int4,
                    )
                    from integration.int4_linear import (
                        convert_model_to_int4_linear,
                        enable_modiff_mode_int4_linear,
                        reset_modiff_state_int4_linear,
                    )
                    convert_model_to_optimized_int4(unet)
                    convert_model_to_int4_linear(unet)
                    from integration.buffer_pool import initialize_buffer_pool
                    initialize_buffer_pool(unet, max_batch_size=batch_size, device='cuda')
                    enable_int4(unet, True)
                    enable_modiff_mode_int4_linear(unet, True)
                    reset_fn = lambda: (reset_int4(unet), reset_modiff_state_int4_linear(unet))

                sampler = DDIMSampler(model)

                # Warmup
                reset_fn()
                with torch.inference_mode():
                    sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
                torch.cuda.synchronize()

                # Two timed batches
                times = []
                for _ in range(2):
                    reset_fn()
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.inference_mode():
                        sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)

                avg_time = sum(times) / len(times)
                tps = avg_time / batch_size
                tpstep = tps / steps * 1000

                mode_results[batch_size] = {
                    'batch_size': batch_size,
                    'total_time_s': avg_time,
                    'time_per_sample': tps,
                    'time_per_step_ms': tpstep,
                    'throughput_samples_per_sec': batch_size / avg_time,
                }
                print(f"  Time/sample: {tps:.3f}s | Time/step: {tpstep:.2f}ms | Throughput: {batch_size/avg_time:.2f} samples/s")

                cleanup_model(model)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  OOM at batch_size={batch_size}, skipping.")
                    torch.cuda.empty_cache()
                    mode_results[batch_size] = None
                else:
                    raise

        results[mode] = mode_results

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    all_data = {
        'metadata': {
            'gpu': GPU_NAME,
            'sm_capability': f"{SM_CAP[0]}.{SM_CAP[1]}",
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }

    print(f"GPU: {GPU_NAME} (SM {SM_CAP[0]}.{SM_CAP[1]})")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 70)

    # --- Experiment 1 ---
    all_data['exp1_pipeline'] = experiment_1_pipeline_speedup(
        steps=200, num_samples=128, batch_size=32
    )

    # --- Experiment 2 ---
    all_data['exp2_breakdown'] = experiment_2_breakdown(
        steps=50, num_batches=2, batch_size=8
    )

    # --- Experiment 3 ---
    all_data['exp3_conv'] = experiment_3_conv_layer_analysis(
        batch_sizes=[8]
    )

    # --- Experiment 4 ---
    all_data['exp4_linear'] = experiment_4_linear_layer_analysis(
        batch_sizes=[8]
    )

    # --- Experiment 5 ---
    all_data['exp5_ablation'] = experiment_5_batch_ablation(
        batch_sizes=[1, 2, 4, 8, 16], steps=50
    )

    # Save all data
    out_path = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nAll experiment data saved to {out_path}")

    return all_data


if __name__ == "__main__":
    main()
