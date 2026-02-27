#!/usr/bin/env python3
"""
LDM Pipeline Time Breakdown: INT4 vs INT8
==========================================

This script instruments the actual LDM diffusion pipeline to provide
a detailed per-component time breakdown, showing exactly where time
is spent in INT4 vs INT8 mode.

Components profiled:
1. Quantization (dynamic scale computation + quantize + pack)
2. CUTLASS convolution kernel
3. Dequantization + cache accumulation (MoDiff)
4. Scale accumulate (weight_scale * conv_output)
5. Sub_absmax_scale (fused residual computation)
6. Non-quantized overhead (GroupNorm, SiLU, attention, skip connections, etc.)
7. Memory format conversions

Reuses:
- integration/benchmark_ldm.py model loading pipeline
- integration/int8_optimized.py, int4_optimized.py
- integration/profiler.py
"""

import os
import sys
import json
import time
import csv
import warnings

# Set memory management policy
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np

warnings.filterwarnings('ignore', message='Could not initialize NNPACK')
warnings.filterwarnings('ignore', category=UserWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from integration.profiler import profiler, Profiler


# ============================================================================
# Hook-based profiler for non-quantized layers
# ============================================================================

class LayerProfiler:
    """Profile individual layers by type using forward hooks."""
    
    def __init__(self):
        self.hooks = []
        self.events = {}  # layer_type -> list of (start, end)
        self._active_start = {}  # module_id -> start_event
    
    def _make_pre_hook(self, layer_type):
        def hook(module, input):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._active_start[id(module)] = (layer_type, ev)
        return hook
    
    def _make_post_hook(self, layer_type):
        def hook(module, input, output):
            key = id(module)
            if key in self._active_start:
                lt, start_ev = self._active_start.pop(key)
                end_ev = torch.cuda.Event(enable_timing=True)
                end_ev.record()
                if lt not in self.events:
                    self.events[lt] = []
                self.events[lt].append((start_ev, end_ev))
        return hook
    
    def register(self, model):
        """Attach hooks to all layers in the model."""
        from integration.int8_optimized import OptimizedInt8Conv2d
        from integration.int4_optimized import OptimizedInt4Conv2d
        
        for name, module in model.named_modules():
            if isinstance(module, OptimizedInt8Conv2d):
                layer_type = "OptimizedInt8Conv2d"
            elif isinstance(module, OptimizedInt4Conv2d):
                layer_type = "OptimizedInt4Conv2d"
            elif isinstance(module, nn.GroupNorm):
                layer_type = "GroupNorm"
            elif isinstance(module, nn.SiLU):
                layer_type = "SiLU"
            elif isinstance(module, nn.Conv2d):
                layer_type = "Conv2d_other"
            elif 'attention' in name.lower() or 'attn' in type(module).__name__.lower():
                layer_type = "Attention"
            elif isinstance(module, nn.Linear):
                layer_type = "Linear"
            else:
                continue
            
            h1 = module.register_forward_pre_hook(self._make_pre_hook(layer_type))
            h2 = module.register_forward_hook(self._make_post_hook(layer_type))
            self.hooks.extend([h1, h2])
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
    
    def collect(self):
        """Synchronize and compute timings."""
        torch.cuda.synchronize()
        result = {}
        for lt, pairs in self.events.items():
            total_ms = 0.0
            for start_ev, end_ev in pairs:
                total_ms += start_ev.elapsed_time(end_ev)
            result[lt] = {
                'total_ms': total_ms,
                'count': len(pairs),
                'avg_ms': total_ms / len(pairs) if pairs else 0,
            }
        return result
    
    def reset(self):
        self.events.clear()
        self._active_start.clear()


# ============================================================================
# Detailed per-operation profiler using CUDA events
# ============================================================================

class DetailedOpProfiler:
    """Monkey-patches quantized conv layers to time each sub-operation."""
    
    def __init__(self):
        self.records = {}  # op_name -> list of elapsed_ms
        self._patches = []
    
    def patch_int8_layer(self, layer, layer_name):
        """Patch an OptimizedInt8Conv2d to time internal operations."""
        import modiff_cutlass
        original_forward = layer.forward
        records = self.records
        
        def profiled_forward(x):
            # We'll intercept _forward_modulated which is the hot path
            return original_forward(x)
        
        # We instrument at a higher level by using the existing profiler hooks
        pass
    
    def patch_int4_layer(self, layer, layer_name):
        """Patch an OptimizedInt4Conv2d to time internal operations."""
        pass
    
    def collect(self):
        return dict(self.records)


# ============================================================================
# Pipeline Benchmark
# ============================================================================

def run_pipeline_breakdown(mode, steps=50, num_batches=4, batch_size=8):
    """Run a detailed pipeline breakdown for a given mode."""
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    
    config_path = 'configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml'
    ckpt_path = 'models/ldm/lsun_churches256/model.ckpt'
    
    print(f"\n{'='*70}")
    print(f"Pipeline Breakdown: {mode.upper()}")
    print(f"Steps={steps}, Batches={num_batches}, BatchSize={batch_size}")
    print(f"{'='*70}")
    
    # Load model
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    model = model.to(memory_format=torch.channels_last)
    
    # Disable gradient checkpointing
    for m in model.modules():
        if hasattr(m, 'use_checkpoint'):
            m.use_checkpoint = False
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    AttentionBlock.forward = lambda self, x: self._forward(x)
    
    # Apply ResBlock fusion
    from integration.fused_resblock import fuse_resblocks_in_module
    fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)
    
    shape = (4, 32, 32)
    
    if mode == 'int8':
        from integration.int8_optimized import (
            convert_model_to_optimized_int8, enable_modiff_mode as enable_int8,
            reset_modiff_state as reset_int8
        )
        convert_model_to_optimized_int8(model.model.diffusion_model)
        from integration.buffer_pool import initialize_buffer_pool
        initialize_buffer_pool(model.model.diffusion_model, max_batch_size=batch_size, device='cuda')
        
        calib_path = 'integration/int8_calibration.pt'
        if os.path.exists(calib_path):
            from integration.int8_optimized import apply_static_scales, get_calibration_config
            scales = torch.load(calib_path, weights_only=True)
            config = get_calibration_config()
            config.scales = scales
            config.is_calibrated = True
            apply_static_scales(model.model.diffusion_model, scales)
        
        enable_int8(model.model.diffusion_model, True)
        reset_fn = lambda: reset_int8(model.model.diffusion_model)
        
    elif mode == 'int4':
        from integration.int4_optimized import (
            convert_model_to_optimized_int4, enable_modiff_mode as enable_int4,
            reset_modiff_state as reset_int4, apply_int4_static_scales
        )
        convert_model_to_optimized_int4(model.model.diffusion_model)
        from integration.buffer_pool import initialize_buffer_pool
        initialize_buffer_pool(model.model.diffusion_model, max_batch_size=batch_size, device='cuda')
        
        calib_path = 'integration/int4_calibration.pt'
        if os.path.exists(calib_path):
            scales = torch.load(calib_path, weights_only=True)
            apply_int4_static_scales(model.model.diffusion_model, scales)
        
        enable_int4(model.model.diffusion_model, True)
        reset_fn = lambda: reset_int4(model.model.diffusion_model)
        
    elif mode == 'int8_baseline':
        from integration.int8_optimized import (
            convert_model_to_optimized_int8, enable_modiff_mode as enable_int8,
            reset_modiff_state as reset_int8
        )
        convert_model_to_optimized_int8(model.model.diffusion_model)
        enable_int8(model.model.diffusion_model, False)
        reset_fn = lambda: reset_int8(model.model.diffusion_model)
        
    elif mode == 'int4_baseline':
        from integration.int4_optimized import (
            convert_model_to_optimized_int4, enable_modiff_mode as enable_int4,
            reset_modiff_state as reset_int4
        )
        convert_model_to_optimized_int4(model.model.diffusion_model)
        enable_int4(model.model.diffusion_model, False)
        reset_fn = lambda: reset_int4(model.model.diffusion_model)
        
    elif mode == 'fp16':
        # Convert model weights to fp16 so that CUDA kernels run in fp16.
        # Without this, weights stay in fp32 and autocast must cast them on
        # every forward pass, causing significant overhead (~5x slowdown).
        model = model.half()
        reset_fn = lambda: None
    elif mode == 'fp32':
        reset_fn = lambda: None
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    sampler = DDIMSampler(model)
    
    # Determine autocast settings once so warmup and timed runs are consistent
    use_autocast = mode != 'fp32'
    dtype = torch.float16 if use_autocast else None
    
    # Enable detailed profiling via the existing Profiler class
    Profiler.enabled = True
    Profiler.reset()
    
    # Register layer-level hooks
    layer_profiler = LayerProfiler()
    layer_profiler.register(model.model.diffusion_model)
    
    # Warmup — use the same autocast context as timed runs to avoid
    # first-run type-conversion overhead skewing the measurement
    print("Warming up...")
    reset_fn()
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
        sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    torch.cuda.synchronize()
    
    # Reset profilers
    Profiler.reset()
    layer_profiler.reset()
    
    # Timed runs
    print(f"Running {num_batches} batches...")
    
    torch.cuda.synchronize()
    total_start = time.time()
    
    for batch_idx in range(num_batches):
        reset_fn()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
            sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    
    torch.cuda.synchronize()
    total_time = time.time() - total_start
    
    # Collect results
    Profiler.collect()
    layer_results = layer_profiler.collect()
    layer_profiler.remove_hooks()
    
    # Build results dict
    results = {
        'mode': mode,
        'total_time_s': total_time,
        'num_batches': num_batches,
        'batch_size': batch_size,
        'steps': steps,
        'total_samples': num_batches * batch_size,
        'time_per_sample_s': total_time / (num_batches * batch_size),
        'time_per_step_ms': total_time / (num_batches * batch_size * steps) * 1000,
    }
    
    # Internal profiler stats (from profiler.py instrumentation in int4/int8_optimized.py)
    internal_stats = {}
    for name, duration in Profiler._stats.items():
        count = Profiler._counts[name]
        internal_stats[name] = {
            'total_s': duration,
            'count': count,
            'avg_ms': (duration / count * 1000) if count > 0 else 0,
            'pct_of_total': (duration / total_time * 100) if total_time > 0 else 0,
        }
    results['internal_profiler'] = internal_stats
    
    # Layer-level stats
    results['layer_profiler'] = layer_results
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {mode.upper()}")
    print(f"{'='*70}")
    print(f"Total time:      {total_time:.2f}s")
    print(f"Time/sample:     {results['time_per_sample_s']:.3f}s")
    print(f"Time/step:       {results['time_per_step_ms']:.2f}ms")
    
    if internal_stats:
        print(f"\n--- Internal Profiler (from int*_optimized.py) ---")
        print(f"{'Component':<40} | {'Total(s)':<10} | {'Calls':<8} | {'Avg(ms)':<10} | {'%Total':<8}")
        print("-" * 82)
        sorted_stats = sorted(internal_stats.items(), key=lambda x: x[1]['total_s'], reverse=True)
        for name, s in sorted_stats:
            print(f"{name:<40} | {s['total_s']:<10.4f} | {s['count']:<8} | {s['avg_ms']:<10.3f} | {s['pct_of_total']:<8.1f}")
    
    if layer_results:
        print(f"\n--- Layer-Level Timing (hooks) ---")
        print(f"{'Layer Type':<30} | {'Total(ms)':<12} | {'Calls':<8} | {'Avg(ms)':<10}")
        print("-" * 65)
        sorted_layers = sorted(layer_results.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        total_layer_ms = sum(v['total_ms'] for v in layer_results.values())
        for lt, s in sorted_layers:
            pct = s['total_ms'] / total_layer_ms * 100 if total_layer_ms > 0 else 0
            print(f"{lt:<30} | {s['total_ms']:<12.2f} | {s['count']:<8} | {s['avg_ms']:<10.3f} | {pct:.1f}%")
    
    # Disable profiler
    Profiler.enabled = False
    
    # Cleanup
    del model, sampler
    torch.cuda.empty_cache()
    
    return results


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    modes_to_test = ['fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline']
    
    # Use smaller settings for the breakdown (still representative)
    steps = 50
    num_batches = 2
    batch_size = 8
    
    all_results = {}
    
    for mode in modes_to_test:
        try:
            result = run_pipeline_breakdown(mode, steps=steps, num_batches=num_batches, batch_size=batch_size)
            all_results[mode] = result
        except Exception as e:
            print(f"ERROR in {mode}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_path = os.path.join(output_dir, "pipeline_breakdown_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    # Comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE: Time per step (ms)")
    print(f"{'='*80}")
    print(f"{'Mode':<18} | {'Total/step(ms)':<15} | {'Speedup vs FP32':<15}")
    print("-" * 55)
    
    fp32_ms = all_results.get('fp32', {}).get('time_per_step_ms', 1)
    for mode in modes_to_test:
        if mode in all_results:
            ms = all_results[mode]['time_per_step_ms']
            spd = fp32_ms / ms if ms > 0 else 0
            print(f"{mode:<18} | {ms:<15.3f} | {spd:<15.2f}x")
    
    # INT8 vs INT4 detailed comparison
    if 'int8' in all_results and 'int4' in all_results:
        print(f"\n{'='*80}")
        print("INT8 vs INT4: Where does the time go?")
        print(f"{'='*80}")
        
        int8 = all_results['int8']
        int4 = all_results['int4']
        
        print(f"\n{'Component':<40} | {'INT8(ms)':<12} | {'INT4(ms)':<12} | {'Ratio':<8}")
        print("-" * 78)
        
        # Merge all component names
        all_components = set()
        for k in int8.get('layer_profiler', {}):
            all_components.add(k)
        for k in int4.get('layer_profiler', {}):
            all_components.add(k)
        
        for comp in sorted(all_components):
            t8 = int8.get('layer_profiler', {}).get(comp, {}).get('total_ms', 0)
            t4 = int4.get('layer_profiler', {}).get(comp, {}).get('total_ms', 0)
            ratio = t4 / t8 if t8 > 0 else float('inf')
            print(f"{comp:<40} | {t8:<12.2f} | {t4:<12.2f} | {ratio:<8.2f}")


if __name__ == "__main__":
    main()
