"""
Bottleneck Analysis: Per-component time breakdown and optimization experiments.

This script profiles the UNet at the component level to identify where time is
spent, then tests concrete optimization approaches.

Experiments:
    1. Per-component time breakdown (conv, attention, GN+SiLU, embedding, other)
    2. FP16 vs FP32 cache accumulation overhead
    3. torch.compile on non-CUTLASS ops
    4. Attention optimization (scaled_dot_product vs naive)
    5. Combined best-of approach

Usage:
    python integration/benchmarks/benchmark_bottleneck.py
"""

import os
import sys
import time
import json
import gc
import warnings
import contextlib

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings('ignore')

sys.path.insert(0, os.getcwd())

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf


# ============================================================================
# Experiment 1: Per-component UNet time breakdown
# ============================================================================

class LayerTimingHook:
    """Records per-layer forward time using CUDA events."""

    def __init__(self):
        self.records = defaultdict(list)  # category -> list of (ms)
        self._hooks = []
        self._start_events = {}
        self._end_events = {}

    def _classify(self, module, name):
        """Classify a module into a performance category."""
        from integration.kernels.int8_optimized import OptimizedInt8Conv2d
        from integration.kernels.int4_optimized import OptimizedInt4Conv2d

        if isinstance(module, (OptimizedInt8Conv2d, OptimizedInt4Conv2d)):
            return "int_conv"
        elif isinstance(module, nn.Conv2d):
            return "fp_conv"

        # Check for attention
        cls_name = type(module).__name__
        if 'Attention' in cls_name or 'attention' in name:
            return "attention"

        if isinstance(module, nn.GroupNorm):
            return "groupnorm"

        from integration.fused_ops.fused_resblock import FusedGroupNormSiLU
        if isinstance(module, FusedGroupNormSiLU):
            return "gn_silu_fused"

        if isinstance(module, nn.SiLU):
            return "silu"

        if isinstance(module, nn.Linear):
            return "linear"

        # Time embedding layers
        if 'time_embed' in name or 'temb' in name:
            return "time_embed"

        # Upsample / Downsample
        if 'Upsample' in cls_name or 'Downsample' in cls_name:
            return "resample"

        return None  # skip unclassified

    def attach(self, model, prefix=""):
        """Attach hooks to all leaf modules in the UNet."""
        for name, module in model.named_modules():
            cat = self._classify(module, name)
            if cat is None:
                continue

            uid = f"{cat}::{name}"

            def make_pre_hook(uid_):
                def hook(mod, inp):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    self._start_events[uid_] = ev
                return hook

            def make_post_hook(uid_, cat_):
                def hook(mod, inp, out):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    self._end_events[uid_] = ev
                    # We'll compute timings after sync
                return hook

            h1 = module.register_forward_pre_hook(make_pre_hook(uid))
            h2 = module.register_forward_hook(make_post_hook(uid, cat))
            self._hooks.append((h1, h2, uid, cat))

    def collect(self):
        """After cuda sync, compute elapsed times from recorded events."""
        torch.cuda.synchronize()
        for h1, h2, uid, cat in self._hooks:
            if uid in self._start_events and uid in self._end_events:
                ms = self._start_events[uid].elapsed_time(self._end_events[uid])
                self.records[cat].append(ms)
        self._start_events.clear()
        self._end_events.clear()

    def remove_hooks(self):
        for h1, h2, uid, cat in self._hooks:
            h1.remove()
            h2.remove()
        self._hooks.clear()

    def summarize(self):
        """Return per-category total and breakdown."""
        summary = {}
        for cat, times in self.records.items():
            summary[cat] = {
                'total_ms': sum(times),
                'count': len(times),
                'mean_ms': np.mean(times) if times else 0,
                'max_ms': max(times) if times else 0,
            }
        return summary


# ============================================================================
# Experiment 2: Kernel-level micro-benchmarks for optimization proposals
# ============================================================================

def benchmark_fp16_cache_accumulation(shapes, num_iterations=100):
    """Compare FP32 vs FP16 cache accumulation overhead.

    Tests whether using FP16 for a_hat/o_hat caches can reduce the
    memory bandwidth bottleneck in the MoDiff step1 kernel.
    """
    import modiff_cutlass

    results = {}
    for N, C, H, W in shapes:
        x = torch.randn(N, C, H, W, device='cuda').to(memory_format=torch.channels_last)
        cache_fp32 = torch.randn_like(x)
        cache_fp16 = cache_fp32.half()

        # Simulate step1 path: residual = x - cache, then quantize
        residual_buf = torch.empty_like(x)
        absmax_buf = torch.zeros(1, device='cuda')
        scale_buf = torch.empty(1, device='cuda')
        inv_scale_buf = torch.empty(1, device='cuda')
        retire_count = torch.zeros(1, device='cuda', dtype=torch.int32)
        smooth_inv = torch.empty(0, device='cuda')

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

        # FP32 cache path (current)
        for _ in range(10):
            absmax_buf.zero_()
            retire_count.zero_()
            modiff_cutlass.step1_quantize_fprop(
                x, cache_fp32, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                retire_count, 127.0, smooth_inv)

        for i in range(num_iterations):
            absmax_buf.zero_()
            retire_count.zero_()
            start_events[i].record()
            modiff_cutlass.step1_quantize_fprop(
                x, cache_fp32, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                retire_count, 127.0, smooth_inv)
            end_events[i].record()

        torch.cuda.synchronize()
        fp32_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        # FP16 simulation: compute residual with FP16 cache (manual path)
        for _ in range(10):
            residual_fp16 = x - cache_fp16.float()
            abs_max = residual_fp16.abs().amax()
            s = 127.0 / torch.clamp(abs_max, min=1e-6)
            r_dq = (residual_fp16 * s).round().clamp(-127, 127) / s
            cache_fp16.add_(r_dq.half())
            x_int8 = modiff_cutlass.scale_quantize_int8(
                residual_fp16.contiguous(memory_format=torch.channels_last), s.view(1))

        for i in range(num_iterations):
            start_events[i].record()
            residual_fp16 = x - cache_fp16.float()
            abs_max = residual_fp16.abs().amax()
            s = 127.0 / torch.clamp(abs_max, min=1e-6)
            r_dq = (residual_fp16 * s).round().clamp(-127, 127) / s
            cache_fp16.add_(r_dq.half())
            x_int8 = modiff_cutlass.scale_quantize_int8(
                residual_fp16.contiguous(memory_format=torch.channels_last), s.view(1))
            end_events[i].record()

        torch.cuda.synchronize()
        fp16_manual_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        # Bandwidth analysis
        numel = N * C * H * W
        fp32_cache_bytes = numel * 8  # read + write FP32 cache
        fp16_cache_bytes = numel * 4  # read + write FP16 cache
        bytes_saved = fp32_cache_bytes - fp16_cache_bytes

        results[f"{N}x{C}x{H}x{W}"] = {
            'fp32_cache_step1_ms': fp32_ms,
            'fp16_cache_step1_ms': fp16_manual_ms,
            'speedup': fp32_ms / max(fp16_manual_ms, 1e-9),
            'fp32_cache_bytes_mib': fp32_cache_bytes / (1024*1024),
            'fp16_cache_bytes_mib': fp16_cache_bytes / (1024*1024),
            'bytes_saved_mib': bytes_saved / (1024*1024),
        }

        del x, cache_fp32, cache_fp16, residual_buf
        torch.cuda.empty_cache()

    return results


def benchmark_attention_variants(shapes, num_iterations=100):
    """Compare attention implementations: naive vs SDPA (flash/memory-efficient).

    Tests the potential speedup from using torch.nn.functional.scaled_dot_product_attention
    instead of the manual QKV matmul + softmax + matmul pattern.
    """
    results = {}

    for N, C, spatial in shapes:
        seq_len = spatial * spatial  # H×W flattened
        num_heads = 8
        head_dim = C // num_heads

        q = torch.randn(N * num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        scale = head_dim ** -0.5

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

        # Naive attention: QK^T softmax V
        for _ in range(10):
            attn = torch.bmm(q, k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = torch.bmm(attn, v)

        for i in range(num_iterations):
            start_events[i].record()
            attn = torch.bmm(q, k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = torch.bmm(attn, v)
            end_events[i].record()

        torch.cuda.synchronize()
        naive_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        # SDPA (flash attention / memory-efficient)
        q_4d = q.view(N, num_heads, seq_len, head_dim)
        k_4d = k.view(N, num_heads, seq_len, head_dim)
        v_4d = v.view(N, num_heads, seq_len, head_dim)

        for _ in range(10):
            out_sdpa = F.scaled_dot_product_attention(q_4d, k_4d, v_4d)

        for i in range(num_iterations):
            start_events[i].record()
            out_sdpa = F.scaled_dot_product_attention(q_4d, k_4d, v_4d)
            end_events[i].record()

        torch.cuda.synchronize()
        sdpa_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        # Memory: naive materializes N*H*seq*seq attention matrix
        attn_matrix_bytes = N * num_heads * seq_len * seq_len * 2  # FP16
        attn_matrix_mib = attn_matrix_bytes / (1024*1024)

        results[f"B{N}_C{C}_S{spatial}x{spatial}"] = {
            'seq_len': seq_len,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'naive_ms': naive_ms,
            'sdpa_ms': sdpa_ms,
            'speedup': naive_ms / max(sdpa_ms, 1e-9),
            'attn_matrix_mib': attn_matrix_mib,
            'memory_saved_mib': attn_matrix_mib,  # SDPA doesn't materialize
        }

        del q, k, v, q_4d, k_4d, v_4d
        torch.cuda.empty_cache()

    return results


def benchmark_triton_gn_silu(shapes, num_iterations=100):
    """Compare GN+SiLU implementations: separate PyTorch vs fused Triton.

    Tests the actual benefit of the Triton fused GroupNorm+SiLU kernel.
    """
    from integration.fused_ops.fused_gn_silu import TritonGroupNormSiLU

    results = {}
    num_groups = 32

    for N, C, H, W in shapes:
        x = torch.randn(N, C, H, W, device='cuda').to(memory_format=torch.channels_last)

        # Separate path: nn.GroupNorm + nn.SiLU
        gn = nn.GroupNorm(num_groups, C).cuda()
        silu = nn.SiLU()

        # Fused Triton path
        triton_gn_silu = TritonGroupNormSiLU(gn).cuda()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

        # Separate
        for _ in range(10):
            y = silu(gn(x))

        for i in range(num_iterations):
            start_events[i].record()
            y = silu(gn(x))
            end_events[i].record()

        torch.cuda.synchronize()
        separate_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        # Triton fused
        for _ in range(10):
            y2 = triton_gn_silu(x)

        for i in range(num_iterations):
            start_events[i].record()
            y2 = triton_gn_silu(x)
            end_events[i].record()

        torch.cuda.synchronize()
        triton_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        # F.group_norm + F.silu (compiler-fusible)
        w, b = gn.weight, gn.bias
        for _ in range(10):
            y3 = F.silu(F.group_norm(x, num_groups, w, b, gn.eps))

        for i in range(num_iterations):
            start_events[i].record()
            y3 = F.silu(F.group_norm(x, num_groups, w, b, gn.eps))
            end_events[i].record()

        torch.cuda.synchronize()
        functional_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        results[f"{N}x{C}x{H}x{W}"] = {
            'separate_ms': separate_ms,
            'triton_fused_ms': triton_ms,
            'functional_ms': functional_ms,
            'triton_speedup_vs_separate': separate_ms / max(triton_ms, 1e-9),
            'functional_speedup_vs_separate': separate_ms / max(functional_ms, 1e-9),
        }

        del x, gn, silu, triton_gn_silu
        torch.cuda.empty_cache()

    return results


def benchmark_torch_compile_overhead(shapes, num_iterations=50):
    """Measure the benefit of torch.compile on GroupNorm+SiLU+Conv pipelines."""
    results = {}
    num_groups = 32

    for N, C, H, W in shapes:
        x = torch.randn(N, C, H, W, device='cuda').to(memory_format=torch.channels_last)

        # Build a mini-pipeline: GN → SiLU → Conv → GN → SiLU → Conv
        pipeline = nn.Sequential(
            nn.GroupNorm(num_groups, C),
            nn.SiLU(),
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(num_groups, C),
            nn.SiLU(),
            nn.Conv2d(C, C, 3, padding=1),
        ).cuda().to(memory_format=torch.channels_last)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

        # Eager mode
        for _ in range(10):
            y = pipeline(x)

        for i in range(num_iterations):
            start_events[i].record()
            y = pipeline(x)
            end_events[i].record()

        torch.cuda.synchronize()
        eager_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        # torch.compile
        compiled = torch.compile(pipeline, mode="reduce-overhead")
        # Warm up compilation
        for _ in range(5):
            y = compiled(x)
        torch.cuda.synchronize()

        for i in range(num_iterations):
            start_events[i].record()
            y = compiled(x)
            end_events[i].record()

        torch.cuda.synchronize()
        compiled_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

        results[f"{N}x{C}x{H}x{W}"] = {
            'eager_ms': eager_ms,
            'compiled_ms': compiled_ms,
            'speedup': eager_ms / max(compiled_ms, 1e-9),
        }

        del x, pipeline, compiled
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Experiment 3: Full-model per-component profiling
# ============================================================================

def run_model_component_profiling(config_path, ckpt_path, batch_size=32, num_steps=10):
    """Run the LDM model and measure time per component category.

    Returns a detailed breakdown of where time goes: conv, attention, normalization, etc.
    """
    from integration.fused_ops.fused_resblock import fuse_resblocks_in_module
    from integration.kernels.int8_optimized import (
        convert_model_to_optimized_int8,
        enable_modiff_mode as enable_modiff_mode_int8,
        reset_modiff_state as reset_modiff_state_int8,
    )

    print("Loading model...")
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    model = model.to(memory_format=torch.channels_last)

    for m in model.modules():
        if hasattr(m, 'use_checkpoint'):
            m.use_checkpoint = False

    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    AttentionBlock.forward = lambda self, x: self._forward(x)

    fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)

    # Convert to INT8
    convert_model_to_optimized_int8(model.model.diffusion_model)
    from integration.utils.buffer_pool import initialize_buffer_pool
    initialize_buffer_pool(model.model.diffusion_model, max_batch_size=batch_size, device='cuda')
    enable_modiff_mode_int8(model.model.diffusion_model, True)

    sampler = DDIMSampler(model)
    shape = (4, 32, 32)

    # Step 1: Measure total UNet step time (wall clock)
    print(f"Measuring total per-step time (batch={batch_size}, steps={num_steps})...")
    reset_modiff_state_int8(model.model.diffusion_model)
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        sampler.sample(S=num_steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    torch.cuda.synchronize()

    # Timed run
    reset_modiff_state_int8(model.model.diffusion_model)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        sampler.sample(S=num_steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    torch.cuda.synchronize()
    t1 = time.time()
    total_time_ms = (t1 - t0) * 1000
    per_step_ms = total_time_ms / num_steps
    print(f"  Total time: {total_time_ms:.1f}ms, per-step: {per_step_ms:.2f}ms")

    # Step 2: Hook-based component profiling
    # We do a single step at a time with hooks to attribute time
    print("Running hook-based component profiling...")
    timer = LayerTimingHook()
    timer.attach(model.model.diffusion_model)

    reset_modiff_state_int8(model.model.diffusion_model)
    # Run num_steps individually for hook collection
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        sampler.sample(S=num_steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
        timer.collect()

    summary = timer.summarize()
    timer.remove_hooks()

    result = {
        'total_per_step_ms': per_step_ms,
        'batch_size': batch_size,
        'num_steps': num_steps,
        'components': {}
    }

    total_hooked = 0
    for cat, info in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
        per_step = info['total_ms'] / num_steps
        result['components'][cat] = {
            'total_ms': info['total_ms'],
            'per_step_ms': per_step,
            'count_per_step': info['count'] / num_steps,
            'pct_of_step': 0,  # fill in after
        }
        total_hooked += per_step

    # Compute percentages relative to hooked total
    for cat in result['components']:
        result['components'][cat]['pct_of_step'] = (
            result['components'][cat]['per_step_ms'] / per_step_ms * 100
        )

    result['total_hooked_per_step_ms'] = total_hooked
    result['overhead_per_step_ms'] = per_step_ms - total_hooked
    result['overhead_pct'] = (per_step_ms - total_hooked) / per_step_ms * 100

    # Print summary
    print(f"\n{'Category':<20} {'Per-Step (ms)':<15} {'% of Total':<12} {'Count/Step':<12}")
    print("-" * 60)
    for cat, info in sorted(result['components'].items(), key=lambda x: -x[1]['per_step_ms']):
        print(f"{cat:<20} {info['per_step_ms']:>10.3f}ms   {info['pct_of_step']:>8.1f}%   {info['count_per_step']:>8.0f}")
    print("-" * 60)
    print(f"{'Hooked total':<20} {total_hooked:>10.3f}ms   {total_hooked/per_step_ms*100:>8.1f}%")
    print(f"{'Unhooked overhead':<20} {per_step_ms - total_hooked:>10.3f}ms   {(per_step_ms - total_hooked)/per_step_ms*100:>8.1f}%")
    print(f"{'Wall-clock step':<20} {per_step_ms:>10.3f}ms   {'100.0':>8}%")

    del model, sampler
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = 'integration/results/extended'
    os.makedirs(output_dir, exist_ok=True)

    config_path = 'models/ldm/lsun_churches256/config.yaml'
    ckpt_path = 'models/ldm/lsun_churches256/model.ckpt'
    batch_size = 32
    num_iters = 100

    # UNet layer shapes from model config: channel_mult=[1,2,2,4,4], base=192
    conv_shapes = [
        (batch_size, 192, 32, 32),
        (batch_size, 384, 16, 16),
        (batch_size, 384, 8, 8),
        (batch_size, 768, 4, 4),
        (batch_size, 768, 2, 2),
    ]

    # Attention shapes: (batch, channels, spatial_size)
    # attention_resolutions: [1,2,4,8] → at 32², 16², 8², 4²
    attn_shapes = [
        (batch_size, 192, 32),   # 32×32, seq_len=1024
        (batch_size, 384, 16),   # 16×16, seq_len=256
        (batch_size, 384, 8),    # 8×8, seq_len=64
        (batch_size, 768, 4),    # 4×4, seq_len=16
    ]

    results = {}

    # ── Experiment 1: Full model per-component profiling ──────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Per-component UNet time breakdown")
    print("=" * 70)
    try:
        model_profile = run_model_component_profiling(
            config_path, ckpt_path, batch_size=batch_size, num_steps=20
        )
        results['model_profile'] = model_profile
    except Exception as e:
        print(f"Model profiling failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Experiment 2: Attention optimization ──────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Attention optimization (naive BMM vs SDPA)")
    print("=" * 70)
    try:
        attn_results = benchmark_attention_variants(attn_shapes, num_iterations=num_iters)
        results['attention'] = attn_results
        print(f"\n{'Shape':<25} {'Naive (ms)':<12} {'SDPA (ms)':<12} {'Speedup':<10} {'Attn Mem (MiB)':<15}")
        print("-" * 75)
        for key, val in attn_results.items():
            print(f"{key:<25} {val['naive_ms']:<12.3f} {val['sdpa_ms']:<12.3f} {val['speedup']:<10.2f}x {val['attn_matrix_mib']:<15.1f}")
    except Exception as e:
        print(f"Attention benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Experiment 3: Triton GN+SiLU ─────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: GroupNorm+SiLU — Separate vs Triton vs Functional")
    print("=" * 70)
    try:
        gn_results = benchmark_triton_gn_silu(conv_shapes, num_iterations=num_iters)
        results['gn_silu'] = gn_results
        print(f"\n{'Shape':<25} {'Separate (ms)':<15} {'Triton (ms)':<15} {'F.gn+silu (ms)':<15} {'Triton ↑':<10}")
        print("-" * 80)
        for key, val in gn_results.items():
            print(f"{key:<25} {val['separate_ms']:<15.3f} {val['triton_fused_ms']:<15.3f} {val['functional_ms']:<15.3f} {val['triton_speedup_vs_separate']:<10.2f}x")
    except Exception as e:
        print(f"GN+SiLU benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Experiment 4: FP16 cache accumulation ─────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: FP16 vs FP32 cache accumulation")
    print("=" * 70)
    try:
        cache_results = benchmark_fp16_cache_accumulation(conv_shapes, num_iterations=num_iters)
        results['fp16_cache'] = cache_results
        print(f"\n{'Shape':<25} {'FP32 (ms)':<12} {'FP16 (ms)':<12} {'Speedup':<10} {'Bytes saved (MiB)':<15}")
        print("-" * 75)
        for key, val in cache_results.items():
            print(f"{key:<25} {val['fp32_cache_step1_ms']:<12.3f} {val['fp16_cache_step1_ms']:<12.3f} {val['speedup']:<10.2f}x {val['bytes_saved_mib']:<15.1f}")
    except Exception as e:
        print(f"FP16 cache benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Experiment 5: torch.compile ───────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: torch.compile on GN+SiLU+Conv pipeline")
    print("=" * 70)
    try:
        compile_shapes = conv_shapes[:3]  # skip very small spatial
        compile_results = benchmark_torch_compile_overhead(compile_shapes, num_iterations=50)
        results['torch_compile'] = compile_results
        print(f"\n{'Shape':<25} {'Eager (ms)':<12} {'Compiled (ms)':<15} {'Speedup':<10}")
        print("-" * 62)
        for key, val in compile_results.items():
            print(f"{key:<25} {val['eager_ms']:<12.3f} {val['compiled_ms']:<15.3f} {val['speedup']:<10.2f}x")
    except Exception as e:
        print(f"torch.compile benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Save results ──────────────────────────────────────────────────
    results_path = os.path.join(output_dir, 'bottleneck_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()
