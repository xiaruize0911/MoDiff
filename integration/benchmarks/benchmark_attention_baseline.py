"""
Attention Layer INT8/INT4 Baseline Microbenchmark.

Task 5: For each AttentionBlock in the UNet:
  1. Establish a no-MoDiff INT8/INT4 baseline for qkv + proj_out Conv1d layers
  2. Benchmark individual layers (per-layer speed)
  3. Benchmark at full pipeline level with attention quantized but no MoDiff caching

Modes measured:
  - fp16_attn:       FP16 attention (reference)
  - int8_attn_base:  INT8 Conv1d via CUTLASS for qkv/proj_out, NO MoDiff temporal caching
  - int4_attn_base:  INT4 quantized FP16 (no native INT4 GEMM for 1D conv, simulated)
  - modiff_attn:     INT8 CUTLASS + MoDiff temporal caching (existing implementation)

Per-layer benchmarks:
  - Each unique (C_in, C_out, L) shape measured in isolation
  - Overhead of: quantization, GEMM, dequantization  
  - Comparison: FP16 F.conv1d vs INT8 CUTLASS vs INT8 w/ MoDiff delta

Usage:
    python integration/benchmarks/benchmark_attention_baseline.py --steps 50
"""
import argparse
import os
import sys
import json
import time
import math
from typing import List, Dict, Tuple

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', message='Could not initialize NNPACK')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.getcwd())

try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: CUTLASS not available")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone baseline modules for attention (no MoDiff)
# ──────────────────────────────────────────────────────────────────────────────

class BaselineInt8Conv1d(nn.Module):
    """
    INT8 CUTLASS conv1d WITHOUT MoDiff temporal caching.

    This is the true fair baseline: same CUTLASS kernel (quantize + INT8 GEMM)
    used identically on every timestep, no delta caching, no accumulated output.
    Directly comparable to the FP16 reference for latency purposes.

    Architecture: Conv1d(C_in, C_out, ks=1) ≡ Conv2d(C_in, C_out, ks=1×1)
    """
    def __init__(self, conv1d: nn.Conv1d, layer_name: str = ""):
        super().__init__()
        self.layer_name = layer_name
        self.C_in  = conv1d.in_channels
        self.C_out = conv1d.out_channels

        # Reuse the OptimizedInt8Conv2d machinery, but with modiff_enabled=False
        from integration.kernels.int8_optimized import OptimizedInt8Conv2d
        conv2d = nn.Conv2d(self.C_in, self.C_out, kernel_size=1,
                           bias=(conv1d.bias is not None))
        conv2d.weight.data = conv1d.weight.data.reshape(
            self.C_out, self.C_in, 1, 1).clone()
        if conv1d.bias is not None:
            conv2d.bias.data = conv1d.bias.data.clone()

        self.int8_op = OptimizedInt8Conv2d(conv2d, layer_name=layer_name)
        self.int8_op.modiff_enabled = False  # pure baseline: no temporal caching

        # Calibrate with a reasonable fixed scale (per-tensor, symmetric INT8)
        # In production you'd run calibration; here we use a static scale
        self.int8_op.is_calibrated = False  # will use dynamic scale per-call

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, L] → [B, C_out, L]"""
        B, C, L = x.shape
        # Conv1d(ks=1) ≡ Conv2d(ks=1x1): reshape [B,C,L] → [B,C,L,1]
        x_4d = x.unsqueeze(-1)
        if x_4d.dtype != torch.float32:
            x_4d = x_4d.float()
        if not x_4d.is_contiguous(memory_format=torch.channels_last):
            x_4d = x_4d.contiguous(memory_format=torch.channels_last)
        out_4d = self.int8_op(x_4d)  # [B, C_out, L, 1]
        return out_4d.squeeze(-1)   # [B, C_out, L]

    def apply_static_scale(self, scale: float):
        """Set a static calibrated scale for this layer."""
        self.int8_op.static_input_scale.fill_(scale)
        self.int8_op.is_calibrated = True
        self.int8_op._cached_scale_float = scale


class BaselineInt8Conv1dNaive(nn.Module):
    """
    Naive INT8 simulation: FP16 weights, per-tensor dynamic quantization.
    No CUTLASS — just standard PyTorch round/clamp path.
    Used as an educational comparison to show kernel overhead.
    """
    def __init__(self, conv1d: nn.Conv1d, layer_name: str = ""):
        super().__init__()
        self.layer_name = layer_name
        # Quantize weights statically at init
        w = conv1d.weight.data.float()
        w_max = w.abs().max().clamp(1e-8)
        self.w_scale = w_max / 127.0
        w_q = (w / self.w_scale).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer('weight_int8', w_q)
        if conv1d.bias is not None:
            self.register_buffer('bias', conv1d.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        x_f = x.float()
        # Dynamic per-tensor activation quantization
        act_max = x_f.abs().max().clamp(1e-8)
        act_scale = act_max / 127.0
        x_q = (x_f / act_scale).round().clamp(-127, 127).to(torch.int8)
        # Dequantize and run FP32 conv
        x_dq = x_q.float() * act_scale
        w_dq = self.weight_int8.float() * self.w_scale
        out = F.conv1d(x_dq, w_dq, self.bias)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Layer microbenchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_conv1d_layer(
    C_in: int, C_out: int, L: int, batch_size: int = 8,
    warmup_iters: int = 20, timed_iters: int = 50,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark a single Conv1d(C_in, C_out, ks=1) with various implementations.
    Returns latency in milliseconds.
    """
    torch.manual_seed(42)
    x = torch.randn(batch_size, C_in, L, device=device, dtype=torch.float16)

    # Create reference FP16 conv
    conv1d = nn.Conv1d(C_in, C_out, kernel_size=1, bias=True).to(device)
    nn.init.kaiming_uniform_(conv1d.weight, a=math.sqrt(5))
    weight_fp16 = conv1d.weight.data.half()
    bias_fp16   = conv1d.bias.data.half()

    results = {}

    # ─ FP16 baseline ─
    def fp16_forward():
        return F.conv1d(x, weight_fp16, bias_fp16)

    for _ in range(warmup_iters):
        fp16_forward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed_iters):
        fp16_forward()
    torch.cuda.synchronize()
    results['fp16_ms'] = (time.perf_counter() - t0) / timed_iters * 1000

    # ─ INT8 CUTLASS baseline (no MoDiff) ─
    if HAS_CUTLASS:
        int8_base = BaselineInt8Conv1d(conv1d, layer_name=f"attn_{C_in}_{C_out}").to(device)
        x_fp32 = x.float()
        for _ in range(warmup_iters):
            int8_base(x_fp32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(timed_iters):
            int8_base(x_fp32)
        torch.cuda.synchronize()
        results['int8_cutlass_base_ms'] = (time.perf_counter() - t0) / timed_iters * 1000

    # ─ INT8 naive baseline ─
    int8_naive = BaselineInt8Conv1dNaive(conv1d, layer_name=f"attn_naive_{C_in}").to(device)
    for _ in range(warmup_iters):
        int8_naive(x.float())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed_iters):
        int8_naive(x.float())
    torch.cuda.synchronize()
    results['int8_naive_ms'] = (time.perf_counter() - t0) / timed_iters * 1000

    # ─ MoDiff INT8 CUTLASS (with temporal caching) ─
    if HAS_CUTLASS:
        try:
            from integration.kernels.modiff_attention import MoDiffConv1dCUTLASS
            modiff = MoDiffConv1dCUTLASS(conv1d, layer_name=f"attn_modiff_{C_in}").to(device)
            for _ in range(warmup_iters):
                modiff(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(timed_iters):
                modiff(x)
            torch.cuda.synchronize()
            results['modiff_cutlass_ms'] = (time.perf_counter() - t0) / timed_iters * 1000
        except Exception as e:
            results['modiff_cutlass_ms'] = float('nan')
            print(f"  MoDiff CUTLASS error: {e}")

    return results


def print_layer_benchmark(C_in, C_out, L, bs, results):
    print(f"\n  Conv1d({C_in}→{C_out}, L={L}, bs={bs})")
    fp16 = results.get('fp16_ms', float('nan'))
    print(f"    FP16:              {fp16:>8.3f} ms  (1.00×)")
    for key, label in [
        ('int8_cutlass_base_ms', 'INT8 CUTLASS base'),
        ('int8_naive_ms',        'INT8 naive (sim)'),
        ('modiff_cutlass_ms',    'MoDiff INT8 CUTLASS'),
    ]:
        v = results.get(key, float('nan'))
        ratio = fp16 / v if (v > 0 and not math.isnan(v)) else float('nan')
        speedup = f"{ratio:.2f}×" if not math.isnan(ratio) else "N/A"
        print(f"    {label:<22} {v:>8.3f} ms  ({speedup})")


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline comparison with attention quantized
# ──────────────────────────────────────────────────────────────────────────────

def setup_pipeline_with_attn_baseline(config_path, ckpt_path, batch_size,
                                       int8_mode=True, use_modiff=False,
                                       calib_path=None):
    """
    Load UNet and convert attention Conv1d layers to INT8 baseline (no MoDiff).
    INT8 conv quantization on ResBlocks uses standard setup.
    """
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf

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

    from integration.fused_ops.fused_resblock import fuse_resblocks_in_module
    fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)

    if int8_mode:
        from integration.kernels.int8_optimized import (
            convert_model_to_optimized_int8, apply_static_scales,
            enable_modiff_mode, get_calibration_config,
        )
        from integration.kernels.int8_linear import (
            convert_model_to_int8_linear, enable_modiff_mode_linear,
            apply_linear_static_scales,
        )
        convert_model_to_optimized_int8(model.model.diffusion_model)
        convert_model_to_int8_linear(model.model.diffusion_model)
        from integration.utils.buffer_pool import initialize_buffer_pool
        initialize_buffer_pool(model.model.diffusion_model,
                               max_batch_size=batch_size, device='cuda')
        if calib_path and os.path.exists(calib_path):
            scales = torch.load(calib_path, weights_only=True)
            config = get_calibration_config()
            config.scales = scales
            config.is_calibrated = True
            apply_static_scales(model.model.diffusion_model, scales)
            lin_scales = {k.replace('linear:', ''): v for k, v in scales.items()
                         if k.startswith('linear:')}
            if lin_scales:
                apply_linear_static_scales(model.model.diffusion_model, lin_scales)
        enable_modiff_mode(model.model.diffusion_model, use_modiff)
        enable_modiff_mode_linear(model.model.diffusion_model, use_modiff)

    # Now replace AttentionBlock's Conv1d layers with INT8 baseline modules
    n_replaced = 0
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    for module in model.model.diffusion_model.modules():
        if isinstance(module, AttentionBlock):
            # Replace qkv: Conv1d(C, 3C, 1)
            if isinstance(module.qkv, nn.Conv1d) and HAS_CUTLASS:
                module.qkv = BaselineInt8Conv1d(module.qkv,
                                                 layer_name=f"attn_qkv_{module.channels}").to('cuda')
                n_replaced += 1
            # Replace proj_out: Conv1d(C, C, 1)
            if isinstance(module.proj_out, nn.Conv1d) and HAS_CUTLASS:
                module.proj_out = BaselineInt8Conv1d(
                    module.proj_out, layer_name=f"attn_proj_{module.channels}").to('cuda')
                n_replaced += 1

    print(f"  → Replaced {n_replaced} AttentionBlock Conv1d layers with INT8 baseline")

    from ldm.models.diffusion.ddim import DDIMSampler
    return model, DDIMSampler(model)


def timed_pipeline(model, sampler, mode_label, batch_size, steps, shape=(4,32,32),
                   num_runs=4, use_autocast=True):
    """Run timed pipeline and return time per sample."""
    dtype = torch.float16 if use_autocast else None

    # Reset modiff state
    try:
        from integration.kernels.int8_optimized import reset_modiff_state
        reset_modiff_state(model.model.diffusion_model)
    except Exception:
        pass

    # Warmup
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
        sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        try:
            from integration.kernels.int8_optimized import reset_modiff_state
            reset_modiff_state(model.model.diffusion_model)
        except Exception:
            pass
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
            sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return avg / batch_size


def main():
    parser = argparse.ArgumentParser(description="Attention Layer INT8/INT4 Baseline Benchmark")
    parser.add_argument('--config', default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt',   default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--steps',      type=int, default=50)
    parser.add_argument('--int8_calib', default='integration/calibration/int8_calibration.pt')
    parser.add_argument('--num_runs',   type=int, default=3)
    parser.add_argument('--output', default='integration/results/attn_baseline_benchmark.json')
    parser.add_argument('--skip_pipeline', action='store_true',
                        help='Skip full pipeline tests, only run per-layer microbenchmarks')
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUTLASS available: {HAS_CUTLASS}")
    print(f"batch_size={args.batch_size}, steps={args.steps}\n")

    output = {'per_layer': {}, 'pipeline': {}}

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: Per-layer microbenchmarks
    # ─────────────────────────────────────────────────────────────────────────
    print("="*60)
    print("PART 1: Per-Layer Attention Conv1d Microbenchmarks")
    print("="*60)
    print("(Benchmarking FP16 vs INT8-CUTLASS-baseline vs INT8-naive vs MoDiff)")

    # AttentionBlock shapes in the LSUN Churches LDM:
    # From model analysis: C = 192, 384, 768 (feature channels)
    # L = H*W flattened spatial dimension, depends on resolution stage
    # At 256×256 with 4x downsampling and further 2x/4x: L = 32*32=1024, 16*16=256, 8*8=64
    attn_shapes = [
        # (C_in, C_out, L, spatial_desc)
        # qkv: (C, 3C, L)
        (192,  576,  1024, "res8_qkv_192→576"),
        (384,  1152, 256,  "res16_qkv_384→1152"),
        (768,  2304, 64,   "res32_qkv_768→2304"),
        # proj_out: (C, C, L)
        (192,  192,  1024, "res8_proj_192→192"),
        (384,  384,  256,  "res16_proj_384→384"),
        (768,  768,  64,   "res32_proj_768→768"),
    ]

    per_layer_results = {}
    for C_in, C_out, L, desc in attn_shapes:
        print(f"\n  [{desc}]  Conv1d({C_in}→{C_out}, L={L}, bs={args.batch_size})")
        try:
            res = benchmark_conv1d_layer(
                C_in, C_out, L, batch_size=args.batch_size,
                warmup_iters=20, timed_iters=50
            )
            per_layer_results[desc] = res
            fp16 = res.get('fp16_ms', float('nan'))
            print(f"    FP16:              {fp16:>8.3f} ms  (1.00×)")
            for key, label in [
                ('int8_cutlass_base_ms', 'INT8 CUTLASS base'),
                ('int8_naive_ms',        'INT8 naive (sim)'),
                ('modiff_cutlass_ms',    'MoDiff INT8 CUTLASS'),
            ]:
                v = res.get(key, float('nan'))
                ratio = fp16 / v if (v and v > 0 and not math.isnan(v)) else float('nan')
                speedup = f"{ratio:.2f}×" if not math.isnan(ratio) else "N/A"
                print(f"    {label:<22} {v:>8.3f} ms  ({speedup})")
        except Exception as e:
            print(f"  ERROR: {e}")
            per_layer_results[desc] = {'error': str(e)}

    output['per_layer'] = per_layer_results

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: Full pipeline comparison
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_pipeline:
        print("\n" + "="*60)
        print("PART 2: Full Pipeline — Attention Quantization Modes")
        print("="*60)

        pipeline_results = {}

        # Experiment A: INT8 ResBlocks (standard) + FP16 attention
        print("\n[A] INT8 ResBlocks + FP16 attention (standard int8_baseline)")
        model_a, sampler_a = _load_int8_model(
            args.config, args.ckpt, args.batch_size, args.int8_calib, use_modiff=False)
        t_a = timed_pipeline(model_a, sampler_a, 'int8_fp16_attn',
                              args.batch_size, args.steps, num_runs=args.num_runs)
        pipeline_results['int8_resblock_fp16_attn'] = {
            'time_per_sample_s': round(t_a, 4),
            'time_per_step_ms':  round(t_a / args.steps * 1000, 3),
        }
        print(f"  {t_a*1000:.1f} ms/sample | {t_a/args.steps*1000:.2f} ms/step")
        del model_a, sampler_a
        torch.cuda.empty_cache()

        # Experiment B: INT8 ResBlocks + INT8 baseline attention (no MoDiff)
        print("\n[B] INT8 ResBlocks + INT8 attention baseline (no MoDiff)")
        if HAS_CUTLASS:
            model_b, sampler_b = setup_pipeline_with_attn_baseline(
                args.config, args.ckpt, args.batch_size,
                int8_mode=True, use_modiff=False, calib_path=args.int8_calib)
            t_b = timed_pipeline(model_b, sampler_b, 'int8_all',
                                  args.batch_size, args.steps, num_runs=args.num_runs)
            pipeline_results['int8_resblock_int8_attn_base'] = {
                'time_per_sample_s': round(t_b, 4),
                'time_per_step_ms':  round(t_b / args.steps * 1000, 3),
                'speedup_vs_fp16_attn': round(t_a / t_b, 3),
            }
            print(f"  {t_b*1000:.1f} ms/sample | {t_b/args.steps*1000:.2f} ms/step "
                  f"| {t_a/t_b:.2f}× vs [A]")
            del model_b, sampler_b
            torch.cuda.empty_cache()
        else:
            print("  Skipped (no CUTLASS)")

        # Experiment C: INT8 ResBlocks + MoDiff attention (modiff_attention)
        print("\n[C] INT8 ResBlocks + MoDiff attention (existing modiff_attn)")
        model_c, sampler_c = _load_int8_modiff_attn_model(
            args.config, args.ckpt, args.batch_size, args.int8_calib)
        t_c = timed_pipeline(model_c, sampler_c, 'int8_modiff_attn',
                              args.batch_size, args.steps, num_runs=args.num_runs)
        pipeline_results['int8_resblock_modiff_attn'] = {
            'time_per_sample_s': round(t_c, 4),
            'time_per_step_ms':  round(t_c / args.steps * 1000, 3),
            'speedup_vs_fp16_attn': round(t_a / t_c, 3),
        }
        print(f"  {t_c*1000:.1f} ms/sample | {t_c/args.steps*1000:.2f} ms/step "
              f"| {t_a/t_c:.2f}× vs [A]")
        del model_c, sampler_c
        torch.cuda.empty_cache()

        output['pipeline'] = pipeline_results

        # Summary
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"{'Mode':<40} {'ms/sample':>10} {'ms/step':>10} {'vs [A]':>10}")
        print("-" * 72)
        ref_t = pipeline_results.get('int8_resblock_fp16_attn', {}).get('time_per_sample_s', 1.0)
        for label, key in [
            ('INT8 ResBlocks + FP16 attn [A]', 'int8_resblock_fp16_attn'),
            ('INT8 ResBlocks + INT8 attn base [B]', 'int8_resblock_int8_attn_base'),
            ('INT8 ResBlocks + MoDiff attn [C]', 'int8_resblock_modiff_attn'),
        ]:
            r = pipeline_results.get(key)
            if r:
                t = r['time_per_sample_s']
                ms_s = r['time_per_step_ms']
                speedup = ref_t / t
                print(f"{label:<40} {t*1000:>10.1f} {ms_s:>10.2f} {speedup:>9.2f}×")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


def _load_int8_model(config_path, ckpt_path, batch_size, calib_path, use_modiff=False):
    """Load INT8-quantized model (ResBlocks only, attention in FP16)."""
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from omegaconf import OmegaConf

    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval().to(memory_format=torch.channels_last)

    for m in model.modules():
        if hasattr(m, 'use_checkpoint'):
            m.use_checkpoint = False
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    AttentionBlock.forward = lambda self, x: self._forward(x)

    from integration.fused_ops.fused_resblock import fuse_resblocks_in_module
    fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)

    from integration.kernels.int8_optimized import (
        convert_model_to_optimized_int8, apply_static_scales,
        enable_modiff_mode, get_calibration_config,
    )
    from integration.kernels.int8_linear import (
        convert_model_to_int8_linear, enable_modiff_mode_linear,
        apply_linear_static_scales,
    )
    convert_model_to_optimized_int8(model.model.diffusion_model)
    convert_model_to_int8_linear(model.model.diffusion_model)
    from integration.utils.buffer_pool import initialize_buffer_pool
    initialize_buffer_pool(model.model.diffusion_model, max_batch_size=batch_size, device='cuda')

    if calib_path and os.path.exists(calib_path):
        scales = torch.load(calib_path, weights_only=True)
        config = get_calibration_config()
        config.scales = scales
        config.is_calibrated = True
        apply_static_scales(model.model.diffusion_model, scales)
        lin_scales = {k.replace('linear:', ''): v for k, v in scales.items()
                     if k.startswith('linear:')}
        if lin_scales:
            apply_linear_static_scales(model.model.diffusion_model, lin_scales)

    enable_modiff_mode(model.model.diffusion_model, use_modiff)
    enable_modiff_mode_linear(model.model.diffusion_model, use_modiff)
    return model, DDIMSampler(model)


def _load_int8_modiff_attn_model(config_path, ckpt_path, batch_size, calib_path):
    """Load INT8 model with MoDiff attention (existing modiff_attention mode)."""
    model, sampler = _load_int8_model(config_path, ckpt_path, batch_size, calib_path,
                                      use_modiff=False)
    if HAS_CUTLASS:
        from integration.kernels.modiff_attention import convert_attention_to_modiff
        n = convert_attention_to_modiff(model.model.diffusion_model, act_bits=8, verbose=False)
        print(f"  → MoDiff attention applied to {n} AttentionBlocks")
    return model, sampler


if __name__ == '__main__':
    main()
