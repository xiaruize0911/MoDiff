"""
Attention Layer Ablation Benchmark.

Studies the effect of skipping attention layers on speed vs quality:
  - fp16_full:       Full FP16 model (attention included) — reference
  - fp16_skip_attn:  FP16 but AttentionBlock replaced with identity (skip)
  - int8_full:       INT8 conv + linear, attention in FP16 — standard
  - int8_skip_attn:  INT8 conv + linear, attention skipped

Reports:
  1. Time/step breakdown with/without attention
  2. % time savings from skipping attention
  3. Per-mode speedup vs FP32 baseline

Usage:
    python integration/benchmarks/benchmark_skip_attention.py --steps 50
"""
import argparse
import os
import sys
import json
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore', message='Could not initialize NNPACK')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.getcwd())

try:
    from integration.kernels.int8_optimized import (
        convert_model_to_optimized_int8, apply_static_scales,
        enable_modiff_mode as enable_modiff_int8,
        reset_modiff_state as reset_modiff_int8,
        get_calibration_config as get_calib_int8,
    )
    HAS_INT8 = True
except ImportError:
    HAS_INT8 = False

try:
    from integration.kernels.int4_optimized import (
        convert_model_to_optimized_int4, apply_int4_static_scales,
        enable_modiff_mode as enable_modiff_int4,
        reset_modiff_state as reset_modiff_int4,
    )
    HAS_INT4 = True
except ImportError:
    HAS_INT4 = False

try:
    from integration.kernels.int8_linear import (
        convert_model_to_int8_linear, enable_modiff_mode_linear,
        reset_modiff_state_linear, apply_linear_static_scales,
    )
    HAS_INT8_LINEAR = True
except ImportError:
    HAS_INT8_LINEAR = False

try:
    from integration.kernels.int4_linear import (
        convert_model_to_int4_linear, enable_modiff_mode_int4_linear,
        reset_modiff_state_int4_linear, apply_int4_linear_static_scales,
    )
    HAS_INT4_LINEAR = True
except ImportError:
    HAS_INT4_LINEAR = False

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


def load_model(config_path, ckpt_path):
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval()


def setup_mode(config_path, ckpt_path, base_mode, skip_attention, batch_size, calib_path=None):
    """
    base_mode: 'fp16' | 'int8' | 'int4'
    skip_attention: bool — replace AttentionBlock forward with identity
    """
    model = load_model(config_path, ckpt_path)
    model = model.to(memory_format=torch.channels_last)

    for m in model.modules():
        if hasattr(m, 'use_checkpoint'):
            m.use_checkpoint = False

    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    if skip_attention:
        # Replace attention with identity pass-through
        AttentionBlock.forward = lambda self, x: x
        print("  → AttentionBlock: SKIPPED (identity)")
    else:
        AttentionBlock.forward = lambda self, x: self._forward(x)
        print("  → AttentionBlock: ACTIVE")

    from integration.fused_ops.fused_resblock import fuse_resblocks_in_module
    fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)

    if base_mode in ('int8', 'int8_baseline') and HAS_INT8:
        from integration.kernels.int8_optimized import OptimizedInt8Conv2d
        n_conv = sum(1 for m in model.model.diffusion_model.modules() if isinstance(m, nn.Conv2d))
        convert_model_to_optimized_int8(model.model.diffusion_model)
        if HAS_INT8_LINEAR:
            convert_model_to_int8_linear(model.model.diffusion_model)
        from integration.utils.buffer_pool import initialize_buffer_pool
        initialize_buffer_pool(model.model.diffusion_model, max_batch_size=batch_size, device='cuda')
        if calib_path and os.path.exists(calib_path):
            scales = torch.load(calib_path, weights_only=True)
            config = get_calib_int8()
            config.scales = scales
            config.is_calibrated = True
            n = apply_static_scales(model.model.diffusion_model, scales)
            print(f"  → INT8 calibration: {n} conv scales loaded")
            if HAS_INT8_LINEAR:
                lin_scales = {k.replace('linear:', ''): v for k, v in scales.items() if k.startswith('linear:')}
                if lin_scales:
                    apply_linear_static_scales(model.model.diffusion_model, lin_scales)
        enable = (base_mode == 'int8')
        enable_modiff_int8(model.model.diffusion_model, enable)
        if HAS_INT8_LINEAR:
            enable_modiff_mode_linear(model.model.diffusion_model, enable)
    elif base_mode in ('int4', 'int4_baseline') and HAS_INT4:
        convert_model_to_optimized_int4(model.model.diffusion_model)
        if HAS_INT4_LINEAR:
            convert_model_to_int4_linear(model.model.diffusion_model)
        from integration.utils.buffer_pool import initialize_buffer_pool
        initialize_buffer_pool(model.model.diffusion_model, max_batch_size=batch_size, device='cuda')
        if calib_path and os.path.exists(calib_path):
            scales = torch.load(calib_path, weights_only=True)
            n = apply_int4_static_scales(model.model.diffusion_model, scales)
            print(f"  → INT4 calibration: {n} conv scales loaded")
            if HAS_INT4_LINEAR:
                lin_scales = {k.replace('linear:', ''): v for k, v in scales.items() if k.startswith('linear:')}
                if lin_scales:
                    apply_int4_linear_static_scales(model.model.diffusion_model, lin_scales)
        enable = (base_mode == 'int4')
        enable_modiff_int4(model.model.diffusion_model, enable)
        if HAS_INT4_LINEAR:
            enable_modiff_mode_int4_linear(model.model.diffusion_model, enable)

    return model, DDIMSampler(model)


def timed_inference(model, sampler, mode_base, batch_size, steps, shape=(4, 32, 32),
                    use_autocast=True, dtype=torch.float16, num_batches=4):
    """Run warmed-up timed inference, return time-per-sample."""
    def reset():
        if mode_base in ('int8', 'int8_baseline') and HAS_INT8:
            reset_modiff_int8(model.model.diffusion_model)
        if mode_base in ('int8', 'int8_baseline') and HAS_INT8_LINEAR:
            reset_modiff_state_linear(model.model.diffusion_model)
        if mode_base in ('int4', 'int4_baseline') and HAS_INT4:
            reset_modiff_int4(model.model.diffusion_model)
        if mode_base in ('int4', 'int4_baseline') and HAS_INT4_LINEAR:
            reset_modiff_state_int4_linear(model.model.diffusion_model)

    reset()
    # Warmup
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
        sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_batches):
        reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
            sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return avg / batch_size  # per sample


def main():
    parser = argparse.ArgumentParser(description="Attention Layer Skip Ablation")
    parser.add_argument('--config', type=str, default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt', type=str, default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--int8_calib', type=str, default='integration/calibration/int8_calibration.pt')
    parser.add_argument('--int4_calib', type=str, default='integration/calibration/int4_calibration.pt')
    parser.add_argument('--num_batches', type=int, default=4)
    parser.add_argument('--output', type=str, default='integration/results/skip_attention_ablation.json')
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Steps={args.steps}, batch_size={args.batch_size}\n")

    # Experiments: (base_mode, skip_attention, label)
    experiments = [
        ('fp16', False, 'fp16_full_attn'),
        ('fp16', True,  'fp16_skip_attn'),
        ('int8', False, 'int8_full_attn'),
        ('int8', True,  'int8_skip_attn'),
        ('int4', False, 'int4_full_attn'),
        ('int4', True,  'int4_skip_attn'),
        ('int8_baseline', False, 'int8_base_full_attn'),
        ('int8_baseline', True,  'int8_base_skip_attn'),
    ]

    results = {}

    for base_mode, skip_attn, label in experiments:
        if 'int8' in base_mode and not HAS_INT8:
            print(f"Skipping {label}: INT8 not available")
            continue
        if 'int4' in base_mode and not HAS_INT4:
            print(f"Skipping {label}: INT4 not available")
            continue

        print(f"\n{'='*55}\n{label.upper()}\n{'='*55}")
        calib = args.int4_calib if 'int4' in base_mode else args.int8_calib
        model, sampler = setup_mode(
            args.config, args.ckpt, base_mode, skip_attn, args.batch_size,
            calib_path=calib if 'fp' not in base_mode else None
        )

        use_ac = 'fp32' not in base_mode
        t = timed_inference(model, sampler, base_mode, args.batch_size,
                            args.steps, use_autocast=use_ac,
                            dtype=torch.float16 if use_ac else None,
                            num_batches=args.num_batches)

        results[label] = {
            'time_per_sample_s': round(t, 4),
            'time_per_step_ms': round(t / args.steps * 1000, 3),
            'throughput': round(1.0 / t, 3),
            'skip_attention': skip_attn,
            'base_mode': base_mode,
        }
        print(f"  {t*1000:.1f} ms/sample | {t/args.steps*1000:.2f} ms/step | {1/t:.2f} samples/s")

        del model, sampler
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print("ATTENTION SKIP ABLATION — Results")
    print(f"{'='*70}")
    print(f"{'Mode':<28} {'ms/sample':>10} {'ms/step':>10} {'samples/s':>12} {'vs full_attn':>14}")
    print("-" * 74)

    for base_mode in ['fp16', 'int8', 'int8_baseline', 'int4']:
        full_key = f"{base_mode}_full_attn"
        skip_key = f"{base_mode}_skip_attn"
        if full_key not in results:
            continue
        r_full = results[full_key]
        t_full = r_full['time_per_sample_s']
        print(f"{full_key:<28} {t_full*1000:>10.1f} {r_full['time_per_step_ms']:>10.2f} {r_full['throughput']:>12.2f} {'(reference)':>14}")
        if skip_key in results:
            r_skip = results[skip_key]
            t_skip = r_skip['time_per_sample_s']
            speedup = t_full / t_skip
            savings = (1 - t_skip / t_full) * 100
            print(f"{skip_key:<28} {t_skip*1000:>10.1f} {r_skip['time_per_step_ms']:>10.2f} {r_skip['throughput']:>12.2f} {f'+{savings:.1f}% faster':>14}")
        print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
