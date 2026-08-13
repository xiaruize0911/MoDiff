"""
Batch Size Ablation Benchmark.

Studies how increasing batch_size affects latency/throughput for FP32, FP16, INT8, INT4.
Runs the full DDIM pipeline at each batch size and collects:
  - Time per sample (latency)
  - Samples per second (throughput)
  - Speedup of each quantized mode vs FP32 at the same batch size

Usage:
    python integration/benchmarks/benchmark_batchsize.py --steps 50
    python integration/benchmarks/benchmark_batchsize.py --batch_sizes 1 4 8 16 32 --steps 200
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
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.getcwd())

from integration.benchmarks.benchmark_ldm import BenchmarkRunner, load_model

# Import quantization modules
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
from omegaconf import OmegaConf


def _pref_calib(bits):
    """CALIBRATION_PREFERENCE for a bit width -- see the note in report/kernel_suites_bench.py.

    Used as the FALLBACK for --int{8,4}_calib. Those flags used to default to the stub-checkpoint
    literal; defaulting them to None instead would have been worse than the bug, because the consumer
    below skips loading when the path is falsy, so an unset flag would silently run the model on
    per-call DYNAMIC scales rather than on the wrong static ones. Resolve, do not blank.
    """
    from integration.benchmarks.benchmark_ldm import CALIBRATION_PREFERENCE, _pick
    return _pick(CALIBRATION_PREFERENCE[f"int{bits}"], "calibration")


def setup_model_for_mode(config_path, ckpt_path, mode, batch_size, calib_path=None):
    """Load and configure model for given mode."""
    from ldm.util import instantiate_from_config
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

    if mode in ('int8', 'int8_baseline') and HAS_INT8:
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
            apply_static_scales(model.model.diffusion_model, scales)
            if HAS_INT8_LINEAR:
                lin_scales = {k.replace('linear:', ''): v for k, v in scales.items() if k.startswith('linear:')}
                if lin_scales:
                    apply_linear_static_scales(model.model.diffusion_model, lin_scales)
        enable = (mode == 'int8')
        enable_modiff_int8(model.model.diffusion_model, enable)
        if HAS_INT8_LINEAR:
            enable_modiff_mode_linear(model.model.diffusion_model, enable)
    elif mode in ('int4', 'int4_baseline') and HAS_INT4:
        convert_model_to_optimized_int4(model.model.diffusion_model)
        if HAS_INT4_LINEAR:
            convert_model_to_int4_linear(model.model.diffusion_model)
        from integration.utils.buffer_pool import initialize_buffer_pool
        initialize_buffer_pool(model.model.diffusion_model, max_batch_size=batch_size, device='cuda')
        if calib_path and os.path.exists(calib_path):
            scales = torch.load(calib_path, weights_only=True)
            apply_int4_static_scales(model.model.diffusion_model, scales)
            if HAS_INT4_LINEAR:
                lin_scales = {k.replace('linear:', ''): v for k, v in scales.items() if k.startswith('linear:')}
                if lin_scales:
                    apply_int4_linear_static_scales(model.model.diffusion_model, lin_scales)
        enable = (mode == 'int4')
        enable_modiff_int4(model.model.diffusion_model, enable)
        if HAS_INT4_LINEAR:
            enable_modiff_mode_int4_linear(model.model.diffusion_model, enable)

    return model, DDIMSampler(model)


def benchmark_at_batch_size(model, sampler, mode, batch_size, steps, shape=(4, 32, 32),
                             num_batches=4, use_autocast=True, dtype=torch.float16):
    """Run timed inference at a given batch_size and return time-per-sample."""
    # Reset MoDiff state
    if mode in ('int8', 'int8_baseline') and HAS_INT8:
        reset_modiff_int8(model.model.diffusion_model)
    if mode in ('int8', 'int8_baseline') and HAS_INT8_LINEAR:
        reset_modiff_state_linear(model.model.diffusion_model)
    if mode in ('int4', 'int4_baseline') and HAS_INT4:
        reset_modiff_int4(model.model.diffusion_model)
    if mode in ('int4', 'int4_baseline') and HAS_INT4_LINEAR:
        reset_modiff_state_int4_linear(model.model.diffusion_model)

    # Warmup
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
        sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
    torch.cuda.synchronize()

    # Reset again post-warmup
    if mode in ('int8', 'int8_baseline') and HAS_INT8:
        reset_modiff_int8(model.model.diffusion_model)
    if mode in ('int4', 'int4_baseline') and HAS_INT4:
        reset_modiff_int4(model.model.diffusion_model)

    times = []
    for _ in range(num_batches):
        if mode in ('int8', 'int8_baseline') and HAS_INT8:
            reset_modiff_int8(model.model.diffusion_model)
        if mode in ('int4', 'int4_baseline') and HAS_INT4:
            reset_modiff_int4(model.model.diffusion_model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
            sampler.sample(S=steps, batch_size=batch_size, shape=shape, eta=0.0, verbose=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_total = sum(times) / len(times)
    return avg_total / batch_size  # seconds per sample


def main():
    parser = argparse.ArgumentParser(description="Batch Size Ablation Benchmark")
    parser.add_argument('--config', type=str, default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt', type=str, default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['fp32', 'fp16', 'int8', 'int4'],
                        choices=['fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline'])
    parser.add_argument('--int8_calib', type=str, default=None)
    parser.add_argument('--int4_calib', type=str, default=None)
    parser.add_argument('--num_batches', type=int, default=4, help='Timed repetitions per batch size')
    parser.add_argument('--output', type=str, default='integration/results/batchsize_ablation.json')
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Steps: {args.steps} | Batch sizes: {args.batch_sizes}")
    print(f"Modes: {args.modes}\n")

    results = {}  # mode -> {bs: time_per_sample}

    for mode in args.modes:
        print(f"\n{'='*50}\nMode: {mode.upper()}\n{'='*50}")
        results[mode] = {}
        calib = ((args.int4_calib or _pref_calib(4)) if 'int4' in mode
                 else (args.int8_calib or _pref_calib(8)))
        max_bs = max(args.batch_sizes)

        model, sampler = setup_model_for_mode(
            args.config, args.ckpt, mode, max_bs,
            calib_path=calib if 'fp' not in mode else None
        )
        use_ac = mode != 'fp32'
        dtype = torch.float16 if use_ac else None

        for bs in args.batch_sizes:
            print(f"  batch_size={bs}...", end='', flush=True)
            try:
                t = benchmark_at_batch_size(
                    model, sampler, mode, bs, args.steps,
                    num_batches=args.num_batches,
                    use_autocast=use_ac, dtype=dtype
                )
                throughput = 1.0 / t
                results[mode][bs] = {'time_per_sample_s': round(t, 4),
                                     'throughput': round(throughput, 3)}
                print(f" {t*1000:.1f} ms/sample  ({throughput:.2f} samples/s)")
            except Exception as e:
                print(f" ERROR: {e}")
                results[mode][bs] = None

        del model, sampler
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*70}")
    print("BATCH SIZE ABLATION SUMMARY — Time per Sample (ms)")
    print(f"{'='*70}")
    header = f"{'BatchSize':<12}" + "".join(f"{m:<18}" for m in args.modes)
    print(header)
    print("-" * (12 + 18 * len(args.modes)))
    for bs in args.batch_sizes:
        row = f"{bs:<12}"
        for mode in args.modes:
            v = results[mode].get(bs)
            if v:
                row += f"{v['time_per_sample_s']*1000:>8.1f} ms    "
            else:
                row += f"{'ERROR':<18}"
        print(row)

    print(f"\nINT vs FP32 Speedup")
    print("-" * (12 + 18 * len(args.modes)))
    fp32_times = {bs: results.get('fp32', {}).get(bs, {}).get('time_per_sample_s', None)
                  for bs in args.batch_sizes}
    for bs in args.batch_sizes:
        row = f"{bs:<12}"
        fp32_t = fp32_times.get(bs)
        for mode in args.modes:
            v = results[mode].get(bs)
            if v and fp32_t:
                speedup = fp32_t / v['time_per_sample_s']
                row += f"{speedup:>7.2f}x         "
            else:
                row += f"{'—':<18}"
        print(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
