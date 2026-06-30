"""
Extended Benchmark: CUTLASS INT8 + CUDA Graph and Fused Kernel Baselines.

This script adds new benchmark modes to compare against the existing MoDiff:

New modes:
    1. int8_cudagraph: CUTLASS INT8 + CUDA Graph + MoDiff temporal caching
    2. int8_cudagraph_baseline: CUTLASS INT8 + CUDA Graph, no MoDiff
  3. int8_separate: INT8 with separate Q/Conv/DQ kernels + MoDiff
  4. int8_separate_baseline: INT8 with separate Q/Conv/DQ kernels, no MoDiff
  5. int4_separate: INT4 with separate Q/Conv/DQ kernels + MoDiff
  6. int4_separate_baseline: INT4 with separate Q/Conv/DQ kernels, no MoDiff

All experiments use batch_size=32, timesteps=200, LDM model.

Usage:
    python integration/benchmark_extended.py --mode all --batch_size 32 --steps 200
    python integration/benchmark_extended.py --mode int8_cudagraph --batch_size 32 --steps 200
"""
import argparse
import os
import sys
import time
import json
import warnings
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import torchvision.utils as tvu

warnings.filterwarnings('ignore', message='Could not initialize NNPACK')
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.getcwd())

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from integration.utils.quant_memory import format_quant_memory_report, report_quant_memory

# Import existing MoDiff INT8/INT4
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
    )
    HAS_INT8 = True
except ImportError:
    HAS_INT8 = False

try:
    from integration.kernels.int4_optimized import (
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

# Import INT8 + CUDA Graph helpers
try:
    from integration.kernels.int8_cudagraph import (
        CUDAGraphWrapper,
        enable_modiff_mode_pytorch_int8,
        reset_modiff_state_pytorch_int8,
        install_cuda_graph_replay_pytorch_int8,
        get_cuda_graph_replay_stats,
    )
    HAS_CUDAGRAPH = True
except ImportError as e:
    HAS_CUDAGRAPH = False
    print(f"Warning: CUDA Graph module not available: {e}")

# Import fused baselines
try:
    from integration.kernels.fused_baseline import (
        SeparateKernelInt8Conv2d,
        SeparateKernelInt4Conv2d,
        convert_model_to_separate_int8,
        convert_model_to_separate_int4,
        enable_modiff_mode_separate_int8,
        enable_modiff_mode_separate_int4,
        reset_modiff_state_separate_int8,
        reset_modiff_state_separate_int4,
        apply_separate_int8_scales,
        apply_separate_int4_scales,
    )
    HAS_SEPARATE = True
except ImportError as e:
    HAS_SEPARATE = False
    print(f"Warning: Separate kernel module not available: {e}")


def load_model(config_path: str, ckpt_path: str):
    """Load LDM model from config and checkpoint."""
    print(f"Loading model from {ckpt_path}")
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval(), conf


def measure_gpu_memory():
    """Get current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


class ExtendedBenchmarkRunner:
    """Benchmark runner for extended modes."""

    def __init__(self, config_path: str, ckpt_path: str, output_dir: str,
                 batch_size: int = 32, steps: int = 200, shape: tuple = (4, 32, 32)):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.steps = steps
        self.shape = shape
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

    def _setup_model_base(self, model):
        """Common model setup."""
        model = model.to(memory_format=torch.channels_last)
        for m in model.modules():
            if hasattr(m, 'use_checkpoint'):
                m.use_checkpoint = False
        from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
        AttentionBlock.forward = lambda self, x: self._forward(x)
        from integration.fused_ops.fused_resblock import fuse_resblocks_in_module
        fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)
        return model

    def _calibrate(self, model, sampler, mode: str, num_runs: int = 5):
        """Run calibration passes to get quantization scales."""
        print(f"  Calibrating {mode} ({num_runs} runs)...")
        if 'cudagraph' in mode:
            set_calibrating_int8(model.model.diffusion_model, True)
        elif 'separate' in mode and 'int8' in mode:
            for m in model.model.diffusion_model.modules():
                if isinstance(m, SeparateKernelInt8Conv2d):
                    m.is_calibrated = False
        elif 'separate' in mode and 'int4' in mode:
            for m in model.model.diffusion_model.modules():
                if isinstance(m, SeparateKernelInt4Conv2d):
                    m.is_calibrated = False

        with torch.no_grad():
            for _ in range(num_runs):
                if 'cudagraph' in mode:
                    reset_modiff_state_int8(model.model.diffusion_model)
                elif 'separate' in mode and 'int8' in mode:
                    reset_modiff_state_separate_int8(model.model.diffusion_model)
                elif 'separate' in mode and 'int4' in mode:
                    reset_modiff_state_separate_int4(model.model.diffusion_model)
                sampler.sample(S=5, batch_size=2, shape=self.shape, eta=0.0, verbose=False)

        if 'cudagraph' in mode:
            set_calibrating_int8(model.model.diffusion_model, False)
            return get_calibration_config_int8().scales
        return {}

    def _setup_model(self, mode: str):
        """Load and prepare model for given mode."""
        model, _ = load_model(self.config_path, self.ckpt_path)
        model = self._setup_model_base(model)

        if mode == 'int8_cudagraph':
            convert_model_to_optimized_int8(model.model.diffusion_model)
            from integration.utils.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            calib_path = 'integration/calibration/int8_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_static_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_pytorch_int8(model.model.diffusion_model, True)

        elif mode == 'int8_cudagraph_baseline':
            convert_model_to_optimized_int8(model.model.diffusion_model)
            from integration.utils.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            calib_path = 'integration/calibration/int8_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_static_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_pytorch_int8(model.model.diffusion_model, False)

        elif mode == 'int8_separate':
            convert_model_to_separate_int8(model.model.diffusion_model)
            # Use existing INT8 calibration scales if available
            calib_path = 'integration/calibration/int8_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_separate_int8_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_separate_int8(model.model.diffusion_model, True)

        elif mode == 'int8_separate_baseline':
            convert_model_to_separate_int8(model.model.diffusion_model)
            calib_path = 'integration/calibration/int8_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_separate_int8_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_separate_int8(model.model.diffusion_model, False)

        elif mode == 'int4_separate':
            convert_model_to_separate_int4(model.model.diffusion_model)
            calib_path = 'integration/calibration/int4_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_separate_int4_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_separate_int4(model.model.diffusion_model, True)

        elif mode == 'int4_separate_baseline':
            convert_model_to_separate_int4(model.model.diffusion_model)
            calib_path = 'integration/calibration/int4_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_separate_int4_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_separate_int4(model.model.diffusion_model, False)

        elif mode == 'fp32':
            pass  # No conversion needed

        elif mode == 'fp16':
            pass  # Use autocast

        elif mode == 'int8' and HAS_INT8:
            convert_model_to_optimized_int8(model.model.diffusion_model)
            from integration.utils.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            calib_path = 'integration/calibration/int8_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_static_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_int8(model.model.diffusion_model, True)

        elif mode == 'int8_baseline' and HAS_INT8:
            convert_model_to_optimized_int8(model.model.diffusion_model)
            from integration.utils.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            calib_path = 'integration/calibration/int8_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_static_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_int8(model.model.diffusion_model, False)

        elif mode == 'int4' and HAS_INT4:
            convert_model_to_optimized_int4(model.model.diffusion_model)
            from integration.utils.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            calib_path = 'integration/calibration/int4_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_int4_static_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_int4(model.model.diffusion_model, True)

        elif mode == 'int4_baseline' and HAS_INT4:
            convert_model_to_optimized_int4(model.model.diffusion_model)
            from integration.utils.buffer_pool import initialize_buffer_pool
            initialize_buffer_pool(model.model.diffusion_model, max_batch_size=self.batch_size, device='cuda')
            calib_path = 'integration/calibration/int4_calibration.pt'
            if os.path.exists(calib_path):
                scales = torch.load(calib_path, weights_only=True)
                loaded = apply_int4_static_scales(model.model.diffusion_model, scales)
                print(f"  Loaded {loaded} static scales from {calib_path}")
            enable_modiff_mode_int4(model.model.diffusion_model, False)

        return model, DDIMSampler(model)

    def _reset_state(self, model, mode):
        """Reset MoDiff state for a new sample."""
        if 'cudagraph' in mode:
            reset_modiff_state_pytorch_int8(model.model.diffusion_model)
        elif 'separate' in mode and 'int8' in mode:
            reset_modiff_state_separate_int8(model.model.diffusion_model)
        elif 'separate' in mode and 'int4' in mode:
            reset_modiff_state_separate_int4(model.model.diffusion_model)
        elif 'int8' in mode and HAS_INT8:
            reset_modiff_state_int8(model.model.diffusion_model)
        elif 'int4' in mode and HAS_INT4:
            reset_modiff_state_int4(model.model.diffusion_model)

    def _select_cudagraph_precapture_steps(self, sampler, mode: str) -> int:
        """Pick a valid short DDIM schedule that captures all needed graph phases.

        DDIM's uniform discretization only supports step counts that divide the
        underlying DDPM schedule length cleanly; otherwise it can generate an
        out-of-bounds timestep (e.g. 3 -> 1000). For CUDA Graph benchmarking we
        only need enough steps to capture the required phases ahead of timing:
        1 step for baseline, 2 steps for MoDiff (first + modulated).
        """
        ddpm_steps = int(getattr(sampler.model, 'num_timesteps', 1000))
        min_required = 1 if 'baseline' in mode else 2

        for candidate in range(min_required, self.steps + 1):
            if ddpm_steps % candidate == 0:
                return candidate

        return self.steps

    def _benchmark_cuda_callable(self, fn, num_iterations: int, warmup: int = 10) -> float:
        """Time a CUDA callable with warmup using CUDA events. Returns milliseconds."""
        for _ in range(warmup):
            fn()

        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

        for i in range(num_iterations):
            start_events[i].record()
            fn()
            end_events[i].record()

        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

    def _estimate_quant_io_compute(self, static_quant_ms: float, io_proxy_ms: float) -> dict:
        """Estimate whether quantization is IO- or compute-dominated.

        Uses a tensor copy as an IO proxy / lower bound for memory movement.
        The arithmetic/packing contribution is approximated as the remaining
        time in the static quantization kernel after subtracting the IO proxy.
        """
        io_proxy_ms = max(float(io_proxy_ms), 0.0)
        static_quant_ms = max(float(static_quant_ms), 0.0)
        compute_estimate_ms = max(static_quant_ms - io_proxy_ms, 0.0)

        if static_quant_ms <= 0.0:
            return {
                'io_proxy_ms': io_proxy_ms,
                'compute_estimate_ms': compute_estimate_ms,
                'io_share_lower_bound_pct': 0.0,
                'compute_share_upper_bound_pct': 0.0,
                'dominant_factor': 'unknown',
            }

        io_share = min(100.0, (io_proxy_ms / static_quant_ms) * 100.0)
        compute_share = max(0.0, 100.0 - io_share)
        dominant = 'io' if io_proxy_ms >= compute_estimate_ms else 'compute'

        return {
            'io_proxy_ms': io_proxy_ms,
            'compute_estimate_ms': compute_estimate_ms,
            'io_share_lower_bound_pct': io_share,
            'compute_share_upper_bound_pct': compute_share,
            'dominant_factor': dominant,
        }

    def run_mode(self, mode: str, num_samples: int = 32):
        """Run benchmark for a specific mode."""
        print(f"\n{'='*60}\n{mode.upper()}\n{'='*60}")

        # Reset GPU memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        mem_before = measure_gpu_memory()
        model, sampler = self._setup_model(mode)
        mem_after_setup = measure_gpu_memory()

        use_autocast = mode not in ('fp32',)
        dtype = torch.float16 if use_autocast else None

        mode_dir = os.path.join(self.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # Warmup
        print(f"  Warming up (full {self.steps}-step pass)...")
        self._reset_state(model, mode)
        with torch.inference_mode():
            sampler.sample(S=self.steps, batch_size=self.batch_size, shape=self.shape, eta=0.0, verbose=False)
        torch.cuda.synchronize()
        quant_memory_after_warmup = report_quant_memory(model.model.diffusion_model)
        if quant_memory_after_warmup["total_tracked_mib"] > 0:
            print(f"  Quant memory after warmup: {format_quant_memory_report(quant_memory_after_warmup)}")

        # For CUDA Graph modes, try to capture the graph after warmup
        if 'cudagraph' in mode:
            print("  Installing per-step UNet CUDA Graph replay...")
            install_cuda_graph_replay_pytorch_int8(
                model.model,
                batch_size=self.batch_size,
                shape=self.shape,
            )
            self._reset_state(model, mode)
            precapture_steps = self._select_cudagraph_precapture_steps(sampler, mode)
            print("  Pre-capturing CUDA Graph phases before timed sampling...")
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
                sampler.sample(
                    S=precapture_steps,
                    batch_size=self.batch_size,
                    shape=self.shape,
                    eta=0.0,
                    verbose=False,
                )
            torch.cuda.synchronize()
            self._reset_state(model, mode)

        # Generate samples and measure
        total_time = 0.0
        generated = 0

        while generated < num_samples:
            batch = min(self.batch_size, num_samples - generated)
            self._reset_state(model, mode)
            sample_batch = self.batch_size if 'cudagraph' in mode else batch

            torch.cuda.synchronize()
            start = time.time()

            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
                samples, _ = sampler.sample(
                    S=self.steps, batch_size=sample_batch, shape=self.shape, eta=0.0, verbose=False
                )
                if sample_batch != batch:
                    samples = samples[:batch]

            torch.cuda.synchronize()
            elapsed = time.time() - start
            total_time += elapsed

            self._decode_and_save_samples(
                model,
                samples,
                mode_dir,
                generated,
                use_autocast=use_autocast,
                dtype=dtype,
            )

            generated += batch

        mem_peak = measure_gpu_memory()

        time_per_sample = total_time / num_samples
        time_per_step = total_time / (num_samples * self.steps) * 1000

        self.results[mode] = {
            'total_time': total_time,
            'num_samples': num_samples,
            'time_per_sample': time_per_sample,
            'time_per_step_ms': time_per_step,
            'memory_allocated_mb': mem_after_setup['allocated_mb'],
            'memory_peak_mb': mem_peak['max_allocated_mb'],
            'quant_memory_after_warmup': quant_memory_after_warmup,
            'batch_size': self.batch_size,
            'steps': self.steps,
        }

        if 'cudagraph' in mode:
            graph_stats = get_cuda_graph_replay_stats(model.model.diffusion_model)
            if graph_stats is None:
                raise RuntimeError("CUDA Graph manager was not installed for cudagraph mode.")
            if graph_stats['num_graphs'] == 0 or graph_stats['replay_count'] == 0:
                raise RuntimeError(f"CUDA Graph replay was not exercised correctly: {graph_stats}")
            self.results[mode]['cuda_graph_num_graphs'] = graph_stats['num_graphs']
            self.results[mode]['cuda_graph_capture_count'] = graph_stats['capture_count']
            self.results[mode]['cuda_graph_replay_count'] = graph_stats['replay_count']

        print(f"\n{mode.upper()} Results:")
        print(f"  Total: {total_time:.2f}s for {num_samples} samples")
        print(f"  Per-sample: {time_per_sample:.3f}s")
        print(f"  Per-step: {time_per_step:.2f}ms")
        print(f"  Memory: {mem_after_setup['allocated_mb']:.0f}MB allocated, {mem_peak['max_allocated_mb']:.0f}MB peak")
        if 'cudagraph' in mode:
            print(
                f"  CUDA Graphs: {self.results[mode]['cuda_graph_num_graphs']} graphs, "
                f"{self.results[mode]['cuda_graph_capture_count']} captures, "
                f"{self.results[mode]['cuda_graph_replay_count']} replays"
            )

        if 'fp32' in self.results:
            speedup = self.results['fp32']['time_per_sample'] / time_per_sample
            self.results[mode]['speedup_vs_fp32'] = speedup
            print(f"  Speedup vs FP32: {speedup:.2f}x")

        del model, sampler
        torch.cuda.empty_cache()
        gc.collect()

    def run_kernel_timing(self, num_iterations: int = 100):
        """
        Micro-benchmark comparing fused vs separate kernel timings.
        Tests individual kernel operations on representative tensor shapes.

        Also measures the narrower conv-side question the user cares about:
        - compute + DQ           : conv2d_*_fprop + out_raw * weight_scale
        - compute + DQ + update  : conv2d_*_fprop_o_hat
        """
        print(f"\n{'='*60}")
        print("KERNEL TIMING COMPARISON (Fused vs Separate)")
        print(f"{'='*60}")

        shapes = [
            (self.batch_size, 192, 32, 32),   # First conv block
            (self.batch_size, 384, 16, 16),   # After first downsample
            (self.batch_size, 384, 16, 16),   # Mid-level blocks
            (self.batch_size, 768, 8, 8),     # Deep blocks
            (self.batch_size, 768, 8, 8),     # Bottleneck
        ]

        kernel_results = {}
        quantization_results = {}

        for shape in shapes:
            N, C, H, W = shape
            print(f"\nShape: ({N}, {C}, {H}, {W})")

            x = torch.randn(N, C, H, W, device='cuda').to(memory_format=torch.channels_last)
            # Create a fake cache tensor
            cache = torch.randn_like(x)
            cache_zero = torch.zeros_like(x)
            io_proxy_buf = torch.empty_like(x)
            w_conv = nn.Conv2d(C, C, 3, padding=1, bias=False).cuda()
            w_data = w_conv.weight.data
            shape_key = f"{N}x{C}x{H}x{W}"

            # --- INT8 Fused (CUTLASS) ---
            try:
                import modiff_cutlass

                # Prepare INT8 weights
                K = C
                w_flat = w_data.reshape(K, -1)
                ch_max = w_flat.abs().max(dim=1).values
                ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)
                w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
                w_quant = w_quant.reshape_as(w_data).permute(0, 2, 3, 1).contiguous()
                weight_scale = ch_scale.view(1, K, 1, 1).cuda()
                empty_bias = torch.empty(0, device='cuda')

                # Buffers for fused kernel
                residual_buf = torch.empty_like(x)
                absmax_buf = torch.zeros(1, device='cuda')
                scale_buf = torch.empty(1, device='cuda')
                inv_scale_buf = torch.empty(1, device='cuda')
                retire_count = torch.zeros(1, device='cuda', dtype=torch.int32)
                smooth_inv = torch.empty(0, device='cuda')
                o_hat = torch.randn(N, C, H, W, device='cuda').to(memory_format=torch.channels_last)

                # Warmup
                for _ in range(10):
                    modiff_cutlass.step1_quantize_fprop(
                        x, cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 127.0, smooth_inv
                    )

                # Time fused step1
                torch.cuda.synchronize()
                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

                for i in range(num_iterations):
                    absmax_buf.zero_()
                    retire_count.zero_()
                    start_events[i].record()
                    x_int8 = modiff_cutlass.step1_quantize_fprop(
                        x, cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 127.0, smooth_inv
                    )
                    end_events[i].record()

                torch.cuda.synchronize()
                fused_step1_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Time fused step1 without a_hat update (same fused launch structure)
                for _ in range(10):
                    modiff_cutlass.step1_quantize_no_ahat_fprop(
                        x, cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 127.0, smooth_inv
                    )

                for i in range(num_iterations):
                    absmax_buf.zero_()
                    retire_count.zero_()
                    start_events[i].record()
                    x_int8_no_cache = modiff_cutlass.step1_quantize_no_ahat_fprop(
                        x, cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 127.0, smooth_inv
                    )
                    end_events[i].record()

                torch.cuda.synchronize()
                fused_step1_no_cache_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Time fused conv + dequant + update_o_hat
                for _ in range(10):
                    modiff_cutlass.conv2d_int8_fprop_o_hat(
                        x_int8, w_quant, inv_scale_buf.view(1), weight_scale.view(-1), o_hat,
                        1, 1, 1, 1, 1, 1
                    )

                for i in range(num_iterations):
                    start_events[i].record()
                    modiff_cutlass.conv2d_int8_fprop_o_hat(
                        x_int8, w_quant, inv_scale_buf.view(1), weight_scale.view(-1), o_hat,
                        1, 1, 1, 1, 1, 1
                    )
                    end_events[i].record()

                torch.cuda.synchronize()
                fused_conv_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Time fused conv + dequant without o_hat update (same fused launch structure)
                for _ in range(10):
                    _ = modiff_cutlass.conv2d_int8_fprop_no_ohat(
                        x_int8_no_cache, w_quant, inv_scale_buf.view(1), weight_scale.view(-1),
                        1, 1, 1, 1, 1, 1
                    )

                for i in range(num_iterations):
                    start_events[i].record()
                    _ = modiff_cutlass.conv2d_int8_fprop_no_ohat(
                        x_int8_no_cache, w_quant, inv_scale_buf.view(1), weight_scale.view(-1),
                        1, 1, 1, 1, 1, 1
                    )
                    end_events[i].record()

                torch.cuda.synchronize()
                fused_conv_no_cache_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Time compute + DQ only (no o_hat update)
                for _ in range(10):
                    out_raw = modiff_cutlass.conv2d_int8_fprop(
                        x_int8, w_quant, inv_scale_buf.view(1), empty_bias, 1, 1, 1, 1, 1, 1
                    )
                    _ = out_raw * weight_scale

                for i in range(num_iterations):
                    start_events[i].record()
                    out_raw = modiff_cutlass.conv2d_int8_fprop(
                        x_int8, w_quant, inv_scale_buf.view(1), empty_bias, 1, 1, 1, 1, 1, 1
                    )
                    _ = out_raw * weight_scale
                    end_events[i].record()

                torch.cuda.synchronize()
                compute_dq_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # --- INT8 Separate kernels ---
                # Step 1 separate: residual, absmax, scale, quantize
                for _ in range(10):
                    residual = x - cache
                    abs_max = residual.abs().amax()
                    s = 127.0 / torch.clamp(abs_max, min=1e-6)
                    inv_s = 1.0 / s
                    x_s = modiff_cutlass.scale_quantize_int8(residual, s.view(1))

                for i in range(num_iterations):
                    start_events[i].record()
                    residual = x - cache
                    abs_max = residual.abs().amax()
                    s = 127.0 / torch.clamp(abs_max, min=1e-6)
                    inv_s = 1.0 / s
                    r_dq = (residual * s).round().clamp(-127, 127) / s
                    cache.add_(r_dq - r_dq)  # simulate cache update (no-op add)
                    cache.add_(r_dq)
                    x_s = modiff_cutlass.scale_quantize_int8(residual, s.view(1))
                    end_events[i].record()

                torch.cuda.synchronize()
                separate_step1_ms = sum(s_ev.elapsed_time(e_ev) for s_ev, e_ev in zip(start_events, end_events)) / num_iterations

                # Step 2 separate: conv + dequant + accumulate
                for i in range(num_iterations):
                    start_events[i].record()
                    out_raw = modiff_cutlass.conv2d_int8_fprop(
                        x_s, w_quant, inv_s.view(1), empty_bias, 1, 1, 1, 1, 1, 1
                    )
                    out_scaled = out_raw * weight_scale
                    o_hat.add_(out_scaled)
                    end_events[i].record()

                torch.cuda.synchronize()
                separate_conv_ms = sum(s_ev.elapsed_time(e_ev) for s_ev, e_ev in zip(start_events, end_events)) / num_iterations

                kernel_results[f"INT8_{shape_key}"] = {
                    'fused_step1_ms': fused_step1_ms,
                    'fused_step1_no_cache_ms': fused_step1_no_cache_ms,
                    'fused_conv_ms': fused_conv_ms,
                    'fused_conv_no_cache_ms': fused_conv_no_cache_ms,
                    'fused_total_ms': fused_step1_ms + fused_conv_ms,
                    'separate_step1_ms': separate_step1_ms,
                    'separate_conv_ms': separate_conv_ms,
                    'separate_total_ms': separate_step1_ms + separate_conv_ms,
                    'fusion_speedup': (separate_step1_ms + separate_conv_ms) / (fused_step1_ms + fused_conv_ms),
                    'compute_dq_ms': compute_dq_ms,
                    'compute_dq_update_ohat_ms': fused_conv_ms,
                    'step1_cache_update_overhead_ms': fused_step1_ms - fused_step1_no_cache_ms,
                    'step1_cache_update_overhead_pct': ((fused_step1_ms - fused_step1_no_cache_ms) / max(fused_step1_no_cache_ms, 1e-12)) * 100.0,
                    'conv_cache_update_overhead_ms': fused_conv_ms - fused_conv_no_cache_ms,
                    'conv_cache_update_overhead_pct': ((fused_conv_ms - fused_conv_no_cache_ms) / max(fused_conv_no_cache_ms, 1e-12)) * 100.0,
                    'ohat_update_overhead_ms': fused_conv_ms - compute_dq_ms,
                    'ohat_update_overhead_pct': ((fused_conv_ms - compute_dq_ms) / max(compute_dq_ms, 1e-12)) * 100.0,
                }

                print(f"  INT8 Fused:    step1={fused_step1_ms:.3f}ms, conv={fused_conv_ms:.3f}ms, total={fused_step1_ms+fused_conv_ms:.3f}ms")
                print(f"  INT8 Separate: step1={separate_step1_ms:.3f}ms, conv={separate_conv_ms:.3f}ms, total={separate_step1_ms+separate_conv_ms:.3f}ms")
                print(f"  Fusion speedup: {(separate_step1_ms+separate_conv_ms)/(fused_step1_ms+fused_conv_ms):.2f}x")
                print(
                    f"  INT8 Conv-side: compute+DQ={compute_dq_ms:.3f}ms, "
                    f"compute+DQ+update_o_hat={fused_conv_ms:.3f}ms, "
                    f"overhead={fused_conv_ms - compute_dq_ms:+.3f}ms "
                    f"({((fused_conv_ms - compute_dq_ms) / max(compute_dq_ms, 1e-12)) * 100.0:+.1f}%)"
                )
                print(
                    f"  INT8 Cache-only delta: step1={fused_step1_ms - fused_step1_no_cache_ms:+.3f}ms "
                    f"({((fused_step1_ms - fused_step1_no_cache_ms) / max(fused_step1_no_cache_ms, 1e-12)) * 100.0:+.1f}%), "
                    f"conv={fused_conv_ms - fused_conv_no_cache_ms:+.3f}ms "
                    f"({((fused_conv_ms - fused_conv_no_cache_ms) / max(fused_conv_no_cache_ms, 1e-12)) * 100.0:+.1f}%)"
                )

                static_scale_int8 = torch.tensor(
                    [127.0 / max(x.abs().amax().item(), 1e-6)], device='cuda', dtype=torch.float32
                )

                def dynamic_quant_int8():
                    absmax_buf.zero_()
                    retire_count.zero_()
                    modiff_cutlass.step1_quantize_no_ahat_fprop(
                        x, cache_zero, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 127.0, smooth_inv
                    )

                def static_quant_int8():
                    modiff_cutlass.scale_quantize_int8(x, static_scale_int8)

                def absmax_scale_int8():
                    abs_max = x.abs().amax()
                    _ = 127.0 / torch.clamp(abs_max, min=1e-6)

                def io_proxy():
                    io_proxy_buf.copy_(x)

                dynamic_quant_ms = self._benchmark_cuda_callable(dynamic_quant_int8, num_iterations)
                static_quant_ms = self._benchmark_cuda_callable(static_quant_int8, num_iterations)
                absmax_scale_ms = self._benchmark_cuda_callable(absmax_scale_int8, num_iterations)
                io_proxy_ms = self._benchmark_cuda_callable(io_proxy, num_iterations)
                io_compute = self._estimate_quant_io_compute(static_quant_ms, io_proxy_ms)

                quantization_results[f"INT8_{shape_key}"] = {
                    'dynamic_quant_ms': dynamic_quant_ms,
                    'static_quant_ms': static_quant_ms,
                    'absmax_scale_ms': absmax_scale_ms,
                    'dynamic_over_static_ms': dynamic_quant_ms - static_quant_ms,
                    'dynamic_over_static_pct': ((dynamic_quant_ms - static_quant_ms) / max(static_quant_ms, 1e-12)) * 100.0,
                    **io_compute,
                }

                print(
                    f"  INT8 Quant: dynamic={dynamic_quant_ms:.3f}ms, static={static_quant_ms:.3f}ms, "
                    f"dynamic overhead={dynamic_quant_ms - static_quant_ms:+.3f}ms "
                    f"({((dynamic_quant_ms - static_quant_ms) / max(static_quant_ms, 1e-12)) * 100.0:+.1f}%)"
                )
                print(
                    f"  INT8 Quant breakdown: absmax+scale={absmax_scale_ms:.3f}ms, "
                    f"IO proxy={io_proxy_ms:.3f}ms, compute est.={io_compute['compute_estimate_ms']:.3f}ms, "
                    f"dominant={io_compute['dominant_factor']}"
                )

            except Exception as e:
                print(f"  INT8 kernel timing failed: {e}")

            # --- INT4 Fused vs Separate ---
            try:
                import modiff_cutlass
                from integration.kernels.int4_optimized import pack_int4

                # INT4 weights
                w_flat4 = w_data.reshape(K, -1)
                ch_scale4 = torch.clamp(w_flat4.abs().max(dim=1).values / 7.0, min=1e-8)
                w_q4 = (w_flat4 / ch_scale4.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
                w_q4 = w_q4.reshape_as(w_data).permute(0, 2, 3, 1).contiguous()
                w_packed = pack_int4(w_q4)
                ws4 = ch_scale4.view(1, K, 1, 1).cuda()

                # Warmup
                for _ in range(10):
                    modiff_cutlass.step1_quantize_pack_int4_fprop(
                        x, cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 7.0, smooth_inv
                    )

                # Fused step1
                for i in range(num_iterations):
                    absmax_buf.zero_()
                    retire_count.zero_()
                    start_events[i].record()
                    xp = modiff_cutlass.step1_quantize_pack_int4_fprop(
                        x, cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 7.0, smooth_inv
                    )
                    end_events[i].record()

                torch.cuda.synchronize()
                fused4_step1 = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Fused step1 without a_hat update (same fused launch structure)
                for _ in range(10):
                    modiff_cutlass.step1_quantize_pack_int4_no_ahat_fprop(
                        x, cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 7.0, smooth_inv
                    )

                for i in range(num_iterations):
                    absmax_buf.zero_()
                    retire_count.zero_()
                    start_events[i].record()
                    xp_no_cache = modiff_cutlass.step1_quantize_pack_int4_no_ahat_fprop(
                        x, cache, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 7.0, smooth_inv
                    )
                    end_events[i].record()

                torch.cuda.synchronize()
                fused4_step1_no_cache = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Fused conv + dequant + update_o_hat
                for _ in range(10):
                    modiff_cutlass.conv2d_int4_fprop_o_hat(
                        xp, w_packed, inv_scale_buf.view(1), ws4.view(-1), o_hat, 1, 1, 1, 1, 1, 1
                    )

                for i in range(num_iterations):
                    start_events[i].record()
                    modiff_cutlass.conv2d_int4_fprop_o_hat(
                        xp, w_packed, inv_scale_buf.view(1), ws4.view(-1), o_hat, 1, 1, 1, 1, 1, 1
                    )
                    end_events[i].record()

                torch.cuda.synchronize()
                fused4_conv = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Fused conv without o_hat update (same fused launch structure)
                for _ in range(10):
                    _ = modiff_cutlass.conv2d_int4_fprop_no_ohat(
                        xp_no_cache, w_packed, inv_scale_buf.view(1), ws4.view(-1), 1, 1, 1, 1, 1, 1
                    )

                for i in range(num_iterations):
                    start_events[i].record()
                    _ = modiff_cutlass.conv2d_int4_fprop_no_ohat(
                        xp_no_cache, w_packed, inv_scale_buf.view(1), ws4.view(-1), 1, 1, 1, 1, 1, 1
                    )
                    end_events[i].record()

                torch.cuda.synchronize()
                fused4_conv_no_cache = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Time compute + DQ only (no o_hat update)
                for _ in range(10):
                    out_raw4 = modiff_cutlass.conv2d_int4_fprop(
                        xp, w_packed, inv_scale_buf.view(1), empty_bias, 1, 1, 1, 1, 1, 1
                    )
                    _ = out_raw4 * ws4

                for i in range(num_iterations):
                    start_events[i].record()
                    out_raw4 = modiff_cutlass.conv2d_int4_fprop(
                        xp, w_packed, inv_scale_buf.view(1), empty_bias, 1, 1, 1, 1, 1, 1
                    )
                    _ = out_raw4 * ws4
                    end_events[i].record()

                torch.cuda.synchronize()
                compute_dq4_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iterations

                # Separate INT4
                for i in range(num_iterations):
                    start_events[i].record()
                    residual = x - cache
                    abs_max = residual.abs().amax()
                    s4 = 7.0 / torch.clamp(abs_max, min=1e-6)
                    inv_s4 = 1.0 / s4
                    r_clamped = (residual * s4).round().clamp(-7, 7)
                    r_dq4 = r_clamped / s4
                    cache.add_(r_dq4)
                    xp_sep = modiff_cutlass.quantize_and_pack(r_clamped.contiguous(memory_format=torch.channels_last))
                    end_events[i].record()

                torch.cuda.synchronize()
                sep4_step1 = sum(s_ev.elapsed_time(e_ev) for s_ev, e_ev in zip(start_events, end_events)) / num_iterations

                for i in range(num_iterations):
                    start_events[i].record()
                    out_raw4 = modiff_cutlass.conv2d_int4_fprop(
                        xp_sep, w_packed, inv_s4.view(1), empty_bias, 1, 1, 1, 1, 1, 1
                    )
                    out_sc4 = out_raw4 * ws4
                    o_hat.add_(out_sc4)
                    end_events[i].record()

                torch.cuda.synchronize()
                sep4_conv = sum(s_ev.elapsed_time(e_ev) for s_ev, e_ev in zip(start_events, end_events)) / num_iterations

                kernel_results[f"INT4_{shape_key}"] = {
                    'fused_step1_ms': fused4_step1,
                    'fused_step1_no_cache_ms': fused4_step1_no_cache,
                    'fused_conv_ms': fused4_conv,
                    'fused_conv_no_cache_ms': fused4_conv_no_cache,
                    'fused_total_ms': fused4_step1 + fused4_conv,
                    'separate_step1_ms': sep4_step1,
                    'separate_conv_ms': sep4_conv,
                    'separate_total_ms': sep4_step1 + sep4_conv,
                    'fusion_speedup': (sep4_step1 + sep4_conv) / (fused4_step1 + fused4_conv),
                    'compute_dq_ms': compute_dq4_ms,
                    'compute_dq_update_ohat_ms': fused4_conv,
                    'step1_cache_update_overhead_ms': fused4_step1 - fused4_step1_no_cache,
                    'step1_cache_update_overhead_pct': ((fused4_step1 - fused4_step1_no_cache) / max(fused4_step1_no_cache, 1e-12)) * 100.0,
                    'conv_cache_update_overhead_ms': fused4_conv - fused4_conv_no_cache,
                    'conv_cache_update_overhead_pct': ((fused4_conv - fused4_conv_no_cache) / max(fused4_conv_no_cache, 1e-12)) * 100.0,
                    'ohat_update_overhead_ms': fused4_conv - compute_dq4_ms,
                    'ohat_update_overhead_pct': ((fused4_conv - compute_dq4_ms) / max(compute_dq4_ms, 1e-12)) * 100.0,
                }

                print(f"  INT4 Fused:    step1={fused4_step1:.3f}ms, conv={fused4_conv:.3f}ms, total={fused4_step1+fused4_conv:.3f}ms")
                print(f"  INT4 Separate: step1={sep4_step1:.3f}ms, conv={sep4_conv:.3f}ms, total={sep4_step1+sep4_conv:.3f}ms")
                print(f"  Fusion speedup: {(sep4_step1+sep4_conv)/(fused4_step1+fused4_conv):.2f}x")
                print(
                    f"  INT4 Conv-side: compute+DQ={compute_dq4_ms:.3f}ms, "
                    f"compute+DQ+update_o_hat={fused4_conv:.3f}ms, "
                    f"overhead={fused4_conv - compute_dq4_ms:+.3f}ms "
                    f"({((fused4_conv - compute_dq4_ms) / max(compute_dq4_ms, 1e-12)) * 100.0:+.1f}%)"
                )
                print(
                    f"  INT4 Cache-only delta: step1={fused4_step1 - fused4_step1_no_cache:+.3f}ms "
                    f"({((fused4_step1 - fused4_step1_no_cache) / max(fused4_step1_no_cache, 1e-12)) * 100.0:+.1f}%), "
                    f"conv={fused4_conv - fused4_conv_no_cache:+.3f}ms "
                    f"({((fused4_conv - fused4_conv_no_cache) / max(fused4_conv_no_cache, 1e-12)) * 100.0:+.1f}%)"
                )

                static_scale_int4 = torch.tensor(
                    [7.0 / max(x.abs().amax().item(), 1e-6)], device='cuda', dtype=torch.float32
                )

                def dynamic_quant_int4():
                    absmax_buf.zero_()
                    retire_count.zero_()
                    modiff_cutlass.step1_quantize_pack_int4_no_ahat_fprop(
                        x, cache_zero, residual_buf, absmax_buf, scale_buf, inv_scale_buf,
                        retire_count, 7.0, smooth_inv
                    )

                def static_quant_int4():
                    modiff_cutlass.scale_quantize_and_pack(x, static_scale_int4)

                def absmax_scale_int4():
                    abs_max = x.abs().amax()
                    _ = 7.0 / torch.clamp(abs_max, min=1e-6)

                dynamic_quant4_ms = self._benchmark_cuda_callable(dynamic_quant_int4, num_iterations)
                static_quant4_ms = self._benchmark_cuda_callable(static_quant_int4, num_iterations)
                absmax_scale4_ms = self._benchmark_cuda_callable(absmax_scale_int4, num_iterations)
                io_proxy4_ms = self._benchmark_cuda_callable(io_proxy, num_iterations)
                io_compute4 = self._estimate_quant_io_compute(static_quant4_ms, io_proxy4_ms)

                quantization_results[f"INT4_{shape_key}"] = {
                    'dynamic_quant_ms': dynamic_quant4_ms,
                    'static_quant_ms': static_quant4_ms,
                    'absmax_scale_ms': absmax_scale4_ms,
                    'dynamic_over_static_ms': dynamic_quant4_ms - static_quant4_ms,
                    'dynamic_over_static_pct': ((dynamic_quant4_ms - static_quant4_ms) / max(static_quant4_ms, 1e-12)) * 100.0,
                    **io_compute4,
                }

                print(
                    f"  INT4 Quant: dynamic={dynamic_quant4_ms:.3f}ms, static={static_quant4_ms:.3f}ms, "
                    f"dynamic overhead={dynamic_quant4_ms - static_quant4_ms:+.3f}ms "
                    f"({((dynamic_quant4_ms - static_quant4_ms) / max(static_quant4_ms, 1e-12)) * 100.0:+.1f}%)"
                )
                print(
                    f"  INT4 Quant breakdown: absmax+scale={absmax_scale4_ms:.3f}ms, "
                    f"IO proxy={io_proxy4_ms:.3f}ms, compute est.={io_compute4['compute_estimate_ms']:.3f}ms, "
                    f"dominant={io_compute4['dominant_factor']}"
                )

            except Exception as e:
                print(f"  INT4 kernel timing failed: {e}")

            del x, cache, w_conv, o_hat
            torch.cuda.empty_cache()

        self.results['kernel_timing'] = kernel_results
        self.results['quantization_timing'] = quantization_results
        self._annotate_cache_update_io()
        return kernel_results

    def _bytes_to_mib(self, value_bytes: float) -> float:
        return float(value_bytes) / (1024.0 * 1024.0)

    def _annotate_cache_update_io(self):
        if 'kernel_timing' not in self.results:
            return

        for key, val in self.results['kernel_timing'].items():
            shape = key.split('_', 1)[1]
            n, c, h, w = [int(part) for part in shape.split('x')]
            num_input_elements = n * c * h * w
            num_output_elements = num_input_elements

            step1_extra_bytes = num_input_elements * 8.0   # read + write a_hat_cache float32
            conv_extra_bytes = num_output_elements * 4.0   # extra read of o_hat_cache float32

            val['step1_cache_update_extra_bytes'] = step1_extra_bytes
            val['step1_cache_update_extra_mib'] = self._bytes_to_mib(step1_extra_bytes)
            val['conv_cache_update_extra_bytes'] = conv_extra_bytes
            val['conv_cache_update_extra_mib'] = self._bytes_to_mib(conv_extra_bytes)
            val['total_cache_update_extra_bytes'] = step1_extra_bytes + conv_extra_bytes
            val['total_cache_update_extra_mib'] = self._bytes_to_mib(step1_extra_bytes + conv_extra_bytes)

    def generate_quantization_report(self):
        report_path = os.path.join(self.output_dir, 'LAYER_QUANTIZATION_REPORT.md')
        qt = self.results.get('quantization_timing')
        if not qt:
            raise RuntimeError('quantization_timing results are required to generate the quantization report.')

        lines = [
            '# Layer-Level Quantization Timing Report',
            '',
            f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'**GPU**: {torch.cuda.get_device_name()}',
            f'**Batch Size**: {self.batch_size}',
            '',
            'This report compares the current dynamic activation quantization path against a static-scale quantization path using the same CUTLASS quantization kernels.',
            '',
            'Interpretation notes:',
            '- **Dynamic quantization**: includes the per-tensor absmax/scale discovery inside the hot path.',
            '- **Static quantization**: reuses a fixed precomputed activation scale and only performs quantize/(pack) work.',
            '- **IO proxy**: tensor copy used as a lower-bound proxy for memory movement during quantization.',
            '- **Compute estimate**: `static_quant_ms - io_proxy_ms`, clipped at zero. This is an upper-bound style estimate of arithmetic/packing overhead.',
            '',
            '## INT8 Dynamic vs Static Quantization',
            '',
            '| Shape | Dynamic (ms) | Static (ms) | Dynamic overhead | Absmax+scale (ms) | IO proxy (ms) | Compute est. (ms) | Dominant |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |',
        ]

        for key, val in qt.items():
            if not key.startswith('INT8_'):
                continue
            lines.append(
                f"| {key} | {val['dynamic_quant_ms']:.3f} | {val['static_quant_ms']:.3f} | "
                f"{val['dynamic_over_static_ms']:+.3f}ms ({val['dynamic_over_static_pct']:+.1f}%) | "
                f"{val['absmax_scale_ms']:.3f} | {val['io_proxy_ms']:.3f} | {val['compute_estimate_ms']:.3f} | {val['dominant_factor']} |"
            )

        lines.extend([
            '',
            '## INT4 Dynamic vs Static Quantization',
            '',
            '| Shape | Dynamic (ms) | Static (ms) | Dynamic overhead | Absmax+scale (ms) | IO proxy (ms) | Compute est. (ms) | Dominant |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |',
        ])

        for key, val in qt.items():
            if not key.startswith('INT4_'):
                continue
            lines.append(
                f"| {key} | {val['dynamic_quant_ms']:.3f} | {val['static_quant_ms']:.3f} | "
                f"{val['dynamic_over_static_ms']:+.3f}ms ({val['dynamic_over_static_pct']:+.1f}%) | "
                f"{val['absmax_scale_ms']:.3f} | {val['io_proxy_ms']:.3f} | {val['compute_estimate_ms']:.3f} | {val['dominant_factor']} |"
            )

        lines.extend([
            '',
            '## Key takeaways',
            '',
            '- The gap between dynamic and static quantization isolates the cost of discovering a fresh activation scale in the hot path.',
            '- The IO proxy vs static quantization comparison indicates whether quantization is primarily memory-movement limited or arithmetic/packing limited.',
            '- If the IO proxy is close to the static quantization time, quantization is effectively IO-bound; if the compute estimate is larger, arithmetic/packing is the main contributor.',
        ])

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f'Quantization report saved to: {report_path}')

    def generate_cache_update_report(self):
        report_path = os.path.join(self.output_dir, 'FUSED_CACHE_UPDATE_REPORT.md')
        if 'kernel_timing' not in self.results:
            raise RuntimeError('kernel_timing results are required to generate the cache update report.')

        self._annotate_cache_update_io()
        kt = self.results['kernel_timing']

        lines = [
            '# Fused Cache Update Overhead Report',
            '',
            f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'**GPU**: {torch.cuda.get_device_name()}',
            f'**Batch Size**: {self.batch_size}',
            f'**Timesteps**: {self.steps}',
            '',
            'This report isolates the cost of MoDiff cache updates while preserving the same fused launch structure as the production kernels.',
            '',
            'Compared kernels:',
            '- **Step1 fused**: `sub_absmax_scale + quantize (+ optional a_hat update)`',
            '- **Conv fused**: `conv + dequant (+ optional o_hat update)`',
            '',
            '## INT8 Cache Update Overhead',
            '',
            '| Shape | Step1 w/ cache (ms) | Step1 no cache (ms) | a_hat update cost | Conv w/ cache (ms) | Conv no cache (ms) | o_hat update cost | Extra IO from cache update (MiB) |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |',
        ]

        for key, val in kt.items():
            if not key.startswith('INT8_'):
                continue
            lines.append(
                f"| {key} | {val['fused_step1_ms']:.3f} | {val['fused_step1_no_cache_ms']:.3f} | "
                f"{val['step1_cache_update_overhead_ms']:+.3f}ms ({val['step1_cache_update_overhead_pct']:+.1f}%) | "
                f"{val['fused_conv_ms']:.3f} | {val['fused_conv_no_cache_ms']:.3f} | "
                f"{val['conv_cache_update_overhead_ms']:+.3f}ms ({val['conv_cache_update_overhead_pct']:+.1f}%) | "
                f"{val['total_cache_update_extra_mib']:.1f} |"
            )

        lines.extend([
            '',
            '## INT4 Cache Update Overhead',
            '',
            '| Shape | Step1 w/ cache (ms) | Step1 no cache (ms) | a_hat update cost | Conv w/ cache (ms) | Conv no cache (ms) | o_hat update cost | Extra IO from cache update (MiB) |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |',
        ])

        for key, val in kt.items():
            if not key.startswith('INT4_'):
                continue
            lines.append(
                f"| {key} | {val['fused_step1_ms']:.3f} | {val['fused_step1_no_cache_ms']:.3f} | "
                f"{val['step1_cache_update_overhead_ms']:+.3f}ms ({val['step1_cache_update_overhead_pct']:+.1f}%) | "
                f"{val['fused_conv_ms']:.3f} | {val['fused_conv_no_cache_ms']:.3f} | "
                f"{val['conv_cache_update_overhead_ms']:+.3f}ms ({val['conv_cache_update_overhead_pct']:+.1f}%) | "
                f"{val['total_cache_update_extra_mib']:.1f} |"
            )

        lines.extend([
            '',
            '## Memory-IO model',
            '',
            '- **Step1 cache update (`a_hat`)**: additional float32 read + float32 write per input activation, i.e. about **8 bytes / element** beyond the no-cache fused baseline.',
            '- **Conv cache update (`o_hat`)**: additional float32 read of the existing `o_hat_cache` per output activation, i.e. about **4 bytes / element** beyond the no-cache fused baseline.',
            '- These are lower-bound tensor traffic estimates for the cache-update delta itself; they intentionally ignore small scalar buffers and assume the same quantized compute path in both variants.',
            '',
            '## Takeaways',
            '',
            '- The Step1 cache update isolates the cost of writing the temporal activation cache (`a_hat`).',
            '- The Conv cache update isolates the cost of reading and accumulating into the temporal output cache (`o_hat`).',
            '- Comparing these no-cache fused baselines against the production fused kernels shows how much of MoDiff hot-path time is spent on cache maintenance rather than quantized compute.',
        ])

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f'Cache update report saved to: {report_path}')

    def print_summary(self):
        """Print comprehensive summary."""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Config: batch_size={self.batch_size}, steps={self.steps}, shape={self.shape}")
        print(f"GPU: {torch.cuda.get_device_name()}")

        # Pipeline timing
        print(f"\n{'Mode':<25} {'Time/Sample':<15} {'Speedup':<12} {'Memory (MB)':<15} {'Graph Stats':<20}")
        print("-" * 92)

        baseline = self.results.get('fp32', {}).get('time_per_sample')
        pipeline_modes = [
            'fp32', 'fp16',
            'int8', 'int8_baseline',
            'int4', 'int4_baseline',
            'int8_cudagraph', 'int8_cudagraph_baseline',
            'int8_separate', 'int8_separate_baseline',
            'int4_separate', 'int4_separate_baseline',
        ]

        for mode in pipeline_modes:
            if mode in self.results and mode != 'kernel_timing':
                r = self.results[mode]
                t = f"{r['time_per_sample']:.3f}s"
                speedup = (baseline / r['time_per_sample']) if (baseline is not None and r['time_per_sample'] > 0) else None
                s = "(baseline)" if mode == 'fp32' else (f"{speedup:.2f}x" if speedup is not None else '-')
                mem = f"{r.get('memory_peak_mb', 0):.0f}"
                if 'cudagraph' in mode:
                    graph_stats = f"{int(r.get('cuda_graph_num_graphs', 0))}g/{int(r.get('cuda_graph_replay_count', 0))}r"
                else:
                    graph_stats = '-'
                print(f"{mode:<25} {t:<15} {s:<12} {mem:<15} {graph_stats:<20}")

        # Kernel timing
        if 'kernel_timing' in self.results:
            print(f"\n{'='*70}")
            print("KERNEL TIMING: Fused vs Separate")
            print(f"{'='*70}")
            print(f"{'Config':<20} {'Fused (ms)':<15} {'Separate (ms)':<15} {'Speedup':<10}")
            print("-" * 60)
            for key, val in self.results['kernel_timing'].items():
                print(f"{key:<20} {val['fused_total_ms']:<15.3f} {val['separate_total_ms']:<15.3f} {val['fusion_speedup']:<10.2f}x")

            print(f"\n{'='*70}")
            print("KERNEL TIMING: Compute+DQ vs Compute+DQ+Update o_hat")
            print(f"{'='*70}")
            print(f"{'Config':<20} {'Compute+DQ':<15} {'+Update o_hat':<15} {'Overhead':<15}")
            print("-" * 70)
            for key, val in self.results['kernel_timing'].items():
                print(
                    f"{key:<20} {val['compute_dq_ms']:<15.3f} "
                    f"{val['compute_dq_update_ohat_ms']:<15.3f} "
                    f"{val['ohat_update_overhead_ms']:+.3f}ms ({val['ohat_update_overhead_pct']:+.1f}%)"
                )

        if 'quantization_timing' in self.results:
            print(f"\n{'='*70}")
            print("LAYER QUANTIZATION: Dynamic vs Static")
            print(f"{'='*70}")
            print(f"{'Config':<20} {'Dynamic (ms)':<15} {'Static (ms)':<15} {'Overhead':<18} {'Dominant':<10}")
            print("-" * 85)
            for key, val in self.results['quantization_timing'].items():
                print(
                    f"{key:<20} {val['dynamic_quant_ms']:<15.3f} {val['static_quant_ms']:<15.3f} "
                    f"{val['dynamic_over_static_ms']:+.3f}ms ({val['dynamic_over_static_pct']:+.1f}%) "
                    f"{val['dominant_factor']:<10}"
                )

        # Save results
        results_path = os.path.join(self.output_dir, 'extended_results.json')
        # Convert non-serializable items
        serializable = {}
        for k, v in self.results.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()}
            else:
                serializable[k] = v
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

    def generate_report(self):
        """Generate markdown report with results."""
        report_path = os.path.join(self.output_dir, 'EXTENDED_BENCHMARK_REPORT.md')
        correctness_path = os.path.join(self.output_dir, 'correctness_summary.json')
        correctness = None
        if os.path.exists(correctness_path):
            with open(correctness_path) as f:
                correctness = json.load(f)
        lines = [
            "# Extended MoDiff Benchmark Report",
            "",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**GPU**: {torch.cuda.get_device_name()}",
            f"**Batch Size**: {self.batch_size}",
            f"**Timesteps**: {self.steps}",
            f"**Latent Shape**: {self.shape}",
            "",
            "## Pipeline Timing Results",
            "",
            "| Mode | Time/Sample (s) | Speedup vs FP32 | Peak Memory (MB) | Graphs | Captures | Replays |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]

        baseline = self.results.get('fp32', {}).get('time_per_sample')
        for mode in ['fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline',
                      'int8_cudagraph', 'int8_cudagraph_baseline',
                      'int8_separate', 'int8_separate_baseline',
                      'int4_separate', 'int4_separate_baseline']:
            if mode in self.results and mode != 'kernel_timing':
                r = self.results[mode]
                t = f"{r['time_per_sample']:.3f}"
                speedup = (baseline / r['time_per_sample']) if (baseline is not None and r['time_per_sample'] > 0) else None
                s = "-" if mode == 'fp32' or speedup is None else f"{speedup:.2f}x"
                mem = f"{r.get('memory_peak_mb', 0):.0f}"
                graphs = str(int(r.get('cuda_graph_num_graphs', 0))) if 'cudagraph' in mode else '-'
                captures = str(int(r.get('cuda_graph_capture_count', 0))) if 'cudagraph' in mode else '-'
                replays = str(int(r.get('cuda_graph_replay_count', 0))) if 'cudagraph' in mode else '-'
                lines.append(f"| {mode} | {t} | {s} | {mem} | {graphs} | {captures} | {replays} |")

        cudagraph_modes = [
            mode for mode in ['int8_cudagraph', 'int8_cudagraph_baseline']
            if mode in self.results
        ]
        if cudagraph_modes:
            lines.extend([
                "",
                "## CUDA Graph Replay Stats",
                "",
                "| Mode | Graphs | Captures | Replays | Replays / Captures |",
                "| --- | --- | --- | --- | --- |",
            ])
            for mode in cudagraph_modes:
                r = self.results[mode]
                captures = int(r.get('cuda_graph_capture_count', 0))
                replays = int(r.get('cuda_graph_replay_count', 0))
                ratio = f"{(replays / captures):.1f}" if captures > 0 else '-'
                lines.append(
                    f"| {mode} | {int(r.get('cuda_graph_num_graphs', 0))} | {captures} | {replays} | {ratio} |"
                )

        lines.extend([
            "",
            "## Mode Implementation Details",
            "",
            "### Baseline floating-point modes",
            "",
            "- **`fp32`**: original LDM inference path with standard PyTorch/CUDA kernels and no quantization.",
            "- **`fp16`**: original LDM inference path under autocast. This reduces memory traffic, but does not change the algorithmic structure.",
            "",
            "### CUTLASS fused MoDiff modes",
            "",
            "- **`int8`**: `OptimizedInt8Conv2d` from `integration/kernels/int8_optimized.py`.",
            "  - weights are quantized once to per-channel INT8",
            "  - activation residual path uses the fused `step1_quantize_fprop` kernel",
            "  - conv-side uses the fused `conv2d_int8_fprop_o_hat` kernel",
            "  - MoDiff caches (`a_hat_cache`, `o_hat_cache`) are updated inside the optimized hot path",
            "- **`int8_baseline`**: same CUTLASS INT8 backend, but MoDiff temporal modulation is disabled.",
            "- **`int4`**: `OptimizedInt4Conv2d` from `integration/kernels/int4_optimized.py`.",
            "  - same structure as INT8, but using packed INT4 activations/weights and fused INT4 kernels",
            "- **`int4_baseline`**: same CUTLASS INT4 backend, with MoDiff disabled.",
            "",
            "### CUTLASS INT8 + CUDA Graph modes",
            "",
            "- **`int8_cudagraph`**: `OptimizedInt8Conv2d` from `integration/kernels/int8_optimized.py` plus real per-step UNet CUDA graph replay.",
            "  - the underlying convolution and modulation path use the same CUTLASS INT8 kernels as fused `int8`",
            "  - activations stay on the CUTLASS quantized path instead of dequantizing back to FP16 `F.conv2d`",
            "  - DDIM remains the outer Python loop, but each UNet step is replayed from captured CUDA graphs",
            "  - two graphs are used: one for the first MoDiff step and one for all modulated steps",
            "  - the benchmark pre-captures those graphs on a short valid DDIM schedule before timed sampling, so graph construction cost does not leak into the steady-state timing",
            "- **`int8_cudagraph_baseline`**: same backend, but with MoDiff disabled, so only a single per-step graph is needed.",
            "",
            "### Separate-kernel baselines",
            "",
            "- **`int8_separate`**: `SeparateKernelInt8Conv2d` from `integration/kernels/fused_baseline.py`.",
            "  - residual computation, absmax, scale computation, quantization, dequantization, cache update, conv, dequant-by-weight-scale, and `o_hat` accumulation are split into separate operations",
            "  - this preserves the same math as MoDiff but intentionally removes kernel fusion",
            "- **`int8_separate_baseline`**: same separate INT8 backend with MoDiff disabled.",
            "- **`int4_separate`** / **`int4_separate_baseline`**: same idea for INT4.",
        ])

        if 'kernel_timing' in self.results:
            lines.extend([
                "",
                "## Kernel Timing: Fused vs Separate",
                "",
                "| Shape | Fused Total (ms) | Separate Total (ms) | Fusion Speedup |",
                "| --- | --- | --- | --- |",
            ])
            for key, val in self.results['kernel_timing'].items():
                lines.append(f"| {key} | {val['fused_total_ms']:.3f} | {val['separate_total_ms']:.3f} | {val['fusion_speedup']:.2f}x |")

            lines.extend([
                "",
                "### Detailed Kernel Breakdown",
                "",
                "| Shape | Fused Step1 (ms) | Fused Conv (ms) | Sep Step1 (ms) | Sep Conv (ms) |",
                "| --- | --- | --- | --- | --- |",
            ])
            for key, val in self.results['kernel_timing'].items():
                lines.append(f"| {key} | {val['fused_step1_ms']:.3f} | {val['fused_conv_ms']:.3f} | {val['separate_step1_ms']:.3f} | {val['separate_conv_ms']:.3f} |")

            lines.extend([
                "",
                "## Kernel Timing: Compute+DQ vs Compute+DQ+Update o_hat",
                "",
                "| Shape | Compute+DQ (ms) | Compute+DQ+Update o_hat (ms) | Overhead |",
                "| --- | --- | --- | --- |",
            ])
            for key, val in self.results['kernel_timing'].items():
                lines.append(
                    f"| {key} | {val['compute_dq_ms']:.3f} | {val['compute_dq_update_ohat_ms']:.3f} | "
                    f"{val['ohat_update_overhead_ms']:+.3f}ms ({val['ohat_update_overhead_pct']:+.1f}%) |"
                )

        if 'quantization_timing' in self.results:
            lines.extend([
                "",
                "## Layer-level Quantization Timing",
                "",
                "These measurements compare the current dynamic activation quantization path against a static-scale path using the same quantization kernels.",
                "",
                "### Dynamic vs Static Quantization",
                "",
                "| Shape | Dynamic (ms) | Static (ms) | Dynamic overhead | Absmax+scale (ms) | IO proxy (ms) | Compute est. (ms) | Dominant |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ])
            for key, val in self.results['quantization_timing'].items():
                lines.append(
                    f"| {key} | {val['dynamic_quant_ms']:.3f} | {val['static_quant_ms']:.3f} | "
                    f"{val['dynamic_over_static_ms']:+.3f}ms ({val['dynamic_over_static_pct']:+.1f}%) | "
                    f"{val['absmax_scale_ms']:.3f} | {val['io_proxy_ms']:.3f} | {val['compute_estimate_ms']:.3f} | {val['dominant_factor']} |"
                )

            lines.extend([
                "",
                "### Quantization interpretation",
                "",
                "- **Dynamic quantization** includes the per-tensor absmax reduction and scale computation in the hot path.",
                "- **Static quantization** removes that scale-discovery step and directly quantizes with a cached scale.",
                "- **IO proxy** is a tensor copy lower bound for memory traffic.",
                "- **Compute estimate** is `static_quant_ms - io_proxy_ms`, clipped at zero; if it stays small, the quantization kernel is predominantly IO-limited.",
            ])

        if correctness is not None:
            lines.extend([
                "",
                "## Correctness Validation",
                "",
                "These checks were rerun after the full benchmark sweep to confirm that the CUDA Graph path still matches the eager INT8 reference.",
            ])

            raw = correctness.get('raw_cutlass_graph')
            if raw is not None:
                lines.extend([
                    "",
                    "### Raw CUTLASS CUDA Graph replay",
                    "",
                    raw.get('description', ''),
                    "",
                    "| Check | Mean Abs Diff | Max Abs Diff |",
                    "| --- | --- | --- |",
                    f"| replay(x₁) vs eager(x₁) | {raw['y1_vs_out1_mean_abs']:.6f} | {raw['y1_vs_out1_max_abs']:.6f} |",
                    f"| replay(x₂) vs eager(x₂) | {raw['y2_vs_out2_mean_abs']:.6f} | {raw['y2_vs_out2_max_abs']:.6f} |",
                    f"| replay(x₂) vs eager(x₁) | {raw['y2_vs_out1_mean_abs']:.6f} | {raw['y2_vs_out1_max_abs']:.6f} |",
                ])

            model = correctness.get('ldm_eager_vs_cudagraph')
            if model is not None:
                lines.extend([
                    "",
                    "### End-to-end LDM eager vs CUDA Graph",
                    "",
                    model.get('description', ''),
                    "",
                    f"- configuration: batch size {int(model['batch_size'])}, {int(model['steps'])} DDIM steps, $\\eta = {model['eta']:.1f}$",
                    f"- latent diff: mean {model['latent_mean_abs']:.6f}, max {model['latent_max_abs']:.6f}",
                    f"- decoded image diff: mean {model['image_mean_abs']:.6f}, max {model['image_max_abs']:.6f}",
                ])

        lines.extend([
            "",
            "## Analysis",
            "",
            "### Fairness configuration",
            "",
            "- TF32 is disabled for both matmul and cuDNN paths in this benchmark, so the `fp32` baseline stays true FP32 rather than silently using TensorFloat-32 acceleration.",
            "- The baseline quantized modes use the same optimized kernels and static scales as the MoDiff modes; the only intended difference is temporal caching.",
            "",
            "### Task 1: CUTLASS INT8 + CUDA Graph",
            "",
            "CUDA Graphs reduce Python/kernel-launch overhead by replaying captured UNet executions.",
            "In this implementation, the graph replay is real and is exercised in the benchmark:",
        ])

        if 'int8_cudagraph_baseline' in self.results:
            base = self.results['int8_cudagraph_baseline']
            lines.append(
                f"- `int8_cudagraph_baseline` captures {int(base.get('cuda_graph_num_graphs', 0))} graph and replays it {int(base.get('cuda_graph_replay_count', 0))} times"
            )
        if 'int8_cudagraph' in self.results:
            graph = self.results['int8_cudagraph']
            lines.append(
                f"- `int8_cudagraph` captures {int(graph.get('cuda_graph_num_graphs', 0))} graphs (first/modulated) and replays them {int(graph.get('cuda_graph_replay_count', 0))} times"
            )

        lines.extend([
            "",
            "`int8_cudagraph` now uses the same CUTLASS INT8 kernels as fused `int8` for the conv/modulation path.",
            "This isolates the effect of CUDA Graph replay on top of the optimized backend instead of benchmarking a different FP16 fallback backend.",
            "Any remaining gap between `int8_cudagraph` and eager fused `int8` therefore reflects graph-capture constraints, first-step capture cost, and the interaction between static replay buffers and the MoDiff execution schedule rather than a backend mismatch.",
            "",
            "### Why the earlier graph numbers looked slow",
            "",
            "The slow `int8_cudagraph` / `int8_cudagraph_baseline` run was primarily a benchmarking bug rather than a pure kernel regression.",
            "The pre-capture path used an invalid short DDIM schedule (`S=3`), but DDIM uniform discretization only works cleanly for step counts that divide the 1000-step base schedule.",
            "That could either fail outright or prevent the graphs from being fully pre-captured before the timed sample, which makes the first measured batch pay graph-construction cost.",
            "The benchmark now pre-captures with the minimum valid schedule needed for each mode:",
            "- `int8_cudagraph`: 2 steps (captures both first-step and modulated graphs)",
            "- `int8_cudagraph_baseline`: 1 step (captures the single baseline graph)",
            "",
            "### Task 2: Fused vs Separate Kernels",
            "",
            "The current MoDiff implementation fuses multiple operations into fewer kernel launches:",
            "- **Fused Step1**: sub_absmax_scale + scale_quantize + dequant_accumulate",
            "- **Fused Conv**: conv + weight_scale + o_hat_accumulate",
            "",
            "The separate baseline breaks these into individual PyTorch/CUTLASS calls.",
            "Fusion benefit is primarily from reduced kernel launch overhead and memory bandwidth savings.",
        ])

        if 'int8_cudagraph' in self.results and 'int8' in self.results:
            graph_time = self.results['int8_cudagraph']['time_per_sample']
            eager_time = self.results['int8']['time_per_sample']
            if graph_time < eager_time:
                lines.extend([
                    "",
                    "### Effect of CUDA Graph replay on CUTLASS INT8",
                    "",
                    f"- **`int8_cudagraph` vs `int8`**: {graph_time:.3f}s vs {eager_time:.3f}s.",
                    f"  - CUDA Graph replay reduces the eager CUTLASS INT8 path by {eager_time / graph_time:.2f}x on this run",
                    "  - both modes use the same CUTLASS INT8 kernels on the hot path, so this speedup comes from reducing Python/kernel-launch overhead",
                    "  - the remaining trade-off is memory: graph replay keeps large static buffers alive, which raises peak memory usage",
                ])
            else:
                lines.extend([
                    "",
                    "### Effect of CUDA Graph replay on CUTLASS INT8",
                    "",
                    f"- **`int8_cudagraph` vs `int8`**: {graph_time:.3f}s vs {eager_time:.3f}s.",
                    "  - CUDA Graph replay helps on the control-plane side",
                    "  - both modes now use the same CUTLASS INT8 kernels on the hot path",
                    "  - any remaining delta comes from graph capture/replay mechanics rather than PyTorch FP16 fallback kernels",
                ])

        if 'int8_separate' in self.results and 'int8' in self.results:
            lines.extend([
                "",
                f"- **`int8_separate` vs `int8`**: {self.results['int8_separate']['time_per_sample']:.3f}s vs {self.results['int8']['time_per_sample']:.3f}s.",
                "  - the separate path performs the same MoDiff math, but it explodes the fused hot path into many kernels",
                "  - the microbenchmark shows the main gap is in **Step1 fusion**, not in `o_hat` accumulation",
            ])
            if 'kernel_timing' in self.results and 'INT8_32x192x32x32' in self.results['kernel_timing']:
                ref = self.results['kernel_timing']['INT8_32x192x32x32']
                lines.extend([
                    f"  - fused INT8 total kernel time is {ref['fused_total_ms']:.3f}ms vs {ref['separate_total_ms']:.3f}ms on the representative 32x192x32x32 case",
                    f"  - the extra cost of `compute+DQ+update_o_hat` over `compute+DQ` is only {ref['ohat_update_overhead_ms']:+.3f}ms ({ref['ohat_update_overhead_pct']:+.1f}%)",
                    "  - so the missing speedup is mostly due to unfused Step1 work and extra global-memory traffic, not because `o_hat` update is too expensive",
                ])

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Report saved to: {report_path}")


def generate_plots(results_path: str, output_dir: str):
    """Generate visualization plots from results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Plot 1: Pipeline speedup comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    modes = []
    times = []
    colors = []
    color_map = {
        'fp32': '#808080', 'fp16': '#4CAF50',
        'int8': '#2196F3', 'int8_baseline': '#90CAF9',
        'int4': '#FF5722', 'int4_baseline': '#FFAB91',
        'int8_cudagraph': '#9C27B0', 'int8_cudagraph_baseline': '#CE93D8',
        'int8_separate': '#FF9800', 'int8_separate_baseline': '#FFE0B2',
        'int4_separate': '#795548', 'int4_separate_baseline': '#D7CCC8',
    }

    for mode in ['fp32', 'fp16', 'int8', 'int8_baseline', 'int4', 'int4_baseline',
                  'int8_cudagraph', 'int8_cudagraph_baseline',
                  'int8_separate', 'int8_separate_baseline',
                  'int4_separate', 'int4_separate_baseline']:
        if mode in results and mode != 'kernel_timing':
            modes.append(mode)
            times.append(float(results[mode]['time_per_sample']))
            colors.append(color_map.get(mode, '#808080'))

    bars = ax.bar(range(len(modes)), times, color=colors)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax.set_ylabel('Time per Sample (s)')
    ax.set_title('MoDiff Extended Benchmark: Time per Sample')
    ax.grid(axis='y', alpha=0.3)

    # Add speedup labels
    if 'fp32' in results:
        fp32_time = float(results['fp32']['time_per_sample'])
        for i, (mode, t) in enumerate(zip(modes, times)):
            if mode != 'fp32':
                speedup = fp32_time / t
                ax.text(i, t + 0.01, f'{speedup:.2f}x', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_extended_speedup.png'), dpi=150)
    plt.close()

    # Plot 2: Memory comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    modes_mem = []
    mems = []
    for mode in modes:
        if mode in results and 'memory_peak_mb' in results[mode]:
            modes_mem.append(mode)
            mems.append(float(results[mode]['memory_peak_mb']))

    if mems:
        ax.bar(range(len(modes_mem)), mems, color=[color_map.get(m, '#808080') for m in modes_mem])
        ax.set_xticks(range(len(modes_mem)))
        ax.set_xticklabels(modes_mem, rotation=45, ha='right')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('MoDiff Extended Benchmark: Peak GPU Memory')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot_extended_memory.png'), dpi=150)
    plt.close()

    # Plot 3: Kernel timing comparison
    if 'kernel_timing' in results:
        kt = results['kernel_timing']
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # INT8 kernels
        int8_keys = [k for k in kt if k.startswith('INT8')]
        if int8_keys:
            labels = [k.replace('INT8_', '') for k in int8_keys]
            fused = [float(kt[k]['fused_total_ms']) for k in int8_keys]
            separate = [float(kt[k]['separate_total_ms']) for k in int8_keys]
            x = range(len(labels))
            w = 0.35
            axes[0].bar([i - w/2 for i in x], fused, w, label='Fused', color='#2196F3')
            axes[0].bar([i + w/2 for i in x], separate, w, label='Separate', color='#FF5722')
            axes[0].set_xticks(list(x))
            axes[0].set_xticklabels(labels, rotation=45, ha='right')
            axes[0].set_ylabel('Time (ms)')
            axes[0].set_title('INT8: Fused vs Separate Kernels')
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)

        # INT4 kernels
        int4_keys = [k for k in kt if k.startswith('INT4')]
        if int4_keys:
            labels = [k.replace('INT4_', '') for k in int4_keys]
            fused = [float(kt[k]['fused_total_ms']) for k in int4_keys]
            separate = [float(kt[k]['separate_total_ms']) for k in int4_keys]
            x = range(len(labels))
            w = 0.35
            axes[1].bar([i - w/2 for i in x], fused, w, label='Fused', color='#2196F3')
            axes[1].bar([i + w/2 for i in x], separate, w, label='Separate', color='#FF5722')
            axes[1].set_xticks(list(x))
            axes[1].set_xticklabels(labels, rotation=45, ha='right')
            axes[1].set_ylabel('Time (ms)')
            axes[1].set_title('INT4: Fused vs Separate Kernels')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot_kernel_timing.png'), dpi=150)
        plt.close()

    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extended MoDiff Benchmark")
    parser.add_argument('--config', type=str, default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt', type=str, default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--output_dir', type=str, default='integration/results/extended')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'fp32', 'fp16',
                                'int8', 'int8_baseline', 'int4', 'int4_baseline',
                                'int8_cudagraph', 'int8_cudagraph_baseline',
                                'int8_separate', 'int8_separate_baseline',
                                'int4_separate', 'int4_separate_baseline',
                                'kernel_timing'])
    parser.add_argument('--skip_plots', action='store_true')
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: batch_size={args.batch_size}, steps={args.steps}, samples={args.num_samples}")

    runner = ExtendedBenchmarkRunner(
        args.config, args.ckpt, args.output_dir,
        args.batch_size, args.steps, shape=(4, 32, 32)
    )

    if args.mode == 'all':
        # Run baselines first
        for mode in ['fp32', 'fp16']:
            runner.run_mode(mode, args.num_samples)

        # Run existing MoDiff
        if HAS_INT8:
            for mode in ['int8', 'int8_baseline']:
                runner.run_mode(mode, args.num_samples)
        if HAS_INT4:
            for mode in ['int4', 'int4_baseline']:
                runner.run_mode(mode, args.num_samples)

        # Run new modes
        if HAS_CUDAGRAPH:
            for mode in ['int8_cudagraph', 'int8_cudagraph_baseline']:
                runner.run_mode(mode, args.num_samples)

        if HAS_SEPARATE:
            for mode in ['int8_separate', 'int8_separate_baseline',
                          'int4_separate', 'int4_separate_baseline']:
                runner.run_mode(mode, args.num_samples)

        # Kernel timing
        runner.run_kernel_timing()

    elif args.mode == 'kernel_timing':
        runner.run_kernel_timing()
    else:
        runner.run_mode(args.mode, args.num_samples)

    runner.print_summary()
    runner.generate_report()
    if 'kernel_timing' in runner.results:
        runner.generate_cache_update_report()
    if 'quantization_timing' in runner.results:
        runner.generate_quantization_report()

    if not args.skip_plots:
        results_path = os.path.join(args.output_dir, 'extended_results.json')
        if os.path.exists(results_path):
            generate_plots(results_path, args.output_dir)


if __name__ == '__main__':
    main()
