"""Static vs Dynamic quantization benchmark for baseline and MoDiff modes.

This benchmark compares the available full-model quantization configurations for
both INT8 and INT4:
    - baseline + dynamic quantization
    - baseline + static quantization
    - MoDiff + dynamic quantization
    - MoDiff + static quantization

For each mode it records:
    - sampling time (decode/save excluded from timed region)
    - GPU memory after setup and peak during generation
    - deterministic quality comparison images using the same initial latent noise

Notes
-----
The current MoDiff implementation keeps residual error-compensation quantization
on a dynamic path inside the temporal caching step. Static calibration still
applies to the standard/first-step forward path and linear layers, so the
"static MoDiff" modes here reflect the actual static-capable MoDiff variant in
this repository.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
import types
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torchvision.utils as tvu
from omegaconf import OmegaConf

warnings.filterwarnings('ignore', message='Could not initialize NNPACK')
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.getcwd())

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from integration.kernels.int8_optimized import (
    convert_model_to_optimized_int8,
    enable_modiff_mode as enable_modiff_mode_int8,
    reset_modiff_state as reset_modiff_state_int8,
    set_calibrating as set_calibrating_int8,
    reset_calibration as reset_calibration_int8,
    get_calibration_config as get_calibration_config_int8,
    export_int8_static_scales,
    apply_static_scales,
)
from integration.kernels.int8_linear import (
    convert_model_to_int8_linear,
    enable_modiff_mode_linear,
    reset_modiff_state_linear,
    set_calibrating_linear,
    export_linear_static_scales,
    apply_linear_static_scales,
)
from integration.kernels.int4_optimized import (
    convert_model_to_optimized_int4,
    enable_modiff_mode as enable_modiff_mode_int4,
    reset_modiff_state as reset_modiff_state_int4,
    set_calibrating_int4,
    export_int4_static_scales,
    apply_int4_static_scales,
)
from integration.kernels.int4_linear import (
    convert_model_to_int4_linear,
    enable_modiff_mode_int4_linear,
    reset_modiff_state_int4_linear,
    set_calibrating_int4_linear,
    export_int4_linear_static_scales,
    apply_int4_linear_static_scales,
)
from integration.utils.buffer_pool import initialize_buffer_pool
from integration.fused_ops.fused_resblock import fuse_resblocks_in_module, print_fusion_summary


@dataclass(frozen=True)
class ModeSpec:
    precision: str  # int8 | int4
    static: bool
    modiff: bool

    @property
    def name(self) -> str:
        return f"{self.precision}_{'static' if self.static else 'dynamic'}_{'modiff' if self.modiff else 'baseline'}"

    @property
    def label(self) -> str:
        return f"{self.precision.upper()} {'static' if self.static else 'dynamic'} {'MoDiff' if self.modiff else 'baseline'}"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config_path: str, ckpt_path: str):
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = pl_sd.get('state_dict', pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(state_dict, strict=False)
    return model.cuda().eval(), conf


def measure_gpu_memory() -> Dict[str, float]:
    torch.cuda.synchronize()
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def decode_latents(model, latents: torch.Tensor, use_autocast: bool, dtype: Optional[torch.dtype], chunk_size: int = 8) -> torch.Tensor:
    decoded_chunks = []
    for start in range(0, latents.shape[0], chunk_size):
        end = min(start + chunk_size, latents.shape[0])
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
            decoded = model.decode_first_stage(latents[start:end])
        decoded = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0).cpu()
        decoded_chunks.append(decoded)
    return torch.cat(decoded_chunks, dim=0)


class StaticDynamicBenchmark:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        output_dir: str,
        batch_size: int,
        steps: int,
        num_samples: int,
        quality_samples: int,
        calibration_steps: int,
        calibration_runs: int,
        timing_repeats: int,
        seed: int,
    ):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.steps = steps
        self.num_samples = num_samples
        self.quality_samples = quality_samples
        self.calibration_steps = calibration_steps
        self.calibration_runs = calibration_runs
        self.timing_repeats = timing_repeats
        self.seed = seed
        self.shape = (4, 32, 32)
        self.results: Dict[str, Dict[str, float | int | str | bool]] = {}
        self.calibration_cache: Dict[Tuple[str, bool], Dict[str, float]] = {}
        self.mode_order = [
            ModeSpec('int8', False, False),
            ModeSpec('int8', True, False),
            ModeSpec('int8', False, True),
            ModeSpec('int8', True, True),
            ModeSpec('int4', False, False),
            ModeSpec('int4', True, False),
            ModeSpec('int4', False, True),
            ModeSpec('int4', True, True),
        ]
        self.selected_precisions = ['int8', 'int4']

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self._samples_root(), exist_ok=True)
        os.makedirs(self._quality_root(), exist_ok=True)
        os.makedirs(self._calibration_root(), exist_ok=True)

    def _mode_seed(self, spec: ModeSpec, offset: int = 0) -> int:
        workload_key = f"{spec.precision}_{'modiff' if spec.modiff else 'baseline'}"
        key = sum(ord(ch) for ch in workload_key)
        return self.seed + key + offset

    def _make_timed_latents(self, spec: ModeSpec) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        generator = torch.Generator(device='cuda')
        generator.manual_seed(self._mode_seed(spec, offset=1000))
        warmup = torch.randn((self.batch_size, *self.shape), device='cuda', generator=generator)

        timed_batches: List[torch.Tensor] = []
        generated = 0
        while generated < self.num_samples:
            batch = min(self.batch_size, self.num_samples - generated)
            timed_batches.append(torch.randn((batch, *self.shape), device='cuda', generator=generator))
            generated += batch
        return warmup, timed_batches

    def _samples_root(self) -> str:
        return os.path.join(self.output_dir, 'samples')

    def _quality_root(self) -> str:
        return os.path.join(self.output_dir, 'quality')

    def _calibration_root(self) -> str:
        return os.path.join(self.output_dir, 'calibration')

    def _prepare_model(self, model):
        model = model.to(memory_format=torch.channels_last)
        for module in model.modules():
            if hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = False
        from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
        AttentionBlock.forward = lambda self, x: self._forward(x)
        fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)
        return model

    def _reset_quant_state(self, unet, precision: str):
        if precision == 'int8':
            reset_modiff_state_int8(unet)
            reset_modiff_state_linear(unet)
        else:
            reset_modiff_state_int4(unet)
            reset_modiff_state_int4_linear(unet)

    def _set_modiff(self, unet, precision: str, enabled: bool):
        if precision == 'int8':
            enable_modiff_mode_int8(unet, enabled)
            enable_modiff_mode_linear(unet, enabled)
        else:
            enable_modiff_mode_int4(unet, enabled)
            enable_modiff_mode_int4_linear(unet, enabled)

    def _apply_scales(self, unet, precision: str, scales: Dict[str, float]) -> Tuple[int, int]:
        if precision == 'int8':
            conv_loaded = apply_static_scales(unet, scales)
            linear_scales = {k.replace('linear:', ''): v for k, v in scales.items() if k.startswith('linear:')}
            linear_loaded = apply_linear_static_scales(unet, linear_scales) if linear_scales else 0
            return conv_loaded, linear_loaded

        conv_loaded = apply_int4_static_scales(unet, scales)
        linear_scales = {k.replace('linear:', ''): v for k, v in scales.items() if k.startswith('linear:')}
        linear_loaded = apply_int4_linear_static_scales(unet, linear_scales) if linear_scales else 0
        return conv_loaded, linear_loaded

    def _collect_scales(self, unet, precision: str) -> Dict[str, float]:
        if precision == 'int8':
            scales = export_int8_static_scales(unet)
            linear_scales = export_linear_static_scales(unet)
        else:
            scales = export_int4_static_scales(unet)
            linear_scales = export_int4_linear_static_scales(unet)
        merged = dict(scales)
        for key, value in linear_scales.items():
            merged[f'linear:{key}'] = value
        return merged

    def _run_calibration(self, spec: ModeSpec) -> Dict[str, float]:
        cache_key = (spec.precision, spec.modiff)
        if cache_key in self.calibration_cache:
            return self.calibration_cache[cache_key]

        print(f"\n[calibration] {spec.label}")
        model, _ = load_model(self.config_path, self.ckpt_path)
        model = self._prepare_model(model)
        unet = model.model.diffusion_model

        if spec.precision == 'int8':
            convert_model_to_optimized_int8(unet)
            convert_model_to_int8_linear(unet)
        else:
            convert_model_to_optimized_int4(unet)
            convert_model_to_int4_linear(unet)

        initialize_buffer_pool(unet, max_batch_size=max(self.batch_size, self.quality_samples), device='cuda')
        self._set_modiff(unet, spec.precision, spec.modiff)
        sampler = DDIMSampler(model)

        cal_batch = min(self.batch_size, 8)
        cal_steps = min(self.calibration_steps, self.steps)

        if spec.precision == 'int8':
            observed_absmax: Dict[str, float] = {}
            original_scale_fns = {}
            reset_calibration_int8()
            set_calibrating_linear(unet, True)

            for module in unet.modules():
                if hasattr(module, 'layer_name') and hasattr(module, '_compute_scale_tensor'):
                    original_scale_fns[module.layer_name] = module._compute_scale_tensor

                    def wrapped_compute_scale_tensor(self, x, _orig=module._compute_scale_tensor, _seen=observed_absmax):
                        abs_max = float(x.detach().abs().amax().item())
                        current = _seen.get(self.layer_name, 0.0)
                        if abs_max > current:
                            _seen[self.layer_name] = abs_max
                        return _orig(x)

                    module._compute_scale_tensor = types.MethodType(wrapped_compute_scale_tensor, module)
        else:
            set_calibrating_int4(unet, True)
            set_calibrating_int4_linear(unet, True)

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            for _ in range(self.calibration_runs):
                self._reset_quant_state(unet, spec.precision)
                sampler.sample(S=cal_steps, batch_size=cal_batch, shape=self.shape, eta=0.0, verbose=False)

        if spec.precision == 'int8':
            for module in unet.modules():
                if hasattr(module, 'layer_name') and module.layer_name in original_scale_fns:
                    module._compute_scale_tensor = original_scale_fns[module.layer_name]
            set_calibrating_linear(unet, False)
            get_calibration_config_int8().finalize()
        else:
            set_calibrating_int4(unet, False)
            set_calibrating_int4_linear(unet, False)

        scales = self._collect_scales(unet, spec.precision)
        if spec.precision == 'int8':
            conv_scales = {name: 127.0 / max(value, 1e-6) for name, value in observed_absmax.items()}
            scales.update(conv_scales)
        calib_path = os.path.join(self._calibration_root(), f"{spec.name}_scales.pt")
        torch.save(scales, calib_path)
        print(f"  collected {len(scales)} total scales -> {calib_path}")

        self.calibration_cache[cache_key] = scales
        del model, sampler
        torch.cuda.empty_cache()
        gc.collect()
        return scales

    def _build_mode(self, spec: ModeSpec):
        model, _ = load_model(self.config_path, self.ckpt_path)
        model = self._prepare_model(model)
        unet = model.model.diffusion_model

        if spec.precision == 'int8':
            convert_model_to_optimized_int8(unet)
            convert_model_to_int8_linear(unet)
        else:
            convert_model_to_optimized_int4(unet)
            convert_model_to_int4_linear(unet)

        initialize_buffer_pool(unet, max_batch_size=max(self.batch_size, self.quality_samples), device='cuda')
        self._set_modiff(unet, spec.precision, spec.modiff)

        conv_loaded = 0
        linear_loaded = 0
        if spec.static:
            scales = self._run_calibration(spec)
            conv_loaded, linear_loaded = self._apply_scales(unet, spec.precision, scales)

        sampler = DDIMSampler(model)
        return model, sampler, conv_loaded, linear_loaded

    def _save_samples(self, decoded: torch.Tensor, mode_dir: str, start_index: int):
        for local_idx, image in enumerate(decoded):
            tvu.save_image(image, os.path.join(mode_dir, f'{start_index + local_idx:05d}.png'))

    def run_mode(self, spec: ModeSpec):
        print(f"\n{'=' * 80}\n{spec.label}\n{'=' * 80}")
        mode_dir = os.path.join(self._samples_root(), spec.name)
        if os.path.exists(mode_dir):
            shutil.rmtree(mode_dir)
        os.makedirs(mode_dir, exist_ok=True)

        torch.cuda.empty_cache()
        gc.collect()
        model, sampler, conv_loaded, linear_loaded = self._build_mode(spec)
        warmup_latents, timed_latents = self._make_timed_latents(spec)
        mem_after_setup = measure_gpu_memory()
        torch.cuda.reset_peak_memory_stats()

        use_autocast = True
        dtype = torch.float16

        print(f"  warmup: {self.steps} steps @ batch {self.batch_size}")
        self._reset_quant_state(model.model.diffusion_model, spec.precision)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=dtype):
            sampler.sample(
                S=self.steps,
                batch_size=self.batch_size,
                shape=self.shape,
                eta=0.0,
                verbose=False,
                x_T=warmup_latents.clone(),
            )
        torch.cuda.synchronize()

        repeat_totals: List[float] = []
        final_repeat_samples: List[torch.Tensor] = []
        for repeat_idx in range(self.timing_repeats):
            total_time = 0.0
            repeat_samples: List[torch.Tensor] = []
            generated = 0
            for latent_batch in timed_latents:
                batch = latent_batch.shape[0]
                self._reset_quant_state(model.model.diffusion_model, spec.precision)

                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=dtype):
                    samples, _ = sampler.sample(
                        S=self.steps,
                        batch_size=batch,
                        shape=self.shape,
                        eta=0.0,
                        verbose=False,
                        x_T=latent_batch.clone(),
                    )
                torch.cuda.synchronize()
                total_time += time.time() - t0
                if repeat_idx == self.timing_repeats - 1:
                    repeat_samples.append(samples.detach().cpu())
                generated += batch

            repeat_totals.append(total_time)
            if repeat_idx == self.timing_repeats - 1:
                final_repeat_samples = repeat_samples

        total_time = float(np.mean(repeat_totals))
        time_std = float(np.std(repeat_totals))
        generated = sum(batch.shape[0] for batch in timed_latents)

        save_index = 0
        for sample_batch in final_repeat_samples:
            decoded = decode_latents(model, sample_batch.cuda(non_blocking=True), use_autocast=True, dtype=dtype)
            self._save_samples(decoded, mode_dir, save_index)
            save_index += decoded.shape[0]

        mem_peak = measure_gpu_memory()
        result = {
            'mode': spec.name,
            'label': spec.label,
            'precision': spec.precision,
            'static': spec.static,
            'modiff': spec.modiff,
            'num_samples': generated,
            'steps': self.steps,
            'batch_size': self.batch_size,
            'total_time_s': total_time,
            'repeat_times_s': repeat_totals,
            'timing_std_s': time_std,
            'time_per_sample_s': total_time / generated,
            'time_per_step_ms': total_time / (generated * self.steps) * 1000.0,
            'memory_allocated_mb': mem_after_setup['allocated_mb'],
            'memory_peak_mb': mem_peak['max_allocated_mb'],
            'loaded_conv_scales': conv_loaded,
            'loaded_linear_scales': linear_loaded,
        }
        self.results[spec.name] = result

        print(f"  total: {result['total_time_s']:.2f}s for {generated} samples")
        if self.timing_repeats > 1:
            print(f"  repeats: {', '.join(f'{value:.2f}s' for value in repeat_totals)} (std {time_std:.2f}s)")
        print(f"  per-sample: {result['time_per_sample_s']:.3f}s")
        print(f"  per-step: {result['time_per_step_ms']:.2f}ms")
        print(f"  memory: {result['memory_allocated_mb']:.0f}MB allocated, {result['memory_peak_mb']:.0f}MB peak")
        if spec.static:
            print(f"  static scales loaded: conv={conv_loaded}, linear={linear_loaded}")
        else:
            print("  static scales loaded: conv=0, linear=0 (dynamic path)")

        del model, sampler
        torch.cuda.empty_cache()
        gc.collect()

    def _generate_fp32_reference(self, x_T: torch.Tensor) -> torch.Tensor:
        model, _ = load_model(self.config_path, self.ckpt_path)
        model = self._prepare_model(model)
        sampler = DDIMSampler(model)
        with torch.inference_mode():
            samples, _ = sampler.sample(S=self.steps, batch_size=x_T.shape[0], shape=self.shape, eta=0.0, verbose=False, x_T=x_T.clone())
        decoded = decode_latents(model, samples, use_autocast=False, dtype=None)
        del model, sampler
        torch.cuda.empty_cache()
        gc.collect()
        return decoded

    def _render_quality_figure(self, precision: str, rows: List[Tuple[str, torch.Tensor]], metric_lines: Dict[str, str]) -> str:
        from PIL import Image, ImageDraw, ImageFont

        nrows = len(rows)
        ncols = rows[0][1].shape[0]
        sample_image = rows[0][1][0]
        cell_h = int(sample_image.shape[1])
        cell_w = int(sample_image.shape[2])
        top_margin = 70
        left_margin = 320
        row_gap = 20
        canvas_w = left_margin + ncols * cell_w
        canvas_h = top_margin + nrows * cell_h + max(0, nrows - 1) * row_gap

        canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        title = f'{precision.upper()} static vs dynamic quality comparison (same initial noise)'
        draw.text((20, 15), title, fill='black', font=font)
        for col_idx in range(ncols):
            draw.text((left_margin + col_idx * cell_w + 8, 45), f'Sample {col_idx}', fill='black', font=font)

        for row_idx, (row_label, images) in enumerate(rows):
            y = top_margin + row_idx * (cell_h + row_gap)
            metric_suffix = metric_lines.get(row_label, '')
            label = row_label if not metric_suffix else f"{row_label}\n{metric_suffix}"
            draw.multiline_text((15, y + 8), label, fill='black', font=font, spacing=4)

            for col_idx in range(ncols):
                img = (images[col_idx].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
                pil_img = Image.fromarray(img)
                x = left_margin + col_idx * cell_w
                canvas.paste(pil_img, (x, y))

        out_path = os.path.join(self._quality_root(), f'{precision}_quality_comparison.png')
        canvas.save(out_path)
        return out_path

    def run_quality_comparison(self):
        print(f"\n{'=' * 80}\nQUALITY COMPARISON\n{'=' * 80}")
        set_seed(self.seed)
        x_T = torch.randn((self.quality_samples, *self.shape), device='cuda')
        fp32_reference = self._generate_fp32_reference(x_T)
        quality_summary: Dict[str, Dict[str, Dict[str, float | str]]] = {}

        for precision in self.selected_precisions:
            rows: List[Tuple[str, torch.Tensor]] = [('FP32 reference', fp32_reference)]
            metric_lines: Dict[str, str] = {}
            precision_summary: Dict[str, Dict[str, float | str]] = {}

            relevant_modes = [spec for spec in self.mode_order if spec.precision == precision]
            for spec in relevant_modes:
                model, sampler, conv_loaded, linear_loaded = self._build_mode(spec)
                self._reset_quant_state(model.model.diffusion_model, spec.precision)
                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                    samples, _ = sampler.sample(
                        S=self.steps,
                        batch_size=self.quality_samples,
                        shape=self.shape,
                        eta=0.0,
                        verbose=False,
                        x_T=x_T.clone(),
                    )
                decoded = decode_latents(model, samples, use_autocast=True, dtype=torch.float16)
                rows.append((spec.label, decoded))

                mae = float((decoded - fp32_reference).abs().mean().item())
                max_abs = float((decoded - fp32_reference).abs().max().item())
                mse = float(torch.mean((decoded - fp32_reference) ** 2).item())
                psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
                precision_summary[spec.name] = {
                    'label': spec.label,
                    'mae_vs_fp32': mae,
                    'max_abs_vs_fp32': max_abs,
                    'psnr_vs_fp32_db': float(psnr),
                    'loaded_conv_scales': conv_loaded,
                    'loaded_linear_scales': linear_loaded,
                }
                metric_lines[spec.label] = f"MAE {mae:.4f} | PSNR {psnr:.2f}dB"

                row_dir = os.path.join(self._quality_root(), precision, spec.name)
                os.makedirs(row_dir, exist_ok=True)
                self._save_samples(decoded, row_dir, 0)

                del model, sampler
                torch.cuda.empty_cache()
                gc.collect()

            fig_path = self._render_quality_figure(precision, rows, metric_lines)
            precision_summary['figure_path'] = fig_path
            quality_summary[precision] = precision_summary
            print(f"  {precision.upper()} quality figure -> {fig_path}")

        with open(os.path.join(self.output_dir, 'quality_summary.json'), 'w') as f:
            json.dump(quality_summary, f, indent=2)
        return quality_summary

    def write_report(self, quality_summary: Dict[str, Dict[str, Dict[str, float | str]]]):
        report_path = os.path.join(self.output_dir, 'STATIC_DYNAMIC_BENCHMARK_REPORT.md')
        lines = [
            '# Static vs Dynamic Baseline and MoDiff Benchmark Report',
            '',
            f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'**GPU**: {torch.cuda.get_device_name()}',
            f'**Batch Size**: {self.batch_size}',
            f'**Timesteps**: {self.steps}',
            f'**Timed Samples per Mode**: {self.num_samples}',
            f'**Quality Samples per Mode**: {self.quality_samples}',
            '',
            '## Progress and debugging notes',
            '',
            '- Built a dedicated benchmark to compare static vs dynamic quantization for both baseline and MoDiff variants.',
            '- Generated fresh per-experiment calibration scales for all static modes instead of reusing shared legacy calibration files.',
            '- Debug note: the current INT8 conv path does not populate built-in conv calibration scales on its own because it uses the GPU-only scale path; this benchmark compensates with an explicit hook-based INT8 conv-scale calibration pass so the static INT8 rows are real static runs.',
            '- Kept decode/save outside the timed region so the numbers isolate denoising throughput rather than PNG I/O.',
            '- Used the same initial latent noise (`x_T`) for all quality comparisons so visual differences come from the quantization mode rather than random sampling drift.',
            '- Important implementation detail: the current MoDiff path in this repo keeps residual error compensation dynamically quantized by design. The `static` MoDiff rows therefore represent the repository\'s actual static-capable MoDiff configuration: static calibrated standard/first-step path + static linear scales, while residual compensation stays dynamic.',
            '',
            '## Timing and memory results',
            '',
            '| Mode | Time / sample (s) | Time / step (ms) | Allocated after setup (MB) | Peak memory (MB) | Loaded conv scales | Loaded linear scales |',
            '| --- | --- | --- | --- | --- | --- | --- |',
        ]

        for spec in self.mode_order:
            result = self.results[spec.name]
            lines.append(
                f"| {result['label']} | {result['time_per_sample_s']:.3f} | {result['time_per_step_ms']:.2f} | "
                f"{result['memory_allocated_mb']:.0f} | {result['memory_peak_mb']:.0f} | "
                f"{int(result['loaded_conv_scales'])} | {int(result['loaded_linear_scales'])} |"
            )

        for precision in self.selected_precisions:
            lines.extend([
                '',
                f'## {precision.upper()} static vs dynamic summary',
                '',
            ])
            baseline_dynamic = self.results[f'{precision}_dynamic_baseline']
            baseline_static = self.results[f'{precision}_static_baseline']
            modiff_dynamic = self.results[f'{precision}_dynamic_modiff']
            modiff_static = self.results[f'{precision}_static_modiff']

            baseline_speedup = baseline_dynamic['time_per_sample_s'] / baseline_static['time_per_sample_s']
            modiff_speedup = modiff_dynamic['time_per_sample_s'] / modiff_static['time_per_sample_s']
            baseline_mem_delta = baseline_static['memory_peak_mb'] - baseline_dynamic['memory_peak_mb']
            modiff_mem_delta = modiff_static['memory_peak_mb'] - modiff_dynamic['memory_peak_mb']

            lines.extend([
                f"- Baseline static vs dynamic speedup: **{baseline_speedup:.2f}x** ({baseline_dynamic['time_per_sample_s']:.3f}s → {baseline_static['time_per_sample_s']:.3f}s).",
                f"- MoDiff static vs dynamic speedup: **{modiff_speedup:.2f}x** ({modiff_dynamic['time_per_sample_s']:.3f}s → {modiff_static['time_per_sample_s']:.3f}s).",
                f"- Baseline peak-memory delta (static - dynamic): **{baseline_mem_delta:+.0f} MB**.",
                f"- MoDiff peak-memory delta (static - dynamic): **{modiff_mem_delta:+.0f} MB**.",
            ])

            if self.timing_repeats > 1:
                lines.extend([
                    f"- Baseline timing repeat std-dev: **{baseline_dynamic['timing_std_s']:.2f}s** dynamic vs **{baseline_static['timing_std_s']:.2f}s** static.",
                    f"- MoDiff timing repeat std-dev: **{modiff_dynamic['timing_std_s']:.2f}s** dynamic vs **{modiff_static['timing_std_s']:.2f}s** static.",
                    '- Timed runs reuse the same pre-generated initial latents (`x_T`) across compared modes so static vs dynamic timing is measured on identical denoising workloads.',
                ])

            q = quality_summary[precision]
            lines.extend([
                '',
                f'### {precision.upper()} image-quality comparison against FP32',
                '',
                '| Mode | MAE vs FP32 | Max abs diff | PSNR vs FP32 (dB) |',
                '| --- | --- | --- | --- |',
            ])
            for mode_key in [
                f'{precision}_dynamic_baseline',
                f'{precision}_static_baseline',
                f'{precision}_dynamic_modiff',
                f'{precision}_static_modiff',
            ]:
                q_row = q[mode_key]
                lines.append(
                    f"| {q_row['label']} | {q_row['mae_vs_fp32']:.4f} | {q_row['max_abs_vs_fp32']:.4f} | {q_row['psnr_vs_fp32_db']:.2f} |"
                )

            lines.extend([
                '',
                f'Quality figure: `{q["figure_path"]}`',
                '',
                '### Visual inspection notes',
                '',
                '_Pending manual visual review._',
            ])

        lines.extend([
            '',
            '## Initial conclusions',
            '',
            '- Static calibration should reduce per-sample time whenever the benchmark can reuse loaded activation scales instead of recomputing them at runtime.',
            '- The static-vs-dynamic gain is expected to be strongest in the baseline path because that path can fully replace repeated activation-scale discovery with cached scales.',
            '- MoDiff quality should remain closer to FP32 than the baseline variants because temporal residual compensation is still active.',
        ])

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        return report_path

    def save_results(self):
        with open(os.path.join(self.output_dir, 'static_dynamic_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)

    def run(self, precisions: List[str]):
        selected = [spec for spec in self.mode_order if spec.precision in precisions]
        self.mode_order = selected
        self.selected_precisions = precisions
        for spec in selected:
            self.run_mode(spec)
        self.save_results()
        quality_summary = self.run_quality_comparison()
        report_path = self.write_report(quality_summary)
        print(f"\nReport written to: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Static vs dynamic quantization benchmark for baseline and MoDiff modes')
    parser.add_argument('--config', type=str, default='configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml')
    parser.add_argument('--ckpt', type=str, default='models/ldm/lsun_churches256/model.ckpt')
    parser.add_argument('--output_dir', type=str, default='integration/results/static_dynamic')
    parser.add_argument('--precision', type=str, choices=['all', 'int8', 'int4'], default='all')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--quality_samples', type=int, default=4)
    parser.add_argument('--calibration_steps', type=int, default=20)
    parser.add_argument('--calibration_runs', type=int, default=3)
    parser.add_argument('--timing_repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260319)
    args = parser.parse_args()

    precisions = ['int8', 'int4'] if args.precision == 'all' else [args.precision]
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Precisions: {', '.join(precisions)}")
    print(f"Config: steps={args.steps}, batch_size={args.batch_size}, num_samples={args.num_samples}, quality_samples={args.quality_samples}")

    bench = StaticDynamicBenchmark(
        config_path=args.config,
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        steps=args.steps,
        num_samples=args.num_samples,
        quality_samples=args.quality_samples,
        calibration_steps=args.calibration_steps,
        calibration_runs=args.calibration_runs,
        timing_repeats=args.timing_repeats,
        seed=args.seed,
    )
    bench.run(precisions)


if __name__ == '__main__':
    main()
