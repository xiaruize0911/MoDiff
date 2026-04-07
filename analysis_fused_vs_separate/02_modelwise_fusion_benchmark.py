#!/usr/bin/env python3
"""Model-wise fused-vs-separate MoDiff benchmark.

This script benchmarks the full LSUN-Churches LDM sampling pipeline using the
repository's real MoDiff implementations:

- fused INT8 / INT4 (`OptimizedInt8Conv2d`, `OptimizedInt4Conv2d`)
- separate INT8 / INT4 (`SeparateKernelInt8Conv2d`, `SeparateKernelInt4Conv2d`)

The timed region covers the full DDIM denoising call, excludes decode / image
save, uses warmup runs plus timed iterations × repeats, and resets MoDiff state
before every timed call so runs start from the same cache state.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import random
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", message="Could not initialize NNPACK")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from integration.fused_ops.fused_resblock import fuse_resblocks_in_module
from integration.kernels.int8_optimized import (
    apply_static_scales,
    convert_model_to_optimized_int8,
    enable_modiff_mode as enable_modiff_mode_int8,
    export_int8_static_scales,
    reset_modiff_state as reset_modiff_state_int8,
    set_calibrating as set_calibrating_int8,
)
from integration.kernels.int4_optimized import (
    apply_int4_static_scales,
    convert_model_to_optimized_int4,
    enable_modiff_mode as enable_modiff_mode_int4,
    export_int4_static_scales,
    reset_modiff_state as reset_modiff_state_int4,
    set_calibrating_int4,
)
from integration.kernels.fused_baseline import (
    apply_separate_int8_scales,
    apply_separate_int4_scales,
    convert_model_to_separate_int8,
    convert_model_to_separate_int4,
    enable_modiff_mode_separate_int8,
    enable_modiff_mode_separate_int4,
    reset_modiff_state_separate_int8,
    reset_modiff_state_separate_int4,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def ensure_working_conv_backend() -> str:
    """Return the active convolution backend, falling back if cuDNN is broken."""
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
        return "cuda-no-cudnn"

    try:
        x = torch.randn(1, 8, 8, 8, device="cuda", dtype=torch.float32)
        conv = torch.nn.Conv2d(8, 8, 3, padding=1).cuda().eval()
        with torch.inference_mode():
            _ = conv(x)
        del x, conv
        torch.cuda.synchronize()
        return "cudnn"
    except RuntimeError as exc:
        message = str(exc)
        if "CUDNN_STATUS_NOT_INITIALIZED" not in message:
            raise
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print("[warn] cuDNN initialization failed; falling back to non-cuDNN CUDA convolutions.")
        return "cuda-no-cudnn"


@dataclass(frozen=True)
class ModeSpec:
    precision: str  # int8 | int4
    implementation: str  # fused | separate

    @property
    def name(self) -> str:
        return f"{self.precision}_{self.implementation}_modiff"

    @property
    def label(self) -> str:
        return f"{self.precision.upper()} {'fused' if self.implementation == 'fused' else 'separate'} MoDiff"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config_path: str, ckpt_path: str):
    conf = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(conf.model)
    model.load_state_dict(state_dict, strict=False)
    return model.cuda().eval(), conf


def measure_gpu_memory() -> Dict[str, float]:
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def release_calibration_only_buffers(model) -> float:
    """Drop calibration-only fused buffers so inference memory is measured fairly.

    Optimized fused conv modules keep a full-precision `_orig_weight` clone to
    support later SmoothQuant calibration. The benchmark never calibrates after
    model construction, so those clones are dead weight and can distort memory
    comparisons against the separate baseline by ~1.8 GB on this UNet.

    Returns the approximate amount of GPU memory released in MB.
    """
    released_bytes = 0
    for module in model.modules():
        if hasattr(module, "_orig_weight"):
            orig_weight = getattr(module, "_orig_weight")
            if orig_weight is not None:
                released_bytes += orig_weight.numel() * orig_weight.element_size()
                module._orig_weight = None
        if hasattr(module, "_act_channel_max"):
            module._act_channel_max = None
    return released_bytes / 1024.0 / 1024.0


@contextlib.contextmanager
def suppress_sampler_output(enabled: bool = True):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class ModelwiseFusionBenchmark:
    def __init__(
        self,
        *,
        config_path: str,
        ckpt_path: str,
        output_dir: str,
        batch_size: int,
        steps: int,
        timing_iterations: int,
        timing_repeats: int,
        warmup_runs: int,
        seed: int,
        quant_mode: str,
        static_calibration_runs: int,
        int8_scale_path: Optional[str],
        int4_scale_path: Optional[str],
    ):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.steps = steps
        self.timing_iterations = timing_iterations
        self.timing_repeats = timing_repeats
        self.warmup_runs = warmup_runs
        self.seed = seed
        self.quant_mode = quant_mode
        self.static_calibration_runs = static_calibration_runs
        self.shape = (4, 32, 32)
        self.int8_scale_path = int8_scale_path
        self.int4_scale_path = int4_scale_path
        self.results: Dict[str, Dict[str, object]] = {}
        self._static_scale_cache: Dict[str, Dict[str, object]] = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare_model(self, model):
        model = model.to(memory_format=torch.channels_last)
        for module in model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = False
        from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
        AttentionBlock.forward = lambda self, x: self._forward(x)
        fuse_resblocks_in_module(model.model.diffusion_model, inplace=True)
        return model

    def _scale_path_for_precision(self, precision: str) -> Optional[str]:
        candidate = self.int8_scale_path if precision == "int8" else self.int4_scale_path
        if candidate is None:
            return None
        return candidate if os.path.isabs(candidate) else os.path.join(REPO_ROOT, candidate)

    def _convert_unet(self, spec: ModeSpec, unet) -> None:
        if spec.precision == "int8":
            if spec.implementation == "fused":
                convert_model_to_optimized_int8(unet)
                enable_modiff_mode_int8(unet, True)
            else:
                convert_model_to_separate_int8(unet)
                enable_modiff_mode_separate_int8(unet, True)
        else:
            if spec.implementation == "fused":
                convert_model_to_optimized_int4(unet)
                enable_modiff_mode_int4(unet, True)
            else:
                convert_model_to_separate_int4(unet)
                enable_modiff_mode_separate_int4(unet, True)

    def _set_calibrating_for_precision(self, precision: str, unet, calibrating: bool) -> None:
        if precision == "int8":
            set_calibrating_int8(unet, calibrating)
        else:
            set_calibrating_int4(unet, calibrating)

    def _export_static_scales(self, precision: str, unet) -> Dict[str, float]:
        if precision == "int8":
            return export_int8_static_scales(unet)
        return export_int4_static_scales(unet)

    def _probe_matching_scale_count(self, precision: str, scales: Dict[str, float]) -> int:
        probe_spec = ModeSpec(precision, "fused")
        model, _ = load_model(self.config_path, self.ckpt_path)
        model = self._prepare_model(model)
        unet = model.model.diffusion_model
        self._convert_unet(probe_spec, unet)
        layer_names = {
            module.layer_name
            for module in unet.modules()
            if hasattr(module, "layer_name") and getattr(module, "layer_name", "")
        }
        match_count = sum(1 for key in scales if key in layer_names)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return match_count

    def _make_calibration_latents(self) -> List[torch.Tensor]:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(self.seed + 101)
        return [
            torch.randn((self.batch_size, *self.shape), device="cuda", generator=generator)
            for _ in range(max(1, self.static_calibration_runs))
        ]

    def _generate_static_scales(self, precision: str) -> Tuple[Dict[str, float], str]:
        print(f"  calibrating {precision.upper()} static scales from representative DDIM samples...")
        calibration_spec = ModeSpec(precision, "fused")
        model, _ = load_model(self.config_path, self.ckpt_path)
        model = self._prepare_model(model)
        unet = model.model.diffusion_model
        self._convert_unet(calibration_spec, unet)

        if precision == "int8":
            enable_modiff_mode_int8(unet, False)
        else:
            enable_modiff_mode_int4(unet, False)

        sampler = DDIMSampler(model)
        calibration_latents = self._make_calibration_latents()
        self._set_calibrating_for_precision(precision, unet, True)
        try:
            for index, latent in enumerate(calibration_latents, start=1):
                print(f"    calibration run {index}/{len(calibration_latents)}")
                with suppress_sampler_output(), torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                    sampler.sample(
                        S=self.steps,
                        batch_size=self.batch_size,
                        shape=self.shape,
                        eta=0.0,
                        verbose=False,
                        x_T=latent.clone(),
                    )
                torch.cuda.synchronize()
        finally:
            self._set_calibrating_for_precision(precision, unet, False)

        scales = self._export_static_scales(precision, unet)
        scale_path = os.path.join(self.output_dir, f"{precision}_generated_static_scales.pt")
        torch.save(scales, scale_path)

        del model, sampler
        torch.cuda.empty_cache()
        gc.collect()
        return scales, scale_path

    def _resolve_static_scales(self, precision: str) -> Dict[str, object]:
        cached = self._static_scale_cache.get(precision)
        if cached is not None:
            return cached

        scale_path = self._scale_path_for_precision(precision)
        if scale_path is not None and os.path.exists(scale_path):
            file_scales = torch.load(scale_path, weights_only=True)
            match_count = self._probe_matching_scale_count(precision, file_scales)
            if match_count > 0:
                info = {
                    "scales": dict(file_scales),
                    "path": scale_path,
                    "status": "loaded-file",
                    "match_count": match_count,
                }
                self._static_scale_cache[precision] = info
                return info
            print(f"  ignoring {precision.upper()} scale file with 0 matching quantized conv keys: {scale_path}")

        generated_scales, generated_path = self._generate_static_scales(precision)
        info = {
            "scales": generated_scales,
            "path": generated_path,
            "status": "generated-calibration",
            "match_count": len(generated_scales),
        }
        self._static_scale_cache[precision] = info
        return info

    def _apply_quant_mode_scales(self, spec: ModeSpec, unet) -> Tuple[int, Optional[str], str, str]:
        if self.quant_mode == "dynamic":
            return 0, None, "disabled-dynamic", "dynamic-disabled"

        info = self._resolve_static_scales(spec.precision)
        scales = info["scales"]
        if spec.precision == "int8":
            if spec.implementation == "fused":
                loaded = apply_static_scales(unet, scales)
            else:
                loaded = apply_separate_int8_scales(unet, scales)
        else:
            if spec.implementation == "fused":
                loaded = apply_int4_static_scales(unet, scales)
            else:
                loaded = apply_separate_int4_scales(unet, scales)

        status = info["status"] if loaded > 0 else f"{info['status']}-no-match"
        return loaded, info["path"], status, str(info["status"])

    def _build_mode(self, spec: ModeSpec):
        model, _ = load_model(self.config_path, self.ckpt_path)
        model = self._prepare_model(model)
        unet = model.model.diffusion_model
        released_calibration_mb = 0.0

        self._convert_unet(spec, unet)
        loaded_scales, scale_path, scale_status, scale_source = self._apply_quant_mode_scales(spec, unet)
        if spec.implementation == "fused":
            released_calibration_mb = release_calibration_only_buffers(unet)
            torch.cuda.empty_cache()
        sampler = DDIMSampler(model)
        return model, sampler, loaded_scales, scale_path, scale_status, scale_source, released_calibration_mb

    def _reset_state(self, spec: ModeSpec, unet) -> None:
        if spec.precision == "int8":
            if spec.implementation == "fused":
                reset_modiff_state_int8(unet)
            else:
                reset_modiff_state_separate_int8(unet)
        else:
            if spec.implementation == "fused":
                reset_modiff_state_int4(unet)
            else:
                reset_modiff_state_separate_int4(unet)

    def _make_timed_latents(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(self.seed)
        warmup = torch.randn((self.batch_size, *self.shape), device="cuda", generator=generator)
        timed = [
            torch.randn((self.batch_size, *self.shape), device="cuda", generator=generator)
            for _ in range(self.timing_iterations)
        ]
        return warmup, timed

    def run_mode(self, spec: ModeSpec) -> None:
        print(f"\n{'=' * 80}\n{spec.label}\n{'=' * 80}")

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        warmup_latent, timed_latents = self._make_timed_latents()
        model, sampler, loaded_scales, scale_path, scale_status, scale_source, released_calibration_mb = self._build_mode(spec)
        unet = model.model.diffusion_model
        mem_after_setup = measure_gpu_memory()

        use_autocast = True
        dtype = torch.float16

        for warmup_idx in range(self.warmup_runs):
            print(f"  warmup {warmup_idx + 1}/{self.warmup_runs}: {self.steps} DDIM steps @ batch {self.batch_size}")
            self._reset_state(spec, unet)
            with suppress_sampler_output(), torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
                sampler.sample(
                    S=self.steps,
                    batch_size=self.batch_size,
                    shape=self.shape,
                    eta=0.0,
                    verbose=False,
                    x_T=warmup_latent.clone(),
                )
            torch.cuda.synchronize()

        mem_after_warmup = measure_gpu_memory()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        repeat_means_ms: List[float] = []
        repeat_totals_ms: List[float] = []
        repeat_iteration_ms: List[List[float]] = []

        for repeat_idx in range(self.timing_repeats):
            iteration_ms: List[float] = []
            for latent_batch in timed_latents:
                self._reset_state(spec, unet)
                torch.cuda.synchronize()
                t0 = time.time()
                with suppress_sampler_output(), torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
                    sampler.sample(
                        S=self.steps,
                        batch_size=self.batch_size,
                        shape=self.shape,
                        eta=0.0,
                        verbose=False,
                        x_T=latent_batch.clone(),
                    )
                torch.cuda.synchronize()
                iteration_ms.append((time.time() - t0) * 1000.0)

            repeat_iteration_ms.append(iteration_ms)
            repeat_totals_ms.append(sum(iteration_ms))
            repeat_means_ms.append(sum(iteration_ms) / len(iteration_ms))

        mem_peak = measure_gpu_memory()
        mean_call_ms = float(sum(repeat_means_ms) / len(repeat_means_ms))
        median_call_ms = float(statistics.median(repeat_means_ms))
        std_call_ms = float(statistics.pstdev(repeat_means_ms)) if len(repeat_means_ms) > 1 else 0.0
        peak_delta_mb = max(0.0, mem_peak["max_allocated_mb"] - mem_after_warmup["allocated_mb"])

        result = {
            "mode": spec.name,
            "label": spec.label,
            "precision": spec.precision,
            "implementation": spec.implementation,
            "quant_mode": self.quant_mode,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "warmup_runs": self.warmup_runs,
            "timing_iterations": self.timing_iterations,
            "timing_repeats": self.timing_repeats,
            "timed_calls": self.timing_iterations * self.timing_repeats,
            "repeat_mean_call_ms": repeat_means_ms,
            "repeat_total_ms": repeat_totals_ms,
            "per_repeat_iteration_ms": repeat_iteration_ms,
            "mean_call_ms": mean_call_ms,
            "median_call_ms": median_call_ms,
            "std_call_ms": std_call_ms,
            "min_call_ms": float(min(repeat_means_ms)),
            "max_call_ms": float(max(repeat_means_ms)),
            "time_per_sample_ms": mean_call_ms / self.batch_size,
            "time_per_step_ms": mean_call_ms / (self.batch_size * self.steps),
            "memory_allocated_mb": mem_after_warmup["allocated_mb"],
            "memory_reserved_mb": mem_after_warmup["reserved_mb"],
            "memory_setup_allocated_mb": mem_after_setup["allocated_mb"],
            "memory_setup_reserved_mb": mem_after_setup["reserved_mb"],
            "memory_ready_allocated_mb": mem_after_warmup["allocated_mb"],
            "memory_ready_reserved_mb": mem_after_warmup["reserved_mb"],
            "memory_peak_mb": mem_peak["max_allocated_mb"],
            "memory_peak_delta_mb": peak_delta_mb,
            "loaded_static_scales": int(loaded_scales),
            "scale_path": scale_path,
            "scale_status": scale_status,
            "scale_source": scale_source,
            "released_calibration_buffers_mb": released_calibration_mb,
            "timing_mode": "synchronized_wall_clock_per_full_sampling_call",
            "decode_in_timed_region": False,
            "reset_state_before_each_call": True,
            "sampler_output_suppressed": True,
            "buffer_pool_enabled": False,
            "peak_measurement_scope": "timed-region-after-warmup",
        }
        self.results[spec.name] = result

        print(f"  mean call: {mean_call_ms:.2f} ms")
        print(f"  per sample: {result['time_per_sample_ms']:.2f} ms")
        print(f"  per step: {result['time_per_step_ms']:.4f} ms")
        print(f"  repeat means: {', '.join(f'{value:.2f}' for value in repeat_means_ms)} ms")
        print(f"  setup allocated: {result['memory_setup_allocated_mb']:.0f} MB")
        print(f"  ready allocated: {result['memory_ready_allocated_mb']:.0f} MB")
        print(f"  timed peak memory: {result['memory_peak_mb']:.0f} MB")
        print(f"  timed peak delta: {result['memory_peak_delta_mb']:.0f} MB")
        if released_calibration_mb > 0.0:
            print(f"  released calibration-only buffers: {released_calibration_mb:.0f} MB")
        if self.quant_mode == "dynamic":
            print("  loaded static scales: 0 (dynamic mode disables cached activation scales)")
        elif scale_path is not None and loaded_scales > 0:
            print(f"  loaded static scales: {loaded_scales} ({scale_status}) from {scale_path}")
        elif scale_path is not None:
            print(f"  loaded static scales: 0 ({scale_status}: {scale_path})")
        else:
            print("  loaded static scales: 0 (static mode had no usable scale source)")

        del model, sampler
        torch.cuda.empty_cache()
        gc.collect()

    def write_results(self) -> str:
        path = os.path.join(self.output_dir, "modelwise_fused_vs_separate_results.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)
        return path

    def write_report(self) -> str:
        report_path = os.path.join(self.output_dir, "MODELWISE_FUSED_VS_SEPARATE_REPORT.md")
        quant_mode = next(iter(self.results.values())).get("quant_mode", self.quant_mode) if self.results else self.quant_mode
        lines: List[str] = [
            f"# Model-wise fused-vs-separate MoDiff benchmark ({quant_mode} quantization)",
            "",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**GPU**: {torch.cuda.get_device_name()}",
            f"**Config**: `{self.config_path}`",
            f"**Checkpoint**: `{self.ckpt_path}`",
            f"**Batch Size**: {self.batch_size}",
            f"**Timesteps**: {self.steps}",
            f"**Quant Mode**: {quant_mode}",
            "",
            "Timing notes:",
            f"- Each number is the mean full-sampling-call latency over {self.timing_repeats} timed repeats × {self.timing_iterations} iterations.",
            "- Timed region covers the full DDIM denoising call and excludes decode / image save.",
            "- MoDiff state is reset before every timed call, outside the timed region.",
            "- Sampler stdout/stderr and progress bars are suppressed during warmup and timed calls so console I/O does not pollute the timing.",
            "- Peak GPU memory is reset **after warmup** and measured only over the timed region, so one-off setup allocations do not distort the comparison.",
            "- The benchmark intentionally leaves the global buffer pool disabled because the current pool pre-allocates oversized residual buffers for fused layers and inflates memory without benefiting these kernels.",
            "- Fused layers also drop their calibration-only FP32 `_orig_weight` clones after setup, because those buffers are only needed for later SmoothQuant calibration and otherwise exaggerate inference memory.",
            "- The same pre-generated latent tensors are reused across compared modes so fused and separate paths denoise identical workloads.",
            (
                "- Static mode applies one fixed activation scale per quantized layer; when a supplied scale file has no matching conv keys, the benchmark auto-calibrates fresh scales from representative DDIM sampling calls."
                if quant_mode == "static"
                else "- Dynamic mode intentionally disables static activation scales so each call recomputes its own per-tensor activation scale."
            ),
            "",
            "## Timing summary",
            "",
            "| Mode | Mean call (ms) | Std over repeat means (ms) | Time/sample (ms) | Time/step (ms) | Ready memory (MB) | Timed peak (MB) | Peak Δ (MB) | Loaded scales | Scale status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]

        mode_order = [
            "int8_fused_modiff",
            "int8_separate_modiff",
            "int4_fused_modiff",
            "int4_separate_modiff",
        ]
        for mode in mode_order:
            if mode not in self.results:
                continue
            result = self.results[mode]
            lines.append(
                f"| {result['label']} | {result['mean_call_ms']:.2f} | {result['std_call_ms']:.2f} | "
                f"{result['time_per_sample_ms']:.2f} | {result['time_per_step_ms']:.4f} | {result.get('memory_ready_allocated_mb', result['memory_allocated_mb']):.0f} | {result['memory_peak_mb']:.0f} | {result.get('memory_peak_delta_mb', 0.0):.0f} | {result['loaded_static_scales']} | {result.get('scale_status', 'unknown')} |"
            )

        lines.extend([
            "",
            "## Fusion speedup",
            "",
        ])

        if "int8_fused_modiff" in self.results and "int8_separate_modiff" in self.results:
            fused = self.results["int8_fused_modiff"]["mean_call_ms"]
            separate = self.results["int8_separate_modiff"]["mean_call_ms"]
            lines.append(f"- **INT8 fused vs separate**: {separate / fused:.2f}x faster ({separate:.2f} ms → {fused:.2f} ms).")

        if "int4_fused_modiff" in self.results and "int4_separate_modiff" in self.results:
            fused = self.results["int4_fused_modiff"]["mean_call_ms"]
            separate = self.results["int4_separate_modiff"]["mean_call_ms"]
            lines.append(f"- **INT4 fused vs separate**: {separate / fused:.2f}x faster ({separate:.2f} ms → {fused:.2f} ms).")

        lines.extend([
            "",
            "## Calibration notes",
            "",
            (
                "- Static activation scales are shared between the fused and separate variants of each precision so the comparison isolates kernel fusion rather than calibration drift."
                if quant_mode == "static"
                else "- Dynamic mode ignores static calibration files by design; both fused and separate paths recompute activation scales online."
            ),
            "",
            "## Memory notes",
            "",
            "- Reported peak memory is the **timed-region peak after warmup**, not the whole-process peak since process start.",
            "- The earlier inflated fused-memory readings were caused by two setup-time artifacts: benchmark-side buffer-pool preallocation and retained calibration-only `_orig_weight` clones inside fused modules.",
            "- The rebuilt benchmark disables the former and releases the latter before timing.",
            "- After those artifacts are removed, the remaining INT8 fused post-warmup gap is mostly backend/workspace retention rather than model-owned tensors: Python-visible extra fused state is only the persistent `_residual_buf` (~44 MB on this UNet), while roughly 0.5 GB remains allocated until the fused INT8 model is destroyed. INT4 does not show the same lingering footprint.",
        ])

        for mode in mode_order:
            if mode not in self.results:
                continue
            result = self.results[mode]
            scale_path = result.get("scale_path") or "n/a"
            lines.append(
                f"- {result['label']}: scale source **{result.get('scale_source', 'unknown')}**, status **{result.get('scale_status', 'unknown')}**, applied scales **{result['loaded_static_scales']}**, path `{scale_path}`."
            )

        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return report_path

    def run(self, precisions: List[str]) -> None:
        mode_order = [
            ModeSpec("int8", "fused"),
            ModeSpec("int8", "separate"),
            ModeSpec("int4", "fused"),
            ModeSpec("int4", "separate"),
        ]
        for spec in mode_order:
            if spec.precision in precisions:
                self.run_mode(spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-model fused-vs-separate MoDiff benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        help="latent-diffusion config used to instantiate the full model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/lsun_churches256/model.ckpt",
        help="checkpoint for the LSUN-Churches LDM model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_fused_vs_separate/modelwise_results",
        help="directory for JSON/Markdown outputs",
    )
    parser.add_argument("--precision", choices=["all", "int8", "int4"], default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--timing-iterations", type=int, default=2, help="full sampling calls per repeat")
    parser.add_argument("--timing-repeats", type=int, default=3, help="number of timed repeats")
    parser.add_argument("--warmup-runs", type=int, default=1, help="full warmup sampling calls before timing")
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument(
        "--quant-mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="dynamic recomputes activation scales online; static applies cached per-layer activation scales",
    )
    parser.add_argument(
        "--static-calibration-runs",
        type=int,
        default=1,
        help="representative DDIM sampling runs used to auto-calibrate static scales when a matching scale file is unavailable",
    )
    parser.add_argument(
        "--int8-scale-path",
        type=str,
        default="integration/calibration/int8_calibration.pt",
        help="optional INT8 static-scale file; static mode auto-calibrates if the file is missing or has no matching conv keys",
    )
    parser.add_argument(
        "--int4-scale-path",
        type=str,
        default="integration/calibration/int4_calibration.pt",
        help="optional INT4 static-scale file; static mode auto-calibrates if the file is missing or has no matching conv keys",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")

    config_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO_ROOT, args.ckpt)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(REPO_ROOT, args.output_dir)

    if not os.path.exists(config_path):
        raise SystemExit(f"Config file not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"Checkpoint file not found: {ckpt_path}\n"
            "Pass --ckpt with the correct LSUN-Churches model checkpoint to run the full-model benchmark."
        )

    set_seed(args.seed)
    backend = ensure_working_conv_backend()

    precisions = ["int8", "int4"] if args.precision == "all" else [args.precision]
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Conv backend: {backend}")
    print(f"Precisions: {', '.join(precisions)}")
    print(f"Quant mode: {args.quant_mode}")
    print(
        f"Config: steps={args.steps}, batch_size={args.batch_size}, "
        f"timing_iterations={args.timing_iterations}, timing_repeats={args.timing_repeats}"
    )

    bench = ModelwiseFusionBenchmark(
        config_path=config_path,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        steps=args.steps,
        timing_iterations=args.timing_iterations,
        timing_repeats=args.timing_repeats,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        quant_mode=args.quant_mode,
        static_calibration_runs=args.static_calibration_runs,
        int8_scale_path=args.int8_scale_path,
        int4_scale_path=args.int4_scale_path,
    )
    bench.run(precisions)
    json_path = bench.write_results()
    report_path = bench.write_report()

    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved Markdown report to {report_path}")


if __name__ == "__main__":
    main()
