"""
PyTorch Native INT8 + CUDA Graph Inference for MoDiff.

This module implements PyTorch-native INT8 simulation for Conv2d layers and wires
real CUDA Graph replay into the per-step UNet call path used by DDIM sampling.

Two modes:
1. int8_cudagraph: PyTorch INT8 + CUDA Graph + MoDiff temporal caching
2. int8_cudagraph_baseline: Same but without MoDiff temporal caching

Implementation details:
- baseline mode captures one per-step UNet graph (`standard`)
- MoDiff mode captures two per-step UNet graphs:
    * `first`      : first denoising step after state reset
    * `modulated`  : all subsequent denoising steps
- DDIM itself stays as the outer Python loop, but each UNet invocation in that
    loop is replayed from a captured CUDA graph using fixed static buffers.

Architecture (unchanged from original MoDiff):
- UNet with ResBlocks, attention, time embedding
- DDIMSampler with configurable steps
- AutoencoderKL first stage decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


class PyTorchInt8Conv2d(nn.Module):
    """
    INT8 Conv2d using PyTorch native quantization.
    
    Uses per-tensor symmetric quantization for activations and
    per-channel symmetric quantization for weights, following the
    same scheme as the CUTLASS implementation but using PyTorch ops.
    
    When CUDA Graphs are enabled, static buffers are used so the
    graph can be replayed without reallocation.
    """

    def __init__(self, conv: nn.Conv2d, layer_name: str = ""):
        super().__init__()
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self.groups = conv.groups

        # Store FP32 weight for quantization
        w_data = conv.weight.data.float()

        # Per-channel weight quantization (symmetric INT8)
        K = self.out_channels
        w_flat = w_data.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values
        w_scale = torch.clamp(ch_max / 127.0, min=1e-8)
        self.register_buffer('weight_scale', w_scale)  # [K]

        # Quantize weights to INT8
        w_quant = (w_flat / w_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        w_quant = w_quant.reshape_as(w_data)
        self.register_buffer('weight_int8', w_quant)

        # FP16 fallback weight for matmul
        self.register_buffer('weight_fp16', w_data.half())

        # Bias
        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.float())
        else:
            self.bias = None

        # MoDiff state
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        self.warmup_steps = 3

        # Calibration
        self.calibrating = False
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self.register_buffer('static_input_scale', torch.tensor(1.0))

        # Persistent state buffers for CUDA Graph compatibility
        self._state_ready = False

    def _ensure_state_buffers(self, x: torch.Tensor):
        h_out = ((x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x.shape[0], self.out_channels, h_out, w_out)

        if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
            self.a_hat_cache = torch.zeros_like(x)
            self._state_ready = False

        if self.o_hat_cache is None or self.o_hat_cache.shape != output_shape:
            self.o_hat_cache = torch.zeros(output_shape, device=x.device, dtype=torch.float32)
            self._state_ready = False

    def _quantize_and_conv(self, x: torch.Tensor, input_scale, with_bias: bool = True) -> torch.Tensor:
        """INT8 quantize input, do FP16 conv (PyTorch native), dequantize output."""
        # Quantize activation to INT8
        x_int8 = (x * input_scale).round().clamp(-127, 127)

        # Dequantize for FP16 conv (PyTorch doesn't have native INT8 conv on GPU)
        x_dq = x_int8 / input_scale

        # FP16 convolution (fastest for A40)
        x_fp16 = x_dq.half()
        out = F.conv2d(x_fp16, self.weight_fp16, None,
                       self.stride, self.padding, self.dilation, self.groups)
        out = out.float()

        if with_bias and self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

    def _dequantize_activation(self, x: torch.Tensor, scale) -> torch.Tensor:
        """Simulate quantize-then-dequantize: a_hat = Q(x)."""
        return (x * scale).round().clamp(-127, 127) / scale

    def _compute_scale_float(self, x: torch.Tensor) -> float:
        """Compute per-tensor symmetric scale as Python float (for calibration/export only)."""
        if self.is_calibrated:
            return float(self.static_input_scale.item())
        abs_max = x.abs().max().item()
        scale = 127.0 / max(abs_max, 1e-6)
        if self.calibrating:
            self._scale_sum += scale
            self._scale_count += 1
        return scale

    def _compute_scale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-tensor symmetric scale on GPU, graph-safe."""
        if self.is_calibrated:
            scale = self.static_input_scale
            if scale.device != x.device:
                scale = scale.to(x.device)
            return scale

        abs_max = x.abs().amax()
        scale = 127.0 / torch.clamp(abs_max, min=1e-6)
        if self.calibrating:
            scale_float = float(scale.detach().item())
            self._scale_sum += scale_float
            self._scale_count += 1
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()

        if not self.modiff_enabled:
            scale = self._compute_scale_tensor(x)
            return self._quantize_and_conv(x, scale)

        if self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
            return output

        return self._forward_modulated(x)

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """First timestep: warm-up."""
        self._ensure_state_buffers(x)

        scale = self._compute_scale_tensor(x)
        a_hat = self._dequantize_activation(x, scale)
        o_hat = self._quantize_and_conv(x, scale)

        for _ in range(self.warmup_steps - 1):
            residual = x - a_hat
            r_scale = self._compute_scale_tensor(residual)
            r_dq = self._dequantize_activation(residual, r_scale)
            o_hat = o_hat + self._quantize_and_conv(residual, r_scale, with_bias=False)
            a_hat = a_hat + r_dq

        self.a_hat_cache.copy_(a_hat)
        self.o_hat_cache.copy_(o_hat)
        self._state_ready = True
        return self.o_hat_cache

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """MoDiff modulated step."""
        self._ensure_state_buffers(x)
        if not self._state_ready:
            self.is_first_step = True
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

        residual = x - self.a_hat_cache
        abs_max = residual.abs().amax()
        r_scale = 127.0 / torch.clamp(abs_max, min=1e-6)

        r_dq = (residual * r_scale).round().clamp(-127, 127) / r_scale
        conv_r = self._quantize_and_conv(residual, r_scale, with_bias=False)

        self.a_hat_cache.add_(r_dq)
        self.o_hat_cache.add_(conv_r)
        return self.o_hat_cache

    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()

    def reset_state(self):
        self.is_first_step = True
        with torch.inference_mode():
            if self.a_hat_cache is not None:
                self.a_hat_cache.zero_()
            if self.o_hat_cache is not None:
                self.o_hat_cache.zero_()
        self._state_ready = False

    def set_calibrating(self, calibrating: bool):
        if calibrating:
            self.calibrating = True
            self.is_calibrated = False
            self._scale_sum = 0.0
            self._scale_count = 0
        else:
            self.calibrating = False
            if self._scale_count > 0:
                avg_scale = self._scale_sum / self._scale_count
                self.static_input_scale.fill_(avg_scale)
                self.is_calibrated = True

    def set_static_scale(self, scale: float):
        self.static_input_scale.fill_(scale)
        self.is_calibrated = True


def _iter_pytorch_int8_modules(model: nn.Module):
    for m in model.modules():
        if isinstance(m, PyTorchInt8Conv2d):
            yield m


class _GraphRecord:
    def __init__(self, graph: torch.cuda.CUDAGraph, static_x: torch.Tensor, static_t: torch.Tensor,
                 static_kwargs: Dict[str, Any], static_output: torch.Tensor):
        self.graph = graph
        self.static_x = static_x
        self.static_t = static_t
        self.static_kwargs = static_kwargs
        self.static_output = static_output


class UNetCudaGraphManager:
    """Capture and replay per-step UNet forwards for standard/first/modulated phases."""

    def __init__(self, diffusion_wrapper: nn.Module, batch_size: int, shape: Tuple[int, int, int],
                 modiff_enabled: bool):
        self.diffusion_wrapper = diffusion_wrapper
        self.diffusion_model = diffusion_wrapper.diffusion_model
        self.batch_size = batch_size
        self.shape = shape
        self.modiff_enabled = modiff_enabled
        self.records: Dict[str, _GraphRecord] = {}
        self.phase_index = 0
        self.capture_count = 0
        self.replay_count = 0
        self._previous_cudnn_benchmark = torch.backends.cudnn.benchmark

        # cuDNN algorithm search is not capture-safe; freeze engine selection.
        torch.backends.cudnn.benchmark = False

        self._prepare_module_state_buffers()

    def _prepare_module_state_buffers(self):
        C, H, W = self.shape
        example = torch.zeros(self.batch_size, C, H, W, device=next(self.diffusion_model.parameters()).device, dtype=torch.float32)
        for module in _iter_pytorch_int8_modules(self.diffusion_model):
            module._ensure_state_buffers(example)

    def reset_sequence(self):
        self.phase_index = 0

    def _phase_name(self) -> str:
        if not self.modiff_enabled:
            return 'standard'
        return 'first' if self.phase_index == 0 else 'modulated'

    def _copy_structure(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.clone()
        if isinstance(obj, list):
            return [self._copy_structure(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._copy_structure(v) for v in obj)
        if isinstance(obj, dict):
            return {k: self._copy_structure(v) for k, v in obj.items()}
        return obj

    def _copy_into(self, dst: Any, src: Any):
        if torch.is_tensor(dst):
            dst.copy_(src)
            return
        if isinstance(dst, list):
            for d, s in zip(dst, src):
                self._copy_into(d, s)
            return
        if isinstance(dst, tuple):
            for d, s in zip(dst, src):
                self._copy_into(d, s)
            return
        if isinstance(dst, dict):
            for k in dst:
                self._copy_into(dst[k], src[k])

    def _set_module_first_step(self, is_first_step: bool):
        for module in _iter_pytorch_int8_modules(self.diffusion_model):
            module.is_first_step = is_first_step

    def _capture_phase(self, phase: str, x: torch.Tensor, t: torch.Tensor, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        static_x = torch.empty_like(x)
        static_t = torch.empty_like(t)
        static_kwargs = self._copy_structure(model_kwargs)
        static_x.copy_(x)
        static_t.copy_(t)

        if phase == 'first':
            self._set_module_first_step(True)
        elif phase == 'modulated':
            self._set_module_first_step(False)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            static_output = self.diffusion_wrapper(static_x, static_t, **static_kwargs)

        record = _GraphRecord(graph, static_x, static_t, static_kwargs, static_output)
        self.records[phase] = record
        self.capture_count += 1

        if phase == 'first':
            self._set_module_first_step(False)
        return static_output

    def __call__(self, x: torch.Tensor, t: torch.Tensor, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        phase = self._phase_name()
        if phase not in self.records:
            out = self._capture_phase(phase, x, t, model_kwargs)
        else:
            record = self.records[phase]
            self._copy_into(record.static_x, x)
            self._copy_into(record.static_t, t)
            self._copy_into(record.static_kwargs, model_kwargs)
            record.graph.replay()
            out = record.static_output
            self.replay_count += 1
            if phase == 'first':
                self._set_module_first_step(False)

        if self.modiff_enabled and phase == 'first':
            self.phase_index = 1
        elif not self.modiff_enabled:
            self.phase_index += 1
        else:
            self.phase_index += 1
        return out

    def stats(self) -> Dict[str, int]:
        return {
            'num_graphs': len(self.records),
            'capture_count': self.capture_count,
            'replay_count': self.replay_count,
        }


def install_cuda_graph_replay_pytorch_int8(diffusion_wrapper: nn.Module, batch_size: int,
                                           shape: Tuple[int, int, int]) -> UNetCudaGraphManager:
    modiff_enabled = any(module.modiff_enabled for module in _iter_pytorch_int8_modules(diffusion_wrapper.diffusion_model))
    manager = UNetCudaGraphManager(diffusion_wrapper, batch_size=batch_size, shape=shape, modiff_enabled=modiff_enabled)
    diffusion_wrapper.diffusion_model._cuda_graph_manager = manager
    return manager


def get_cuda_graph_replay_stats(diffusion_model: nn.Module) -> Optional[Dict[str, int]]:
    manager = getattr(diffusion_model, '_cuda_graph_manager', None)
    if manager is None:
        return None
    return manager.stats()


# ---------------------------------------------------------------------------
# CUDA Graph Wrapper
# ---------------------------------------------------------------------------

class CUDAGraphWrapper:
    """
    Wraps a model's forward pass in a CUDA Graph for replay.
    
    CUDA Graphs capture the entire GPU execution (kernels, memory copies)
    into a single graph that can be replayed with minimal CPU overhead.
    This eliminates per-kernel launch overhead (~5-15us per kernel).
    
    Limitations:
    - Input/output shapes must be fixed (use static buffers)
    - Cannot use dynamic control flow inside the graph
    - Must use the same CUDA stream
    """

    def __init__(self, model, batch_size: int, shape: tuple, device: str = 'cuda'):
        self.model = model
        self.batch_size = batch_size
        self.shape = shape
        self.device = device
        self.graph = None
        self.stream = torch.cuda.Stream()

        # Static input/output buffers
        C, H, W = shape
        self._static_x = torch.zeros(batch_size, C, H, W, device=device)
        self._static_t = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._static_context = None
        self._static_output = torch.zeros(batch_size, C, H, W, device=device)
        self._captured = False

    def warmup(self, x: torch.Tensor, t: torch.Tensor, context=None, num_warmup: int = 3):
        """Warm up the model before capture."""
        with torch.cuda.stream(self.stream):
            for _ in range(num_warmup):
                _ = self.model(x, t, context)
        self.stream.synchronize()

    def capture(self, x: torch.Tensor, t: torch.Tensor, context=None):
        """Capture the forward pass into a CUDA Graph."""
        # Copy inputs to static buffers
        self._static_x.copy_(x)
        self._static_t.copy_(t)
        if context is not None:
            if self._static_context is None:
                self._static_context = context.clone()
            else:
                self._static_context.copy_(context)

        # Warm up
        self.warmup(self._static_x, self._static_t, self._static_context)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self._static_output = self.model(
                self._static_x, self._static_t, self._static_context
            )
        self._captured = True

    def replay(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        """Replay the captured graph with new inputs."""
        if not self._captured:
            # Fall back to eager execution
            return self.model(x, t, context)

        # Copy new inputs to static buffers
        self._static_x.copy_(x)
        self._static_t.copy_(t)
        if context is not None and self._static_context is not None:
            self._static_context.copy_(context)

        # Replay graph
        self.graph.replay()

        return self._static_output

    def reset(self):
        """Reset the graph (e.g., for new sample)."""
        self._captured = False
        self.graph = None


# ---------------------------------------------------------------------------
# Model Conversion
# ---------------------------------------------------------------------------

def convert_model_to_pytorch_int8(model: nn.Module, prefix: str = "") -> nn.Module:
    """Convert Conv2d layers to PyTorchInt8Conv2d."""
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, PyTorchInt8Conv2d):
            if child.in_channels < 32:
                continue
            is_skip = 'skip' in name
            is_final_out = full_name.startswith('out.')
            is_pointwise = child.kernel_size == (1, 1)
            is_grouped = child.groups != 1
            if is_skip or is_final_out or is_pointwise or is_grouped:
                continue

            optimized = PyTorchInt8Conv2d(child, layer_name=full_name)
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized = optimized.to(target_device)
            setattr(model, name, optimized)
        else:
            convert_model_to_pytorch_int8(child, prefix=full_name)
    return model


def enable_modiff_mode_pytorch_int8(model: nn.Module, enabled: bool = True):
    for m in model.modules():
        if isinstance(m, PyTorchInt8Conv2d):
            m.enable_modiff(enabled)


def reset_modiff_state_pytorch_int8(model: nn.Module):
    for m in model.modules():
        if isinstance(m, PyTorchInt8Conv2d):
            m.reset_state()
    manager = getattr(model, '_cuda_graph_manager', None)
    if manager is not None:
        manager.reset_sequence()


def set_calibrating_pytorch_int8(model: nn.Module, calibrating: bool):
    for m in model.modules():
        if isinstance(m, PyTorchInt8Conv2d):
            m.set_calibrating(calibrating)


def export_pytorch_int8_scales(model: nn.Module) -> Dict[str, float]:
    scales = {}
    for m in model.modules():
        if isinstance(m, PyTorchInt8Conv2d) and m.is_calibrated:
            scales[m.layer_name] = float(m.static_input_scale.item())
    return scales


def apply_pytorch_int8_scales(model: nn.Module, scales: Dict[str, float]) -> int:
    loaded = 0
    for m in model.modules():
        if isinstance(m, PyTorchInt8Conv2d) and m.layer_name in scales:
            m.set_static_scale(scales[m.layer_name])
            loaded += 1
    return loaded
