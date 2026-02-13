"""
CUTLASS INT8 Conv2d with SmoothQuant + MoDiff Error-Compensated Modulation.

Uses true INT8×INT8 tensor core kernels via CUTLASS for maximum throughput.
Implements SmoothQuant to migrate per-channel activation variance into weights,
and MoDiff paper's error-compensated modulation across diffusion timesteps.

MoDiff equations (Gao et al., ICML 2025):
    t=T (first step):
        a_hat_T = Q(a_T)                                    -- Eq. (ec1)
        o_hat_T = A(a_hat_T) + bias                         -- Eq. (ec2)
    t<T (modulated steps):
        a_hat_t = Q(a_t - a_hat_{t+1}) + a_hat_{t+1}        -- Eq. (ec5)
        o_hat_t = A(Q(a_t - a_hat_{t+1})) + o_hat_{t+1}     -- Eq. (ec6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from integration.profiler import profiler

try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: modiff_cutlass extension not found.")


class OptimizedInt8Conv2d(nn.Module):
    """
    CUTLASS-based INT8 Conv2d.
    Replaces previous implementations for better generality.
    """
    def __init__(self, conv: nn.Conv2d, layer_name: str = "", use_compile: bool = False):
        super().__init__()
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        K = self.out_channels

        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self.groups = conv.groups

        w_data = conv.weight.data  # [K, C_in, R, S]

        # --- SmoothQuant ---
        self.register_buffer('smooth_scale', torch.ones(1, self.in_channels, 1, 1))
        self.register_buffer('_smooth_inv', torch.ones(1, self.in_channels, 1, 1))
        self.register_buffer('_orig_weight', w_data.clone(), persistent=False)

        # --- Per-output-channel symmetric INT8 weight quantization ---
        w_flat = w_data.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values  # [K]
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)  # [K]
        self.register_buffer('weight_scale_channel', ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        w_quant = w_quant.reshape_as(w_data)
        # CUTLASS expects NHWC (K, R, S, C) for weights
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        self.register_buffer('weight_int8', w_nhwc)

        # --- Bias ---
        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.view(1, -1, 1, 1))
        else:
            self.bias = None

        self._empty_bias = None
        self.use_cutlass = HAS_CUTLASS and self.groups == 1

        # --- MoDiff state ---
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        self.step_count = 0
        self.warmup_steps = 3  # Reduced from 5: 3 steps sufficient for convergence

        # --- Calibration state ---
        self.calibrating = False
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self.register_buffer('static_input_scale', torch.tensor(1.0, dtype=torch.float32))
        self._act_channel_max: Optional[torch.Tensor] = None
        self._cached_scale_float: Optional[float] = None
        self._cached_alpha_tensor: Optional[torch.Tensor] = None

        # --- SmoothQuant identity flag for fast path ---
        self._smooth_is_identity = True

        # --- Fused kernel persistent buffers (lazy-initialized) ---
        self._residual_buf: Optional[torch.Tensor] = None
        self._scale_buf: Optional[torch.Tensor] = None
        self._inv_scale_buf: Optional[torch.Tensor] = None
        self._absmax_buf: Optional[torch.Tensor] = None
        self._retire_count: Optional[torch.Tensor] = None

    # ==================================================================
    # Quantization helpers
    # ==================================================================

    def _compute_activation_scale(self, x: torch.Tensor, is_residual: bool = False) -> float:
        """Per-tensor symmetric activation scale: 127 / max(|x|).
        Used during calibration and first-step only (slow path with .item() sync).
        """
        if self.calibrating:
            abs_max = x.abs().max().item()
            scale = 127.0 / max(abs_max, 1e-6)
            if not is_residual:
                self._scale_sum += scale
                self._scale_count += 1
                with torch.no_grad():
                    ch_max = x.abs().amax(dim=(0, 2, 3))
                    if self._act_channel_max is None:
                        self._act_channel_max = ch_max.clone()
                    else:
                        torch.max(self._act_channel_max, ch_max, out=self._act_channel_max)
            return scale

        if is_residual or not self.is_calibrated:
            abs_max = x.abs().max().item()
            return 127.0 / max(abs_max, 1e-6)

        if self._cached_scale_float is None:
            self._cached_scale_float = float(self.static_input_scale.item())
        return self._cached_scale_float

    def _compute_scale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """GPU-only per-tensor scale computation. No .item() sync.
        Returns 1-element GPU tensor = 127.0 / max(|x|, 1e-6).
        Used on the modulated hot path to avoid CPU-GPU synchronization.
        """
        abs_max = x.abs().amax()
        return 127.0 / torch.clamp(abs_max, min=1e-6)

    def _dequantize_activation(self, x: torch.Tensor, input_scale) -> torch.Tensor:
        """Simulate quantize-then-dequantize: a_hat = Q(x) in FP32.
        input_scale can be float or 1-element tensor.
        """
        return (x * input_scale).round().clamp(-127, 127) / input_scale

    def _int8_conv(self, x: torch.Tensor, input_scale, with_bias: bool = True) -> torch.Tensor:
        """INT8 x INT8 convolution via CUTLASS tensor core kernel.
        input_scale can be float or 1-element GPU tensor.
        """
        if self.use_cutlass:
            if isinstance(input_scale, (int, float)):
                alpha = 1.0 / input_scale
                if (self._cached_alpha_tensor is not None
                        and self._cached_scale_float is not None
                        and input_scale == self._cached_scale_float):
                    scale_tensor = self._cached_alpha_tensor
                else:
                    scale_tensor = torch.tensor([alpha], device=x.device, dtype=torch.float32)
                x_scaled = x * input_scale
                if not x_scaled.is_contiguous(memory_format=torch.channels_last):
                    x_scaled = x_scaled.contiguous(memory_format=torch.channels_last)
                x_int8 = x_scaled.round().clamp(-127, 127).to(torch.int8)
            else:
                # Tensor path: use fused scale+quantize kernel (no CPU sync)
                scale_tensor = (1.0 / input_scale).view(1)
                if not x.is_contiguous(memory_format=torch.channels_last):
                    x = x.contiguous(memory_format=torch.channels_last)
                x_int8 = modiff_cutlass.scale_quantize_int8(x, input_scale)

            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, device=x.device)

            out_raw = modiff_cutlass.conv2d_int8_fprop(
                x_int8,
                self.weight_int8,
                scale_tensor,
                self._empty_bias,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1]
            )
            # Dequantize per-channel
            out = out_raw * self.weight_scale_channel
        else:
            raise RuntimeError(
                f"CUTLASS INT8 kernel unavailable for layer {self.layer_name} "
                f"(groups={self.groups}). Build modiff_cutlass extension."
            )

        if with_bias and self.bias is not None:
            out = out + self.bias
        return out

    def _int8_conv_fused(self, x: torch.Tensor, scale: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
        """Optimized INT8 conv for modulated path: scale and inv_scale already computed on GPU.
        No .item() sync, no 1/scale computation kernel. Uses device pointer alpha for CUTLASS.
        Returns RAW (unscaled) CUTLASS output — caller applies weight_scale_channel via scale_accumulate.
        """
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        x_int8 = modiff_cutlass.scale_quantize_int8(x, scale)

        if self._empty_bias is None or self._empty_bias.device != x.device:
            self._empty_bias = torch.empty(0, device=x.device)

        return modiff_cutlass.conv2d_int8_fprop(
            x_int8,
            self.weight_int8,
            inv_scale.view(1),
            self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

    # ==================================================================
    # Forward paths
    # ==================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_start = profiler.start("Layer: OptimizedInt8Conv2d.forward")

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        # SmoothQuant: equalize per-channel activation ranges
        # For modulated path, SmoothQuant is fused into sub_absmax_scale kernel
        if not self._smooth_is_identity and (not self.modiff_enabled or self.is_first_step):
            x = x * self._smooth_inv

        if not self.modiff_enabled:
            output = self._forward_standard(x)
        elif self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
        else:
            output = self._forward_modulated(x)

        profiler.stop("Layer: OptimizedInt8Conv2d.forward", fwd_start)
        return output

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard INT8 forward without MoDiff modulation."""
        input_scale = self._compute_activation_scale(x)
        return self._int8_conv(x, input_scale, with_bias=True)

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """First timestep (t=T): warm-up with repeated quantisation."""
        input_scale = self._compute_activation_scale(x)
        a_hat = self._dequantize_activation(x, input_scale)
        o_hat = self._int8_conv(x, input_scale, with_bias=True)

        for _ in range(self.warmup_steps - 1):
            residual = x - a_hat
            r_scale = self._compute_activation_scale(residual, is_residual=True)
            conv_r = self._int8_conv(residual, r_scale, with_bias=False)
            r_dq = self._dequantize_activation(residual, r_scale)
            a_hat = a_hat + r_dq
            o_hat = o_hat + conv_r

        self.a_hat_cache = a_hat
        self.o_hat_cache = o_hat
        return o_hat.clone()

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """MoDiff modulated step (t<T). No periodic reset per paper.
        Uses fused sub+absmax+scale kernel and device pointer alpha to minimize
        kernel launches and avoid CPU-GPU synchronization.
        SmoothQuant multiply is fused into sub_absmax_scale when applicable.
        """
        self.step_count += 1

        if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
            self.is_first_step = True
            if not self._smooth_is_identity:
                x = x * self._smooth_inv
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

        # Lazy-init persistent buffers (reused across timesteps, never reallocated)
        if self._residual_buf is None or self._residual_buf.shape != x.shape:
            self._residual_buf = torch.empty_like(x)
            self._scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
            self._inv_scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
            self._absmax_buf = torch.zeros(1, device=x.device, dtype=torch.float32)
            self._retire_count = torch.zeros(1, device=x.device, dtype=torch.int32)

        # Lazy-init smooth_inv flat tensor (1-D contiguous for kernel)
        if not hasattr(self, '_smooth_inv_flat'):
            if not self._smooth_is_identity:
                self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
            else:
                self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

        # Fused: [smooth * x] - cache, absmax reduction, scale + inv_scale
        # SmoothQuant is fused into the kernel (no extra element-wise multiply)
        modiff_cutlass.sub_absmax_scale(
            x, self.a_hat_cache, self._residual_buf,
            self._absmax_buf, self._scale_buf, self._inv_scale_buf,
            self._retire_count, 127.0, self._smooth_inv_flat
        )

        # Quantize + CUTLASS conv: uses inv_scale directly as device pointer alpha
        conv_residual = self._int8_conv_fused(self._residual_buf, self._scale_buf, self._inv_scale_buf)

        # Fused dequant + cache accumulate: 5 kernel launches → 1
        modiff_cutlass.dequant_accumulate_int8(self._residual_buf, self.a_hat_cache, self._scale_buf)

        # Fused: o_hat_cache += conv_raw * weight_scale_channel (2 kernels → 1)
        modiff_cutlass.scale_accumulate(conv_residual, self.weight_scale_channel.view(-1), self.o_hat_cache)
        return self.o_hat_cache

    # ==================================================================
    # MoDiff controls
    # ==================================================================

    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()

    def reset_state(self):
        self.is_first_step = True
        self.a_hat_cache = None
        self.o_hat_cache = None
        self.step_count = 0

    # ==================================================================
    # Calibration + SmoothQuant
    # ==================================================================

    def begin_calibration(self):
        self.calibrating = True
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self._act_channel_max = None

    def end_calibration(self):
        self.calibrating = False
        if self._scale_count == 0:
            return

        if self._act_channel_max is not None and self._orig_weight is not None:
            self._apply_smoothquant()

        if self._act_channel_max is not None:
            s = self.smooth_scale.view(-1)
            smoothed_ch_max = self._act_channel_max / s
            smoothed_global_max = smoothed_ch_max.max().item()
            static_scale = 127.0 / max(smoothed_global_max, 1e-6)
        else:
            static_scale = self._scale_sum / self._scale_count

        self.static_input_scale.fill_(float(static_scale))
        self.is_calibrated = True
        self._cached_scale_float = float(static_scale)
        alpha = 1.0 / float(static_scale)
        self._cached_alpha_tensor = torch.tensor(
            [alpha], device=self.static_input_scale.device, dtype=torch.float32
        )
        self._smooth_inv.copy_(1.0 / self.smooth_scale)
        self._smooth_is_identity = bool(torch.allclose(
            self._smooth_inv,
            torch.ones_like(self._smooth_inv),
            atol=1e-6
        ))
        self._orig_weight = None

    def _apply_smoothquant(self):
        """SmoothQuant: fold per-channel activation scales into weights."""
        act_max = self._act_channel_max
        w = self._orig_weight
        K = self.out_channels

        w_dev = w.to(act_max.device)
        w_by_cin = w_dev.reshape(K, self.in_channels, -1)
        w_max = w_by_cin.abs().amax(dim=(0, 2))

        ratio = act_max / torch.clamp(w_max, min=1e-8)
        s = ratio.sqrt().clamp(min=1e-4, max=1e4)

        self.smooth_scale.copy_(s.view(1, -1, 1, 1))

        w_smoothed = w_dev * s.view(1, -1, 1, 1)
        w_flat = w_smoothed.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)

        self.weight_scale_channel.copy_(ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        w_quant = w_quant.reshape(K, self.in_channels, *self.kernel_size)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        self.weight_int8.data = w_nhwc.to(self.weight_int8.device)

    def set_calibrating(self, calibrating: bool):
        if calibrating:
            self.begin_calibration()
        else:
            self.end_calibration()

    def set_static_scale(self, scale: float):
        self.static_input_scale.fill_(float(scale))
        self.is_calibrated = True
        self._cached_scale_float = float(scale)
        alpha = 1.0 / float(scale)
        self._cached_alpha_tensor = torch.tensor(
            [alpha], device=self.static_input_scale.device, dtype=torch.float32
        )


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

def convert_model_to_optimized_int8(model: nn.Module, prefix: str = "", use_compile: bool = False) -> nn.Module:
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, OptimizedInt8Conv2d):
            if child.in_channels < 32:
                continue
            is_skip = 'skip' in name
            is_final_out = full_name.startswith('out.')
            is_pointwise = child.kernel_size == (1, 1)
            is_grouped = child.groups != 1

            if is_skip or is_final_out or is_pointwise or is_grouped:
                continue

            optimized_conv = OptimizedInt8Conv2d(child, layer_name=full_name, use_compile=use_compile)
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized_conv = optimized_conv.to(target_device)
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int8(child, prefix=full_name, use_compile=use_compile)

    # Convert to channels_last for PyTorch perf, then restore weight_int8
    model = model.to(memory_format=torch.channels_last)
    for m in model.modules():
        if isinstance(m, OptimizedInt8Conv2d):
            m.weight_int8.data = m.weight_int8.data.contiguous()
    return model


# ---------------------------------------------------------------------------
# Global calibration helpers
# ---------------------------------------------------------------------------

class CalibrationConfig:
    def __init__(self):
        self.is_calibrated = False
        self.scales = {}

    def update(self, layer_name: str, scale: float):
        self.scales[layer_name] = float(scale)

    def get_scale(self, layer_name: str):
        return self.scales.get(layer_name, None)

    def load(self, path):
        self.scales = torch.load(path, weights_only=True)
        self.is_calibrated = True

    def save(self, path):
        torch.save(self.scales, path)

    def finalize(self):
        self.is_calibrated = True


_calib_config = CalibrationConfig()


def get_calibration_config():
    return _calib_config


def reset_calibration():
    _calib_config.scales = {}
    _calib_config.is_calibrated = False


def enable_modiff_mode(model, enabled=True):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.enable_modiff(enabled)


def reset_modiff_state(model):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.reset_state()


def set_calibrating(model, calibrating):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.set_calibrating(calibrating)
            if not calibrating and module.is_calibrated:
                _calib_config.update(module.layer_name, float(module.static_input_scale.item()))
    if not calibrating:
        _calib_config.finalize()


def export_int8_static_scales(model: nn.Module) -> Dict[str, float]:
    scales = {}
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d) and module.is_calibrated:
            scales[module.layer_name] = float(module.static_input_scale.item())
    return scales


def apply_static_scales(model, *args, **kwargs):
    scales = kwargs.get('scales', None)
    if scales is None and len(args) > 0 and isinstance(args[0], dict):
        scales = args[0]
    if scales is None:
        return 0

    loaded = 0
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d) and module.layer_name in scales:
            module.set_static_scale(scales[module.layer_name])
            loaded += 1
    _calib_config.scales = dict(scales)
    _calib_config.is_calibrated = True
    return loaded


# Stubs for backward compatibility with benchmark_ldm.py
def convert_model_to_optimized_int8_static(model, sample_inputs=None, num_timesteps=None, device='cuda', **kwargs):
    model = convert_model_to_optimized_int8(model)
    if sample_inputs is not None and len(sample_inputs) > 0:
        set_calibrating(model, True)
        with torch.no_grad():
            for x in sample_inputs[:16]:
                t = torch.randint(0, 1000, (x.shape[0],), device=x.device)
                _ = model(x, t, None)
        set_calibrating(model, False)
    return model


def calibrate_int8_static_scales(model, *args, **kwargs):
    return export_int8_static_scales(model)

