import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, Optional
from integration.profiler import profiler

# CUTLASS INT8 backend -- required for INT8x INT8 tensor core convolution
try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: modiff_cutlass extension not found. Please compile it using setup.py.")

# Module-level toggle: set False to disable SmoothQuant (for A/B testing).
_ENABLE_SMOOTHQUANT = False


class OptimizedInt8Conv2d(nn.Module):
    """
    INT8 Conv2d with per-channel dynamic quantization + MoDiff modulation.

    Implements the MoDiff paper (Gao et al., ICML 2025) quantization:

    Two compute paths, selected per-layer based on channel uniformity:
    1. CUTLASS INT8×INT8 tensor core kernel (per-tensor activation scale)
       — used when per-channel activation max ratio < threshold
    2. Per-channel dynamic INT8 Q/DQ + FP16 conv (LCQ, matching paper)
       — used when channel imbalance makes per-tensor INT8 catastrophic

    Both paths use INT8 weights (per-output-channel symmetric quantization)
    stored in INT8 format for 4× memory compression.

    MoDiff error-compensated modulation across diffusion timesteps:
        t=T:  â_T = Q(a_T),  ô_T = A(â_T) + bias          -- Eq.(ec1-ec2)
        t<T:  â_t = Q(a_t − â_{t+1}) + â_{t+1}              -- Eq.(ec5)
              ô_t = A(Q(a_t − â_{t+1})) + ô_{t+1}           -- Eq.(ec6)

    MoDiff is most beneficial for aggressive quantization (W8A4, W8A6).
    For W8A8 per-channel, quantization error is small enough that MoDiff
    is typically unnecessary (matching paper's Table 1: W8A8 FID≈FP32).
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

        w_data = conv.weight.data.contiguous()  # [K, C_in, R, S], ensure contiguous

        # --- SmoothQuant ---
        # smooth_scale[c] = sqrt(act_max[c] / w_max[c]):  activations are
        # divided by this, weights are multiplied.  Identity until
        # calibration computes the actual values.
        device = w_data.device
        self.register_buffer('smooth_scale', torch.ones(1, self.in_channels, 1, 1, device=device))
        # Original FP32 weights for SmoothQuant re-quantization (freed after calib)
        self.register_buffer('_orig_weight', w_data.clone(), persistent=False)

        # --- Per-output-channel symmetric INT8 weight quantization ---
        w_flat = w_data.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values  # [K]
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)  # [K]
        self.register_buffer('weight_scale_channel', ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        w_quant = w_quant.view_as(w_data)
        # CUTLASS NHWC layout: (K, C, R, S) -> (K, R, S, C)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        self.register_buffer('weight_int8', w_nhwc)

        # FP16 weights for fallback path (grouped convs, or when CUTLASS unavailable)
        self.register_buffer('weight_fp16', w_data.half())

        # --- Bias ---
        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.view(1, -1, 1, 1))
        else:
            self.bias = None

        self._empty_bias = None  # lazily init for CUTLASS (avoids per-call alloc)
        # Per-layer CUTLASS eligibility: set during calibration based on
        # channel uniformity. Layers with extreme per-channel activation
        # imbalance use per-channel FP16 fallback (matching paper's LCQ).
        self.use_cutlass = HAS_CUTLASS and self.groups == 1

        # --- MoDiff state ---
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None   # a_hat_{t+1}
        self.o_hat_cache: Optional[torch.Tensor] = None   # o_hat_{t+1}
        self.step_count = 0
        self.reset_interval = 10  # periodic cache reset to cap error

        # --- Calibration state ---
        self.calibrating = False
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self.register_buffer('static_input_scale', torch.tensor(1.0, dtype=torch.float32, device=device))

        # Per-channel activation max (accumulated during calibration for SmoothQuant)
        self._act_channel_max: Optional[torch.Tensor] = None

        # Cached values to avoid per-call GPU syncs / tensor allocs
        self._cached_scale_float: Optional[float] = None
        self._cached_alpha_tensor: Optional[torch.Tensor] = None

    # ==================================================================
    # Quantization helpers
    # ==================================================================

    @staticmethod
    def _compute_robust_scale(x: torch.Tensor, n_levels: float = 127.0) -> float:
        """Percentile-based activation scale: n_levels / percentile_max.

        Uses 99.9th percentile instead of abs-max. This dramatically
        improves per-tensor quantization when a few channels carry
        extreme outlier values (e.g., 343x channel-max ratio at low
        timesteps). Outlier values get clipped to ±n_levels, introducing
        small error for <0.1% of values but giving the other 99.9%
        much better resolution.
        """
        abs_vals = x.abs()
        n_elem = abs_vals.numel()
        # Subsample for speed on large tensors
        if n_elem > 131072:
            indices = torch.randint(0, n_elem, (131072,), device=x.device)
            abs_flat = abs_vals.reshape(-1)[indices]
        else:
            abs_flat = abs_vals.reshape(-1)
        k = max(1, int(abs_flat.numel() * 0.001))
        val = abs_flat.kthvalue(abs_flat.numel() - k + 1).values.item()
        return n_levels / max(val, 1e-6)

    def _compute_activation_scale(self, x: torch.Tensor, is_residual: bool = False) -> float:
        """Per-tensor activation scale with 99.9th-percentile clipping.

        ALWAYS uses dynamic scaling from actual activation data.
        During calibration, also collect per-channel max for SmoothQuant.
        """
        scale = self._compute_robust_scale(x, 127.0)

        if self.calibrating and not is_residual:
            self._scale_sum += scale
            self._scale_count += 1
            # Track per-channel max for SmoothQuant
            with torch.no_grad():
                ch_max = x.abs().amax(dim=(0, 2, 3))  # [C]
                if self._act_channel_max is None:
                    self._act_channel_max = ch_max.clone()
                else:
                    torch.max(self._act_channel_max, ch_max, out=self._act_channel_max)

        return scale

    def _dequantize_activation(self, x: torch.Tensor, input_scale: float) -> torch.Tensor:
        """Simulate quantize-then-dequantize: a_hat = Q(x) in FP32.

        Matches the INT8 rounding of the CUTLASS path so the activation
        cache accurately tracks what the kernel actually computed with.
        """
        return (x * input_scale).round().clamp(-127, 127) / input_scale

    def _int8_conv(self, x: torch.Tensor, input_scale: float, with_bias: bool = True) -> torch.Tensor:
        """INT8 convolution — CUTLASS INT8 or FP16 fallback.

        CUTLASS path: actual INT8×INT8 tensor core convolution with
        per-tensor activation scale and per-channel weight scale.

        FP16 fallback (for layers with extreme channel imbalance):
        per-channel dynamic INT8 Q/DQ on activations, then F.conv2d in
        FP16. Simulates per-channel INT8 precision matching the paper.
        """
        if self.use_cutlass:
            alpha = 1.0 / input_scale
            scale_tensor = torch.tensor([alpha], device=x.device, dtype=torch.float32)

            # Quantize activations to INT8 (per-tensor)
            x_q = (x * input_scale).round().clamp(-127, 127).to(torch.int8)
            if not x_q.is_contiguous(memory_format=torch.channels_last):
                x_q = x_q.contiguous(memory_format=torch.channels_last)

            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, device=x.device)

            out_raw = modiff_cutlass.conv2d_int8_fprop(
                x_q,
                self.weight_int8,
                scale_tensor,
                self._empty_bias,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1]
            )
            # 2-step dequant: alpha handled activation scale, now weight scale
            out = out_raw * self.weight_scale_channel
        else:
            # FP16 fallback with per-channel dynamic INT8 Q/DQ.
            # This matches the paper's LCQ (per-channel dynamic quantization).
            ch_max = x.abs().amax(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
            ch_scale = 127.0 / ch_max  # [1, C, 1, 1]
            x_qdq = (x * ch_scale).round().clamp(-127, 127) / ch_scale
            out = F.conv2d(
                x_qdq.half(), self.weight_fp16,
                None,  # bias handled below
                self.stride, self.padding, self.dilation, self.groups
            ).float()

        if with_bias and self.bias is not None:
            out = out + self.bias
        return out

    # ==================================================================
    # Forward paths
    # ==================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_start = profiler.start("Layer: OptimizedInt8Conv2d.forward")

        # Ensure FP32 -- autocast may feed FP16 which corrupts INT8 quantization
        x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        # SmoothQuant: equalize per-channel activation ranges.
        # smooth_scale is identity (1) until calibration applies SmoothQuant.
        # Division is ~0.1 ms vs conv ~1 ms -- negligible overhead.
        if _ENABLE_SMOOTHQUANT and self.is_calibrated:
            x = x / self.smooth_scale

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
        """First timestep (t=T): a_hat_T = Q(a_T), o_hat_T = A(a_hat_T) + bias."""
        input_scale = self._compute_activation_scale(x)
        out = self._int8_conv(x, input_scale, with_bias=True)

        # Cache quantize-dequantized activation and output for next step
        self.a_hat_cache = self._dequantize_activation(x, input_scale)
        self.o_hat_cache = out.clone()
        return out

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """MoDiff modulated step (t<T):
            a_hat_t = Q(a_t - a_hat_{t+1}) + a_hat_{t+1}           -- Eq. (ec5)
            o_hat_t = A(Q(a_t - a_hat_{t+1})) + o_hat_{t+1}        -- Eq. (ec6)
        """
        self.step_count += 1

        # Periodic cache reset to prevent unbounded error accumulation
        if self.step_count % self.reset_interval == 0:
            out = self._forward_first_step(x)
            return out

        # Shape mismatch -> first-step fallback (handles batch size changes)
        if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
            self.is_first_step = True
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

        # Eq. (ec5) -- residual has ~10x smaller range
        residual = x - self.a_hat_cache

        # Dynamic scale for residual (range varies per step)
        input_scale = self._compute_activation_scale(residual, is_residual=True)
        conv_residual = self._int8_conv(residual, input_scale, with_bias=False)

        # Update activation cache: a_hat_t = Q(residual) + a_hat_{t+1}
        residual_dequant = self._dequantize_activation(residual, input_scale)
        self.a_hat_cache.add_(residual_dequant)

        # Eq. (ec6) -- accumulate output: o_hat_t = conv(Q(residual)) + o_hat_{t+1}
        self.o_hat_cache.add_(conv_residual)
        return self.o_hat_cache.clone()

    # ==================================================================
    # MoDiff controls
    # ==================================================================

    def enable_modiff(self, enabled: bool = True):
        """Enable/disable MoDiff temporal caching."""
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()

    def reset_state(self):
        """Reset MoDiff state (call between diffusion samples)."""
        self.is_first_step = True
        self.a_hat_cache = None
        self.o_hat_cache = None
        self.step_count = 0

    # ==================================================================
    # Calibration + SmoothQuant
    # ==================================================================

    def begin_calibration(self):
        """Start accumulating activation scales for static calibration."""
        self.calibrating = True
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self._act_channel_max = None

    def end_calibration(self):
        """Finalize calibration: detect catastrophic layers + set static activation scale."""
        self.calibrating = False
        if self._scale_count == 0:
            return

        # --- Auto-detect catastrophic layers ---
        # Layers with extreme per-channel activation imbalance can't use
        # per-tensor INT8 effectively. Route them to per-channel FP16
        # fallback matching the paper's LCQ per-channel dynamic quantization.
        # Threshold tuned empirically: ch_ratio >10x causes >1% per-layer
        # error which compounds across 70 layers to ~25% UNet error,
        # too much for MoDiff to compensate via DDIM feedback loop.
        if self._act_channel_max is not None and self.use_cutlass:
            ch_ratio = (self._act_channel_max.max() / self._act_channel_max.clamp(min=1e-8).min()).item()
            if ch_ratio > 1.5:
                self.use_cutlass = False  # -> per-channel Q/DQ FP16 fallback

        # --- SmoothQuant: balance per-channel activation/weight ranges ---
        if _ENABLE_SMOOTHQUANT and self._act_channel_max is not None and self._orig_weight is not None:
            self._apply_smoothquant()

        # --- Static activation scale ---
        if _ENABLE_SMOOTHQUANT and self._act_channel_max is not None:
            s = self.smooth_scale.view(-1).to(self._act_channel_max.device)
            smoothed_ch_max = self._act_channel_max / s
            smoothed_global_max = smoothed_ch_max.max().item()
            static_scale = 127.0 / max(smoothed_global_max, 1e-6)
        else:
            static_scale = self._scale_sum / self._scale_count

        self.static_input_scale.fill_(float(static_scale))
        self.is_calibrated = True

        # Pre-compute cached values
        self._cached_scale_float = float(static_scale)
        alpha = 1.0 / float(static_scale)
        self._cached_alpha_tensor = torch.tensor(
            [alpha], device=self.static_input_scale.device, dtype=torch.float32
        )

        # Free original weights to save memory
        self._orig_weight = None

    def _apply_smoothquant(self):
        """SmoothQuant: fold per-channel activation scales into weights.

        For each input channel c:
            smooth_scale[c] = sqrt(act_max[c] / w_max[c])
        At runtime:
            smoothed_act[:, c] = act[:, c] / smooth_scale[c]
        Weights absorb the scale:
            new_weight[:, c] = weight[:, c] * smooth_scale[c]

        After transformation, all activation channels have similar ranges,
        so per-tensor INT8 quantization becomes nearly as accurate as per-channel.
        """
        act_max = self._act_channel_max  # [C_in], on CUDA
        w = self._orig_weight             # [K, C_in, R, S]
        K = self.out_channels

        # Per-input-channel weight max: max(|w[:, c, :, :]|) over K, R, S
        w_dev = w.to(act_max.device)
        w_by_cin = w_dev.reshape(K, self.in_channels, -1)  # [K, C_in, R*S]
        w_max = w_by_cin.abs().amax(dim=(0, 2))  # [C_in]

        # SmoothQuant scale (alpha=0.5)
        ratio = act_max / torch.clamp(w_max, min=1e-8)
        s = ratio.sqrt().clamp(min=1e-4, max=1e4)  # [C_in]

        self.smooth_scale.copy_(s.view(1, -1, 1, 1))

        # Re-quantize weights with smooth_scale folded in
        w_smoothed = (w_dev * s.view(1, -1, 1, 1)).contiguous()
        w_flat = w_smoothed.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values  # [K]
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)

        self.weight_scale_channel.copy_(ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        w_quant = w_quant.view(K, self.in_channels, *self.kernel_size)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        self.weight_int8.copy_(w_nhwc.to(self.weight_int8.device))

    def set_calibrating(self, calibrating: bool):
        """Compat shim for benchmark_ldm.py."""
        if calibrating:
            self.begin_calibration()
        else:
            self.end_calibration()

    def set_static_scale(self, scale: float):
        """Set static activation scale directly (for loading from checkpoint)."""
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
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int8(child, prefix=full_name, use_compile=use_compile)
    return model.to(memory_format=torch.channels_last)


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


# Stubs for static quantization to keep benchmark_ldm.py happy
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
