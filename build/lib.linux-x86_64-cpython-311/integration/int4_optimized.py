
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from integration.profiler import profiler

# Try to import the compiled extension
try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: modiff_cutlass extension not found. Please compile it using setup.py.")

def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack int8 tensor to packed int4 (2 elements per byte).
    Input: Int8 Tensor usually [K, C, R, S] or [N, H, W, C]
    Output: Int8 Tensor (uint8 view) with last dim halved.
    """
    # Assume input is Int8 range [-8, 7]
    shape = list(tensor.shape)
    last_dim = shape[-1]
    
    if last_dim % 2 != 0:
        raise ValueError(f"Last dimension {last_dim} must be divisible by 2 for INT4 packing")
    
    # Reshape to separate adjacent pairs
    new_shape = shape[:-1] + [last_dim // 2, 2]
    reshaped = tensor.view(new_shape)
    
    low = reshaped[..., 0] & 0x0F
    high = (reshaped[..., 1] & 0x0F) << 4
    
    packed = (low | high).to(torch.int8) 
    return packed

class OptimizedInt4Conv2d(nn.Module):
    """
    CUTLASS-based INT4 Conv2d with MoDiff Error-Compensated Modulation.

    Implements the MoDiff paper's core equations:
        First step (t=T):
            â_T = Q(a_T)                              -- Eq. (ec1)
            ô_T = Conv(â_T) + bias                     -- Eq. (ec2)
        Subsequent steps (t<T):
            â_t = Q(a_t - â_{t+1}) + â_{t+1}          -- Eq. (ec5)
            ô_t = Conv(Q(a_t - â_{t+1})) + ô_{t+1}    -- Eq. (ec6)

    Key insight: The residual (a_t - â_{t+1}) has ~10x smaller range,
    enabling INT4 quantization with much less error.
    """
    def __init__(self, conv: nn.Conv2d, layer_name: str = "", use_compile: bool = False):
        super().__init__()
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        
        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self.groups = conv.groups

        # Keep FP16 weights for quality-preserving activation-quantized conv.
        # MoDiff primarily targets activation quantization quality; naive W4
        # PTQ without a calibrated checkpoint can cause strong blur.
        self.register_buffer('weight_fp16', conv.weight.data.half())

        # Optional packed INT4 weights path (kept for compatibility/experiments).
        # Conv weight is (K, C, R, S).
        w_data = conv.weight.data
        K = self.out_channels

        # Per-channel max-based weight quantization (along output channel dim).
        # Real model weights have very different scales per output channel —
        # per-tensor max wastes INT4 bins on outlier channels (cos drops to 0.85),
        # per-tensor 3-sigma clips outliers. Per-channel max gives each channel
        # its own scale so every channel uses the full INT4 dynamic range.
        w_flat = w_data.view(K, -1)  # [K, C*R*S]
        ch_max = w_flat.abs().max(dim=1).values  # [K]
        ch_scale = torch.clamp(ch_max / 7.0, min=1e-8)  # [K]

        # Pre-reshape for broadcast: [1, K, 1, 1] avoids .view() per forward
        self.register_buffer('weight_scale_channel', ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
        w_quant = w_quant.view_as(w_data)

        # Permute to Physical NHWC (K, R, S, C)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        
        # Pack
        if self.in_channels % 2 == 0:
            self.weight_packed = pack_int4(w_nhwc)
        else:
            self.weight_packed = None

        if conv.bias is not None:
             self.register_buffer('bias', conv.bias.data.view(1, -1, 1, 1))
        else:
             self.bias = None

        # Pre-allocated empty tensor for CUTLASS bias arg (avoids alloc per call)
        self._empty_bias = None  # lazily initialized on first forward

        # MoDiff state
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None  # â_{t+1}: dequantized activation cache
        self.o_hat_cache: Optional[torch.Tensor] = None  # ô_{t+1}: output cache
        self.step_count = 0
        self.reset_interval = 10  # Periodic cache reset every K steps to cap error accumulation

        # Calibration state for static activation scaling
        self.calibrating = False
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self.register_buffer('static_input_scale', torch.tensor(1.0, dtype=torch.float32))

        # Cached values to avoid per-call GPU syncs and tensor allocs
        self._cached_scale_float: Optional[float] = None
        self._cached_alpha_tensor: Optional[torch.Tensor] = None

    def _compute_robust_scale(self, x: torch.Tensor) -> float:
        """Compute robust activation scale with 99.9th-percentile clipping."""
        flat = x.detach().abs().reshape(-1)
        # Subsample for speed on large feature maps.
        if flat.numel() > 131072:
            stride = max(flat.numel() // 131072, 1)
            flat = flat[::stride]
        clip = torch.quantile(flat, 0.999).item()
        return 7.0 / max(clip, 1e-6)

    def _compute_activation_scale(self, x: torch.Tensor, is_residual: bool = False) -> float:
        """Compute per-tensor activation scale using max-based quantization.

        Max-based avoids clipping any values, which is important for INT4
        where each clipped value loses significant signal. The slightly wider
        bins are compensated by MoDiff's residual quantization at subsequent
        timesteps (residuals have ~10x smaller range).

        For MoDiff residuals, we always use dynamic max scaling because their
        range varies significantly per step.
        """
        if self.calibrating:
            scale = self._compute_robust_scale(x)
            if not is_residual:
                self._scale_sum += scale
                self._scale_count += 1
            return scale

        if (is_residual or not self.is_calibrated):
            abs_max = x.abs().max().item()
            return 7.0 / max(abs_max, 1e-6)

        # Use cached Python float to avoid .item() GPU sync every call
        if self._cached_scale_float is None:
            self._cached_scale_float = float(self.static_input_scale.item())
        return self._cached_scale_float

    def _int4_conv(self, x: torch.Tensor, input_scale: float, with_bias: bool = True) -> torch.Tensor:
        """INT4 forward path: CUTLASS INT4 when available, otherwise safe fallback."""
        if HAS_CUTLASS and self.weight_packed is not None and self.groups == 1:
            alpha = 1.0 / input_scale

            # Only reuse cached tensor when alpha actually matches the static
            # scale (i.e. standard forward, not residual/dynamic).  Residual
            # convolutions have a completely different dynamic scale so using
            # the cached static alpha would corrupt their output by ~10x.
            if (self._cached_alpha_tensor is not None
                    and self._cached_scale_float is not None
                    and input_scale == self._cached_scale_float):
                scale_tensor = self._cached_alpha_tensor
            else:
                scale_tensor = torch.tensor([alpha], device=x.device, dtype=torch.float32)

            x_scaled = x * input_scale
            x_packed = modiff_cutlass.quantize_and_pack(x_scaled)

            # Lazy init empty bias tensor
            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, device=x.device)

            out_raw = modiff_cutlass.conv2d_int4_fprop(
                x_packed,
                self.weight_packed,
                scale_tensor,
                self._empty_bias,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1]
            )

            # Per-channel weight dequantization
            out = out_raw * self.weight_scale_channel
        else:
            # Quantize-dequantize activations to emulate INT4 activation precision.
            # Used when CUTLASS backend is unavailable.
            x_qdq = self._dequantize_activation(x, input_scale)
            out = F.conv2d(
                x_qdq.half(),
                self.weight_fp16,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            ).float()

        if with_bias and self.bias is not None:
            out = out + self.bias
        return out

    def _dequantize_activation(self, x: torch.Tensor, input_scale: float) -> torch.Tensor:
        """Simulate quantize-dequantize to get â = Q(x) in FP32.

        Matches the rounding behaviour of the CUDA quantize_and_pack kernel
        so the cache accurately tracks what CUTLASS actually computed with.
        """
        return (x * input_scale).round().clamp(-7, 7) / input_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_start = profiler.start("Layer: OptimizedInt4Conv2d.forward")

        # Ensure FP32 — autocast may feed FP16 from upstream ops and the
        # CUTLASS kernel calls data_ptr<float>() which would crash or
        # silently reinterpret FP16 bits as garbage FP32 values.
        x = x.float()

        # Ensure channels_last for CUTLASS NHWC path
        if not x.is_contiguous(memory_format=torch.channels_last):
             x = x.contiguous(memory_format=torch.channels_last)

        if not self.modiff_enabled:
            output = self._forward_standard(x)
        elif self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
        else:
            output = self._forward_modulated(x)

        profiler.stop("Layer: OptimizedInt4Conv2d.forward", fwd_start)
        return output

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard INT4 forward without MoDiff modulation."""
        input_scale = self._compute_activation_scale(x)
        return self._int4_conv(x, input_scale, with_bias=True)

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        First timestep (t=T):
            â_T = Q(a_T)                   -- Eq. (ec1)
            ô_T = Conv(â_T) + bias          -- Eq. (ec2)
        """
        input_scale = self._compute_activation_scale(x)
        out = self._int4_conv(x, input_scale, with_bias=True)

        # Cache dequantized activation and output for next step
        self.a_hat_cache = self._dequantize_activation(x, input_scale)
        self.o_hat_cache = out.clone()
        return out

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoDiff modulated step (t<T):
            â_t = Q(a_t - â_{t+1}) + â_{t+1}           -- Eq. (ec5)
            ô_t = Conv(Q(a_t - â_{t+1})) + ô_{t+1}     -- Eq. (ec6)

        Key insight: the residual (a_t - â_{t+1}) has ~10x smaller range than
        a_t, so INT4 quantization on the residual has ~10x less quantization
        error.  This is the core benefit of MoDiff.

        Periodic cache reset: Every `reset_interval` steps, the caches are
        discarded and a fresh first-step is computed. This caps accumulated
        quantization error at O(√K) instead of O(√N) where K << N.
        """
        self.step_count += 1

        # Periodic cache reset to prevent unbounded error accumulation.
        # After K modulated steps, accumulated error grows as O(√K).
        # Resetting every reset_interval steps caps the max error.
        if self.step_count % self.reset_interval == 0:
            out = self._forward_first_step(x)
            return out

        # Shape mismatch → fall back to first-step (handles batch size changes)
        if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
            self.is_first_step = True
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

        # Eq. (ec5) — compute residual, which has ~10x smaller range
        residual = x - self.a_hat_cache

        # Quantize residual + conv (no bias — bias was already added at t=T)
        # Residuals ALWAYS use dynamic max-scaling for best accuracy
        input_scale = self._compute_activation_scale(residual, is_residual=True)
        conv_residual = self._int4_conv(residual, input_scale, with_bias=False)

        # Update activation cache: â_t = Q(residual) + â_{t+1}
        residual_dequant = self._dequantize_activation(residual, input_scale)
        self.a_hat_cache.add_(residual_dequant)

        # Eq. (ec6) — accumulate output in-place: ô_t = conv(Q(residual)) + ô_{t+1}
        self.o_hat_cache.add_(conv_residual)
        return self.o_hat_cache.clone()

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
        self.step_count = 0  # CRITICAL: Reset step counter to prevent incorrect periodic resets

    def begin_calibration(self):
        """Start accumulating activation scales for static calibration."""
        self.calibrating = True
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0

    def end_calibration(self):
        """Finalize static activation scale from accumulated stats."""
        self.calibrating = False
        if self._scale_count > 0:
            avg_scale = self._scale_sum / self._scale_count
            self.static_input_scale.fill_(float(avg_scale))
            self.is_calibrated = True
            # Pre-compute cached values
            self._cached_scale_float = float(avg_scale)
            alpha = 1.0 / float(avg_scale)
            self._cached_alpha_tensor = torch.tensor([alpha], device=self.static_input_scale.device, dtype=torch.float32)

    def set_static_scale(self, scale: float):
        """Set static activation scale directly (for loading from checkpoint)."""
        self.static_input_scale.fill_(float(scale))
        self.is_calibrated = True
        # Pre-compute cached values
        self._cached_scale_float = float(scale)
        alpha = 1.0 / float(scale)
        self._cached_alpha_tensor = torch.tensor([alpha], device=self.static_input_scale.device, dtype=torch.float32)

def convert_model_to_optimized_int4(model: nn.Module, prefix: str = "", use_compile: bool = False) -> nn.Module:
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, OptimizedInt4Conv2d):
            if child.in_channels < 32:
                continue
            # Skip sensitive layers that destroy signal quality when quantized:
            # 1. Skip connections (residual path - critical for signal preservation)
            # 2. Final output projection (directly affects output quality)
            # 3. 1x1 projections (channel mixing/projection layers are highly sensitive)
            # 4. Grouped/depthwise convs (unsupported/fragile in current INT4 path)
            is_skip = 'skip' in name
            # Check if this is the top-level output conv (e.g. model.out.2)
            is_final_out = full_name.startswith('out.')
            is_pointwise = child.kernel_size == (1, 1)
            is_grouped = child.groups != 1
            
            if is_skip or is_final_out or is_pointwise or is_grouped:
                continue
                
            optimized_conv = OptimizedInt4Conv2d(child, layer_name=full_name, use_compile=use_compile)
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int4(child, prefix=full_name, use_compile=use_compile)
    return model.to(memory_format=torch.channels_last)

def enable_modiff_mode(model: nn.Module, enabled: bool = True):
    """Enable/disable MoDiff temporal caching for all INT4 layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            module.enable_modiff(enabled)


def reset_modiff_state(model: nn.Module):
    """Reset MoDiff state for all INT4 layers (call between samples)."""
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            module.reset_state()


def set_calibrating_int4(model: nn.Module, calibrating: bool):
    """Enable/disable calibration mode for INT4 layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            if calibrating:
                module.begin_calibration()
            else:
                module.end_calibration()


def export_int4_static_scales(model: nn.Module) -> Dict[str, float]:
    """Export calibrated INT4 activation scales by layer name."""
    scales = {}
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d) and module.is_calibrated:
            scales[module.layer_name] = float(module.static_input_scale.item())
    return scales


def apply_int4_static_scales(model: nn.Module, scales: Dict[str, float]) -> int:
    """Apply precomputed INT4 activation scales by layer name."""
    loaded = 0
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            if module.layer_name in scales:
                module.set_static_scale(scales[module.layer_name])
                loaded += 1
    return loaded
