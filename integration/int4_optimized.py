
import torch
import torch.nn as nn
from typing import Optional
from integration.profiler import profiler
print(f"DEBUG: int4_optimized loaded. Profiler ID: {id(profiler)}")

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

        # 1. Weight quantization and packing
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

        # Store per-channel scale for 2-step dequantization
        self.register_buffer('weight_scale_channel', ch_scale)  # [K]

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
        w_quant = w_quant.view_as(w_data)

        # Permute to Physical NHWC (K, R, S, C)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        
        # Pack
        if self.in_channels % 2 == 0:
            self.weight_packed = pack_int4(w_nhwc)
        else:
            raise ValueError(f"Input channels {self.in_channels} not divisible by 2")

        if conv.bias is not None:
             self.register_buffer('bias', conv.bias.data)
        else:
             self.bias = None

        # MoDiff state
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None  # â_{t+1}: dequantized activation cache
        self.o_hat_cache: Optional[torch.Tensor] = None  # ô_{t+1}: output cache
        self.step_count = 0
        self.reset_interval = 10  # Periodic cache reset every K steps to cap error accumulation

    def _compute_activation_scale(self, x: torch.Tensor) -> float:
        """Compute per-tensor activation scale using max-based quantization.

        Max-based avoids clipping any values, which is important for INT4
        where each clipped value loses significant signal. The slightly wider
        bins are compensated by MoDiff's residual quantization at subsequent
        timesteps (residuals have ~10x smaller range).
        """
        abs_max = x.abs().max().item()
        return 7.0 / max(abs_max, 1e-6)

    def _int4_conv(self, x: torch.Tensor, input_scale: float, with_bias: bool = True) -> torch.Tensor:
        """Core INT4 quantize + CUTLASS conv + per-channel dequantization.

        Uses 2-step dequantization for per-channel weight quantization:
          Step 1 (CUTLASS): out_raw = (1/input_scale) * int32_accum
          Step 2 (PyTorch): out = out_raw * weight_scale_channel + bias

        This is mathematically equivalent to:
          out[k] = sum_{c,r,s} (a_q / input_scale) * (w_q * ch_scale[k]) + bias[k]
        which gives exact per-channel dequantization.
        """
        # Step 1: CUTLASS with alpha = 1/input_scale (undo activation scaling)
        # No bias — handled in Step 2 for per-channel correctness
        alpha = 1.0 / input_scale
        scale_tensor = torch.tensor([alpha], device=x.device, dtype=torch.float32)

        x_scaled = x * input_scale
        x_packed = modiff_cutlass.quantize_and_pack(x_scaled)

        out_raw = modiff_cutlass.conv2d_int4_fprop(
             x_packed,
             self.weight_packed,
             scale_tensor,
             torch.empty(0, device=x.device),  # no bias in CUTLASS
             self.stride[0], self.stride[1],
             self.padding[0], self.padding[1],
             self.dilation[0], self.dilation[1]
        )

        # Step 2: Per-channel weight dequantization + bias
        # out_raw[k] = (1/input_scale) * sum(a_q * w_q[k])
        # out[k] = out_raw[k] * ch_scale[k] + bias[k]
        out = out_raw * self.weight_scale_channel.view(1, -1, 1, 1)
        if with_bias and self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

    def _dequantize_activation(self, x: torch.Tensor, input_scale: float) -> torch.Tensor:
        """Simulate quantize-dequantize to get â = Q(x) in FP32.

        Matches the rounding behaviour of the CUDA quantize_and_pack kernel
        so the cache accurately tracks what CUTLASS actually computed with.
        """
        return (x * input_scale).round().clamp(-7, 7) / input_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_start = profiler.start("Layer: OptimizedInt4Conv2d.forward")

        if not HAS_CUTLASS:
             raise RuntimeError("modiff_cutlass backend missing.")

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
        input_scale = self._compute_activation_scale(residual)
        conv_residual = self._int4_conv(residual, input_scale, with_bias=False)

        # Update activation cache: â_t = Q(residual) + â_{t+1}
        residual_dequant = self._dequantize_activation(residual, input_scale)
        self.a_hat_cache = self.a_hat_cache + residual_dequant

        # Eq. (ec6) — accumulate output: ô_t = conv(Q(residual)) + ô_{t+1}
        self.o_hat_cache = self.o_hat_cache + conv_residual
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

def convert_model_to_optimized_int4(model: nn.Module, prefix: str = "", use_compile: bool = False) -> nn.Module:
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, OptimizedInt4Conv2d):
            if child.in_channels < 32:
                 continue
            # Skip sensitive layers that destroy signal quality when quantized:
            # 1. Skip connections (residual path - critical for signal preservation)
            # 2. Final output projection (directly affects output quality)
            # 3. First input projection (sets the initial signal range)
            is_skip = 'skip' in name
            is_output = (prefix == '' and name == 'out') or name == 'out'
            # Check if this is the top-level output conv (e.g. model.out.2)
            is_final_out = full_name.startswith('out.')
            
            if is_skip or is_final_out:
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
    """Set calibration mode (INT4 uses dynamic per-tensor quantization)."""
    pass  # INT4 uses dynamic quantization — no static calibration needed
