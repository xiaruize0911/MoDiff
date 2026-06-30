"""
CUTLASS INT8 Linear Layer with MoDiff Error-Compensated Modulation.

For the time-embedding linear layers in the UNet (typically [batch, 768] -> [batch, 384/768/1536]),
torch._int_mm is slower than F.linear due to small M dimensions (batch_size < 32).

This implementation uses FP16 F.linear for the actual GEMM (fastest for small matrices)
while implementing MoDiff's error-compensated temporal caching for quality improvement.
When CUTLASS is available, the modulated path uses fused CUDA kernels
(sub_absmax_scale, dequant_accumulate_int8) on reshaped [B,D,1,1] channels_last
tensors, reducing ~12 kernel launches to ~5.

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

try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False


class OptimizedInt8Linear(nn.Module):
    """
    FP16-accelerated Linear layer with MoDiff temporal caching.

    Uses per-tensor symmetric INT8 quantization for activation cache tracking
    and FP16 F.linear for actual computation (fastest for small M dimensions).

    When modiff_cutlass is available, uses fused CUDA kernels for the
    residual computation and cache updates (sub_absmax_scale, dequant_accumulate_int8)
    to minimize kernel launch overhead.
    """

    def __init__(self, linear: nn.Linear, layer_name: str = "",
                 backend: str = "fp16", int_gemm_min_m: int = 64):
        super().__init__()
        self.layer_name = layer_name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.backend = backend
        self.int_gemm_min_m = int_gemm_min_m

        # Store weights in FP16 for fast matmul
        self.register_buffer('weight_fp16', linear.weight.data.half())

        # Store per-tensor INT8 weights for the optional true INT GEMM backend.
        weight_fp32 = linear.weight.data.float()
        weight_absmax = weight_fp32.abs().max()
        weight_scale = torch.clamp(weight_absmax / 127.0, min=1e-8)
        weight_int8_t = torch.round(weight_fp32 / weight_scale).clamp(-128, 127).to(torch.int8).t().contiguous()
        self.register_buffer('weight_int8_t', weight_int8_t)
        self.register_buffer('weight_int8_awq', weight_int8_t.t().contiguous())
        self.register_buffer('weight_dequant_scale', weight_scale.float().reshape(1))

        # --- Bias ---
        if linear.bias is not None:
            self.register_buffer('bias', linear.bias.data.half())
        else:
            self.bias = None

        # --- MoDiff state ---
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        self.step_count = 0
        self.warmup_steps = 3

        # --- Calibration state ---
        self.calibrating = False
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self.register_buffer('static_input_scale', torch.tensor(1.0, dtype=torch.float32))
        self._cached_scale_float: Optional[float] = None
        self._cached_scale_tensor: Optional[torch.Tensor] = None

        # --- Fused kernel persistent buffers (lazy-initialized) ---
        self._residual_buf: Optional[torch.Tensor] = None
        self._scale_buf: Optional[torch.Tensor] = None
        self._inv_scale_buf: Optional[torch.Tensor] = None
        self._absmax_buf: Optional[torch.Tensor] = None
        self._retire_count: Optional[torch.Tensor] = None
        self._smooth_inv_flat: Optional[torch.Tensor] = None
        self.standard_output_fp16 = False

    # ==================================================================
    # Quantization helpers
    # ==================================================================

    def _compute_activation_scale(self, x: torch.Tensor, is_residual: bool = False) -> float:
        """Per-tensor symmetric activation scale: 127 / max(|x|)."""
        if self.calibrating:
            abs_max = x.abs().max().item()
            scale = 127.0 / max(abs_max, 1e-6)
            if not is_residual:
                self._scale_sum += scale
                self._scale_count += 1
            return scale

        if is_residual or not self.is_calibrated:
            abs_max = x.abs().max().item()
            return 127.0 / max(abs_max, 1e-6)

        if self._cached_scale_float is None:
            self._cached_scale_float = float(self.static_input_scale.item())
        return self._cached_scale_float

    def _dequantize_activation(self, x: torch.Tensor, input_scale: float) -> torch.Tensor:
        """Simulate quantize-then-dequantize: a_hat = Q(x)."""
        return (x * input_scale).round().clamp(-127, 127) / input_scale

    def _fp16_linear(self, x: torch.Tensor, with_bias: bool = True) -> torch.Tensor:
        """Fast FP16 linear. Input can be FP32 (auto-cast) or FP16."""
        x_fp16 = x.half() if x.dtype != torch.float16 else x
        out = F.linear(x_fp16, self.weight_fp16, self.bias if with_bias else None)
        return out if self.standard_output_fp16 else out.float()

    def _int8_gemm_linear(self, x: torch.Tensor, with_bias: bool = True,
                          input_scale: Optional[float | torch.Tensor] = None) -> torch.Tensor:
        """True W8A8 GEMM path: quantize activation, int8 GEMM, dequantize."""
        x_2d = x.reshape(-1, self.in_features)
        if x_2d.shape[0] < self.int_gemm_min_m or not x_2d.is_cuda:
            return self._fp16_linear(x, with_bias=with_bias)

        from modiff_triton.kernels.gemm_w8a8 import gemm_w8a8

        if input_scale is None:
            abs_max = x_2d.abs().amax()
            act_dequant_scale = torch.clamp(abs_max / 127.0, min=1e-8)
        else:
            # Existing calibration stores quantization scale (127 / absmax).
            if isinstance(input_scale, torch.Tensor):
                act_dequant_scale = 1.0 / torch.clamp(input_scale.float(), min=1e-8)
            else:
                act_dequant_scale = torch.tensor(
                    1.0 / max(float(input_scale), 1e-8),
                    device=x_2d.device,
                    dtype=torch.float32,
                )

        x_int8 = torch.round(x_2d / act_dequant_scale).clamp(-128, 127).to(torch.int8)
        bias = self.bias if (with_bias and self.bias is not None) else None
        out = gemm_w8a8(
            x_int8,
            self.weight_int8_t,
            act_dequant_scale,
            self.weight_dequant_scale,
            bias,
        )
        out = out.reshape(*x.shape[:-1], self.out_features)
        return out.half() if self.standard_output_fp16 else out

    def _awq_gemm_linear(self, x: torch.Tensor, with_bias: bool = True,
                         input_scale: Optional[float | torch.Tensor] = None) -> torch.Tensor:
        """AWQ W8A8 GEMM path used by benchmark_ldm.py when --linear_backend awq."""
        x_2d = x.reshape(-1, self.in_features)
        if (
            x_2d.shape[0] < self.int_gemm_min_m
            or not x_2d.is_cuda
            or self.out_features % 2 != 0
        ):
            return self._fp16_linear(x, with_bias=with_bias)

        from modiff_triton.kernels.awq_w8a8 import (
            awq_fused_quant_gemm_w8a8,
            awq_gemm_w8a8,
        )

        bias = self.bias if (with_bias and self.bias is not None) else None
        if input_scale is None:
            out = awq_fused_quant_gemm_w8a8(
                x_2d,
                self.weight_int8_awq,
                self.weight_dequant_scale,
                bias,
                weight_is_awq_layout=True,
            )
        else:
            # Existing calibration stores q_scale = 127 / absmax. AWQ GEMM
            # consumes dequant scale = absmax / 127.
            if isinstance(input_scale, torch.Tensor):
                q_scale = input_scale.to(device=x_2d.device, dtype=torch.float32)
            else:
                q_scale = torch.tensor(float(input_scale), device=x_2d.device, dtype=torch.float32)
            act_dequant_scale = 1.0 / torch.clamp(q_scale, min=1e-8)
            x_int8 = torch.round(x_2d.float() / act_dequant_scale).clamp(-128, 127).to(torch.int8)
            scale_a = act_dequant_scale.reshape(1).expand(x_2d.shape[0]).to(torch.float16)
            out = awq_gemm_w8a8(
                x_int8,
                self.weight_int8_awq,
                self.weight_dequant_scale,
                scale_a,
                bias,
                weight_is_awq_layout=True,
            )

        out = out.reshape(*x.shape[:-1], self.out_features)
        return out if self.standard_output_fp16 else out.float()

    def _linear(self, x: torch.Tensor, with_bias: bool = True,
                input_scale: Optional[float | torch.Tensor] = None) -> torch.Tensor:
        if self.backend == "int_gemm":
            return self._int8_gemm_linear(x, with_bias=with_bias, input_scale=input_scale)
        if self.backend == "awq":
            return self._awq_gemm_linear(x, with_bias=with_bias, input_scale=input_scale)
        return self._fp16_linear(x, with_bias=with_bias)

    def _quant_dequant_int8(self, x: torch.Tensor, input_scale: Optional[float] = None):
        """Return dequantized activation and the quantization scale used by legacy MoDiff code."""
        if input_scale is None:
            q_scale = self._compute_activation_scale(x)
        else:
            q_scale = input_scale
        a_hat = self._dequantize_activation(x, q_scale)
        return a_hat, q_scale

    # ==================================================================
    # Forward paths
    # ==================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.modiff_enabled:
            return self._linear(x, with_bias=True)

        if x.dtype != torch.float32:
            x = x.float()

        if self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
            return output

        return self._forward_modulated(x)

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """First timestep (t=T): warm-up with repeated quantisation.

        Computes a_hat_T = Q(a_T) and o_hat_T = A(a_hat_T) + bias using
        iterative refinement (warmup_steps iterations).
        """
        input_scale = self._cached_scale_float if self.is_calibrated else self._compute_activation_scale(x)
        if input_scale is None:
            input_scale = float(self.static_input_scale.item())
        a_hat = self._dequantize_activation(x, input_scale)
        o_hat = self._linear(a_hat, with_bias=True, input_scale=input_scale)

        for _ in range(self.warmup_steps - 1):
            residual = x - a_hat
            r_scale = input_scale if self.is_calibrated else self._compute_activation_scale(residual, is_residual=True)
            r_dq = self._dequantize_activation(residual, r_scale)
            o_hat = o_hat + self._linear(r_dq, with_bias=False, input_scale=r_scale)
            a_hat = a_hat + r_dq

        self.a_hat_cache = a_hat
        self.o_hat_cache = o_hat
        return o_hat.clone()

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """MoDiff modulated step (t<T): error-compensated temporal caching."""
        self.step_count += 1

        if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
            self.is_first_step = True
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

        if self.is_calibrated:
            return self._forward_modulated_static(x)

        if HAS_CUTLASS:
            return self._forward_modulated_fused(x)
        else:
            return self._forward_modulated_fallback(x)

    def _forward_modulated_static(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._cached_scale_float
        if scale is None:
            scale = float(self.static_input_scale.item())
            self._cached_scale_float = scale

        residual = x - self.a_hat_cache
        r_dq = (residual * scale).round().clamp(-127, 127) / scale
        linear_r = self._linear(r_dq, with_bias=False, input_scale=scale)

        self.a_hat_cache.add_(r_dq)
        self.o_hat_cache.add_(linear_r)
        return self.o_hat_cache

    def _forward_modulated_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Modulated path using fused CUTLASS kernels on [B,D,1,1] channels_last."""
        B = x.shape[0]
        D = x.shape[-1]

        # Reshape to 4D channels_last for CUTLASS kernels
        x_4d = x.reshape(B, D, 1, 1).contiguous(memory_format=torch.channels_last)
        a_hat_4d = self.a_hat_cache.reshape(B, D, 1, 1).contiguous(
            memory_format=torch.channels_last)

        # Lazy-init persistent kernel buffers
        if self._residual_buf is None or self._residual_buf.shape != x_4d.shape:
            self._residual_buf = torch.empty_like(x_4d)
            self._scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
            self._inv_scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
            self._absmax_buf = torch.zeros(1, device=x.device, dtype=torch.float32)
            self._retire_count = torch.zeros(1, device=x.device, dtype=torch.int32)
            self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

        # Fused kernel 1: residual = x - a_hat, absmax, scale = 127/absmax
        modiff_cutlass.sub_absmax_scale(
            x_4d, a_hat_4d, self._residual_buf,
            self._absmax_buf, self._scale_buf, self._inv_scale_buf,
            self._retire_count, 127.0, self._smooth_inv_flat
        )

        # Quant-dequant residual + FP16 matmul
        residual_2d = self._residual_buf.reshape(B, D)
        r_dq = (residual_2d * self._scale_buf).round().clamp(-127, 127) * self._inv_scale_buf
        linear_r = self._linear(r_dq, with_bias=False, input_scale=self._scale_buf)

        # Fused kernel 2: dequant + a_hat cache accumulate
        modiff_cutlass.dequant_accumulate_int8(
            self._residual_buf, a_hat_4d, self._scale_buf
        )
        self.a_hat_cache = a_hat_4d.reshape(B, D)

        # Update o_hat cache
        self.o_hat_cache.add_(linear_r)
        return self.o_hat_cache

    def _forward_modulated_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Modulated path using pure PyTorch (fallback without CUTLASS)."""
        residual = x - self.a_hat_cache

        abs_max = residual.abs().amax()
        scale = 127.0 / torch.clamp(abs_max, min=1e-6)
        r_dq = (residual * scale).round().clamp(-127, 127) / scale

        linear_r = self._linear(r_dq, with_bias=False, input_scale=scale)

        self.a_hat_cache.add_(r_dq)
        self.o_hat_cache.add_(linear_r)
        return self.o_hat_cache

    # ==================================================================
    # MoDiff controls
    # ==================================================================

    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()

    def set_standard_output_fp16(self, enabled: bool = True):
        self.standard_output_fp16 = enabled

    def reset_state(self):
        self.is_first_step = True
        self.a_hat_cache = None
        self.o_hat_cache = None
        self.step_count = 0

    # ==================================================================
    # Calibration
    # ==================================================================

    def begin_calibration(self):
        self.calibrating = True
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0

    def end_calibration(self):
        self.calibrating = False
        if self._scale_count == 0:
            return
        static_scale = self._scale_sum / self._scale_count
        self.static_input_scale.fill_(float(static_scale))
        self.is_calibrated = True
        self._cached_scale_float = float(static_scale)

    def set_calibrating(self, calibrating: bool):
        if calibrating:
            self.begin_calibration()
        else:
            self.end_calibration()

    def set_static_scale(self, scale: float):
        self.static_input_scale.fill_(float(scale))
        self.is_calibrated = True
        self._cached_scale_float = float(scale)
        self._cached_scale_tensor = torch.tensor([float(scale)], device=self.static_input_scale.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

def convert_model_to_int8_linear(model: nn.Module, prefix: str = "",
                                  min_features: int = 128,
                                  backend: str = "fp16",
                                  int_gemm_min_m: int = 64) -> nn.Module:
    """Convert nn.Linear layers to OptimizedInt8Linear with MoDiff support."""
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and not isinstance(child, OptimizedInt8Linear):
            if child.in_features < min_features:
                continue
            optimized = OptimizedInt8Linear(
                child,
                layer_name=full_name,
                backend=backend,
                int_gemm_min_m=int_gemm_min_m,
            )
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized = optimized.to(target_device)
            setattr(model, name, optimized)
        else:
            convert_model_to_int8_linear(
                child,
                prefix=full_name,
                min_features=min_features,
                backend=backend,
                int_gemm_min_m=int_gemm_min_m,
            )
    return model


# ---------------------------------------------------------------------------
# Global helpers
# ---------------------------------------------------------------------------

def enable_modiff_mode_linear(model: nn.Module, enabled: bool = True):
    """Enable/disable MoDiff mode for all OptimizedInt8Linear layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt8Linear):
            module.enable_modiff(enabled)


def reset_modiff_state_linear(model: nn.Module):
    """Reset MoDiff temporal caches for all OptimizedInt8Linear layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt8Linear):
            module.reset_state()


def set_standard_output_fp16_linear(model: nn.Module, enabled: bool = True):
    """Return FP16 outputs from baseline OptimizedInt8Linear layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt8Linear):
            module.set_standard_output_fp16(enabled)


def set_calibrating_linear(model: nn.Module, calibrating: bool):
    """Set calibration mode for all OptimizedInt8Linear layers."""
    for module in model.modules():
        if isinstance(module, OptimizedInt8Linear):
            module.set_calibrating(calibrating)


def export_linear_static_scales(model: nn.Module) -> Dict[str, float]:
    """Export static scales from all calibrated OptimizedInt8Linear layers."""
    scales = {}
    for module in model.modules():
        if isinstance(module, OptimizedInt8Linear) and module.is_calibrated:
            scales[module.layer_name] = float(module.static_input_scale.item())
    return scales


def apply_linear_static_scales(model: nn.Module, scales: Dict[str, float]) -> int:
    """Apply pre-calibrated static scales to OptimizedInt8Linear layers."""
    loaded = 0
    for module in model.modules():
        if isinstance(module, OptimizedInt8Linear) and module.layer_name in scales:
            module.set_static_scale(scales[module.layer_name])
            loaded += 1
    return loaded
