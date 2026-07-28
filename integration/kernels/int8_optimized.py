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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from integration.utils.profiler import profiler

try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: modiff_cutlass extension not found.")

# Per-shape CUTLASS tile autotuning for the deep-fuse int8 conv. On first call
# each conv times all tile configs on its actual input and caches the fastest
# (the cuDNN-style per-shape selection). Kill-switch: MODIFF_DISABLE_CONV_AUTOTUNE=1
# reverts to the single fixed 128^3 tile.
_CONV_AUTOTUNE = (os.environ.get("MODIFF_DISABLE_CONV_AUTOTUNE") != "1"
                  and HAS_CUTLASS and hasattr(modiff_cutlass, "conv2d_int8_dequant_fp16_tuned"))


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
        # No .clone(): w_data is conv.weight.data (already detached from autograd),
        # and only ever read (never mutated) by _apply_smoothquant(). Aliasing it
        # avoids a full-weight D2D copy at construction time that is pure waste
        # whenever calibration is loaded from a file instead of run live (the
        # common case — see apply_static_scales, which never calls
        # begin_calibration()/end_calibration() so this buffer would otherwise
        # sit around unused for the life of the model).
        self.register_buffer('_orig_weight', w_data, persistent=False)

        # --- Per-output-channel symmetric INT8 weight quantization ---
        w_flat = w_data.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values  # [K]
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)  # [K]
        self.register_buffer('weight_scale_channel', ch_scale.view(1, K, 1, 1))
        self.register_buffer('weight_scale_channel_half', ch_scale.half().contiguous())

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
        self._cached_scale_tensor: Optional[torch.Tensor] = None  # for _forward_standard fused path
        self.standard_output_fp16 = False
        self._standard_output_buf: Optional[torch.Tensor] = None
        # INT8 conv->conv chaining (TensorRT-style): when output_requant_scale is
        # set (to the NEXT conv's static_input_scale), forward_to_int8 emits int8
        # requantized by it (+ optional fused ReLU), so the next conv reads int8
        # directly with no fp16 round-trip. See integration/fused_ops/chained_bottleneck.py.
        self.output_requant_scale: Optional[torch.Tensor] = None
        self.fuse_output_relu: bool = False
        self._int8_output_buf: Optional[torch.Tensor] = None
        # Autotuned CUTLASS tile config id for the deep-fuse int8 conv (lazy, per
        # this conv's shape). None = not yet tuned; -1 = fixed default tile.
        self._tuned_config_id: Optional[int] = None
        # Persistent scratch for the cast-free fp16 quantize in _forward_standard
        # (see there): a zeroed a_hat lets the fused step1 kernel consume fp16
        # activations directly, avoiding a per-layer fp16->fp32 cast.
        self._zero_ahat_buf: Optional[torch.Tensor] = None
        self._empty_smooth: Optional[torch.Tensor] = None

        # --- SmoothQuant identity flag for fast path ---
        self._smooth_is_identity = True

        # --- SiLU fusion: set by fused_resblock.py's wire_silu_fusion() when
        # this layer directly follows a ResBlock's GroupNorm (i.e. it's a
        # ResBlock in_conv/out_conv). When True, callers pass the *pre-SiLU*
        # activation and this layer applies SiLU itself -- either fused into
        # the quantize kernel (fast path) or via a plain F.silu(x) call
        # (first-step/uncalibrated fallback) -- see forward().
        self.fuse_input_silu = False

        # --- Fused kernel persistent buffers (lazy-initialized) ---
        self._residual_buf: Optional[torch.Tensor] = None
        self._scale_buf: Optional[torch.Tensor] = None
        self._inv_scale_buf: Optional[torch.Tensor] = None
        self._absmax_buf: Optional[torch.Tensor] = None
        self._retire_count: Optional[torch.Tensor] = None

        # --- Dynamic (uncalibrated) baseline buffers: no cache, so these are
        # smaller/separate from the MoDiff buffers above (no _residual_buf,
        # no a_hat/o_hat needed) ---
        self._dyn_scale_buf: Optional[torch.Tensor] = None
        self._dyn_inv_scale_buf: Optional[torch.Tensor] = None
        self._dyn_absmax_buf: Optional[torch.Tensor] = None
        self._dyn_retire_count: Optional[torch.Tensor] = None

    def _ensure_state_buffers(self, x: torch.Tensor):
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        h_out = ((x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x.shape[0], self.out_channels, h_out, w_out)
        cache_dtype = torch.float16 if self.is_calibrated else torch.float32

        if (self.a_hat_cache is None or self.a_hat_cache.shape != x.shape
                or self.a_hat_cache.dtype != cache_dtype):
            self.a_hat_cache = torch.zeros(
                x.shape, device=x.device, dtype=cache_dtype
            ).contiguous(memory_format=torch.channels_last)
        if (self.o_hat_cache is None or self.o_hat_cache.shape != output_shape
                or self.o_hat_cache.dtype != cache_dtype):
            self.o_hat_cache = torch.zeros(
                output_shape, device=x.device, dtype=cache_dtype
            ).contiguous(memory_format=torch.channels_last)

        if self.is_calibrated:
            self._residual_buf = None
        elif self._residual_buf is None or self._residual_buf.shape != x.shape:
            self._residual_buf = torch.empty_like(x)
        if self._scale_buf is None or self._scale_buf.device != x.device:
            self._scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        if self._inv_scale_buf is None or self._inv_scale_buf.device != x.device:
            self._inv_scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        if self._absmax_buf is None or self._absmax_buf.device != x.device:
            self._absmax_buf = torch.zeros(1, device=x.device, dtype=torch.float32)
        if self._retire_count is None or self._retire_count.device != x.device:
            self._retire_count = torch.zeros(1, device=x.device, dtype=torch.int32)

        if not hasattr(self, '_smooth_inv_flat') or self._smooth_inv_flat.device != x.device:
            if not self._smooth_is_identity:
                self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
            else:
                self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

    def _module_output(self) -> torch.Tensor:
        # Used to force-cast a fp16 o_hat_cache up to fp32 here, which meant every
        # calibrated MoDiff conv call materialized a full extra fp32 copy of its
        # output on the way out -- only for the very next op (autocast-managed
        # conv/linear, or our own autocast-disabled GroupNorm+SiLU) to want fp16
        # again anyway. The rest of the fp16-autocast pipeline already tolerates
        # fp16 activations natively, so just return the cache as-is.
        return self.o_hat_cache

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

    def _ensure_dynamic_buffers(self, x: torch.Tensor):
        if self._dyn_scale_buf is None or self._dyn_scale_buf.device != x.device:
            self._dyn_scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
            self._dyn_inv_scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
            self._dyn_absmax_buf = torch.zeros(1, device=x.device, dtype=torch.float32)
            self._dyn_retire_count = torch.zeros(1, device=x.device, dtype=torch.int32)

    def _int8_conv_dynamic_fused(self, x: torch.Tensor, with_bias: bool = True) -> torch.Tensor:
        """Cache-free dynamic (uncalibrated) INT8 conv: fuses the absmax
        reduction + scale/inv_scale computation into one kernel
        (dynamic_quantize_int8_fprop -> compute_dynamic_scale), instead of
        the generic _int8_conv's tensor-scale path, which does the reduction
        via a plain `.abs().amax()` PyTorch call and a separate `1.0/scale`
        reciprocal — this collapses those into the same fused kernel already
        used elsewhere, and avoids ever materializing a residual buffer
        (there is no cache here, so unlike the MoDiff dynamic path there is
        nothing to subtract).
        """
        # dynamic_quantize_int8_fprop's kernels read x via data_ptr<float>(); unlike
        # _int8_conv's tensor-scale branch (which happens to promote to fp32 through
        # `x * input_scale` in its scalar branch, or would hard-error via data_ptr<float>()
        # in its tensor branch), cast explicitly so this path is correct regardless of
        # what dtype the previous layer produced.
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_dynamic_buffers(x)

        x_int8 = modiff_cutlass.dynamic_quantize_int8_fprop(
            x, self._dyn_absmax_buf, self._dyn_scale_buf,
            self._dyn_inv_scale_buf, self._dyn_retire_count
        )

        if self._empty_bias is None or self._empty_bias.device != x.device:
            self._empty_bias = torch.empty(0, device=x.device)

        out_raw = modiff_cutlass.conv2d_int8_fprop(
            x_int8,
            self.weight_int8,
            self._dyn_inv_scale_buf.view(1),
            self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        out = out_raw * self.weight_scale_channel
        if with_bias and self.bias is not None:
            out = out + self.bias
        return out

    # ==================================================================
    # Forward paths
    # ==================================================================

    def _can_fuse_input_silu(self, x: torch.Tensor) -> bool:
        """True when this call can take the fused SiLU+quantize kernel path:
        `x` must be the pre-activation ResBlock GroupNorm output (see
        fuse_input_silu / fused_resblock.py's wire_silu_fusion), not yet
        SiLU'd, and everything the fused kernel requires (calibrated, FP16
        cache, matching shape/dtype) must already hold.
        """
        return (self.fuse_input_silu and self.modiff_enabled and not self.is_first_step
                and self.is_calibrated and HAS_CUTLASS and self.use_cutlass
                and self.a_hat_cache is not None
                and self.a_hat_cache.dtype == torch.float16
                and self.a_hat_cache.shape == x.shape
                and x.dtype == torch.float16)

    def _forward_modulated_static_fused_silu(self, x: torch.Tensor) -> torch.Tensor:
        """Same as _forward_modulated's calibrated CUTLASS branch, but `x` is
        the pre-activation input -- SiLU is applied inline inside
        step1_static_quantize_fprop_silu's CUDA kernel instead of a separate
        F.silu(x) Python call over the whole activation tensor beforehand.
        """
        self.step_count += 1
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_state_buffers(x)

        if self._cached_alpha_tensor is None or self._cached_alpha_tensor.device != x.device:
            scale = float(self.static_input_scale.item())
            self._cached_scale_float = scale
            self._cached_alpha_tensor = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)

        if not hasattr(self, '_smooth_inv_flat') or self._smooth_inv_flat.device != x.device:
            if not self._smooth_is_identity:
                self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
            else:
                self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

        p_step1 = profiler.start("MoDiff INT8 Static Step1 (fused SiLU)")
        x_int8 = modiff_cutlass.step1_static_quantize_fprop_silu(
            x,
            self.a_hat_cache,
            self.static_input_scale.view(1),
            self._smooth_inv_flat,
        )
        profiler.stop("MoDiff INT8 Static Step1 (fused SiLU)", p_step1)

        p_conv = profiler.start("MoDiff INT8 Static Conv2d")
        (modiff_cutlass.conv2d_int8_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int8_fprop_o_hat)(
            x_int8,
            self.weight_int8,
            self._cached_alpha_tensor.view(1),
            self.weight_scale_channel.view(-1),
            self.o_hat_cache,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        profiler.stop("MoDiff INT8 Static Conv2d", p_conv)
        return self._module_output()

    def can_gn_fuse_modiff(self, x: torch.Tensor) -> bool:
        """Eligibility for the fused GroupNorm+SiLU+delta-quantize modiff path
        (group_norm_silu_delta_quantize_nhwc). Same gate as _can_fuse_input_silu
        (fuse_input_silu + calibrated + fp16 cache present + shape/dtype match +
        not first step) plus groups==1 and cuda. The caller (_prequant_gn_conv)
        additionally checks the GN-native conditions (channels_last, C%ng==0)
        and that the kernel is available."""
        return (self._can_fuse_input_silu(x)
                and getattr(self, 'groups', 1) == 1
                and x.is_cuda)

    def forward_gn_fused_modiff(self, x, gn_weight, gn_bias, num_groups, eps,
                                mod_scale2d, mod_shift2d, residual=None):
        """Fused GroupNorm(+scale-shift mod)+SiLU + INT8 temporal-delta quantize
        + o_hat conv, in one GN-quantize kernel + one conv. Replaces the
        standalone GroupNorm kernel + step1_static_quantize_fprop_silu that
        _forward_modulated_static_fused_silu runs back-to-back, removing the fp16
        `normed` round-trip between them. Bit-identical to that two-kernel path
        (the kernel replicates the fp16 rounding of `normed` before SiLU, and the
        a_hat update `cache += q/scale` is unchanged). Caller must have verified
        can_gn_fuse_modiff(x). `mod_scale2d`/`mod_shift2d` are [N,C] (or empty)
        matching x.dtype; `residual` (fp16, or None) is added to the output."""
        self.step_count += 1
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_state_buffers(x)
        if self._cached_alpha_tensor is None or self._cached_alpha_tensor.device != x.device:
            scale = float(self.static_input_scale.item())
            self._cached_scale_float = scale
            self._cached_alpha_tensor = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)

        p_step1 = profiler.start("MoDiff INT8 GN-fused Step1 (GN+SiLU+delta)")
        x_int8 = modiff_cutlass.group_norm_silu_delta_quantize_nhwc(
            x, gn_weight, gn_bias, self.a_hat_cache, num_groups, eps, True,
            self.static_input_scale.view(1), self._smooth_inv_flat,
            mod_scale2d, mod_shift2d)
        profiler.stop("MoDiff INT8 GN-fused Step1 (GN+SiLU+delta)", p_step1)

        if residual is not None:
            # EVT dual-store (same conv2d_int8_evt_o_hat_residual kernel
            # forward_modiff_fused_silu_residual already uses): fold the ResBlock
            # skip-add into the o_hat conv's accumulate epilogue instead of the
            # separate eager `out + residual` below -- removes an elementwise-add
            # kernel AND (for the GN-fusion out_conv->next-block-in_conv edge)
            # restores a direct producer/consumer relationship between this
            # conv's output and the next GroupNorm's input, with no intervening
            # op. Caller's can_gn_fuse_modiff(x) precondition (is_calibrated=True)
            # already guarantees a_hat/o_hat_cache are fp16 (see
            # _ensure_state_buffers), matching forward_modiff_fused_silu_residual's
            # own (unconditional) use of the EVT kernel.
            residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
            out = torch.empty_like(self.o_hat_cache)
            p_conv = profiler.start("MoDiff INT8 Static Conv2d (o_hat+residual)")
            modiff_cutlass.conv2d_int8_evt_o_hat_residual(
                x_int8, self.weight_int8, self._cached_alpha_tensor.view(1),
                self.weight_scale_channel.view(-1), self.o_hat_cache, residual, out,
                self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1])
            profiler.stop("MoDiff INT8 Static Conv2d (o_hat+residual)", p_conv)
            return out

        p_conv = profiler.start("MoDiff INT8 Static Conv2d")
        (modiff_cutlass.conv2d_int8_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int8_fprop_o_hat)(
            x_int8, self.weight_int8, self._cached_alpha_tensor.view(1),
            self.weight_scale_channel.view(-1), self.o_hat_cache,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        profiler.stop("MoDiff INT8 Static Conv2d", p_conv)
        return self._module_output()

    def forward_modiff_fused_silu_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Same as _forward_modulated_static_fused_silu, but fuses the ResBlock
        skip-add into the o_hat conv's accumulate epilogue
        (conv2d_int8_fprop_o_hat_residual): returns o_hat + residual in one pass,
        with NO trailing aten::add and WITHOUT polluting the o_hat cache (the cache
        write is byte-identical to the non-residual path). Caller must have verified
        _can_fuse_input_silu(x). `residual` is the ResBlock skip (cast to fp16
        channels_last here)."""
        self.step_count += 1
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_state_buffers(x)
        if self._cached_alpha_tensor is None or self._cached_alpha_tensor.device != x.device:
            scale = float(self.static_input_scale.item())
            self._cached_scale_float = scale
            self._cached_alpha_tensor = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)

        p_step1 = profiler.start("MoDiff INT8 Static Step1 (fused SiLU)")
        x_int8 = modiff_cutlass.step1_static_quantize_fprop_silu(
            x, self.a_hat_cache, self.static_input_scale.view(1), self._smooth_inv_flat)
        profiler.stop("MoDiff INT8 Static Step1 (fused SiLU)", p_step1)

        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
        out = torch.empty_like(self.o_hat_cache)
        p_conv = profiler.start("MoDiff INT8 Static Conv2d (o_hat+residual)")
        # EVT dual-store: o_hat += acc*alpha*weight_scale (in place) and out = o_hat_new +
        # residual, in ONE conv pass -- removes the fp32 conv_out round-trip of the old
        # conv2d_int8_fprop_o_hat_residual (verified bit-exact o_hat + out; ~1.4-1.8x faster b128).
        modiff_cutlass.conv2d_int8_evt_o_hat_residual(
            x_int8, self.weight_int8, self._cached_alpha_tensor.view(1),
            self.weight_scale_channel.view(-1), self.o_hat_cache, residual, out,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        profiler.stop("MoDiff INT8 Static Conv2d (o_hat+residual)", p_conv)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_start = profiler.start("Layer: OptimizedInt8Conv2d.forward")

        if self.fuse_input_silu:
            if self._can_fuse_input_silu(x):
                output = self._forward_modulated_static_fused_silu(x)
                profiler.stop("Layer: OptimizedInt8Conv2d.forward", fwd_start)
                return output
            # Fast path not applicable this call (first step / uncalibrated /
            # dtype mismatch) -- the caller passed the pre-activation input
            # expecting this layer to apply SiLU itself, so do it explicitly.
            x = F.silu(x)

        # The calibrated MoDiff modulated path's CUDA kernel (step1_static_quantize_fprop)
        # reads fp16 x directly, so skip this cast there -- it used to force a full-tensor
        # fp16->fp32 copy of every activation before every quantized conv call, which cost
        # more GPU time than the quantized conv itself (see FusedGroupNormSiLU's sibling
        # analysis; profiling showed aten::copy_ as the single largest kernel-time bucket
        # in int8/int4 mode). The other paths (calibration, uncalibrated dynamic MoDiff)
        # still use fp32-only kernels, so they keep the upfront cast.
        if x.dtype != torch.float32 and (self.calibrating or (self.modiff_enabled and not self.is_calibrated)):
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
        """Standard INT8 forward without MoDiff modulation.

        When static scales are available (is_calibrated=True), uses the same
        fused CUDA kernels as the MoDiff modulated path:
            scale_quantize_int8 → conv2d_int8_fprop
        This avoids separate PyTorch round/clamp/cast kernels and is the
        only fair baseline against which to measure temporal caching overhead.

        When not calibrated, falls back to the naive PyTorch path (which
        includes a CPU-GPU sync via .item() in _compute_activation_scale).
        """
        if self.is_calibrated and HAS_CUTLASS and self.use_cutlass:
            self._ensure_conv_caches(x.device)
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
            if x.dtype == torch.float16:
                # Cache-free static quantize (baseline: no temporal cache). Reads fp16 x directly
                # (no fp32 cast) and does NOT touch a_hat — dropping the per-call a_hat zero-fill +
                # a_hat read+write that step1_static_quantize_fprop(x, a_hat=0) wasted. Output is
                # bit-identical (residual=x-0=x). SmoothQuant already applied upstream -> smooth empty.
                if self._empty_smooth is None or self._empty_smooth.device != x.device:
                    self._empty_smooth = torch.empty(0, device=x.device, dtype=torch.float32)
                x_int8 = modiff_cutlass.step1_static_quantize_noahat_fprop(
                    x, self.static_input_scale.view(1), self._empty_smooth
                )
            else:
                x_for_quant = x if x.dtype == torch.float32 else x.float()
                x_int8 = modiff_cutlass.scale_quantize_int8(x_for_quant, self._cached_scale_tensor)
            return self._conv_from_int8(x_int8)
        # Fallback: during calibration we need the host-visible scale path so the
        # module can accumulate static activation statistics. Outside calibration
        # we use the fully-fused cache-free dynamic-scale kernel (no cache, no
        # residual -- see _int8_conv_dynamic_fused) instead of _compute_scale_tensor's
        # separate .amax() + reciprocal + _int8_conv's tensor-scale branch, which
        # needlessly cost 2 extra small kernel launches on every uncalibrated call.
        if self.calibrating:
            input_scale = self._compute_activation_scale(x)
            return self._int8_conv(x, input_scale, with_bias=True)
        return self._int8_conv_dynamic_fused(x, with_bias=True)

    def _apply(self, *args, **kwargs):
        """Keep the packed INT8 weight standard-contiguous through any tensor
        transform. `model.to(memory_format=torch.channels_last)` (applied to make
        activations channels_last) also reformats the [K,R,S,C] `weight_int8` buffer
        to a channels_last stride -- which for R,S>1 (3x3 convs) silently transposes
        the physical layout the CUTLASS conv kernel reads, producing garbage output
        (1x1 convs are unaffected: channels_last == contiguous there). This was
        invisible to random-weight consistency and speed checks; only real accuracy
        vs fp16 exposed it. Re-contiguating after the transform costs one small copy."""
        out = super()._apply(*args, **kwargs)
        wi = getattr(self, "weight_int8", None)
        if wi is not None and wi.dim() == 4 and not wi.is_contiguous():
            self.weight_int8 = wi.contiguous()
        return out

    def _ensure_conv_caches(self, device):
        """Lazy-init the per-tensor scale caches reused by the calibrated conv path."""
        if self._cached_scale_float is None:
            self._cached_scale_float = float(self.static_input_scale.item())
        if self._cached_alpha_tensor is None:
            self._cached_alpha_tensor = torch.tensor(
                [1.0 / self._cached_scale_float], device=device, dtype=torch.float32)
        if self._cached_scale_tensor is None:
            self._cached_scale_tensor = torch.tensor(
                [self._cached_scale_float], device=device, dtype=torch.float32)
        if self._empty_bias is None or self._empty_bias.device != device:
            self._empty_bias = torch.empty(0, device=device)
        if self._empty_smooth is None or self._empty_smooth.device != device:
            self._empty_smooth = torch.empty(0, device=device, dtype=torch.float32)

    def _ensure_tuned_config(self, x_int8: torch.Tensor, output_shape) -> Optional[int]:
        """Lazily pick the fastest CUTLASS tile config for this conv's shape by
        timing all configs on the actual int8 input (cuDNN-style per-shape select).
        Cached in _tuned_config_id. Returns -1 (use fixed default tile) when
        autotuning is disabled or the deep-fuse tuned kernel is unavailable."""
        if self._tuned_config_id is not None:
            return self._tuned_config_id
        if not _CONV_AUTOTUNE:
            self._tuned_config_id = -1
            return -1
        ncfg = modiff_cutlass.conv2d_int8_num_tuned_configs()
        buf = torch.empty(output_shape, device=x_int8.device, dtype=torch.float16
                          ).contiguous(memory_format=torch.channels_last)
        wscale_h = self.weight_scale_channel_half.view(-1)
        args = (x_int8, self.weight_int8, self._cached_alpha_tensor, wscale_h, buf)
        strides = (self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                   self.dilation[0], self.dilation[1])
        best_t, best_id = float("inf"), -1
        for cid in range(ncfg):
            try:
                for _ in range(3):
                    modiff_cutlass.conv2d_int8_dequant_fp16_tuned(*args, cid, *strides)
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(10):
                    modiff_cutlass.conv2d_int8_dequant_fp16_tuned(*args, cid, *strides)
                e.record(); torch.cuda.synchronize()
                t = s.elapsed_time(e)
            except Exception:
                continue
            if t < best_t:
                best_t, best_id = t, cid
        self._tuned_config_id = best_id  # -1 if every config failed -> fixed default
        return self._tuned_config_id

    def _conv_from_int8(self, x_int8: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the calibrated INT8 conv (dequant/bias/store dispatch) on an already
        -quantized channels_last int8 activation. Shared by _forward_standard (which
        quantizes first) and forward_from_int8 (which skips the quantize).

        If `residual` (fp16 channels_last, same shape as the conv output) is given,
        it is added in the store epilogue (fusing a ResBlock skip-add) rather than
        as a separate aten::add."""
        self._ensure_conv_caches(x_int8.device)
        h_out = ((x_int8.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x_int8.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x_int8.shape[0], self.out_channels, h_out, w_out)
        bias_fused = False
        residual_fused = False
        if self.standard_output_fp16:
            if (self._standard_output_buf is None
                    or self._standard_output_buf.shape != output_shape
                    or self._standard_output_buf.device != x_int8.device
                    or self._standard_output_buf.dtype != torch.float16):
                self._standard_output_buf = torch.empty(
                    output_shape, device=x_int8.device, dtype=torch.float16
                ).contiguous(memory_format=torch.channels_last)
            deep_ok = (self.out_channels % 8 == 0
                       and hasattr(modiff_cutlass, "conv2d_int8_fprop_deepfuse_bias_residual_fp16"))
            if deep_ok:
                cid = self._ensure_tuned_config(x_int8, output_shape)
                cid = cid if cid is not None else -1
                if self.bias is None and residual is None:
                    # No bias/residual -> deep-fuse writes final fp16 directly (no store pass).
                    if cid >= 0:
                        return modiff_cutlass.conv2d_int8_dequant_fp16_tuned(
                            x_int8, self.weight_int8, self._cached_alpha_tensor,
                            self.weight_scale_channel_half.view(-1), self._standard_output_buf, cid,
                            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                            self.dilation[0], self.dilation[1])
                    return modiff_cutlass.conv2d_int8_fprop_dequant_fp16_prealloc(
                        x_int8, self.weight_int8, self._cached_alpha_tensor,
                        self.weight_scale_channel_half.view(-1), self._standard_output_buf,
                        self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                        self.dilation[0], self.dilation[1])
                # EVT single-pass: acc*alpha*weight_scale[k] + bias[k] + residual[elem] -> fp16,
                # no fp32/fp16 scratch (replaces the deep-fuse scratch + bias_residual_store pass).
                # Beats even the best autotuned deep-fuse cid at b128 (scratch IO dominates the
                # fixed-tile cost); ~fp16-ulp vs the old 2x-rounded path (single-round -> more
                # accurate). weight_scale/bias are read FP32 in the visitor tree.
                if self.bias is not None:
                    if getattr(self, '_evt_bias_f32', None) is None or self._evt_bias_f32.numel() != self.bias.numel():
                        self._evt_bias_f32 = self.bias.view(-1).float().contiguous()
                    bias_arg = self._evt_bias_f32
                else:
                    bias_arg = self._empty_bias
                res_arg = (residual if residual is not None
                           else torch.empty(0, device=x_int8.device, dtype=torch.float16))
                return modiff_cutlass.conv2d_int8_evt_bias_residual_fp16(
                    x_int8, self.weight_int8, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), bias_arg, res_arg,
                    self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
            # Fallback (out_channels % 8 != 0 or kernel unavailable): fp32-temp store paths.
            if residual is not None:
                bias_arg = (self.bias.view(-1).contiguous()
                            if self.bias is not None else self._empty_bias)
                return modiff_cutlass.conv2d_int8_fprop_no_ohat_prealloc_bias_residual(
                    x_int8, self.weight_int8, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), bias_arg, residual,
                    self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
            if self.bias is not None and hasattr(modiff_cutlass, "conv2d_int8_fprop_no_ohat_prealloc_bias"):
                out = modiff_cutlass.conv2d_int8_fprop_no_ohat_prealloc_bias(
                    x_int8, self.weight_int8, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), self.bias.view(-1).contiguous(),
                    self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
                bias_fused = True
            else:
                out = modiff_cutlass.conv2d_int8_fprop_no_ohat_prealloc(
                    x_int8, self.weight_int8, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
        else:
            out_raw = modiff_cutlass.conv2d_int8_fprop(
                x_int8, self.weight_int8, self._cached_alpha_tensor, self._empty_bias,
                self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1])
            out = out_raw * self.weight_scale_channel
        if self.bias is not None and not bias_fused:
            bias = self.bias.to(out.dtype) if out.dtype != self.bias.dtype else self.bias
            out = out + bias
        # Residual not fused in the epilogue (non-fp16 output / kernel unavailable):
        # add it here so behaviour is identical, just unfused.
        if residual is not None and not residual_fused:
            out = out + (residual.to(out.dtype) if out.dtype != residual.dtype else residual)
        return out

    def forward_from_int8(self, x_int8: torch.Tensor,
                          residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Baseline fast path: the activation is already quantized to int8 (with SiLU
        applied upstream, e.g. by the GN->int8 fusion in fused_resblock.py), so skip
        the per-layer quantize (K1) and go straight to the conv. Only valid when
        calibrated + not modiff_enabled. Optional `residual` (fp16 channels_last) is
        fused into the store epilogue as the ResBlock skip-add."""
        if not x_int8.is_contiguous(memory_format=torch.channels_last):
            x_int8 = x_int8.contiguous(memory_format=torch.channels_last)
        return self._conv_from_int8(x_int8, residual=residual)

    def forward_from_int8_dual(self, x_int8: torch.Tensor, residual: torch.Tensor,
                               requant_scale: torch.Tensor, apply_relu: bool = True):
        """Cross-block-chaining conv3: dequant + bias + fp16 skip-residual + ReLU, and
        emit BOTH the fp16 block output (x_{N+1}, for the next block's identity) AND
        that output requantized to int8 by `requant_scale` (= the next block conv1's
        static_input_scale) -- in one fused store. This folds the block-entry quantize
        (the standalone per-block K1) into this conv3's epilogue. Returns
        (out_fp16, out_int8), both channels_last. Requires the deep-fuse dual kernel,
        out_channels%8==0, calibrated + standard_output_fp16."""
        assert self.standard_output_fp16, "dual store requires standard_output_fp16"
        assert hasattr(modiff_cutlass, "conv2d_int8_fprop_deepfuse_bias_residual_dual"), \
            "dual-store kernel unavailable (rebuild the extension)"
        assert self.out_channels % 8 == 0, "dual store requires out_channels%8==0"
        self._ensure_conv_caches(x_int8.device)
        if not x_int8.is_contiguous(memory_format=torch.channels_last):
            x_int8 = x_int8.contiguous(memory_format=torch.channels_last)
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        h_out = ((x_int8.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x_int8.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x_int8.shape[0], self.out_channels, h_out, w_out)
        if (self._standard_output_buf is None
                or self._standard_output_buf.shape != output_shape
                or self._standard_output_buf.device != x_int8.device
                or self._standard_output_buf.dtype != torch.float16):
            self._standard_output_buf = torch.empty(
                output_shape, device=x_int8.device, dtype=torch.float16
            ).contiguous(memory_format=torch.channels_last)
        if (self._int8_output_buf is None
                or self._int8_output_buf.shape != output_shape
                or self._int8_output_buf.device != x_int8.device):
            self._int8_output_buf = torch.empty(
                output_shape, device=x_int8.device, dtype=torch.int8
            ).contiguous(memory_format=torch.channels_last)
        bias_arg = (self.bias.view(-1).contiguous() if self.bias is not None else self._empty_bias)
        cid = self._ensure_tuned_config(x_int8, output_shape)
        modiff_cutlass.conv2d_int8_fprop_deepfuse_bias_residual_dual(
            x_int8, self.weight_int8, self._cached_alpha_tensor,
            self.weight_scale_channel_half.view(-1), bias_arg, residual.half(),
            requant_scale.view(1), self._standard_output_buf, self._int8_output_buf,
            apply_relu, cid if cid is not None else -1,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        return self._standard_output_buf, self._int8_output_buf

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize an fp16/fp32 activation to channels_last int8 using this conv's
        calibrated static_input_scale -- the block-entry K1 for int8 chaining. Reuses
        the cast-free fp16 path from _forward_standard (step1 with a zeroed a_hat)."""
        self._ensure_conv_caches(x.device)
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        if x.dtype == torch.float16:
            if (self._zero_ahat_buf is None
                    or self._zero_ahat_buf.shape != x.shape
                    or self._zero_ahat_buf.device != x.device):
                self._zero_ahat_buf = torch.zeros_like(x)
            else:
                self._zero_ahat_buf.zero_()
            return modiff_cutlass.step1_static_quantize_fprop(
                x, self._zero_ahat_buf, self.static_input_scale.view(1), self._empty_smooth)
        x_for_quant = x if x.dtype == torch.float32 else x.float()
        return modiff_cutlass.scale_quantize_int8(x_for_quant, self._cached_scale_tensor)

    def forward_to_int8(self, x_int8: torch.Tensor, apply_relu: bool = True) -> torch.Tensor:
        """INT8-in, INT8-out conv for chaining: dequant + optional ReLU + requantize
        by output_requant_scale (the next conv's input scale), in one fused kernel,
        so the next conv reads int8 directly. Requires output_requant_scale set,
        calibrated, use_cutlass. Returns a channels_last int8 tensor."""
        self._ensure_conv_caches(x_int8.device)
        assert self.output_requant_scale is not None, "output_requant_scale not wired"
        if not x_int8.is_contiguous(memory_format=torch.channels_last):
            x_int8 = x_int8.contiguous(memory_format=torch.channels_last)
        h_out = ((x_int8.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x_int8.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x_int8.shape[0], self.out_channels, h_out, w_out)
        if (self._int8_output_buf is None
                or self._int8_output_buf.shape != output_shape
                or self._int8_output_buf.device != x_int8.device):
            self._int8_output_buf = torch.empty(
                output_shape, device=x_int8.device, dtype=torch.int8
            ).contiguous(memory_format=torch.channels_last)
        bias_arg = (self.bias.view(-1).contiguous()
                    if self.bias is not None else self._empty_bias)
        # Deep-fuse path: fold per-channel weight_scale into the CUTLASS GEMM
        # epilogue (fp16, no fp32 temporary), then bias+ReLU+requant->int8. Removes
        # the fp32 intermediate the plain path pays. Requires out_channels%8==0.
        if (self.out_channels % 8 == 0
                and hasattr(modiff_cutlass, "conv2d_int8_fprop_deepfuse_relu_requant_int8")):
            cid = self._ensure_tuned_config(x_int8, tuple(self._int8_output_buf.shape))
            return modiff_cutlass.conv2d_int8_fprop_deepfuse_relu_requant_int8(
                x_int8, self.weight_int8, self._cached_alpha_tensor,
                self.weight_scale_channel_half.view(-1), bias_arg,
                self.output_requant_scale.view(1), self._int8_output_buf, apply_relu,
                cid if cid is not None else -1,
                self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1])
        return modiff_cutlass.conv2d_int8_fprop_relu_requant_int8(
            x_int8, self.weight_int8, self._cached_alpha_tensor,
            self.weight_scale_channel.view(-1), bias_arg,
            self.output_requant_scale.view(1), self._int8_output_buf, apply_relu,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """First timestep (t=T): warm-up with repeated quantisation.

        Does not call _ensure_state_buffers(): a_hat/o_hat are computed fresh
        below and adopted directly as the cache (see the assignment at the end),
        so pre-zeroing a persistent buffer just to immediately .copy_() over it
        would be a wasted full-tensor allocation + D2D copy per layer. The other
        scratch buffers _ensure_state_buffers() sets up (_residual_buf,
        _scale_buf, ...) aren't used by this method — they're allocated by
        _forward_modulated()'s own _ensure_state_buffers() call on step 2.

        Unlike _forward_modulated's hot path, this one still needs fp32 x:
        the tensor-scale branch of _int8_conv calls the vectorized
        scale_quantize_int8 kernel, which reinterpret_casts its input as
        float4 and would read garbage (wrong stride, half the needed bytes)
        given fp16 memory. This only runs once per layer per sample() call
        (t=T warm-up), so the cast's cost is negligible next to the N-1
        _forward_modulated calls that now skip it.
        """
        if x.dtype != torch.float32:
            x = x.float()
        if self.is_calibrated:
            input_scale = self.static_input_scale
            if input_scale.device != x.device:
                input_scale = input_scale.to(x.device)
        elif self.calibrating:
            input_scale = self._compute_activation_scale(x)
        else:
            input_scale = self._compute_scale_tensor(x)

        a_hat = self._dequantize_activation(x, input_scale)
        o_hat = self._int8_conv(x, input_scale, with_bias=True)

        for _ in range(self.warmup_steps - 1):
            residual = x - a_hat
            if self.is_calibrated:
                r_scale = input_scale
            elif self.calibrating:
                r_scale = self._compute_activation_scale(residual, is_residual=True)
            else:
                r_scale = self._compute_scale_tensor(residual)
            conv_r = self._int8_conv(residual, r_scale, with_bias=False)
            r_dq = self._dequantize_activation(residual, r_scale)
            a_hat = a_hat + r_dq
            o_hat = o_hat + conv_r

        cache_dtype = torch.float16 if self.is_calibrated else torch.float32
        self.a_hat_cache = a_hat.to(cache_dtype).contiguous(memory_format=torch.channels_last)
        self.o_hat_cache = o_hat.to(cache_dtype).contiguous(memory_format=torch.channels_last)
        return self._module_output()

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

        self._ensure_state_buffers(x)

        if self.is_calibrated and HAS_CUTLASS and self.use_cutlass:
            if self._cached_alpha_tensor is None or self._cached_alpha_tensor.device != x.device:
                scale = float(self.static_input_scale.item())
                self._cached_scale_float = scale
                self._cached_alpha_tensor = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)

            if not hasattr(self, '_smooth_inv_flat') or self._smooth_inv_flat.device != x.device:
                if not self._smooth_is_identity:
                    self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
                else:
                    self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

            p_step1 = profiler.start("MoDiff INT8 Static Step1")
            x_int8 = modiff_cutlass.step1_static_quantize_fprop(
                x,
                self.a_hat_cache,
                self.static_input_scale.view(1),
                self._smooth_inv_flat,
            )
            profiler.stop("MoDiff INT8 Static Step1", p_step1)

            p_conv = profiler.start("MoDiff INT8 Static Conv2d")
            (modiff_cutlass.conv2d_int8_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int8_fprop_o_hat)(
                x_int8,
                self.weight_int8,
                self._cached_alpha_tensor.view(1),
                self.weight_scale_channel.view(-1),
                self.o_hat_cache,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1]
            )
            profiler.stop("MoDiff INT8 Static Conv2d", p_conv)
            return self._module_output()

        # Kernel 1 Fused C++ Backend Call:
        # Fuses sub_absmax_scale, dequant_accumulate, and scale_quantize into 1 python launch.
        p_step1 = profiler.start("MoDiff INT8 Fused Step1")
        x_int8 = modiff_cutlass.step1_quantize_fprop(
            x, self.a_hat_cache, self._residual_buf,
            self._absmax_buf, self._scale_buf, self._inv_scale_buf,
            self._retire_count, 127.0, self._smooth_inv_flat
        )
        profiler.stop("MoDiff INT8 Fused Step1", p_step1)

        p_conv = profiler.start("MoDiff INT8 Fused Conv2d")
        (modiff_cutlass.conv2d_int8_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int8_fprop_o_hat)(
            x_int8,
            self.weight_int8,
            self._inv_scale_buf.view(1),
            self.weight_scale_channel.view(-1),
            self.o_hat_cache,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        profiler.stop("MoDiff INT8 Fused Conv2d", p_conv)
        return self._module_output()

    # ==================================================================
    # MoDiff controls
    # ==================================================================

    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()

    def set_standard_output_fp16(self, enabled: bool = True):
        self.standard_output_fp16 = enabled
        if not enabled:
            self._standard_output_buf = None

    def reset_state(self):
        self.is_first_step = True
        with torch.inference_mode():
            if self.a_hat_cache is not None:
                self.a_hat_cache.zero_()
            if self.o_hat_cache is not None:
                self.o_hat_cache.zero_()
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
        """SmoothQuant: derive the per-in-channel smooth scale from the calibrated
        activation range and fold it into the weights."""
        act_max = self._act_channel_max
        w = self._orig_weight
        K = self.out_channels

        w_dev = w.to(act_max.device)
        w_by_cin = w_dev.reshape(K, self.in_channels, -1)
        w_max = w_by_cin.abs().amax(dim=(0, 2))

        ratio = act_max / torch.clamp(w_max, min=1e-8)
        s = ratio.sqrt().clamp(min=1e-4, max=1e4)

        self._fold_weights_with_smooth(s)

    def _fold_weights_with_smooth(self, s: torch.Tensor):
        """Fold a given per-in-channel SmoothQuant scale `s` ([C_in]) into the weights:
        set smooth_scale, then requantize the *original* fp weights against their smoothed
        per-output-channel range. Shared by _apply_smoothquant (live calibration, `s` from
        activation stats) and set_static_calibration (`s` restored from a checkpoint).
        Requires _orig_weight to still be present."""
        w = self._orig_weight
        K = self.out_channels

        w_dev = w.to(s.device)
        self.smooth_scale.copy_(s.view(1, -1, 1, 1))

        w_smoothed = w_dev * s.view(1, -1, 1, 1)
        w_flat = w_smoothed.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)

        self.weight_scale_channel.copy_(ch_scale.view(1, K, 1, 1))
        self.weight_scale_channel_half.copy_(ch_scale.half().to(self.weight_scale_channel_half.device))

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
        self._cached_scale_tensor = torch.tensor(
            [float(scale)], device=self.static_input_scale.device, dtype=torch.float32
        )

    def set_static_calibration(self, scale: float, smooth_scale: Optional[torch.Tensor] = None):
        """Restore a full static calibration (per-tensor `scale` + optional per-in-channel
        `smooth_scale`) from an exported checkpoint. The smooth scale is re-folded into the
        freshly converted (unsmoothed) weights so smoothed activations meet smoothed
        weights at inference, keeping the SmoothQuant-derived static scale correct. See
        OptimizedInt4Conv2d.set_static_calibration for the full rationale; int8's 8-bit
        range hides the mismatch that the int4 path suffers, but the treatment is mirrored
        for consistency. Falls back to scale-only if the weights can't be re-folded
        (_orig_weight already released)."""
        if smooth_scale is not None and self._orig_weight is not None:
            s = torch.as_tensor(smooth_scale, dtype=torch.float32,
                                device=self.smooth_scale.device).reshape(-1)
            self._fold_weights_with_smooth(s)
            self._smooth_inv.copy_(1.0 / self.smooth_scale)
            self._smooth_is_identity = bool(torch.allclose(
                self._smooth_inv, torch.ones_like(self._smooth_inv), atol=1e-6))
            if hasattr(self, '_smooth_inv_flat'):
                del self._smooth_inv_flat
        self.set_static_scale(scale)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

def convert_model_to_optimized_int8(model: nn.Module, prefix: str = "", use_compile: bool = False,
                                     skip_pointwise: bool = True) -> nn.Module:
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, OptimizedInt8Conv2d):
            if child.in_channels < 32:
                continue
            is_skip = 'skip' in name
            is_final_out = full_name.startswith('out.')
            is_pointwise = child.kernel_size == (1, 1)
            is_grouped = child.groups != 1

            if is_skip or is_final_out or is_grouped:
                continue
            if is_pointwise and skip_pointwise:
                continue

            optimized_conv = OptimizedInt8Conv2d(child, layer_name=full_name, use_compile=use_compile)
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized_conv = optimized_conv.to(target_device)
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int8(child, prefix=full_name, use_compile=use_compile,
                                             skip_pointwise=skip_pointwise)

    # Convert to channels_last for PyTorch perf, then restore weight_int8.
    # Only at the top-level call: this function recurses, and re-running the
    # whole-subtree conversion at every nesting level would re-scramble and
    # re-fix every already-fixed descendant's weight_int8 once per level of
    # nesting instead of once overall.
    if not prefix:
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


def set_standard_output_fp16(model, enabled: bool = True):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.set_standard_output_fp16(enabled)


def set_calibrating(model, calibrating):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.set_calibrating(calibrating)
            if not calibrating and module.is_calibrated:
                _calib_config.update(module.layer_name, float(module.static_input_scale.item()))
    if not calibrating:
        _calib_config.finalize()


def export_int8_static_scales(model: nn.Module) -> Dict[str, object]:
    """Export the static calibration per int8 conv. Identity-SmoothQuant layers export as
    a bare float (legacy format); SmoothQuant layers export as
    ``{"static_scale": float, "smooth_scale": cpu fp32 tensor [C_in]}`` so smoothing can be
    restored on apply. Mirrors export_int4_static_scales (int8's wider range hides the
    quality loss the int4 path suffers, but the round-trip is kept faithful)."""
    scales = {}
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d) and module.is_calibrated:
            if module._smooth_is_identity:
                scales[module.layer_name] = float(module.static_input_scale.item())
            else:
                scales[module.layer_name] = {
                    "static_scale": float(module.static_input_scale.item()),
                    "smooth_scale": module.smooth_scale.detach().to("cpu", torch.float32).reshape(-1).clone(),
                }
    return scales


def apply_static_scales(model, *args, **kwargs):
    """Load static calibration produced by export_int8_static_scales. Accepts both the
    legacy flat ``{name: float}`` format and the richer ``{name: {"static_scale":...,
    "smooth_scale":...}}`` format (mixed is fine)."""
    scales = kwargs.get('scales', None)
    if scales is None and len(args) > 0 and isinstance(args[0], dict):
        scales = args[0]
    if scales is None:
        return 0

    loaded = 0
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d) and module.layer_name in scales:
            entry = scales[module.layer_name]
            if isinstance(entry, dict):
                module.set_static_calibration(entry["static_scale"], entry.get("smooth_scale"))
            else:
                module.set_static_scale(float(entry))
            loaded += 1
    # Keep _calib_config a plain {name: float} map (its historical contract): downstream
    # consumers (benchmark_ldm.py) treat these values as scalars.
    _calib_config.scales = {
        k: (v["static_scale"] if isinstance(v, dict) else v) for k, v in scales.items()
    }
    _calib_config.is_calibrated = True
    return loaded

