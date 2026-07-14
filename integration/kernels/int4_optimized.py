
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from integration.utils.profiler import profiler

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
    CUTLASS-based INT4 Conv2d with SmoothQuant + MoDiff Error-Compensated Modulation.

    Architecture follows the MoDiff paper (Gao et al., ICML 2025):
    - INT4 weight x INT4 activation via CUTLASS tensor core kernels
    - SmoothQuant migrates per-channel activation range differences into
      the weights, so per-tensor activation quantization (needed by the
      INT4 matmul HW) becomes nearly as accurate as per-channel.
    - MoDiff error-compensated modulation across diffusion timesteps
      prevents temporal error accumulation.

    Equations (from the paper):
        t=T (first step):
            a_hat_T = Q(a_T)                                    -- Eq. (ec1)
            o_hat_T = A(a_hat_T) + bias                         -- Eq. (ec2)
        t<T (modulated steps):
            a_hat_t = Q(a_t - a_hat_{t+1}) + a_hat_{t+1}        -- Eq. (ec5)
            o_hat_t = A(Q(a_t - a_hat_{t+1})) + o_hat_{t+1}     -- Eq. (ec6)

    The residual (a_t - a_hat_{t+1}) has ~10x smaller range than a_t, so
    INT4 quantization error is dramatically reduced on modulated steps.
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
        # No .clone(): see the matching comment in int8_optimized.py.
        self.register_buffer('_orig_weight', w_data, persistent=False)

        # --- Per-output-channel symmetric INT4 weight quantization ---
        w_flat = w_data.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values  # [K]
        ch_scale = torch.clamp(ch_max / 7.0, min=1e-8)  # [K]
        self.register_buffer('weight_scale_channel', ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
        w_quant = w_quant.reshape_as(w_data)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()

        # Pack INT4 (2 values per byte) — registered buffer so .to() moves it
        if self.in_channels % 2 == 0:
            self.register_buffer('weight_packed', pack_int4(w_nhwc))
        else:
            self.register_buffer('weight_packed', torch.empty(0, dtype=torch.int8))

        # --- Bias ---
        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.view(1, -1, 1, 1))
        else:
            self.bias = None

        self._empty_bias = None
        self.use_cutlass = HAS_CUTLASS and self.groups == 1 and self.in_channels % 2 == 0

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
        # Persistent scratch for the cast-free fp16 quantize in _forward_standard
        # (see there): a zeroed a_hat lets the fused step1 kernel consume fp16
        # activations directly, avoiding a per-layer fp16->fp32 cast.
        self._zero_ahat_buf: Optional[torch.Tensor] = None
        self._empty_smooth: Optional[torch.Tensor] = None

        # --- SmoothQuant identity flag for fast path ---
        self._smooth_is_identity = True

        # --- SiLU fusion: set by fused_resblock.py's wire_silu_fusion() when
        # this layer directly follows a ResBlock's GroupNorm (i.e. it's a
        # ResBlock in_conv/out_conv). See OptimizedInt8Conv2d's identical flag
        # for the full rationale.
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

    def _cache_dtype(self) -> torch.dtype:
        return torch.float16 if self.is_calibrated else torch.float32

    def _new_cache_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            tensor.shape, device=tensor.device, dtype=self._cache_dtype()
        ).contiguous(memory_format=torch.channels_last)

    def _module_output(self) -> torch.Tensor:
        # See OptimizedInt8Conv2d._module_output: forcing fp32 here just to have
        # the next fp16-autocast op cast back down again is a wasted full-tensor
        # copy. Return the cache (fp16 when calibrated) as-is.
        return self.o_hat_cache

    # ==================================================================
    # Quantization helpers
    # ==================================================================

    def _compute_activation_scale(self, x: torch.Tensor, is_residual: bool = False) -> float:
        """Per-tensor symmetric activation scale: 7 / max(|x|).
        Used during calibration and first-step only (slow path with .item() sync).
        """
        if self.calibrating:
            abs_max = x.abs().max().item()
            scale = 7.0 / max(abs_max, 1e-6)
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
            return 7.0 / max(abs_max, 1e-6)

        if self._cached_scale_float is None:
            self._cached_scale_float = float(self.static_input_scale.item())
        return self._cached_scale_float

    def _compute_scale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """GPU-only per-tensor scale computation. No .item() sync.
        Returns 1-element GPU tensor = 7.0 / max(|x|, 1e-6).
        Used on the modulated hot path to avoid CPU-GPU synchronization.
        """
        abs_max = x.abs().amax()
        return 7.0 / torch.clamp(abs_max, min=1e-6)

    def _dequantize_activation(self, x: torch.Tensor, input_scale) -> torch.Tensor:
        """Simulate quantize-then-dequantize: a_hat = Q(x) in FP32.
        input_scale can be float or 1-element tensor.
        """
        return (x * input_scale).round().clamp(-7, 7) / input_scale

    def _int4_conv(self, x: torch.Tensor, input_scale, with_bias: bool = True) -> torch.Tensor:
        """INT4 x INT4 convolution via CUTLASS tensor core kernel.
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
                x_packed = modiff_cutlass.quantize_and_pack(x_scaled)
            else:
                # Tensor path: use fused scale+quantize+pack kernel (no CPU sync)
                scale_tensor = (1.0 / input_scale).view(1)
                if not x.is_contiguous(memory_format=torch.channels_last):
                    x = x.contiguous(memory_format=torch.channels_last)
                x_packed = modiff_cutlass.scale_quantize_and_pack(x, input_scale)

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
            out = out_raw * self.weight_scale_channel
        else:
            raise RuntimeError(
                f"CUTLASS INT4 kernel unavailable for layer {self.layer_name} "
                f"(groups={self.groups}, in_ch={self.in_channels}). "
                f"Build modiff_cutlass extension."
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

    def _int4_conv_dynamic_fused(self, x: torch.Tensor, with_bias: bool = True) -> torch.Tensor:
        """Cache-free dynamic (uncalibrated) INT4 conv: fuses the absmax
        reduction + scale/inv_scale computation into one kernel
        (dynamic_quantize_pack_int4_fprop -> compute_dynamic_scale), instead
        of the generic _int4_conv's tensor-scale path, which does the
        reduction via a plain `.abs().amax()` PyTorch call and a separate
        `1.0/scale` reciprocal. See int8_optimized.py's
        _int8_conv_dynamic_fused for the identical INT8 rationale.
        """
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_dynamic_buffers(x)

        x_packed = modiff_cutlass.dynamic_quantize_pack_int4_fprop(
            x, self._dyn_absmax_buf, self._dyn_scale_buf,
            self._dyn_inv_scale_buf, self._dyn_retire_count
        )

        if self._empty_bias is None or self._empty_bias.device != x.device:
            self._empty_bias = torch.empty(0, device=x.device)

        out_raw = modiff_cutlass.conv2d_int4_fprop(
            x_packed,
            self.weight_packed,
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

    def _int4_conv_fused(self, x: torch.Tensor, scale: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
        """Optimized INT4 conv for modulated path: scale and inv_scale already computed on GPU.
        No .item() sync, no 1/scale computation kernel. Uses device pointer alpha for CUTLASS.
        Returns RAW (unscaled) CUTLASS output — caller applies weight_scale_channel via scale_accumulate.
        """
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        x_packed = modiff_cutlass.scale_quantize_and_pack(x, scale)

        if self._empty_bias is None or self._empty_bias.device != x.device:
            self._empty_bias = torch.empty(0, device=x.device)

        return modiff_cutlass.conv2d_int4_fprop(
            x_packed,
            self.weight_packed,
            inv_scale.view(1),
            self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

    # ==================================================================
    # Forward paths
    # ==================================================================

    def _can_fuse_input_silu(self, x: torch.Tensor) -> bool:
        """See OptimizedInt8Conv2d._can_fuse_input_silu for the rationale."""
        return (self.fuse_input_silu and self.modiff_enabled and not self.is_first_step
                and self.is_calibrated and HAS_CUTLASS and self.use_cutlass
                and self.a_hat_cache is not None
                and self.a_hat_cache.dtype == torch.float16
                and self.a_hat_cache.shape == x.shape
                and x.dtype == torch.float16)

    def _forward_modulated_static_fused_silu(self, x: torch.Tensor) -> torch.Tensor:
        """Same as _forward_modulated's calibrated CUTLASS branch, but `x` is
        the pre-activation input -- SiLU is applied inline inside
        step1_static_quantize_pack_int4_fprop_silu's CUDA kernel instead of a
        separate F.silu(x) Python call over the whole activation tensor first.
        """
        self.step_count += 1
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        if self._cached_alpha_tensor is None or self._cached_alpha_tensor.device != x.device:
            scale = float(self.static_input_scale.item())
            self._cached_scale_float = scale
            self._cached_alpha_tensor = torch.tensor([1.0 / scale], device=x.device, dtype=torch.float32)
        if not hasattr(self, '_smooth_inv_flat') or self._smooth_inv_flat.device != x.device:
            if not self._smooth_is_identity:
                self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
            else:
                self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

        p_step1 = profiler.start("MoDiff INT4 Static Step1 (fused SiLU)")
        x_packed = modiff_cutlass.step1_static_quantize_pack_int4_fprop_silu(
            x,
            self.a_hat_cache,
            self.static_input_scale.view(1),
            self._smooth_inv_flat,
        )
        profiler.stop("MoDiff INT4 Static Step1 (fused SiLU)", p_step1)

        p_conv = profiler.start("MoDiff INT4 Static Conv2d")
        modiff_cutlass.conv2d_int4_fprop_o_hat(
            x_packed,
            self.weight_packed,
            self._cached_alpha_tensor.view(1),
            self.weight_scale_channel.view(-1),
            self.o_hat_cache,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        profiler.stop("MoDiff INT4 Static Conv2d", p_conv)
        return self._module_output()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_start = profiler.start("Layer: OptimizedInt4Conv2d.forward")

        if self.fuse_input_silu:
            if self._can_fuse_input_silu(x):
                output = self._forward_modulated_static_fused_silu(x)
                profiler.stop("Layer: OptimizedInt4Conv2d.forward", fwd_start)
                return output
            # Fast path not applicable this call -- caller passed pre-activation
            # input expecting this layer to apply SiLU itself, so do it explicitly.
            x = F.silu(x)

        # See OptimizedInt8Conv2d.forward for the rationale: the calibrated MoDiff
        # modulated path's kernel (step1_static_quantize_pack_int4_fprop) now reads
        # fp16 x directly, so skip the upfront full-tensor fp32 cast there. Other
        # paths (calibration, uncalibrated dynamic MoDiff) still use fp32-only
        # kernels and keep it.
        if x.dtype != torch.float32 and (self.calibrating or (self.modiff_enabled and not self.is_calibrated)):
            x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        if not self.modiff_enabled:
            # SmoothQuant: equalize per-channel activation ranges
            if not self._smooth_is_identity:
                x = x * self._smooth_inv
            output = self._forward_standard(x)
        elif self.is_first_step:
            if not self._smooth_is_identity:
                x = x * self._smooth_inv
            output = self._forward_first_step(x)
            self.is_first_step = False
        else:
            # Modulated path: SmoothQuant is fused into sub_absmax_scale kernel
            output = self._forward_modulated(x)

        profiler.stop("Layer: OptimizedInt4Conv2d.forward", fwd_start)
        return output

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard INT4 forward without MoDiff modulation.

        When static scales are available (is_calibrated=True), uses the same
        fused CUDA kernels as the MoDiff modulated path:
            scale_quantize_and_pack → conv2d_int4_fprop
        This is the only fair baseline against which to measure temporal caching overhead.

        When not calibrated, falls back to the naive PyTorch path.
        """
        if self.is_calibrated and HAS_CUTLASS and self.use_cutlass:
            self._ensure_conv_caches(x.device)
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
            if x.dtype == torch.float16:
                # Cast-free quantize: scale_quantize_and_pack reinterpret-casts its
                # input as float4 and so requires fp32, forcing an fp16->fp32 cast
                # here whose bandwidth (2x traffic + an extra launch) exceeds the
                # quantize itself. step1_static_quantize_pack_int4_fprop consumes
                # fp16 directly; a zeroed a_hat makes it quantize x as-is,
                # bit-identically to scale_quantize_and_pack(x.float(), scale). Any
                # SmoothQuant scaling is already applied to x upstream, so smooth
                # is empty (identity) here.
                # NOTE: step1_static_quantize_pack_int4_fprop UPDATES a_hat in
                # place, so the buffer must be re-zeroed every call -- otherwise
                # the 2nd+ step quantizes against a stale a_hat and silently
                # corrupts the output.
                if (self._zero_ahat_buf is None
                        or self._zero_ahat_buf.shape != x.shape
                        or self._zero_ahat_buf.device != x.device):
                    self._zero_ahat_buf = torch.zeros_like(x)
                else:
                    self._zero_ahat_buf.zero_()
                if self._empty_smooth is None or self._empty_smooth.device != x.device:
                    self._empty_smooth = torch.empty(0, device=x.device, dtype=torch.float32)
                x_packed = modiff_cutlass.step1_static_quantize_pack_int4_fprop(
                    x, self._zero_ahat_buf, self.static_input_scale.view(1), self._empty_smooth
                )
            else:
                x_for_quant = x if x.dtype == torch.float32 else x.float()
                x_packed = modiff_cutlass.scale_quantize_and_pack(x_for_quant, self._cached_scale_tensor)
            return self._conv_from_int4(x_packed, x.shape[2], x.shape[3])
        # Fallback: during calibration we need the host-visible scale path so the
        # module can accumulate static activation statistics (the .item() sync in
        # _compute_activation_scale is required there). Outside calibration we use
        # the fully-fused cache-free dynamic-scale kernel (see
        # _int4_conv_dynamic_fused) -- this used to call _compute_activation_scale
        # here too, which cost a CPU-GPU sync on every uncalibrated forward call
        # for no reason (the tensor path below never needed a host-visible float).
        if self.calibrating:
            input_scale = self._compute_activation_scale(x)
            return self._int4_conv(x, input_scale, with_bias=True)
        return self._int4_conv_dynamic_fused(x, with_bias=True)

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

    def _conv_from_int4(self, x_packed: torch.Tensor, h_in: int, w_in: int) -> torch.Tensor:
        """Run the calibrated INT4 conv (dequant/bias/store dispatch) on an
        already-quantized+packed activation ([N, H, W, C/2], the layout produced by
        scale_quantize_and_pack). Shared by _forward_standard (which quantizes first)
        and forward_from_int4 (which skips the quantize). h_in/w_in are the conv
        input's spatial dims (== x_packed.shape[1:3])."""
        self._ensure_conv_caches(x_packed.device)
        h_out = ((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x_packed.shape[0], self.out_channels, h_out, w_out)
        bias_fused = False
        if self.standard_output_fp16:
            if (self._standard_output_buf is None
                    or self._standard_output_buf.shape != output_shape
                    or self._standard_output_buf.device != x_packed.device
                    or self._standard_output_buf.dtype != torch.float16):
                self._standard_output_buf = torch.empty(
                    output_shape, device=x_packed.device, dtype=torch.float16
                ).contiguous(memory_format=torch.channels_last)
            if self.bias is not None and hasattr(modiff_cutlass, "conv2d_int4_fprop_no_ohat_prealloc_bias"):
                out = modiff_cutlass.conv2d_int4_fprop_no_ohat_prealloc_bias(
                    x_packed, self.weight_packed, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), self.bias.view(-1).contiguous(),
                    self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
                bias_fused = True
            else:
                out = modiff_cutlass.conv2d_int4_fprop_no_ohat_prealloc(
                    x_packed, self.weight_packed, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
        else:
            out_raw = modiff_cutlass.conv2d_int4_fprop(
                x_packed, self.weight_packed, self._cached_alpha_tensor, self._empty_bias,
                self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1])
            out = out_raw * self.weight_scale_channel
        if self.bias is not None and not bias_fused:
            bias = self.bias.to(out.dtype) if out.dtype != self.bias.dtype else self.bias
            out = out + bias
        return out

    def forward_from_int4(self, x_packed: torch.Tensor, h_in: int, w_in: int) -> torch.Tensor:
        """Baseline fast path: the activation is already quantized+packed to int4
        (SiLU applied upstream by the GN->int4 fusion in fused_resblock.py), so skip
        the per-layer quantize+pack and go straight to the conv. Only valid when
        calibrated + not modiff_enabled."""
        if not x_packed.is_contiguous():
            x_packed = x_packed.contiguous()
        return self._conv_from_int4(x_packed, h_in, w_in)

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """First timestep (t=T): warm-up with repeated quantisation.

        Paper Appendix B.6: 4-5 warm-up steps converge error.

        Still needs fp32 x (unlike _forward_modulated's calibrated hot path):
        _int4_conv's tensor-scale branch calls scale_quantize_and_pack, which
        reads its input via a vectorized float2 pointer cast and would read
        garbage given fp16 memory. Only runs once per layer per sample() call.
        """
        if x.dtype != torch.float32:
            x = x.float()
        input_scale = self.static_input_scale if self.is_calibrated else self._compute_activation_scale(x)
        a_hat = self._dequantize_activation(x, input_scale)
        o_hat = self._int4_conv(x, input_scale, with_bias=True)

        for _ in range(self.warmup_steps - 1):
            residual = x - a_hat
            r_scale = input_scale if self.is_calibrated else self._compute_activation_scale(residual, is_residual=True)
            conv_r  = self._int4_conv(residual, r_scale, with_bias=False)
            r_dq    = self._dequantize_activation(residual, r_scale)
            a_hat   = a_hat + r_dq
            o_hat   = o_hat + conv_r

        cache_dtype = self._cache_dtype()
        self.a_hat_cache = a_hat.to(cache_dtype).contiguous(memory_format=torch.channels_last)
        self.o_hat_cache = o_hat.to(cache_dtype).contiguous(memory_format=torch.channels_last)
        return self._module_output()

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """MoDiff modulated step (t<T).  No periodic reset per paper.
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
        if self.a_hat_cache.dtype != self._cache_dtype():
            self.is_first_step = True
            if not self._smooth_is_identity:
                x = x * self._smooth_inv
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

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

            p_step1 = profiler.start("MoDiff INT4 Static Step1")
            x_packed = modiff_cutlass.step1_static_quantize_pack_int4_fprop(
                x,
                self.a_hat_cache,
                self.static_input_scale.view(1),
                self._smooth_inv_flat,
            )
            profiler.stop("MoDiff INT4 Static Step1", p_step1)

            p_conv = profiler.start("MoDiff INT4 Static Conv2d")
            modiff_cutlass.conv2d_int4_fprop_o_hat(
                x_packed,
                self.weight_packed,
                self._cached_alpha_tensor.view(1),
                self.weight_scale_channel.view(-1),
                self.o_hat_cache,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1]
            )
            profiler.stop("MoDiff INT4 Static Conv2d", p_conv)
            return self._module_output()

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

        # Kernel 1 Fused C++ Backend Call:
        # Fuses sub_absmax_scale, scale_quantize_and_pack, and dequant_accumulate into 1 python launch.
        p_step1 = profiler.start("MoDiff INT4 Fused Step1")
        x_packed = modiff_cutlass.step1_quantize_pack_int4_fprop(
            x, self.a_hat_cache, self._residual_buf,
            self._absmax_buf, self._scale_buf, self._inv_scale_buf,
            self._retire_count, 7.0, self._smooth_inv_flat
        )
        profiler.stop("MoDiff INT4 Fused Step1", p_step1)

        p_conv = profiler.start("MoDiff INT4 Fused Conv2d")
        modiff_cutlass.conv2d_int4_fprop_o_hat(
            x_packed,
            self.weight_packed,
            self._inv_scale_buf.view(1),
            self.weight_scale_channel.view(-1),
            self.o_hat_cache,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        profiler.stop("MoDiff INT4 Fused Conv2d", p_conv)
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
            static_scale = 7.0 / max(smoothed_global_max, 1e-6)
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
        ch_scale = torch.clamp(ch_max / 7.0, min=1e-8)

        self.weight_scale_channel.copy_(ch_scale.view(1, K, 1, 1))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
        w_quant = w_quant.reshape(K, self.in_channels, *self.kernel_size)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()

        if self.in_channels % 2 == 0:
            self.weight_packed.data = pack_int4(w_nhwc).to(self.weight_packed.device)

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


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

def convert_model_to_optimized_int4(model: nn.Module, prefix: str = "", use_compile: bool = False,
                                     skip_pointwise: bool = True) -> nn.Module:
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, OptimizedInt4Conv2d):
            if child.in_channels < 32:
                continue
            # Require even in_channels for INT4 packing
            if child.in_channels % 2 != 0:
                continue
            is_skip = 'skip' in name
            is_final_out = full_name.startswith('out.')
            is_pointwise = child.kernel_size == (1, 1)
            is_grouped = child.groups != 1

            if is_skip or is_final_out or is_grouped:
                continue
            if is_pointwise and skip_pointwise:
                continue

            optimized_conv = OptimizedInt4Conv2d(child, layer_name=full_name, use_compile=use_compile)
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized_conv = optimized_conv.to(target_device)
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int4(child, prefix=full_name, use_compile=use_compile,
                                             skip_pointwise=skip_pointwise)

    # Convert remaining layers to channels_last for PyTorch perf,
    # then restore weight_packed buffers which must stay standard-contiguous
    # (CUTLASS reads raw memory in packed NHWC row-major order).
    # Only at the top-level call: this function recurses, and re-running the
    # whole-subtree conversion at every nesting level would re-scramble and
    # re-fix every already-fixed descendant's weight_packed once per level of
    # nesting instead of once overall.
    if not prefix:
        model = model.to(memory_format=torch.channels_last)
        for m in model.modules():
            if isinstance(m, OptimizedInt4Conv2d):
                m.weight_packed.data = m.weight_packed.data.contiguous()
    return model


# ---------------------------------------------------------------------------
# Global helpers
# ---------------------------------------------------------------------------

def enable_modiff_mode(model: nn.Module, enabled: bool = True):
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            module.enable_modiff(enabled)


def reset_modiff_state(model: nn.Module):
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            module.reset_state()


def set_standard_output_fp16(model: nn.Module, enabled: bool = True):
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            module.set_standard_output_fp16(enabled)


def set_calibrating_int4(model: nn.Module, calibrating: bool):
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            if calibrating:
                module.begin_calibration()
            else:
                module.end_calibration()


def export_int4_static_scales(model: nn.Module) -> Dict[str, float]:
    scales = {}
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d) and module.is_calibrated:
            scales[module.layer_name] = float(module.static_input_scale.item())
    return scales


def apply_int4_static_scales(model: nn.Module, scales: Dict[str, float]) -> int:
    loaded = 0
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            if module.layer_name in scales:
                module.set_static_scale(scales[module.layer_name])
                loaded += 1
    return loaded
