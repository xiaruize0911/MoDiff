
import os
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

# Per-shape CUTLASS tile autotuning for the int4 conv (mirror of the int8 tuned tile
# set). Each conv times all tile configs on its actual packed input and caches the
# fastest. Shared kill-switch with int8: MODIFF_DISABLE_CONV_AUTOTUNE=1 -> fixed tile.
_INT4_CONV_AUTOTUNE = (os.environ.get("MODIFF_DISABLE_CONV_AUTOTUNE") != "1"
                       and HAS_CUTLASS and hasattr(modiff_cutlass, "conv2d_int4_fprop_tuned"))

# Deep-fuse int4 dequant store (default ON): fold the per-channel weight_scale into
# the CUTLASS int4 epilogue (fp16 out, no fp32 temp) + a from_half bias/residual
# store, instead of the fp32 conv_out + scale_bias[_residual]_store<float> path.
# Halves the dequant-store bandwidth. Disable with MODIFF_DISABLE_INT4_DEEPFUSE=1.
_INT4_DEEPFUSE_STORE = os.environ.get("MODIFF_DISABLE_INT4_DEEPFUSE") != "1"

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
        # FP16 per-channel weight scale for the deep-fuse conv epilogue (folds
        # weight_scale into the CUTLASS GEMM -> fp16 out, no fp32 temp). Kept in sync
        # with weight_scale_channel wherever the latter is (re)computed.
        self.register_buffer('weight_scale_channel_half', ch_scale.half().contiguous())

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
        (modiff_cutlass.conv2d_int4_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int4_fprop_o_hat)(
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

    def can_gn_fuse_modiff(self, x: torch.Tensor) -> bool:
        """Eligibility for the fused GroupNorm+SiLU+delta-quantize+pack modiff
        path (group_norm_silu_delta_quantize_pack_nhwc). See
        OptimizedInt8Conv2d.can_gn_fuse_modiff. int4 also needs even
        channels-per-group (checked by the caller alongside the GN conditions)."""
        return (self._can_fuse_input_silu(x)
                and getattr(self, 'groups', 1) == 1
                and x.is_cuda)

    def forward_gn_fused_modiff(self, x, gn_weight, gn_bias, num_groups, eps,
                                mod_scale2d, mod_shift2d, residual=None):
        """int4 counterpart of OptimizedInt8Conv2d.forward_gn_fused_modiff:
        fused GroupNorm(+mod)+SiLU + INT4 delta-quantize+pack + o_hat conv,
        replacing the standalone GN kernel + step1_static_quantize_pack_int4_fprop_silu.
        Bit-identical to that two-kernel path. Caller must have verified
        can_gn_fuse_modiff(x) and the even-CPG / GN-native conditions."""
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

        p_step1 = profiler.start("MoDiff INT4 GN-fused Step1 (GN+SiLU+delta+pack)")
        x_packed = modiff_cutlass.group_norm_silu_delta_quantize_pack_nhwc(
            x, gn_weight, gn_bias, self.a_hat_cache, num_groups, eps, True,
            self.static_input_scale.view(1), self._smooth_inv_flat,
            mod_scale2d, mod_shift2d)
        profiler.stop("MoDiff INT4 GN-fused Step1 (GN+SiLU+delta+pack)", p_step1)

        p_conv = profiler.start("MoDiff INT4 Static Conv2d")
        (modiff_cutlass.conv2d_int4_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int4_fprop_o_hat)(
            x_packed, self.weight_packed, self._cached_alpha_tensor.view(1),
            self.weight_scale_channel.view(-1), self.o_hat_cache,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        profiler.stop("MoDiff INT4 Static Conv2d", p_conv)
        out = self._module_output()
        if residual is not None:
            out = out + residual.to(out.dtype)
        return out

    def forward_modiff_fused_silu_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """int4 counterpart of OptimizedInt8Conv2d.forward_modiff_fused_silu_residual:
        step1 delta-quantize+pack + o_hat conv with the ResBlock skip-add fused into
        the accumulate epilogue (conv2d_int4_fprop_o_hat_residual). Returns o_hat +
        residual; o_hat cache write is byte-identical to the non-residual path.
        Caller must have verified _can_fuse_input_silu(x)."""
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
            x, self.a_hat_cache, self.static_input_scale.view(1), self._smooth_inv_flat)
        profiler.stop("MoDiff INT4 Static Step1 (fused SiLU)", p_step1)

        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
        out = torch.empty_like(self.o_hat_cache)
        p_conv = profiler.start("MoDiff INT4 Static Conv2d (o_hat+residual)")
        # EVT dual-store (see int8 counterpart): o_hat RMW + residual in one conv pass,
        # no fp32 round-trip. Bit-exact o_hat + out vs conv2d_int4_fprop_o_hat_residual.
        modiff_cutlass.conv2d_int4_evt_o_hat_residual(
            x_packed, self.weight_packed, self._cached_alpha_tensor.view(1),
            self.weight_scale_channel.view(-1), self.o_hat_cache, residual, out,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        profiler.stop("MoDiff INT4 Static Conv2d (o_hat+residual)", p_conv)
        return out

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
                # Cache-free static quantize+pack (baseline: no temporal cache). Reads fp16 x
                # directly (no fp32 cast) and does NOT touch a_hat — drops the per-call a_hat
                # zero-fill + a_hat read+write that step1_static_quantize_pack_int4_fprop(x, a_hat=0)
                # wasted. Bit-identical output (residual=x-0=x). SmoothQuant applied upstream -> empty.
                if self._empty_smooth is None or self._empty_smooth.device != x.device:
                    self._empty_smooth = torch.empty(0, device=x.device, dtype=torch.float32)
                x_packed = modiff_cutlass.step1_static_quantize_pack_int4_noahat_fprop(
                    x, self.static_input_scale.view(1), self._empty_smooth
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

    def _apply(self, *args, **kwargs):
        """Keep the packed INT4 weight standard-contiguous through any tensor
        transform. `model.to(memory_format=torch.channels_last)` reformats the 4D
        [K,R,S,C/2] `weight_packed` buffer to a channels_last stride, which for R,S>1
        (3x3 convs) silently transposes the layout the CUTLASS int4 conv kernel reads
        -> garbage output (1x1 unaffected). See the twin guard in OptimizedInt8Conv2d."""
        out = super()._apply(*args, **kwargs)
        wp = getattr(self, "weight_packed", None)
        if wp is not None and wp.dim() == 4 and not wp.is_contiguous():
            self.weight_packed = wp.contiguous()
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

    def _conv_from_int4(self, x_packed: torch.Tensor, h_in: int, w_in: int,
                        residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the calibrated INT4 conv (dequant/bias/store dispatch) on an
        already-quantized+packed activation ([N, H, W, C/2], the layout produced by
        scale_quantize_and_pack). Shared by _forward_standard (which quantizes first)
        and forward_from_int4 (which skips the quantize). h_in/w_in are the conv
        input's spatial dims (== x_packed.shape[1:3]). Optional `residual` (fp16
        channels_last, same shape as output) is fused into the store epilogue as the
        ResBlock skip-add."""
        self._ensure_conv_caches(x_packed.device)
        h_out = ((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x_packed.shape[0], self.out_channels, h_out, w_out)
        bias_fused = False
        residual_fused = False
        if self.standard_output_fp16:
            if (self._standard_output_buf is None
                    or self._standard_output_buf.shape != output_shape
                    or self._standard_output_buf.device != x_packed.device
                    or self._standard_output_buf.dtype != torch.float16):
                self._standard_output_buf = torch.empty(
                    output_shape, device=x_packed.device, dtype=torch.float16
                ).contiguous(memory_format=torch.channels_last)
            # EVT single-pass (acc*alpha*weight_scale[k] + bias[k] + residual[elem] -> fp16, no
            # scratch): replaces the deep-fuse scratch + from_half bias/residual store, and beats
            # even the best autotuned deep-fuse tile at b128. int4 is tile-picky and the EVT tile
            # is fixed, so on a can_implement miss we cache it and fall through to the autotuned
            # deep-fuse path below. weight_scale/bias are read FP32 in the visitor tree.
            if (_INT4_DEEPFUSE_STORE and getattr(self, '_evt_d1_ok', True)
                    and hasattr(modiff_cutlass, "conv2d_int4_evt_bias_residual_fp16")):
                if self.bias is not None:
                    if getattr(self, '_evt_bias_f32', None) is None or self._evt_bias_f32.numel() != self.bias.numel():
                        self._evt_bias_f32 = self.bias.view(-1).float().contiguous()
                    bias_arg = self._evt_bias_f32
                else:
                    bias_arg = self._empty_bias
                res_arg = (residual if residual is not None
                           else torch.empty(0, device=x_packed.device, dtype=torch.float16))
                try:
                    return modiff_cutlass.conv2d_int4_evt_bias_residual_fp16(
                        x_packed, self.weight_packed, self._cached_alpha_tensor,
                        self.weight_scale_channel.view(-1), bias_arg, res_arg,
                        self._standard_output_buf,
                        self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                        self.dilation[0], self.dilation[1])
                except RuntimeError:
                    self._evt_d1_ok = False  # fixed EVT tile can't implement this shape; use deep-fuse below

            # Deep-fuse: fold the per-channel weight_scale into the CUTLASS int4
            # epilogue (writes fully-scaled fp16, NO fp32 temp), then a from_half
            # bias/residual store -- half the store bandwidth of the fp32-input
            # scale_bias[_residual]_store<float> fallback below (the int4 analogue of
            # int8's conv2d_int8_fprop_deepfuse_bias_residual_fp16). Gated on a tile
            # config known to implement this shape (cid>=0, verified during timing);
            # shapes with no implementable deep-fuse tile fall through to fp32.
            if _INT4_DEEPFUSE_STORE and hasattr(modiff_cutlass, "conv2d_int4_fprop_deepfuse_bias_residual_fp16"):
                cid = self._ensure_tuned_config(x_packed)
                if cid is not None and cid >= 0:
                    bias_arg = (self.bias.view(-1).contiguous()
                                if self.bias is not None else self._empty_bias)
                    res_arg = (residual if residual is not None
                               else torch.empty(0, device=x_packed.device, dtype=torch.float16))
                    return modiff_cutlass.conv2d_int4_fprop_deepfuse_bias_residual_fp16(
                        x_packed, self.weight_packed, self._cached_alpha_tensor,
                        self.weight_scale_channel_half.view(-1), bias_arg, res_arg,
                        self._standard_output_buf, cid,
                        self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                        self.dilation[0], self.dilation[1])
            if residual is not None and hasattr(modiff_cutlass, "conv2d_int4_fprop_no_ohat_prealloc_bias_residual"):
                bias_arg = (self.bias.view(-1).contiguous()
                            if self.bias is not None else self._empty_bias)
                return modiff_cutlass.conv2d_int4_fprop_no_ohat_prealloc_bias_residual(
                    x_packed, self.weight_packed, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), bias_arg, residual,
                    self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
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
        if residual is not None and not residual_fused:
            out = out + (residual.to(out.dtype) if out.dtype != residual.dtype else residual)
        return out

    def forward_from_int4(self, x_packed: torch.Tensor, h_in: int, w_in: int,
                          residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Baseline fast path: the activation is already quantized+packed to int4
        (SiLU applied upstream by the GN->int4 fusion in fused_resblock.py), so skip
        the per-layer quantize+pack and go straight to the conv. Only valid when
        calibrated + not modiff_enabled. Optional `residual` (fp16 channels_last) is
        fused into the store epilogue as the ResBlock skip-add."""
        if not x_packed.is_contiguous():
            x_packed = x_packed.contiguous()
        return self._conv_from_int4(x_packed, h_in, w_in, residual=residual)

    def _ensure_tuned_config(self, x_packed: torch.Tensor) -> int:
        """Lazily pick the fastest int4 CUTLASS tile for this conv's shape by timing
        all configs on the actual packed input (mirror of OptimizedInt8Conv2d). Cached
        in _tuned_config_id. Returns -1 (fixed default tile) when autotune is off or
        unavailable."""
        cid = getattr(self, "_tuned_config_id", None)
        if cid is not None:
            return cid
        if not _INT4_CONV_AUTOTUNE or not hasattr(modiff_cutlass, "conv2d_int4_dequant_fp16_tuned"):
            self._tuned_config_id = -1
            return -1
        self._ensure_conv_caches(x_packed.device)
        ncfg = modiff_cutlass.conv2d_int4_num_tuned_configs()
        h_in, w_in = x_packed.shape[1], x_packed.shape[2]
        h_out, w_out = self._out_hw(h_in, w_in)
        buf = torch.empty((x_packed.shape[0], self.out_channels, h_out, w_out),
                          device=x_packed.device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        wsh = self.weight_scale_channel_half.view(-1)
        strides = (self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                   self.dilation[0], self.dilation[1])
        # time the actual deep-fuse kernel (fp16 out, weight_scale folded in)
        best_t, best_id = float("inf"), -1
        for c in range(ncfg):
            try:
                for _ in range(3):
                    modiff_cutlass.conv2d_int4_dequant_fp16_tuned(
                        x_packed, self.weight_packed, self._cached_alpha_tensor, wsh, buf, c, *strides)
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(10):
                    modiff_cutlass.conv2d_int4_dequant_fp16_tuned(
                        x_packed, self.weight_packed, self._cached_alpha_tensor, wsh, buf, c, *strides)
                e.record(); torch.cuda.synchronize()
                t = s.elapsed_time(e)
            except Exception:
                continue
            if t < best_t:
                best_t, best_id = t, c
        self._tuned_config_id = best_id  # -1 if all configs failed -> fixed default
        return self._tuned_config_id

    def _out_hw(self, h_in: int, w_in: int):
        h_out = ((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        return h_out, w_out

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize+pack an fp16/fp32 activation to channels_last packed int4
        ([N,H,W,C/2]) using this conv's static_input_scale -- the block-entry K1 for
        int4 chaining. Mirror of OptimizedInt8Conv2d.quantize_input."""
        self._ensure_conv_caches(x.device)
        xf = x if x.dtype == torch.float32 else x.float()
        if not xf.is_contiguous(memory_format=torch.channels_last):
            xf = xf.contiguous(memory_format=torch.channels_last)
        # Use static_input_scale directly (not _cached_scale_tensor, which can be stale
        # if the caches were populated before set_static_scale) -- mirrors int8.
        return modiff_cutlass.scale_quantize_and_pack(xf, self.static_input_scale.view(1))

    def _ensure_packed_out_buf(self, N, h_out, w_out, device):
        K = self.out_channels
        shape = (N, h_out, w_out, K // 2)
        buf = getattr(self, "_int4_output_packed_buf", None)
        if buf is None or tuple(buf.shape) != shape or buf.device != device:
            self._int4_output_packed_buf = torch.empty(shape, device=device, dtype=torch.int8).contiguous()
        return self._int4_output_packed_buf

    def forward_to_int4(self, x_packed: torch.Tensor, h_in: int, w_in: int,
                        requant_scale: torch.Tensor, apply_relu: bool = True) -> torch.Tensor:
        """INT4-in, INT4-out conv for chaining: dequant + bias + optional ReLU +
        requantize+pack to `requant_scale` (the next conv's input scale), so the next
        conv reads packed int4 directly with no fp16 round-trip. Returns packed int4
        [N,h_out,w_out,K/2]. The int4 analogue of OptimizedInt8Conv2d.forward_to_int8."""
        self._ensure_conv_caches(x_packed.device)
        if not x_packed.is_contiguous():
            x_packed = x_packed.contiguous()
        h_out, w_out = self._out_hw(h_in, w_in)
        out = self._ensure_packed_out_buf(x_packed.shape[0], h_out, w_out, x_packed.device)
        bias_arg = (self.bias.view(-1).contiguous() if self.bias is not None else self._empty_bias)
        cid = self._ensure_tuned_config(x_packed)
        modiff_cutlass.conv2d_int4_fprop_relu_requant_int4(
            x_packed, self.weight_packed, self._cached_alpha_tensor,
            self.weight_scale_channel_half.view(-1), bias_arg, requant_scale.view(1),
            out, apply_relu, cid,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        return out

    def forward_from_int4_dual(self, x_packed: torch.Tensor, h_in: int, w_in: int,
                               residual: torch.Tensor, requant_scale: torch.Tensor,
                               apply_relu: bool = True):
        """Cross-block-chaining conv3: dequant + bias + fp16 skip-residual + ReLU,
        emitting BOTH the fp16 block output (x_{N+1}) AND its requantized packed int4
        (the next block conv1's input) in one store -- fusing the block-entry quantize.
        Returns (out_fp16, out_packed_int4). Requires standard_output_fp16."""
        assert self.standard_output_fp16, "dual store requires standard_output_fp16"
        assert hasattr(modiff_cutlass, "conv2d_int4_fprop_bias_residual_dual"), \
            "int4 dual-store kernel unavailable (rebuild the extension)"
        self._ensure_conv_caches(x_packed.device)
        if not x_packed.is_contiguous():
            x_packed = x_packed.contiguous()
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        h_out, w_out = self._out_hw(h_in, w_in)
        N, K = x_packed.shape[0], self.out_channels
        output_shape = (N, K, h_out, w_out)
        if (self._standard_output_buf is None
                or self._standard_output_buf.shape != output_shape
                or self._standard_output_buf.device != x_packed.device
                or self._standard_output_buf.dtype != torch.float16):
            self._standard_output_buf = torch.empty(
                output_shape, device=x_packed.device, dtype=torch.float16
            ).contiguous(memory_format=torch.channels_last)
        out_packed = self._ensure_packed_out_buf(N, h_out, w_out, x_packed.device)
        bias_arg = (self.bias.view(-1).contiguous() if self.bias is not None else self._empty_bias)
        cid = self._ensure_tuned_config(x_packed)
        modiff_cutlass.conv2d_int4_fprop_bias_residual_dual(
            x_packed, self.weight_packed, self._cached_alpha_tensor,
            self.weight_scale_channel_half.view(-1), bias_arg, residual.half(),
            requant_scale.view(1), self._standard_output_buf, out_packed, apply_relu, cid,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        return self._standard_output_buf, out_packed

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
            (modiff_cutlass.conv2d_int4_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int4_fprop_o_hat)(
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
        (modiff_cutlass.conv2d_int4_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int4_fprop_o_hat)(
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
        set smooth_scale, then requantize+repack the *original* fp weights against their
        smoothed per-output-channel range. Shared by _apply_smoothquant (live calibration,
        where `s` is derived from the activation stats) and set_static_calibration (which
        restores `s` from an exported checkpoint). Requires _orig_weight to still be present."""
        w = self._orig_weight
        K = self.out_channels

        w_dev = w.to(s.device)
        self.smooth_scale.copy_(s.view(1, -1, 1, 1))

        w_smoothed = w_dev * s.view(1, -1, 1, 1)
        w_flat = w_smoothed.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values
        ch_scale = torch.clamp(ch_max / 7.0, min=1e-8)

        self.weight_scale_channel.copy_(ch_scale.view(1, K, 1, 1))
        self.weight_scale_channel_half.copy_(ch_scale.half().to(self.weight_scale_channel_half.device))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
        w_quant = w_quant.reshape(K, self.in_channels, *self.kernel_size)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()

        if self.in_channels % 2 == 0:
            self.weight_packed.data = pack_int4(w_nhwc).to(self.weight_packed.device).contiguous()

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
        """Restore a full static calibration from an exported checkpoint: the per-tensor
        activation `scale` AND, when the layer used SmoothQuant, the per-in-channel
        `smooth_scale`. The smooth scale is re-folded into the (freshly converted, still
        unsmoothed) weights via _fold_weights_with_smooth, so at inference the smoothed
        activations meet smoothed weights and the static scale — which was derived from
        the *smoothed* activation range — is correct.

        Without restoring smooth_scale (the old apply path, i.e. plain set_static_scale),
        a SmoothQuant-derived static scale gets applied to UNsmoothed weights with no
        activation smoothing: int8 masks the mismatch (8-bit range) but int4's 4-bit
        range degrades it ~2x (first-step rel ~0.40 vs ~0.20). If the weights can't be
        re-folded (odd in_channels, or _orig_weight already released), we fall back to
        that scale-only path rather than smoothing activations against unsmoothed weights,
        which would be strictly worse."""
        can_refold = (smooth_scale is not None and self._orig_weight is not None
                      and self.in_channels % 2 == 0)
        if can_refold:
            s = torch.as_tensor(smooth_scale, dtype=torch.float32,
                                device=self.smooth_scale.device).reshape(-1)
            self._fold_weights_with_smooth(s)
            self._smooth_inv.copy_(1.0 / self.smooth_scale)
            self._smooth_is_identity = bool(torch.allclose(
                self._smooth_inv, torch.ones_like(self._smooth_inv), atol=1e-6))
            # Invalidate the lazily-cached flat smooth vector so the forward path
            # rebuilds it from the restored smooth_scale.
            if hasattr(self, '_smooth_inv_flat'):
                del self._smooth_inv_flat
        self.set_static_scale(scale)


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


def export_int4_static_scales(model: nn.Module) -> Dict[str, object]:
    """Export the static calibration per int4 conv for checkpoint reuse.

    A layer with identity SmoothQuant is exported as a bare float (the legacy format,
    still loadable by any old consumer). A layer that used SmoothQuant is exported as
    ``{"static_scale": float, "smooth_scale": cpu fp32 tensor [C_in]}`` so the smoothing
    can be restored on apply — without it, int4 checkpoint reuse silently loses ~2x
    accuracy (see set_static_calibration). Serialize with torch.save (values may be
    tensors)."""
    scales = {}
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d) and module.is_calibrated:
            if module._smooth_is_identity:
                scales[module.layer_name] = float(module.static_input_scale.item())
            else:
                scales[module.layer_name] = {
                    "static_scale": float(module.static_input_scale.item()),
                    "smooth_scale": module.smooth_scale.detach().to("cpu", torch.float32).reshape(-1).clone(),
                }
    return scales


def apply_int4_static_scales(model: nn.Module, scales: Dict[str, object]) -> int:
    """Load static calibration produced by export_int4_static_scales. Accepts both the
    legacy flat ``{name: float}`` format and the richer ``{name: {"static_scale":...,
    "smooth_scale":...}}`` format (mixed within one dict is fine)."""
    loaded = 0
    for module in model.modules():
        if isinstance(module, OptimizedInt4Conv2d):
            if module.layer_name in scales:
                entry = scales[module.layer_name]
                if isinstance(entry, dict):
                    module.set_static_calibration(entry["static_scale"], entry.get("smooth_scale"))
                else:
                    module.set_static_scale(float(entry))
                loaded += 1
    return loaded
