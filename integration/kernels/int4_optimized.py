
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

#: Table length for the MoDiff per-step delta scale. Matches int8_optimized.MODIFF_MAX_STEPS.
MODIFF_MAX_STEPS = 256
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
                       and HAS_CUTLASS
                       # The int4 autotuner times conv2d_int4_dequant_fp16_tuned, not
                       # conv2d_int4_fprop_tuned, which this used to probe and which nothing calls.
                       and hasattr(modiff_cutlass, "conv2d_int4_dequant_fp16_tuned"))

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


def _int4_weight_scale(w_flat: torch.Tensor, Q: float = 7.0) -> torch.Tensor:
    """Per-output-channel symmetric INT4 weight scale, chosen to minimise reconstruction MSE.

    The obvious choice, scale = absmax/Q, is what this replaced. At 4 bits it is a poor one: the
    real checkpoint's conv weights have a median max/median ratio of 6.5 and up to 24.9, so one
    outlier in a channel stretches the 15-level grid for all ~1700 of that channel's weights.
    Measured over the 87 quantized convs (relative Frobenius error of the reconstructed weight):

        per-channel absmax   0.1825 median, 0.4493 worst   <- what this replaced
        per-channel p99.9    0.1498 median, 0.3203 worst
        per-channel MSE      0.1254 median, 0.2609 worst   <- this
        group-128 absmax     0.1226 median, 0.2206 worst

    So the MSE search recovers 96% of what group-wise quantization would buy, and unlike group-wise
    it is free: the CUTLASS int4 conv epilogue folds ONE fp16 scale per output channel, and this
    keeps that layout exactly. Group-wise would need the scale applied inside the K-loop.

    Cost is at load only -- 13 candidate clips x one pass each, per layer.

    Why this matters for MoDiff specifically: MoDiff is an ACTIVATION method, so none of the weight
    error above is reachable by it. At W4A4 the weight error alone (0.18) is a large fraction of
    the mode's total latent relL2 (0.44), which is why W4A4+MoDiff underperforms the paper's W8A4
    result by so much.
    """
    #: DEFAULT IS mse, on a PAIRED A/B (same 4 seeds per arm, batch 16, DDIM 50, warm-up run
    #: discarded), W4A4+MoDiff latent relL2 vs fp16:
    #:     absmax  0.5067 +- 0.0195   (0.5248, 0.4983, 0.5206, 0.4833)
    #:     mse     0.4689 +- 0.0093   (0.4737, 0.4688, 0.4772, 0.4559)
    #: 4/4 seeds improve, mean -7.5%, effect ~2x the spread. An earlier UNPAIRED single-seed check
    #: had this backwards (0.4437 -> 0.4981 "worse") -- in that same comparison the int8 rows, which
    #: this code cannot affect, also moved 10-30%, which is what exposed it as noise. Note the
    #: end-to-end gain (-7.5%) is much smaller than the weight-reconstruction gain (-31%): lower
    #: ||W-Q(W)|| is not the same objective as lower output error, and clipping outliers trades away
    #: some of the salient weights AWQ exists to protect. MODIFF_INT4_WSCALE=absmax restores the old
    #: rule for A/B without a rebuild.
    mode = os.environ.get("MODIFF_INT4_WSCALE", "mse")
    am = w_flat.abs().max(dim=1).values
    if mode == "absmax":
        return torch.clamp(am / Q, min=1e-8)
    best_err = None
    best_scale = None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4):
        sc = torch.clamp(am * r / Q, min=1e-8)
        e = (((w_flat / sc[:, None]).round().clamp(-Q, Q) * sc[:, None] - w_flat) ** 2).sum(1)
        if best_err is None:
            best_err, best_scale = e, sc
        else:
            m = e < best_err
            best_err = torch.where(m, e, best_err)
            best_scale = torch.where(m, sc, best_scale)
    return best_scale


#: "layer|entry_point" pairs observed quantizing with a non-zero activation zero point that the
#: entry point does not apply. As of 2026-08-13 only group_norm_silu_quantize_pack_nhwc honours
#: z; the plan scoped fix #2 at ~6 entry points. Collected so the gap is visible instead of
#: silently numeric -- symmetric codes against a zp-corrected bias is worse than either choice,
#: and is what produced relL2 7-22 in the first end-to-end run.
_ZP_UNSUPPORTED = set()

#: EXPERIMENT ONLY (MODIFF_ZP_PREPAD=1): pad the packed activation with the code z instead of letting
#: CUTLASS zero-fill it, so an asymmetric grid is padded CORRECTLY. Read at call time, not import time,
#: so an in-process A/B can flip it between arms. See _conv_from_int4 for the mechanism and why it is
#: off by default (it materializes a padded copy per conv).
#: Does the extension have the fused code-pad kernel? Checked once at import, like the other HAS_* gates.
HAS_PAD_PACKED_INT4_CODE = hasattr(modiff_cutlass, "pad_packed_int4_code") if HAS_CUTLASS else False


def _zp_prepad_enabled():
    return os.environ.get("MODIFF_ZP_PREPAD", "0") == "1"

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
        ch_scale = _int4_weight_scale(w_flat)  # [K]; MSE-optimal clip, see the helper
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

        # --- Sum of quantized weights per output channel, for the activation zero point ---
        # An asymmetric activation grid stores a_q = round(a*s) + z, so a = (a_q - z)/s and
        #     sum_i w_q[k,i] * a[i] = (sum_i w_q[k,i]*a_q[i] - z * sum_i w_q[k,i]) * ws[k] / s.
        # The second term is a CONSTANT PER OUTPUT CHANNEL, so it folds into the bias and the GEMM
        # never has to know about the zero point. This buffer is that sum; it is exact (int4 codes in
        # fp32) and is computed here, once, from the same w_quant the packing uses -- deriving it
        # later from weight_packed would mean unpacking nibbles and could drift from what shipped.
        self.register_buffer('weight_sum_q',
                             w_quant.reshape(K, -1).sum(dim=1).float().view(1, K, 1, 1))
        #: Activation zero point, per tensor. ZERO unless an asymmetric calibration sets it, and at
        #: zero every expression below reduces exactly to the symmetric path -- which is what
        #: test_int4_zero_point.py asserts bit-for-bit rather than approximately.
        self.register_buffer('static_input_zp', torch.zeros(1, dtype=torch.float32))
        #: HOST MIRROR of static_input_zp. The guards and the routing have to ask "is there a
        #: zero point?" on every quantize, and reading the CUDA buffer with .item() forces a
        #: device sync -- ~12k of them per sample at 62 convs x 200 steps. Measured cost of the
        #: first version, which did exactly that: W4A4 MoDiff 59.52 -> 61.63 ms/step and PTQ
        #: 59.04 -> 61.73. Updated only where the buffer is (calibration time), never in a
        #: forward.
        self._zp_float = 0.0

        # --- Bias ---
        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.view(1, -1, 1, 1))
        else:
            self.bias = None
        #: The bias as it arrived, before any zero-point correction is folded in. Kept because the
        #: correction depends on the activation scale: re-calibrating must recompute it from the
        #: ORIGINAL bias, not from an already-corrected one, or the correction compounds.
        self._orig_bias = None if self.bias is None else self.bias.clone()

        self._empty_bias = None
        self.use_cutlass = HAS_CUTLASS and self.groups == 1 and self.in_channels % 2 == 0

        # --- MoDiff state ---
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        self.step_count = 0
        #: 5 rounds, per the paper's Appendix D.5 warm-up. See OptimizedInt8Conv2d.warmup_steps for
        #: the measured contraction; at 4 bits it is 0.4006 -> 0.00001 over 5 rounds, and 0 over any
        #: number of rounds with the static grid this used to pass.
        self.warmup_steps = max(1, int(os.environ.get("MODIFF_WARMUP_STEPS", "5")))

        # --- Calibration state ---
        self.calibrating = False
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self.register_buffer('static_input_scale', torch.tensor(1.0, dtype=torch.float32))
        # --- MoDiff per-step delta-scale table (W4A4). The int8 twin of this landed 2026-08-03;
        # int4 was left out, which meant int4 MoDiff had only the activation grid to quantize the
        # delta on -- and per Theorem 4.3 that predicts no error reduction at all. Measured: int4
        # MoDiff static 0.7772 vs int4_baseline 0.7830, i.e. it bought essentially nothing. The
        # int8 table by contrast is a 4.4x improvement (0.1850 -> 0.0422), so this is the single
        # largest quality gap left in the W4A4 path.
        self.register_buffer('static_delta_scale',
                             torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32))
        self.register_buffer('static_delta_alpha',      # 1/scale, the CUTLASS epilogue alpha
                             torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32))
        self.register_buffer('is_delta_calibrated', torch.tensor(False))
        #: Host mirror of is_delta_calibrated -- the hot path must never read the device buffer
        #: (one GPU->CPU sync per modulated conv per step; measured ~5 ms/step for the int8 twin).
        self._delta_cal = False
        self._delta_calib = False
        self._delta_absmax_obs: Optional[torch.Tensor] = None
        self._delta_obs_code_max: Optional[float] = None
        self._delta_obs_clipped_frac: Optional[float] = None
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

        #: Delta-quantizer mode, W4A4. Same contract as OptimizedInt8Conv2d.delta_dynamic, with
        #: Q=7 instead of 127. STATIC is the default since 2026-08-12 for paper fidelity, not
        #: because it measures better -- see the int8 twin for the price and the one env var that
        #: reverts it.
        #:
        #: `static` used to be a WEAKER thing here than at int8: this path had no per-step delta
        #: table (Stage 1 was int8-only), so it fell back to quantizing the temporal delta on the
        #: *activation* grid -- 15 levels spanning the whole activation range, which per the
        #: paper's Theorem 4.3 leaves the quantization error unchanged from baseline, i.e. MoDiff
        #: buys only error feedback. That fallback still exists and still warns, but it is no
        #: longer what `static` means by default: int4_delta_qdiff.pt is now calibrated
        #: (--modulate --quant_mode qdiff at 4 bits) and loaded by _load_delta_table. With only
        #: 4 bits to spend, clipping and coarseness both bite harder than at int8, so the gap to
        #: `dynamic` is wider here than at W8A8.
        self.delta_dynamic = os.environ.get("MODIFF_DELTA_MODE", "static").lower() != "static"
        #: `MODIFF_DELTA_CLIP` is RETIRED -- see OptimizedInt8Conv2d for the sweep it produced and
        #: why setting it now raises instead of being ignored. At W4A4 it cost nothing to remove:
        #: the curve was flat within noise from 0.35 to 1.0 (relL2 0.42-0.46). The int8 class does
        #: the raising, and every W4A4 run constructs those too, so this class does not repeat it.
        #: See OptimizedInt8Conv2d._delta_should_refresh -- recompute the dynamic scale every Nth
        #: modulated step, reuse it in between. 1 = exact.
        #: 4. See OptimizedInt8Conv2d.delta_refresh: K=1 is the paper's formulation and was tried as
        #: the default on 2026-08-06, but it never wins on a paired sweep and loses badly at 3 and 2
        #: activation bits, because a per-step absmax is set by one outlier where a held scale is
        #: smoothed.
        self.delta_refresh = max(1, int(os.environ.get("MODIFF_DELTA_REFRESH", "4")))
        #: Free absmax reporting, INT4 twin. DEFAULT OFF -- see OptimizedInt8Conv2d.delta_report for
        #: the full reasoning. At W4A4 it does not merely degrade, it DIVERGES: latent relL2 0.4746
        #: (off) -> 11.6553 (on), measured 2026-08-04. With only 15 levels, the extra staleness
        #: (the scale is published on a refresh step and consumed across the following window, so up
        #: to 2*delta_refresh steps old) clips, and clipping compounds through the error-feedback
        #: term. Kept only as the recorded result.
        self.delta_report = os.environ.get("MODIFF_DELTA_REPORT", "0") == "1"
        self.delta_report_safety = float(os.environ.get("MODIFF_DELTA_SAFETY", "1.15"))
        self._delta_seeded = False

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

    def effective_code_utilisation(self, x: torch.Tensor, fused_silu: bool = False) -> float:
        """max |value| this layer's static quantizer will see, in CODE units. Q (=7) is full scale.

        See OptimizedInt8Conv2d.effective_code_utilisation for why this exists and for the two
        wrong ways of computing it that this replaces. int4's Q is 7, so the headroom is far
        smaller and a mis-provisioned scale bites correspondingly harder.
        """
        xs = x.detach().float()
        if fused_silu:
            xs = F.silu(xs)
        if not self._smooth_is_identity:
            xs = xs * self._smooth_inv.to(xs.device, torch.float32)
        return float(xs.abs().max().item() * float(self.static_input_scale.item()))

    def _module_output(self) -> torch.Tensor:
        # See OptimizedInt8Conv2d._module_output: forcing fp32 here just to have
        # the next fp16-autocast op cast back down again is a wasted full-tensor
        # copy. Return the cache (fp16 when calibrated) as-is.
        #
        # See the int8 twin: casting fp32 -> fp16 here is deliberately NOT done. The uncalibrated
        # window is an fp32-flavoured pipeline, and the one consumer that requires fp16 enforces it
        # itself (quantized_std_attention._proj_with_residual).
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

    def _dequantize_activation(self, x: torch.Tensor, input_scale,
                               activation_grid: bool = True) -> torch.Tensor:
        """Simulate quantize-then-dequantize: a_hat = Q(x) in FP32.
        input_scale can be float or 1-element tensor.

        WITH AN ACTIVATION ZERO POINT the round trip is a_q = clamp(round(a*s) + z, -7, 7) and
        a = (a_q - z)/s, so both the clamp and the un-shift have to be here. Dropping the -z would
        leave a_hat offset by z/s at t=T, and MoDiff carries a_hat forward through every remaining
        step -- the error would not decay, it would seed the whole trajectory.

        `activation_grid=False` MEANS x IS A RESIDUAL, not an activation, and z does not apply: it
        cancels in a difference, the warm-up loop's conv adds no bias to compensate for it, and the
        per-round scale is not the calibrated activation scale. This mirrors _int4_conv's with_bias
        gate exactly -- the two must agree, because a_hat accumulates r_dq while o_hat accumulates the
        conv of the SAME codes. If one applies z and the other does not, the caches diverge.

        At z = 0 both branches are the original expression term for term, which is what
        test_int4_zero_point.py's bit-identity gate rests on.
        """
        z = getattr(self, "_zp_float", 0.0) if activation_grid else 0.0
        if z == 0.0:
            return (x * input_scale).round().clamp(-7, 7) / input_scale
        return (((x * input_scale).round() + z).clamp(-7, 7) - z) / input_scale

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
                # quantize_and_pack takes PRE-SCALED values and has no zero-point arity, so this
                # branch cannot honour z. It is only reached with a python-float input_scale, which
                # means an uncalibrated layer -- a layer with a zero point is calibrated by
                # construction (the zp arrives with the scale in set_static_calibration).
                # Guarded only when the bias is added. A bias-free call here is the warm-up loop
                # quantizing a RESIDUAL on a dynamic scale, which needs no zero point, so there is
                # nothing to guard -- and it cannot use grid="delta" either, because that exemption
                # verifies an a_hat CACHE and the warm-up residual is taken against a local a_hat
                # that has not been stored yet.
                if with_bias:
                    self._zp_unsupported("quantize_and_pack (float-scale branch)")
                x_packed = modiff_cutlass.quantize_and_pack(x_scaled)
            else:
                # Tensor path: use fused scale+quantize+pack kernel (no CPU sync)
                scale_tensor = (1.0 / input_scale).view(1)
                if not x.is_contiguous(memory_format=torch.channels_last):
                    x = x.contiguous(memory_format=torch.channels_last)
                # ACTIVATION ZERO POINT (fix #2), AND THIS IS THE SITE THAT MATTERED. This is
                # MoDiff's t=T entry point (_forward_modulated -> _forward_first_step -> here): the
                # one quantize per layer per sample whose conv adds the zp-corrected bias, seeding
                # the o_hat that every later step accumulates into. It had no guard until
                # 2026-08-13, which is why the census in docs/zero_point_2026-08-13/FINDINGS.md
                # listed 62 harmless delta sites and not this one, and why the MoDiff arm measured
                # relL2 7.3057 -- a divergence, not a grid.
                #
                # GATED ON with_bias, WHICH IS THE WHOLE DISCRIMINATION. z exists to pair with the
                # -z*sum(w_q) term _refold_zp_bias folded into the bias, so it applies exactly where
                # that bias is added. The two callers separate cleanly on that test:
                #
                #   _forward_first_step  with_bias=True   x is the ACTIVATION, on the calibrated
                #                                         grid, and the bias is added -> z applies
                #   its warm-up loop     with_bias=False  x is a RESIDUAL (x - a_hat) on a dynamic
                #                                         per-round scale, conv is bias-free -> z
                #                                         must NOT be applied
                #
                # Measured, not assumed: without this gate the MoDiff arm made 350 zp-quantize calls
                # per run (70 convs x 5) instead of 70 -- every warm-up residual was being shifted by
                # z against a conv that adds no bias to compensate, i.e. the same class of defect
                # (codes and bias disagreeing about z) that this whole fix exists to remove, just
                # pointing the other way.
                zp = getattr(self, "_zp_float", 0.0) if with_bias else 0.0
                if zp != 0.0:
                    if not hasattr(modiff_cutlass, "scale_quantize_and_pack_zp"):
                        raise RuntimeError(
                            "activation zero point is set but modiff_cutlass lacks "
                            "scale_quantize_and_pack_zp -- rebuild the extension")
                    x_packed = modiff_cutlass.scale_quantize_and_pack_zp(x, input_scale, zp)
                else:
                    x_packed = modiff_cutlass.scale_quantize_and_pack(x, input_scale)

            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, device=x.device)

            # Same correct-asymmetric-padding emulation as _conv_from_int4, and it is needed HERE too
            # for the MoDiff arm: this is the t=T conv, the only one on that arm that reads the
            # activation grid and adds the corrected bias. Its delta steps do NOT need it -- they
            # quantize a difference on a symmetric delta grid, where code 0 IS delta 0, so zero-filling
            # is already correct there.
            pad_h, pad_w = self.padding[0], self.padding[1]
            if (_zp_prepad_enabled() and zp != 0.0 and (pad_h or pad_w)):
                x_packed, _, _ = self._prepad_packed_with_zp(
                    x_packed, x_packed.shape[1], x_packed.shape[2])
                pad_h = pad_w = 0

            out_raw = modiff_cutlass.conv2d_int4_fprop(
                x_packed,
                self.weight_packed,
                scale_tensor,
                self._empty_bias,
                self.stride[0], self.stride[1],
                pad_h, pad_w,
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

    # ==================================================================
    # Forward paths
    # ==================================================================

    # ------------------------------------------------------------------
    # Dynamic delta scale (W4A4). Mirrors OptimizedInt8Conv2d's helpers with Q=7.
    # ------------------------------------------------------------------
    Q_DELTA = 7.0

    #: DELIBERATE CLIPPING ON THE DELTA GRID, and the single largest W4A4 quality lever found so far.
    #:
    #: Sizing the grid to the observed absmax is the obvious choice and it is wrong at 4 bits. The
    #: MoDiff residual is heavy-tailed: covering its full range spends the 15 codes on a tail almost
    #: nothing lands in, and the bulk gets a step size it cannot afford. Dividing the assumed range
    #: by this ratio saturates the tail and buys resolution everywhere else.
    #:
    #: Swept act-only on the real checkpoint (DDIM S=50, batch 6, 3 seeds, fp16 weights so the delta
    #: grid is the only variable -- docs/paper_repro_2026-08-12/data/delta_clip_sweep.json):
    #:
    #:     ratio    1     2     4     8     16    21    32
    #:     relL2  .4945 .3362 .1773 .1147 .2193 .2542 .3117
    #:
    #: A clean U with its minimum at 8, worth 4.3x over the unclipped table. For reference the
    #: paper's OWN per-layer delta values, dropped into this datapath, read 0.2452 -- so a single
    #: swept constant beats importing them, because the optimum follows the trajectory and ours is
    #: not theirs (different weights, EMA, calibration set and step count).
    #:
    #: THIS IS PROTOCOL-DEPENDENT and must not be read as universal. The residual shrinks with step
    #: count (measured absmax 0.83 at S=50 against 0.50 at S=500), so a run at a very different S
    #: should re-sweep. MODIFF_DELTA_CLIP_RATIO exists for that.
    #:
    #: int8 keeps 1.0: 255 levels do not have this problem and the sweep was never run there. Do not
    #: copy the 8 across without measuring -- that assumption is what the int4/int8 twins keep
    #: getting wrong.
    DELTA_CLIP_RATIO = float(os.environ.get("MODIFF_DELTA_CLIP_RATIO", "8.0"))

    #: The same lever on the ACTIVATION grid, and for the same reason. `silu(gn(x))` is one-sided
    #: (SiLU bottoms out at -0.2785, unbounded above) with |max|/|min| measured at 19.91x, so a
    #: symmetric grid sized to the positive tail leaves only 5 of 15 codes carrying >0.1% of the
    #: mass -- an effective 2.32 bits of a nominal 3.91
    #: (docs/state_report_2026-08-12/data/int4_code_use.json).
    #:
    #: Under-sizing the range saturates that tail and gives the bulk a usable step. Swept act-only:
    #: assumed absmax /1 reads 1.1499, /6.7 reads 0.4519. Set from a real-kernel sweep, not the
    #: fake-quant one -- the harness runs fp16 weights, and once the activation grid is fine the
    #: weight error (0.2728 on its own) stops being negligible, so fake quant reads systematically
    #: optimistic in exactly this regime. Measured: at DELTA_CLIP_RATIO 8 the harness predicted
    #: 0.1147 and the kernels delivered 0.3099.
    #:
    #: A zero point would recover most of this properly rather than by saturation, and is worth a
    #: further 1.23x on top of the best clip -- see the plan's fix #2. This constant is the part that
    #: needs no format or kernel change.
    #: Swept on the REAL kernels, both axes, 3 seeds
    #: (docs/paper_repro_2026-08-12/data/act_clip_sweep_real.json):
    #:
    #:     ratio      1      2      3     4.5    6.7     10
    #:     PTQ     .8647  .5482  .4968  .4692  .5312  .6373     <- clean U, minimum at 4.5
    #:     MoDiff  .3090  .3176  .3074  .3095  .3121  .3361     <- flat, 1.09x across the sweep
    #:
    #: 4.5 is chosen for the PTQ axis and costs the MoDiff axis nothing, because MoDiff reads this
    #: grid only at t=T and then refines a_hat with 5 warm-up rounds -- the same reason the
    #: "paper leaves a_T unquantized" hypothesis came out per-seed identical. One constant serves
    #: both; they do not need separate ratios.
    #:
    #: THIS ONLY HELPS HEAVY-TAILED DATA, and post-SiLU activations are heavy-tailed. On GAUSSIAN
    #: data it is a straight loss, because there is no tail to trade away: test_int4_conv's synthetic
    #: fixture (randn weights, randn input) goes 0.221 -> 0.340 at ratio 4.5, while the real network's
    #: PTQ arm goes 0.8647 -> 0.4692. Both are correct. Do not "fix" the constant by reverting it
    #: because a synthetic conv got worse -- that fixture has an |max|/|min| of 1.26 against the real
    #: activations' 19.91, which is the whole reason clipping pays on one and not the other.
    #: golden/int4_conv_res32_3x3.pt was refreshed for this change; MODIFF_ACT_CLIP_RATIO=1.0
    #: reproduces the previous golden bit-exactly, which is how the attribution was proved.
    ACT_CLIP_RATIO = float(os.environ.get("MODIFF_ACT_CLIP_RATIO", "4.5"))

    def _ensure_delta_dyn_bufs(self, device):
        """Allocate the 4 reduction buffers the dynamic delta scale needs.

        The uncalibrated path allocates these too (see the lazy-init block in
        _forward_modulated), but only *after* the calibrated branch has already returned -- so
        the calibrated path, which is the one dynamic mode runs on, needs its own init.
        """
        if self._scale_buf is None or self._scale_buf.device != device:
            self._scale_buf = torch.empty(1, device=device, dtype=torch.float32)
            self._inv_scale_buf = torch.empty(1, device=device, dtype=torch.float32)
            self._absmax_buf = torch.zeros(1, device=device, dtype=torch.float32)
            self._retire_count = torch.zeros(1, device=device, dtype=torch.int32)

    # ------------------------------------------------------------------
    # MoDiff per-step delta-scale table (W4A4). Same purpose as OptimizedInt8Conv2d's, but the
    # CALIBRATION mechanism has to differ, and that is the whole story here.
    #
    # Q is 7, not 127. The int8 twin recovers the delta's range by observing max|code| under a known
    # scale, so its resolution is one quantizer step -- fine out of 255 levels, useless out of 15.
    # Measured: observing int4 codes gave code_max == 1 for all 70 layers (visible as "step gain
    # median 1.72x max 1.72x", identical everywhere, which back-solves to code_max == 1), and the
    # resulting table made latent error 2.1x WORSE than no table (0.7772 -> 1.6630).
    #
    # So this path observes the delta absmax EXACTLY instead, by running the layer in dynamic mode
    # during calibration and reading back what the absmax kernel computed. See
    # _observe_delta_absmax.
    # ------------------------------------------------------------------

    def _delta_step_index(self) -> int:
        """Table index for the modulated step about to run. step_count is incremented at the top of
        every modulated forward and is 0 after _forward_first_step, so the first modulated step sees
        1 -> index 0. Longer runs than the table clamp to its last entry."""
        return min(max(self.step_count - 1, 0), MODIFF_MAX_STEPS - 1)

    def _observe_delta_absmax(self):
        """Record this step's EXACT delta absmax, read from what the dynamic kernel just computed.

        Why not observe max|code| the way the int8 twin does. That trick recovers the range as
        code_max/scale_used, so its resolution is one quantizer step -- fine with 255 levels, useless
        with 15. Measured directly: observing int4 codes gave code_max == 1 for all 70 layers, i.e.
        a range estimate good to about a factor of two, and the resulting table made latent error
        2.1x WORSE than no table at all (0.7772 -> 1.6630).

        So calibration instead runs the layer in DYNAMIC mode, where delta_absmax_fp16 /
        gn_delta_absmax_flat_kernel compute max|delta| exactly, and reads the answer back out of
        `_inv_scale_buf` (which holds absmax/Q by construction). Two benefits beyond resolution:
        the calibration trajectory is quantized with a good scale rather than a deliberately coarse
        observation scale, and the resulting static table reproduces the dynamic scale's per-step
        envelope -- which is why the int8 table lands within a few percent of dynamic.

        Runs entirely on device (a max into a resident [MODIFF_MAX_STEPS] buffer), so no host sync.
        """
        if not self._delta_calib or self._inv_scale_buf is None:
            return
        if self._delta_absmax_obs is None or self._delta_absmax_obs.device != self._inv_scale_buf.device:
            self._delta_absmax_obs = torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32,
                                                 device=self._inv_scale_buf.device)
        i = self._delta_step_index()
        am = self._inv_scale_buf.view(()) * self.Q_DELTA      # inv_scale = absmax / Q
        self._delta_absmax_obs[i] = torch.maximum(self._delta_absmax_obs[i], am)

    def begin_delta_calibration(self, reset: bool = False):
        """Arm exact delta-absmax observation. The layer runs in dynamic mode for the pass, so the
        absmax is computed by the kernel rather than inferred through a 4-bit quantizer -- see
        _observe_delta_absmax for why the int8 code-observation trick cannot be reused here.
        `reset` is vestigial, kept for signature parity with the int8 twin."""
        self._delta_calib = True
        self._delta_absmax_obs = None
        self._delta_calib_was_dynamic = self.delta_dynamic
        self._delta_calib_was_refresh = self.delta_refresh
        self.delta_dynamic = True      # exact per-call absmax
        self.delta_refresh = 1         # every step, or the per-step table would be undersampled
        self.is_delta_calibrated.fill_(False)
        self._delta_cal = False

    def end_delta_calibration(self, safety: float = 1.02, smooth: bool = True) -> bool:
        """Turn the observed per-step absmax into the scale table. Returns True if set."""
        self._delta_calib = False
        # Restore whatever mode the caller had; the dynamic forcing was for the observation pass.
        self.delta_dynamic = getattr(self, "_delta_calib_was_dynamic", self.delta_dynamic)
        self.delta_refresh = getattr(self, "_delta_calib_was_refresh", self.delta_refresh)
        if self._delta_absmax_obs is None or not self.is_calibrated:
            return False
        absmax = self._delta_absmax_obs.detach().to("cpu", torch.float64)
        seen = absmax > 0
        if not bool(seen.any()):
            return False
        self._delta_obs_code_max = None            # not applicable: observation is not quantized
        self._delta_obs_clipped_frac = 0.0         # exact absmax cannot clip by construction

        # Forward-fill steps never reached, then back-fill the head. The delta range is flat in the
        # tail, so a short calibration run still yields a usable table for a longer production run.
        last = 0.0
        for i in range(absmax.numel()):
            if seen[i]:
                last = float(absmax[i])
            elif last > 0.0:
                absmax[i] = last
        first = next((float(absmax[i]) for i in range(absmax.numel()) if absmax[i] > 0), 0.0)
        absmax[absmax <= 0] = first
        if first <= 0.0:
            return False

        if smooth:
            # Running max over a 3-wide window: one noisy batch must not shrink a neighbour's scale.
            a = absmax.clone()
            for i in range(absmax.numel()):
                lo, hi = max(0, i - 1), min(absmax.numel(), i + 2)
                absmax[i] = float(a[lo:hi].max())

        # DELTA_CLIP_RATIO deliberately under-sizes the grid; see the constant for the sweep.
        scale = self.Q_DELTA / (absmax * safety / self.DELTA_CLIP_RATIO).clamp_min(1e-12)
        self.static_delta_scale.copy_(scale.to(torch.float32))
        self.static_delta_alpha.copy_((1.0 / scale).to(torch.float32))
        self.is_delta_calibrated.fill_(True)
        self._delta_cal = True
        self._delta_absmax_obs = None
        return True

    def delta_calibration_report(self) -> Dict[str, object]:
        """Diagnostics: how much finer the delta grid is than the activation grid it replaces."""
        if not bool(self.is_delta_calibrated):
            return {"layer": self.layer_name, "calibrated": False}
        act = float(self.static_input_scale.item())
        tail = self.static_delta_scale.detach().to("cpu", torch.float64)
        pos = tail[tail > 0]
        return {"layer": self.layer_name, "calibrated": True,
                "act_scale": act,
                "delta_scale_median": float(pos.median()) if pos.numel() else 0.0,
                "step_gain_tail": (float(pos.median()) / act) if (pos.numel() and act > 0) else None,
                "obs_code_max": self._delta_obs_code_max,
                "obs_clipped_frac": self._delta_obs_clipped_frac}

    def _delta_should_refresh(self) -> bool:
        """See OptimizedInt8Conv2d._delta_should_refresh. step_count is 1 on the first modulated
        step, so that step always refreshes -- required, since the buffers hold nothing before."""
        k = self.delta_refresh
        return k <= 1 or ((self.step_count - 1) % k) == 0

    def _delta_scale_args_i4(self, device, x=None, fused_silu=False):
        """(quantize_scale, conv_alpha) for the current step.

        Static mode returns the activation scale and its cached reciprocal -- the pre-existing
        behaviour. Dynamic mode reduces max|delta| for this call and returns device views of the
        result, so the 15 available int4 levels span the delta's own range instead of the whole
        activation's. `fused_silu` must match whether the paired quantize kernel applies SiLU,
        or the reduced and quantized expressions differ and the no-clip guarantee is void.
        """
        if self.delta_dynamic:
            self._ensure_delta_dyn_bufs(device)
            if x is not None and self._delta_should_refresh():
                modiff_cutlass.delta_absmax_fp16(
                    x, self.a_hat_cache, self._absmax_buf, self._scale_buf, self._inv_scale_buf,
                    self._retire_count, self.Q_DELTA,
                    self._smooth_inv_flat, fused_silu)
            return self._scale_buf.view(1), self._inv_scale_buf.view(1)
        # Static mode: prefer the per-step delta table when it has been calibrated. Falling back to
        # the activation scale reproduces the pre-table behaviour, which per Theorem 4.3 leaves the
        # quantization error unchanged from baseline -- so warn once rather than fail silently.
        if not self._delta_cal and bool(self.is_delta_calibrated):
            self._delta_cal = True
        if self._delta_cal:
            i = self._delta_step_index()
            return self.static_delta_scale[i:i + 1], self.static_delta_alpha[i:i + 1]
        if not getattr(self, "_warned_no_delta_calib", False):
            self._warned_no_delta_calib = True
            print(f"⚠ {self.layer_name or type(self).__name__}: no INT4 MoDiff delta calibration; "
                  f"quantizing the temporal delta on the FULL-ACTIVATION grid with only 15 levels. "
                  f"Per Theorem 4.3 this leaves the error unchanged -- MoDiff buys only error "
                  f"feedback. Run the int4 delta calibration.")
        return self.static_input_scale.view(1), self._cached_alpha_tensor.view(1)

    def _delta_report_on_i4(self) -> bool:
        """Free reporting is live: dynamic, enabled, seeded, and on a refresh step."""
        return (self.delta_dynamic and self.delta_report and self._delta_seeded
                and self._delta_should_refresh())

    def _delta_gn_dynamic_args_i4(self, device):
        """The five trailing arguments of group_norm_silu_delta_quantize_pack_nhwc: empty
        tensors in static mode, the real reduction buffers in dynamic mode (where the kernel
        finds the scale itself from silu(gn(x)), which only exists inside it)."""
        if getattr(self, "_empty_f32", None) is None or self._empty_f32.device != device:
            self._empty_f32 = torch.empty(0, device=device, dtype=torch.float32)
        e = self._empty_f32
        if not self.delta_dynamic:
            return e, e, e, e, self.Q_DELTA, False, 1.0
        self._ensure_delta_dyn_bufs(device)
        if self._delta_report_on_i4():
            # The quantize kernel both quantizes with the published scale and publishes the next
            # one: no separate absmax pass at all.
            return (self._absmax_buf, self._scale_buf, self._inv_scale_buf,
                    self._retire_count, self.Q_DELTA,
                    True, self.delta_report_safety)
        if not self._delta_should_refresh():
            # Reuse the last measured scale; empty buffers make the kernel skip its absmax pass.
            return e, e, e, e, self.Q_DELTA, False, 1.0
        self._delta_seeded = True      # this pass publishes a scale the next window can ride on
        return (self._absmax_buf, self._scale_buf, self._inv_scale_buf,
                self._retire_count, self.Q_DELTA, False, 1.0)

    def _can_fuse_input_silu(self, x: torch.Tensor) -> bool:
        """See OptimizedInt8Conv2d._can_fuse_input_silu for the rationale."""
        return (self.fuse_input_silu and self.modiff_enabled and not self.is_first_step
                and self.is_calibrated and HAS_CUTLASS and self.use_cutlass
                and self.a_hat_cache is not None
                and self.a_hat_cache.dtype == torch.float16
                and self.a_hat_cache.shape == x.shape
                and x.dtype == torch.float16)

    def _zp_unsupported(self, where, grid="activation"):
        """Raise/record when this conv's activation zero point is about to be ignored.

        `grid` NAMES WHAT THE CALLING SITE QUANTIZES, and it is the whole point of this signature:

          "activation" -- the site quantizes the activation itself on the activation grid, and its
                          conv adds the bias that _refold_zp_bias corrected. z MUST be applied here.
                          Ignoring it is not "the old behaviour": it is symmetric codes against a
                          corrected bias, which is worse than either choice alone.

          "delta"      -- the site quantizes a temporal DELTA (x - a_hat) and feeds a BIAS-FREE
                          o_hat accumulate. z is inapplicable by construction: it cancels in a
                          difference of activations, and adding it would additionally corrupt the
                          a_hat update, which dequantizes as q/s. Nothing is wrong at such a site.

        THIS PARAMETER REPLACED A CENSUS THAT CLASSIFIED BY ENTRY POINT NAME AND WAS WRONG BOTH WAYS.
        docs/zero_point_2026-08-13/FINDINGS.md counted 70 contaminated pairs on the MoDiff arm; 62 of
        them were step1_static_quantize_pack_int4_fprop, i.e. delta sites, and were false positives,
        while the one site that actually mattered -- _int4_conv's t=T scale_quantize_and_pack -- had no
        guard at all and so never appeared in the census. Measured, per call, at the kernel boundary:
        docs/zp_coverage_2026-08-13/data/site_census.json.

        COST: in the shipped symmetric configuration this is one attribute read and one float compare,
        exactly as before -- the `grid` branch sits AFTER the z == 0 exit. That ordering is deliberate:
        the previous version of these guards cost 3.5% of throughput by reading a CUDA scalar here
        (fixed in 2a2b1c3), and a guard on a hot path has to be free.
        """
        z = getattr(self, "_zp_float", 0.0)
        if z == 0.0:
            return
        if grid == "delta":
            # Only reachable in an asymmetric configuration, so this verification is free in the
            # shipped one. It checks the DECLARATION rather than trusting it: a delta site without an
            # a_hat cache is not quantizing a delta, and then the exemption below would be a hole.
            if self.a_hat_cache is None:
                raise RuntimeError(
                    f"{self.layer_name}: {where} declared grid='delta' but this conv has no a_hat "
                    f"cache, so it is not quantizing a delta -- the zero-point exemption does not "
                    f"apply and the codes would be symmetric against a corrected bias.")
            return
        name = self.layer_name or type(self).__name__
        _ZP_UNSUPPORTED.add(f"{name}|{where}")
        if os.environ.get("MODIFF_ZP_STRICT", "1") == "1":
            raise RuntimeError(
                f"{name}: activation zero point {z:+.0f} set, but {where} does not apply it while "
                f"the bias carries its correction. Zero the zp for this layer, or teach {where}.")

    def can_gn_fuse_modiff_cat2(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """Eligibility for the decoder skip-concat fold, decided from SHAPES ONLY.

        The obvious way to write this is to build a probe tensor of the concatenated shape and hand it
        to can_gn_fuse_modiff. That is what the first version did, and it allocated a full
        [N, C1+C2, H, W] fp16 tensor per block per step -- the size of the concatenation this
        optimisation exists to avoid touching. It would have consumed the whole saving to decide
        whether to take it. Everything the predicate needs is a shape or a flag, so nothing is
        allocated here.

        Mirrors _can_fuse_input_silu + can_gn_fuse_modiff exactly; if either changes, this must too.
        """
        if not (self.fuse_input_silu and self.modiff_enabled and not self.is_first_step
                and self.is_calibrated and HAS_CUTLASS and self.use_cutlass
                and getattr(self, 'groups', 1) == 1):
            return False
        if self.a_hat_cache is None or self.a_hat_cache.dtype != torch.float16:
            return False
        if a.dtype != torch.float16 or b.dtype != torch.float16 or not a.is_cuda:
            return False
        # a_hat_cache is indexed as the CONCATENATION, which is what the kernel writes and the apply
        # pass reads, so it must match that shape and not either half's.
        cat_shape = (a.shape[0], a.shape[1] + b.shape[1], a.shape[2], a.shape[3])
        return tuple(self.a_hat_cache.shape) == cat_shape

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

        d_scale, d_alpha = self._delta_scale_args_i4(x.device, x, fused_silu=True)

        p_step1 = profiler.start("MoDiff INT4 Static Step1 (fused SiLU)")
        self._zp_unsupported("step1_static_quantize_pack_int4_fprop_silu", grid="delta")
        x_packed = modiff_cutlass.step1_static_quantize_pack_int4_fprop_silu(
            x,
            self.a_hat_cache,
            d_scale,
            self._smooth_inv_flat,
        )
        profiler.stop("MoDiff INT4 Static Step1 (fused SiLU)", p_step1)
        if self._delta_calib:
            self._observe_delta_absmax()

        p_conv = profiler.start("MoDiff INT4 Static Conv2d")
        (modiff_cutlass.conv2d_int4_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int4_fprop_o_hat)(
            x_packed,
            self.weight_packed,
            d_alpha,
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
                                mod_scale2d, mod_shift2d, residual=None, x2=None):
        """int4 counterpart of OptimizedInt8Conv2d.forward_gn_fused_modiff:
        fused GroupNorm(+mod)+SiLU + INT4 delta-quantize+pack + o_hat conv,
        replacing the standalone GN kernel + step1_static_quantize_pack_int4_fprop_silu.
        Bit-identical to that two-kernel path. Caller must have verified
        can_gn_fuse_modiff(x) and the even-CPG / GN-native conditions.

        `x2` is the DECODER SKIP-CONCAT FOLD. When given, (x, x2) are the two halves the UNet
        would otherwise have concatenated with cat2_channels_last_fp16, and the kernel reads
        them in place while emitting the concatenation itself -- so the tensor is read twice
        in total instead of three times. Returns (out, cat) in that mode instead of out,
        because the ResBlock still consumes the concatenation twice more (the 1x1 skip conv
        and the out-conv's residual); handing it back is what makes splitting the skip conv
        unnecessary. Measured 51% of cat2 saved, ~1.65% end to end, and gated to
        bit-exactness against cat2 + this same path in integration/tests/test_cat2_gn_fold.py.
        """
        self.step_count += 1
        if x2 is not None:
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
            if not x2.is_contiguous(memory_format=torch.channels_last):
                x2 = x2.contiguous(memory_format=torch.channels_last)
        elif not x.is_contiguous(memory_format=torch.channels_last):
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

        # silu(gn(x)) exists only inside the fused kernel, so in dynamic mode the kernel does
        # the reduction itself (reusing its GN statistics) and the conv alpha comes from the
        # buffer it writes. See the int8 counterpart forward_gn_fused_modiff.
        gn_dyn = self._delta_gn_dynamic_args_i4(x.device)
        # d_scale QUANTIZES silu(gn(x)-a_hat) in the pack kernel below and d_alpha DEQUANTIZES the
        # int accumulation in the conv epilogue, so they must be reciprocals of the SAME quantity.
        # Take both from one source, exactly as the int8 twin does (int8_optimized.py:932).
        #
        # They used to be taken from two: d_alpha from _cached_alpha_tensor (1/static_input_scale)
        # and d_scale from _delta_scale_args_i4. That was consistent while static mode meant "use
        # the ACTIVATION scale for the delta" -- the pre-table behaviour this line's old comment
        # described. When the per-step delta table was added, _delta_scale_args_i4 started returning
        # static_delta_scale[i] while this site kept the activation alpha, so every fused-GN MoDiff
        # conv quantized on the delta grid and dequantized on the activation one. The error is
        # exactly the delta/activation gain -- measured 2.05x on the shipped W4A4 tables, and it
        # only appears once a table is actually loaded, which nothing did until 2026-08-12.
        d_scale, d_alpha = self._delta_scale_args_i4(x.device)
        if self.delta_dynamic:
            # Dynamic mode: the kernel writes the scale it used into _scale_buf and its reciprocal
            # into _inv_scale_buf. Override BOTH, never one.
            d_scale = self._scale_buf.view(1)
            d_alpha = self._inv_scale_buf.view(1)

        cat = None
        if x2 is not None:
            p_step1 = profiler.start("MoDiff INT4 GN-fused Step1 cat2 (concat+GN+SiLU+delta+pack)")
            x_packed, cat = modiff_cutlass.group_norm_silu_delta_quantize_pack_cat2_nhwc(
                x, x2, gn_weight, gn_bias, self.a_hat_cache, num_groups, eps, True,
                d_scale, self._smooth_inv_flat,
                mod_scale2d, mod_shift2d, *gn_dyn)
            profiler.stop("MoDiff INT4 GN-fused Step1 cat2 (concat+GN+SiLU+delta+pack)", p_step1)
        else:
            p_step1 = profiler.start("MoDiff INT4 GN-fused Step1 (GN+SiLU+delta+pack)")
            x_packed = modiff_cutlass.group_norm_silu_delta_quantize_pack_nhwc(
                x, gn_weight, gn_bias, self.a_hat_cache, num_groups, eps, True,
                d_scale, self._smooth_inv_flat,
                mod_scale2d, mod_shift2d, *gn_dyn)
            profiler.stop("MoDiff INT4 GN-fused Step1 (GN+SiLU+delta+pack)", p_step1)
        if self._delta_calib:
            self._observe_delta_absmax()

        if residual is not None:
            # EVT dual-store (see int8 counterpart OptimizedInt8Conv2d.forward_gn_fused_modiff
            # and this file's own forward_modiff_fused_silu_residual, which already uses this
            # kernel unconditionally): fold the ResBlock skip-add into the o_hat conv's
            # accumulate epilogue instead of the separate eager `out + residual` below.
            residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
            out = torch.empty_like(self.o_hat_cache)
            p_conv = profiler.start("MoDiff INT4 Static Conv2d (o_hat+residual)")
            modiff_cutlass.conv2d_int4_evt_o_hat_residual(
                x_packed, self.weight_packed, d_alpha,
                self.weight_scale_channel.view(-1), self.o_hat_cache, residual, out,
                self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1])
            profiler.stop("MoDiff INT4 Static Conv2d (o_hat+residual)", p_conv)
            return (out, cat) if cat is not None else out

        p_conv = profiler.start("MoDiff INT4 Static Conv2d")
        (modiff_cutlass.conv2d_int4_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int4_fprop_o_hat)(
            x_packed, self.weight_packed, d_alpha,
            self.weight_scale_channel.view(-1), self.o_hat_cache,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        profiler.stop("MoDiff INT4 Static Conv2d", p_conv)
        out = self._module_output()
        return (out, cat) if cat is not None else out

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

        d_scale, d_alpha = self._delta_scale_args_i4(x.device, x, fused_silu=True)

        p_step1 = profiler.start("MoDiff INT4 Static Step1 (fused SiLU)")
        self._zp_unsupported("step1_static_quantize_pack_int4_fprop_silu", grid="delta")
        x_packed = modiff_cutlass.step1_static_quantize_pack_int4_fprop_silu(
            x, self.a_hat_cache, d_scale, self._smooth_inv_flat)
        profiler.stop("MoDiff INT4 Static Step1 (fused SiLU)", p_step1)
        if self._delta_calib:
            self._observe_delta_absmax()

        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
        out = torch.empty_like(self.o_hat_cache)
        p_conv = profiler.start("MoDiff INT4 Static Conv2d (o_hat+residual)")
        # EVT dual-store (see int8 counterpart): o_hat RMW + residual in one conv pass,
        # no fp32 round-trip. Bit-exact o_hat + out vs conv2d_int4_fprop_o_hat_residual.
        modiff_cutlass.conv2d_int4_evt_o_hat_residual(
            x_packed, self.weight_packed, d_alpha,
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
                # Activation grid feeding _conv_from_int4's bias, so z applies. This kernel was left
                # GUARDED rather than taught at first, on the grounds that the 2026-08-13 census found
                # it unreachable on both shipped W4A4 arms (every calibrated conv goes through the GN
                # fusion or the resize kernel). That was correct but made the coverage claim
                # CONDITIONAL: a configuration that did reach it got a hard error instead of a result.
                # Taught 2026-08-13 so the claim is unconditional -- it is the same one-line `+ z`, and
                # this kernel takes no a_hat, so unlike upsample2x_* it has only the one role.
                zp = getattr(self, "_zp_float", 0.0)
                if zp != 0.0:
                    if not hasattr(modiff_cutlass, "step1_static_quantize_pack_int4_noahat_fprop_zp"):
                        raise RuntimeError(
                            "activation zero point is set but modiff_cutlass lacks "
                            "step1_static_quantize_pack_int4_noahat_fprop_zp -- rebuild the extension")
                    x_packed = modiff_cutlass.step1_static_quantize_pack_int4_noahat_fprop_zp(
                        x, self.static_input_scale.view(1), self._empty_smooth, zp)
                else:
                    x_packed = modiff_cutlass.step1_static_quantize_pack_int4_noahat_fprop(
                        x, self.static_input_scale.view(1), self._empty_smooth)
            else:
                x_for_quant = x if x.dtype == torch.float32 else x.float()
                zp = getattr(self, "_zp_float", 0.0)
                x_packed = (
                    modiff_cutlass.scale_quantize_and_pack_zp(
                        x_for_quant, self._cached_scale_tensor, zp) if zp != 0.0
                    else modiff_cutlass.scale_quantize_and_pack(
                        x_for_quant, self._cached_scale_tensor))
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

    def _conv_from_int4_o_hat(self, x_packed: torch.Tensor, h_in: int, w_in: int,
                             alpha: torch.Tensor) -> torch.Tensor:
        """INT4 twin of OptimizedInt8Conv2d._conv_from_int8_o_hat. h_in/w_in are passed explicitly
        because the packed tensor is a [N,H,W,C/2] byte buffer whose logical shape carries no
        spatial extents."""
        h_out = ((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                 // self.stride[0]) + 1
        w_out = ((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                 // self.stride[1]) + 1
        shape = (x_packed.shape[0], self.out_channels, h_out, w_out)
        dtype = torch.float16 if self.is_calibrated else torch.float32
        if (self.o_hat_cache is None or self.o_hat_cache.shape != shape
                or self.o_hat_cache.dtype != dtype):
            self.o_hat_cache = torch.zeros(shape, device=x_packed.device, dtype=dtype
                                           ).contiguous(memory_format=torch.channels_last)
        p = profiler.start("MoDiff INT4 Static Conv2d (from codes)")
        (modiff_cutlass.conv2d_int4_evt_o_hat if self.o_hat_cache.dtype == torch.float16
         else modiff_cutlass.conv2d_int4_fprop_o_hat)(
            x_packed, self.weight_packed, alpha, self.weight_scale_channel.view(-1),
            self.o_hat_cache,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        profiler.stop("MoDiff INT4 Static Conv2d (from codes)", p)
        return self._module_output()

    def _prepad_packed_with_zp(self, x_packed: torch.Tensor, h_in: int, w_in: int):
        """Spatially pad a packed int4 activation with the code `z`, returning (padded, h, w).

        x_packed is [N, H, W, C/2] bytes, low nibble = even channel. Both nibbles of the pad byte are
        z, so every padded tap decodes to code z and dequantizes to (z - z)/s = 0, which is what a
        zero-padded convolution is supposed to contribute.

        The nibble is TWO'S COMPLEMENT in 4 bits (z = -5 -> 0xB), matching how unpack_int4 and the
        kernels read it, and the byte is then reinterpreted as signed int8 for the tensor's dtype. Built
        with torch.full + a slice copy rather than F.pad, because F.pad's constant value goes through a
        float conversion that would not round-trip a negative int8 pad byte reliably.
        """
        ph, pw = int(self.padding[0]), int(self.padding[1])
        if HAS_PAD_PACKED_INT4_CODE:
            # ONE PASS, one allocation. The eager version below is three traversals plus a second
            # allocation per conv and measured +7.1% ms/step -- it gave back exactly the accuracy the
            # zero point buys. Gated bit-for-bit against that version in test_int4_zp_prepad.py, which
            # is why the eager path is kept rather than deleted.
            out = modiff_cutlass.pad_packed_int4_code(x_packed, ph, pw, self._zp_float)
            return out, h_in + 2 * ph, w_in + 2 * pw
        return self._prepad_packed_with_zp_eager(x_packed, h_in, w_in)

    def _prepad_packed_with_zp_eager(self, x_packed: torch.Tensor, h_in: int, w_in: int):
        """Pure-PyTorch reference for _prepad_packed_with_zp. Kept as the thing the CUDA kernel is
        gated against, and as the fallback when the extension predates it.

        torch.full + a slice copy rather than F.pad, because F.pad's constant value goes through a float
        conversion that would not round-trip a negative int8 pad byte reliably."""
        ph, pw = int(self.padding[0]), int(self.padding[1])
        nib = int(self._zp_float) & 0x0F
        byte = nib | (nib << 4)
        if byte > 127:
            byte -= 256
        N, H, W, Cb = x_packed.shape
        out = torch.full((N, H + 2 * ph, W + 2 * pw, Cb), byte,
                         dtype=x_packed.dtype, device=x_packed.device)
        out[:, ph:ph + H, pw:pw + W, :] = x_packed
        return out.contiguous(), h_in + 2 * ph, w_in + 2 * pw

    def _conv_from_int4(self, x_packed: torch.Tensor, h_in: int, w_in: int,
                        residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the calibrated INT4 conv (dequant/bias/store dispatch) on an
        already-quantized+packed activation ([N, H, W, C/2], the layout produced by
        scale_quantize_and_pack). Shared by _forward_standard (which quantizes first)
        and forward_from_int4 (which skips the quantize). h_in/w_in are the conv
        input's spatial dims (== x_packed.shape[1:3]). Optional `residual` (fp16
        channels_last, same shape as output) is fused into the store epilogue as the
        ResBlock skip-add."""
        # CORRECT ASYMMETRIC PADDING (MODIFF_ZP_PREPAD=1), experimental. CUTLASS's implicit GEMM
        # zero-fills padded taps, so with a zero point a padded tap reads code 0, which the grid
        # dequantizes to -z/s instead of 0, while the folded bias subtracts a per-output-CHANNEL
        # correction for a sample never taken. The correct padding VALUE is code z.
        #
        # This does it without a new kernel: pad the PACKED tensor with the byte whose two nibbles are
        # both z, then run the conv with padding=0 over the enlarged input. Exact, because every padded
        # tap then reads code z and dequantizes to (z - z)/s = 0 -- the same emulation
        # docs/zp_coverage_2026-08-13/scripts/zp_padding_probe.py used to identify the defect, moved
        # onto the real datapath.
        #
        # NOT THE SHIPPED PATH, and off by default: it materializes a padded copy of the activation per
        # conv, which is the round-trip the fused kernels exist to avoid. It exists to measure what fix
        # #2 is worth when its padding is RIGHT, so the negative answer rests on a measurement rather
        # than on an isolated ceiling. At z = 0 the pad byte is 0, i.e. exactly what CUTLASS inserts, so
        # this path is bit-identical to the normal one -- which is how it is gated.
        if (_zp_prepad_enabled() and getattr(self, "_zp_float", 0.0) != 0.0
                and (self.padding[0] or self.padding[1])):
            xp, hp, wp = self._prepad_packed_with_zp(x_packed, h_in, w_in)
            saved = self.padding
            self.padding = (0, 0)
            try:
                return self._conv_from_int4(xp, hp, wp, residual=residual)
            finally:
                self.padding = saved
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
        # Activation grid, and the conv that consumes these codes adds the corrected bias, so z
        # applies here exactly as it does in _int4_conv's tensor branch.
        zp = getattr(self, "_zp_float", 0.0)
        if zp != 0.0:
            return modiff_cutlass.scale_quantize_and_pack_zp(
                xf, self.static_input_scale.view(1), zp)
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
            # Dynamic per-round scale. Passing the static activation grid here (what this did) makes
            # the warm-up loop a no-op -- see OptimizedInt8Conv2d._forward_first_step.
            r_scale = (self._compute_scale_tensor(residual) if self.is_calibrated
                       else self._compute_activation_scale(residual, is_residual=True))
            conv_r  = self._int4_conv(residual, r_scale, with_bias=False)
            # activation_grid=False pairs with with_bias=False above: this is a residual on a dynamic
            # scale, so no zero point on either side. The two calls quantize the SAME codes -- a_hat
            # accumulates r_dq while o_hat accumulates their conv -- so they must agree about z.
            r_dq    = self._dequantize_activation(residual, r_scale, activation_grid=False)
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

            # No SiLU here: step1_static_quantize_pack_int4_fprop quantizes x itself.
            d_scale, d_alpha = self._delta_scale_args_i4(x.device, x, fused_silu=False)

            p_step1 = profiler.start("MoDiff INT4 Static Step1")
            self._zp_unsupported("step1_static_quantize_pack_int4_fprop", grid="delta")
            x_packed = modiff_cutlass.step1_static_quantize_pack_int4_fprop(
                x,
                self.a_hat_cache,
                d_scale,
                self._smooth_inv_flat,
            )
            profiler.stop("MoDiff INT4 Static Step1", p_step1)
            if self._delta_calib:
                self._observe_delta_absmax()

            p_conv = profiler.start("MoDiff INT4 Static Conv2d")
            (modiff_cutlass.conv2d_int4_evt_o_hat if self.o_hat_cache.dtype == torch.float16 else modiff_cutlass.conv2d_int4_fprop_o_hat)(
                x_packed,
                self.weight_packed,
                d_alpha,
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

        # ACT_CLIP_RATIO deliberately under-sizes the grid; see the constant for the sweep.
        static_scale = float(static_scale) * self.ACT_CLIP_RATIO
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

        # Identity smoothing where the weight range is zero -- see the int8 twin. Hygiene, not
        # numerics: a uniform s cancels against the scale, so codes are unchanged. It stops
        # static_input_scale and smooth_scale from being pinned to a clamp ceiling (~1e4x off), which
        # is what makes those fields readable in diagnostics.
        dead = w_max <= 1e-12
        if bool(dead.any()):
            s = torch.where(dead, torch.ones_like(s), s)

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
        # Same MSE-optimal scale as __init__; the two sites must agree or a SmoothQuant refold
        # would silently revert the layer to absmax.
        ch_scale = _int4_weight_scale(w_flat)

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

    def _refold_zp_bias(self):
        """Fold the activation zero point's per-output-channel correction into the stored bias.

        bias_eff[k] = bias[k] - z * sum_i w_q[k,i] * ws[k] / s

        Done ONCE here rather than in the forward path, because z and s are fixed by calibration and
        do not change during sampling. That is what keeps the zero point out of every conv call site,
        the EVT epilogue and the GEMM entirely -- the arithmetic is identical, it just happens at
        calibration time. At z = 0 the correction is exactly 0.0 and `bias` is restored byte-for-byte
        from _orig_bias, so an asymmetric file and a symmetric one cannot drift apart.

        A conv with NO bias gains one when z != 0: the correction is a real per-channel offset and
        there is nowhere else to put it. It is dropped again (back to None) if z returns to 0.
        """
        z = float(self.static_input_zp.item())
        self._zp_float = z          # calibration-time sync is fine; the forward path reads the mirror
        # ZERO-PADDED CONVS CANNOT USE THIS FOLD, and that is measured, not suspected.
        #
        # The fold subtracts z * sum_i w_q[k,i] -- the sum over the WHOLE kernel, one constant per
        # output channel. A padded output pixel's implicit GEMM reads literal 0 on its missing taps,
        # which dequantizes to (0 - z)/s = -z/s rather than 0, so that pixel needs the sum over only
        # the taps it actually has. The residual is exactly -z * sum_{missing} w_q[k] * ws[k] / s,
        # confirmed to 1-2.6% against the kernel in integration/tests/test_int4_zp_padding.py, and it
        # lands on the border ring: 23% of pixels at 16x16, 44% at 8x8, 75% at 4x4. End to end it cost
        # +70% (PTQ) and +170% (MoDiff) with coverage otherwise complete.
        #
        # REFUSING RATHER THAN CORRECTING, because the correction is not worth building: an asymmetric
        # grid reduces the reconstruction error of silu(gn(x)) by only 1.06x once the symmetric grid is
        # clipped (docs/zp_coverage_2026-08-13/data/zp_activation_error.json, 70 convs) -- under the
        # 1.15x bar this tree set for fix #2 in zp_headroom.py, and that is a CEILING before any
        # downstream dilution. Fix #3's clip ratio already took the slack the zero point was after.
        #
        # MODIFF_ZP_ALLOW_PADDED=1 re-enables it for experiments. It does not make it correct.
        if z != 0.0 and tuple(self.padding) != (0, 0) \
                and os.environ.get("MODIFF_ZP_ALLOW_PADDED", "0") != "1":
            raise RuntimeError(
                f"{self.layer_name or type(self).__name__}: activation zero point {z:+.0f} on a conv "
                f"with padding {tuple(self.padding)}. The bias fold is per-output-channel and the "
                f"padding error is per-output-pixel, so this configuration is wrong on the border "
                f"ring by -z*sum(missing w_q)*ws/s -- see docs/zp_coverage_2026-08-13/FINDINGS.md. "
                f"Set MODIFF_ZP_ALLOW_PADDED=1 only to reproduce that measurement.")
        s_in = float(self.static_input_scale.item())
        if z == 0.0 or s_in == 0.0:
            self.bias = None if self._orig_bias is None else self._orig_bias.clone()
            return
        ws = self.weight_scale_channel.view(1, -1, 1, 1).to(torch.float32)
        corr = -(z / s_in) * self.weight_sum_q.to(torch.float32) * ws
        base = self._orig_bias
        if base is None:
            self.bias = corr.to(self.weight_scale_channel.dtype)
        else:
            self.bias = (base.to(torch.float32) + corr).to(base.dtype)
        if getattr(self, '_evt_bias_f32', None) is not None:
            # Invalidate the cached fp32 flattening the EVT epilogue reads, or it keeps the old bias.
            self._evt_bias_f32 = None

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
        # The correction scales as 1/s, so it must be refolded whenever s changes.
        self._refold_zp_bias()

    def set_static_calibration(self, scale: float, smooth_scale: Optional[torch.Tensor] = None,
                               zero_point: float = 0.0):
        """Restore a full static calibration from an exported checkpoint: the per-tensor
        activation `scale` AND, when the layer used SmoothQuant, the per-in-channel
        `smooth_scale`. The smooth scale is re-folded into the (freshly converted, still
        unsmoothed) weights via _fold_weights_with_smooth, so at inference the smoothed
        activations meet smoothed weights and the static scale — which was derived from
        the *smoothed* activation range — is correct.

        Without restoring smooth_scale (the old apply path, i.e. plain set_static_scale),
        a SmoothQuant-derived static scale gets applied to UNsmoothed weights with no
        activation smoothing: int8 masks the mismatch (8-bit range) but int4's 4-bit
        range degrades it ~2x (ONE layer's MoDiff first step, rel ~0.40 against ~0.20 for
        live calibration; d77c516). If the weights can't be re-folded (odd in_channels, or
        _orig_weight already released), we fall back to that scale-only path rather than
        smoothing activations against unsmoothed weights, which would be strictly worse.

        SMOOTHQUANT IS NOT IN THE SHIPPED PATH ANY MORE, so this whole branch is now the
        fallback rather than the norm. The tree follows the paper (README:96,
        --modulate --quant_mode qdiff --cali_min_max), and Q-Diffusion has no SmoothQuant:
        int4_calibration_qdiff.pt is therefore BARE FLOATS, which routes through
        set_static_scale and leaves smoothing identity. This method still runs, exactly as
        written, for the _realckpt.pt fallback and for any externally exported dict — the
        semantics are unchanged, only which file the default resolves to.

        Keep the ~2x above in proportion: it is ONE layer's first step on a synthetic conv.
        test_int4_export_apply's fixture cannot even reproduce the effect (its per-channel
        activation spread is 1.26 against the real checkpoint's 4.83, so its s is near
        uniform and cancels), which is why that gate asserts bit-exact state restoration
        rather than an accuracy margin."""
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
        # Written BEFORE set_static_scale, which is what folds the correction into the bias.
        self.static_input_zp.fill_(float(zero_point))
        self.set_static_scale(scale)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

def convert_model_to_optimized_int4(model: nn.Module, prefix: str = "", use_compile: bool = False,
                                     skip_pointwise: bool = True,
                                     _memo: dict = None) -> nn.Module:
    """Wrap every eligible nn.Conv2d in OptimizedInt4Conv2d.

    DEDUPLICATED BY OBJECT IDENTITY (`_memo`), added 2026-08-13 to fix a 114 MiB leak.

    FusedResBlock aliases one conv under two attributes -- `fused.in_conv` IS
    `fused.original.in_layers[-1]` (fused_resblock.py:756), and likewise for out_conv. This walk
    recurses over named_children(), so it reached the SAME nn.Conv2d down two paths and wrapped it
    TWICE, into two independent modules each holding its own packed int4 weights. Only the one
    `forward` uses was ever called or calibrated; the other 70 sat inert with modiff_enabled=True.

    114 MiB, NOT THE 1014.6 MiB THIS DOCSTRING USED TO CLAIM -- corrected 2026-08-13 after measuring
    it (docs/zp_coverage_2026-08-13/scripts/orphan_wrapper_memory.py, which builds the model twice,
    once with the memo defeated, and diffs the allocator). 1014.6 MiB is the sum of every tensor the
    orphans REFERENCE, and 901 MiB of that aliases storage that was already there: both wrappers wrap
    ONE nn.Conv2d, so its fp16 weight is one allocation with two references. What the orphans actually
    allocated for themselves -- packed int4 weights and their own buffers -- is 113.6 MiB by
    per-storage sum and 114.3 MiB by allocator delta, which agree to 0.7%. Still worth fixing; not 37%
    of the model.

    The memo makes the second path reuse the first wrapper, so one object is referenced from both.

    THE NAME MATTERS AND IS NOT COSMETIC. apply_int{{4,8}}_static_scales matches on
    `module.layer_name`, and the calibration files key on the NON-`.original.` path, which is why 70
    of 140 wrappers loaded a scale before this fix. named_children() registers `original` before
    `in_conv`, so the first wrapper created carries the `.original.` name -- keeping it would have
    silently dropped calibration to 0 layers while looking like a pure memory win. The memo therefore
    upgrades layer_name to the non-`.original.` path when it sees it.
    """
    if _memo is None:
        _memo = {}
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

            hit = _memo.get(id(child))
            if hit is not None:
                # Same underlying conv, reached by a second alias. Reuse the one wrapper, and prefer
                # the name the calibration files key on (see the docstring).
                if ".original." in (getattr(hit, "layer_name", "") or "") \
                        and ".original." not in full_name:
                    hit.layer_name = full_name
                setattr(model, name, hit)
                continue
            optimized_conv = OptimizedInt4Conv2d(child, layer_name=full_name, use_compile=use_compile)
            _memo[id(child)] = optimized_conv
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized_conv = optimized_conv.to(target_device)
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int4(child, prefix=full_name, use_compile=use_compile,
                                             skip_pointwise=skip_pointwise,
                                             _memo=_memo)

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


# ----------------------------------------------------------------------
# MoDiff per-step delta-scale table: model-level driver. Mirrors the int8 helpers in
# int8_optimized.py one-for-one, so a caller can drive either bit-width the same way.
# ----------------------------------------------------------------------

def begin_delta_calibration_int4(model: nn.Module, reset: bool = False) -> int:
    """Arm delta-range observation on every calibrated int4 conv. Returns how many were armed.

    Must run AFTER the ordinary activation calibration: the observation scale is derived from
    static_input_scale, and t=T's Q(a_T) needs it too.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, OptimizedInt4Conv2d) and m.is_calibrated:
            m.begin_delta_calibration(reset=reset)
            n += 1
    return n


def end_delta_calibration_int4(model: nn.Module, safety: float = 1.02) -> int:
    """Convert observations into per-step tables. Returns how many layers got one."""
    n = 0
    for m in model.modules():
        if isinstance(m, OptimizedInt4Conv2d):
            if m.end_delta_calibration(safety=safety):
                n += 1
    return n


def export_int4_delta_scales(model: nn.Module) -> Dict[str, object]:
    """Export the delta table keyed by layer name. Deliberately a SEPARATE artifact from
    export_int4_static_scales -- the two describe different quantities (activation range vs
    temporal-delta range) and are valid for different modes."""
    out = {}
    for m in model.modules():
        if isinstance(m, OptimizedInt4Conv2d) and bool(m.is_delta_calibrated):
            out[m.layer_name] = m.static_delta_scale.detach().to("cpu", torch.float32).clone()
    return out


def apply_int4_delta_scales(model: nn.Module, table: Dict[str, object]) -> int:
    """Load a table from export_int4_delta_scales. Returns how many layers were filled."""
    if not table:
        return 0
    loaded = 0
    for m in model.modules():
        if not isinstance(m, OptimizedInt4Conv2d) or m.layer_name not in table:
            continue
        t = table[m.layer_name].to(m.static_delta_scale.device, torch.float32)
        n = min(t.numel(), m.static_delta_scale.numel())
        m.static_delta_scale[:n].copy_(t[:n])
        if n < m.static_delta_scale.numel():          # forward-fill a shorter saved table
            m.static_delta_scale[n:].fill_(float(t[n - 1]))
        m.static_delta_alpha.copy_(1.0 / m.static_delta_scale.clamp_min(1e-12))
        m.is_delta_calibrated.fill_(True)
        m._delta_cal = True
        loaded += 1
    return loaded


def delta_calibration_report_int4(model: nn.Module):
    """Per-layer diagnostics: how much finer the delta grid is than the activation grid."""
    return [m.delta_calibration_report() for m in model.modules()
            if isinstance(m, OptimizedInt4Conv2d) and bool(m.is_delta_calibrated)]


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
                    module.set_static_calibration(entry["static_scale"], entry.get("smooth_scale"),
                                                  float(entry.get("zero_point", 0.0)))
                else:
                    module.set_static_scale(float(entry))
                loaded += 1
    return loaded
