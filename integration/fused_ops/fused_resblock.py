"""
Fused ResBlock operations for MoDiff optimizations.

This module provides fused implementations of ResBlock operations to reduce
memory bandwidth and kernel launch overhead. The main optimizations:

1. Fused GroupNorm + SiLU: prefers a native channels_last CUDA kernel
   (csrc/kernels/group_norm_silu.cu, see _group_norm_silu below) that reads
   and writes NHWC-physical memory directly, so it never forces the
   channels_last -> NCHW -> channels_last round-trip that plain F.group_norm
   causes for every downstream quantized conv. Falls back to F.group_norm +
   F.silu with autocast locally disabled (see FusedGroupNormSiLU below) when
   the native kernel doesn't apply (CPU, non-channels_last input, dtype other
   than fp32/fp16).
2. Fused residual addition: Combines skip connection with final output
3. In-place operations where possible to reduce memory allocations

Measured speedup (autocast-disabled F.group_norm path, before the native
channels_last kernel was added): ~11% wall-time reduction on the fp16
LSUN-churches UNet (no measurable effect in int8/int4 modes -- see
FusedGroupNormSiLU docstring for why their GroupNorm inputs are already fp32
for an unrelated reason).

After adding the native channels_last kernel: full `benchmark_ldm.py --mode
all --steps 200 --batch_size 16 --num_samples 32` run, time/sample before ->
after (int8/int4 unlike fp16 DO benefit here, since their GroupNorm inputs
are channels_last fp32, the exact case the old F.group_norm round-trip hit):
    fp32:          1.180s -> 1.151s  (~2%)
    fp16:          0.419s -> 0.400s  (~5%)
    int8 (MoDiff): 0.442s -> 0.341s  (~23%)
    int8_baseline: 0.417s -> 0.318s  (~24%)
    int4 (MoDiff): 0.409s -> 0.318s  (~22%)
    int4_baseline: 0.409s -> 0.343s  (~16%)
The int4 (MoDiff) vs int4_baseline ordering above (0.318s vs 0.343s) did NOT
reproduce under controlled re-profiling (matched 30-step runs with static
calibration explicitly loaded): there they came out statistically tied
(1308ms vs 1311ms total GPU time, ~0.2% apart) -- so treat that single-run
flip as cuDNN benchmark-mode algorithm-selection noise across separate model
loads, not a real effect of this kernel. See analysis_int4_vs_int8/ for why
int4 doesn't show its ~2x theoretical speedup over int8 pipeline-wide even
though the raw CUTLASS INT4 conv kernel itself does (~1.8x faster than INT8's
in this same profiling run): conv is only ~8-14% of total GPU time, while
attention (unquantized, ~38-40%) and non-quantized fp32/fp16 convs (~17-18%)
are identical between precisions and dominate the pipeline.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- int8-conv-output quality probe (MODIFF_CONV_INT8_OUT = pt | pc) --------------
# Fake-quant the in_conv fp16 output through int8 before the out-norm GN reads it,
# to measure the quality ceiling of "conv writes int8 -> GN reads int8" BEFORE
# building the CUDA kernel. pt = per-tensor scale, pc = per-output-channel scale.
_CONV_INT8_OUT = os.environ.get("MODIFF_CONV_INT8_OUT", "")
def _fake_quant_int8_out(h):
    if _CONV_INT8_OUT == "pt":
        s = h.abs().amax().clamp_min(1e-8) / 127.0
        return (torch.round(h / s).clamp_(-127, 127)) * s
    if _CONV_INT8_OUT == "pc":                       # per-channel over (N,H,W), dim=1=C
        amax = h.abs().amax(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
        s = amax / 127.0
        return (torch.round(h / s).clamp_(-127, 127)) * s
    return h

# Import TimestepBlock for proper integration with TimestepEmbedSequential
try:
    from ldm.modules.diffusionmodules.openaimodel import TimestepBlock
    HAS_TIMESTEP_BLOCK = True
except ImportError:
    # Fallback if import fails
    HAS_TIMESTEP_BLOCK = False
    class TimestepBlock(nn.Module):
        pass

try:
    import modiff_cutlass
    HAS_NATIVE_GN_SILU = hasattr(modiff_cutlass, "group_norm_silu_nhwc")
    HAS_GN_SILU_QUANTIZE = hasattr(modiff_cutlass, "group_norm_silu_quantize_nhwc")
    HAS_GN_SILU_QUANTIZE_PACK = hasattr(modiff_cutlass, "group_norm_silu_quantize_pack_nhwc")
    # MoDiff GN->delta-quantize fusion (default ON): fuse GroupNorm(+mod)+SiLU into
    # the delta-quantize + a_hat update, replacing the standalone GN kernel + separate
    # step1 pass and removing the intermediate fp16 `normed` round-trip. Bit-identical
    # to that two-kernel path (repo gate ALL PASS).
    #
    # The earlier single group-major fused kernel was OPT-IN/OFF because it regressed
    # ~2-3 ms/step -- its pass 2 did the fp16 a_hat read-modify-write group-major, so
    # in NHWC a group's a_hat access was strided by C (~4-8x uncoalesced at the
    # dominant low-CPG / high-spatial shapes), costing more than the round-trip it
    # saved. The kernel is now SPLIT (group_norm_silu.cu) into a group-major stats
    # reduction + a flat, fully-coalesced apply pass; measured kernel A/B (b128) beats
    # the two-kernel default at every real conv-input shape (int8 1.07-1.18x, int4
    # 1.17-1.42x) and the old fused kernel by up to 2.34x. Disable with
    # MODIFF_DISABLE_GN_MODIFF_FUSION=1. See docs/benchmark_5mode_2026-07-20.
    HAS_GN_SILU_DELTA_QUANTIZE = hasattr(modiff_cutlass, "group_norm_silu_delta_quantize_nhwc")
    HAS_GN_SILU_DELTA_QUANTIZE_PACK = hasattr(modiff_cutlass, "group_norm_silu_delta_quantize_pack_nhwc")
    # MoDiff o_hat + residual fusion (default ON): fold the ResBlock skip-add into
    # the o_hat conv's accumulate epilogue (conv2d_intX_fprop_o_hat_residual),
    # removing the trailing aten::add. Low-risk (elementwise, coalesced; the o_hat
    # cache write is byte-identical). Disable with MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION=1.
    HAS_O_HAT_RESIDUAL = hasattr(modiff_cutlass, "conv2d_int8_fprop_o_hat_residual")
except ImportError:
    modiff_cutlass = None
    HAS_NATIVE_GN_SILU = False
    HAS_GN_SILU_QUANTIZE = False
    HAS_GN_SILU_QUANTIZE_PACK = False
    HAS_GN_SILU_DELTA_QUANTIZE = False
    HAS_GN_SILU_DELTA_QUANTIZE_PACK = False
    HAS_O_HAT_RESIDUAL = False

# Kill-switch for the GN->intX K1-fusion (baseline int8/int4). Set
# MODIFF_DISABLE_GN_INT8_FUSION=1 to fall back to the exact two-kernel
# (GroupNorm+SiLU, then standalone quantize[/pack]) path -- used for A/B
# benchmarking and as a production safety switch. Pure optimization either way.
if os.environ.get("MODIFF_DISABLE_GN_INT8_FUSION") == "1":
    HAS_GN_SILU_QUANTIZE = False
    HAS_GN_SILU_QUANTIZE_PACK = False

if os.environ.get("MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION") == "1":
    HAS_O_HAT_RESIDUAL = False

# Kill-switch for the MoDiff GN->delta-quantize fusion (default ON). Set
# MODIFF_DISABLE_GN_MODIFF_FUSION=1 to fall back to the exact two-kernel modiff path
# (standalone GroupNorm(+SiLU) kernel, then step1_static_quantize[_pack]_fprop_silu)
# -- used for A/B benchmarking and as a production safety switch.
if os.environ.get("MODIFF_DISABLE_GN_MODIFF_FUSION") == "1":
    HAS_GN_SILU_DELTA_QUANTIZE = False
    HAS_GN_SILU_DELTA_QUANTIZE_PACK = False


def _group_norm_silu(x, num_groups, weight, bias, eps, apply_silu, mod_scale=None, mod_shift=None):
    """GroupNorm (+ optional SiLU), preferring the native channels_last CUDA
    kernel (csrc/kernels/group_norm_silu.cu) that reads/writes NHWC-physical
    memory directly. F.group_norm always returns NCHW-contiguous output
    regardless of its input's memory format, which forces a real copy back
    to channels_last for every downstream quantized conv (profiled and
    confirmed on this pipeline) -- the native kernel never materializes that
    NCHW intermediate. Falls back to F.group_norm(+F.silu) with autocast
    disabled locally for shapes/dtypes/devices the kernel doesn't cover
    (CPU, non-channels_last input, non-power-of-32-divisible channel counts,
    dtypes other than fp32/fp16).
    """
    can_use_native = (
        HAS_NATIVE_GN_SILU
        and x.is_cuda
        and x.dtype in (torch.float32, torch.float16)
        and x.is_contiguous(memory_format=torch.channels_last)
        and x.size(1) % num_groups == 0
    )
    # Optional use_scale_shift_norm modulation `normed*(1+scale)+shift`, folded into
    # the kernel (fp16 output) so the MoDiff modulated path avoids a separate
    # elementwise mul+add before its delta-quantize.
    if mod_scale is not None:
        N, C = x.size(0), x.size(1)
        ms2d = mod_scale.reshape(N, C).contiguous()
        sh2d = mod_shift.reshape(N, C).contiguous()
    else:
        ms2d = sh2d = x.new_empty(0)
    if can_use_native:
        return modiff_cutlass.group_norm_silu_nhwc(x, weight, bias, num_groups, eps, apply_silu, ms2d, sh2d)
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        out = F.group_norm(x, num_groups, weight, bias, eps)
        if mod_scale is not None:
            out = out * (1 + mod_scale) + mod_shift
        return F.silu(out) if apply_silu else out


def _prequant_common_ok(conv):
    """Common gates for the GN->intX K1-fusion, shared by int8 and int4.

    Baseline-only: the MoDiff modulated path (modiff_enabled=True) does an in-place
    a_hat cache update inside its quantize that a fused GN-emit would bypass and
    corrupt, so we require modiff_enabled=False. Grouped convs / non-fp16 output fall
    back to the exact path -- pure optimization, never a correctness change.

    SmoothQuant (non-identity _smooth_inv, used by int4) is NOT excluded: the fused
    group_norm_silu_quantize[_pack]_nhwc kernel applies a per-channel smooth_inv before
    the quantize (see _prequant_gn_conv), so it is folded into the fused path rather
    than paying an eager `x * smooth_inv` multiply + separate quantize.
    """
    return (
        getattr(conv, 'is_calibrated', False)
        and not getattr(conv, 'modiff_enabled', True)
        and getattr(conv, 'standard_output_fp16', False)
        and getattr(conv, 'use_cutlass', False)
        and getattr(conv, 'groups', 1) == 1
    )


def _prequant_gn_conv_modiff(x, gn, conv, mod_scale=None, mod_shift=None, residual=None):
    """MoDiff (temporal-cache) counterpart of the baseline GN->intX fusion:
    fuse GroupNorm(+scale-shift mod)+SiLU into the conv's delta-quantize + a_hat
    update kernel (group_norm_silu_delta_quantize[_pack]_nhwc), then run the o_hat
    conv, in place of the standalone GroupNorm kernel + step1_static_quantize_fprop_silu
    two-kernel pass. Bit-identical to that path (the kernel replicates the fp16
    `normed` rounding before SiLU). Returns the conv output (with residual added
    when given), or None to fall through to the existing modiff two-kernel path
    (first step / uncalibrated / kernel unavailable / not native-GN eligible).
    """
    if conv is None or not getattr(conv, 'modiff_enabled', False):
        return None
    if not hasattr(conv, 'can_gn_fuse_modiff') or not conv.can_gn_fuse_modiff(x):
        return None
    is_int4 = hasattr(conv, 'forward_from_int4')
    if is_int4 and not HAS_GN_SILU_DELTA_QUANTIZE_PACK:
        return None
    if not is_int4 and not HAS_GN_SILU_DELTA_QUANTIZE:
        return None

    ng = gn.num_groups
    N, C = x.size(0), x.size(1)
    # GN-native eligibility (same conditions as _group_norm_silu's can_use_native).
    if not (x.is_cuda and x.dtype in (torch.float32, torch.float16)
            and x.is_contiguous(memory_format=torch.channels_last) and C % ng == 0):
        return None
    if is_int4 and (C % 2 != 0 or (C // ng) % 2 != 0):
        return None

    weight, bias = gn._cast_params(x.dtype)
    if mod_scale is not None:
        ms2d = mod_scale.reshape(N, C).contiguous()
        sh2d = mod_shift.reshape(N, C).contiguous()
    else:
        ms2d = sh2d = x.new_empty(0)
    if residual is not None:
        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
    return conv.forward_gn_fused_modiff(x, weight, bias, ng, gn.eps, ms2d, sh2d, residual)


def _modiff_out_conv(conv, h, residual_arg):
    """Run a modiff out-conv, folding the ResBlock skip-add into the o_hat conv's
    accumulate epilogue when eligible (conv2d_intX_fprop_o_hat_residual via
    forward_modiff_fused_silu_residual). `h` is the pre-SiLU GroupNorm output.
    Returns (output, residual_fused): residual_fused=True means the skip is already
    added (caller must NOT add it again). Falls back to the plain conv (skip added
    by the caller) when not eligible (first step / uncalibrated / no residual /
    kernel unavailable)."""
    if (residual_arg is not None and HAS_O_HAT_RESIDUAL
            and getattr(conv, 'modiff_enabled', False)
            and hasattr(conv, 'forward_modiff_fused_silu_residual')
            and hasattr(conv, '_can_fuse_input_silu') and conv._can_fuse_input_silu(h)):
        return conv.forward_modiff_fused_silu_residual(h, residual_arg), True
    return conv(h), False


def _prequant_gn_conv(x, gn, conv, mod_scale=None, mod_shift=None, residual=None):
    """If (GroupNorm+SiLU `gn` -> quantized conv `conv`) is eligible for the
    GN->intX K1-fusion, run GroupNorm(+optional scale-shift modulation)+SiLU
    emitting the conv's quantized input directly (int8, or packed int4) and then
    conv.forward_from_intX, returning the conv output. Otherwise return None so the
    caller uses the normal path.

    `mod_scale`/`mod_shift` (each [N, C, 1, 1] from the timestep embedding, or None)
    add the use_scale_shift_norm modulation `normed*(1+scale)+shift` between the GN
    affine and the SiLU, folding it (and the SiLU) into the one kernel too.
    `residual` (the ResBlock skip tensor, or None) is fused into the conv's store
    epilogue as a skip-add, removing the trailing aten::add. The quantize multiplier
    is conv.static_input_scale (=127/absmax); smooth_inv is identity (gated above).
    """
    # MoDiff temporal-cache path: fuse GroupNorm(+mod)+SiLU into the delta-quantize
    # (+ a_hat update) kernel, replacing the standalone GN kernel + step1 two-kernel
    # pass. Bit-identical to that path; only kicks in once the layer is calibrated
    # with an fp16 a_hat cache (step >= 2). See _prequant_gn_conv_modiff.
    modiff_fused = _prequant_gn_conv_modiff(x, gn, conv, mod_scale, mod_shift, residual)
    if modiff_fused is not None:
        return modiff_fused

    if not _prequant_common_ok(conv):
        return None
    # Residual must be fp16 channels_last matching the conv output's flat layout.
    if residual is not None:
        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
    is_int4 = hasattr(conv, 'forward_from_int4')
    if is_int4 and not HAS_GN_SILU_QUANTIZE_PACK:
        return None
    if not is_int4 and (not HAS_GN_SILU_QUANTIZE or not hasattr(conv, 'forward_from_int8')):
        return None

    ng, eps = gn.num_groups, gn.eps
    w, b = gn._cast_params(x.dtype)
    conv._ensure_conv_caches(x.device)
    scale = conv._cached_scale_tensor          # fp32 [1], =static_input_scale
    # SmoothQuant: fold the per-channel smooth_inv into the fused GN->quantize kernel
    # (it applies `out *= smooth_inv[c]` before the quantize), matching the eager
    # `x * smooth_inv` the standard path would otherwise pay. int8 is identity -> empty.
    if getattr(conv, '_smooth_is_identity', True):
        smooth_inv = conv._empty_smooth        # empty -> identity
    else:
        smooth_inv = conv._smooth_inv.view(-1).to(torch.float32).contiguous()
    N, C = x.size(0), x.size(1)
    native_ok = (
        x.is_cuda
        and x.dtype in (torch.float32, torch.float16)
        and x.is_contiguous(memory_format=torch.channels_last)
        and C % ng == 0
    )

    # Modulation: kernel wants [N, C] contiguous same-dtype tensors (nullptr when
    # empty); the fallback broadcasts the original [N, C, 1, 1] form.
    if mod_scale is not None:
        ms2d = mod_scale.reshape(N, C).contiguous()
        sh2d = mod_shift.reshape(N, C).contiguous()
    else:
        ms2d = sh2d = x.new_empty(0)

    if is_int4:
        # int4 packs channel pairs within a group -> needs even channels-per-group.
        if C % 2 != 0 or (C // ng) % 2 != 0:
            return None
        h_in, w_in = x.shape[2], x.shape[3]
        if native_ok:
            packed = modiff_cutlass.group_norm_silu_quantize_pack_nhwc(
                x, w, b, ng, eps, True, scale, smooth_inv, ms2d, sh2d)
        else:
            packed = modiff_cutlass.scale_quantize_and_pack(
                _gn_mod_silu_fp32_cl(x, ng, w, b, eps, mod_scale, mod_shift), scale)
        return conv.forward_from_int4(packed, h_in, w_in, residual=residual)
    else:
        if native_ok:
            q = modiff_cutlass.group_norm_silu_quantize_nhwc(
                x, w, b, ng, eps, True, scale, smooth_inv, ms2d, sh2d)
        else:
            q = modiff_cutlass.scale_quantize_int8(
                _gn_mod_silu_fp32_cl(x, ng, w, b, eps, mod_scale, mod_shift), scale)
        return conv.forward_from_int8(q, residual=residual)


def _gn_mod_silu_fp32_cl(x, num_groups, weight, bias, eps, mod_scale, mod_shift):
    """Fallback for _prequant_gn_conv: GroupNorm (+ optional scale-shift
    modulation) + SiLU, returned as an fp32 channels_last tensor (the standalone
    scale_quantize[_and_pack] kernels read via float vectors). Matches the native
    kernel's op order exactly."""
    if mod_scale is None:
        h = _group_norm_silu(x, num_groups, weight, bias, eps, apply_silu=True)
    else:
        h = _group_norm_silu(x, num_groups, weight, bias, eps, apply_silu=False)
        h = h * (1 + mod_scale) + mod_shift
        h = F.silu(h)
    h = h if h.dtype == torch.float32 else h.float()
    if not h.is_contiguous(memory_format=torch.channels_last):
        h = h.contiguous(memory_format=torch.channels_last)
    return h


class FusedGroupNormSiLU(nn.Module):
    """
    Fused GroupNorm + SiLU activation.

    This replaces the common pattern:
        nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU()
        )

    Delegates to _group_norm_silu(), which prefers the native channels_last
    CUDA kernel and falls back to PyTorch's native F.group_norm + F.silu with
    autocast locally disabled around the call.

    F.group_norm is on autocast's fp32-cast list (numerical stability for its
    mean/var reduction), so under `torch.amp.autocast(dtype=torch.float16)`
    a plain F.group_norm call on fp16 input silently costs two extra dtype-
    cast kernels: fp16->fp32 forced by autocast before group_norm runs, and
    fp32->fp16 forced by autocast before the next (fp16-autocast-listed)
    conv runs -- since F.silu itself isn't autocast-managed and just passes
    the fp32 straight through in between. PyTorch's native GroupNorm CUDA
    kernel already accumulates its mean/var reduction in fp32 internally
    regardless of the tensor's dtype (standard practice for numerical
    stability), so it doesn't actually need autocast's help -- disabling
    autocast for just this call lets it run natively in fp16 (no materialized
    fp32 tensor, no cast kernels) while keeping PyTorch's own optimized
    implementation.

    Measured end-to-end on the fp16 LSUN-churches UNet: ~24% reduction in the
    dtype-cast/direct-copy kernel bucket and ~23% reduction in the group_norm
    bucket, for an overall ~11% wall-time reduction (3356ms -> 2996ms at
    batch=168, 10 DDIM steps). This does NOT meaningfully help int8/int4
    modes: profiling showed their FusedGroupNormSiLU inputs are already fp32
    by the time they arrive (the quantize/dequant kernels' accumulator output
    is fp32 for precision, independent of autocast), so there's no cast for
    this fix to eliminate there -- fixing that would mean changing the
    dequant kernels' output dtype, a separate, riskier change not made here.

    This also beat a hand-written Triton fused kernel that was tried first:
    numerically correct, but slower than plain F.group_norm for this model's
    channel/resolution sizes (~3.4ms/call vs ~1.9ms/call for the
    autocast-disabled native path, in the same pipeline).
    """
    def __init__(self, num_groups, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # weight/bias stay fp32 (the nn.Parameter default) while activations run
        # fp16 under this inference-only quantized pipeline, so every call used to
        # re-launch 2 small cast kernels just to match x's dtype. Since the
        # parameters don't change between calls in eval/inference mode, cache the
        # cast result per dtype instead of recomputing it on every forward.
        self._cast_dtype = None
        self._cast_weight = None
        self._cast_bias = None

        # Set by wire_silu_fusion() below when the following conv is a
        # calibrated OptimizedInt8Conv2d/OptimizedInt4Conv2d with
        # fuse_input_silu=True: that layer applies SiLU itself (fused into
        # its quantize kernel, or via its own F.silu fallback), so this
        # module should return its GroupNorm-only output instead of also
        # applying SiLU here.
        self.skip_silu = False

    def _cast_params(self, dtype):
        if self._cast_dtype is not dtype:
            # .detach() first: Parameter.to(dtype) is a no-op returning `self`
            # (still a Parameter) when dtype already matches, which would let
            # nn.Module's __setattr__ register _cast_weight as a real parameter
            # slot on that call -- then break on the next call with a different
            # dtype, since a registered parameter slot rejects a plain Tensor.
            self._cast_weight = self.weight.detach().to(dtype) if self.weight is not None else None
            self._cast_bias = self.bias.detach().to(dtype) if self.bias is not None else None
            self._cast_dtype = dtype
        return self._cast_weight, self._cast_bias

    def forward(self, x):
        weight, bias = self._cast_params(x.dtype)
        return _group_norm_silu(x, self.num_groups, weight, bias, self.eps, not self.skip_silu)


class FusedResBlock(TimestepBlock):
    """
    Optimized ResBlock with fused operations.
    
    This is a drop-in replacement for the standard ResBlock that fuses:
    - GroupNorm + SiLU in in_layers
    - GroupNorm + SiLU in out_layers
    - Residual addition
    
    Compatible with both openaimodel.ResBlock and model.ResnetBlock.
    Inherits from TimestepBlock for compatibility with TimestepEmbedSequential.
    """
    def __init__(self, original_resblock):
        super().__init__()
        self.original = original_resblock
        
        # Detect which ResBlock type we're wrapping
        if hasattr(original_resblock, 'in_layers'):
            # openaimodel.ResBlock
            self.resblock_type = 'openai'
            self._wrap_openai_resblock()
        elif hasattr(original_resblock, 'norm1'):
            # model.ResnetBlock
            self.resblock_type = 'resnet'
            self._wrap_resnet_block()
        else:
            raise ValueError("Unknown ResBlock type")
    
    def _wrap_openai_resblock(self):
        """Wrap openaimodel.ResBlock"""
        # Replace in_layers: GroupNorm + SiLU + Conv
        # Extract the normalization layer
        norm_layer = self.original.in_layers[0]
        if isinstance(norm_layer, nn.GroupNorm):
            self.fused_in_norm_silu = FusedGroupNormSiLU(
                norm_layer.num_groups, norm_layer.num_channels, norm_layer.eps, norm_layer.affine
            ).to(norm_layer.weight.device)
            self.fused_in_norm_silu.weight.data.copy_(norm_layer.weight.data)
            self.fused_in_norm_silu.bias.data.copy_(norm_layer.bias.data)
            # Keep the conv layer
            self.in_conv = self.original.in_layers[-1]

        # Replace out_layers: GroupNorm + SiLU + Dropout + Conv
        out_norm_layer = self.original.out_layers[0]
        if isinstance(out_norm_layer, nn.GroupNorm):
            self.fused_out_norm_silu = FusedGroupNormSiLU(
                out_norm_layer.num_groups, out_norm_layer.num_channels, out_norm_layer.eps, out_norm_layer.affine
            ).to(out_norm_layer.weight.device)
            self.fused_out_norm_silu.weight.data.copy_(out_norm_layer.weight.data)
            self.fused_out_norm_silu.bias.data.copy_(out_norm_layer.bias.data)
            # Keep dropout and conv
            self.out_dropout = self.original.out_layers[2]
            self.out_conv = self.original.out_layers[3]
        
        # Keep other components
        self.emb_layers = self.original.emb_layers
        self.skip_connection = self.original.skip_connection
        self.h_upd = self.original.h_upd if hasattr(self.original, 'h_upd') else nn.Identity()
        self.x_upd = self.original.x_upd if hasattr(self.original, 'x_upd') else nn.Identity()
        self.updown = self.original.updown if hasattr(self.original, 'updown') else False
        self.use_scale_shift_norm = self.original.use_scale_shift_norm
    
    def _wrap_resnet_block(self):
        """Wrap model.ResnetBlock"""
        # Replace norm1 + nonlinearity (SiLU)
        norm1 = self.original.norm1
        if isinstance(norm1, nn.GroupNorm):
            self.fused_norm1_silu = FusedGroupNormSiLU(
                norm1.num_groups, norm1.num_channels, norm1.eps, norm1.affine
            ).to(norm1.weight.device)
            self.fused_norm1_silu.weight.data.copy_(norm1.weight.data)
            self.fused_norm1_silu.bias.data.copy_(norm1.bias.data)

        # Replace norm2 + nonlinearity (SiLU)
        norm2 = self.original.norm2
        if isinstance(norm2, nn.GroupNorm):
            self.fused_norm2_silu = FusedGroupNormSiLU(
                norm2.num_groups, norm2.num_channels, norm2.eps, norm2.affine
            ).to(norm2.weight.device)
            self.fused_norm2_silu.weight.data.copy_(norm2.weight.data)
            self.fused_norm2_silu.bias.data.copy_(norm2.bias.data)
        
        # Keep convolutions and other components
        self.conv1 = self.original.conv1
        self.conv2 = self.original.conv2
        self.dropout = self.original.dropout
        
        # Time embedding projection
        self.temb_proj = self.original.temb_proj if hasattr(self.original, 'temb_proj') else None
        
        # Skip connection
        if hasattr(self.original, 'conv_shortcut'):
            self.skip_connection = self.original.conv_shortcut
        elif hasattr(self.original, 'nin_shortcut'):
            self.skip_connection = self.original.nin_shortcut
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, emb=None, split=0):
        """Forward pass with fused operations"""
        if self.resblock_type == 'openai':
            return self._forward_openai(x, emb, split)
        else:
            return self._forward_resnet(x, emb)
    
    def _forward_openai(self, x, emb, split=0):
        """Fused forward for openaimodel.ResBlock"""
        # Input path: fused GroupNorm + SiLU, then Conv
        if self.updown:
            h = self.fused_in_norm_silu(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_conv(h)
        else:
            # K1->GN fusion: GroupNorm+SiLU emits the conv's quantized input
            # directly (int8, or packed int4), so in_conv skips its own quantize.
            fused = _prequant_gn_conv(x, self.fused_in_norm_silu, self.in_conv)
            if fused is not None:
                h = fused
            else:
                h = self.fused_in_norm_silu(x)
                h = self.in_conv(h)

        # Time embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        # Skip/residual source (computed with the live x, after any updown resize).
        # When split==0 we hand it to the out-conv so the skip-add is fused into the
        # conv's store epilogue; the split>0 path can't fuse and adds it at the end.
        if split > 0:
            skip = self.skip_connection(x, split=split)
            residual_arg = None
        else:
            skip = self.skip_connection(x)
            residual_arg = skip
        residual_fused = False

        # int8-conv-output quality probe: round-trip in_conv's output through int8
        # before the out-norm GN consumes it (h feeds only the out GN; skip=x).
        if _CONV_INT8_OUT:
            h = _fake_quant_int8_out(h)

        # Output path: fused GroupNorm + SiLU (+ scale-shift modulation) + skip-add,
        # then Conv. out_dropout is nn.Dropout (no-op in eval), elided on the fused path.
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            fused = _prequant_gn_conv(h, self.fused_out_norm_silu, self.out_conv,
                                      mod_scale=scale, mod_shift=shift, residual=residual_arg)
            if fused is not None:
                h = fused
                residual_fused = residual_arg is not None
            else:
                # Not eligible for the GN->intX fusion (e.g. the MoDiff modulated
                # path, whose delta-quantize needs the fp16 activation). Still fold
                # the scale-shift modulation `normed*(1+scale)+shift` into the GN
                # kernel (fp16 output) so it isn't a separate elementwise mul+add.
                weight, bias = self.fused_out_norm_silu._cast_params(h.dtype)
                apply_silu_here = not getattr(self.out_conv, 'fuse_input_silu', False)
                h = _group_norm_silu(h, self.fused_out_norm_silu.num_groups,
                                     weight, bias, self.fused_out_norm_silu.eps,
                                     apply_silu=apply_silu_here,
                                     mod_scale=scale, mod_shift=shift)
                # If out_conv applies SiLU itself (fuse_input_silu, e.g. MoDiff's
                # step1_silu kernel), apply_silu_here is False and SiLU is deferred.
                h = self.out_dropout(h)
                # MoDiff: fold the skip-add into the o_hat conv's accumulate epilogue
                # (no trailing aten::add) when eligible; else plain conv + skip later.
                h, residual_fused = _modiff_out_conv(self.out_conv, h, residual_arg)
        else:
            h = h + emb_out
            fused = _prequant_gn_conv(h, self.fused_out_norm_silu, self.out_conv,
                                      residual=residual_arg)
            if fused is not None:
                h = fused
                residual_fused = residual_arg is not None
            else:
                h = self.fused_out_norm_silu(h)
                h = self.out_dropout(h)
                h, residual_fused = _modiff_out_conv(self.out_conv, h, residual_arg)

        # Residual addition (skipped when already fused into the out-conv epilogue)
        if residual_fused:
            return h
        return torch.add(skip, h)
    
    def _forward_resnet(self, x, temb):
        """Fused forward for model.ResnetBlock"""
        # Input path
        h = self.fused_norm1_silu(x)
        h = self.conv1(h)
        
        # Time embedding modulation
        if temb is not None and self.temb_proj is not None:
            # Apply SiLU to temb and project
            temb_emb = F.silu(temb)
            h = h + self.temb_proj(temb_emb)[:, :, None, None]
        
        # Output path
        h = self.fused_norm2_silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection + residual addition (fused)
        skip = self.skip_connection(x) if not isinstance(self.skip_connection, nn.Identity) else x
        return torch.add(skip, h)


def fuse_resblocks_in_module(module, inplace=True):
    """
    Recursively replace all ResBlock instances with FusedResBlock.
    
    Args:
        module: PyTorch module to process
        inplace: If True, modify module in place. Otherwise, return a copy.
    
    Returns:
        Modified module with fused ResBlocks
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)
    
    fused_count = 0
    
    for name, child in list(module.named_children()):
        # Check if this is a ResBlock
        child_class_name = child.__class__.__name__
        if child_class_name in ['ResBlock', 'ResnetBlock']:
            # Replace with fused version
            fused_block = FusedResBlock(child)
            setattr(module, name, fused_block)
            fused_count += 1
            print(f"✓ Fused ResBlock: {name} ({child_class_name})")
        else:
            # Recursively process children
            child_fused = fuse_resblocks_in_module(child, inplace=True)
            fused_count += sum(1 for m in child_fused.modules() if isinstance(m, FusedResBlock))
    
    return module


def wire_silu_fusion(module):
    """
    Wire SiLU fusion between each FusedResBlock's GroupNorm and its quantized
    conv, for ResBlocks whose in_conv/out_conv have already been converted to
    OptimizedInt8Conv2d/OptimizedInt4Conv2d (call this AFTER
    convert_model_to_optimized_int8/int4, once per model setup).

    For each eligible conv, sets `fuse_input_silu=True` so the conv applies
    SiLU itself -- fused into its quantize kernel on the calibrated hot path,
    or via a plain F.silu(x) call otherwise (see OptimizedInt8Conv2d.forward)
    -- and pairs it with `fused_in_norm_silu.skip_silu=True` so the preceding
    GroupNorm no longer also applies SiLU (avoiding a double activation).

    `out_conv` is always eligible: nothing spatial sits between its GroupNorm
    and the conv (out_dropout is `nn.Dropout`, a no-op in eval/inference
    mode). `in_conv` is only eligible when `updown=False`: for `updown=True`
    ResBlocks, `h_upd` (a spatial resize) runs between GroupNorm and in_conv,
    and SiLU does not commute with resizing -- deferring SiLU into in_conv
    there would apply it after the resize instead of before, changing the
    computation, not just how it's scheduled. Returns the number of conv
    layers wired.
    """
    try:
        from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    except ImportError:
        OptimizedInt8Conv2d = None
    try:
        from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    except ImportError:
        OptimizedInt4Conv2d = None

    fusable_types = tuple(t for t in (OptimizedInt8Conv2d, OptimizedInt4Conv2d) if t is not None)
    if not fusable_types:
        return 0

    wired = 0
    for m in module.modules():
        if not isinstance(m, FusedResBlock) or m.resblock_type != 'openai':
            continue
        if (not m.updown and hasattr(m, 'in_conv') and hasattr(m, 'fused_in_norm_silu')
                and isinstance(m.in_conv, fusable_types)):
            m.in_conv.fuse_input_silu = True
            m.fused_in_norm_silu.skip_silu = True
            wired += 1
        if hasattr(m, 'out_conv') and isinstance(m.out_conv, fusable_types):
            m.out_conv.fuse_input_silu = True
            if hasattr(m, 'fused_out_norm_silu'):
                m.fused_out_norm_silu.skip_silu = True
            wired += 1
    return wired


class FusedUpsample(nn.Module):
    """Wraps an ldm `Upsample` module, fusing its `F.interpolate(nearest, 2x)` into the
    following conv's quantize prologue (`upsample2x_quantize_noahat_fprop` /
    `_pack_noahat_fprop`, csrc/kernels/quantize/modiff_delta_quantize.cu) when eligible --
    reads the SMALL pre-upsample tensor once and writes the quantized LARGE tensor
    directly, so the fp16 upsampled intermediate `F.interpolate` would materialize (and
    the plain quantize kernel would then re-read) is never allocated. Bit-identical to
    the unfused path (verified: F.interpolate -> step1_static_quantize[_pack]_noahat_fprop
    vs upsample2x_quantize[_pack]_noahat_fprop, incl. non-identity SmoothQuant, since a
    per-channel scale commutes with nearest-neighbor spatial replication).

    Baseline-only (mirrors _prequant_common_ok): the MoDiff modulated path's SmoothQuant
    and delta-quantize semantics aren't replicated here, and this only ever engages for
    `Upsample.conv`, which is never inside a ResBlock skip (no residual to fold in).
    Falls back to the original two-step `self.orig(x)` otherwise (fp32/uncalibrated/
    modiff-enabled/grouped conv/no conv/dims!=2).
    """
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def _fusable(self, x, conv):
        if conv is None or self.orig.dims != 2 or x.dtype != torch.float16 or not x.is_cuda:
            return False
        return (getattr(conv, 'is_calibrated', False)
                and not getattr(conv, 'modiff_enabled', True)
                and getattr(conv, 'use_cutlass', False)
                and getattr(conv, 'groups', 1) == 1)

    def forward(self, x):
        assert x.shape[1] == self.orig.channels
        conv = getattr(self.orig, 'conv', None) if self.orig.use_conv else None
        if not self._fusable(x, conv):
            return self.orig(x)
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        conv._ensure_conv_caches(x.device)
        smooth_inv = (conv._empty_smooth if getattr(conv, '_smooth_is_identity', True)
                      else conv._smooth_inv.view(-1).to(torch.float32).contiguous())
        is_int4 = hasattr(conv, 'weight_packed')
        if is_int4:
            x_q = modiff_cutlass.upsample2x_quantize_pack_noahat_fprop(
                x, conv.static_input_scale.view(1), smooth_inv)
            return conv._conv_from_int4(x_q, x.shape[2] * 2, x.shape[3] * 2)
        x_q = modiff_cutlass.upsample2x_quantize_noahat_fprop(
            x, conv.static_input_scale.view(1), smooth_inv)
        return conv._conv_from_int8(x_q)


def convert_upsample_to_fused(module):
    """Recursively replace ldm `Upsample` instances with `FusedUpsample` (call this AFTER
    convert_model_to_optimized_int8/int4, once per model setup, same convention as
    wire_silu_fusion). Returns the number of Upsample modules wrapped."""
    try:
        from ldm.modules.diffusionmodules.openaimodel import Upsample
    except ImportError:
        return 0
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Upsample) and not isinstance(child, FusedUpsample):
            setattr(module, name, FusedUpsample(child))
            n += 1
        else:
            n += convert_upsample_to_fused(child)
    return n


def print_fusion_summary(module):
    """Print summary of fused blocks in the module"""
    fused_blocks = [m for m in module.modules() if isinstance(m, FusedResBlock)]
    openai_blocks = sum(1 for m in fused_blocks if m.resblock_type == 'openai')
    resnet_blocks = sum(1 for m in fused_blocks if m.resblock_type == 'resnet')
    
    print(f"\n{'='*60}")
    print(f"ResBlock Fusion Summary")
    print(f"{'='*60}")
    print(f"Total fused blocks: {len(fused_blocks)}")
    print(f"  - OpenAI ResBlocks: {openai_blocks}")
    print(f"  - ResNet Blocks: {resnet_blocks}")
    print(f"Fusion method: native channels_last GroupNorm(+SiLU) CUDA kernel, "
          f"falling back to F.group_norm + F.silu with autocast locally disabled "
          f"(HAS_NATIVE_GN_SILU={HAS_NATIVE_GN_SILU})")
    print(f"{'='*60}\n")
