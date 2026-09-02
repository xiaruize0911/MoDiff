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
# Fake-quant the in_conv fp16 output through int8 before the out-norm GN reads it, to
# measure the quality ceiling of "conv writes int8 -> GN reads int8". pt = per-tensor
# scale, pc = per-output-channel scale. Off unless the env var is set.
#
# STATUS: this probe already returned its answer -- quality is fine (+0.0023..0.0033
# rel-err, far inside the 0.02 gate), and the GN half of the fusion was then built and
# verified (group_norm_silu_dequant_quantize_nhwc in csrc/kernels/norm/group_norm_silu.cu,
# see its STATUS comment). The blocker is the CONV half: it needs a direct-int8-output
# CUTLASS epilogue, without which the handoff moves more bytes than it saves. Kept because
# it is the measurement harness for that remaining work, not leftover debugging.
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
    HAS_GN_SILU_QUANTIZE_PACK_ZP = hasattr(modiff_cutlass,
                                           "group_norm_silu_quantize_pack_nhwc_zp")
    # GN FAST-REDUCE (default ON, 2026-08-16). `..._fast` is the SAME entry point with
    # fast_reduce=true: 128-512 threads and a pair-major pass 1 instead of the generic heuristic's
    # up-to-1024. Identical signature, identical math, different reduction ORDER.
    #
    # The attention paths have taken it since it was written; this file did not, and this file owns the
    # bulk of the GN time -- so the family sat at 10-65% of the A40's 696 GB/s while the fix was in the
    # tree. Measured over the captured ResBlock shapes at their real call counts
    # (docs/gn_fast_reduce_2026-08-16): int8 14.51 -> 7.60 ms/step (1.91x), int4 14.93 -> 7.37 (2.03x).
    #
    # Numerics: a different fp32 reduction order moves the mean/inv_std in the last bits, so a value
    # sitting exactly on a code boundary can land either side. Measured at <=1 code on <=1.6e-5% of
    # elements across all 16 shapes, both precisions, with production modulation. Not bit-identical,
    # and the flag exists so that is falsifiable rather than asserted.
    # Read at CALL time, not import time, so an in-process A/B can flip it between arms -- the same
    # reason _updown_fuse_refresh() below is a function and not a constant.
    def _gnq(name):
        """The fast entry point when it exists and is enabled, else the one this file always used."""
        if os.environ.get("MODIFF_GN_FAST", "1") == "1":
            f = getattr(modiff_cutlass, name + "_fast", None)
            if f is not None:
                return f
        return getattr(modiff_cutlass, name)
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
    # Probe the EVT symbol, which is what the code actually calls. It used to probe
    # conv2d_int8_fprop_o_hat_residual -- the superseded fp32-round-trip version -- so deleting
    # that dead symbol would have silently turned this fusion OFF rather than failing loudly.
    HAS_O_HAT_RESIDUAL = hasattr(modiff_cutlass, "conv2d_int8_evt_o_hat_residual")
    # Upsample(nearest,2x)+quantize fusion (default ON): fold a ResBlock's updown
    # h_upd (Upsample, use_conv=False) into the following in_conv's quantize step, the
    # same upsample2x_quantize[_pack]_noahat kernel FusedUpsample already uses for
    # standalone Upsample(use_conv=True) modules -- see FusedUpsample.
    # Disable with MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION=1.
    HAS_GN_SILU_DELTA_QUANTIZE_RESIZE = hasattr(
        modiff_cutlass, "group_norm_silu_delta_quantize_resize_nhwc")
    HAS_GN_SILU_QUANTIZE_RESIZE = hasattr(
        modiff_cutlass, "group_norm_silu_quantize_resize_nhwc")
    #: Activation-zero-point twins (fix #2), added 2026-08-13. Separate flags rather than assuming a
    #: rebuild: an older .so with the symmetric entry points must keep working, and a non-zero zero
    #: point must then REFUSE rather than quantize symmetrically against a corrected bias.
    HAS_GN_SILU_QUANTIZE_RESIZE_ZP = hasattr(
        modiff_cutlass, "group_norm_silu_quantize_resize_nhwc_zp")
    HAS_UPSAMPLE_QUANTIZE_PACK_ZP = hasattr(
        modiff_cutlass, "upsample2x_quantize_pack_noahat_fprop_zp")
    HAS_UPSAMPLE_QUANTIZE = hasattr(modiff_cutlass, "upsample2x_quantize_noahat_fprop")
    HAS_UPSAMPLE_QUANTIZE_PACK = hasattr(modiff_cutlass, "upsample2x_quantize_pack_noahat_fprop")
    # Downsample(avg_pool,2x2)+quantize fusion (default ON): the down-direction sibling of
    # the upsample fusion above. avg_pool cannot commute with quantize (averaging codes !=
    # averaging then quantizing), so this does NOT reorder anything -- it fuses avg_pool's
    # own arithmetic with the following in_conv's quantize step into one kernel, computing
    # each 2x2 window's fp16-rounded average (bit-exact to nn.AvgPool2d's actual output,
    # see avgpool4_as_stored in modiff_delta_quantize.cu) and quantizing it directly,
    # without ever materializing the fp16 pooled intermediate (FusedUpsample's path).
    # Disable with MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION=1.
    HAS_AVGPOOL_QUANTIZE = hasattr(modiff_cutlass, "avgpool2x_quantize_noahat_fprop")
    HAS_AVGPOOL_QUANTIZE_PACK = hasattr(modiff_cutlass, "avgpool2x_quantize_pack_noahat_fprop")
except ImportError:
    modiff_cutlass = None

    def _gnq(name):                     # every native path is gated on native_ok/HAS_*, so unreachable
        raise RuntimeError("modiff_cutlass is not importable")
    HAS_GN_SILU_DELTA_QUANTIZE_RESIZE = False
    HAS_GN_SILU_QUANTIZE_RESIZE = False
    HAS_NATIVE_GN_SILU = False
    HAS_GN_SILU_QUANTIZE = False
    HAS_GN_SILU_QUANTIZE_PACK = False
    HAS_GN_SILU_QUANTIZE_PACK_ZP = False
    HAS_GN_SILU_QUANTIZE_RESIZE_ZP = False
    HAS_UPSAMPLE_QUANTIZE_PACK_ZP = False
    HAS_GN_SILU_DELTA_QUANTIZE = False
    HAS_GN_SILU_DELTA_QUANTIZE_PACK = False
    HAS_O_HAT_RESIDUAL = False
    HAS_UPSAMPLE_QUANTIZE = False
    HAS_UPSAMPLE_QUANTIZE_PACK = False
    HAS_AVGPOOL_QUANTIZE = False
    HAS_AVGPOOL_QUANTIZE_PACK = False

if os.environ.get("MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION") == "1":
    HAS_UPSAMPLE_QUANTIZE = False
    HAS_UPSAMPLE_QUANTIZE_PACK = False
    HAS_UPSAMPLE_QUANTIZE_PACK_ZP = False

if os.environ.get("MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION") == "1":
    HAS_AVGPOOL_QUANTIZE = False
    HAS_AVGPOOL_QUANTIZE_PACK = False

# Kill-switch for the GN->intX K1-fusion (baseline int8/int4). Set
# MODIFF_DISABLE_GN_INT8_FUSION=1 to fall back to the exact two-kernel
# (GroupNorm+SiLU, then standalone quantize[/pack]) path -- used for A/B
# benchmarking and as a production safety switch. Pure optimization either way.
if os.environ.get("MODIFF_DISABLE_GN_INT8_FUSION") == "1":
    HAS_GN_SILU_QUANTIZE = False
    HAS_GN_SILU_QUANTIZE_PACK = False
    HAS_GN_SILU_QUANTIZE_PACK_ZP = False

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


#: Layers observed taking a quantize path that does NOT apply the activation zero point, while their
#: bias carries the -z*sum(w_q) correction. That combination is strictly wrong -- symmetric codes
#: against a corrected bias -- and it is how the first end-to-end zero-point run produced relL2 7-22
#: instead of a result. Recorded (and, under MODIFF_ZP_STRICT=1, raised on) so the gap is visible
#: rather than silently numeric. The plan scoped fix #2 at ~6 quantize entry points; as of 2026-08-13
#: exactly one of them (group_norm_silu_quantize_pack_nhwc) honours z.
ZP_UNSUPPORTED_HITS = set()


def _zp_unsupported(conv, where, grid="activation"):
    """Record/raise when a conv with a non-zero zero point is quantized by a path that ignores it.

    `grid` names what the SITE quantizes -- see OptimizedInt4Conv2d._zp_unsupported for the full
    contract. "delta" sites (x - a_hat, feeding a bias-free o_hat accumulate) are z-free by
    construction and exempt; classifying them as gaps is what made the first census count 70 where
    the measured number is 8 (docs/zp_coverage_2026-08-13/data/site_census.json).
    """
    # Host mirror, never .item() -- see OptimizedInt4Conv2d._zp_float. This function is called on
    # every quantize, so a device sync here cost ~2 ms/step before it was removed.
    if getattr(conv, "_zp_float", 0.0) == 0.0:
        return
    if grid == "delta":
        # Verify the declaration rather than trust it; only reachable in an asymmetric config.
        if getattr(conv, "a_hat_cache", None) is None:
            raise RuntimeError(
                f"{getattr(conv, 'layer_name', None)}: {where} declared grid='delta' but the conv "
                f"has no a_hat cache, so it is not quantizing a delta")
        return
    name = getattr(conv, 'layer_name', None) or where
    ZP_UNSUPPORTED_HITS.add(name)
    if os.environ.get("MODIFF_ZP_STRICT", "1") == "1":
        raise RuntimeError(
            f"{name}: activation zero point {getattr(conv, '_zp_float', 0.0):+.0f} is set, but this layer is "
            f"quantized by {where}, which does not apply it -- while the bias already carries the "
            f"matching -z*sum(w_q) correction. Set the zero point to 0 for this layer or teach "
            f"{where} the zero point.")


def _skip_concat_fallback(a, b):
    """Materialize the decoder skip concatenation when the fold declines.

    Reuses openaimodel._skip_concat so there is ONE definition of what this concatenation is -- it
    already falls back from cat2_channels_last_fp16 to torch.cat on odd channel counts, fp32 and CPU.
    Imported lazily to keep integration/ from importing ldm/ at module scope.
    """
    try:
        from ldm.modules.diffusionmodules.openaimodel import _skip_concat
        return _skip_concat(a, b)
    except Exception:
        return torch.cat([a, b], dim=1)


def _prequant_gn_conv_modiff(x, gn, conv, mod_scale=None, mod_shift=None, residual=None,
                             x2=None):
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
    # THE DECODER SKIP-CONCAT FOLD. `x2` means x is only the first half; every eligibility check
    # below is about the tensor the GroupNorm will actually see, which is the CONCATENATION -- so a
    # probe of that shape is built and checked, rather than checking the half and hoping. Only the
    # int4 fold kernel exists (cat2_gn_stats_fp16 is fp16/int4-only), and it requires C1 % 32 == 0
    # so no warp straddles the two buffers; both are verified here so an ineligible shape falls back
    # to the ordinary concat path instead of failing inside CUDA.
    if x2 is not None:
        if (not HAS_GN_SILU_DELTA_QUANTIZE_PACK or not hasattr(conv, 'forward_from_int4')
                or x.dtype != torch.float16 or x2.dtype != torch.float16
                or int(x.shape[1]) % 32 != 0
                or not x.is_contiguous(memory_format=torch.channels_last)
                or not x2.is_contiguous(memory_format=torch.channels_last)):
            return None
        # SHAPE-ONLY eligibility. The first version built a probe tensor of the concatenated shape
        # and passed it to can_gn_fuse_modiff -- allocating, per block per step, exactly the tensor
        # this optimisation exists to stop touching. It also crashed: new_empty() takes no
        # memory_format on this torch. can_gn_fuse_modiff_cat2 decides from shapes and flags alone.
        if not hasattr(conv, 'can_gn_fuse_modiff_cat2') or not conv.can_gn_fuse_modiff_cat2(x, x2):
            return None
        cat_shape = (x.shape[0], x.shape[1] + x2.shape[1], x.shape[2], x.shape[3])
        x_eff = None            # no concatenated tensor exists yet; the kernel will emit it
    else:
        if not hasattr(conv, 'can_gn_fuse_modiff') or not conv.can_gn_fuse_modiff(x):
            return None
        x_eff = x
    is_int4 = hasattr(conv, 'forward_from_int4')
    if is_int4 and not HAS_GN_SILU_DELTA_QUANTIZE_PACK:
        return None
    if not is_int4 and not HAS_GN_SILU_DELTA_QUANTIZE:
        return None

    ng = gn.num_groups
    if x2 is not None:
        # The halves' own layout/dtype were checked above; what remains is the CONCATENATION's
        # divisibility, which is a property of the shape tuple and needs no tensor.
        N, C = cat_shape[0], cat_shape[1]
        if C % ng != 0:
            return None
        ref = x
    else:
        N, C = x_eff.size(0), x_eff.size(1)
        # GN-native eligibility (same conditions as _group_norm_silu's can_use_native).
        if not (x_eff.is_cuda and x_eff.dtype in (torch.float32, torch.float16)
                and x_eff.is_contiguous(memory_format=torch.channels_last) and C % ng == 0):
            return None
        ref = x_eff
    if is_int4 and (C % 2 != 0 or (C // ng) % 2 != 0):
        return None

    weight, bias = gn._cast_params(ref.dtype)
    if mod_scale is not None:
        ms2d = mod_scale.reshape(N, C).contiguous()
        sh2d = mod_shift.reshape(N, C).contiguous()
    else:
        ms2d = sh2d = ref.new_empty(0)
    if residual is not None:
        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
    if x2 is not None:
        # Returns (out, cat): the fold emits the concatenation, which the caller still needs for the
        # skip conv and the out-conv residual.
        return conv.forward_gn_fused_modiff(x, weight, bias, ng, gn.eps, ms2d, sh2d, residual,
                                            x2=x2)
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


def _prequant_gn_conv(x, gn, conv, mod_scale=None, mod_shift=None, residual=None, x2=None):
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
    # MODIFF_CONV_BLOCKK=32: fuse GN(+mod)(+SiLU) into a BLOCKWISE-along-C quantize and run
    # conv2d_int8_blockk. Injected before the per-tensor folds below because it replaces both of
    # them; returns None when the layer or the env is not eligible, and then nothing changes.
    # HAS_GN_SILU_* gate this too. Without it the hook fires above the kill-switch checks below,
    # so a run asking for the UNFUSED or scalar-CTRL arm silently got the fused blockwise path --
    # which made "unfused" and "fused" the same measurement and made CTRL impossible to measure.
    if (x2 is None and hasattr(conv, 'blockk_gn_fused')
            and not conv._conv_blockk_ctrl()
            and (HAS_GN_SILU_DELTA_QUANTIZE if getattr(conv, 'modiff_enabled', False)
                 else HAS_GN_SILU_QUANTIZE)):
        bk = conv.blockk_gn_fused(x, *gn._cast_params(x.dtype), gn.num_groups, gn.eps, True,
                                  mod_scale, mod_shift, residual)
        if bk is not None:
            return bk

    # MoDiff temporal-cache path: fuse GroupNorm(+mod)+SiLU into the delta-quantize
    # (+ a_hat update) kernel, replacing the standalone GN kernel + step1 two-kernel
    # pass. Bit-identical to that path; only kicks in once the layer is calibrated
    # with an fp16 a_hat cache (step >= 2). See _prequant_gn_conv_modiff.
    modiff_fused = _prequant_gn_conv_modiff(x, gn, conv, mod_scale, mod_shift, residual, x2=x2)
    if modiff_fused is not None:
        return modiff_fused
    # The fold is MoDiff-only. If it did not take, the caller must fall back to materializing the
    # concatenation itself -- signalled by returning None rather than by silently running the
    # non-folded path on `x`, which is only the first half and would be a wrong-shape conv.
    if x2 is not None:
        return None

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
        # ACTIVATION ZERO POINT (fix #2). Routed ONLY when a calibration actually set one, so the
        # symmetric path keeps calling the exact entry point it always did. The _zp kernel is
        # bit-identical at z=0, but not calling it at all is a stronger guarantee than trusting that.
        # The matching -z*sum(w_q) term is already folded into conv.bias by _refold_zp_bias.
        zp = getattr(conv, "_zp_float", 0.0)   # host mirror; see _zp_unsupported
        if native_ok and zp != 0.0:
            if not HAS_GN_SILU_QUANTIZE_PACK_ZP:
                # A non-zero zp with no kernel to honour it would quantize symmetrically while the
                # bias carries a correction for a zero point that was never applied -- worse than
                # either choice alone. Refuse rather than silently pick one.
                raise RuntimeError(
                    "conv has a non-zero activation zero point but modiff_cutlass lacks "
                    "group_norm_silu_quantize_pack_nhwc_zp -- rebuild the extension")
            packed = _gnq("group_norm_silu_quantize_pack_nhwc_zp")(
                x, w, b, ng, eps, True, scale, smooth_inv, ms2d, sh2d, 0, zp)
        elif native_ok:
            packed = _gnq("group_norm_silu_quantize_pack_nhwc")(
                x, w, b, ng, eps, True, scale, smooth_inv, ms2d, sh2d)
        else:
            # The non-native fallback now honours z too (scale_quantize_and_pack_zp, 2026-08-13), so
            # a zero point no longer forces a layer down the native path to stay correct.
            gn = _gn_mod_silu_fp32_cl(x, ng, w, b, eps, mod_scale, mod_shift)
            if zp != 0.0:
                if not hasattr(modiff_cutlass, "scale_quantize_and_pack_zp"):
                    raise RuntimeError(
                        "activation zero point is set but this layer fell back to the non-native "
                        "scale_quantize_and_pack path and modiff_cutlass has no _zp variant -- it "
                        "would quantize symmetrically against a bias corrected for one; rebuild")
                packed = modiff_cutlass.scale_quantize_and_pack_zp(gn, scale, zp)
            else:
                packed = modiff_cutlass.scale_quantize_and_pack(gn, scale)
        return conv.forward_from_int4(packed, h_in, w_in, residual=residual)
    else:
        if native_ok:
            q = _gnq("group_norm_silu_quantize_nhwc")(
                x, w, b, ng, eps, True, scale, smooth_inv, ms2d, sh2d)
        else:
            q = modiff_cutlass.scale_quantize_int8(
                _gn_mod_silu_fp32_cl(x, ng, w, b, eps, mod_scale, mod_shift), scale)
        return conv.forward_from_int8(q, residual=residual)


#: Read at call time, not at import, so an in-process A/B can flip it between arms.
def _updown_fuse_refresh():
    return os.environ.get("MODIFF_UPDOWN_FUSE_REFRESH", "1") == "1"


class _UpdownFuseRefreshFlag:
    """Truthy iff the fusion should cover dynamic refresh steps. A live view of the env var
    rather than a snapshot, so `os.environ[...] = "0"` takes effect on the next call."""
    def __bool__(self):
        return _updown_fuse_refresh()


_UPDOWN_FUSE_REFRESH = _UpdownFuseRefreshFlag()


class _UpdownA4Flag:
    """Truthy iff the updown fusion should honour the conv's activation bit-width. Live view of
    MODIFF_UPDOWN_A4 so an in-process A/B can flip it between arms; see the call site."""
    def __bool__(self):
        return os.environ.get("MODIFF_UPDOWN_A4", "1") == "1"


_UPDOWN_A4 = _UpdownA4Flag()


def _delta_gn_dynamic_args_any(conv, device, is_int4):
    """The 8 trailing dynamic-scale arguments of group_norm_silu_delta_quantize_resize_nhwc,
    from whichever accessor this conv class provides.

    int8 (`_delta_gn_dynamic_args`) already returns all 8. int4 (`_delta_gn_dynamic_args_i4`)
    returns 7 -- it has no bit-width flag to return, because packed-int4 storage saturates at 7 by
    the format itself and its sibling kernel group_norm_silu_delta_quantize_pack_nhwc therefore
    takes no such argument. False is correct here for the same reason: on the PACK path the resize
    kernel derives its limit from PACK, so the flag is redundant rather than wrong, and the int4 arm
    stays bit-identical to that sibling.
    """
    if is_int4:
        return tuple(conv._delta_gn_dynamic_args_i4(device)) + (False,)
    args = tuple(conv._delta_gn_dynamic_args(device))
    if not _UPDOWN_A4:
        # MODIFF_UPDOWN_A4=0 forces these eight layers back to the int8 store's literal 127, which
        # is what they did before 2026-08-10: this kernel took no ceiling, so it clamped at 127
        # while the other 62 convs honoured act_q. It exists as the CONTROL for measuring what that
        # defect was worth -- the before-arm cannot otherwise be reproduced, since the code is gone.
        # Only observable at MODIFF_ACT_BITS=4 with MODIFF_DELTA_REFRESH>1; see
        # docs/updown_refresh_fusion_2026-08-10/.
        args = args[:-1] + (False,)
    return args


def _prequant_gn_resize_conv_modiff(x, gn, h_upd, conv, mod_scale=None, mod_shift=None):
    # MODIFF_CONV_BLOCKK: this fold emits PER-TENSOR codes and accumulates o_hat through
    # _conv_from_int8_o_hat with a scalar alpha. Under blockwise that mixes conventions on the
    # 8 resize in_conv layers -- step 1 writes a_hat/o_hat blockwise, steps 2+ accumulate
    # per-tensor on top -- which measured relL2 0.4951 instead of 0.0641 and raised no error,
    # because _conv_from_int8_o_hat had no guard (it has one now). Disable the fold so those
    # layers take one convention end to end.
    # MODIFF_ACT_BLOCK is here too, not just CONV_BLOCKK. This fold reaches the conv through
    # _conv_from_int8_o_hat with PER-TENSOR codes, so under either the sim or the blockwise path
    # these 8 resize in_conv layers would run per-tensor no matter what granularity the arm asked
    # for. docs/act_budget_2026-09-02 was measured before _conv_from_int8_o_hat had a guard and so
    # had 8 of 70 layers silently excluded from its granularity sweep.
    if (os.environ.get("MODIFF_CONV_BLOCKK", "0") not in ("0", "")
            or os.environ.get("MODIFF_ACT_BLOCK", "0") not in ("0", "")):
        return None
    """MoDiff twin of _prequant_gn_resize_conv: GN+SiLU+resize+delta-quantize+a_hat in ONE kernel.

    The eight updown ResBlocks previously got NO fusion under MoDiff -- the sibling below gates on
    `not modiff`, so MoDiff fell back to a standalone PyTorch resize followed by a separate
    delta-quantize. Measured cost of that fallback at batch 128 (2026-08-04): +1.20 ms/step nearest
    upsample, +0.44 avg_pool, +0.71 GN+SiLU-only, i.e. 2.35 ms/step and the largest remaining
    non-intrinsic MoDiff overhead.

    a_hat stays at the POST-resize (conv input) resolution, the same layout the unfused path uses,
    so this is a pure fusion.

    Returns the conv output, or None to fall through.

    REFRESH STEPS (fixed 2026-08-10). This used to decline whenever the dynamic scale had to be
    re-measured, because the kernel took its scale as a device pointer and computed no absmax. That
    read as "one step in four falls back" -- but MODIFF_DELTA_REFRESH=1 is the paper's own
    configuration, and there EVERY step is a refresh, so the fusion never fired at all and all
    eight updown ResBlocks ran unfused. The 2026-08-07 traces show it plainly: 6/8 of these calls
    fused at K=4 and 0/8 at K=1, with 8 `upsample_nearest2d` + 8 `avg_pool2d` + 8 standalone
    `group_norm_silu_nhwc` launches in their place.

    The kernel now carries the same dynamic-scale contract as the 62 other convs'
    group_norm_silu_delta_quantize_nhwc, so this passes `_delta_gn_dynamic_args` straight through
    and a refresh step stays fused: a reduction-only launch of the same kernel measures the range,
    the quantizing launch consumes it. Two launches, but still one fewer full pass over `x` than
    the unfused route, and no fp16 resized intermediate at all.

    `MODIFF_UPDOWN_FUSE_REFRESH=0` restores the old decline. It exists because the win is small
    enough (+0.46 ms/step over the eight shapes at batch 128) that cross-session drift is the same
    order, so the honest measurement is an A/B in ONE process -- which needs both behaviours
    available at once. It doubles as the revert switch.
    """
    if not (HAS_GN_SILU_DELTA_QUANTIZE_RESIZE and conv is not None):
        return None
    if not getattr(conv, 'modiff_enabled', False) or not getattr(conv, 'is_calibrated', False):
        return None
    if getattr(conv, 'is_first_step', True):
        return None                       # t=T has no cache to subtract yet
    if not getattr(conv, 'use_cutlass', False) or getattr(conv, 'groups', 1) != 1:
        return None
    # Dynamic mode needs the conv's four reduction buffers. _ensure_state_buffers allocates them,
    # but it is keyed on the conv's own input tensor and this fusion never materializes one -- it
    # hands the kernel the PRE-resize activation -- so allocate them here. Idempotent, and the only
    # reason this path used to require that an unfused step had run first.
    if getattr(conv, 'delta_dynamic', False):
        if not _UPDOWN_FUSE_REFRESH and conv._delta_should_refresh():
            return None                   # pre-2026-08-10 behaviour; see the docstring
        if getattr(conv, '_scale_buf', None) is None and not _UPDOWN_FUSE_REFRESH:
            return None                   # the old path relied on an unfused step running first
        conv._ensure_delta_dyn_bufs(x.device)
    try:
        from ldm.modules.diffusionmodules.openaimodel import Upsample, Downsample
    except ImportError:
        return None
    h_upd = getattr(h_upd, 'orig', h_upd)
    if isinstance(h_upd, Upsample):
        direction = 1
    elif isinstance(h_upd, Downsample):
        direction = -1
    else:
        return None
    if h_upd.use_conv or h_upd.dims != 2:
        return None
    if not (x.is_cuda and x.dtype in (torch.float32, torch.float16)):
        return None
    ng, eps = gn.num_groups, gn.eps
    N, C, H, W = x.shape
    if C % ng != 0 or C % 2 != 0 or (C // ng) % 2 != 0:
        return None
    if direction < 0 and (H % 2 or W % 2):
        return None
    Ho, Wo = (H * 2, W * 2) if direction > 0 else (H // 2, W // 2)
    is_int4 = hasattr(conv, 'forward_from_int4')
    if not is_int4 and not hasattr(conv, '_conv_from_int8'):
        return None
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)

    # The conv owns the a_hat cache and sizes it to ITS input, which is the post-resize shape.
    # Bail rather than reshape if it is not there yet -- a wrong-shaped cache would silently
    # corrupt o_hat for every remaining timestep.
    ah = getattr(conv, 'a_hat_cache', None)
    if (ah is None or ah.dtype not in (torch.float16, torch.int8)
            or ah.numel() != N * C * Ho * Wo):
        return None

    conv.step_count += 1
    w, b = gn._cast_params(x.dtype)
    d_scale, d_alpha = (conv._delta_scale_args_i4(x.device) if is_int4
                        else conv._delta_scale_args(x.device))
    # ORDER MATTERS, and it is the same order forward_gn_fused_modiff uses: read the pair currently
    # IN FORCE first, because _delta_gn_dynamic_args flips it on a reporting step. On a separate-pass
    # refresh step the pair in force IS what the reduction launch writes into, so the conv's alpha
    # (d_alpha) picks up the freshly measured value; on a reporting step it is the previous window's,
    # which is exactly what that mode quantizes with.
    dyn = _delta_gn_dynamic_args_any(conv, x.device, is_int4)
    if getattr(conv, 'delta_dynamic', False):
        cur = conv._cur_scale_pair() if hasattr(conv, '_cur_scale_pair') else (
            conv._scale_buf, conv._inv_scale_buf)
        d_scale, d_alpha = cur[0].view(1), cur[1].view(1)
    # _empty_smooth is lazily created, so it may still be None here -- build the empty sentinel
    # rather than passing None into the kernel.
    if getattr(conv, '_smooth_is_identity', True):
        smooth_inv = torch.empty(0, device=x.device, dtype=torch.float32)
    else:
        smooth_inv = conv._smooth_inv.view(-1).to(torch.float32).contiguous()
    if mod_scale is not None:
        ms2d = mod_scale.reshape(N, C).contiguous()
        sh2d = mod_shift.reshape(N, C).contiguous()
    else:
        ms2d = sh2d = x.new_empty(0)

    _zp_unsupported(conv, "group_norm_silu_delta_quantize_resize_nhwc", grid="delta")
    write_ahat = not conv._skip_cache_store()
    conv._begin_ahat_kernel(write_ahat) if hasattr(conv, '_begin_ahat_kernel') else None
    ahat_scale = conv._ahat_scale_arg() if hasattr(conv, '_ahat_scale_arg') else torch.empty(
        0, device=x.device, dtype=torch.float32)
    x_q = modiff_cutlass.group_norm_silu_delta_quantize_resize_nhwc(
        x, w, b, ng, eps, True, d_scale, smooth_inv, ms2d, sh2d, 0, direction, is_int4, ah, *dyn,
        write_ahat, ahat_scale)
    if conv._delta_calib:
        conv._observe_delta_absmax() if is_int4 else conv._observe_delta_codes(x_q)
    out = (conv._conv_from_int4_o_hat(x_q, Ho, Wo, d_alpha) if is_int4
           else conv._conv_from_int8_o_hat(x_q, d_alpha))
    return conv._after_ahat_write(out)


def _prequant_gn_resize_conv(x, gn, h_upd, conv, mod_scale=None, mod_shift=None):
    """updown ResBlock: GroupNorm+SiLU+quantize+2x resize in ONE kernel, then the conv.

    Replaces the two-kernel pair this path used to run -- `group_norm_silu_nhwc` (fp16 out)
    followed by `{upsample2x,avgpool2x}_quantize_[pack_]noahat_fprop`. GroupNorm emitted fp16
    there because, split across kernels, the DOWN direction is only exact with the quantize on
    the far side of the resize: averaging four already-quantized codes rounds each input before
    averaging, which is not the same as averaging and rounding once. Inside a single kernel the
    2x2 average is taken on the fp32 post-SiLU values, so both directions fuse and the result is
    closer to fp32 than the old pair was -- measured rel-L2 vs an fp32 reference falls from
    ~0.006-0.007 to <=0.0003 (integration/tests/test_gn_resize_fusion.py), and the pair of
    kernels becomes one at 1.45-5.6x, median ~2.9x, on the eight real updown shapes.

    Returns the conv output, or None to fall through to `h_upd(h)` -> `conv(h)`.
    """
    if not HAS_GN_SILU_QUANTIZE_RESIZE:
        return None
    try:
        from ldm.modules.diffusionmodules.openaimodel import Upsample, Downsample
    except ImportError:
        return None
    h_upd = getattr(h_upd, 'orig', h_upd)   # unwrap FusedUpsample if that pass ran
    if isinstance(h_upd, Upsample):
        direction = 1
    elif isinstance(h_upd, Downsample):
        direction = -1
    else:
        return None
    if h_upd.use_conv or h_upd.dims != 2:
        return None
    if not _prequant_common_ok(conv):
        return None
    if not (x.is_cuda and x.dtype in (torch.float32, torch.float16)):
        return None
    ng, eps = gn.num_groups, gn.eps
    N, C, H, W = x.shape
    if C % ng != 0:
        return None
    if direction < 0 and (H % 2 or W % 2):
        return None
    is_int4 = hasattr(conv, 'forward_from_int4')
    if C % 2 != 0 or (C // ng) % 2 != 0:      # both variants store channel PAIRS
        return None
    if not is_int4 and not hasattr(conv, '_conv_from_int8'):
        return None
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)

    w, b = gn._cast_params(x.dtype)
    conv._ensure_conv_caches(x.device)
    scale = conv._cached_scale_tensor
    smooth_inv = (conv._empty_smooth if getattr(conv, '_smooth_is_identity', True)
                  else conv._smooth_inv.view(-1).to(torch.float32).contiguous())
    if mod_scale is not None:
        ms2d = mod_scale.reshape(N, C).contiguous()
        sh2d = mod_shift.reshape(N, C).contiguous()
    else:
        ms2d = sh2d = x.new_empty(0)

    # ACTIVATION ZERO POINT (fix #2). PTQ updown ResBlocks: the activation grid feeding a conv that
    # adds the corrected bias -- 8 of the 8 real gaps the 2026-08-13 census found on the PTQ arm
    # (docs/zp_coverage_2026-08-13/data/site_census.json). Routed only when a calibration actually
    # set a zero point, so the symmetric path keeps calling the exact entry point it always did.
    zp = getattr(conv, "_zp_float", 0.0)
    if zp != 0.0 and is_int4:
        if not HAS_GN_SILU_QUANTIZE_RESIZE_ZP:
            raise RuntimeError(
                "conv has a non-zero activation zero point but modiff_cutlass lacks "
                "group_norm_silu_quantize_resize_nhwc_zp -- rebuild the extension")
        x_q = modiff_cutlass.group_norm_silu_quantize_resize_nhwc_zp(
            x, w, b, ng, eps, True, scale, smooth_inv, ms2d, sh2d, 0, direction, True, zp)
    else:
        # int8 has no bias correction for an int4 zero point; the guard names that rather than
        # letting the kernel's own TORCH_CHECK surface as an opaque failure deep in a sample.
        _zp_unsupported(conv, "group_norm_silu_quantize_resize_nhwc")
        x_q = modiff_cutlass.group_norm_silu_quantize_resize_nhwc(
            x, w, b, ng, eps, True, scale, smooth_inv, ms2d, sh2d, 0, direction, is_int4)
    Ho, Wo = (H * 2, W * 2) if direction > 0 else (H // 2, W // 2)
    return conv._conv_from_int4(x_q, Ho, Wo) if is_int4 else conv._conv_from_int8(x_q)


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
        # The decoder skip-concat fold hands a (h, skip) TUPLE, and only _forward_openai understands
        # it. This UNet's output_blocks are all openai-type so the resnet path never sees one today --
        # which is exactly why it would be a silent trap for the next model family wired through
        # FusedResBlock. Materialize and continue.
        if isinstance(x, tuple):
            x = _skip_concat_fallback(*x)
        return self._forward_resnet(x, emb)

    def _forward_openai(self, x, emb, split=0):
        """Fused forward for openaimodel.ResBlock"""
        # THE DECODER SKIP-CONCAT FOLD ARRIVES AS A TUPLE. openaimodel's decoder loop hands
        # (h, hs.pop()) instead of their concatenation, so the fold kernel can read both halves in
        # place and emit the concatenation itself -- one pass over the tensor instead of two.
        #
        # Only the non-updown int4 MoDiff in-path can consume it. Everything else -- and every shape
        # the kernel rejects -- materializes the concatenation right here and proceeds exactly as
        # before, so this is a fast path with a complete fallback rather than a new requirement.
        # A ResBlock is always the FIRST layer of a decoder output block, so no other module ever
        # sees the tuple.
        x_halves = None
        if isinstance(x, tuple):
            x_halves = x
            if self.updown:
                x = _skip_concat_fallback(*x_halves)
                x_halves = None
            else:
                x = None            # materialized below, either by the fold or by the fallback

        if self.updown:
            # GroupNorm+SiLU+quantize+resize in one kernel (see _prequant_gn_resize_conv),
            # which takes the RAW x -- the normalisation happens inside it. GroupNorm is
            # therefore only computed separately on the fallback path (fp16 mode, modiff,
            # uncalibrated, use_conv=True, ...), where the conv needs an fp16 input anyway.
            fused = _prequant_gn_resize_conv_modiff(x, self.fused_in_norm_silu, self.h_upd,
                                                    self.in_conv)
            if fused is None:
                fused = _prequant_gn_resize_conv(x, self.fused_in_norm_silu, self.h_upd,
                                                 self.in_conv)
            if fused is not None:
                h = fused
            else:
                h = self.fused_in_norm_silu(x)
                h = self.h_upd(h)
                h = self.in_conv(h)
            x = self.x_upd(x)
        else:
            # K1->GN fusion: GroupNorm+SiLU emits the conv's quantized input
            # directly (int8, or packed int4), so in_conv skips its own quantize.
            if x_halves is not None:
                # Fold attempt. Returns (out, cat) on success -- `cat` is the concatenation the
                # kernel produced, which the skip conv and the out-conv residual below still need.
                folded = _prequant_gn_conv(x_halves[0], self.fused_in_norm_silu, self.in_conv,
                                           x2=x_halves[1])
                if folded is not None:
                    h, x = folded
                    fused = h
                else:
                    # Declined (wrong dtype/layout, C1 % 32, uncalibrated, non-MoDiff, int8 path).
                    # Materialize and take the ordinary route.
                    x = _skip_concat_fallback(*x_halves)
                    fused = _prequant_gn_conv(x, self.fused_in_norm_silu, self.in_conv)
            else:
                fused = _prequant_gn_conv(x, self.fused_in_norm_silu, self.in_conv)
            if fused is not None:
                h = fused
            else:
                h = self.fused_in_norm_silu(x)
                h = self.in_conv(h)

        # Skip connection is independent of the time-embed projection / out GN.
        if split > 0:
            skip = self.skip_connection(x, split=split)
            residual_arg = None
        else:
            skip = self.skip_connection(x)
            residual_arg = skip

        # Time embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
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
        """Eligible for the fused upsample+quantize path, in EITHER mode.

        This used to require `not modiff_enabled`, which excluded 16 standalone Upsample->conv layers
        from the fusion whenever MoDiff was on -- they fell back to F.interpolate plus a separate
        delta-quantize. That exclusion is gone: `upsample2x_quantize_[pack_]noahat_fprop` now takes an
        OPTIONAL a_hat_cache, so the same kernel does the baseline quantize (empty cache) or the
        MoDiff delta-quantize (real cache). No MoDiff twin kernel was needed -- the loop already
        grids over output elements, so it already visits each a_hat entry exactly once.
        """
        if conv is None or self.orig.dims != 2 or x.dtype != torch.float16 or not x.is_cuda:
            return False
        if not (getattr(conv, 'is_calibrated', False)
                and getattr(conv, 'use_cutlass', False)
                and getattr(conv, 'groups', 1) == 1):
            return False
        if getattr(conv, 'modiff_enabled', False):
            # MoDiff needs a cache to subtract, sized to the conv's (post-upsample) input, and the
            # first step has none. It also needs a published scale: on a dynamic refresh step the
            # scale must be re-measured, which this kernel does not do, so defer to the unfused path
            # there (one step in MODIFF_DELTA_REFRESH).
            if getattr(conv, 'is_first_step', True):
                return False
            if getattr(conv, 'delta_dynamic', False):
                if conv._delta_should_refresh():
                    return False
                if getattr(conv, '_scale_buf', None) is None:
                    return False
            ah = getattr(conv, 'a_hat_cache', None)
            n, c, h, w = x.shape
            if (ah is None or ah.dtype not in (torch.float16, torch.int8)
                    or ah.numel() != n * c * h * 2 * w * 2):
                return False
        return True

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
        if smooth_inv is None:
            smooth_inv = torch.empty(0, device=x.device, dtype=torch.float32)
        is_int4 = hasattr(conv, 'weight_packed')
        modiff = getattr(conv, 'modiff_enabled', False)
        if modiff:
            conv.step_count += 1
            ah = conv.a_hat_cache
            d_scale, d_alpha = (conv._delta_scale_args_i4(x.device) if is_int4
                                else conv._delta_scale_args(x.device))
            if getattr(conv, 'delta_dynamic', False):
                d_scale, d_alpha = conv._scale_buf.view(1), conv._inv_scale_buf.view(1)
        else:
            ah = torch.empty(0, device=x.device, dtype=torch.float16)
            d_scale, d_alpha = conv.static_input_scale.view(1), None
        if is_int4:
            # ONE KERNEL, TWO ROLES, and the zero point belongs to only one of them:
            #   modiff=False -> `ah` is empty, this quantizes the ACTIVATION on the activation grid,
            #                   and _conv_from_int4 adds the zp-corrected bias.  z applies.
            #   modiff=True  -> `ah` is the a_hat cache, this quantizes the DELTA and advances the
            #                   cache, and _conv_from_int4_o_hat adds no bias at all.  z does not
            #                   apply, and passing it would corrupt the cache update.
            # The kernel TORCH_CHECKs the second case, so this branch cannot get it wrong silently.
            zp = getattr(conv, "_zp_float", 0.0)
            write_ahat = (not conv._skip_cache_store()) if modiff else True
            if modiff and hasattr(conv, '_begin_ahat_kernel'):
                conv._begin_ahat_kernel(write_ahat)
            ahat_scale = (conv._ahat_scale_arg() if hasattr(conv, '_ahat_scale_arg')
                          else torch.empty(0, device=x.device, dtype=torch.float32))
            if zp != 0.0 and not modiff:
                if not HAS_UPSAMPLE_QUANTIZE_PACK_ZP:
                    raise RuntimeError(
                        "conv has a non-zero activation zero point but modiff_cutlass lacks "
                        "upsample2x_quantize_pack_noahat_fprop_zp -- rebuild the extension")
                x_q = modiff_cutlass.upsample2x_quantize_pack_noahat_fprop_zp(
                    x, d_scale, smooth_inv, ah, zp)
            else:
                _zp_unsupported(conv, "upsample2x_quantize_pack_noahat_fprop",
                                grid="delta" if modiff else "activation")
                x_q = modiff_cutlass.upsample2x_quantize_pack_noahat_fprop(
                    x, d_scale, smooth_inv, ah, write_ahat, ahat_scale)
            out = (conv._conv_from_int4_o_hat(x_q, x.shape[2] * 2, x.shape[3] * 2, d_alpha)
                   if modiff else conv._conv_from_int4(x_q, x.shape[2] * 2, x.shape[3] * 2))
            if modiff:
                return conv._after_ahat_write(out)
            return out
        write_ahat = (not conv._skip_cache_store()) if modiff else True
        if modiff and hasattr(conv, '_begin_ahat_kernel'):
            conv._begin_ahat_kernel(write_ahat)
        ahat_scale = (conv._ahat_scale_arg() if hasattr(conv, '_ahat_scale_arg')
                      else torch.empty(0, device=x.device, dtype=torch.float32))
        x_q = modiff_cutlass.upsample2x_quantize_noahat_fprop(
            x, d_scale, smooth_inv, ah, write_ahat, ahat_scale)
        out = (conv._conv_from_int8_o_hat(x_q, d_alpha) if modiff
               else conv._conv_from_int8(x_q))
        if modiff:
            return conv._after_ahat_write(out)
        return out


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
