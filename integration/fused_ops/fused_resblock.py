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
except ImportError:
    modiff_cutlass = None
    HAS_NATIVE_GN_SILU = False
    HAS_GN_SILU_QUANTIZE = False
    HAS_GN_SILU_QUANTIZE_PACK = False

# Kill-switch for the GN->intX K1-fusion (baseline int8/int4). Set
# MODIFF_DISABLE_GN_INT8_FUSION=1 to fall back to the exact two-kernel
# (GroupNorm+SiLU, then standalone quantize[/pack]) path -- used for A/B
# benchmarking and as a production safety switch. Pure optimization either way.
if os.environ.get("MODIFF_DISABLE_GN_INT8_FUSION") == "1":
    HAS_GN_SILU_QUANTIZE = False
    HAS_GN_SILU_QUANTIZE_PACK = False


def _group_norm_silu(x, num_groups, weight, bias, eps, apply_silu):
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
    if can_use_native:
        return modiff_cutlass.group_norm_silu_nhwc(x, weight, bias, num_groups, eps, apply_silu)
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        out = F.group_norm(x, num_groups, weight, bias, eps)
        return F.silu(out) if apply_silu else out


def _prequant_common_ok(conv):
    """Common gates for the GN->intX K1-fusion, shared by int8 and int4.

    Baseline-only: the MoDiff modulated path (modiff_enabled=True) does an in-place
    a_hat cache update inside its quantize that a fused GN-emit would bypass and
    corrupt, so we require modiff_enabled=False. SmoothQuant / grouped convs /
    non-fp16 output all fall back to the exact path -- pure optimization, never a
    correctness change.
    """
    return (
        getattr(conv, 'is_calibrated', False)
        and not getattr(conv, 'modiff_enabled', True)
        and getattr(conv, 'standard_output_fp16', False)
        and getattr(conv, 'use_cutlass', False)
        and getattr(conv, 'groups', 1) == 1
        and getattr(conv, '_smooth_is_identity', False)
    )


def _prequant_gn_conv(x, gn, conv, mod_scale=None, mod_shift=None):
    """If (GroupNorm+SiLU `gn` -> quantized conv `conv`) is eligible for the
    GN->intX K1-fusion, run GroupNorm(+optional scale-shift modulation)+SiLU
    emitting the conv's quantized input directly (int8, or packed int4) and then
    conv.forward_from_intX, returning the conv output. Otherwise return None so the
    caller uses the normal path.

    `mod_scale`/`mod_shift` (each [N, C, 1, 1] from the timestep embedding, or None)
    add the use_scale_shift_norm modulation `normed*(1+scale)+shift` between the GN
    affine and the SiLU, folding it (and the SiLU) into the one kernel too. The
    quantize multiplier is conv.static_input_scale (=127/absmax); smooth_inv is
    identity (gated above).
    """
    if not _prequant_common_ok(conv):
        return None
    is_int4 = hasattr(conv, 'forward_from_int4')
    if is_int4 and not HAS_GN_SILU_QUANTIZE_PACK:
        return None
    if not is_int4 and (not HAS_GN_SILU_QUANTIZE or not hasattr(conv, 'forward_from_int8')):
        return None

    ng, eps = gn.num_groups, gn.eps
    w, b = gn._cast_params(x.dtype)
    conv._ensure_conv_caches(x.device)
    scale = conv._cached_scale_tensor          # fp32 [1], =static_input_scale
    smooth_inv = conv._empty_smooth            # empty -> identity
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
        return conv.forward_from_int4(packed, h_in, w_in)
    else:
        if native_ok:
            q = modiff_cutlass.group_norm_silu_quantize_nhwc(
                x, w, b, ng, eps, True, scale, smooth_inv, ms2d, sh2d)
        else:
            q = modiff_cutlass.scale_quantize_int8(
                _gn_mod_silu_fp32_cl(x, ng, w, b, eps, mod_scale, mod_shift), scale)
        return conv.forward_from_int8(q)


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
        
        # Output path: fused GroupNorm + SiLU, then Dropout + Conv
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            # K1->GN fusion with scale-shift modulation folded in: GroupNorm ->
            # normed*(1+scale)+shift -> SiLU -> quantize, all in one kernel, then
            # out_conv.forward_from_intX. out_dropout is nn.Dropout (no-op in eval).
            fused = _prequant_gn_conv(h, self.fused_out_norm_silu, self.out_conv,
                                      mod_scale=scale, mod_shift=shift)
            if fused is not None:
                h = fused
            else:
                # Manual GroupNorm since we need scale/shift modulation
                weight, bias = self.fused_out_norm_silu._cast_params(h.dtype)
                h_norm = _group_norm_silu(h, self.fused_out_norm_silu.num_groups,
                                           weight, bias,
                                           self.fused_out_norm_silu.eps,
                                           apply_silu=False)
                h = h_norm * (1 + scale) + shift
                # If out_conv is a calibrated quantized conv with fuse_input_silu
                # set (see wire_silu_fusion), it applies SiLU itself (fused into
                # its quantize kernel) -- skip the separate F.silu(h) pass here.
                if not getattr(self.out_conv, 'fuse_input_silu', False):
                    h = F.silu(h)
                h = self.out_dropout(h)
                h = self.out_conv(h)
        else:
            h = h + emb_out
            # K1->GN fusion: GroupNorm+SiLU emits the conv's quantized input
            # directly; out_conv skips its own quantize. out_dropout is nn.Dropout
            # (no-op in eval/inference), so it's elided on the fused path.
            fused = _prequant_gn_conv(h, self.fused_out_norm_silu, self.out_conv)
            if fused is not None:
                h = fused
            else:
                h = self.fused_out_norm_silu(h)
                h = self.out_dropout(h)
                h = self.out_conv(h)
        
        # Fused residual addition
        if split > 0:
            return torch.add(self.skip_connection(x, split=split), h)
        else:
            return torch.add(self.skip_connection(x), h)
    
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
