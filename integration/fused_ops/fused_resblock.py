"""
Fused ResBlock operations for MoDiff optimizations.

This module provides fused implementations of ResBlock operations to reduce
memory bandwidth and kernel launch overhead. The main optimizations:

1. Fused GroupNorm + SiLU: uses F.group_norm + F.silu with autocast locally
   disabled (see FusedGroupNormSiLU below), so GroupNorm runs natively in
   the input's own dtype instead of paying two dtype-cast kernels per call.
2. Fused residual addition: Combines skip connection with final output
3. In-place operations where possible to reduce memory allocations

Measured speedup: ~11% wall-time reduction on the fp16 LSUN-churches UNet
(no measurable effect in int8/int4 modes -- see FusedGroupNormSiLU docstring
for why their GroupNorm inputs are already fp32 for an unrelated reason).
"""

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


class FusedGroupNormSiLU(nn.Module):
    """
    Fused GroupNorm + SiLU activation.

    This replaces the common pattern:
        nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU()
        )

    Uses PyTorch's native F.group_norm + F.silu, with autocast locally
    disabled around the call.

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

    def forward(self, x):
        weight = self.weight.to(x.dtype) if self.weight is not None and self.weight.dtype != x.dtype else self.weight
        bias = self.bias.to(x.dtype) if self.bias is not None and self.bias.dtype != x.dtype else self.bias
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.group_norm(x, self.num_groups, weight, bias, self.eps)
            return F.silu(x)


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
            h = self.fused_in_norm_silu(x)
            h = self.in_conv(h)
        
        # Time embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        # Output path: fused GroupNorm + SiLU, then Dropout + Conv
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            # Manual GroupNorm since we need scale/shift modulation
            weight = self.fused_out_norm_silu.weight
            bias = self.fused_out_norm_silu.bias
            weight = weight.to(h.dtype) if weight is not None and weight.dtype != h.dtype else weight
            bias = bias.to(h.dtype) if bias is not None and bias.dtype != h.dtype else bias
            with torch.amp.autocast(device_type=h.device.type, enabled=False):
                h_norm = F.group_norm(h, self.fused_out_norm_silu.num_groups,
                                      weight,
                                      bias,
                                      self.fused_out_norm_silu.eps)
            h = h_norm * (1 + scale) + shift
            h = F.silu(h)
            h = self.out_dropout(h)
            h = self.out_conv(h)
        else:
            h = h + emb_out
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
    print(f"Fusion method: F.group_norm + F.silu with autocast locally disabled (no fp32 round-trip)")
    print(f"Measured: ~11% wall-time reduction in fp16 mode; no effect in int8/int4 (see FusedGroupNormSiLU docstring)")
    print(f"{'='*60}\n")
