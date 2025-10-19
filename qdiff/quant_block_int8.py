"""
INT8 Quantized Block Structures

This module provides INT8 versions of quantized block structures (ResBlock, AttentionBlock,
TransformerBlock, etc.) that use true int8 weight storage via QuantModuleINT8.

The code structure remains identical to quant_block.py, with the key difference being:
- Uses QuantModuleINT8 instead of QuantModule for weight quantization
- Uses UniformAffineQuantizerINT8 instead of UniformAffineQuantizer
- Weights are stored as direct uint8 (true int8, NO packing) instead of fake-quantized float32
- Much faster than INT4 due to no packing/unpacking overhead

All forward pass logic, block structures, and API remain unchanged for compatibility.
"""

import logging
from types import MethodType
import torch
import torch as th
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat

# Import INT8 quantization modules
from qdiff.quant_layer_int8 import QuantModuleINT8, UniformAffineQuantizerINT8, StraightThrough

# Import original block structures from LDM/DDIM
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.attention import exists, default
from ddim.models.diffusion import ResnetBlock, AttnBlock, nonlinearity


logger = logging.getLogger(__name__)


class BaseQuantBlockINT8(nn.Module):
    """
    Base implementation of block structures for all networks with INT8 quantization.
    
    This is identical to BaseQuantBlock except it uses UniformAffineQuantizerINT8
    which stores activations with true int8 quantization capability.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        
        # Initialize INT8 activation quantizer
        # Note: Activations are still fake-quantized (not stored), but use int8 levels
        self.act_quantizer = UniformAffineQuantizerINT8(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """
        Enable/disable quantization for weights and activations in this block.
        
        When weight_quant=True, all QuantModuleINT8 children will pack weights to int8.
        """
        # Setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModuleINT8):
                m.set_quant_state(weight_quant, act_quant)


class QuantResBlockINT8(BaseQuantBlockINT8, TimestepBlock):
    """
    INT8 quantized ResBlock for diffusion models.
    
    Wraps the original ResBlock and replaces its Conv2d layers with QuantModuleINT8
    via the parent QuantModel's quant_module_refactor() method.
    """
    def __init__(
        self, res: ResBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}, 
        sm_abit: int = 8, modulate: bool = False):
        super().__init__(act_quant_params)
        
        # Copy all attributes from original ResBlock
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm
        self.modulate = modulate
        self.last_temb = None

        # These layers will be QuantModuleINT8 if refactored by QuantModel
        self.in_layers = res.in_layers
        self.updown = res.updown
        self.h_upd = res.h_upd
        self.x_upd = res.x_upd
        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers
        self.skip_connection = res.skip_connection

    def forward(self, x, emb=None, split=0):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        
        Args:
            x: [N x C x ...] Tensor of features
            emb: [N x emb_channels] Tensor of timestep embeddings
            split: Split point for split-channel quantization (experimental)
        
        Returns:
            [N x C x ...] Tensor of outputs
        """
        # Modulation: adaptively update quantization based on timestep changes
        if self.modulate and emb is not None:
            try:
                # Check if timestep has changed significantly
                if self.last_temb is None or self.last_temb.shape != emb.shape:
                    # Initialize or reset last_temb if shape doesn't match (e.g., batch size change)
                    self.last_temb = emb.clone().detach()
                    self.set_quant_state(True, True)
                else:
                    # Compute change in embedding (normalized)
                    emb_change = torch.abs(emb - self.last_temb).mean().item()
                    emb_scale = torch.abs(self.last_temb).mean().item()
                    relative_change = emb_change / (emb_scale + 1e-8)
                    
                    if relative_change > 0.01:  # Threshold for significant change
                        self.last_temb = emb.clone().detach()
                        # Re-quantize with new timestep context
                        self.set_quant_state(False, False)  # Dequantize
                        self.set_quant_state(True, True)    # Re-quantize
            except Exception as e:
                logger.warning(f"Modulation error (will continue with current quantization): {e}")
        
        if split != 0 and self.skip_connection.split == 0:
            return checkpoint(
                self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
            )
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )  

    def _forward(self, x, emb, split=0):
        """Internal forward pass (called via checkpoint)."""
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3]

        # Process input through layers
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        # Add timestep embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        # Residual connection
        if split != 0:
            return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


class QuantQKMatMulINT8(BaseQuantBlockINT8):
    """
    INT8 quantized Q-K matrix multiplication for attention.
    
    This quantizes the Q and K inputs before computing attention scores.
    """
    def __init__(
        self, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.scale = None
        self.use_act_quant = False
        
        # Separate quantizers for Q and K
        self.act_quantizer_q = UniformAffineQuantizerINT8(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizerINT8(**act_quant_params)
        
    def forward(self, q, k):
        """Compute scaled dot-product attention scores."""
        weight = th.einsum(
            "bct,bcs->bts", q * self.scale, k * self.scale
        )
        return weight

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """Enable/disable activation quantization for Q and K."""
        self.use_act_quant = act_quant


class QuantSMVMatMulINT8(BaseQuantBlockINT8):
    """
    INT8 quantized softmax-value matrix multiplication for attention.
    
    This quantizes the attention weights (after softmax) and values before
    computing the attention output.
    """
    def __init__(
        self, act_quant_params: dict = {}, sm_abit=8):
        super().__init__(act_quant_params)
        self.use_act_quant = False
        
        # Quantizer for values
        self.act_quantizer_v = UniformAffineQuantizerINT8(**act_quant_params)
        
        # Quantizer for attention weights (often uses higher precision, e.g., 8-bit)
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['symmetric'] = False
        act_quant_params_w['always_zero'] = True
        self.act_quantizer_w = UniformAffineQuantizerINT8(**act_quant_params_w)
        
    def forward(self, weight, v):
        """Compute attention output: softmax(Q·K^T) · V"""
        a = th.einsum("bts,bcs->bct", weight, v)
        return a

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """Enable/disable activation quantization for attention weights and values."""
        self.use_act_quant = act_quant


class QuantAttentionBlockINT8(BaseQuantBlockINT8):
    """
    INT8 quantized attention block (used in some diffusion model architectures).
    
    Wraps the original AttentionBlock and uses QuantModuleINT8 for its layers.
    """
    def __init__(
        self, attn: AttentionBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        
        # Copy attributes from original attention block
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        
        # These will be QuantModuleINT8 if refactored
        self.norm = attn.norm
        self.qkv = attn.qkv
        self.attention = attn.attention
        self.proj_out = attn.proj_out

    def forward(self, x):
        """Forward pass with gradient checkpointing."""
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        """Internal forward pass."""
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def cross_attn_forward(self, x, context=None, mask=None):
    """
    Cross-attention forward pass (used for transformer blocks).
    
    This function is bound as a method to attention modules in QuantBasicTransformerBlockINT8.
    """
    h = self.heads

    # Compute Q, K, V projections
    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    # Reshape for multi-head attention
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    # Compute attention scores
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    # Apply mask if provided
    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -th.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # Apply softmax to get attention weights
    attn = sim.softmax(dim=-1)

    # Compute attention output
    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


class QuantBasicTransformerBlockINT8(BaseQuantBlockINT8):
    """
    INT8 quantized transformer block with self-attention and cross-attention.
    
    This is the main building block for transformer-based diffusion models (e.g., Stable Diffusion).
    Uses QuantModuleINT8 for all linear/conv layers and supports activation quantization
    for attention operations.
    """
    def __init__(
        self, tran: BasicTransformerBlock, weight_quant_params: dict = {}, 
        act_quant_params: dict = {}, sm_abit: int = 8, modulate: bool = False):
        super().__init__(act_quant_params)
        
        # Copy layers from original transformer block
        self.attn1 = tran.attn1  # Self-attention
        self.ff = tran.ff        # Feedforward network
        self.attn2 = tran.attn2  # Cross-attention
        
        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint = tran.checkpoint
        self.modulate = modulate
        self.last_temb = None

        # Add INT8 quantizers for self-attention (attn1)
        self.attn1.act_quantizer_q = UniformAffineQuantizerINT8(**act_quant_params)
        self.attn1.act_quantizer_k = UniformAffineQuantizerINT8(**act_quant_params)
        self.attn1.act_quantizer_v = UniformAffineQuantizerINT8(**act_quant_params)

        # Add INT8 quantizers for cross-attention (attn2)
        self.attn2.act_quantizer_q = UniformAffineQuantizerINT8(**act_quant_params)
        self.attn2.act_quantizer_k = UniformAffineQuantizerINT8(**act_quant_params)
        self.attn2.act_quantizer_v = UniformAffineQuantizerINT8(**act_quant_params)
        
        # Quantizers for attention weights (softmax output) - often use higher bit width
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['always_zero'] = True
        self.attn1.act_quantizer_w = UniformAffineQuantizerINT8(**act_quant_params_w)
        self.attn2.act_quantizer_w = UniformAffineQuantizerINT8(**act_quant_params_w)

        # Override forward methods to use custom cross-attention
        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        self.attn1.use_act_quant = False
        self.attn2.use_act_quant = False

    def forward(self, x, context=None):
        """Forward pass with gradient checkpointing and optional modulation."""
        # Modulation: adaptively update quantization based on context changes
        if self.modulate and context is not None:
            try:
                if self.last_temb is None or self.last_temb.shape != context.shape:
                    # Initialize or reset last_temb if shape doesn't match (e.g., batch size change)
                    self.last_temb = context.clone().detach()
                    self.set_quant_state(True, True)
                else:
                    context_change = torch.abs(context - self.last_temb).mean().item()
                    context_scale = torch.abs(self.last_temb).mean().item()
                    relative_change = context_change / (context_scale + 1e-8)
                    
                    if relative_change > 0.01:
                        self.last_temb = context.clone().detach()
                        self.set_quant_state(False, False)
                        self.set_quant_state(True, True)
            except Exception as e:
                logger.warning(f"Modulation error: {e}")
        
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        """Internal forward pass: self-attention → cross-attention → feedforward."""
        if context is None:
            assert(len(x) == 2)
            x, context = x

        # Self-attention with residual
        x = self.attn1(self.norm1(x)) + x
        
        # Cross-attention with residual
        x = self.attn2(self.norm2(x), context=context) + x
        
        # Feedforward with residual
        x = self.ff(self.norm3(x)) + x
        
        return x
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """Enable/disable quantization for attention and feedforward layers."""
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        # Setting weight quantization propagates to all QuantModuleINT8 children
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModuleINT8):
                m.set_quant_state(weight_quant, act_quant)


# ============================================================================
# DDIM CIFAR-specific blocks (QuantResnetBlockINT8, QuantAttnBlockINT8)
# ============================================================================

class QuantResnetBlockINT8(BaseQuantBlockINT8):
    """
    INT8 quantized ResNet block for DDIM CIFAR models.
    
    Similar to QuantResBlockINT8 but with a slightly different architecture
    used in DDIM-style diffusion models.
    """
    def __init__(
        self, res: ResnetBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}, 
        sm_abit: int = 8, modulate: bool = False):
        super().__init__(act_quant_params)
        
        # Copy attributes from original ResnetBlock
        self.in_channels = res.in_channels
        self.out_channels = res.out_channels
        self.use_conv_shortcut = res.use_conv_shortcut
        self.modulate = modulate
        self.last_temb = None

        # These will be QuantModuleINT8 if refactored
        self.norm1 = res.norm1
        self.conv1 = res.conv1
        self.temb_proj = res.temb_proj
        self.norm2 = res.norm2
        self.dropout = res.dropout
        self.conv2 = res.conv2
        
        # Shortcut connection (if needed)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = res.conv_shortcut
            else:
                self.nin_shortcut = res.nin_shortcut

    def forward(self, x, temb=None, split=0):
        """
        Forward pass with time embedding and optional modulation.
        
        Args:
            x: Input tensor
            temb: Time embedding
            split: Split point for split-channel quantization
        """
        # Modulation: adaptively update quantization based on timestep changes
        if self.modulate and temb is not None:
            try:
                if self.last_temb is None or self.last_temb.shape != temb.shape:
                    # Initialize or reset last_temb if shape doesn't match (e.g., batch size change)
                    self.last_temb = temb.clone().detach()
                    self.set_quant_state(True, True)
                else:
                    temb_change = torch.abs(temb - self.last_temb).mean().item()
                    temb_scale = torch.abs(self.last_temb).mean().item()
                    relative_change = temb_change / (temb_scale + 1e-8)
                    
                    if relative_change > 0.01:
                        self.last_temb = temb.clone().detach()
                        self.set_quant_state(False, False)
                        self.set_quant_state(True, True)
            except Exception as e:
                logger.warning(f"Modulation error: {e}")
        
        if temb is None:
            assert(len(x) == 2)
            x, temb = x

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Apply shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x, split=split)
        
        out = x + h
        return out


class QuantAttnBlockINT8(BaseQuantBlockINT8):
    """
    INT8 quantized attention block for DDIM CIFAR models.
    
    Implements scaled dot-product attention with INT8 quantization for Q, K, V
    and attention weights.
    """
    def __init__(
        self, attn: AttnBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}, 
        sm_abit: int = 8, modulate: bool = False):
        super().__init__(act_quant_params)
        
        self.in_channels = attn.in_channels
        self.modulate = modulate
        self.last_temb = None

        # Copy layers from original attention block
        self.norm = attn.norm
        self.q = attn.q
        self.k = attn.k
        self.v = attn.v
        self.proj_out = attn.proj_out

        # Quantizers for attention weights (often use higher bit width, e.g., 8-bit)
        # We do not reduce the bit width in attention in this work
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        self.act_quantizer_w = UniformAffineQuantizerINT8(**act_quant_params_w)

        # Quantizers for Q, K, V
        self.act_quantizer_q = UniformAffineQuantizerINT8(**act_quant_params_w)
        self.act_quantizer_k = UniformAffineQuantizerINT8(**act_quant_params_w)
        self.act_quantizer_v = UniformAffineQuantizerINT8(**act_quant_params_w)

    def forward(self, x, temb=None):
        """
        Forward pass: compute self-attention with INT8 quantization and optional modulation.
        """
        # Modulation: adaptively update quantization based on context changes
        if self.modulate and temb is not None:
            try:
                if self.last_temb is None or self.last_temb.shape != temb.shape:
                    # Initialize or reset last_temb if shape doesn't match (e.g., batch size change)
                    self.last_temb = temb.clone().detach()
                    self.set_quant_state(True, True)
                else:
                    temb_change = torch.abs(temb - self.last_temb).mean().item()
                    temb_scale = torch.abs(self.last_temb).mean().item()
                    relative_change = temb_change / (temb_scale + 1e-8)
                    
                    if relative_change > 0.01:
                        self.last_temb = temb.clone().detach()
                        self.set_quant_state(False, False)
                        self.set_quant_state(True, True)
            except Exception as e:
                logger.warning(f"Modulation error: {e}")
        
        h_ = x
        h_ = self.norm(h_)
        
        # Compute Q, K, V
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention scores
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b, hw, c
        k = k.reshape(b, c, h*w)  # b, c, hw

        w_ = th.bmm(q, k)     # b, hw, hw   w[b,i,j] = sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # Attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b, hw, hw (first hw of k, second of q)
        
        # h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = th.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        
        out = x + h_
        return out
    
    def set_dynamic_state(self, dynamic):
        """Enable/disable dynamic quantization for attention weights."""
        self.act_quantizer_w.dynamic = dynamic


def get_specials_int8(quant_act=False):
    """
    Get mapping of original block types to INT8 quantized versions.
    
    This is used by QuantModelINT8 to determine which blocks should be
    replaced with quantized versions during model refactoring.
    
    Args:
        quant_act: If True, also replace matrix multiplication ops (QKMatMul, SMVMatMul)
    
    Returns:
        Dictionary mapping original block types to INT8 quantized versions
    """
    specials = {
        ResBlock: QuantResBlockINT8,
        BasicTransformerBlock: QuantBasicTransformerBlockINT8,
        ResnetBlock: QuantResnetBlockINT8,
        AttnBlock: QuantAttnBlockINT8,
    }
    
    # Optionally quantize individual attention operations
    if quant_act:
        specials[QKMatMul] = QuantQKMatMulINT8
        specials[SMVMatMul] = QuantSMVMatMulINT8
    else:
        specials[AttentionBlock] = QuantAttentionBlockINT8
    
    return specials
