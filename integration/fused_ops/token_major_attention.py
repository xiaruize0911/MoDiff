"""Token-major AttentionBlock: a drop-in, numerically-identical replacement for
ldm.modules.diffusionmodules.openaimodel.AttentionBlock that eliminates the
layout-copy overhead profiled around attention.

The original block runs channel-major ([N, C, T]) with 1x1 Conv1d qkv/proj, so
with the channels_last activations this pipeline uses it pays three forced copies
per block (measured ~2.5 ms/iter total at batch 32 on churches):
  1. `x.reshape(b, c, -1)` on a channels_last input  -> NCHW reorder copy
  2. `qkv.reshape(...).permute(...).contiguous()`     -> channel-major -> flash layout
  3. `a.permute(...).reshape(...)` + final reshape     -> flash -> channel-major, then NCHW

A 1x1 Conv1d over the channel dim is exactly an nn.Linear, so running the block
token-major ([N, T, C]) makes copies (1) and (3)'s reshapes free views of the
channels_last memory, and feeds flash-attn strided views without `.contiguous()`.
Only the post-attention transpose (SDPA output [N,H,T,hd] -> [N,T,C]) remains.

Weights are copied bit-for-bit from the Conv1d layers, so the result is
numerically identical up to GroupNorm-kernel / flash-stride rounding (~1e-3 e2e).
This is shared attention code, so it speeds up every mode (fp16/int8/int4) equally.

Kill-switch: MODIFF_DISABLE_TOKEN_MAJOR_ATTN=1 skips the conversion.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from integration.fused_ops.fused_resblock import _group_norm_silu

try:
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, QKVAttentionLegacy
    _HAS_ATTN = True
except Exception:
    _HAS_ATTN = False
    AttentionBlock = QKVAttentionLegacy = None


class TokenMajorAttentionBlock(nn.Module):
    """Drop-in for AttentionBlock (QKVAttentionLegacy head order) that runs
    token-major so the reshape/permute copies collapse to free views."""

    def __init__(self, orig):
        super().__init__()
        assert isinstance(orig.attention, QKVAttentionLegacy), \
            "TokenMajorAttentionBlock only supports the legacy (split-heads-first) order"
        C = orig.channels
        self.channels = C
        self.num_heads = orig.num_heads
        self.head_dim = C // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Reuse the original GroupNorm (params + num_groups + eps) verbatim.
        self.norm = orig.norm
        # 1x1 Conv1d -> Linear: weight [outC, inC, 1] -> [outC, inC], same math.
        qkv_conv, proj_conv = orig.qkv, orig.proj_out
        dev, dt = qkv_conv.weight.device, qkv_conv.weight.dtype
        self.qkv = nn.Linear(C, 3 * C).to(device=dev, dtype=dt)
        self.proj = nn.Linear(C, C).to(device=dev, dtype=dt)
        with torch.no_grad():
            self.qkv.weight.copy_(qkv_conv.weight.reshape(3 * C, C))
            self.qkv.bias.copy_(qkv_conv.bias)
            self.proj.weight.copy_(proj_conv.weight.reshape(C, C))
            self.proj.bias.copy_(proj_conv.bias)

        # Cache dtype-matched GroupNorm affine params (mirrors FusedGroupNormSiLU).
        self._gn_cast_dtype = None
        self._gn_w = None
        self._gn_b = None

    def _gn_params(self, dtype):
        if self._gn_cast_dtype is not dtype:
            self._gn_w = self.norm.weight.detach().to(dtype) if self.norm.weight is not None else None
            self._gn_b = self.norm.bias.detach().to(dtype) if self.norm.bias is not None else None
            self._gn_cast_dtype = dtype
        return self._gn_w, self._gn_b

    def forward(self, x):
        # x: [N, C, H, W] (channels_last in this pipeline). All the .permute/.reshape
        # below are free views when x is channels_last-contiguous.
        b, c, H, W = x.shape
        T = H * W
        nh, hd = self.num_heads, self.head_dim

        # GroupNorm over channels via the native NHWC kernel -> stays channels_last.
        w, bnorm = self._gn_params(x.dtype)
        xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)

        # channels_last [N,C,H,W] (physical [N,H,W,C]) -> token-major [N,T,C] view.
        x_in_tok = x.permute(0, 2, 3, 1).reshape(b, T, c)
        xn_tok = xn.permute(0, 2, 3, 1).reshape(b, T, c)

        # qkv: Linear [N,T,C] -> [N,T,3C]; channel order matches the Conv1d exactly:
        # (num_heads, {q,k,v}, head_dim). Split into flash layout [N,H,T,hd] via views.
        qkv = self.qkv(xn_tok).view(b, T, nh, 3, hd)
        q, k, v = qkv.unbind(3)                       # each [N,T,H,hd] (views)
        q = q.transpose(1, 2)                          # [N,H,T,hd], last dim stride 1
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # [N,H,T,hd]

        # [N,H,T,hd] -> [N,T,C] (head-major C), proj, residual, back to channels_last.
        a = a.transpose(1, 2).reshape(b, T, c)         # only remaining copy
        h = self.proj(a)
        out_tok = x_in_tok + h                          # [N,T,C]
        # [N,T,C] -> [N,H,W,C] -> channels_last [N,C,H,W] (free views).
        return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)


def convert_attention_to_token_major(module, verbose=False):
    """Recursively replace plain AttentionBlock instances with the token-major
    variant. Skips blocks whose qkv/proj are not vanilla Conv1d (e.g. already
    MoDiff-wrapped for attn_modiff modes) and the non-legacy attention order.
    No-op if MODIFF_DISABLE_TOKEN_MAJOR_ATTN=1. Returns the count converted."""
    if not _HAS_ATTN or os.environ.get("MODIFF_DISABLE_TOKEN_MAJOR_ATTN") == "1":
        return 0
    converted = 0
    for name, child in list(module.named_children()):
        if isinstance(child, AttentionBlock):
            ok = (isinstance(child.attention, QKVAttentionLegacy)
                  and isinstance(child.qkv, nn.Conv1d)
                  and isinstance(child.proj_out, nn.Conv1d)
                  and child.channels % child.num_heads == 0)
            if ok:
                setattr(module, name, TokenMajorAttentionBlock(child))
                converted += 1
                if verbose:
                    print(f"  token-major attention: {name} (C={child.channels}, heads={child.num_heads})")
            elif verbose:
                print(f"  skip attention {name} (not a plain legacy Conv1d block)")
        else:
            converted += convert_attention_to_token_major(child, verbose=verbose)
    return converted
