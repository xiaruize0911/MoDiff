"""Token-major AttentionBlock: a drop-in, numerically-identical replacement for
ldm.modules.diffusionmodules.openaimodel.AttentionBlock that eliminates the
layout-copy overhead profiled around attention.

The original block runs channel-major ([N, C, T]) with 1x1 Conv1d qkv/proj, so
with the channels_last activations this pipeline uses it pays three forced copies
per block (measured ~2.5 ms/iter total at batch 32 on churches):
  1. `x.reshape(b, c, -1)` on a channels_last input  -> NCHW reorder copy
  2. `qkv.reshape(...).permute(...).contiguous()`     -> channel-major -> attn layout
  3. `a.permute(...).reshape(...)` + final reshape     -> attn -> channel-major, then NCHW

A 1x1 Conv1d over the channel dim is exactly an nn.Linear, so running the block
token-major ([N, T, C]) makes copies (1) and (3)'s reshapes free views of the
channels_last memory, and feeds SDPA strided views without `.contiguous()`.
Only the post-attention transpose (SDPA output [N,H,T,hd] -> [N,T,C]) remains.

Attention runs on the explicit **math (non-flash) SDPA backend** (see
`_SDPA_MATH_CTX`): the QK^T / AV products are then plain cuBLAS batched GEMMs
that can be intercepted/quantized, unlike the opaque fused-flash kernel. This is
the permanent design of this block, not a toggle.

Weights are copied bit-for-bit from the Conv1d layers, so the result is
numerically identical up to GroupNorm-kernel / SDPA rounding (~1e-3 e2e).
This is shared attention code, so it speeds up every mode (fp16/int8/int4) equally.

Kill-switch: MODIFF_DISABLE_TOKEN_MAJOR_ATTN=1 skips the conversion.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention runs on the math (non-flash) SDPA backend permanently, so the
# QK^T/AV products stay as regular cuBLAS GEMMs (interceptable/quantizable). The
# try/except only picks the right API for the installed torch; both force math.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA_MATH_CTX = lambda: sdpa_kernel(SDPBackend.MATH)
except Exception:  # pragma: no cover - older torch
    from contextlib import contextmanager
    @contextmanager
    def _SDPA_MATH_CTX():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield

from integration.fused_ops.fused_resblock import _group_norm_silu

try:
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, QKVAttentionLegacy
    _HAS_ATTN = True
except Exception:
    _HAS_ATTN = False
    AttentionBlock = QKVAttentionLegacy = None

try:
    import modiff_cutlass as _mc
    _HAS_FUSED_GN_QKV = hasattr(_mc, "fused_gn_qkv")
except Exception:
    _HAS_FUSED_GN_QKV = False

# Constant that absorbs the CUTLASS fprop-fusion ReLU: bias carries +SHIFT so the
# pre-ReLU value (x-mean)*rstd + SHIFT is always >= 0 (normalized activations are
# ~unit variance). Must match the value used in fused_gn_qkv.cu's caller.
_FUSE_SHIFT = 16.0
# The fused conv's threadblock tile has kM=128, so it is only correct when a tile
# stays within one sample, i.e. tokens T = H*W is a multiple of 128.
_FUSE_TILE_M = 128


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

        # Fused GroupNorm->qkv via the custom CUTLASS per-sample mainloop-fusion conv.
        # Default ON; kill-switch MODIFF_FUSE_GN_QKV=0. Folded conv weight + epilogue
        # bias are built lazily on first forward. See csrc/kernels/fused_gn_qkv.cu.
        self._fuse_gn_qkv = os.environ.get("MODIFF_FUSE_GN_QKV", "1") != "0"
        self._fused_ready = False
        self._fused_conv_w = None    # [3C,1,1,C] fp16 KRSC = qkv.weight * gn.weight
        self._fused_epi_bias = None  # [3C] fp16 = qkv.bias + qkv.w@gn.b - SHIFT*colsum(Wf)

    def _gn_params(self, dtype):
        if self._gn_cast_dtype is not dtype:
            self._gn_w = self.norm.weight.detach().to(dtype) if self.norm.weight is not None else None
            self._gn_b = self.norm.bias.detach().to(dtype) if self.norm.bias is not None else None
            self._gn_cast_dtype = dtype
        return self._gn_w, self._gn_b

    def _ensure_fused(self):
        """Lazily fold GroupNorm's affine into the qkv conv weight + epilogue bias.
        The +SHIFT that absorbs the mainloop ReLU is added to the activation bias in
        the CUDA stats kernel; here we subtract its induced constant SHIFT*colsum(Wf)
        from the per-output-channel bias so the net result is exact."""
        if self._fused_ready:
            return
        C = self.channels
        w = self.qkv.weight.detach().to(torch.float16)      # [3C, C]
        b = self.qkv.bias.detach().to(torch.float16)        # [3C]
        gw = (self.norm.weight.detach().to(torch.float16) if self.norm.weight is not None
              else torch.ones(C, device=w.device, dtype=torch.float16))
        gb = (self.norm.bias.detach().to(torch.float16) if self.norm.bias is not None
              else torch.zeros(C, device=w.device, dtype=torch.float16))
        Wf = (w * gw[None, :]).contiguous()                 # [3C, C]
        self._fused_conv_w = Wf.view(3 * C, 1, 1, C).contiguous()
        self._fused_epi_bias = (b + w @ gb - _FUSE_SHIFT * Wf.sum(dim=1)).contiguous().to(torch.float16)
        self._fused_ready = True

    def forward(self, x):
        # x: [N, C, H, W] (channels_last in this pipeline). All the .permute/.reshape
        # below are free views when x is channels_last-contiguous.
        b, c, H, W = x.shape
        T = H * W
        nh, hd = self.num_heads, self.head_dim

        # channels_last [N,C,H,W] (physical [N,H,W,C]) -> token-major [N,T,C] view
        # (needed for the residual add regardless of the qkv path).
        x_in_tok = x.permute(0, 2, 3, 1).reshape(b, T, c)

        # Fused GroupNorm->qkv (custom CUTLASS per-sample mainloop fusion): computes
        # the GroupNorm-normalized qkv directly from the raw channels_last activation,
        # skipping the separate GroupNorm kernel + its intermediate write. Gated to
        # fp16 and T a multiple of the conv's tile-M (else the per-sample scale offset
        # would be wrong); otherwise fall back to GroupNorm + cuBLAS below.
        if (self._fuse_gn_qkv and _HAS_FUSED_GN_QKV and x.dtype == torch.float16
                and (T % _FUSE_TILE_M) == 0 and (c % 8) == 0):
            self._ensure_fused()
            qkv_img = _mc.fused_gn_qkv(x, self._fused_conv_w, self._fused_epi_bias,
                                       self.norm.num_groups, self.norm.eps, _FUSE_SHIFT)
            qkv = qkv_img.permute(0, 2, 3, 1).reshape(b, T, nh, 3, hd)
            q, k, v = qkv.unbind(3)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            with _SDPA_MATH_CTX():                        # math SDPA backend (see module header)
                a = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            a = a.transpose(1, 2).reshape(b, T, c)
            out_tok = x_in_tok + self.proj(a)
            return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)

        # GroupNorm over channels via the native NHWC kernel -> stays channels_last.
        w, bnorm = self._gn_params(x.dtype)
        xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
        xn_tok = xn.permute(0, 2, 3, 1).reshape(b, T, c)

        # qkv: Linear [N,T,C] -> [N,T,3C]; channel order matches the Conv1d exactly:
        # (num_heads, {q,k,v}, head_dim). Split into flash layout [N,H,T,hd] via views.
        qkv = self.qkv(xn_tok).view(b, T, nh, 3, hd)
        q, k, v = qkv.unbind(3)                       # each [N,T,H,hd] (views)
        q = q.transpose(1, 2)                          # [N,H,T,hd], last dim stride 1
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with _SDPA_MATH_CTX():                          # math SDPA backend (see module header)
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
