"""Quantized token-major AttentionBlock: extends TokenMajorAttentionBlock with an
int8/int4 quantized score path (QKᵀ / softmax / AV) via the fused flash kernel
`modiff_cutlass.flash_attn_int8`, which never materializes the [N,H,T,T] score
matrix. Drop-in, numerically close to the fp16 block.

Layers (each independently gated):
  - score_bits ∈ {16,8}: 16 → fp16 math SDPA (fallback); 8 → fused int8 flash.
    (int4 scores are a later, FID-gated experiment.)
  - proj_bits ∈ {16,8,4}: qkv/proj projection precision (16 = fp16 for now).
  - modiff: MoDiff temporal delta on qkv/proj only (later milestone).

The qkv→int8 quantization is currently done in PyTorch (per-token Q/K, per-channel
V, head_dim padded to a multiple of 32); it will move into a fused CUDA kernel
(`quantize_qkv_int8`) in a perf pass. The score matmul is never delta-cached
(softmax is nonlinear/bilinear — no clean MoDiff accumulation identity).

Kill-switches: MODIFF_DISABLE_FLASH_INT8=1 forces fp16 SDPA; MODIFF_FLASH_MIN_T
(default 64) keeps small-T blocks on fp16; MODIFF_ATTN_SCORE_BITS overrides bits.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from integration.fused_ops.token_major_attention import (
    TokenMajorAttentionBlock, convert_attention_to_token_major,
    _SDPA_MATH_CTX, _FUSE_SHIFT, _FUSE_TILE_M, _HAS_FUSED_GN_QKV,
    _group_norm_silu,
)

try:
    import modiff_cutlass as _mc
    _HAS_FLASH_INT8 = hasattr(_mc, "flash_attn_int8")
    _HAS_QUANT_QKV = hasattr(_mc, "quantize_qkv_int8")
except Exception:
    _mc = None
    _HAS_FLASH_INT8 = False
    _HAS_QUANT_QKV = False

try:
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, QKVAttentionLegacy
    _HAS_ATTN = True
except Exception:
    _HAS_ATTN = False
    AttentionBlock = QKVAttentionLegacy = None

_FLASH_MIN_T = int(os.environ.get("MODIFF_FLASH_MIN_T", "64"))


def _pad_hd(x, hd_pad):
    hd = x.shape[-1]
    if hd_pad == hd:
        return x.contiguous()
    return F.pad(x, (0, hd_pad - hd)).contiguous()


class QuantizedTokenMajorAttentionBlock(TokenMajorAttentionBlock):
    """TokenMajorAttentionBlock + quantized score path."""

    def __init__(self, orig, *, score_bits=8, proj_bits=16, modiff=False):
        super().__init__(orig)
        env_bits = os.environ.get("MODIFF_ATTN_SCORE_BITS")
        self.score_bits = int(env_bits) if env_bits is not None else score_bits
        self.proj_bits = proj_bits
        self.modiff = modiff
        if os.environ.get("MODIFF_DISABLE_FLASH_INT8") == "1" or not _HAS_FLASH_INT8:
            self.score_bits = 16

    def _quant_flash(self, q, k, v):
        """q,k,v: [B,H,T,hd] fp16 -> [B,H,T,hd] fp16 via fused int8 flash.
        Per-token dequant scales for Q/K, per-channel (head_dim) for V. The
        quantize is a fused CUDA kernel (quantize_qkv_int8) when available, else
        a PyTorch fallback."""
        B, Hh, T, hd = q.shape
        hd_pad = (hd + 31) // 32 * 32
        if _HAS_QUANT_QKV:
            qi, ki, vi, sq, sk, scv = _mc.quantize_qkv_int8(q, k, v, hd_pad)
            return _mc.flash_attn_int8(qi, ki, vi, sq, sk, scv, 1.0 / math.sqrt(hd))

        def qtok(x):  # per-token: scale [B,H,T]
            amax = x.abs().amax(-1).clamp_min(1e-8)
            sc = amax / 127.0
            xi = torch.round(x / sc.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
            return _pad_hd(xi, hd_pad), sc.to(torch.float32).contiguous()

        qi, sq = qtok(q)
        ki, sk = qtok(k)
        amaxv = v.abs().amax(2).clamp_min(1e-8)          # [B,H,hd]
        scv = amaxv / 127.0
        vi = torch.round(v / scv.unsqueeze(2)).clamp_(-127, 127).to(torch.int8)
        vi = _pad_hd(vi, hd_pad)
        scv = scv.to(torch.float32).contiguous()
        return _mc.flash_attn_int8(qi, ki, vi, sq, sk, scv, 1.0 / math.sqrt(hd))

    def forward(self, x):
        b, c, H, W = x.shape
        T = H * W
        nh, hd = self.num_heads, self.head_dim
        x_in_tok = x.permute(0, 2, 3, 1).reshape(b, T, c)

        # ---- qkv (fp16), reusing the fused GN->qkv path or GN+cuBLAS fallback ----
        use_fused = (self._fuse_gn_qkv and _HAS_FUSED_GN_QKV and x.dtype == torch.float16
                     and (T % _FUSE_TILE_M) == 0 and (c % 8) == 0)
        if use_fused:
            self._ensure_fused()
            qkv_img = _mc.fused_gn_qkv(x, self._fused_conv_w, self._fused_epi_bias,
                                       self.norm.num_groups, self.norm.eps, _FUSE_SHIFT)
            qkv = qkv_img.permute(0, 2, 3, 1).reshape(b, T, nh, 3, hd)
        else:
            w, bnorm = self._gn_params(x.dtype)
            xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
            xn_tok = xn.permute(0, 2, 3, 1).reshape(b, T, c)
            qkv = self.qkv(xn_tok).view(b, T, nh, 3, hd)

        q, k, v = qkv.unbind(3)                            # each [b,T,nh,hd]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # [b,nh,T,hd]

        # ---- score path ----
        if self.score_bits == 8 and T >= _FLASH_MIN_T:
            a = self._quant_flash(q.contiguous(), k.contiguous(), v.contiguous())  # [b,nh,T,hd]
        else:
            with _SDPA_MATH_CTX():
                a = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        a = a.transpose(1, 2).reshape(b, T, c)
        out_tok = x_in_tok + self.proj(a)
        return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)


def convert_attention_to_quantized(module, *, score_bits=8, proj_bits=16, modiff=False, verbose=False):
    """Recursively replace plain AttentionBlock instances with the quantized
    token-major variant. Same guards as convert_attention_to_token_major
    (legacy Conv1d block, channels % heads == 0). No-op if
    MODIFF_DISABLE_TOKEN_MAJOR_ATTN=1. Returns the count converted."""
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
                setattr(module, name, QuantizedTokenMajorAttentionBlock(
                    child, score_bits=score_bits, proj_bits=proj_bits, modiff=modiff))
                converted += 1
                if verbose:
                    print(f"  quantized attention: {name} (C={child.channels}, "
                          f"heads={child.num_heads}, score_bits={score_bits})")
        else:
            converted += convert_attention_to_quantized(
                child, score_bits=score_bits, proj_bits=proj_bits, modiff=modiff, verbose=verbose)
    return converted
