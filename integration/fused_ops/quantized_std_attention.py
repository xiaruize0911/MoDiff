"""Materialized quantized STANDARD attention (W8A8 / W4A4) — no flash.

Assembles the batched int8/int4 kernels in csrc/kernels/attn_quant_gemm.cu into a
standard attention block: GroupNorm -> qkv(fp16) -> quantize Q/K/V -> QKᵀ (int GEMM,
materialized [BH,T,T] scores) -> softmax+requant -> AV (int GEMM) -> proj -> residual.

bits=8 (W8A8) is quality-safe (latent rel-err ~0.02-0.03 per block); bits=4 (W4A4) is
aggressive (int4 Q·K·V + 8-level int4 P, rel ~0.35-0.45) and is meant to be compensated
by MoDiff temporal-delta caching across DDIM steps. Drop-in for AttentionBlock; extends
TokenMajorAttentionBlock (reuses its GN + qkv/proj Linears, weight copy, head config).
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from integration.fused_ops.token_major_attention import (
    TokenMajorAttentionBlock, _group_norm_silu)

try:
    import modiff_cutlass as _mc
    _HAS = all(hasattr(_mc, k) for k in
               ("attn_qk_int8", "attn_softmax_requant", "attn_av_int8",
                "attn_qk_int4", "attn_softmax_requant4", "attn_av_int4"))
except Exception:
    _mc = None
    _HAS = False

try:
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, QKVAttentionLegacy
    _HAS_ATTN = True
except Exception:
    _HAS_ATTN = False
    AttentionBlock = QKVAttentionLegacy = None


def _pack_last(t):  # int8 values in [-7,7], pack pairs on the last dim -> [..., n/2]
    lo = t[..., 0::2] & 0xF
    hi = t[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8).contiguous()


class QuantizedStandardAttentionBlock(TokenMajorAttentionBlock):
    """TokenMajorAttentionBlock with the score path (QKᵀ/softmax/AV) quantized to int8/int4."""

    def __init__(self, orig, *, bits=8):
        super().__init__(orig)
        assert bits in (4, 8)
        self.bits = bits
        self.Q = 127 if bits == 8 else 7
        self.hp_qk = (self.head_dim + 31) // 32 * 32 if bits == 8 else (self.head_dim + 63) // 64 * 64
        self.hp_av = (self.head_dim + 63) // 64 * 64          # AV N tile needs %64

    def _qtok(self, x, hd_pad):
        """per-token int8/int4 over the last dim; pad hd->hd_pad. x:[BH,T,hd] -> (xi, scale[BH,T])."""
        sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / self.Q
        xi = torch.round(x / sc).clamp_(-self.Q, self.Q).to(torch.int8)
        xi = F.pad(xi, (0, hd_pad - x.shape[-1])).contiguous()
        if self.bits == 4:
            xi = _pack_last(xi)
        return xi, sc.squeeze(-1).float().contiguous()

    def _qattn(self, q, k, v, T, BH):
        """q,k,v: [BH,T,hd] fp16 -> attention output [BH,T,hd] fp16 (materialized int GEMMs)."""
        hd = self.head_dim
        scale = 1.0 / math.sqrt(hd)
        qi, sq = self._qtok(q, self.hp_qk)
        ki, sk = self._qtok(k, self.hp_qk)
        # V: per-channel-over-T scale; transpose to [BH,hd,T], pad to hp_av, (pack for int4)
        svc = v.abs().amax(1, keepdim=True).clamp_min(1e-8) / self.Q     # [BH,1,hd]
        vi = torch.round(v / svc).clamp_(-self.Q, self.Q).to(torch.int8) # [BH,T,hd]
        vt = F.pad(vi.transpose(1, 2).contiguous(), (0, 0, 0, self.hp_av - hd)).contiguous()  # [BH,hp_av,T]
        sv = F.pad(svc.squeeze(1), (0, self.hp_av - hd)).float().contiguous()                  # [BH,hp_av]
        if self.bits == 8:
            S = _mc.attn_qk_int8(qi, ki)
            P, sp = _mc.attn_softmax_requant(S, sq, sk, scale)
            O = _mc.attn_av_int8(P, vt, sp, sv)
        else:
            vt = _pack_last(vt)                                           # [BH,hp_av,T/2]
            S = _mc.attn_qk_int4(qi, ki, self.hp_qk)
            P, sp = _mc.attn_softmax_requant4(S, sq, sk, scale)
            O = _mc.attn_av_int4(P, vt, sp, sv, T)
        return O[:, :, :hd]                                              # drop hd_pad

    def forward(self, x):
        b, c, H, W = x.shape
        T = H * W
        nh, hd = self.num_heads, self.head_dim
        BH = b * nh
        x_in_tok = x.permute(0, 2, 3, 1).reshape(b, T, c)
        # GroupNorm + qkv (fp16). Reuse fused GN->qkv when eligible, else GN + Linear.
        from integration.fused_ops.token_major_attention import _HAS_FUSED_GN_QKV, _FUSE_TILE_M, _FUSE_SHIFT
        if (self._fuse_gn_qkv and _HAS_FUSED_GN_QKV and x.dtype == torch.float16
                and (T % _FUSE_TILE_M) == 0 and (c % 8) == 0):
            self._ensure_fused()
            qkv_img = _mc.fused_gn_qkv(x, self._fused_conv_w, self._fused_epi_bias,
                                       self.norm.num_groups, self.norm.eps, _FUSE_SHIFT)
            qkv = qkv_img.permute(0, 2, 3, 1).reshape(b, T, nh, 3, hd)
        else:
            w, bnorm = self._gn_params(x.dtype)
            xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
            qkv = self.qkv(xn.permute(0, 2, 3, 1).reshape(b, T, c)).view(b, T, nh, 3, hd)
        q, k, v = qkv.unbind(3)                                          # [b,T,nh,hd]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)   # [b,nh,T,hd]
        # The batched int GEMMs need T%64==0 (N tile); small-T blocks (e.g. 4x4, T=16)
        # fall back to fp16 standard math attention (they are negligibly cheap).
        if T % 64 != 0 or not _HAS:
            from integration.fused_ops.token_major_attention import _SDPA_CTX
            with _SDPA_CTX():
                a = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # [b,nh,T,hd]
        else:
            qf = q.reshape(BH, T, hd).contiguous()
            kf = k.reshape(BH, T, hd).contiguous()
            vf = v.reshape(BH, T, hd).contiguous()
            a = self._qattn(qf, kf, vf, T, BH).reshape(b, nh, T, hd)     # [b,nh,T,hd]
        a = a.transpose(1, 2).reshape(b, T, c)
        out_tok = x_in_tok + self.proj(a)
        return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)


def convert_attention_to_quantized_std(module, *, bits=8, verbose=False):
    """Replace AttentionBlock instances with the quantized standard-attention block."""
    if not (_HAS and _HAS_ATTN):
        raise RuntimeError("quantized std-attention kernels unavailable")
    n = 0
    for name, child in list(module.named_children()):
        if AttentionBlock is not None and isinstance(child, AttentionBlock) \
                and isinstance(child.attention, QKVAttentionLegacy):
            setattr(module, name, QuantizedStandardAttentionBlock(child, bits=bits).to(
                next(child.parameters()).device))
            n += 1
        else:
            n += convert_attention_to_quantized_std(child, bits=bits, verbose=verbose)
    return n
