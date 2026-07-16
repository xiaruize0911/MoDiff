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
    _HAS_FUSED_QKV = (hasattr(_mc, "gemm_w8a8_out_int8") and hasattr(_mc, "gemm_w4a4_out_int8")
                      and hasattr(_mc, "transpose_qkv_int8"))
except Exception:
    _mc = None
    _HAS_FLASH_INT8 = False
    _HAS_QUANT_QKV = False
    _HAS_FUSED_QKV = False

try:
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, QKVAttentionLegacy
    _HAS_ATTN = True
except Exception:
    _HAS_ATTN = False
    AttentionBlock = QKVAttentionLegacy = None

# int8 flash only where it pays: the memory win is ~entirely the largest-T block
# (its T×T score matrix dwarfs the others), and that block is where the int8 kernel
# already matches fp16 SDPA speed. Small-T blocks would cost speed for ~no memory
# benefit, so default the threshold to 512 (T=1024 block only). Override to 64 to
# quantize all attention blocks.
_FLASH_MIN_T = int(os.environ.get("MODIFF_FLASH_MIN_T", "512"))

# Fused qkv-int-output -> flash path (skips the fp16 round-trip): 0=off, 8=W8A8, 4=W4A4.
# The qkv Linear emits int8 directly (gemm_w{8a8,4a4}_out_int8), a light int8 transpose
# reorders to head-major, and the int8 flash consumes it. Scales are calibrated static
# (Q/K per-tensor, V per-channel-over-T) over the first MODIFF_QKV_CALIB_STEPS forwards.
_QKV_FUSED_BITS = int(os.environ.get("MODIFF_QKV_FLASH_FUSED", "0"))
_QKV_CALIB_STEPS = int(os.environ.get("MODIFF_QKV_CALIB_STEPS", "8"))


def _pack_int4_rows(wq):  # int8 values in [-7,7], [N,K] -> packed [N,K/2]
    lo = wq[:, 0::2] & 0xF
    hi = wq[:, 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8).contiguous()


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
        env_pb = os.environ.get("MODIFF_ATTN_PROJ_BITS")
        self.proj_bits = int(env_pb) if env_pb is not None else proj_bits
        self.modiff = modiff
        if os.environ.get("MODIFF_DISABLE_FLASH_INT8") == "1" or not _HAS_FLASH_INT8:
            self.score_bits = 16
        # int4 weight-only projections: snap qkv/proj weights onto the int4 grid
        # (per-output-channel symmetric). Quality prerequisite for any int4-proj
        # runtime path (which needs an AWQ-style dequant-in-GEMM kernel); here it
        # measures the accuracy impact and yields 4x-smaller proj weights.
        if self.proj_bits == 4:
            with torch.no_grad():
                for lin in (self.qkv, self.proj):
                    w = lin.weight
                    s = (w.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-8)
                    lin.weight.copy_((torch.round(w / s).clamp_(-7, 7) * s).to(w.dtype))

        # Fused qkv-int-output -> flash state (populated lazily on first eligible forward).
        self._qkv_fused_bits = int(os.environ.get("MODIFF_QKV_FLASH_FUSED", "0"))
        self._fused_ready = False      # qkv weight quantized
        self._fused_frozen = False     # static scales calibrated
        self._calib_n = 0
        self._amax_in = 0.0
        self._amax_q = 0.0
        self._amax_k = 0.0
        self._amax_v = None            # [nh,hd] running absmax over tokens
        self._flash_scale_cache = {}

    def _quant_flash_packed(self, qkv, nh, hd):
        """qkv: [B,T,nh,3,hd] fp16 (packed) -> [B,nh,T,hd] fp16 via fused int8 flash.
        The fused quantize reads the packed qkv directly (no transpose/contiguous)."""
        hd_pad = (hd + 31) // 32 * 32
        qi, ki, vi, sq, sk, scv = _mc.quantize_qkv_int8(qkv, nh, hd_pad)
        return _mc.flash_attn_int8(qi, ki, vi, sq, sk, scv, 1.0 / math.sqrt(hd))

    # ---- fused qkv-int-output -> flash path (skips the fp16 round-trip) ----
    def _ensure_fused_qkv_weights(self, bits):
        if self._fused_ready:
            return
        Q = 127 if bits == 8 else 7
        with torch.no_grad():
            W = self.qkv.weight.data.float()                    # [3C, C]
            wamax = W.abs().amax(dim=1).clamp_min(1e-8)          # [3C]
            self._qkv_wscale = (wamax / Q).float().contiguous()
            wq = torch.round(W / (wamax / Q)[:, None]).clamp_(-Q, Q).to(torch.int8)
            self._qkv_wint = wq.contiguous() if bits == 8 else _pack_int4_rows(wq)
            self._qkv_bias = (self.qkv.bias.data.float().contiguous()
                              if self.qkv.bias is not None
                              else torch.empty(0, device=W.device, dtype=torch.float32))
        self._fused_ready = True

    def _calib_update(self, xn_tok, qkv_fp, nh, hd):
        with torch.no_grad():
            self._amax_in = max(self._amax_in, float(xn_tok.abs().max()))
            qv = qkv_fp.reshape(-1, nh, 3, hd).float()
            self._amax_q = max(self._amax_q, float(qv[:, :, 0, :].abs().max()))
            self._amax_k = max(self._amax_k, float(qv[:, :, 1, :].abs().max()))
            v_amax = qv[:, :, 2, :].abs().amax(dim=0)            # [nh,hd]
            self._amax_v = v_amax if self._amax_v is None else torch.maximum(self._amax_v, v_amax)
        self._calib_n += 1
        if self._calib_n >= _QKV_CALIB_STEPS:
            self._freeze_fused(nh, hd)

    def _freeze_fused(self, nh, hd):
        Q = 127 if self._qkv_fused_bits == 8 else 7
        dev = self._qkv_wscale.device
        self._qkv_ascale = max(self._amax_in / Q, 1e-8)
        # oscale [3C] = 127/absmax per output column (Q/K per-tensor, V per-channel).
        osc = torch.empty(nh, 3, hd, device=dev, dtype=torch.float32)
        osc[:, 0, :] = 127.0 / max(self._amax_q, 1e-8)
        osc[:, 1, :] = 127.0 / max(self._amax_k, 1e-8)
        osc[:, 2, :] = 127.0 / self._amax_v.clamp_min(1e-8).to(dev)
        self._oscale = osc.reshape(-1).contiguous()
        # flash dequant scales (= absmax/127); broadcast to [B,nh,T]/[B,nh,hd] on demand.
        self._sq_val = max(self._amax_q, 1e-8) / 127.0
        self._sk_val = max(self._amax_k, 1e-8) / 127.0
        self._sv_vec = (self._amax_v.clamp_min(1e-8) / 127.0).to(dev).float().contiguous()  # [nh,hd]
        self._fused_frozen = True

    def _flash_scales(self, B, nh, T, hd, dev):
        c = self._flash_scale_cache.get((B, T))
        if c is None:
            sq = torch.full((B, nh, T), self._sq_val, device=dev, dtype=torch.float32)
            sk = torch.full((B, nh, T), self._sk_val, device=dev, dtype=torch.float32)
            sv = self._sv_vec.view(1, nh, hd).expand(B, nh, hd).contiguous()
            c = (sq, sk, sv)
            self._flash_scale_cache[(B, T)] = c
        return c

    def _fused_qkv_flash(self, x, nh, hd, bits):
        """GN -> quantize -> gemm_*_out_int8 (packed int8 qkv) -> int8 transpose -> flash.
        During calibration (first _QKV_CALIB_STEPS) runs the fp16 path and records absmax."""
        b, cch, H, W = x.shape
        T = H * W
        w, bnorm = self._gn_params(x.dtype)
        xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
        xn_tok = xn.permute(0, 2, 3, 1).reshape(b * T, cch)
        self._ensure_fused_qkv_weights(bits)
        if not self._fused_frozen:
            qkv_fp = self.qkv(xn_tok.to(x.dtype)).view(b, T, nh, 3, hd)
            self._calib_update(xn_tok, qkv_fp, nh, hd)
            return self._quant_flash_packed(qkv_fp, nh, hd)     # correct fp16 output while calibrating
        a_scale = self._qkv_ascale
        xf = xn_tok.half().contiguous()
        if bits == 8:
            xq = _mc.quantize_act_int8(xf, a_scale)
            qkv_i8 = _mc.gemm_w8a8_out_int8(xq, self._qkv_wint, self._qkv_wscale, a_scale,
                                            self._oscale, self._qkv_bias)
        else:
            xq = _mc.quantize_act_int4_pack(xf, a_scale)
            qkv_i8 = _mc.gemm_w4a4_out_int8(xq, self._qkv_wint, self._qkv_wscale, a_scale,
                                            cch, self._oscale, self._qkv_bias)
        hd_pad = (hd + 31) // 32 * 32
        qi, ki, vi = _mc.transpose_qkv_int8(qkv_i8.view(b, T, nh, 3, hd), nh, hd_pad)
        sq, sk, sv = self._flash_scales(b, nh, T, hd, qi.device)
        return _mc.flash_attn_int8(qi, ki, vi, sq, sk, sv, 1.0 / math.sqrt(hd))

    def _quant_flash(self, q, k, v):
        """PyTorch-fallback quantize path (q,k,v: [B,H,T,hd] fp16)."""
        B, Hh, T, hd = q.shape
        hd_pad = (hd + 31) // 32 * 32

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

        # ---- fused qkv-int-output -> flash path (skips the fp16 qkv round-trip) ----
        if (self._qkv_fused_bits in (4, 8) and self.score_bits == 8 and T >= _FLASH_MIN_T
                and _HAS_FUSED_QKV and _HAS_FLASH_INT8 and x.dtype == torch.float16):
            a = self._fused_qkv_flash(x, nh, hd, self._qkv_fused_bits)   # [b,nh,T,hd]
            a = a.transpose(1, 2).reshape(b, T, c)
            out_tok = x_in_tok + self.proj(a)
            return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)

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

        # ---- score path ----
        if self.score_bits == 8 and T >= _FLASH_MIN_T and _HAS_QUANT_QKV:
            # fused quantize reads packed qkv directly (no transpose/contiguous)
            a = self._quant_flash_packed(qkv, nh, hd)                      # [b,nh,T,hd]
        else:
            q, k, v = qkv.unbind(3)                                        # each [b,T,nh,hd]
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # [b,nh,T,hd]
            if self.score_bits == 8 and T >= _FLASH_MIN_T:
                a = self._quant_flash(q.contiguous(), k.contiguous(), v.contiguous())
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
