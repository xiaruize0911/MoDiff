"""Fused (flash) quantized STANDARD attention (W8A8 / W4A4).

GroupNorm -> qkv(fp16) -> quantize Q/K/V -> FUSED flash attention (QKᵀ + online softmax + AV
in one kernel, scores kept in SRAM) -> proj -> residual. This is the SOLE quantized-attention
path; the materialized QKᵀ/softmax/AV int-GEMM path (attn_qk_int8/int4, attn_softmax_requant*,
attn_av_int8/int4) was removed. Eligible blocks (head_dim<=48, T%64==0) run the flash kernel;
ineligible blocks (hd>48 or T%64!=0) fall back to fp16 SDPA math attention.

bits=8 (W8A8) is quality-safe (adds ~0.004 rel-L2 over fp16 attention). bits=4 uses int4-packed
Q/K with int8 (transposed) V and int8 online-softmax P — there is no native int4 flash kernel, so
"int4" here refers to the conv/linear precision; the attention score path is int4 Q/K · int8 V.
Drop-in for AttentionBlock; extends TokenMajorAttentionBlock (reuses its GN + qkv/proj Linears,
weight copy, head config, GN->qkv fusion). Scales self-calibrate over the first _calib_steps
forwards, then freeze to a static single-pass quantize (no runtime absmax reduction).
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
    _HAS_FLASH = hasattr(_mc, "flash_attn_int8_vt") and hasattr(_mc, "flash_attn_int4_vt")
    # packed-qkv quantize (reads interleaved [b,T,nh,3,hd] -> no fp16 transpose copy)
    _HAS_PACKED = hasattr(_mc, "quantize_attn_qkv_packed") and hasattr(_mc, "quantize_attn_qkv_packed_static")
    # fused GN + activation-quantize -> int8 (reused from the conv path) for the qkv W8A8 GEMM input
    _HAS_GN_QUANT = hasattr(_mc, "group_norm_silu_quantize_nhwc")
    # fused GN + activation-quantize+pack -> int4 for the qkv W4A4 GEMM input (int4 parity with GN_QUANT)
    _HAS_GN_PACK = hasattr(_mc, "group_norm_silu_quantize_pack_nhwc")
    # fused transpose+reshape+quantize of the attention output -> int8 [b*T,c] for the proj W8A8 GEMM
    _HAS_ATTN_OUT = hasattr(_mc, "quantize_attn_out_int8")
    # int4 variant: transpose+reshape+int4-quantize+pack -> [b*T,c/2] for the proj W4A4 GEMM
    _HAS_ATTN_OUT_I4 = hasattr(_mc, "quantize_attn_out_int4_pack")
    # flash epilogue writes int8 token-major directly (folds the attention-output quantize into flash)
    _HAS_FLASH_OUT_I8 = hasattr(_mc, "flash_attn_int8_vt_out_i8")
except Exception:
    _mc = None
    _HAS_FLASH = False
    _HAS_PACKED = False
    _HAS_GN_QUANT = False
    _HAS_GN_PACK = False
    _HAS_ATTN_OUT = False
    _HAS_ATTN_OUT_I4 = False
    _HAS_FLASH_OUT_I8 = False

try:
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, QKVAttentionLegacy
    _HAS_ATTN = True
except Exception:
    _HAS_ATTN = False
    AttentionBlock = QKVAttentionLegacy = None


class QuantizedStandardAttentionBlock(TokenMajorAttentionBlock):
    """TokenMajorAttentionBlock with the score path (QKᵀ/softmax/AV) run by the fused int8/int4
    flash kernel (scores in SRAM). bits=8 = W8A8 (quality-safe); bits=4 = int4-packed Q/K + int8 V/P."""

    def __init__(self, orig, *, bits=8, static=False):
        super().__init__(orig)
        assert bits in (4, 8)
        self.bits = bits
        # static kept for API compat; the flash path always self-calibrates over the first
        # _calib_steps forwards then freezes to a static single-pass quantize.
        self.static = bool(static)
        self._calib_steps = int(os.environ.get("MODIFF_ATTN_CALIB_STEPS", "8"))

    def _flash_quant_attn(self, q, k, v, out_i8_inv_scale=None):
        """FUSED int8/int4 flash attention. q,k,v: [b,nh,T,hd] fp16 -> [b,nh,T,hd] fp16.
        Dynamic per-token Q/K + per-channel V quantize, then the fused flash_attn kernel
        (QKᵀ + online softmax + AV in one kernel, scores in SRAM). self.bits selects int8/int4.
        If out_i8_inv_scale is set (int8 path only), the flash epilogue writes int8 token-major
        [b*T, nh*hd] quantized by that scalar (= 1/proj.a_scale) — folds the attention-output
        quantize into flash so proj can consume it directly (4-kernel plan Step 1)."""
        b, nh, T, hd = q.shape
        BH = b * nh
        qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
        if self.bits == 8:
            hd_pad = ((hd + 31) // 32) * 32
            # Task 1: after a short self-calibration, use the STATIC single-pass quantize (no runtime
            # amax reduction -> ~1.3x faster than dynamic; rel-L2 ~0.03). Falls back to dynamic while
            # calibrating. (Full fuse into the qkv-GEMM epilogue is not possible: the per-token amax
            # can't be computed in a GEMM epilogue, so this static single-pass is the floor.)
            if getattr(self, "_fq_frozen2", False):
                qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_static(qm, km, vm, hd_pad, hd_pad, 8,
                                                                      self._fq_sqc, self._fq_skc, self._fq_svv)
            else:
                qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv(qm, km, vm, hd_pad, hd_pad, 8)
                self._fq_aq = max(getattr(self, "_fq_aq", 0.0), q.abs().max().item())
                self._fq_ak = max(getattr(self, "_fq_ak", 0.0), k.abs().max().item())
                avc = v.abs().amax(dim=(0, 1, 2)).float()
                self._fq_av = avc if getattr(self, "_fq_av", None) is None else torch.maximum(self._fq_av, avc)
                self._fq_n = getattr(self, "_fq_n", 0) + 1
                if self._fq_n >= self._calib_steps:
                    self._fq_sqc = self._fq_aq / 127.0; self._fq_skc = self._fq_ak / 127.0
                    svv = torch.ones(hd_pad, device=v.device); svv[:hd] = (self._fq_av / 127.0).clamp_min(1e-8)
                    self._fq_svv = svv.contiguous(); self._fq_frozen2 = True
            qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad)
            vt = vt.view(b, nh, hd_pad, T)                    # already transposed -> pass directly (no copy)
            sq = sq.view(b, nh, T).contiguous(); sk = sk.view(b, nh, T).contiguous()
            sv = sv[..., :hd].contiguous().view(b, nh, hd)
            if out_i8_inv_scale is not None:                                   # int8 token-major [b*T, nh*hd]
                return _mc.flash_attn_int8_vt_out_i8(qi, ki, vt, sq, sk, sv, self.scale, float(out_i8_inv_scale))
            return _mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, self.scale)   # [b,nh,T,hd], V pre-transposed
        # Task 2: int4 fused. ONE-PASS mixed quantize (quantize_attn_qkv_i4qk_i8v): int4-packed Q/K
        # (matches the flash kernel) + int8 transposed V (flash uses int8 PV) in a single sweep of q/k/v
        # -> no eager nibble-pack, no wasted int4-V, no double V quantize. Feeds flash_attn_int4_vt directly.
        hdp4, hdp_v = 64, ((hd + 31) // 32) * 32
        if getattr(self, "_fq4_frozen", False):   # static single-pass (calibrated) — no runtime amax
            q4, k4, vt, sq4, sk4, sv = _mc.quantize_attn_qkv_i4qk_i8v_static(
                qm, km, vm, hdp4, hdp_v, self._fq4_sqc, self._fq4_skc, self._fq4_svv)
        else:
            q4, k4, vt, sq4, sk4, sv = _mc.quantize_attn_qkv_i4qk_i8v(qm, km, vm, hdp4, hdp_v)
            self._fq4_aq = max(getattr(self, "_fq4_aq", 0.0), q.abs().max().item())
            self._fq4_ak = max(getattr(self, "_fq4_ak", 0.0), k.abs().max().item())
            avc = v.abs().amax(dim=(0, 1, 2)).float()
            self._fq4_av = avc if getattr(self, "_fq4_av", None) is None else torch.maximum(self._fq4_av, avc)
            self._fq4_n = getattr(self, "_fq4_n", 0) + 1
            if self._fq4_n >= self._calib_steps:
                self._fq4_sqc = self._fq4_aq / 7.0; self._fq4_skc = self._fq4_ak / 7.0   # int4 Q/K
                svv = torch.ones(hdp_v, device=v.device); svv[:hd] = (self._fq4_av / 127.0).clamp_min(1e-8)  # int8 V
                self._fq4_svv = svv.contiguous(); self._fq4_frozen = True
        q4 = q4.view(b, nh, T, -1); k4 = k4.view(b, nh, T, -1)
        vt = vt.view(b, nh, hdp_v, T)                          # int8 V, already transposed
        sq4 = sq4.view(b, nh, T).contiguous(); sk4 = sk4.view(b, nh, T).contiguous()
        sv = sv[..., :hd].contiguous().view(b, nh, hd)
        return _mc.flash_attn_int4_vt(q4, k4, vt, sq4, sk4, sv, hdp4, self.scale)  # [b,nh,T,hd]

    def _flash_quant_attn_packed(self, qkv, out_i8_inv_scale=None):
        """Same as _flash_quant_attn but reads the interleaved qkv [b,T,nh,3,hd] DIRECTLY via the
        packed quantize kernels -> drops the ~1.2 GB/step fp16 q/k/v.transpose().contiguous() copy
        (the largest 'glue' cost). Output identical: [b,nh,T,hd] fp16 (or int8 [b*T,nh*hd] when
        out_i8_inv_scale is set, int8 path only — folds the attention-output quantize into flash)."""
        b, T, nh, _, hd = qkv.shape
        qkv = qkv.contiguous()
        qv, kv, vv = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :]  # strided views (no copy)
        if self.bits == 8:
            hd_pad = ((hd + 31) // 32) * 32
            if getattr(self, "_fq_frozen2", False):
                qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_packed_static(
                    qkv, nh, T, hd, hd_pad, hd_pad, 8, self._fq_sqc, self._fq_skc, self._fq_svv)
            else:
                qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_packed(qkv, nh, T, hd, hd_pad, hd_pad, 8)
                self._fq_aq = max(getattr(self, "_fq_aq", 0.0), qv.abs().max().item())
                self._fq_ak = max(getattr(self, "_fq_ak", 0.0), kv.abs().max().item())
                avc = vv.abs().amax(dim=(0, 1, 2)).float()
                self._fq_av = avc if getattr(self, "_fq_av", None) is None else torch.maximum(self._fq_av, avc)
                self._fq_n = getattr(self, "_fq_n", 0) + 1
                if self._fq_n >= self._calib_steps:
                    self._fq_sqc = self._fq_aq / 127.0; self._fq_skc = self._fq_ak / 127.0
                    svv = torch.ones(hd_pad, device=qkv.device); svv[:hd] = (self._fq_av / 127.0).clamp_min(1e-8)
                    self._fq_svv = svv.contiguous(); self._fq_frozen2 = True
            qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad); vt = vt.view(b, nh, hd_pad, T)
            sq = sq.view(b, nh, T); sk = sk.view(b, nh, T); sv = sv[..., :hd].contiguous().view(b, nh, hd)
            if out_i8_inv_scale is not None:                                   # int8 token-major [b*T, nh*hd]
                return _mc.flash_attn_int8_vt_out_i8(qi, ki, vt, sq, sk, sv, self.scale, float(out_i8_inv_scale))
            return _mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, self.scale)
        # int4 (int4 Q/K + int8 V), packed
        hdp4, hdp_v = 64, ((hd + 31) // 32) * 32
        if getattr(self, "_fq4_frozen", False):
            q4, k4, vt, sq4, sk4, sv = _mc.quantize_attn_qkv_packed_static(
                qkv, nh, T, hd, hdp4, hdp_v, 4, self._fq4_sqc, self._fq4_skc, self._fq4_svv)
        else:
            q4, k4, vt, sq4, sk4, sv = _mc.quantize_attn_qkv_packed(qkv, nh, T, hd, hdp4, hdp_v, 4)
            self._fq4_aq = max(getattr(self, "_fq4_aq", 0.0), qv.abs().max().item())
            self._fq4_ak = max(getattr(self, "_fq4_ak", 0.0), kv.abs().max().item())
            avc = vv.abs().amax(dim=(0, 1, 2)).float()
            self._fq4_av = avc if getattr(self, "_fq4_av", None) is None else torch.maximum(self._fq4_av, avc)
            self._fq4_n = getattr(self, "_fq4_n", 0) + 1
            if self._fq4_n >= self._calib_steps:
                self._fq4_sqc = self._fq4_aq / 7.0; self._fq4_skc = self._fq4_ak / 7.0
                svv = torch.ones(hdp_v, device=qkv.device); svv[:hd] = (self._fq4_av / 127.0).clamp_min(1e-8)
                self._fq4_svv = svv.contiguous(); self._fq4_frozen = True
        q4 = q4.view(b, nh, T, -1); k4 = k4.view(b, nh, T, -1); vt = vt.view(b, nh, hdp_v, T)
        sq4 = sq4.view(b, nh, T); sk4 = sk4.view(b, nh, T); sv = sv[..., :hd].contiguous().view(b, nh, hd)
        return _mc.flash_attn_int4_vt(q4, k4, vt, sq4, sk4, sv, hdp4, self.scale)

    def forward(self, x):
        b, c, H, W = x.shape
        T = H * W
        nh, hd = self.num_heads, self.head_dim
        x_in_tok = x.permute(0, 2, 3, 1).reshape(b, T, c)
        from integration.fused_ops.token_major_attention import _HAS_FUSED_GN_QKV, _FUSE_TILE_M, _FUSE_SHIFT
        # ---- GN -> qkv: fused GN->qkv (fp16) where eligible, else GroupNorm + qkv Linear ----
        if (self._fuse_gn_qkv and _HAS_FUSED_GN_QKV and x.dtype == torch.float16
                and (T % _FUSE_TILE_M) == 0 and (c % 8) == 0):
            self._ensure_fused()
            qkv_img = _mc.fused_gn_qkv(x, self._fused_conv_w, self._fused_epi_bias,
                                       self.norm.num_groups, self.norm.eps, _FUSE_SHIFT)
            qkv = qkv_img.permute(0, 2, 3, 1).reshape(b, T, nh, 3, hd)
        elif (os.environ.get("MODIFF_FUSE_GN_QKV_I8", "1") != "0" and _HAS_GN_QUANT and x.dtype == torch.float16
                and (c % self.norm.num_groups == 0)
                and hasattr(self.qkv, "can_from_int8") and self.qkv.can_from_int8()):
            # FUSED GN -> activation-quantize (int8) -> qkv W8A8 GEMM: the GroupNorm kernel emits the
            # qkv GEMM's int8 input directly (no separate quantize_act_int8 pass). c%64==0 for every
            # attention block, so the int8 [N,c,H,W] channels feed the AWQ GEMM with no K-pad. Mirrors
            # the conv path's group_norm_silu_quantize_nhwc -> forward_from_int8 fusion.
            w, bnorm = self._gn_params(torch.float16)
            a = self.qkv.a_scale
            if getattr(self, "_gnq_a", None) != a:
                self._gnq_scale = torch.tensor([1.0 / a], device=x.device, dtype=torch.float32)  # 127/absmax
                self._gnq_empty16 = x.new_empty(0)
                self._gnq_empty32 = torch.empty(0, device=x.device, dtype=torch.float32)
                self._gnq_a = a
            x_cl = x if x.is_contiguous(memory_format=torch.channels_last) else x.contiguous(memory_format=torch.channels_last)
            q_i8 = _mc.group_norm_silu_quantize_nhwc(x_cl, w, bnorm, self.norm.num_groups, self.norm.eps,
                                                     False, self._gnq_scale, self._gnq_empty32,
                                                     self._gnq_empty16, self._gnq_empty16)
            q_tok = q_i8.permute(0, 2, 3, 1).reshape(b * T, c).contiguous()
            qkv = self.qkv.forward_from_int8(q_tok).view(b, T, nh, 3, hd)
        elif (os.environ.get("MODIFF_FUSE_GN_QKV_I4", "1") != "0" and _HAS_GN_PACK and x.dtype == torch.float16
                and (c % self.norm.num_groups == 0) and (c % 2 == 0)
                and hasattr(self.qkv, "can_from_int4") and self.qkv.can_from_int4()):
            # int4 parity with the int8 branch above: GroupNorm emits the qkv GEMM's PACKED int4 input
            # directly (group_norm_silu_quantize_pack_nhwc), skipping the standalone quantize_act_int4_pack.
            # forward_from_int4 zero-pads the packed activation for the K%128 blocks (C=192 -> 256).
            # Previously int4 attention fell to the fp16-GN + separate-quantize path below.
            w, bnorm = self._gn_params(torch.float16)
            a = self.qkv.a_scale
            if getattr(self, "_gnq4_a", None) != a:
                self._gnq4_scale = torch.tensor([1.0 / a], device=x.device, dtype=torch.float32)  # 7/absmax
                self._gnq4_empty16 = x.new_empty(0)
                self._gnq4_empty32 = torch.empty(0, device=x.device, dtype=torch.float32)
                self._gnq4_a = a
            x_cl = x if x.is_contiguous(memory_format=torch.channels_last) else x.contiguous(memory_format=torch.channels_last)
            q_i4 = _mc.group_norm_silu_quantize_pack_nhwc(x_cl, w, bnorm, self.norm.num_groups, self.norm.eps,
                                                          False, self._gnq4_scale, self._gnq4_empty32,
                                                          self._gnq4_empty16, self._gnq4_empty16)
            q_tok = q_i4.reshape(b * T, c // 2).contiguous()
            qkv = self.qkv.forward_from_int4(q_tok).view(b, T, nh, 3, hd)
        else:
            w, bnorm = self._gn_params(x.dtype)
            xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
            xn_tok = xn.permute(0, 2, 3, 1).reshape(b, T, c)
            qkv = self.qkv(xn_tok).view(b, T, nh, 3, hd)
        # ---- FUSED flash attention (QKᵀ + online softmax + AV in one kernel, scores in SRAM). The
        # flash kernel needs head_dim<=48 and T%64==0; ineligible blocks (the hd=96 / tiny-T blocks)
        # fall back to fp16 SDPA math attention (quant loses there and they are negligibly cheap). ----
        _use_flash = (_HAS_FLASH and self.head_dim <= 48 and (T % 64) == 0)
        # proj activation-quantize fusion (int8 modes, default on): fold the attention-output quantize
        # into the producer so proj consumes int8 directly (4-kernel plan). Two producers:
        #   (a) flash-eligible blocks -> flash writes int8 token-major (its own epilogue), 1 kernel; OR
        #   (b) fp16-SDPA (hd=96) blocks -> quantize_attn_out_int8 folds transpose+quantize.
        _proj_i8 = (os.environ.get("MODIFF_FUSE_PROJ_I8", "1") != "0"
                    and hasattr(self.proj, "can_from_int8") and self.proj.can_from_int8())
        # int4 proj parity: int4 has no flash-out-i8, so fold the attention-output transpose+quantize+pack
        # into one kernel (quantize_attn_out_int4_pack) feeding proj.forward_from_int4 (which pads C192),
        # replacing the standalone a.transpose().reshape() copy + proj's own quantize_act_int4_pack.
        _proj_i4 = (os.environ.get("MODIFF_FUSE_PROJ_I4", "1") != "0" and _HAS_ATTN_OUT_I4
                    and hasattr(self.proj, "can_from_int4") and self.proj.can_from_int4())

        if _use_flash and _proj_i8 and _HAS_FLASH_OUT_I8:
            inv = 1.0 / self.proj.a_scale
            if _HAS_PACKED:
                xq = self._flash_quant_attn_packed(qkv, out_i8_inv_scale=inv)     # int8 [b*T, c]
            else:
                q, k, v = qkv.unbind(3)
                q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
                xq = self._flash_quant_attn(q, k, v, out_i8_inv_scale=inv)        # int8 [b*T, c]
            out_flat = self.proj.forward_from_int8(xq, residual=x_in_tok)         # proj GEMM + bias + residual
            return out_flat.reshape(b, H, W, c).permute(0, 3, 1, 2)

        # Standard path: produce the fp16 attention output, then proj.
        if _use_flash and _HAS_PACKED:
            a = self._flash_quant_attn_packed(qkv)                       # [b,nh,T,hd], fused
        else:
            q, k, v = qkv.unbind(3)                                       # [b,T,nh,hd]
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)   # [b,nh,T,hd]
            if _use_flash:
                a = self._flash_quant_attn(q, k, v)                      # [b,nh,T,hd], fused
            else:
                from integration.fused_ops.token_major_attention import _SDPA_CTX
                with _SDPA_CTX():
                    a = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # [b,nh,T,hd]
        # proj: for the fp16-out (hd=96 SDPA) blocks, fold transpose+quantize via quantize_attn_out_int8;
        # else standard transpose + proj.
        if _proj_i8 and _HAS_ATTN_OUT and a.dtype == torch.float16:
            xq = _mc.quantize_attn_out_int8(a, self.proj.a_scale)        # int8 [b*T, c]
            out_flat = self.proj.forward_from_int8(xq, residual=x_in_tok)  # [b*T, c], residual fused
            return out_flat.reshape(b, H, W, c).permute(0, 3, 1, 2)
        if _proj_i4 and a.dtype == torch.float16:
            xq = _mc.quantize_attn_out_int4_pack(a, self.proj.a_scale)   # int4 packed [b*T, c/2], transpose+quant+pack fused
            out_flat = self.proj.forward_from_int4(xq, residual=x_in_tok)  # [b*T, c], pad(C192)+bias+residual fused
            return out_flat.reshape(b, H, W, c).permute(0, 3, 1, 2)
        a = a.transpose(1, 2).reshape(b, T, c)
        out_tok = (self.proj(a, residual=x_in_tok) if getattr(self.proj, "_use_bias_res", False)
                   else x_in_tok + self.proj(a))                     # fused residual in proj epilogue
        return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)


def convert_attention_to_quantized_std(module, *, bits=8, static=False, verbose=False):
    """Replace AttentionBlock instances with the fused (flash) quantized attention block."""
    if not (_HAS_FLASH and _HAS_ATTN):
        raise RuntimeError("fused flash quantized-attention kernels unavailable")
    n = 0
    for name, child in list(module.named_children()):
        if AttentionBlock is not None and isinstance(child, AttentionBlock) \
                and isinstance(child.attention, QKVAttentionLegacy):
            setattr(module, name, QuantizedStandardAttentionBlock(child, bits=bits, static=static).to(
                next(child.parameters()).device))
            n += 1
        else:
            n += convert_attention_to_quantized_std(child, bits=bits, static=static, verbose=verbose)
    return n
