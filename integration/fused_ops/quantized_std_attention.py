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
    TokenMajorAttentionBlock, _group_norm_silu,
    _HAS_FUSED_ATTN_OUT, _HAS_ATTN_OUT_I4, _QuantLinearWxAx)

try:
    import modiff_cutlass as _mc
    _HAS_FLASH = hasattr(_mc, "flash_attn_int8_vt") and hasattr(_mc, "flash_attn_int4_vt")
    # packed-qkv quantize (reads interleaved [b,T,nh,3,hd] -> no fp16 transpose copy)
    _HAS_PACKED = hasattr(_mc, "quantize_attn_qkv_packed") and hasattr(_mc, "quantize_attn_qkv_packed_static")
except Exception:
    _mc = None
    _HAS_FLASH = False
    _HAS_PACKED = False

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
        # Flash-vs-fp16 gate. The int flash kernel only pays off on large-enough blocks. Measured
        # on the production packed path (A40, b128, quantize_attn_qkv_packed -> flash_attn_int8_vt
        # vs fp16 MATH SDPA): hd24/T1024 2.34x, hd24/T256 1.37x, hd48/T512 1.13x, hd48/T256 1.01x
        # (break-even), hd48/T128 0.86x, hd48/T64 0.64x. So the old head_dim<=48 & T%64==0 gate
        # dragged the small-T blocks (esp. T=64) into a net LOSS. The crossover is 2D in
        # (head_dim, T) and GPU-dependent -- fp16 SDPA's own efficiency also rises with head_dim,
        # so no static T threshold separates winners from losers (hd24/T256 wins 1.37x while
        # hd48/T256 is only break-even). "auto" (default) therefore MEASURES flash vs fp16 SDPA
        # once per block, right after the quantize scales freeze, and caches the winner.
        # MODIFF_FLASH_GATE=on forces flash on every eligible block (old behaviour); =off keeps
        # eligible blocks on fp16 SDPA.
        self._flash_gate = os.environ.get("MODIFF_FLASH_GATE", "auto")
        self._flash_choice = None    # None = undecided; True/False = frozen decision

    def _flash_quant_attn(self, q, k, v):
        """FUSED int8/int4 flash attention. q,k,v: [b,nh,T,hd] fp16 -> [b,nh,T,hd] fp16.
        Dynamic per-token Q/K + per-channel V quantize, then the fused flash_attn kernel
        (QKᵀ + online softmax + AV in one kernel, scores in SRAM). self.bits selects int8/int4."""
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

    def _flash_quant_attn_packed(self, qkv):
        """Same as _flash_quant_attn but reads the interleaved qkv [b,T,nh,3,hd] DIRECTLY via the
        packed quantize kernels -> drops the ~1.2 GB/step fp16 q/k/v.transpose().contiguous() copy
        (the largest 'glue' cost). Output identical: [b,nh,T,hd] fp16."""
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

    def _scores_scales_frozen(self):
        """True once the per-block Q/K/V quantize scales have self-calibrated and frozen to
        the static single-pass path (so a flash timing reflects steady state, not calibration)."""
        return getattr(self, "_fq_frozen2" if self.bits == 8 else "_fq4_frozen", False)

    def _score_path(self, qkv, use_flash):
        """Attention score path -> a=[b,nh,T,hd]. Flash (fused int8/int4) when use_flash, else
        fp16 MATH SDPA. Uses the packed quantize (reads interleaved qkv, no transpose copy) when
        available; otherwise unbinds to strided q/k/v views."""
        if use_flash and _HAS_PACKED:
            return self._flash_quant_attn_packed(qkv)                    # [b,nh,T,hd]
        q, k, v = qkv.unbind(3)                                          # [b,T,nh,hd]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)   # [b,nh,T,hd]
        if use_flash:
            return self._flash_quant_attn(q, k, v)
        from integration.fused_ops.token_major_attention import _SDPA_CTX
        with _SDPA_CTX():
            return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

    def _autotune_flash(self, qkv):
        """One-shot: time the fused flash score path vs fp16 SDPA on this block's actual qkv and
        return True iff flash is faster. Called once, after the quantize scales freeze, so the
        flash timing uses the static single-pass path. The choice is then cached (see _resolve_flash).
        Measuring (vs a hardcoded T threshold) is necessary because the crossover is 2D in
        (head_dim, T) and GPU-dependent."""
        def _t(use_flash, warm=3, iters=10):
            for _ in range(warm): self._score_path(qkv, use_flash)
            torch.cuda.synchronize()
            s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
            for _ in range(iters): self._score_path(qkv, use_flash)
            e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e)
        return _t(True) < _t(False)

    def _resolve_flash(self, qkv, T):
        """Decide whether this block runs the fused flash score path. Base eligibility is the
        flash kernel's constraint (head_dim<=48, T%64==0). Then per MODIFF_FLASH_GATE:
        on -> always flash when eligible (old behaviour); off -> never; auto (default) -> flash
        during scale calibration to capture the scales, then autotune-and-freeze once frozen."""
        eligible = (_HAS_FLASH and self.head_dim <= 48 and (T % 64) == 0)
        if not eligible or self._flash_gate == "off":
            return False
        if self._flash_gate == "on":
            return True
        if self._flash_choice is not None:
            return self._flash_choice
        if not self._scores_scales_frozen():
            return True                     # still calibrating -> run flash to freeze the scales
        self._flash_choice = self._autotune_flash(qkv)
        return self._flash_choice

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
        else:
            w, bnorm = self._gn_params(x.dtype)
            xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
            xn_tok = xn.permute(0, 2, 3, 1).reshape(b, T, c)
            qkv = self.qkv(xn_tok).view(b, T, nh, 3, hd)
        # ---- Score path (QKᵀ + softmax + AV): fused int8/int4 flash (scores in SRAM) when the
        # gate says it's a win for this block; else fp16 MATH SDPA. The flash kernel needs
        # head_dim<=48 and T%64==0; among eligible blocks the small-T ones lose to fp16, so the
        # gate autotunes per block (see _resolve_flash). ----
        a = self._score_path(qkv, self._resolve_flash(qkv, T))           # [b,nh,T,hd] head-major
        out_tok = self._proj_with_residual(a, x_in_tok, b, T, c)         # proj (+bias +residual), fused
        return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)

    def _proj_with_residual(self, a, x_in_tok, b, T, c):
        """proj on the head-major attention output a=[b,nh,T,hd], adding the ResBlock residual
        x_in_tok=[b,T,c]. Fully-fused path (calibrated static int8/int4 proj): fold the head-major
        -> token-major transpose AND proj's activation quantize into quantize_attn_out_int{8,4}
        (so the fp16 attention output is never materialized and proj skips its own quantize pass),
        then run the W8A8/W4A4 GEMM with bias + the skip residual folded into its store epilogue
        (gemm_wXaX_awq_bias_res). Numerically identical to the transpose+reshape -> proj(., residual)
        fallback below, which is used during calibration and for ineligible proj (uncalibrated /
        modiff / int8-output / int4 needing K-pad, e.g. C=192). Kill-switch MODIFF_FUSE_PROJ_QUANT=0
        (inherited from TokenMajorAttentionBlock)."""
        proj = self.proj
        # int8 always has _awqt_K == in_features (C % 64 == 0); int4 C=192 needs K=256 pad, now handled
        # by quantize_attn_out_int4_pack's k_pad zero-fill (so the fp16 F.pad fallback is avoided).
        if (self._fuse_proj_quant and _QuantLinearWxAx is not None and isinstance(proj, _QuantLinearWxAx)
                and getattr(proj, "_use_bias_res", False) and proj.a_scale is not None
                and not getattr(proj, "_calib", False) and proj.bits in (8, 4)
                and ((proj.bits == 8 and _HAS_FUSED_ATTN_OUT) or (proj.bits == 4 and _HAS_ATTN_OUT_I4))):
            res = x_in_tok.reshape(b * T, c).contiguous()
            bias = proj.bias if proj.bias is not None else a.new_empty(0)
            if proj.bits == 8:
                xq = _mc.quantize_attn_out_int8(a, proj.a_scale)           # int8 [b*T,c]: transpose+quantize
                out = _mc.gemm_w8a8_awq_bias_res(xq, proj.qweight, proj.w_scale, proj.a_scale,
                                                 proj.out_features, bias, res)
            else:
                # k_pad = proj._awqt_K: pack to the GEMM's padded K (zero-fills C..K_pad-1), no fp16 F.pad
                xq = _mc.quantize_attn_out_int4_pack(a, proj.a_scale, proj._awqt_K)   # packed int4 [b*T,K_pad/2]
                out = _mc.gemm_w4a4_awq_bias_res(xq, proj.qweight, proj.w_scale, proj.a_scale,
                                                 proj._awqt_K, proj.out_features, bias, res)
            return out.reshape(b, T, c)
        # fallback: materialize the transpose, then proj (own quantize; bias+residual epilogue if available)
        a = a.transpose(1, 2).reshape(b, T, c)
        return proj(a, residual=x_in_tok) if getattr(proj, "_use_bias_res", False) else x_in_tok + proj(a)


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
