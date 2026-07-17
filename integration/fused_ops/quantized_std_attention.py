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

    def __init__(self, orig, *, bits=8, static=False):
        super().__init__(orig)
        assert bits in (4, 8)
        self.bits = bits
        self.Q = 127 if bits == 8 else 7
        self.hp_qk = (self.head_dim + 31) // 32 * 32 if bits == 8 else (self.head_dim + 63) // 64 * 64
        self.hp_av = (self.head_dim + 63) // 64 * 64          # AV N tile needs %64
        # STATIC path: calibrate per-tensor Q/K scale, per-channel V scale, and a single softmax
        # constant c over the first _calib_steps forwards, then freeze (no runtime reductions). The
        # softmax c only sets the int8/int4 P grid (softmax is shift-invariant), so a single c is
        # lossy for quantized attention (row maxes vary) -- accepted here for MoDiff to compensate.
        self.static = bool(static)
        self._calib_steps = int(os.environ.get("MODIFF_ATTN_CALIB_STEPS", "8"))
        self._frozen = not self.static           # dynamic mode needs no calibration
        self._cn = 0
        self._aq = 0.0; self._ak = 0.0; self._av = None; self._cacc = 0.0
        self._sq_c = None; self._sk_c = None; self._sv_vec = None; self._c = None
        # int8-output qkv->attention fusion (prototype, static W8A8 only): emit int8 qkv directly
        # (gemm_w8a8_out_int8) and consume it in the attention quantize (quantize_attn_qkv_from_i8),
        # skipping the fp16 qkv round-trip + reshape copy. Weights/scales frozen with the attn calib.
        self.fuse_qkv_i8 = (self.static and self.bits == 8
                            and os.environ.get("MODIFF_FUSE_QKV_I8") == "1")
        self._fq_frozen = False
        self._xn_amax = 0.0; self._qcol_amax = None
        self._qkv_wq = None; self._qkv_ws = None; self._a_scale = None
        self._oscale = None; self._qbias = None

    def _fq_accum(self, xn_flat, qkv_flat):
        """Accumulate calibration for the int8-output qkv fusion: per-tensor activation absmax (xn)
        and per-output-column absmax of the fp16 qkv (-> oscale)."""
        self._xn_amax = max(self._xn_amax, float(xn_flat.abs().max()))
        col = qkv_flat.detach().float().abs().amax(0)                    # [3C]
        self._qcol_amax = col if self._qcol_amax is None else torch.maximum(self._qcol_amax, col)

    def _freeze_fq(self):
        W = self.qkv.weight.detach().float()                            # [3C, c]
        ws = (W.abs().amax(1).clamp_min(1e-8) / 127.0)
        self._qkv_wq = torch.round(W / ws.unsqueeze(1)).clamp(-127, 127).to(torch.int8).contiguous()
        self._qkv_ws = ws.float().contiguous()
        self._a_scale = (self._xn_amax / 127.0) or 1e-8
        self._oscale = (127.0 / self._qcol_amax.clamp_min(1e-6)).float().contiguous()
        bb = self.qkv.bias
        self._qbias = (bb.detach().float().contiguous() if bb is not None
                       else torch.empty(0, device=W.device, dtype=torch.float32))
        self._fq_frozen = True

    def _qtok(self, x, hd_pad):
        """per-token int8/int4 over the last dim; pad hd->hd_pad. x:[BH,T,hd] -> (xi, scale[BH,T])."""
        sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / self.Q
        xi = torch.round(x / sc).clamp_(-self.Q, self.Q).to(torch.int8)
        xi = F.pad(xi, (0, hd_pad - x.shape[-1])).contiguous()
        if self.bits == 4:
            xi = _pack_last(xi)
        return xi, sc.squeeze(-1).float().contiguous()

    def _calib_update(self, q, k, v, S):
        """Accumulate static-scale calibration stats; freeze after _calib_steps forwards."""
        self._aq = max(self._aq, q.abs().max().item())
        self._ak = max(self._ak, k.abs().max().item())
        av = v.abs().amax(dim=(0, 1)).float()                 # per-channel over (BH,T) -> [hd]
        self._av = av if self._av is None else torch.maximum(self._av, av)
        self._cacc += S.float().amax(-1).mean().item()        # mean per-row max -> c
        self._cn += 1
        if self._cn >= self._calib_steps:
            Qm = float(self.Q)
            self._sq_c = (self._aq / Qm) or 1e-8
            self._sk_c = (self._ak / Qm) or 1e-8
            sv = torch.ones(self.hp_av, device=self._av.device, dtype=torch.float32)
            sv[:self.head_dim] = (self._av / Qm).clamp_min(1e-8)
            self._sv_vec = sv
            self._c = self._cacc / max(self._cn, 1)
            self._frozen = True

    def _qattn(self, q, k, v, T, BH):
        """q,k,v: [BH,T,hd] fp16 -> attention output [BH,T,hd] fp16. Fully-fused quantized standard
        attention. Static (frozen) path uses calibrated scales + static-c softmax (no runtime
        reductions); dynamic path (also used while calibrating) computes per-token/per-channel/
        per-row stats at runtime."""
        hd = self.head_dim
        scale = 1.0 / math.sqrt(hd)
        if self.static and self._frozen:
            qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_static(
                q, k, v, self.hp_qk, self.hp_av, self.bits, self._sq_c, self._sk_c, self._sv_vec)
            if self.bits == 8:
                S = _mc.attn_qk_int8(qi, ki, sq, sk, scale)
                P, sp = _mc.attn_softmax_requant_static(S, self._c)
                O = _mc.attn_av_int8(P, vt, sp, sv)
            else:
                S = _mc.attn_qk_int4(qi, ki, self.hp_qk, sq, sk, scale)
                P, sp = _mc.attn_softmax_requant4_static(S, self._c)
                O = _mc.attn_av_int4(P, vt, sp, sv, T)
            return O[:, :, :hd]
        # dynamic (and calibration) path
        qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv(q, k, v, self.hp_qk, self.hp_av, self.bits)
        if self.bits == 8:
            S = _mc.attn_qk_int8(qi, ki, sq, sk, scale)       # fp16 pre-scaled logits [BH,T,T]
            P, sp = _mc.attn_softmax_requant(S)               # int8 P + per-row sp
            O = _mc.attn_av_int8(P, vt, sp, sv)
        else:
            S = _mc.attn_qk_int4(qi, ki, self.hp_qk, sq, sk, scale)
            P, sp = _mc.attn_softmax_requant4(S)
            O = _mc.attn_av_int4(P, vt, sp, sv, T)
        if self.static and not self._frozen:
            self._calib_update(q, k, v, S)
        return O[:, :, :hd]                                   # drop hd_pad

    def forward(self, x):
        b, c, H, W = x.shape
        T = H * W
        nh, hd = self.num_heads, self.head_dim
        BH = b * nh
        x_in_tok = x.permute(0, 2, 3, 1).reshape(b, T, c)
        fq_ok = (self.fuse_qkv_i8 and _HAS and x.dtype == torch.float16
                 and T % 64 == 0 and T >= 256 and (c % 8) == 0)
        # ---- FUSED int8-output qkv path (frozen): GN -> int8-out qkv GEMM -> attn quantize-from-int8
        # (no fp16 qkv materialization, no reshape copy). ----
        if fq_ok and self._fq_frozen:
            w, bnorm = self._gn_params(x.dtype)
            xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
            xn_flat = xn.permute(0, 2, 3, 1).reshape(b * T, c).contiguous()
            xq = _mc.quantize_act_int8(xn_flat, self._a_scale)
            qkv_i8 = _mc.gemm_w8a8_out_int8(xq, self._qkv_wq, self._qkv_ws, self._a_scale, self._oscale, self._qbias)
            qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_from_i8(qkv_i8, self._oscale, nh, T, self.hp_qk, self.hp_av)
            S = _mc.attn_qk_int8(qi, ki, sq, sk, self.scale)
            P, sp = _mc.attn_softmax_requant_static(S, self._c)
            O = _mc.attn_av_int8(P, vt, sp, sv)[:, :, :hd]
            a = O.reshape(b, nh, T, hd).transpose(1, 2).reshape(b, T, c)
            out_tok = x_in_tok + self.proj(a)
            return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)
        # ---- normal fp16-qkv path (also gathers the fusion calibration while not yet frozen) ----
        from integration.fused_ops.token_major_attention import _HAS_FUSED_GN_QKV, _FUSE_TILE_M, _FUSE_SHIFT
        need_fq_calib = fq_ok and not self._fq_frozen           # force GN+Linear to expose xn/qkv
        if (not need_fq_calib and self._fuse_gn_qkv and _HAS_FUSED_GN_QKV and x.dtype == torch.float16
                and (T % _FUSE_TILE_M) == 0 and (c % 8) == 0):
            self._ensure_fused()
            qkv_img = _mc.fused_gn_qkv(x, self._fused_conv_w, self._fused_epi_bias,
                                       self.norm.num_groups, self.norm.eps, _FUSE_SHIFT)
            qkv = qkv_img.permute(0, 2, 3, 1).reshape(b, T, nh, 3, hd)
        else:
            w, bnorm = self._gn_params(x.dtype)
            xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
            xn_tok = xn.permute(0, 2, 3, 1).reshape(b, T, c)
            qkv_flat = self.qkv(xn_tok)
            if need_fq_calib:
                self._fq_accum(xn_tok.reshape(b * T, c), qkv_flat.reshape(b * T, 3 * c))
            qkv = qkv_flat.view(b, T, nh, 3, hd)
        q, k, v = qkv.unbind(3)                                          # [b,T,nh,hd]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)   # [b,nh,T,hd]
        # The batched int GEMMs need T%64==0 (N tile); small-T blocks (T<256, e.g. 4x4/8x8)
        # fall back to fp16 standard math attention — quant loses there (measured 0.44x at
        # T=64) and they are negligibly cheap anyway. int8/int4 win 2.2-2.4x on the big T=1024.
        if T % 64 != 0 or T < 256 or not _HAS:
            from integration.fused_ops.token_major_attention import _SDPA_CTX
            with _SDPA_CTX():
                a = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # [b,nh,T,hd]
        else:
            qf = q.reshape(BH, T, hd).contiguous()
            kf = k.reshape(BH, T, hd).contiguous()
            vf = v.reshape(BH, T, hd).contiguous()
            a = self._qattn(qf, kf, vf, T, BH).reshape(b, nh, T, hd)     # [b,nh,T,hd]
        # freeze the fusion weights/scales once the attention calibration has frozen
        if need_fq_calib and self._frozen and not self._fq_frozen:
            self._freeze_fq()
        a = a.transpose(1, 2).reshape(b, T, c)
        out_tok = x_in_tok + self.proj(a)
        return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)


def convert_attention_to_quantized_std(module, *, bits=8, static=False, verbose=False):
    """Replace AttentionBlock instances with the quantized standard-attention block."""
    if not (_HAS and _HAS_ATTN):
        raise RuntimeError("quantized std-attention kernels unavailable")
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
