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
    _HAS_FLASH = hasattr(_mc, "flash_attn_int8") and hasattr(_mc, "flash_attn_int4")
    # packed-qkv quantize (reads interleaved [b,T,nh,3,hd] -> no fp16 transpose copy at line 175)
    _HAS_PACKED = hasattr(_mc, "quantize_attn_qkv_packed") and hasattr(_mc, "quantize_attn_qkv_packed_static")
except Exception:
    _mc = None
    _HAS = False
    _HAS_FLASH = False
    _HAS_PACKED = False

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
        # int8-SCORE path (bits==8 only): QKᵀ writes int8 scores (attn_qk_int8_s8out) and the softmax
        # reads them with a DYNAMIC per-row max (attn_softmax_requant_s8_dyn) -> halves the T×T score
        # write+read (the memory-bound bottleneck) WITHOUT the static-c quality loss. Needs a per-tensor
        # score scale sS, self-calibrated (score absmax) over the first _calib_steps forwards on the fp16-S
        # path, then frozen. Gate MODIFF_ATTN_S8_SCORE=1. Quality-safe (dynamic softmax).
        self.s8_score = (bits == 8 and os.environ.get("MODIFF_ATTN_S8_SCORE") == "1")
        self._sS = None; self._sS_acc = 0.0; self._sS_n = 0
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
        self._qcol_amax = None
        self._oscale = None; self._qkv_wi8 = None; self._epi_i8 = None

    def _fq_accum(self, qkv_flat):
        """Accumulate the per-output-column absmax of the fp16 qkv (-> oscale) during calibration."""
        col = qkv_flat.detach().float().abs().amax(0)                    # [3C]
        self._qcol_amax = col if self._qcol_amax is None else torch.maximum(self._qcol_amax, col)

    def _freeze_fq(self):
        """Freeze the int8-output fused GN->qkv weights: fold the per-column requant oscale into the
        (GN-folded) fused conv weight + epilogue bias so fused_gn_qkv_int8 emits int8 directly."""
        self._ensure_fused()                                            # builds _fused_conv_w [3C,1,1,c], _fused_epi_bias [3C]
        K = 3 * self.channels
        osc = (127.0 / self._qcol_amax.clamp_min(1e-4)).float()         # [3C]
        self._oscale = osc.contiguous()
        self._qkv_wi8 = (self._fused_conv_w.float() * osc.view(K, 1, 1, 1)).half().contiguous()
        self._epi_i8 = torch.round(self._fused_epi_bias.float() * osc).clamp(-127, 127).to(torch.int8).contiguous()
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
            if self.s8_score and self._sS is not None:
                # int8-score path: int8 S write + dynamic int8-score softmax (both T×T passes halved)
                S = _mc.attn_qk_int8_s8out(qi, ki, sq, sk, scale, self._sS)
                P, sp = _mc.attn_softmax_requant_s8_dyn(S, self._sS)
                O = _mc.attn_av_int8(P, vt, sp, sv)
            else:
                S = _mc.attn_qk_int8(qi, ki, sq, sk, scale)   # fp16 pre-scaled logits [BH,T,T]
                P, sp = _mc.attn_softmax_requant(S)           # int8 P + per-row sp
                O = _mc.attn_av_int8(P, vt, sp, sv)
                if self.s8_score and self._sS is None:        # calibrate the score scale, then freeze
                    self._sS_acc += S.float().abs().max().item() / 127.0
                    self._sS_n += 1
                    if self._sS_n >= self._calib_steps:
                        self._sS = self._sS_acc / self._sS_n
        else:
            S = _mc.attn_qk_int4(qi, ki, self.hp_qk, sq, sk, scale)
            P, sp = _mc.attn_softmax_requant4(S)
            O = _mc.attn_av_int4(P, vt, sp, sv, T)
        if self.static and not self._frozen:
            self._calib_update(q, k, v, S)
        return O[:, :, :hd]                                   # drop hd_pad

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

    def forward(self, x):
        b, c, H, W = x.shape
        T = H * W
        nh, hd = self.num_heads, self.head_dim
        BH = b * nh
        x_in_tok = x.permute(0, 2, 3, 1).reshape(b, T, c)
        from integration.fused_ops.token_major_attention import _HAS_FUSED_GN_QKV, _FUSE_TILE_M, _FUSE_SHIFT
        # fusion needs the per-sample fused GN->qkv (T a multiple of its M-tile) and fp16
        fq_ok = (self.fuse_qkv_i8 and _HAS and _HAS_FUSED_GN_QKV and x.dtype == torch.float16
                 and (T % _FUSE_TILE_M) == 0 and T >= 256 and (c % 8) == 0)
        # ---- FUSED int8 path (frozen): fused GN->qkv with INT8 output -> attn quantize-from-int8.
        # Keeps the GN+qkv fusion (fp16 mainloop, int8-clamp epilogue) AND drops the fp16 qkv
        # materialization + reshape copy. ----
        if fq_ok and self._fq_frozen:
            qkv_i8 = _mc.fused_gn_qkv_int8(x, self._qkv_wi8, self._epi_i8,
                                           self.norm.num_groups, self.norm.eps, _FUSE_SHIFT)  # [N,3C,H,W] i8 CL
            qkv_i8_flat = qkv_i8.permute(0, 2, 3, 1).reshape(b * T, 3 * c).contiguous()
            qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_from_i8(qkv_i8_flat, self._oscale, nh, T, self.hp_qk, self.hp_av)
            S = _mc.attn_qk_int8(qi, ki, sq, sk, self.scale)
            P, sp = _mc.attn_softmax_requant_static(S, self._c)
            O = _mc.attn_av_int8(P, vt, sp, sv)[:, :, :hd]
            a = O.reshape(b, nh, T, hd).transpose(1, 2).reshape(b, T, c)
            out_tok = (self.proj(a, residual=x_in_tok) if getattr(self.proj, "_use_bias_res", False)
                       else x_in_tok + self.proj(a))                 # fused residual in proj epilogue
            return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)
        # ---- normal fp16-qkv path (also gathers the fusion's oscale calibration while not frozen) ----
        need_fq_calib = fq_ok and not self._fq_frozen           # force GN+Linear to expose the fp16 qkv
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
                self._fq_accum(qkv_flat.reshape(b * T, 3 * c))
            qkv = qkv_flat.view(b, T, nh, 3, hd)
        q, k, v = qkv.unbind(3)                                          # [b,T,nh,hd]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)   # [b,nh,T,hd]
        # The batched int GEMMs need T%64==0 (N tile); by default small-T blocks (T<256, e.g. 4x4/8x8)
        # fall back to fp16 standard math attention — quant loses there (measured 0.44x at T=64) and
        # they are negligibly cheap anyway. int8/int4 win 2.2-2.4x on the big T=1024.
        # MODIFF_QUANT_ATTN_ALLT=1 (experiment): drop the T>=256 floor to force int8 on the T=64 blocks
        # too (T%64==0 is a HARD kernel constraint, so hd=96 T=16/T=4 blocks still fall back to fp16).
        _min_t = 0 if os.environ.get("MODIFF_QUANT_ATTN_ALLT") == "1" else 256
        # FUSED path (DEFAULT): eligible blocks go through the fused flash kernel (QKᵀ + online softmax
        # + AV in one kernel, scores kept in SRAM -> no [BH,T,T] HBM round-trip, no separate
        # softmax_requant). It's faster AND quality-transparent (int8 fused adds only ~0.004 rel-L2
        # over fp16-attn), so it's on by default; MODIFF_QATTN_FLASH=0 reverts to the materialized path.
        _use_flash = (os.environ.get("MODIFF_QATTN_FLASH", "1") != "0" and _HAS_FLASH
                      and self.head_dim <= 48 and (T % 64) == 0)
        if _use_flash and _HAS_PACKED:
            # packed: read interleaved qkv directly (skips the fp16 transpose+contiguous copy)
            a = self._flash_quant_attn_packed(qkv)                       # [b,nh,T,hd], fused
        elif _use_flash:
            a = self._flash_quant_attn(q, k, v)                          # [b,nh,T,hd], fused
        elif T % 64 != 0 or T < _min_t or not _HAS:
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
        out_tok = (self.proj(a, residual=x_in_tok) if getattr(self.proj, "_use_bias_res", False)
                   else x_in_tok + self.proj(a))                     # fused residual in proj epilogue
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
