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

_QK_I8 = 8
_QK_I4_PACKED = 4
_QK_I4_VALUES_I8_MMA = 84

from integration.fused_ops.token_major_attention import (
    TokenMajorAttentionBlock, _group_norm_silu,
    _HAS_FUSED_ATTN_OUT, _HAS_ATTN_OUT_I4, _QuantLinearWxAx)

try:
    import modiff_cutlass as _mc
    _HAS_FLASH = hasattr(_mc, "flash_attn_int8_vt") and hasattr(_mc, "flash_attn_int4_vt")
    # packed-qkv quantize (reads interleaved [b,T,nh,3,hd] -> no fp16 transpose copy)
    _HAS_PACKED = hasattr(_mc, "quantize_attn_qkv_packed") and hasattr(_mc, "quantize_attn_qkv_packed_static")
    # flash variants that emit the proj-quantized token-major output directly (fuse quantize_attn_out)
    _HAS_FLASH_QOUT = hasattr(_mc, "flash_attn_int8_vt_qout") and hasattr(_mc, "flash_attn_int4_vt_qout")
except Exception:
    _mc = None
    _HAS_FLASH = False
    _HAS_PACKED = False
    _HAS_FLASH_QOUT = False

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
        # Opt-in (MODIFF_FUSE_GN_QKV_INT8=1): even in quant mode, run the fp16 fused GN->qkv (A)
        # for eligible blocks (T%128==0, c%8==0). int8 only (int4 qweight is nibble-packed, can't
        # cheaply rebuild the fp16 weight).
        #
        # SUPERSEDED, and unreachable in steady state: _int8_qkv_epilogue_forward returns before
        # this branch, so the flag only does anything with MODIFF_INT8_QKV_EPILOGUE=0. The
        # 1.37x/1.15x this comment used to claim was measured 2026-07-23 against the then baseline,
        # a week before the QKV int8-epilogue route existed. Re-measured 2026-08-03 against today's
        # routes (docs/gn_qkv_fusion_2026-08-03): 0.901x at T=1024 and 0.754x vs production, and
        # 0.983x/0.854x even against the plain _qkv_from_gn baseline, which has since fused its own
        # GN+quantize. The GN fusion does win its own segment (1.25x on norm+QKV at T=1024) but it
        # forfeits the layout epilogue, which is worth more: production emits Q/K/Vt straight out of
        # the QKV GEMM, so the 307 us K/V gather+transpose pass does not exist, while a fused-GN
        # route has to pay it back. The two fusions are mutually exclusive -- applying per-sample GN
        # scale/bias in the mainloop needs CUTLASS ImplicitGemmConvolutionFusion, whose
        # LinearCombination epilogue cannot also emit the three attention layouts. Kept as a
        # rollback/diagnostic route only.
        self._fuse_gn_qkv_i8 = (bits == 8 and os.environ.get("MODIFF_FUSE_GN_QKV_INT8", "0") != "0")
        # Route 1 (opt-in MODIFF_ROUTE1=1): int8-emitting fused GN->qkv + int8 reshuffle, skipping
        # both the fp16 qkv round-trip and the separate flash quantize. int8 only; kicks in only once
        # flash + its static scales have frozen (calibration uses the normal path).
        # Same status as _fuse_gn_qkv_i8 above: superseded by the QKV int8-epilogue route, which
        # returns first, so this needs MODIFF_INT8_QKV_EPILOGUE=0 to be reachable. Measured
        # 2026-08-03 at 0.942x (T=1024) and 0.815x (T=256) vs production -- the best of the three
        # fused-GN variants, and still a loss. See docs/gn_qkv_fusion_2026-08-03/FINDINGS.md.
        self._route1 = (bits == 8 and os.environ.get("MODIFF_ROUTE1", "0") != "0"
                        and hasattr(_mc, "fused_gn_qkv_i8evt") and hasattr(_mc, "quantize_attn_qkv_from_i8"))
        self._r1_ready = False
        # static kept for API compat; the flash path always self-calibrates over the first
        # _calib_steps forwards then freezes to a static single-pass quantize.
        self.static = bool(static)
        self._calib_steps = int(os.environ.get("MODIFF_ATTN_CALIB_STEPS", "8"))
        # Production routing is deterministic for BOTH bit widths now: quantized attention is on
        # unless an explicit rollback requests "off", and "auto" is not a supported mode.
        #
        # INT4 used to store this string raw while INT8 normalized it. That was a latent bug, not
        # a deliberate difference: an INT4 run with MODIFF_FLASH_GATE=1 (or =0) matched neither
        # "off" nor "on" in _resolve_flash, so it fell through to _autotune_flash and picked its
        # route by timing. Only the literal strings behaved deterministically.
        flash_gate = os.environ.get("MODIFF_FLASH_GATE", "on").lower()
        if flash_gate not in ("0", "1", "off", "on", "false", "true"):
            raise ValueError(
                f"INT{bits} MODIFF_FLASH_GATE accepts only on/off; auto was removed")
        self._flash_gate = (
            "off" if flash_gate in ("0", "off", "false") else "on")
        # Packed-input flash is an explicit diagnostic route: flash reads the
        # interleaved qkv directly, folding the Q/K/V quantize + V-transpose into its smem staging
        # (no separate aq_qtok/aq_vquant + qi/ki/vt HBM round-trip). It loses at T1024, so the
        # production default is off. There is no per-layer autotune; =1 always selects it.
        flash_packed = os.environ.get("MODIFF_FLASH_PACKED", "0").lower()
        int8_persistent = os.environ.get(
            "MODIFF_INT8_PACKED_PERSISTENT", "0").lower()
        int8_flash_preg = os.environ.get(
            "MODIFF_INT8_FLASH_PREG", "0").lower()
        int8_flash_hd24_exact = os.environ.get(
            "MODIFF_INT8_FLASH_HD24_EXACT", "1").lower()
        int8_layout_epilogue = os.environ.get(
            "MODIFF_INT8_QKV_LAYOUT_EPILOGUE", "1").lower()
        int8_compact_epilogue = os.environ.get(
            "MODIFF_INT8_QKV_COMPACT_EPILOGUE", "1").lower()
        int8_qkv_epilogue = os.environ.get(
            "MODIFF_INT8_QKV_EPILOGUE", "1").lower()
        if bits == 8:
            valid_switches = ("0", "1", "off", "on", "false", "true")
            for name, value in (
                    ("MODIFF_FLASH_PACKED", flash_packed),
                    ("MODIFF_INT8_PACKED_PERSISTENT", int8_persistent),
                    ("MODIFF_INT8_FLASH_PREG", int8_flash_preg),
                    ("MODIFF_INT8_FLASH_HD24_EXACT", int8_flash_hd24_exact),
                    ("MODIFF_INT8_QKV_LAYOUT_EPILOGUE", int8_layout_epilogue),
                    ("MODIFF_INT8_QKV_COMPACT_EPILOGUE", int8_compact_epilogue),
                    ("MODIFF_INT8_QKV_EPILOGUE", int8_qkv_epilogue)):
                if value not in valid_switches:
                    raise ValueError(
                        f"INT8 {name} accepts only on/off; auto was removed")
        self._flash_packed = (
            bits == 8 and flash_packed in ("1", "on", "true")
            and hasattr(_mc, "flash_attn_int8_packed_vt"))
        self._int8_persistent = int8_persistent in ("1", "on", "true")
        self._int8_flash_preg = int8_flash_preg in ("1", "on", "true")
        self._int8_flash_hd24_exact = (
            int8_flash_hd24_exact in ("1", "on", "true"))
        self._int8_layout_epilogue = int8_layout_epilogue in ("1", "on", "true")
        self._int8_compact_epilogue = int8_compact_epilogue in ("1", "on", "true")
        self._int8_qkv_epilogue = int8_qkv_epilogue in ("1", "on", "true")
        self._int8_qkv_inv_out = None
        self._int8_layout_qweight = None
        self._int8_layout_wscale = None
        self._int8_layout_inv_out = None
        self._int8_layout_bias = None
        # Q-in-flash is now pinned on rather than timed at runtime. The autotune it replaces ran
        # inside a real forward (3 warmups + 10 timed iterations per side, per block) and cached
        # a decision that could differ between runs. It never had anything to decide at T1024/hd24
        # -- that shape short-circuits earlier -- so it only ever chose for T256 and T64, where
        # the recorded microbenchmarks are decisive and one-sided:
        #   T256/hd48  0.4191 -> 0.3649 ms  (1.15x)
        #   T64/hd48   0.0889 -> 0.0801 ms  (1.11x)
        # both far outside the 1% margin the autotune used. See INT4_ATTENTION_OPTIMIZATION.
        int4_qin = os.environ.get("MODIFF_INT4_Q_IN_FLASH", "on").lower()
        int4_epi = os.environ.get("MODIFF_INT4_QKV_EPILOGUE", "0").lower()
        int4_layout = os.environ.get("MODIFF_INT4_QKV_LAYOUT_EPILOGUE", "1").lower()
        if bits == 4:
            valid = ("0", "1", "off", "on", "false", "true")
            for name, value in (("MODIFF_INT4_Q_IN_FLASH", int4_qin),
                                ("MODIFF_INT4_QKV_EPILOGUE", int4_epi),
                                ("MODIFF_INT4_QKV_LAYOUT_EPILOGUE", int4_layout)):
                if value not in valid:
                    raise ValueError(
                        f"INT4 {name} accepts only on/off; auto was removed")
        self._int4_qin_on = int4_qin not in ("0", "off", "false")
        self._int4_qkv_epilogue_mode = int4_epi
        # Direct-layout INT4 QKV epilogue (T1024/hd24 only), default ON. Measured on A40 at
        # batch 128: the T1024 layer goes 2918.5 -> 2650.3 us, taking the 21-block weighted
        # total from 20.812 to 19.461 ms and putting INT4 0.668 ms AHEAD of INT8's 20.129.
        # Set to 0 for an explicit rollback to the fp16-QKV + K/V-producer route.
        self._int4_layout_epilogue = int4_layout not in ("0", "off", "false")
        # These two were read via os.environ on EVERY forward. Nothing else in this class does
        # that, and it made the route depend on mutable process state mid-run.
        self._int8_kv_compact24 = os.environ.get(
            "MODIFF_INT8_KV_COMPACT24", "0") != "0"
        self._int4_compact_static = os.environ.get(
            "MODIFF_INT4_COMPACT_STATIC", "1") != "0"
        self._int4_layout_qweight = None
        self._int4_layout_wscale = None
        self._int4_layout_inv_out = None
        self._int4_layout_lim = None
        self._int4_codes_inv = None
        self._int4_codes_lim = None
        self._int4_layout_bias = None

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
            if getattr(self, "_fq_frozen2", False) and hasattr(_mc, "flash_attn_int8_vt_static"):
                return _mc.flash_attn_int8_vt_static(
                    qi, ki, vt, sv, self._fq_sqc, self._fq_skc, self.scale)
            return _mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, self.scale)
        # Task 2: int4 fused. ONE-PASS mixed quantize (quantize_attn_qkv_i4qk_i8v): int4-packed Q/K
        # (matches the flash kernel) + int8 transposed V (flash uses int8 PV) in a single sweep of q/k/v
        # -> no eager nibble-pack, no wasted int4-V, no double V quantize. Feeds flash_attn_int4_vt directly.
        #
        # hdp4 is 64 because mma.m16n8k64.s4 has no shorter K. At hd=24 that pads the k-depth 24->64,
        # so 62% of it is zeros and int4 issues the SAME number of mma instructions as int8's
        # m16n8k32 -- measured: int4 is 1.01x int8 at T=1024/hd=24 (no advantage), against 1.30x at
        # hd=48 where the wider instruction does pay. Switching to m16n8k32.s4 for hd<=32 was measured
        # and rejected (data/attn_headroom.json): it would halve K's smem/HBM traffic but not the mma
        # count, and this kernel is nowhere near memory-bound -- even the pessimistic traffic model
        # (K/V re-read grid.y times, every CTA missing L2) puts it at 56% of the 590 GB/s ceiling,
        # and the unit-floor analysis leaves t_hbm 6.8x below the measured time. Upper bound on the
        # whole idea: 1.28x, and only if bandwidth were the binding constraint. It is not.
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
        if getattr(self, "_fq4_frozen", False) and hasattr(_mc, "flash_attn_int4_vt_static"):
            return _mc.flash_attn_int4_vt_static(
                q4, k4, vt, sv, hdp4, self._fq4_sqc, self._fq4_skc, self.scale)
        return _mc.flash_attn_int4_vt(q4, k4, vt, sq4, sk4, sv, hdp4, self.scale)

    def _packed_ref_vt(self, qkv, b, nh, T, hd, hd_pad):
        """Current path (quantize_attn_qkv_packed_static -> flash_attn_int8_vt), frozen scales.
        The reference the packed score path is autotuned against."""
        qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_packed_static(
            qkv, nh, T, hd, hd_pad, hd_pad, 8, self._fq_sqc, self._fq_skc, self._fq_svv)
        qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad); vt = vt.view(b, nh, hd_pad, T)
        sq = sq.view(b, nh, T).contiguous(); sk = sk.view(b, nh, T).contiguous()
        sv = sv[..., :hd].contiguous().view(b, nh, hd)
        return _mc.flash_attn_int8_vt_static(
            qi, ki, vt, sv, self._fq_sqc, self._fq_skc, self.scale)

    def _packed_ref_vt_qout(self, qkv, b, nh, T, hd, hd_pad, proj_a):
        """Current fused-proj path (quantize -> flash_attn_int8_vt_qout). Autotune reference."""
        qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_packed_static(
            qkv, nh, T, hd, hd_pad, hd_pad, 8, self._fq_sqc, self._fq_skc, self._fq_svv)
        qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad); vt = vt.view(b, nh, hd_pad, T)
        sq = sq.view(b, nh, T).contiguous(); sk = sk.view(b, nh, T).contiguous()
        sv = sv[..., :hd].contiguous().view(b, nh, hd)
        return _mc.flash_attn_int8_vt_static_qout(
            qi, ki, vt, sv, self._fq_sqc, self._fq_skc, self.scale, proj_a)

    def _flash_quant_attn_packed(self, qkv):
        """Same as _flash_quant_attn but reads the interleaved qkv [b,T,nh,3,hd] DIRECTLY via the
        packed quantize kernels -> drops the ~1.2 GB/step fp16 q/k/v.transpose().contiguous() copy
        (the largest 'glue' cost). Output identical: [b,nh,T,hd] fp16."""
        b, T, nh, _, hd = qkv.shape
        qkv = qkv.contiguous()
        qv, kv, vv = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :]  # strided views (no copy)
        if self.bits == 8:
            hd_pad = ((hd + 31) // 32) * 32
            # Explicit packed diagnostic path: read qkv directly and fold quantization into flash
            # staging. Production leaves this disabled because it loses at hd24/T1024.
            if self._flash_packed and getattr(self, "_fq_frozen2", False):
                sv_hd = self._fq_svv[:hd].contiguous()
                return _mc.flash_attn_int8_packed_vt(
                    qkv, sv_hd, hd_pad, self._fq_sqc, self._fq_skc, self.scale)
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
            if getattr(self, "_fq_frozen2", False) and hasattr(_mc, "flash_attn_int8_vt_static"):
                return _mc.flash_attn_int8_vt_static(
                    qi, ki, vt, sv, self._fq_sqc, self._fq_skc, self.scale)
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
        if getattr(self, "_fq4_frozen", False) and hasattr(_mc, "flash_attn_int4_vt_static"):
            return _mc.flash_attn_int4_vt_static(
                q4, k4, vt, sv, hdp4, self._fq4_sqc, self._fq4_skc, self.scale)
        return _mc.flash_attn_int4_vt(q4, k4, vt, sq4, sk4, sv, hdp4, self.scale)

    def _scores_scales_frozen(self):
        """True once the per-block Q/K/V quantize scales have self-calibrated and frozen to
        the static single-pass path (so a flash timing reflects steady state, not calibration)."""
        return getattr(self, "_fq_frozen2" if self.bits == 8 else "_fq4_frozen", False)

    def _observe_small_int8_scales(self, qkv):
        """Calibrate T16/T4 static Q/K/V scales while their warmup uses FP16 SDPA.

        Runs for BOTH bit widths. It used to return immediately for bits != 8, which meant the
        six hd=96 INT4 blocks could never reach _fq4_frozen and so were permanently stuck on the
        FP16 SDPA fallback -- there was no INT4 route for them to become eligible for anyway.
        Now that flash_attn_i4values_small_qout exists, they need the frozen scales.
        """
        frozen = "_fq_frozen2" if self.bits == 8 else "_fq4_frozen"
        if getattr(self, frozen, False):
            return
        _, _, _, _, hd = qkv.shape
        qv, kv, vv = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :]
        self._fq_aq = max(getattr(self, "_fq_aq", 0.0), qv.abs().max().item())
        self._fq_ak = max(getattr(self, "_fq_ak", 0.0), kv.abs().max().item())
        avc = vv.abs().amax(dim=(0, 1, 2)).float()
        self._fq_av = (avc if getattr(self, "_fq_av", None) is None
                       else torch.maximum(self._fq_av, avc))
        self._fq_n = getattr(self, "_fq_n", 0) + 1
        if self._fq_n >= self._calib_steps:
            # INT4 keeps values on the signed int4 grid, so the divisor is 7 not 127.
            lvl = 127.0 if self.bits == 8 else 7.0
            hd_pad = ((hd + 31) // 32) * 32
            svv = torch.ones(hd_pad, device=qkv.device)
            svv[:hd] = (self._fq_av / 127.0).clamp_min(1e-8)   # V stays int8 in both paths
            if self.bits == 8:
                self._fq_sqc = self._fq_aq / lvl
                self._fq_skc = self._fq_ak / lvl
                self._fq_svv = svv.contiguous()
                self._fq_frozen2 = True
            else:
                self._fq4_sqc = self._fq_aq / lvl
                self._fq4_skc = self._fq_ak / lvl
                self._fq4_svv = svv.contiguous()
                self._fq4_frozen = True

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

    def _flash_shape_ok(self, T):
        """The flash mma kernels' shape constraint: head_dim <= 48 (FA_MMA_MAXHD=64 after padding)
        and T % 64 == 0 (FA_MMA_WARPS * FA_MMA_BR).

        ONE definition, used by every gate that feeds a flash entry point. `_qkv_i8_ok` used to
        re-derive its own weaker version (`head_dim % 16 == 0` alone), which admitted the hd=96
        blocks -- shapes the flash kernel rejects with "mma-eligible shapes only" and that never ran
        flash in the first place. That divergence is what made route (b) look structurally impossible
        when it was a missing condition. Constraints asserted in
        integration/tests/test_flash_packed_int8_shapes.py."""
        return self.head_dim <= 48 and (T % 64) == 0

    def _resolve_flash(self, qkv, T):
        """Decide whether this block runs the fused flash score path. Base eligibility is the
        flash kernel's constraint (head_dim<=48, T%64==0); beyond that the gate is a pure
        on/off rollback switch for BOTH bit widths. The runtime A/B that used to live here
        (_autotune_flash) is gone: it was only ever reachable through the INT4 gate-string bug
        described in __init__, and it timed two routes inside a production forward."""
        eligible = _HAS_FLASH and self._flash_shape_ok(T)
        return eligible and self._flash_gate == "on"

    def _ensure_route1(self, device, SHIFT):
        """Build the Route-1 fused-conv weight + fp32 EVT bias once the flash static scales freeze.
        oscale folds the frozen flash scales (per-tensor 1/sq,1/sk for Q/K; per-channel 1/sv[d] for V)
        into the (gamma-folded) qkv weight; fp32 bias = oscale*(qkv.bias + w@beta - SHIFT*colsum)."""
        if self._r1_ready:
            return
        C = self.channels; hd = self.head_dim; K = 3 * C
        if getattr(self.qkv, "weight", None) is not None:
            qw = self.qkv.weight.detach().float()                         # [3C,C]
        else:  # QuantLinearWxAx (int8): rebuild fp16 weight from qweight*w_scale
            qw = self.qkv.qweight[:K, :C].float() * self.qkv.w_scale[:K].float()[:, None]
        qb = self.qkv.bias.detach().float()
        gw = (self.norm.weight.detach().float() if self.norm.weight is not None else torch.ones(C, device=device))
        gb = (self.norm.bias.detach().float() if self.norm.bias is not None else torch.zeros(C, device=device))
        Wf = qw * gw[None, :]                                             # [3C,C]
        epi_real = qb + qw @ gb - SHIFT * Wf.sum(1)                       # [3C]
        j = torch.arange(K, device=device); sel = (j // hd) % 3; d = j % hd
        inv = torch.where(sel == 0, torch.tensor(1.0 / float(self._fq_sqc), device=device),
              torch.where(sel == 1, torch.tensor(1.0 / float(self._fq_skc), device=device),
                          1.0 / self._fq_svv[d].float()))                 # oscale [3C]
        self._r1_w = (Wf * inv[:, None]).to(torch.float16).view(K, 1, 1, C).contiguous()
        self._r1_bias = (inv * epi_real).float().contiguous()
        self._r1_ready = True

    def _route1_score(self, x, b, T, nh, hd, SHIFT):
        """int8-emitting fused GN->qkv -> int8 reshuffle -> flash. Returns [b,nh,T,hd] fp16."""
        self._ensure_route1(x.device, SHIFT)
        hd_pad = ((hd + 31) // 32) * 32
        qkv_i8 = _mc.fused_gn_qkv_i8evt(x, self._r1_w, self._r1_bias,
                                        self.norm.num_groups, self.norm.eps, SHIFT)   # int8 [b,3C,H,W] CL
        qkv_pk = qkv_i8.permute(0, 2, 3, 1).reshape(b, T, nh, 3, hd).contiguous()     # [b,T,nh,3,hd]
        qi, ki, vt = _mc.quantize_attn_qkv_from_i8(qkv_pk, nh, T, hd, hd_pad, hd_pad)
        qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad); vt = vt.view(b, nh, hd_pad, T)
        sv = self._fq_svv[:hd].float().view(1, 1, hd).expand(b, nh, hd).contiguous()
        return _mc.flash_attn_int8_vt_static(
            qi, ki, vt, sv, self._fq_sqc, self._fq_skc, self.scale)                  # [b,nh,T,hd]

    def _route1_qout(self, x, b, T, nh, hd, SHIFT, proj_a_scale):
        """Route-1 without the three post-GEMM conversion kernels.

        Q stays in packed INT8 QKV and is loaded by its consuming Flash CTA. A
        single tiled producer gathers K and transposes V, while Flash writes the
        calibrated INT8 projection input directly.
        """
        self._ensure_route1(x.device, SHIFT)
        hd_pad = ((hd + 31) // 32) * 32
        qkv_i8 = _mc.fused_gn_qkv_i8evt(
            x, self._r1_w, self._r1_bias,
            self.norm.num_groups, self.norm.eps, SHIFT)
        qkv_pk = qkv_i8.permute(0, 2, 3, 1).reshape(
            b, T, nh, 3, hd).contiguous()
        ki, vt = _mc.quantize_attn_kv_from_i8(
            qkv_pk, nh, T, hd, hd_pad, hd_pad)
        sv = self._fq_svv[:hd].contiguous()
        return _mc.flash_attn_int8_qi8packed_kv_static_qout(
            qkv_pk, ki.view(b, nh, T, hd_pad),
            vt.view(b, nh, hd_pad, T), sv, hd_pad,
            self._fq_sqc, self._fq_skc, self.scale, proj_a_scale)

    def _int4_layout_epilogue_forward(self, x, x_in_tok, b, T, c, nh, hd):
        """T1024/hd24 INT4: GN-pack -> ONE-launch direct-layout W4A4 QKV epilogue -> exact
        hd24 flash -> W4 proj. The INT4 analogue of _int8_qkv_epilogue_forward's use_layout
        branch: the GEMM emits token-major Q, padded K and transposed Vt itself, so neither the
        FP16 QKV tensor nor the K/V producer nor the rearrange pass exists.
        """
        mixed = (T == 1024 and hd == 24)          # unpacked int4 values through the int8 MMA
        packed = (T in (256, 64) and hd == 48)     # native int4 MMA, nibble-packed Q/K
        # Both hd=96 shapes, so every attention block in the model runs a quantized route and the
        # FP16 SDPA fallback disappears entirely. T16 is a KNOWN performance loss: the dp4a kernel
        # costs ~T^2 while PyTorch's flash is launch-bound and flat, measured 63.7us against 41.7
        # per layer (T4 is the other way round, 8.1 vs 38.4). Taken deliberately for structural
        # uniformity -- one INT4 dataflow, no shape-dependent fallback to reason about. It also
        # removes the separate quant_attn_out_int4_pack pass those blocks needed.
        small = (T in (16, 4) and hd == 96)
        if (not self._int4_layout_epilogue or self.bits != 4
                or not (mixed or packed or small)
                or (not small and not hasattr(_mc, "gemm_w4a4_awq_qkv_i4qk_i8v_layouts"))
                or (mixed and not hasattr(
                    _mc, "flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24"))
                or (packed and not hasattr(_mc, "flash_attn_int4_vt_static_qout"))
                or (small and not (hasattr(_mc, "flash_attn_i4values_small_qout")
                                   and hasattr(_mc, "gemm_w4a4_awq_qkv_codes")))):
            return None
        qkv, proj = self.qkv, self.proj
        if not (self._qout_eligible() and _QuantLinearWxAx is not None
                and isinstance(qkv, _QuantLinearWxAx) and qkv.bits == 4
                and qkv.a_scale is not None and not qkv.modiff
                and hasattr(_mc, "group_norm_silu_quantize_pack_nhwc_fast")):
            return None
        gw, gb = self._gn_params(x.dtype)
        if gw is None or gb is None:
            return None
        # mixed: hp=32 so the s8 MMA runs at k=32. packed: hp=64, the native s4 MMA's k.
        if small:
            # hd=96 needs no layout rearrangement at all: the dp4a kernel reads the compact
            # token-major [b,T,nh,3,hd] code matrix directly, so one GEMM launch feeds it.
            if self._qkv_inv_scale_t is None:
                self._qkv_inv_scale_t = torch.tensor(
                    [1.0 / qkv.a_scale], device=x.device, dtype=torch.float32)
            empty0 = x.new_empty(0)
            qb = qkv.bias
            if qb is None:
                qb = torch.zeros(qkv.out_features, device=x.device, dtype=torch.float16)
            xq_img = _mc.group_norm_silu_quantize_pack_nhwc_fast(
                x, gw, gb, self.norm.num_groups, self.norm.eps, False,
                self._qkv_inv_scale_t, empty0, empty0, empty0, qkv._awqt_K)
            if self._int4_codes_inv is None:
                n_out = qkv.out_features
                iv = torch.zeros(n_out, device=x.device, dtype=torch.float32)
                lm = torch.zeros(n_out, device=x.device, dtype=torch.float32)
                svf = self._fq4_svv[:hd].float()
                for h in range(nh):
                    for sel in range(3):
                        d0 = (h * 3 + sel) * hd
                        if sel == 0:
                            iv[d0:d0 + hd] = 1.0 / float(self._fq4_sqc)
                        elif sel == 1:
                            iv[d0:d0 + hd] = 1.0 / float(self._fq4_skc)
                        else:
                            iv[d0:d0 + hd].copy_(1.0 / svf)
                        lm[d0:d0 + hd] = 7.0 if sel < 2 else 127.0
                self._int4_codes_inv = iv.contiguous()
                self._int4_codes_lim = lm.contiguous()
            codes = _mc.gemm_w4a4_awq_qkv_codes(
                xq_img.reshape(b * T, qkv._awqt_K // 2), qkv.qweight, qkv.w_scale,
                qkv.a_scale, qkv._awqt_K, qkv.out_features, qb, nh, hd,
                self._int4_codes_inv, self._int4_codes_lim)
            xattn = _mc.flash_attn_i4values_small_qout(
                codes.view(b, T, nh, 3, hd), self._fq4_svv[:hd].contiguous(),
                self._fq4_sqc, self._fq4_skc, self.scale, proj.a_scale, proj._awqt_K)
            # gemm_*_awq_bias_res requires an fp16 residual. No-op in production; load-bearing
            # while the conv layers are uncalibrated, where they emit fp32.
            res0 = x_in_tok.reshape(b * T, c).half().contiguous()
            pb0 = proj.bias if proj.bias is not None else empty0
            out0 = _mc.gemm_w4a4_awq_bias_res(
                xattn, proj.qweight, proj.w_scale, proj.a_scale,
                proj._awqt_K, proj.out_features, pb0, res0)
            return out0.reshape(b, T, c)
        hp = 32 if mixed else 64
        layout_n = nh * 3 * hp                           # 768 (mixed) / 1536 (packed)
        if (self._int4_layout_qweight is None
                or self._int4_layout_qweight.size(0) != layout_n):
            # One-time offline channel layout, identical in structure to the INT8 builder below.
            # qweight is nibble-packed along K ([N, K/2]), so re-basing OUTPUT channels is a plain
            # row copy -- the packing is never touched. Padded rows stay zero, which makes the
            # GEMM compute an exact zero in the d >= hd lanes and removes any need to clear tails.
            qb = qkv.bias
            if qb is None:
                qb = torch.zeros(qkv.out_features, device=x.device, dtype=torch.float16)
            lw = torch.zeros(layout_n, qkv.qweight.size(1),
                             device=x.device, dtype=qkv.qweight.dtype)
            lws = torch.zeros(layout_n, device=x.device, dtype=torch.float32)
            liv = torch.zeros(layout_n, device=x.device, dtype=torch.float32)
            llim = torch.zeros(layout_n, device=x.device, dtype=torch.float32)
            lb = torch.zeros(layout_n, device=x.device, dtype=torch.float16)
            svf = self._fq4_svv[:hd].float()
            for h in range(nh):
                for sel in range(3):
                    src = (h * 3 + sel) * hd
                    dst = (h * 3 + sel) * hp
                    lw[dst:dst + hd].copy_(qkv.qweight[src:src + hd])
                    lws[dst:dst + hd].copy_(qkv.w_scale[src:src + hd])
                    lb[dst:dst + hd].copy_(qb[src:src + hd])
                    # Fold the Q/K/V role selection into the per-column vectors so the epilogue
                    # needs no runtime decode at all. Only [:hd] is written -- _fq4_svv is
                    # ones-padded, so slicing the full hp would put 1.0 (not 0) in the pad lanes.
                    if sel == 0:
                        liv[dst:dst + hd] = 1.0 / float(self._fq4_sqc)
                    elif sel == 1:
                        liv[dst:dst + hd] = 1.0 / float(self._fq4_skc)
                    else:
                        liv[dst:dst + hd].copy_(1.0 / svf)
                    llim[dst:dst + hd] = 7.0 if sel < 2 else 127.0
            self._int4_layout_qweight = lw.contiguous()
            self._int4_layout_wscale = lws.contiguous()
            self._int4_layout_inv_out = liv.contiguous()
            self._int4_layout_lim = llim.contiguous()
            self._int4_layout_bias = lb.contiguous()
        if self._qkv_inv_scale_t is None:
            self._qkv_inv_scale_t = torch.tensor(
                [1.0 / qkv.a_scale], device=x.device, dtype=torch.float32)
        empty = x.new_empty(0)
        xq_img = _mc.group_norm_silu_quantize_pack_nhwc_fast(
            x, gw, gb, self.norm.num_groups, self.norm.eps, False,
            self._qkv_inv_scale_t, empty, empty, empty, qkv._awqt_K)
        xq = xq_img.reshape(b * T, qkv._awqt_K // 2)
        q_i8, ki, vt, sv = _mc.gemm_w4a4_awq_qkv_i4qk_i8v_layouts(
            xq, self._int4_layout_qweight, self._int4_layout_wscale, qkv.a_scale,
            qkv._awqt_K, self._int4_layout_inv_out, self._int4_layout_lim,
            self._int4_layout_bias, nh, T, hd, hp, self._fq4_svv[:hd],
            1 if packed else 0)
        if packed:
            # Q/K come back head-major nibble-packed [BH,T,hp/2] -- exactly the layout
            # flash_attn_int4_vt_static_qout already consumes, so no new wrapper is needed.
            xattn = _mc.flash_attn_int4_vt_static_qout(
                q_i8.view(b, nh, T, hp // 2), ki.view(b, nh, T, hp // 2),
                vt.view(b, nh, hp, T), sv, hp,
                self._fq4_sqc, self._fq4_skc, self.scale, proj.a_scale, proj._awqt_K)
        else:
            xattn = _mc.flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24(
                q_i8, ki.view(b, nh, T, hp), vt.view(b, nh, hp, T), sv, hp,
                self._fq4_sqc, self._fq4_skc, self.scale, proj.a_scale, proj._awqt_K)
        # gemm_*_awq_bias_res requires an fp16 residual. No-op in production; load-bearing while
        # the conv layers are uncalibrated, where they emit fp32 (see FINDINGS 2026-08-03).
        res = x_in_tok.reshape(b * T, c).half().contiguous()
        pbias = proj.bias if proj.bias is not None else empty
        out = _mc.gemm_w4a4_awq_bias_res(
            xattn, proj.qweight, proj.w_scale, proj.a_scale,
            proj._awqt_K, proj.out_features, pbias, res)
        return out.reshape(b, T, c)

    def _int4_qkv_epilogue_forward(self, x, x_in_tok, b, T, c, nh, hd):
        """Opt-in experimental GN-pack -> W4A4 QKV quantizing epilogue -> flash -> W4 proj.

        Returns None unless every frozen static-scale/shape requirement is satisfied. The GEMM
        epilogue emits signed-I4 Q/K codes plus I8 V, avoiding the FP16 QKV tensor entirely.
        """
        enabled = self._int4_qkv_epilogue_mode in ("1", "on")
        if not enabled or self.bits != 4 or not hasattr(
                _mc, "gemm_w4a4_awq_qkv_i4qk_i8v"):
            return None
        qkv, proj = self.qkv, self.proj
        if not (self._qout_eligible() and _QuantLinearWxAx is not None
                and isinstance(qkv, _QuantLinearWxAx) and qkv.bits == 4
                and qkv.a_scale is not None and not qkv.modiff
                and hasattr(_mc, "group_norm_silu_quantize_pack_nhwc_fast")):
            return None
        mixed = T == 1024 and hd == 24
        native = T in (64, 256) and hd == 48
        if not (mixed or native):
            return None
        gw, gb = self._gn_params(x.dtype)
        if gw is None or gb is None:
            return None
        if self._qkv_inv_scale_t is None:
            self._qkv_inv_scale_t = torch.tensor(
                [1.0 / qkv.a_scale], device=x.device, dtype=torch.float32)
        empty = x.new_empty(0)
        xq_img = _mc.group_norm_silu_quantize_pack_nhwc_fast(
            x, gw, gb, self.norm.num_groups, self.norm.eps, False,
            self._qkv_inv_scale_t, empty, empty, empty, qkv._awqt_K)
        xq = xq_img.reshape(b * T, qkv._awqt_K // 2)
        storage = _QK_I4_VALUES_I8_MMA if mixed else _QK_I4_PACKED
        hp_qk = 32 if mixed else 64
        hp_v = ((hd + 31) // 32) * 32
        q, k, vt, sv = _mc.gemm_w4a4_awq_qkv_i4qk_i8v(
            xq, qkv.qweight, qkv.w_scale, qkv.a_scale, qkv._awqt_K,
            qkv.out_features, qkv.bias, nh, T, hd, hp_qk, hp_v, storage,
            self._fq4_sqc, self._fq4_skc, self._fq4_svv)
        q = q.view(b, nh, T, -1); k = k.view(b, nh, T, -1)
        vt = vt.view(b, nh, hp_v, T); sv = sv[:hd].contiguous()
        if mixed:
            xattn = _mc.flash_attn_i4values_i8mma_vt_static_qout(
                q, k, vt, sv, hp_qk, self._fq4_sqc, self._fq4_skc,
                self.scale, proj.a_scale, proj._awqt_K)
        else:
            xattn = _mc.flash_attn_int4_vt_static_qout(
                q, k, vt, sv, hp_qk, self._fq4_sqc, self._fq4_skc,
                self.scale, proj.a_scale, proj._awqt_K)
        # gemm_*_awq_bias_res requires an fp16 residual. No-op in production; load-bearing while
        # the conv layers are uncalibrated, where they emit fp32 (see FINDINGS 2026-08-03).
        res = x_in_tok.reshape(b * T, c).half().contiguous()
        pbias = proj.bias if proj.bias is not None else empty
        out = _mc.gemm_w4a4_awq_bias_res(
            xattn, proj.qweight, proj.w_scale, proj.a_scale,
            proj._awqt_K, proj.out_features, pbias, res)
        return out.reshape(b, T, c)

    def _int8_qkv_epilogue_forward(self, x, x_in_tok, b, T, c, nh, hd):
        """Production GN->W8A8 QKV INT8 epilogue feeding direct-Q quantized attention."""
        if (not self._int8_qkv_epilogue or self.bits != 8
                or not getattr(self, "_fq_frozen2", False)
                or not self._qout_eligible()
                or not hasattr(_mc, "gemm_w8a8_awq_out_i8_bias_nout")
                or not hasattr(_mc, "quantize_attn_kv_from_i8")
                or not hasattr(_mc, "flash_attn_int8_qi8packed_kv_static_qout")
                or (T < 64 and not hasattr(
                    _mc, "flash_attn_int8_qi8packed_small_qout"))):
            return None
        qkv, proj = self.qkv, self.proj
        if not (_QuantLinearWxAx is not None
                and isinstance(qkv, _QuantLinearWxAx) and qkv.bits == 8
                and qkv.a_scale is not None and not qkv.modiff
                and qkv._awqt_K == qkv.in_features):
            return None
        gw, gb = self._gn_params(x.dtype)
        if gw is None or gb is None:
            return None
        if self._qkv_inv_scale_t is None:
            self._qkv_inv_scale_t = torch.tensor(
                [1.0 / qkv.a_scale], device=x.device, dtype=torch.float32)
        empty = x.new_empty(0)
        gnq = getattr(_mc, "group_norm_silu_quantize_nhwc_fast",
                      _mc.group_norm_silu_quantize_nhwc)
        xq_img = gnq(
            x, gw, gb, self.norm.num_groups, self.norm.eps, False,
            self._qkv_inv_scale_t, empty, empty, empty)
        xq = xq_img.permute(0, 2, 3, 1).reshape(b * T, c)

        n_pad = qkv.qweight.size(0)
        if (self._int8_qkv_inv_out is None
                or self._int8_qkv_inv_out.numel() != n_pad):
            inv = torch.zeros(n_pad, device=x.device, dtype=torch.float32)
            j = torch.arange(qkv.out_features, device=x.device)
            sel, d = (j // hd) % 3, j % hd
            inv[:qkv.out_features] = torch.where(
                sel == 0,
                torch.tensor(1.0 / float(self._fq_sqc), device=x.device),
                torch.where(
                    sel == 1,
                    torch.tensor(1.0 / float(self._fq_skc), device=x.device),
                    1.0 / self._fq_svv[d].float()))
            self._int8_qkv_inv_out = inv.contiguous()
        qbias = qkv.bias
        if qbias is None:
            qbias = torch.zeros(
                qkv.out_features, device=x.device, dtype=torch.float16)
        sv = self._fq_svv[:hd].contiguous()
        use_layout = (
            self._int8_layout_epilogue and T == 1024 and hd == 24
            and hasattr(_mc, "gemm_w8a8_awq_qkv_i8_layouts")
            and hasattr(_mc, "flash_attn_int8_qi8_kv_static_qout"))
        # T64 is included even though the layout epilogue measured 0.908x there (slower than the
        # plain GEMM + quantize_attn_kv_from_i8 producer it replaces). Structural uniformity is
        # the priority: every hd=48 shape now takes the same route, so there is one INT8 dataflow
        # to reason about instead of two with a shape-dependent exception. Set
        # MODIFF_INT8_QKV_COMPACT_EPILOGUE=0 to fall the whole hd=48 family back.
        use_compact_layout = (
            self._int8_compact_epilogue and T in (256, 64) and hd == 48
            and hasattr(_mc, "gemm_w8a8_awq_qkv_i8_layouts_compact")
            and hasattr(_mc, "flash_attn_int8_qi8_kv_static_qout"))
        if use_compact_layout:
            hd_pad = 64
            q_i8, ki, vt = _mc.gemm_w8a8_awq_qkv_i8_layouts_compact(
                xq, qkv.qweight, qkv.w_scale, qkv.a_scale,
                self._int8_qkv_inv_out, qbias, nh, T, hd, hd_pad)
            xattn = _mc.flash_attn_int8_qi8_kv_static_qout(
                q_i8, ki.view(b, nh, T, hd_pad),
                vt.view(b, nh, hd_pad, T), sv, hd_pad,
                self._fq_sqc, self._fq_skc, self.scale, proj.a_scale)
        elif use_layout:
            hd_pad = ((hd + 31) // 32) * 32
            layout_n = nh * 3 * hd_pad
            if (self._int8_layout_qweight is None
                    or self._int8_layout_qweight.size(0) != layout_n):
                # One-time, deterministic offline channel layout. Padding each
                # head/QKV segment to hd_pad aligns every epilogue segment with
                # the Flash MMA operand and makes its tail values true zeros.
                lw = torch.zeros(
                    layout_n, qkv.qweight.size(1), device=x.device,
                    dtype=qkv.qweight.dtype)
                ls = torch.zeros(layout_n, device=x.device, dtype=torch.float32)
                li = torch.zeros(layout_n, device=x.device, dtype=torch.float32)
                lb = torch.zeros(layout_n, device=x.device, dtype=torch.float16)
                for h in range(nh):
                    for sel in range(3):
                        src = (h * 3 + sel) * hd
                        dst = (h * 3 + sel) * hd_pad
                        lw[dst:dst + hd].copy_(qkv.qweight[src:src + hd])
                        ls[dst:dst + hd].copy_(qkv.w_scale[src:src + hd])
                        li[dst:dst + hd].copy_(
                            self._int8_qkv_inv_out[src:src + hd])
                        lb[dst:dst + hd].copy_(qbias[src:src + hd])
                self._int8_layout_qweight = lw.contiguous()
                self._int8_layout_wscale = ls.contiguous()
                self._int8_layout_inv_out = li.contiguous()
                self._int8_layout_bias = lb.contiguous()
            q_i8, ki, vt = _mc.gemm_w8a8_awq_qkv_i8_layouts(
                xq, self._int8_layout_qweight,
                self._int8_layout_wscale, qkv.a_scale,
                self._int8_layout_inv_out, self._int8_layout_bias,
                nh, T, hd, hd_pad)
            flash_layout_fn = _mc.flash_attn_int8_qi8_kv_static_qout
            if (self._int8_flash_hd24_exact
                    and hasattr(
                        _mc, "flash_attn_int8_qi8_kv_static_qout_hd24")):
                flash_layout_fn = (
                    _mc.flash_attn_int8_qi8_kv_static_qout_hd24)
            xattn = flash_layout_fn(
                q_i8, ki.view(b, nh, T, hd_pad),
                vt.view(b, nh, hd_pad, T), sv, hd_pad,
                self._fq_sqc, self._fq_skc, self.scale, proj.a_scale)
        else:
            qkv_i8 = _mc.gemm_w8a8_awq_out_i8_bias_nout(
                xq, qkv.qweight, qkv.w_scale, qkv.a_scale,
                self._int8_qkv_inv_out, qbias, qkv.out_features)
            qkv_pk = qkv_i8.view(b, T, nh, 3, hd)
        if not use_layout and not use_compact_layout and T in (4, 16) and hd == 96:
            xattn = _mc.flash_attn_int8_qi8packed_small_qout(
                qkv_pk, sv, self._fq_sqc, self._fq_skc,
                self.scale, proj.a_scale)
        elif not use_layout and not use_compact_layout:
            hd_pad = ((hd + 31) // 32) * 32
            ki, vt = _mc.quantize_attn_kv_from_i8(
                qkv_pk, nh, T, hd, hd_pad, hd_pad)
            flash_fn = _mc.flash_attn_int8_qi8packed_kv_static_qout
            if (self._int8_flash_preg and T == 1024 and hd == 24
                    and hasattr(
                        _mc, "flash_attn_int8_qi8packed_kv_static_qout_preg")):
                flash_fn = _mc.flash_attn_int8_qi8packed_kv_static_qout_preg
            xattn = flash_fn(
                qkv_pk, ki.view(b, nh, T, hd_pad),
                vt.view(b, nh, hd_pad, T), sv, hd_pad,
                self._fq_sqc, self._fq_skc, self.scale, proj.a_scale)
        # gemm_*_awq_bias_res requires an fp16 residual. No-op in production; load-bearing while
        # the conv layers are uncalibrated, where they emit fp32 (see FINDINGS 2026-08-03).
        res = x_in_tok.reshape(b * T, c).half().contiguous()
        pbias = proj.bias if proj.bias is not None else empty
        out = _mc.gemm_w8a8_awq_bias_res(
            xattn, proj.qweight, proj.w_scale, proj.a_scale,
            proj.out_features, pbias, res)
        return out.reshape(b, T, c)

    def forward(self, x):
        """Dtype-transparent wrapper around the route dispatch.

        Every quantized route returns fp16, because the GEMM epilogues emit fp16. That is correct in
        production, where the whole pipeline is fp16. It is wrong during the conv layers'
        UNCALIBRATED window: there the surrounding pipeline is fp32, and handing fp16 to the next
        non-quantized conv raises "Input type (c10::Half) and bias type (float) should be the same"
        -- those convs run with autocast locally disabled, so nothing casts on their behalf.
        Enforcing dtype in == dtype out here, once, is what makes MoDiff-mode conv calibration
        possible at all; without it _calibrate_int8 crashes inside its own sampling pass and the only
        way to get scales is to load a file, which in this tree describes a different random network
        than the one loading it. See docs/modiff_correctness_2026-08-03/FINDINGS.md.
        """
        out = self._forward_routes(x)
        return out if out.dtype == x.dtype else out.to(x.dtype)

    def _forward_routes(self, x):
        b, c, H, W = x.shape
        T = H * W
        nh, hd = self.num_heads, self.head_dim
        x_in_tok = x.permute(0, 2, 3, 1).reshape(b, T, c)
        from integration.fused_ops.token_major_attention import _HAS_FUSED_GN_QKV, _FUSE_TILE_M, _FUSE_SHIFT
        i4_layout_out = self._int4_layout_epilogue_forward(x, x_in_tok, b, T, c, nh, hd)
        if i4_layout_out is not None:
            return i4_layout_out.reshape(b, H, W, c).permute(0, 3, 1, 2)
        qkv_epi_out = self._int4_qkv_epilogue_forward(x, x_in_tok, b, T, c, nh, hd)
        if qkv_epi_out is not None:
            return qkv_epi_out.reshape(b, H, W, c).permute(0, 3, 1, 2)
        int8_qkv_epi_out = self._int8_qkv_epilogue_forward(
            x, x_in_tok, b, T, c, nh, hd)
        if int8_qkv_epi_out is not None:
            return int8_qkv_epi_out.reshape(b, H, W, c).permute(0, 3, 1, 2)
        # ---- Route 1 (opt-in): int8-emitting fused GN->qkv + int8 reshuffle -> flash, skipping the
        # fp16 qkv round-trip AND the separate flash quantize. Only after flash + scales have frozen
        # (calibration used the normal path below); eligible int8 blocks with T%128==0, c%8==0. ----
        if (self._route1 and _HAS_FUSED_GN_QKV and x.dtype == torch.float16
                and self._flash_gate == "on"
                and getattr(self, "_fq_frozen2", False)
                and (T % _FUSE_TILE_M) == 0 and (c % 8) == 0):
            if (self._qout_eligible()
                    and hasattr(_mc, "quantize_attn_kv_from_i8")
                    and hasattr(_mc, "flash_attn_int8_qi8packed_kv_static_qout")):
                proj = self.proj
                xq = self._route1_qout(
                    x, b, T, nh, hd, _FUSE_SHIFT, proj.a_scale)
                # gemm_*_awq_bias_res requires an fp16 residual. No-op in production; load-bearing while
                # the conv layers are uncalibrated, where they emit fp32 (see FINDINGS 2026-08-03).
                res = x_in_tok.reshape(b * T, c).half().contiguous()
                pbias = proj.bias if proj.bias is not None else x.new_empty(0)
                out = _mc.gemm_w8a8_awq_bias_res(
                    xq, proj.qweight, proj.w_scale, proj.a_scale,
                    proj.out_features, pbias, res)
                return out.reshape(b, H, W, c).permute(0, 3, 1, 2)
            a = self._route1_score(x, b, T, nh, hd, _FUSE_SHIFT)
            out_tok = self._proj_with_residual(a, x_in_tok, b, T, c)
            return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)
        # ---- GN -> qkv: fused GN->qkv (fp16) where eligible, else GroupNorm + qkv Linear ----
        if ((self._fuse_gn_qkv or self._fuse_gn_qkv_i8) and _HAS_FUSED_GN_QKV and x.dtype == torch.float16
                and (T % _FUSE_TILE_M) == 0 and (c % 8) == 0):
            self._ensure_fused()
            qkv_img = _mc.fused_gn_qkv(x, self._fused_conv_w, self._fused_epi_bias,
                                       self.norm.num_groups, self.norm.eps, _FUSE_SHIFT)
            qkv = qkv_img.permute(0, 2, 3, 1).reshape(b, T, nh, 3, hd)
        else:
            # Falls back to the base class's GN->qkv fusion (group_norm_silu_quantize_nhwc ->
            # W8A8/W4A4 GEMM, skipping qkv's own separate quantize_act_int8 launch) when eligible
            # (calibrated static int8/int4 qkv, no K-pad, GN-native layout); else plain GroupNorm +
            # qkv Linear (which pays quantize_act_int8 as its own launch). Reachable mainly for
            # blocks the fp16 fused_gn_qkv branch above doesn't cover (T%128!=0 or c%8!=0, e.g.
            # hd48/T64) and during the qkv scale calibration window.
            # Explicit None test, not `or`: these are tensors, and `or` invokes __bool__ on a
            # multi-element tensor -- "Boolean value of Tensor with more than one value is ambiguous".
            qkv = self._qkv_from_gn_modiff_fused(x, b, T, c, nh, hd)
            if qkv is None:
                qkv = self._qkv_from_gn(x, b, T, c, nh, hd)
        if qkv is not None and qkv.dtype == torch.int8:
            # Route (b) emitted the packed int8 qkv directly. Straight to flash's gather path; no
            # aq_* pass, no fp16 qkv ever materialised.
            hd_pad = ((hd + 31) // 32) * 32
            a = _mc.flash_attn_int8_packed_vt(
                qkv.contiguous(), self._fq_svv[:hd].contiguous().float(), hd_pad,
                float(self._fq_sqc), float(self._fq_skc), self.scale)
            out_tok = self._proj_with_residual(a, x_in_tok, b, T, c)
            return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)
        if T < 64:
            self._observe_small_int8_scales(qkv)
        # ---- Score path (QKᵀ + softmax + AV): fused int8/int4 flash (scores in SRAM) when the
        # fixed route selects it; else fp16 MATH SDPA. INT8 always selects quantized flash when
        # eligible; any remaining automatic selection here belongs only to the INT4 experiment. ----
        use_flash = self._resolve_flash(qkv, T)
        # ROUTE (a), MoDiff only, opt-in via MODIFF_FUSE_QKV_PACKED=1.
        #
        # Under MoDiff the qkv GEMM must emit fp16 (o_hat is fp16 state), which makes all 21 blocks
        # qout-ineligible, which brings back three `aq_*` re-quantize passes -- 4.60 ms/step of the
        # `attn_quantize` bucket that is 0.00 in every non-MoDiff arm
        # (docs/profile_kernels_layers_2026-08-11).
        #
        # `flash_attn_int8_packed_vt` branches on the dtype of its packed qkv: kHalf runs
        # run_flash_packed<__half>, which quantizes ON LOAD, and int8 runs the gather. Feeding it the
        # fp16 qkv therefore removes the three aq_* passes with no GEMM change at all, and needs no
        # new calibration -- the aq_* kernels it replaces already quantize this exact tensor with
        # _fq_sqc / _fq_skc / _fq_svv, so it moves where the quantize happens, not what it computes.
        #
        # MEASURED AND REFUTED, 2026-08-11: batch 128, 200 steps, paired on one model, this is
        # 121.25 vs 103.23 ms/step -- **18.0 ms SLOWER**, against a 4.60 ms bucket it was meant to
        # remove. The reasoning missed that flash RE-READS k and v for every query block, so
        # "quantize on load" means quantize O(T/block) times rather than once. Quantize-once-then-
        # gather is why the aq_* kernels exist at all; this route pays their cost repeatedly instead.
        #
        # That makes the int8 input the only viable form -- it takes the GATHER path -- which is
        # route (b): gemm_w8a8_awq_o_hat_out_i8 emitting per-column-scaled int8 straight into this
        # same entry point. Its foundation is verified in integration/tests/test_qkv_o_hat_out_i8.py.
        # Kept OFF and kept HERE because the measurement is the argument for (b); deleting it would
        # leave (b) looking like an arbitrary choice between two routes rather than the only one.
        #
        # (Latent relL2 between the arms was 0.01710, larger than the "tiny" predicted from identical
        # scales. Not chased: the route is refuted on speed, so the number has no consumer.)
        if (use_flash and not self._qout_eligible() and self.bits == 8
                and os.environ.get("MODIFF_FUSE_QKV_PACKED") == "1"
                and getattr(self, "_fq_frozen2", False)
                and _mc is not None and hasattr(_mc, "flash_attn_int8_packed_vt")
                and qkv.dtype == torch.float16 and getattr(self, "_fq_svv", None) is not None):
            hd_pad = ((hd + 31) // 32) * 32
            a = _mc.flash_attn_int8_packed_vt(
                qkv.contiguous(), self._fq_svv[:hd].contiguous().float(), hd_pad,
                float(self._fq_sqc), float(self._fq_skc), self.scale)
            out_tok = self._proj_with_residual(a, x_in_tok, b, T, c)
            return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)
        if use_flash and self._qout_eligible():
            # Fused flash+proj: flash emits the proj-quantized token-major output directly (no fp16
            # attn-output materialize, no separate quantize_attn_out pass) -> gemm_wXaX_awq_bias_res.
            out_tok = self._flash_proj_qout(qkv, x_in_tok, b, T, c)
        else:
            a = self._score_path(qkv, use_flash)                         # [b,nh,T,hd] head-major
            out_tok = self._proj_with_residual(a, x_in_tok, b, T, c)     # proj (+bias +residual), fused
        return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)

    def _qkv_from_gn_modiff_fused(self, x, b, T, c, nh, hd):
        """GN + delta-quantize + a_hat update in ONE kernel for a MoDiff qkv. None if not applicable.

        With MODIFF_LINEAR=1 the qkv projection needs Q(GN(x) - a_hat), not Q(GN(x)), so the fused
        GN+quantize kernel this path used (group_norm_silu_quantize_nhwc) no longer fits and the whole
        chain degrades into three separate full-tensor passes. Profiled at batch 128 (see
        docs/delta_clip_2026-08-06/data/fusion_profile.json), those three cost:

            group_norm_silu_nhwc_kernel                          5.94 ms   GN+SiLU, quantize stripped out
            delta_absmax_fp16_kernel                             3.97 ms   the dynamic delta scale
            static_quantize_and_update_ahat_..._half_cache_vec2   5.52 ms   quantize + a_hat write
                                                                -------
                                                                15.43 ms

        The kernel that does all three at once already exists and is already in production on the conv
        path -- group_norm_silu_delta_quantize_nhwc, i.e. gn_apply_delta_quantize_flat_vec2_kernel,
        which serves 62 conv layers for 8.85 ms total. It was simply never wired to the attention qkv.

        Layout is why this is a view and not a copy: a_hat is [M, K] = [b*T, c] row-major, which is
        byte-identical to a channels_last [b, c, H, W] (both are NHWC-physical), so the 4D view the GN
        kernel requires shares storage with the 2D buffer the projection owns. The int8 codes come back
        NHWC too, so the flattening back to [b*T, c] for the GEMM is also free.

        The GEMM is unchanged: gemm_w8a8_awq_o_hat already runs on this path, and its epilogue contract
        (o_hat accumulate, THEN bias, THEN residual) is what keeps the temporal state bias-free.

        Returns None -- falling through to the unfused path -- whenever a precondition is unmet, which
        includes the un-seeded first modulated step. Seeding stays with QuantLinearWxAx.forward so
        there is exactly one place that establishes a_hat/o_hat.
        """
        qkv = self.qkv
        if (_QuantLinearWxAx is None or not isinstance(qkv, _QuantLinearWxAx)
                or not getattr(qkv, "modiff", False) or qkv.bits != 8
                or x.dtype != torch.float16
                or not hasattr(_mc, "group_norm_silu_delta_quantize_nhwc")
                or not hasattr(_mc, "gemm_w8a8_awq_o_hat")):
            return None
        # Un-seeded (first modulated step) or a shape change: let the projection's own forward seed.
        if qkv.a_hat is None or qkv.o_hat is None or qkv.a_hat.shape[0] != b * T:
            return None
        # int4 pads K for the AWQ layout; the 4D view below assumes width == c, so skip if padded.
        if qkv._awqt_K != qkv.in_features or qkv.in_features != c:
            return None
        gw, gb = self._gn_params(x.dtype)
        if gw is None or gb is None:
            return None
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        qkv._ensure_modiff_bufs(x)
        # [b*T, c] -> channels_last [b, c, H, W], sharing storage. H*W == T, and H is not passed in,
        # so recover it from x rather than assuming a square latent.
        Himg, Wimg = x.shape[2], x.shape[3]
        a_hat4 = qkv.a_hat.view(b, Himg, Wimg, c).permute(0, 3, 1, 2)
        if a_hat4.shape != x.shape:
            return None
        e32 = qkv._empty_f32
        # Refresh schedule, same as the projection's own path. Passing EMPTY reduction buffers makes
        # the kernel skip its absmax pass entirely and quantize with whatever `qkv._scale` already
        # holds -- which on a reuse step is the value the last refresh wrote. That is the same
        # contract the conv path uses (_delta_gn_dynamic_args returns empties on a reuse step), so
        # this needs no kernel change.
        #
        # `_step` is advanced HERE and not in QuantLinearWxAx.forward, because on this fused path the
        # projection's forward never runs -- the GN kernel and the o_hat GEMM are called directly.
        # Advancing it in both places would double-count and halve the effective K.
        qkv._step += 1
        dyn = ((qkv._absmax, qkv._scale, qkv._inv_scale, qkv._retire)
               if qkv._delta_should_refresh() else (e32, e32, e32, e32))
        codes = _mc.group_norm_silu_delta_quantize_nhwc(
            x, gw, gb, a_hat4, self.norm.num_groups, self.norm.eps,
            False,                                   # apply_silu: attention GN has no SiLU
            qkv._scale, e32, e32, e32,               # scale (dynamic mode overwrites it), smooth, mod
            *dyn,
            float(qkv.Q), False, 1.0)                # Q_level, report_next, safety
        codes2d = codes.permute(0, 2, 3, 1).reshape(b * T, c)
        # ROUTE (b): one pass advances the fp16 o_hat state AND emits per-column-scaled int8, which
        # flash_attn_int8_packed_vt consumes on its GATHER path -- the three aq_* re-quantize kernels
        # (4.60 ms/step) disappear. Route (a), feeding the same entry point fp16, was measured 18.0 ms
        # SLOWER because flash re-reads k/v per query block and its kHalf path re-quantizes on every
        # load; see the refutation at the branch in _forward_routes. int8 is the only viable form.
        #
        # The GEMM's column order is already (nh, 3, hd) -- the reshape below is the same one the fp16
        # path does -- so the int8 output IS the packed flash buffer with no data movement.
        if (self._qkv_i8_ok(T) and _mc is not None
                and hasattr(_mc, "gemm_w8a8_awq_o_hat_out_i8")):
            inv_out = self._qkv_inv_out_scale(qkv, nh, hd)
            if inv_out is not None:
                oi8 = _mc.gemm_w8a8_awq_o_hat_out_i8(
                    codes2d, qkv.qweight, qkv.w_scale, qkv._inv_scale, qkv.out_features,
                    qkv.o_hat, qkv.bias if qkv.bias is not None else qkv._empty_h, inv_out)
                # [M, out_features] since the 2026-08-11 allocation fix -- no padded tail to slice.
                return oi8.reshape(b, T, nh, 3, hd)
        out = _mc.gemm_w8a8_awq_o_hat(
            codes2d, qkv.qweight, qkv.w_scale, qkv._inv_scale, qkv.out_features,
            qkv.o_hat, qkv._empty_h,                 # qkv has no residual
            qkv.bias if qkv.bias is not None else qkv._empty_h)
        return out.reshape(b, T, nh, 3, hd)

    def _qkv_i8_ok(self, T):
        """Route (b) eligibility: int8 block, a flash-eligible shape, frozen scales, opt-in.

        THREE constraints, and the gate needs all of them because the int8 branch in _forward_routes
        RAISES rather than falling back once this returns True:

          * hd % 16 == 0 -- a MEASURED performance gate since 2026-08-12, no longer a legality one.
            The kernel grew an 8-byte cp.async staging variant so hd=24 (24 B/token) is now legal, and
            it LOSES: 2.930 ms against production's 2.023 at T=1024/batch 128, i.e. 2.11x the mma
            kernel against a 1.44x break-even, about -4.5 ms/step over the 5 blocks. Narrow
            transactions plus the .ca path (cp.async.cg is 16-byte only) cost more than the aq_*
            passes they would delete, and T=1024 re-reads k/v far more often than T=64 does. Keeping
            the condition as hd%16 rather than adding a separate flag because hd=24 is the only
            8-but-not-16 width in this model, so the two are the same set here. Do not widen it
            without re-running integration/tests/bench_flash_packed_vs_unpacked.py.
          * _flash_shape_ok -- the same hd<=48 / T%64 constraint _resolve_flash applies. Without it
            this gate admitted the six hd=96/T=16 blocks, whose hd_pad=128 exceeds FA_MMA_MAXHD; the
            resulting "mma-eligible shapes only" was mis-recorded as hd=48 failing, which made route
            (b) look impossible on every shape. Those blocks never ran flash at all.
          * frozen scales -- _fq_sqc/_fq_skc/_fq_svv are what the GEMM's per-column out scale is
            built from, so the calibration window must be over.

        Net: the 10 hd=48 blocks (T=256 and T=64) take this route. Measured worth on those shapes,
        with the aq_* kernels removed and the packed gather kernel paying part of it back:
        +0.79 ms/step at batch 128 (integration/tests/bench_flash_packed_vs_unpacked.py), against
        1.80x and 1.49x of the mma kernel's time and a 2.0x break-even.
        """
        return (os.environ.get("MODIFF_FUSE_QKV_I8") == "1" and self.bits == 8
                and (self.head_dim % 16) == 0
                and self._flash_shape_ok(T)
                and getattr(self, "_fq_frozen2", False)
                and getattr(self, "_fq_svv", None) is not None
                and hasattr(_mc, "flash_attn_int8_packed_vt"))

    def _qkv_inv_out_scale(self, qkv, nh, hd):
        """[N_pad] f32 reciprocal output scales, at the interleaved (nh, 3, hd) column stride.

        Column c belongs to q/k/v by `(c // hd) % 3`, and within v to head-channel `c % hd`. Built
        once and cached: it depends only on the frozen flash scales. Padded columns get 1.0, which the
        GEMM never writes now that its allocation is [M, n_out].
        """
        cached = getattr(self, "_qkv_inv_out", None)
        if cached is not None and cached.numel() == qkv._awqt_N:
            return cached
        try:
            sv = self._fq_svv[:hd].float()
            inv = torch.ones(qkv._awqt_N, device=qkv.qweight.device, dtype=torch.float32)
            c = torch.arange(3 * hd * nh, device=inv.device)
            which, ch = (c // hd) % 3, c % hd
            per = torch.where(which == 0, 1.0 / float(self._fq_sqc),
                              torch.where(which == 1, 1.0 / float(self._fq_skc),
                                          1.0 / sv.to(inv.device)[ch]))
            inv[:3 * hd * nh] = per.float()
            self._qkv_inv_out = inv.contiguous()
            return self._qkv_inv_out
        except Exception:
            return None

    def _qout_eligible(self):
        """True when the frozen flash path can emit the proj-quantized output directly (item B):
        packed quantize + the qout kernels available, and proj is a calibrated-static, _use_bias_res
        _QuantLinearWxAx whose bits match the block, with the flash Q/K/V scales already frozen."""
        if not (_HAS_PACKED and _HAS_FLASH_QOUT and self._fuse_proj_quant):
            return False
        proj = self.proj
        if not (_QuantLinearWxAx is not None and isinstance(proj, _QuantLinearWxAx)
                and getattr(proj, "_use_bias_res", False) and proj.a_scale is not None
                and not getattr(proj, "_calib", False) and proj.bits == self.bits):
            return False
        return getattr(self, "_fq_frozen2" if self.bits == 8 else "_fq4_frozen", False)

    def _flash_proj_qout(self, qkv, x_in_tok, b, T, c):
        """Steady-state fused flash+proj (frozen scales): the flash kernel writes the attention
        output already proj-quantized token-major (flash_attn_intX_vt_qout), fed straight to
        gemm_wXaX_awq_bias_res (+bias +residual). Bit-identical to
        _score_path(flash) -> _proj_with_residual, but skips the fp16 attn-output round-trip and the
        quantize_attn_out kernel. Only valid under _qout_eligible()."""
        nh, hd = self.num_heads, self.head_dim
        proj = self.proj
        qkv = qkv.contiguous()
        # gemm_*_awq_bias_res requires an fp16 residual. No-op in production; load-bearing while
        # the conv layers are uncalibrated, where they emit fp32 (see FINDINGS 2026-08-03).
        res = x_in_tok.reshape(b * T, c).half().contiguous()
        bias = proj.bias if proj.bias is not None else qkv.new_empty(0)
        if self.bits == 8:
            hd_pad = ((hd + 31) // 32) * 32
            def int8_qin_xq():
                compact_hd24 = (
                    T == 1024 and hd == 24
                    and self._int8_kv_compact24)
                k_storage_hd = hd_pad
                v_storage_hd = hd if compact_hd24 else hd_pad
                ki_, vt_, sv_ = _mc.quantize_attn_kv_packed_static(
                    qkv, nh, T, hd, k_storage_hd, v_storage_hd, 8,
                    self._fq_skc, self._fq_svv[:v_storage_hd].contiguous())
                return _mc.flash_attn_int8_qpacked_kv_static_qout(
                    qkv, ki_.view(b, nh, T, k_storage_hd),
                    vt_.view(b, nh, v_storage_hd, T), sv_[:hd].contiguous(),
                    hd_pad, self._fq_sqc, self._fq_skc,
                    self.scale, proj.a_scale)

            # Optional packed diagnostic route. There is no runtime autotune: selecting its
            # environment flag means always using it for a supported shape.
            if self._flash_packed:
                sv_hd = self._fq_svv[:hd].contiguous()
                persistent_supported = (
                    hasattr(_mc, "flash_attn_int8_packed_persistent_qout")
                    and ((T == 1024 and hd == 24)
                         or (T in (64, 256) and hd == 48)))
                persistent_ok = persistent_supported and self._int8_persistent
                packed_fn = (
                    (lambda: _mc.flash_attn_int8_packed_persistent_qout(
                        qkv, sv_hd, hd_pad, self._fq_sqc, self._fq_skc,
                        self.scale, proj.a_scale))
                    if persistent_ok else
                    (lambda: _mc.flash_attn_int8_packed_vt_qout(
                        qkv, sv_hd, hd_pad, self._fq_sqc, self._fq_skc,
                        self.scale, proj.a_scale)))
                xq = packed_fn()
                out = _mc.gemm_w8a8_awq_bias_res(
                    xq, proj.qweight, proj.w_scale, proj.a_scale,
                    proj.out_features, bias, res)
                return out.reshape(b, T, c)
            # With the qout store fused, quantizing Q once in each consuming CTA wins at every
            # production flash shape: 1024, 256, and 64 tokens. This removes the Q tensor allocation
            # and half of the Q/K preparation pass while retaining prequantized K/V.
            if T >= 64 and hasattr(_mc, "flash_attn_int8_qpacked_kv_static_qout"):
                xq = int8_qin_xq()
                out = _mc.gemm_w8a8_awq_bias_res(
                    xq, proj.qweight, proj.w_scale, proj.a_scale,
                    proj.out_features, bias, res)
                return out.reshape(b, T, c)
            qi, ki, vt, sq, sk, sv = _mc.quantize_attn_qkv_packed_static(
                qkv, nh, T, hd, hd_pad, hd_pad, 8, self._fq_sqc, self._fq_skc, self._fq_svv)
            qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad); vt = vt.view(b, nh, hd_pad, T)
            sq = sq.view(b, nh, T).contiguous(); sk = sk.view(b, nh, T).contiguous()
            sv = sv[..., :hd].contiguous().view(b, nh, hd)
            if hasattr(_mc, "flash_attn_int8_vt_static_qout"):
                xq = _mc.flash_attn_int8_vt_static_qout(
                    qi, ki, vt, sv, self._fq_sqc, self._fq_skc, self.scale, proj.a_scale)
            else:
                xq = _mc.flash_attn_int8_vt_qout(
                    qi, ki, vt, sq, sk, sv, self.scale, proj.a_scale)
            out = _mc.gemm_w8a8_awq_bias_res(xq, proj.qweight, proj.w_scale, proj.a_scale,
                                             proj.out_features, bias, res)
        else:
            hdp4, hdp_v = 64, ((hd + 31) // 32) * 32
            use_compact_i4 = (self._int4_compact_static
                              and hasattr(_mc, "quantize_attn_qkv_packed_static_compact"))
            # hd=24 is a poor fit for native m16n8k64.s4: 62.5% of QK lanes are padding.
            # Keep the exact signed-int4 Q/K grid but store it unpacked and use m16n8k32.s8;
            # the attention output remains packed int4 for the W4 projection.
            if (T == 1024 and hd == 24
                    and hasattr(_mc, "flash_attn_i4values_i8mma_qpacked_kv_static_qout")):
                hdpi8 = 32
                ki, vt, sv = _mc.quantize_attn_kv_packed_static(
                    qkv, nh, T, hd, hdpi8, hdpi8, _QK_I4_VALUES_I8_MMA,
                    self._fq4_skc, self._fq4_svv[:hdpi8])
                ki = ki.view(b, nh, T, hdpi8)
                vt = vt.view(b, nh, hdpi8, T)
                sv = sv[:hd].contiguous()
                xq = _mc.flash_attn_i4values_i8mma_qpacked_kv_static_qout(
                    qkv, ki, vt, sv, hdpi8, self._fq4_sqc, self._fq4_skc,
                    self.scale, proj.a_scale, proj._awqt_K)
                out = _mc.gemm_w4a4_awq_bias_res(
                    xq, proj.qweight, proj.w_scale, proj.a_scale,
                    proj._awqt_K, proj.out_features, bias, res)
                return out.reshape(b, T, c)
            def i4_qin_xq():
                k4_, vt_, sv_ = _mc.quantize_attn_kv_packed_static(
                    qkv, nh, T, hd, hdp4, hdp_v, _QK_I4_PACKED,
                    self._fq4_skc, self._fq4_svv)
                return _mc.flash_attn_int4_qpacked_kv_static_qout(
                    qkv, k4_.view(b, nh, T, -1), vt_.view(b, nh, hdp_v, T),
                    sv_[:hd].contiguous(), hdp4, self._fq4_sqc, self._fq4_skc,
                    self.scale, proj.a_scale, proj._awqt_K)

            def i4_ref_xq():
                if use_compact_i4:
                    q4_, k4_, vt_, sv_ = _mc.quantize_attn_qkv_packed_static_compact(
                        qkv, nh, T, hd, hdp4, hdp_v, _QK_I4_PACKED,
                        self._fq4_sqc, self._fq4_skc, self._fq4_svv)
                else:
                    q4_, k4_, vt_, _sq4, _sk4, sv_ = _mc.quantize_attn_qkv_packed_static(
                        qkv, nh, T, hd, hdp4, hdp_v, _QK_I4_PACKED,
                        self._fq4_sqc, self._fq4_skc, self._fq4_svv)
                sv_arg = sv_[..., :hd].contiguous()
                if sv_arg.dim() != 1:
                    sv_arg = sv_arg.view(b, nh, hd)
                return _mc.flash_attn_int4_vt_static_qout(
                    q4_.view(b, nh, T, -1), k4_.view(b, nh, T, -1),
                    vt_.view(b, nh, hdp_v, T),
                    sv_arg,
                    hdp4, self._fq4_sqc, self._fq4_skc, self.scale,
                    proj.a_scale, proj._awqt_K)

            # Fixed route, no runtime A/B (see the MODIFF_INT4_Q_IN_FLASH note in __init__ for the
            # microbenchmarks that decide it). Only T256/T64 reach here: T1024/hd24 has already
            # returned above, via either the direct-layout epilogue or the i4values short-circuit.
            qin_available = (T >= 64
                             and hasattr(_mc, "flash_attn_int4_qpacked_kv_static_qout"))
            use_qin = qin_available and self._int4_qin_on
            if use_qin:
                xq = i4_qin_xq()
                out = _mc.gemm_w4a4_awq_bias_res(
                    xq, proj.qweight, proj.w_scale, proj.a_scale,
                    proj._awqt_K, proj.out_features, bias, res)
                return out.reshape(b, T, c)
            if use_compact_i4:
                q4, k4, vt, sv = _mc.quantize_attn_qkv_packed_static_compact(
                    qkv, nh, T, hd, hdp4, hdp_v, _QK_I4_PACKED,
                    self._fq4_sqc, self._fq4_skc, self._fq4_svv)
                sq4 = sk4 = None
            else:
                q4, k4, vt, sq4, sk4, sv = _mc.quantize_attn_qkv_packed_static(
                    qkv, nh, T, hd, hdp4, hdp_v, _QK_I4_PACKED,
                    self._fq4_sqc, self._fq4_skc, self._fq4_svv)
            q4 = q4.view(b, nh, T, -1); k4 = k4.view(b, nh, T, -1); vt = vt.view(b, nh, hdp_v, T)
            sv = sv[..., :hd].contiguous()
            if sv.dim() != 1:
                sv = sv.view(b, nh, hd)
            if hasattr(_mc, "flash_attn_int4_vt_static_qout"):
                xq = _mc.flash_attn_int4_vt_static_qout(
                    q4, k4, vt, sv, hdp4, self._fq4_sqc, self._fq4_skc, self.scale,
                    proj.a_scale, proj._awqt_K)
            else:
                sq4 = sq4.view(b, nh, T).contiguous()
                sk4 = sk4.view(b, nh, T).contiguous()
                sv = sv.view(b, nh, hd)
                xq = _mc.flash_attn_int4_vt_qout(
                    q4, k4, vt, sq4, sk4, sv, hdp4, self.scale,
                    proj.a_scale, proj._awqt_K)   # packed int4 [b*T,K_pad/2]
            out = _mc.gemm_w4a4_awq_bias_res(xq, proj.qweight, proj.w_scale, proj.a_scale,
                                             proj._awqt_K, proj.out_features, bias, res)
        return out.reshape(b, T, c)

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
            # gemm_wXaX_awq_bias_res requires an fp16 residual and raises
            # `residual fp16 [M,n_out]` otherwise. Enforce the contract here rather than relying on
            # every upstream emitter: in production x_in_tok is already fp16 so `.half()` is a no-op
            # (torch returns self), but while the conv layers are UNCALIBRATED they emit fp32 and
            # this call is the first thing that sees it. That made MoDiff-mode conv calibration
            # impossible -- _calibrate_int8 crashed inside its own sampling pass, so the only way to
            # get scales was to load a file, and this tree's stub checkpoint means a saved file
            # describes a different random network than the one loading it. See
            # docs/modiff_correctness_2026-08-03/FINDINGS.md.
            res = x_in_tok.reshape(b * T, c).half().contiguous()
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
            # Be dtype-transparent: the GEMM always returns fp16, but during the uncalibrated
            # calibration window the surrounding pipeline is fp32, and handing it fp16 trips the
            # next non-quantized conv ("Input type (c10::Half) and bias type (float) should be the
            # same" -- those convs run with autocast locally disabled, so nothing casts for them).
            # No-op in production, where x_in_tok is already fp16.
            return out.reshape(b, T, c).to(x_in_tok.dtype)
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
