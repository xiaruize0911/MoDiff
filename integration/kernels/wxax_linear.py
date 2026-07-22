"""Weight+activation int quantized Linear (W8A8 / W4A4), static scales, for the
UNet Linear-equivalent layers. One strategy parameterized by bit-width:
per-output-channel symmetric weights + static per-tensor symmetric activations,
running the AWQ-tiling int tensor-core GEMM (`gemm_w8a8_awq` / `gemm_w4a4_awq`).

Weights are quantized once at construction (static). The activation scale is
static (set by calibration via `set_a_scale`); if unset, a dynamic per-tensor
absmax scale is used as a fallback (non-CUDA-graph).

Eligibility (else keep fp16, handled by the converter): out%64==0 and
in%32==0 (int8) / in%64==0 (int4).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sole Linear-GEMM backend: the AWQ-tiling ports gemm_w8a8_awq / gemm_w4a4_awq
# (ldmatrix + XOR bank-swizzle + 128-wide-N tile). These replaced an older hand-written
# family (gemm_w8a8/gemm_w4a4) and AWQ's external reference as the production path on
# 2026-07-18 -- they beat every prior option at the kernel level (int8 vs fp16 1.1-1.4x
# and vs AWQ-ref 4/6 shapes; int4 vs fp16 up to 2.29x, no AWQ int4 kernel exists). See
# docs/quant_speedup_vs_fp16_2026-07-16/. Weights are padded once at construction to the
# kernel tile requirement (N%128 for both; K%64 int8 / K%128 int4); activations get the
# matching zero-pad at call time.
try:
    import modiff_cutlass as _mc
    _HAS = hasattr(_mc, "gemm_w8a8_awq") and hasattr(_mc, "gemm_w4a4_awq")
    # fused bias(+residual) epilogue -> removes the separate `out + bias` / `x + proj(out)` add kernels
    _HAS_BIAS_RES = hasattr(_mc, "gemm_w8a8_awq_bias_res") and hasattr(_mc, "gemm_w4a4_awq_bias_res")
except Exception:
    _mc = None
    _HAS = False
    _HAS_BIAS_RES = False


def _pack4(q):  # q int8 [...,K] in [-7,7] -> [...,K/2] int8 (2 int4/byte, low nibble = even)
    q = q.to(torch.int32)
    lo = q[..., 0::2] & 0xF
    hi = q[..., 1::2] & 0xF
    return ((lo | (hi << 4)).to(torch.int8)).contiguous()


def _eligible(in_f, out_f, bits):
    return _HAS and (out_f % 64 == 0) and (in_f % (64 if bits == 4 else 32) == 0)


class QuantLinearWxAx(nn.Module):
    def __init__(self, lin: nn.Linear, bits: int, modiff: bool = False):
        super().__init__()
        assert bits in (4, 8) and _eligible(lin.in_features, lin.out_features, bits)
        self.bits = bits
        self.Q = 127 if bits == 8 else 7
        self.modiff = modiff
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        W = lin.weight.detach().float()                       # [N,K]
        s = (W.abs().amax(1).clamp_min(1e-8) / self.Q)        # per-output-channel [N]
        Wq = torch.round(W / s.unsqueeze(1)).clamp(-self.Q, self.Q).to(torch.int8)
        # The AWQ-tiling ports need N%128 and K%(64 int8 / 128 int4). Pad the logical weight to
        # [N_pad, K_pad] with zeros (zero rows/cols contribute 0 to the dot product), pad the fp32
        # wscale with 1.0 (padded output cols are sliced off after the GEMM), then (int4) pack two
        # nibbles/byte. Activations get the matching K zero-pad at call time in _gemm.
        Kmul = 64 if bits == 8 else 128
        self._awqt_K = ((self.in_features + Kmul - 1) // Kmul) * Kmul
        self._awqt_N = ((self.out_features + 127) // 128) * 128
        Wq_p = F.pad(Wq, (0, self._awqt_K - self.in_features, 0, self._awqt_N - self.out_features))
        s_p = F.pad(s, (0, self._awqt_N - self.out_features), value=1.0)
        self.register_buffer("qweight", _pack4(Wq_p) if bits == 4 else Wq_p.contiguous())
        self.register_buffer("w_scale", s_p.to(torch.float32).contiguous())
        self.register_buffer("bias", lin.bias.detach().half().contiguous() if lin.bias is not None else None)
        self.a_scale = None                                    # static activation scale (calibrated)
        self._calib = False
        self._amax = 0.0
        self.a_hat = None                                      # MoDiff temporal caches (modiff mode)
        self.o_hat = None
        # int8-OUTPUT GEMM (output-fusion fix): the GEMM writes int8 (half the M*N output write) and the
        # per-column dequant is fused into the bias add (out = int8·out_scale + bias, one epilogue op that
        # replaces the plain +bias). Needs a calibrated per-column output scale. Engages only when bias is
        # present (so dequant folds into it, not an extra pass) and not modiff. Gate MODIFF_LINEAR_OUT_I8=1.
        self._out_i8 = (not modiff and lin.bias is not None
                        and os.environ.get("MODIFF_LINEAR_OUT_I8") == "1"
                        and hasattr(_mc, "gemm_w8a8_awq_out_i8"))
        self._out_amax = None      # [out_features] calibrated per-column output absmax
        self._inv_out_scale = None # [N_pad] f32 = 127/absmax (padded cols = 1)
        self._out_scale = None     # [out_features] f16 = absmax/127 (dequant multiplier)
        # fused bias(+residual) epilogue: the GEMM adds bias[n] (and an optional residual[m,n]) in its
        # store, replacing the separate `out + bias` elementwise add (and, for attention proj, the
        # `x + proj(out)` residual add). Not for modiff (o_hat accumulation) or the int8-output path.
        self._use_bias_res = _HAS_BIAS_RES and not modiff and not self._out_i8

    def set_a_scale(self, s):
        self.a_scale = float(s)

    def reset_modiff(self):
        self.a_hat = None
        self.o_hat = None

    def _gemm(self, xf, a_scale):
        """xf: fp16 [M,K] -> fp16 [M,N] (quantize + AWQ-tiling int GEMM). Zero-pad the
        activation K to the kernel tile width, quantize, GEMM, slice N back to out_features.
        The kernel returns a fresh output tensor each call (no scratch aliasing)."""
        if self._awqt_K != self.in_features:
            xf = F.pad(xf, (0, self._awqt_K - self.in_features)).contiguous()
        if self.bits == 8:
            xq = _mc.quantize_act_int8(xf, a_scale)
            if self._awqt_N != self.out_features:   # padded N -> write unpadded (no slice-copy)
                return _mc.gemm_w8a8_awq_nout(xq, self.qweight, self.w_scale, a_scale, self.out_features)
            return _mc.gemm_w8a8_awq(xq, self.qweight, self.w_scale, a_scale)
        xq = _mc.quantize_act_int4_pack(xf, a_scale)
        if self._awqt_N != self.out_features:   # padded N -> write unpadded (no slice-copy)
            return _mc.gemm_w4a4_awq_nout(xq, self.qweight, self.w_scale, a_scale, self._awqt_K, self.out_features)
        return _mc.gemm_w4a4_awq(xq, self.qweight, self.w_scale, a_scale, self._awqt_K)

    def _gemm_bias_res(self, xf, a_scale, rflat):
        """xf: fp16 [M,K] -> fp16 [M,out_features] with bias (and optional residual rflat [M,out])
        added in the GEMM epilogue. Uses the unpadded-store form (n_out=out_features)."""
        empty = xf.new_empty(0)
        bias = self.bias if self.bias is not None else empty
        res = rflat if rflat is not None else empty
        if self._awqt_K != self.in_features:
            xf = F.pad(xf, (0, self._awqt_K - self.in_features)).contiguous()
        if self.bits == 8:
            xq = _mc.quantize_act_int8(xf, a_scale)
            return _mc.gemm_w8a8_awq_bias_res(xq, self.qweight, self.w_scale, a_scale, self.out_features, bias, res)
        xq = _mc.quantize_act_int4_pack(xf, a_scale)
        return _mc.gemm_w4a4_awq_bias_res(xq, self.qweight, self.w_scale, a_scale, self._awqt_K, self.out_features, bias, res)

    def can_from_int8(self):
        """True when forward_from_int8 is usable: W8A8, calibrated static a_scale, no K-pad
        (in_features % 64 == 0 -> the GN->int8 kernel's C channels feed the AWQ GEMM directly),
        and the fused-bias/residual epilogue path (not modiff / not int8-output)."""
        return (self.bits == 8 and self.a_scale is not None and self._awqt_K == self.in_features
                and self._use_bias_res)

    def forward_from_int8(self, x_i8, residual=None):
        """Fast path for the fused GN->qkv-quantize: the int8 activation [M, in_features] is
        produced upstream by group_norm_silu_quantize_nhwc (so the per-layer quantize_act_int8 is
        skipped). Goes straight to the AWQ W8A8 GEMM with the fused bias(+residual) epilogue.
        Requires can_from_int8()."""
        a_scale = self.a_scale
        rflat = residual.reshape(-1, self.out_features).half().contiguous() if residual is not None else None
        empty = self.bias.new_empty(0) if self.bias is not None else x_i8.new_empty(0, dtype=torch.float16)
        bias = self.bias if self.bias is not None else empty
        res = rflat if rflat is not None else empty
        out = _mc.gemm_w8a8_awq_bias_res(x_i8, self.qweight, self.w_scale, a_scale, self.out_features, bias, res)
        return out

    def can_from_int4(self):
        """True when forward_from_int4 is usable: W4A4, calibrated static a_scale, and the
        fused-bias/residual epilogue path. Unlike can_from_int8 this does NOT require
        _awqt_K == in_features -- forward_from_int4 zero-pads the PACKED activation up to the
        K-tile (C=192 -> 256). The weight's padded K-channels are zero, so the padded
        activation nibbles contribute 0 to the dot product regardless of value."""
        return (self.bits == 4 and self.a_scale is not None and self._use_bias_res)

    def forward_from_int4(self, x_i4, residual=None):
        """int4 counterpart of forward_from_int8 for the fused GN->qkv-quantize path: the PACKED
        int4 activation [M, in_features/2] is produced upstream by group_norm_silu_quantize_pack_nhwc
        (so the per-layer quantize_act_int4_pack is skipped). Zero-pads the packed activation to
        _awqt_K/2 bytes when K-padded (append zero-nibble channels), then runs the AWQ W4A4 GEMM
        with the fused bias(+residual) epilogue. Requires can_from_int4()."""
        if self._awqt_K != self.in_features:
            x_i4 = F.pad(x_i4, (0, (self._awqt_K - self.in_features) // 2)).contiguous()
        rflat = residual.reshape(-1, self.out_features).half().contiguous() if residual is not None else None
        empty = self.bias.new_empty(0) if self.bias is not None else x_i4.new_empty(0, dtype=torch.float16)
        bias = self.bias if self.bias is not None else empty
        res = rflat if rflat is not None else empty
        return _mc.gemm_w4a4_awq_bias_res(x_i4, self.qweight, self.w_scale, self.a_scale,
                                          self._awqt_K, self.out_features, bias, res)

    def forward(self, x, residual=None):
        orig = x.shape
        xf = x.reshape(-1, self.in_features).half().contiguous()
        if self._calib:
            self._amax = max(self._amax, float(xf.abs().max()))
        rflat = residual.reshape(-1, self.out_features).half().contiguous() if residual is not None else None

        if self._use_bias_res:
            # FUSED bias(+residual) epilogue -- the default int8/int4 Linear path
            a_scale = self.a_scale if self.a_scale is not None else ((xf.abs().max().item() / self.Q) or 1e-8)
            out = self._gemm_bias_res(xf, a_scale, rflat)
            return out.reshape(*orig[:-1], self.out_features)

        if self.modiff and self.a_hat is not None:
            # MoDiff: quantize only the temporal DELTA (small range -> less int error),
            # accumulate the GEMM into o_hat. o_hat_t = o_hat_{t-1} + Linear(Q(x - a_hat)).
            delta = xf - self.a_hat
            d_scale = (delta.abs().max().item() / self.Q) or 1e-8
            q = torch.round(delta / d_scale).clamp_(-self.Q, self.Q)
            self.a_hat = self.a_hat + q * d_scale                       # dequantized-delta reconstruction
            self.o_hat = self.o_hat + self._gemm(q.half().contiguous(), d_scale)
            out = self.o_hat
        elif self._out_i8 and self._inv_out_scale is not None:
            # int8-OUTPUT GEMM (output-fusion): GEMM writes int8 (half the M*N output write); the
            # per-column dequant is fused into the bias add via dequant_bias_i8 (one epilogue op).
            a_scale = self.a_scale if self.a_scale is not None else ((xf.abs().max().item() / self.Q) or 1e-8)
            xfp = F.pad(xf, (0, self._awqt_K - self.in_features)).contiguous() if self._awqt_K != self.in_features else xf
            if self.bits == 8:
                oi8 = _mc.gemm_w8a8_awq_out_i8(_mc.quantize_act_int8(xfp, a_scale), self.qweight, self.w_scale, a_scale, self._inv_out_scale)
            else:
                oi8 = _mc.gemm_w4a4_awq_out_i8(_mc.quantize_act_int4_pack(xfp, a_scale), self.qweight, self.w_scale, a_scale, self._awqt_K, self._inv_out_scale)
            if self._awqt_N != self.out_features:
                oi8 = oi8[:, :self.out_features].contiguous()
            out = _mc.dequant_bias_i8(oi8, self._out_scale, self.bias)   # fp16 [M,out], dequant + bias fused
            if rflat is not None: out = out + rflat
            return out.reshape(*orig[:-1], self.out_features)
        else:
            a_scale = self.a_scale if self.a_scale is not None else ((xf.abs().max().item() / self.Q) or 1e-8)
            out = self._gemm(xf, a_scale)
            if self._out_i8 and self._calib:                           # calibrate per-column output absmax
                cur = out.detach().abs().amax(0).float()               # [out_features]
                self._out_amax = cur if self._out_amax is None else torch.maximum(self._out_amax, cur)
            if self.modiff:                                            # first step: seed the caches
                q = torch.round(xf / a_scale).clamp_(-self.Q, self.Q)
                self.a_hat = (q * a_scale).contiguous()
                self.o_hat = out.clone()

        out = out.reshape(*orig[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        if rflat is not None:
            out = out + rflat.reshape(*orig[:-1], self.out_features)
        return out


def set_wxax_calibrating(model, flag):
    for m in model.modules():
        if isinstance(m, QuantLinearWxAx):
            m._calib = flag
            if flag:
                m._amax = 0.0


def finalize_wxax_ascale(model):
    """Set each layer's static activation scale from the calibrated absmax, and (for int8-output
    layers) the per-column output scale from the calibrated output absmax."""
    n = 0
    for m in model.modules():
        if isinstance(m, QuantLinearWxAx) and m._amax > 0:
            m.set_a_scale(m._amax / m.Q)
            m._calib = False
            n += 1
            if m._out_i8 and m._out_amax is not None:
                amax = m._out_amax.clamp_min(1e-6).to(m.qweight.device)      # [out_features]
                inv = torch.ones(m._awqt_N, device=amax.device, dtype=torch.float32)
                inv[:m.out_features] = 127.0 / amax
                m._inv_out_scale = inv.contiguous()                          # [N_pad], padded cols = 1
                m._out_scale = (amax / 127.0).to(torch.float32).contiguous() # [out_features] dequant mult
    return n


def convert_linears_to_wxax(module, bits, modiff=False, verbose=False, _prefix=""):
    """Recursively replace eligible nn.Linear with QuantLinearWxAx(bits, modiff). Returns count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if _eligible(child.in_features, child.out_features, bits):
                setattr(module, name, QuantLinearWxAx(child, bits, modiff).to(child.weight.device))
                n += 1
                if verbose:
                    print(f"  wxax({bits},modiff={modiff}): {_prefix}{name} {child.in_features}->{child.out_features}")
            elif verbose:
                print(f"  skip(fp16): {_prefix}{name} {child.in_features}->{child.out_features}")
        else:
            n += convert_linears_to_wxax(child, bits, modiff, verbose, _prefix + name + ".")
    return n


def reset_wxax_modiff(model):
    """Clear the MoDiff temporal caches (call between DDIM samples)."""
    for m in model.modules():
        if isinstance(m, QuantLinearWxAx):
            m.reset_modiff()
