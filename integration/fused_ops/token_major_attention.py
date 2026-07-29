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

Attention lets **PyTorch choose the SDPA backend** (see `_SDPA_CTX`), which in practice
selects the fused flash kernel. It used to pin MATH -- materializing the [BH,T,T] scores --
which cost 79.3 ms/step against flash's 8.9 ms at b128 and, because fp16 is the baseline
every quantized speedup is divided by, inflated all of those ratios. Set
MODIFF_SDPA_BACKEND=math to get the materialized scores back (the quantizable-attention
study needs them). (Fused int8/int4 flash kernels exist in csrc but are NOT wired into this
fp16 pipeline.)

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

# The fp16 attention SDPA backend. DEFAULT: unpinned -- PyTorch picks, which is flash.
# MODIFF_SDPA_BACKEND=math|flash|efficient forces a specific one.
#
# This default was flipped from MATH after measuring what MATH cost as a baseline. Isolated
# ops had already shown flash 1.9-9.2x faster than MATH on this model's shapes (dominated by
# hd24/T1024 = L0's 9.2x) at ~4e-4 rel-L2 -- an order of magnitude below the ~4e-3 rel-L2 the
# int8 path already adds. End-to-end at b128 the gap is 79.3 ms/step (MATH: softmax 39.3 +
# bmm 20.6 + bmm 19.5) vs 8.9 ms/step (flash). Because fp16 is the denominator of every
# quantized speedup in this project, pinning MATH made the fp16 baseline 1.39x slower than
# plain PyTorch and inflated every published ratio (int4 read 1.98x; against unpinned
# PyTorch it is 1.42x). Keeping MATH as the default to preserve continuity with older
# accuracy numbers was not worth systematically overstating the speed results.
#
# MATH remains available for the quantizable-attention study, which needs the materialized
# [BH,T,T] scores that flash never forms.
#
# Reads the env var on every call (not cached at import time): this module is only
# ever imported once per process, so a module-level `os.environ.get(...)` baked into
# a lambda at import time would freeze whatever backend was set on the FIRST call and
# silently ignore any later os.environ["MODIFF_SDPA_BACKEND"] change in the same
# process (e.g. a multi-mode benchmark loop that flips it per iteration).
try:
    from contextlib import nullcontext
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA_BACKEND_MAP = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                         "efficient": SDPBackend.EFFICIENT_ATTENTION}

    def _SDPA_CTX():
        """Do NOT pin a backend by default -- let PyTorch pick (it chooses flash).

        This used to default to MATH, which was a large, hidden cost: MATH materializes the
        [BH,T,T] score matrix, and measured at b128 that path costs softmax 39.3 ms + two
        bmm GEMMs 20.6 + 19.5 = 79.3 ms/step against 8.9 ms/step for PyTorch's flash kernel
        (9x). Since fp16 is the baseline every quantized speedup is divided by, pinning MATH
        inflated every one of those ratios -- the fp16 baseline came out 1.39x slower than
        plain PyTorch. MATH is still selectable with MODIFF_SDPA_BACKEND=math for the
        quantizable-attention study that needs materialized scores.
        """
        env = os.environ.get("MODIFF_SDPA_BACKEND")
        if not env:
            return nullcontext()
        backend = _SDPA_BACKEND_MAP.get(env.lower())
        return sdpa_kernel(backend) if backend is not None else nullcontext()
except Exception:  # pragma: no cover - older torch
    from contextlib import contextmanager, nullcontext

    @contextmanager
    def _SDPA_CTX():
        env = (os.environ.get("MODIFF_SDPA_BACKEND") or "").lower()
        if env == "math":
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False,
                                                enable_math=True):
                yield
        else:
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
    _HAS_FUSED_ATTN_OUT = hasattr(_mc, "quantize_attn_out_int8")
    _HAS_GN_QUANT = hasattr(_mc, "group_norm_silu_quantize_nhwc")
    _HAS_GN_PACK = hasattr(_mc, "group_norm_silu_quantize_pack_nhwc")      # int4 GN->pack fusion (qkv)
    _HAS_ATTN_OUT_I4 = hasattr(_mc, "quantize_attn_out_int4_pack")         # int4 proj transpose+quant+pack
except Exception:
    _HAS_FUSED_GN_QKV = False
    _HAS_FUSED_ATTN_OUT = False
    _HAS_GN_QUANT = False
    _HAS_GN_PACK = False
    _HAS_ATTN_OUT_I4 = False

try:
    from integration.kernels.wxax_linear import QuantLinearWxAx as _QuantLinearWxAx
except Exception:
    _QuantLinearWxAx = None

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
        # Proj-side quantize fusion: fold the mandatory attn-output transpose+reshape copy
        # AND the int8 proj Linear's separate quantize pass into ONE gather+quantize kernel
        # (quantize_attn_out_int8), feeding gemm_w8a8_awq the int8 activation directly. Only
        # engages once proj is a calibrated, non-modiff W8A8 QuantLinearWxAx; otherwise the
        # standard transpose+reshape+proj path runs (also used during calibration so proj's
        # a_scale is captured). Kill-switch MODIFF_FUSE_PROJ_QUANT=0.
        self._fuse_proj_quant = os.environ.get("MODIFF_FUSE_PROJ_QUANT", "1") != "0"
        # qkv-side quantize fusion: fold the int8 qkv Linear's separate quantize pass into the
        # GroupNorm producer via group_norm_silu_quantize_nhwc (GN in fp32 -> int8 directly),
        # feeding gemm_w8a8_awq without the standalone quantize kernel. Engages only for a
        # calibrated non-modiff W8A8 qkv with affine GN; else the standard GN + qkv Linear runs
        # (also during calibration). Kill-switch MODIFF_FUSE_QKV_QUANT=0.
        self._fuse_qkv_quant = os.environ.get("MODIFF_FUSE_QKV_QUANT", "1") != "0"
        self._qkv_inv_scale_t = None          # cached [1] fp32 device tensor = 1/qkv.a_scale
        self._fused_ready = False
        self._fused_conv_w = None    # [3C,1,1,C] fp16 KRSC = qkv.weight * gn.weight
        self._fused_epi_bias = None  # [3C] fp16 = qkv.bias + qkv.w@gn.b - SHIFT*colsum(Wf)

        # fp16 MATERIALIZED attention (bmm QKᵀ -> our softmax kernel -> bmm AV), used by the
        # static-vs-dynamic study so both fp16 modes share ONE code path and differ only in the
        # softmax: dynamic = 2-pass per-row max; static = 1-pass calibrated c (lossless for c ≥ max,
        # since fp16 has no quant grid). MODIFF_FP16_MATERIALIZED=1 enables it; MODIFF_STATIC_SOFTMAX
        # =1 selects the static softmax. Default off -> the familiar SDPA-math path is used.
        self._materialized = os.environ.get("MODIFF_FP16_MATERIALIZED") == "1"
        self._static_softmax = os.environ.get("MODIFF_STATIC_SOFTMAX") == "1"
        self._sm_calib_steps = int(os.environ.get("MODIFF_ATTN_CALIB_STEPS", "8"))
        self._sm_frozen = not self._static_softmax
        self._sm_cn = 0; self._sm_cacc = 0.0; self._sm_c = None

    def _materialized_attn(self, q, k, v):
        """fp16 materialized attention on [b,nh,T,hd]. Dynamic (2-pass max) or, once calibrated,
        static (1-pass constant c). c is calibrated as a safe upper bound (max row-logit over the
        first N forwards + margin) so exp(S-c) ≤ 1 -> fp16-lossless. Returns [b,nh,T,hd]."""
        b, nh, T, hd = q.shape
        BH = b * nh
        qf = q.reshape(BH, T, hd).contiguous()
        kf = k.reshape(BH, T, hd).contiguous()
        vf = v.reshape(BH, T, hd).contiguous()
        S = (torch.bmm(qf, kf.transpose(1, 2)) * self.scale).half()      # [BH,T,T] fp16 logits
        if self._static_softmax and self._sm_frozen:
            P, rs = _mc.attn_softmax_fp16(S, True, self._sm_c)
        else:
            P, rs = _mc.attn_softmax_fp16(S, False, 0.0)
            if self._static_softmax and not self._sm_frozen:
                # calibrate c to the TYPICAL (mean) per-row max, not the trajectory max: diffusion
                # logit scale drifts ~30x across timesteps, so a max-based c underflows exp at low
                # timesteps -> tiny rowsum -> O blows up. A mean-based c keeps exp(S-c) in range;
                # the clamp-to-1 in the kernel handles the high-logit tail (peaks tie at 1).
                self._sm_cacc += S.float().amax(-1).mean().item()
                self._sm_cn += 1
                if self._sm_cn >= self._sm_calib_steps:
                    self._sm_c = self._sm_cacc / self._sm_cn
                    self._sm_frozen = True
        O = torch.bmm(P, vf) / rs.unsqueeze(-1).half()                   # normalize (rowsum per O-row)
        return O.reshape(b, nh, T, hd)

    def _attn(self, q, k, v, T):
        """Attention on [b,nh,T,hd]: fp16 materialized (opt-in) else MATH SDPA (default).
        (The fused int8/int4 flash kernels exist in csrc but are intentionally NOT wired into
        this pipeline; attention runs fp16 MATH.)"""
        if self._materialized and (T % 8 == 0):
            return self._materialized_attn(q, k, v)
        with _SDPA_CTX():
            return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

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
        if getattr(self.qkv, "weight", None) is not None:
            w = self.qkv.weight.detach().to(torch.float16)      # [3C, C] fp16 Linear
        else:
            # qkv was quantized to QuantLinearWxAx (int8): rebuild the (int8-quantized) fp16 weight
            # from qweight * per-output-channel w_scale, slicing off the AWQ N/K zero-padding.
            # int8 only (int4 qweight is nibble-packed); the gate guards bits==8.
            N, K = 3 * C, C
            qw = self.qkv.qweight[:N, :K].to(torch.float32)
            ws = self.qkv.w_scale[:N].to(torch.float32).unsqueeze(1)
            w = (qw * ws).to(torch.float16)                     # [3C, C]
        b = self.qkv.bias.detach().to(torch.float16)        # [3C]
        gw = (self.norm.weight.detach().to(torch.float16) if self.norm.weight is not None
              else torch.ones(C, device=w.device, dtype=torch.float16))
        gb = (self.norm.bias.detach().to(torch.float16) if self.norm.bias is not None
              else torch.zeros(C, device=w.device, dtype=torch.float16))
        Wf = (w * gw[None, :]).contiguous()                 # [3C, C]
        self._fused_conv_w = Wf.view(3 * C, 1, 1, C).contiguous()
        self._fused_epi_bias = (b + w @ gb - _FUSE_SHIFT * Wf.sum(dim=1)).contiguous().to(torch.float16)
        self._fused_ready = True

    def _apply_proj(self, a, b, T, c):
        """proj Linear on the head-major attention output a=[b,nh,T,hd].
        Fused int8 path: fuse the transpose+reshape layout copy and proj's quantize pass
        into one kernel (quantize_attn_out_int8) -> int GEMM, bypassing proj's own quantize.
        Falls back to the standard transpose+reshape + proj(.) otherwise (and always during
        calibration, so proj.a_scale gets captured). Returns h=[b,T,c]."""
        proj = self.proj
        if (self._fuse_proj_quant and _HAS_FUSED_ATTN_OUT and _QuantLinearWxAx is not None
                and isinstance(proj, _QuantLinearWxAx) and proj.bits in (8, 4) and not proj.modiff
                and proj.a_scale is not None and proj._awqt_K == proj.in_features   # int4: excludes C192
                and (proj.bits == 8 or _HAS_ATTN_OUT_I4)):
            if proj.bits == 8:   # fused transpose+int8-quantize -> W8A8 GEMM
                xq = _mc.quantize_attn_out_int8(a, proj.a_scale)             # int8 [b*T, c]
                if proj._awqt_N != proj.out_features:                        # padded N -> write unpadded (no slice-copy)
                    out = _mc.gemm_w8a8_awq_nout(xq, proj.qweight, proj.w_scale, proj.a_scale, proj.out_features)
                else:
                    out = _mc.gemm_w8a8_awq(xq, proj.qweight, proj.w_scale, proj.a_scale)
            else:                # fused transpose+int4-quantize+pack -> W4A4 GEMM
                # gate above guarantees _awqt_K == in_features here, so k_pad is a no-op pad (== C)
                xq = _mc.quantize_attn_out_int4_pack(a, proj.a_scale, proj._awqt_K)   # int4 packed [b*T, K_pad/2]
                out = _mc.gemm_w4a4_awq(xq, proj.qweight, proj.w_scale, proj.a_scale, proj._awqt_K)
                if proj._awqt_N != proj.out_features:                        # int4 has no unpadded variant -> slice
                    out = out[:, :proj.out_features].contiguous()
            h = out.reshape(b, T, c)
            if proj.bias is not None:
                h = h + proj.bias
            return h
        a = a.transpose(1, 2).reshape(b, T, c)                            # standard layout copy
        return proj(a)

    def _qkv_from_gn(self, x, b, T, c, nh, hd):
        """GroupNorm(no SiLU) -> qkv on the channels_last input x=[N,C,H,W]. Returns qkv=[b,T,nh,3,hd].
        Fused int8 path: group_norm_silu_quantize_nhwc computes GN in fp32 and emits int8 [b*T,c]
        directly (token-major, a free channels_last view), skipping the qkv Linear's own quantize
        pass; then gemm_w8a8_awq. Numerically equal to GN(fp16) + quantize_act_int8 up to one fp16
        rounding on the GN output. Falls back to the standard GN + qkv Linear otherwise (and during
        calibration, so qkv.a_scale is captured)."""
        qkv = self.qkv
        if (self._fuse_qkv_quant and _QuantLinearWxAx is not None and isinstance(qkv, _QuantLinearWxAx)
                and qkv.bits in (8, 4) and not qkv.modiff and qkv.a_scale is not None
                and qkv._awqt_K == qkv.in_features   # int4: excludes C192 (needs K%128 pad) -> falls back
                and x.dtype == torch.float16 and (c % self.norm.num_groups == 0)
                and ((qkv.bits == 8 and _HAS_GN_QUANT) or (qkv.bits == 4 and _HAS_GN_PACK))):
            gw, gb = self._gn_params(x.dtype)
            if gw is not None and gb is not None:
                if self._qkv_inv_scale_t is None:
                    self._qkv_inv_scale_t = torch.tensor([1.0 / qkv.a_scale], device=x.device, dtype=torch.float32)
                empty = x.new_empty(0); ng, eps = self.norm.num_groups, self.norm.eps
                bias = qkv.bias if qkv.bias is not None else empty
                if qkv.bits == 8:   # GN -> int8 (one kernel) -> W8A8 GEMM, bias fused into the epilogue
                    xq_img = _mc.group_norm_silu_quantize_nhwc(x, gw, gb, ng, eps, False, self._qkv_inv_scale_t, empty, empty, empty)
                    xq = xq_img.permute(0, 2, 3, 1).reshape(b * T, c)     # int8 [b*T, c] token-major (free view)
                    out = _mc.gemm_w8a8_awq_bias_res(xq, qkv.qweight, qkv.w_scale, qkv.a_scale, qkv.out_features, bias, empty)
                else:               # GN -> int4-packed (one kernel, gemm_w4a4 layout) -> W4A4 GEMM, bias fused
                    xq_img = _mc.group_norm_silu_quantize_pack_nhwc(x, gw, gb, ng, eps, False, self._qkv_inv_scale_t, empty, empty, empty)
                    xq = xq_img.reshape(b * T, c // 2)                    # int4 packed [b*T, c/2] token-major (free view)
                    out = _mc.gemm_w4a4_awq_bias_res(xq, qkv.qweight, qkv.w_scale, qkv.a_scale, qkv._awqt_K, qkv.out_features, bias, empty)
                out = out.reshape(b, T, qkv.out_features)
                return out.view(b, T, nh, 3, hd)
        # fallback: standard native GroupNorm + int8 qkv Linear
        w, bnorm = self._gn_params(x.dtype)
        xn = _group_norm_silu(x, self.norm.num_groups, w, bnorm, self.norm.eps, apply_silu=False)
        xn_tok = xn.permute(0, 2, 3, 1).reshape(b, T, c)
        return qkv(xn_tok).view(b, T, nh, 3, hd)

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
            a = self._attn(q, k, v, T)
            out_tok = x_in_tok + self._apply_proj(a, b, T, c)
            return out_tok.reshape(b, H, W, c).permute(0, 3, 1, 2)

        # GroupNorm(no SiLU) -> qkv. Fused int8 path folds qkv's quantize into the GN kernel
        # (group_norm_silu_quantize_nhwc); channel order matches the Conv1d exactly:
        # (num_heads, {q,k,v}, head_dim). Split into [N,H,T,hd] via views.
        qkv = self._qkv_from_gn(x, b, T, c, nh, hd)       # [b,T,nh,3,hd]
        q, k, v = qkv.unbind(3)                            # each [N,T,H,hd] (views)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # [N,H,T,hd]
        a = self._attn(q, k, v, T)                         # fp16 SDPA / materialized

        # [N,H,T,hd] -> proj (fused transpose+quantize when int8) -> residual, back to channels_last.
        h = self._apply_proj(a, b, T, c)
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
