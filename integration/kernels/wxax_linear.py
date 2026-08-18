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

#: Table length for the static per-step delta scale. Imported rather than redefined so the two paths
#: cannot drift; the conv twin's table is indexed by the same step counter.
from integration.kernels.int4_optimized import MODIFF_MAX_STEPS            # noqa: E402

#: The projections' delta grid. SWEPT 2026-08-17 with FID on the L1 + static arm, 10k images each:
#:
#:     ratio        1.0      2.0      8.0     no table (per-call absmax)
#:     FID       52.584   54.555  222.515       54.300
#:
#: 1.0 is the answer, and 8.0 -- the CONV path's value, which this constant was seeded from -- is a 4x
#: regression. The two distributions are not the same object. The conv sweep chose 8 because the conv
#: delta is HEAVY-TAILED, so covering its range spends codes on a tail nothing lands in; under-sizing
#: recovers them. `OptimizedInt4Conv2d.DELTA_CLIP_RATIO`'s own docstring records the opposite case in its
#: test fixture -- |max|/|min| = 1.26 got WORSE with the ratio, 0.221 -> 0.340 -- and the projections'
#: delta behaves like that one. Under-sizing its grid 8x just clips.
#:
#: At 1.0 the table BEATS the per-call reduction it replaces (52.584 vs 54.300, -1.72 FID) while removing
#: it from the hot path (-8.81 ms/step, docs/w4a4_quality_2026-08-17 section 6.4). Both axes, one change.
#:
#: This is why the conv constant's docstring says "Do not copy the 8 across without measuring -- that
#: assumption is what the int4/int8 twins keep getting wrong". Seeding from it cost a 4x regression and a
#: ~30-minute window in which the table auto-loaded; see benchmark_ldm's load site.
LINEAR_DELTA_CLIP_RATIO = float(os.environ.get("MODIFF_LINEAR_DELTA_CLIP_RATIO", "1.0"))


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
        #: Recompute the dynamic delta scale every Kth modulated step, reuse it in between -- the
        #: mechanism OptimizedInt8Conv2d.delta_refresh has had all along, which never reached here.
        #:
        #: MEASURED PREMISE (docs/profile_kernels_layers_2026-08-11): the `delta_quantize` kernel
        #: bucket moves +4.7 ms/step going K=4 -> K=1 at conv-only AND +4.7 at conv+proj, i.e. the
        #: projections' contribution is K-INDEPENDENT, because `delta_absmax_fp16` below runs
        #: unconditionally. The projections' own scale recomputation is 1.84 ms/step (`proj`, via
        #: delta_absmax_fp16) plus ~1.9 (`qkv`, via the GN-fused kernel's absmax) = ~3.7 ms/step, and
        #: a K=4 schedule should remove about three quarters of it -- the same ratio the conv path
        #: gets (its gn_delta_absmax drops 8.24 -> 3.54 ms at K=4).
        #:
        #: DEFAULT 1, i.e. OFF, i.e. the pre-existing behaviour bit for bit. This changes numerics on
        #: reuse steps -- a scale up to K-1 steps old, which is exactly what the conv path's code
        #: ceiling was added for -- so it ships as a knob to be measured, not as a default. Note the
        #: paired-seed protocol at batch 8 / DDIM 50 cannot resolve effects below ~10%
        #: (docs/updown_refresh_fusion_2026-08-10), so a quality verdict here needs a bigger budget
        #: than the a4 measurement used.
        self.delta_refresh = max(1, int(os.environ.get("MODIFF_LINEAR_DELTA_REFRESH", "1")))
        self._step = 0
        # ---- STATIC per-step delta table (2026-08-17) ------------------------------------------------
        #: The conv path has one of these; the projections never did, so every modulated call here runs
        #: `delta_absmax_fp16` -- a global reduction over the delta. THREE things follow from that, and
        #: this table addresses all three:
        #:
        #:   1. COST. It is a separate pass per layer per step (+876 ms of the profiled window at batch
        #:      128, docs/OPEN_ITEMS.md on MODIFF_LINEAR).
        #:   2. IT BLOCKS FUSION, which is the important one. A global absmax cannot be fused with the
        #:      kernel that consumes it -- the whole tensor must be reduced before the first element can
        #:      be quantized. That is precisely why the conv path CAN fuse GroupNorm+SiLU+delta-quantize
        #:      into one kernel (`group_norm_silu_delta_quantize_pack_nhwc`) and this path cannot: the
        #:      conv path reads a static table. So the un-fused quantize that costs L1 +2890 ms is a
        #:      CONSEQUENCE of the missing table, not an independent problem.
        #:   3. NONDETERMINISM. docs/OPEN_ITEMS.md A18: L1 is run-to-run nondeterministic at 4.5-6.2/255
        #:      while every L0 arm is bit-exact. `delta_absmax_fp16`'s `_retire` argument is the
        #:      signature of a last-block-retires grid reduction, which is order-dependent. Prime
        #:      suspect, and a static table removes the call entirely.
        self.register_buffer('static_delta_scale', torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32))
        self.register_buffer('static_delta_alpha', torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32))
        self.register_buffer('is_delta_calibrated', torch.tensor(False))
        #: Host mirror -- the hot path must never read the device buffer (one GPU->CPU sync per
        #: modulated layer per step; the conv twin measured ~5 ms/step for exactly this mistake).
        self._delta_cal = False
        self._delta_calib = False
        self._delta_absmax_obs = None
        # Scratch for the fused MoDiff path (lazily sized on first modulated call).
        self._absmax = self._scale = self._inv_scale = self._retire = None
        self._empty_f32 = None
        self._empty_h = None
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

    def _delta_should_refresh(self) -> bool:
        """Whether this modulated step re-measures the delta scale, or reuses the last one.

        Same shape as OptimizedInt8Conv2d._delta_should_refresh: `_step` is 1 on the first modulated
        call, so `(step - 1) % K == 0` always refreshes there -- required, since `_scale` holds
        nothing valid before the first refresh.
        """
        k = self.delta_refresh
        return k <= 1 or ((self._step - 1) % k) == 0

    # ---- static per-step delta table -----------------------------------------------------------------
    def _load_delta_scale_for_step(self) -> None:
        """Point `_scale`/`_inv_scale` at the table entry for this step. No reduction, no sync.

        Writes into the SAME scratch buffers the dynamic path fills, so every consumer downstream
        (`step1_static_quantize*`, the o_hat GEMM's alpha) is unchanged -- that is what makes this a
        drop-in and what will let the fused kernel read the scale from one place.
        """
        i = min(max(self._step - 1, 0), MODIFF_MAX_STEPS - 1)
        self._scale.copy_(self.static_delta_scale[i:i + 1])
        self._inv_scale.copy_(self.static_delta_alpha[i:i + 1])

    def _observe_delta_absmax(self) -> None:
        """Record this step's delta absmax, recovered from what the dynamic reduction just wrote.

        `delta_absmax_fp16` writes `inv_scale = absmax / Q`, so `absmax = inv_scale * Q`. Reading it
        back is how the conv twin observes too (int4_optimized._observe_delta_absmax) -- the point is to
        observe the EXACT quantity the production path would have used, not a second estimate of it.
        """
        i = min(max(self._step - 1, 0), MODIFF_MAX_STEPS - 1)
        if self._delta_absmax_obs is None or self._delta_absmax_obs.device != self._inv_scale.device:
            self._delta_absmax_obs = torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32,
                                                 device=self._inv_scale.device)
        am = self._inv_scale.view(()) * float(self.Q)
        self._delta_absmax_obs[i] = torch.maximum(self._delta_absmax_obs[i], am)

    def begin_delta_calibration(self) -> None:
        """Observe every step, so a per-step table is not undersampled by a refresh schedule."""
        self._delta_calib = True
        self._delta_cal = False
        self.is_delta_calibrated.fill_(False)
        self._delta_calib_was_refresh = self.delta_refresh
        self.delta_refresh = 1
        self._delta_absmax_obs = None

    def end_delta_calibration(self, safety: float = 1.02, smooth: bool = True) -> bool:
        """Turn the observed per-step absmax into the scale table. Returns True if set.

        Identical arithmetic to OptimizedInt4Conv2d.end_delta_calibration -- forward-fill the tail,
        back-fill the head, 3-wide running max, then `scale = Q / (absmax * safety / ratio)`. Copied in
        shape rather than in spirit so a future change to one path is visibly a divergence from the
        other; the ONE deliberate difference is LINEAR_DELTA_CLIP_RATIO, which is not the conv constant.
        """
        self._delta_calib = False
        self.delta_refresh = getattr(self, "_delta_calib_was_refresh", self.delta_refresh)
        if self._delta_absmax_obs is None:
            return False
        absmax = self._delta_absmax_obs.detach().to("cpu", torch.float64)
        seen = absmax > 0
        if not bool(seen.any()):
            return False
        last = 0.0
        for i in range(absmax.numel()):
            if seen[i]:
                last = float(absmax[i])
            elif last > 0.0:
                absmax[i] = last
        first = next((float(absmax[i]) for i in range(absmax.numel()) if absmax[i] > 0), 0.0)
        if first <= 0.0:
            return False
        absmax[absmax <= 0] = first
        if smooth:
            a = absmax.clone()
            for i in range(absmax.numel()):
                lo, hi = max(0, i - 1), min(absmax.numel(), i + 2)
                absmax[i] = float(a[lo:hi].max())
        # self.Q is 127/7 -- the same ceiling the conv twin calls Q_DELTA. One source of truth.
        scale = float(self.Q) / (absmax * safety / LINEAR_DELTA_CLIP_RATIO).clamp_min(1e-12)
        self.static_delta_scale.copy_(scale.to(torch.float32))
        self.static_delta_alpha.copy_((1.0 / scale).to(torch.float32))
        self.is_delta_calibrated.fill_(True)
        self._delta_cal = True
        self._delta_absmax_obs = None
        return True

    def reset_modiff(self):
        self.a_hat = None
        self.o_hat = None
        # The schedule restarts with the caches. Leaving _step running across samples would make the
        # first modulated step of sample 2 a REUSE step, quantizing against a scale measured for a
        # different trajectory -- and _scale would be stale rather than merely old.
        self._step = 0

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

    def _ensure_modiff_bufs(self, xf):
        """1-element device scratch for the dynamic delta scale, plus the two empty sentinels.

        Allocated once per layer and reused, so the hot path performs no allocation. absmax and
        retire must be zero on entry; both kernels self-reset them.
        """
        d = xf.device
        if self._scale is None or self._scale.device != d:
            self._absmax = torch.zeros(1, device=d, dtype=torch.float32)
            self._scale = torch.empty(1, device=d, dtype=torch.float32)
            self._inv_scale = torch.empty(1, device=d, dtype=torch.float32)
            self._retire = torch.zeros(1, device=d, dtype=torch.int32)
            self._empty_f32 = torch.empty(0, device=d, dtype=torch.float32)
            self._empty_h = torch.empty(0, device=d, dtype=torch.float16)

    def forward(self, x, residual=None):
        orig = x.shape
        xf = x.reshape(-1, self.in_features).half().contiguous()
        if self._calib:
            self._amax = max(self._amax, float(xf.abs().max()))
        rflat = residual.reshape(-1, self.out_features).half().contiguous() if residual is not None else None

        # A shape change invalidates the temporal caches: they are indexed by tensor position, so an
        # a_hat built at one M (batch x tokens) means nothing at another. This happens in normal use
        # -- the activation-scale calibration samples at a smaller batch than production -- and it
        # surfaced as "delta_absmax_fp16: a_hat_cache must match x element count" on the int4 path.
        # Dropping the caches makes the next call re-seed, which is the correct semantics.
        if self.modiff and self.a_hat is not None and self.a_hat.shape[0] != xf.shape[0]:
            self.a_hat = None
            self.o_hat = None

        if self._use_bias_res:
            # FUSED bias(+residual) epilogue -- the default int8/int4 Linear path
            a_scale = self.a_scale if self.a_scale is not None else ((xf.abs().max().item() / self.Q) or 1e-8)
            out = self._gemm_bias_res(xf, a_scale, rflat)
            return out.reshape(*orig[:-1], self.out_features)

        if self.modiff and self.a_hat is not None:
            # MoDiff (paper Eqs 13-14) on the FUSED path: three kernels, no eager arithmetic.
            #   a_hat_t = Q(x_t - a_hat_{t+1}) + a_hat_{t+1}
            #   o_hat_t = Linear(Q(x_t - a_hat_{t+1})) + o_hat_{t+1}
            #
            # This used to be ~6 launches of eager PyTorch (subtract, round, clamp, a_hat add, gemm,
            # o_hat add) plus a host sync for the delta absmax, costing +10.9 ms/step at batch 8 --
            # the ONLY reason MoDiff on the Linear layers stayed off by default. The method itself has
            # been correct since Bug 2 was fixed (int4 latent relL2 0.4571 -> 0.4220 with it on).
            #
            # Every kernel below already existed except gemm_w{8a8,4a4}_awq_o_hat, and none of them is
            # a MoDiff-only clone:
            #   delta_absmax_fp16              the conv path's dynamic delta scale, on device, no sync
            #   step1_static_quantize[_pack]   the conv path's delta-quantize + in-place a_hat update.
            #                                  It is dimension-agnostic (it walks x.numel() and only
            #                                  reads size(1) for SmoothQuant, which is empty here), so
            #                                  the 2D [M,K] Linear activation needs no separate kernel.
            #   gemm_*_awq_o_hat               the accumulate folded into the GEMM epilogue.
            self._ensure_modiff_bufs(xf)
            # a_hat is seeded at the PADDED width (see the first-step branch), so pad x to match or
            # the delta subtract is a shape error. int4 pads K to _awqt_K for the AWQ layout; int8
            # usually does not, in which case this is a no-op.
            xq_in = xf
            if self._awqt_K != self.in_features:
                xq_in = F.pad(xf, (0, self._awqt_K - self.in_features)).contiguous()
            assert self.a_hat.shape[-1] == xq_in.shape[-1], (
                f"a_hat width {self.a_hat.shape[-1]} != padded x width {xq_in.shape[-1]}")
            # Dynamic per-call delta scale. Q_level is passed as the int8/int4 code ceiling, and the
            # kernel writes scale = Q/max|delta| plus its reciprocal, so the quantize below cannot
            # clip. Same choice as the conv path, and for the same measured reason.
            #
            # On a REUSE step this whole pass is skipped and `_scale`/`_inv_scale` keep the value the
            # last refresh wrote there -- they are persistent buffers, so no state has to be carried
            # by hand. `step_count`-equivalent is `_step`, incremented here rather than in forward(),
            # so the non-modulated seeding call does not advance the schedule.
            self._step += 1
            # STATIC TABLE takes precedence over the per-call reduction. `_delta_cal` is the HOST
            # mirror; reading is_delta_calibrated here would cost a GPU->CPU sync per layer per step.
            # During calibration (`_delta_calib`) the reduction still runs -- it is what is being
            # observed -- so the two branches are exclusive by construction, not by ordering luck.
            if self._delta_cal and not self._delta_calib:
                self._load_delta_scale_for_step()
            elif self._delta_should_refresh():
                _mc.delta_absmax_fp16(xq_in, self.a_hat, self._absmax, self._scale, self._inv_scale,
                                      self._retire, float(self.Q), self._empty_f32, False)
                if self._delta_calib:
                    self._observe_delta_absmax()
            if self.bits == 8:
                codes = _mc.step1_static_quantize_fprop(xq_in, self.a_hat, self._scale,
                                                        self._empty_f32)
                out = _mc.gemm_w8a8_awq_o_hat(codes, self.qweight, self.w_scale,
                                              self._inv_scale, self.out_features,
                                              self.o_hat, rflat if rflat is not None else self._empty_h,
                                              self.bias if self.bias is not None else self._empty_h)
            else:
                codes = _mc.step1_static_quantize_pack_int4_fprop(xq_in, self.a_hat, self._scale,
                                                                  self._empty_f32)
                out = _mc.gemm_w4a4_awq_o_hat(codes, self.qweight, self.w_scale,
                                              self._inv_scale, self.out_features,
                                              self.o_hat, rflat if rflat is not None else self._empty_h,
                                              self.bias if self.bias is not None else self._empty_h)
            # The GEMM advanced o_hat in place (bias- and residual-free, Eq 9) and returned
            # o_hat_t + bias + residual. So return directly: falling through to the shared
            # bias/residual tail below would apply both a SECOND time. Getting this wrong is what
            # took int8 latent relL2 from 0.039 to 0.300 in the first version of this path -- the
            # bias was dropped entirely instead of being moved to the output.
            return out.reshape(*orig[:-1], self.out_features)
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
                # Seed at the PADDED width: the modulated path quantizes the padded activation, and
                # a_hat must have the same element count or delta_absmax_fp16 rejects it.
                xs = (F.pad(xf, (0, self._awqt_K - self.in_features)).contiguous()
                      if self._awqt_K != self.in_features else xf)
                q = torch.round(xs / a_scale).clamp_(-self.Q, self.Q)
                self.a_hat = (q * a_scale).half().contiguous()
                # o_hat holds Linear(a_hat_T) only -- bias and residual are added at the output on
                # every step, including this one (the tail below does it), so they must NOT be
                # baked into the state here.
                self.o_hat = out.detach().half().contiguous()

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


def begin_wxax_delta_calibration(model) -> int:
    """Arm per-step delta observation on every modulated wxax Linear. Returns how many."""
    n = 0
    for m in model.modules():
        if isinstance(m, QuantLinearWxAx) and m.modiff:
            m.begin_delta_calibration()
            n += 1
    return n


def end_wxax_delta_calibration(model, safety: float = 1.02, smooth: bool = True) -> int:
    """Turn observations into tables. Returns how many layers got one."""
    return sum(int(m.end_delta_calibration(safety, smooth))
               for m in model.modules()
               if isinstance(m, QuantLinearWxAx) and m.modiff)


def export_wxax_delta_scales(model) -> dict:
    """{dotted module path -> per-step scale tensor}, for the calibrated modulated Linears.

    Keyed on `named_modules()` rather than on a name stored at construction: the converter builds a
    prefix for logging but does not keep it, and a stored name is one more thing that can go stale
    against the module tree. A SEPARATE artifact from the conv delta table -- same reason
    export_int4_delta_scales is separate from export_int4_static_scales: different quantity, different
    validity.
    """
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, QuantLinearWxAx) and m.modiff and bool(m.is_delta_calibrated):
            out[name] = m.static_delta_scale.detach().to("cpu", torch.float32).clone()
    if out:
        # THE RATIO GOES IN THE ARTIFACT. It is baked into the values, so a file exported at one ratio
        # and loaded under a different default is silently wrong -- which happened: exporting at 8.0 and
        # then changing the default to 1.0 (the swept value) left `apply`'s `r / LINEAR_DELTA_CLIP_RATIO`
        # dividing by the wrong number, so asking for 1.0 would have re-applied the 8.0 values and read
        # as FID 222 instead of 52.6. Recording it makes that a refusal instead of a wrong number.
        out["__clip_ratio__"] = torch.tensor(float(LINEAR_DELTA_CLIP_RATIO), dtype=torch.float64)
    return out


def apply_wxax_delta_scales(model, table: dict) -> int:
    """Load a table from export_wxax_delta_scales. Returns how many layers were filled.

    MODIFF_LINEAR_DELTA_TABLE_RATIO re-sizes the loaded grid by `r / LINEAR_DELTA_CLIP_RATIO`, the same
    knob the conv path got on 2026-08-17 and for the same reason: the ratio is baked in at export, so
    without this a re-sweep means a re-export. Default 0 = off, so an existing table loads unchanged.
    """
    if not table:
        return 0
    # The ratio the values were BAKED AT, from the artifact. Files written before this was recorded are
    # assumed to be 8.0 -- the seeded default in force when the only such file was produced -- rather
    # than assumed to match today's constant, which is the assumption that would be wrong.
    if "__clip_ratio__" in table:
        baked = float(table["__clip_ratio__"])
    else:
        # LOUD, because a dict without the key is indistinguishable from the one legacy file that
        # genuinely was baked at 8.0 -- including a hand-sliced SUBSET of a newer table, which would
        # then be rescaled 8x for no reason. Silence here is how a wrong number looks right.
        baked = 8.0
        print("  WARNING: linear delta table carries no __clip_ratio__; assuming it was baked at 8.0 "
              "(the pre-2026-08-17 default). If this is a slice of a newer table, carry the key.",
              flush=True)
    r = float(os.environ.get("MODIFF_LINEAR_DELTA_TABLE_RATIO", "0") or 0)
    want = r if r > 0 else LINEAR_DELTA_CLIP_RATIO
    mul = want / baked
    if mul != 1.0:
        print(f"  linear delta table baked at ratio {baked}, want {want}: rescaling by {mul:.4f}",
              flush=True)
    loaded = 0
    for name, m in model.named_modules():
        if not (isinstance(m, QuantLinearWxAx) and m.modiff) or name not in table:
            continue
        if name == "__clip_ratio__":       # metadata, not a layer
            continue
        t = table[name].to(m.static_delta_scale.device, torch.float32) * mul
        k = min(t.numel(), m.static_delta_scale.numel())
        m.static_delta_scale[:k].copy_(t[:k])
        if k < m.static_delta_scale.numel():
            m.static_delta_scale[k:].fill_(float(t[k - 1]))
        m.static_delta_alpha.copy_(1.0 / m.static_delta_scale.clamp_min(1e-12))
        m.is_delta_calibrated.fill_(True)
        m._delta_cal = True
        loaded += 1
    return loaded


def reset_wxax_modiff(model):
    """Clear the MoDiff temporal caches (call between DDIM samples).

    Deliberately does NOT clear the delta table: it is calibration, not state. Clearing it here would
    silently drop every layer back to the per-call reduction on sample 2 -- the exact class of bug the
    conv path's `_step` reset comment warns about.
    """
    for m in model.modules():
        if isinstance(m, QuantLinearWxAx):
            m.reset_modiff()
