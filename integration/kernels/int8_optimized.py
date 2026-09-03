"""
CUTLASS INT8 Conv2d with SmoothQuant + MoDiff Error-Compensated Modulation.

Uses true INT8×INT8 tensor core kernels via CUTLASS for maximum throughput.
Implements SmoothQuant to migrate per-channel activation variance into weights,
and MoDiff paper's error-compensated modulation across diffusion timesteps.

MoDiff equations (Gao et al., ICML 2025):
    t=T (first step):
        a_hat_T = Q(a_T)                                    -- Eq. (ec1)
        o_hat_T = A(a_hat_T) + bias                         -- Eq. (ec2)
    t<T (modulated steps):
        a_hat_t = Q(a_t - a_hat_{t+1}) + a_hat_{t+1}        -- Eq. (ec5)
        o_hat_t = A(Q(a_t - a_hat_{t+1})) + o_hat_{t+1}     -- Eq. (ec6)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from integration.utils.profiler import profiler

try:
    import modiff_cutlass
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: modiff_cutlass extension not found.")

#: EXPERIMENTAL, default OFF. Routes forward_gn_fused_modiff's step1 through the group-major
#: single-kernel GN+delta-quantize+a_hat (group_norm_silu_delta_quantize_nhwc_fused) instead of the
#: shipped channel-major two-kernel path, so it can be measured on real activations in the real
#: dispatch chain. See that kernel's header comment in csrc/modiff/norm/group_norm_silu.cu: isolated
#: synthetic-tensor measurement already found it a regression at this UNet's CPG (0.44x-0.98x); this
#: flag exists to confirm or correct that finding end-to-end rather than trust the microbenchmark.
_GN_GROUPMAJOR = os.environ.get("MODIFF_GN_GROUPMAJOR", "0") == "1"

# Length of the MoDiff per-step delta-scale table. Indexed by the modulated-step ordinal, so it
# must cover the largest DDIM step count anyone runs (the benchmarks use 200). Runs longer than
# this clamp to the last entry, which is safe: the delta range is flat in the tail.
MODIFF_MAX_STEPS = 256

# Per-shape CUTLASS tile autotuning for the deep-fuse int8 conv. On first call
# each conv times all tile configs on its actual input and caches the fastest
# (the cuDNN-style per-shape selection). Kill-switch: MODIFF_DISABLE_CONV_AUTOTUNE=1
# reverts to the single fixed 128^3 tile.
_CONV_AUTOTUNE = (os.environ.get("MODIFF_DISABLE_CONV_AUTOTUNE") != "1"
                  and HAS_CUTLASS and hasattr(modiff_cutlass, "conv2d_int8_dequant_fp16_tuned"))


class OptimizedInt8Conv2d(nn.Module):
    """
    CUTLASS-based INT8 Conv2d.
    Replaces previous implementations for better generality.
    """
    def __init__(self, conv: nn.Conv2d, layer_name: str = "", use_compile: bool = False):
        super().__init__()
        self.layer_name = layer_name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        K = self.out_channels

        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
        self.groups = conv.groups

        w_data = conv.weight.data  # [K, C_in, R, S]

        # --- SmoothQuant ---
        self.register_buffer('smooth_scale', torch.ones(1, self.in_channels, 1, 1))
        self.register_buffer('_smooth_inv', torch.ones(1, self.in_channels, 1, 1))
        # No .clone(): w_data is conv.weight.data (already detached from autograd),
        # and only ever read (never mutated) by _apply_smoothquant(). Aliasing it
        # avoids a full-weight D2D copy at construction time that is pure waste
        # whenever calibration is loaded from a file instead of run live (the
        # common case — see apply_static_scales, which never calls
        # begin_calibration()/end_calibration() so this buffer would otherwise
        # sit around unused for the life of the model).
        self.register_buffer('_orig_weight', w_data, persistent=False)

        # --- Per-output-channel symmetric INT8 weight quantization ---
        w_flat = w_data.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values  # [K]
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)  # [K]
        self.register_buffer('weight_scale_channel', ch_scale.view(1, K, 1, 1))
        self.register_buffer('weight_scale_channel_half', ch_scale.half().contiguous())

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        w_quant = w_quant.reshape_as(w_data)
        # CUTLASS expects NHWC (K, R, S, C) for weights
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        self.register_buffer('weight_int8', w_nhwc)

        # --- Bias ---
        if conv.bias is not None:
            self.register_buffer('bias', conv.bias.data.view(1, -1, 1, 1))
        else:
            self.bias = None

        self._empty_bias = None
        self.use_cutlass = HAS_CUTLASS and self.groups == 1

        # --- MoDiff state ---
        self.modiff_enabled = False
        self.is_first_step = True
        self.a_hat_cache: Optional[torch.Tensor] = None
        self.o_hat_cache: Optional[torch.Tensor] = None
        self.step_count = 0
        #: t=T warm-up rounds. The paper's warm-up is "repeatedly inputting a_T" until the
        #: quantization error contracts away (Appendix D.5), which needs 4-5 rounds at 4 bits. The
        #: previous value of 3 came with the comment "3 steps sufficient for convergence" -- true at
        #: int8, where one round already lands within 2% -- but the loop was ALSO a no-op (see
        #: _forward_first_step), so the count never mattered. Measured contraction of |a_hat - x| /
        #: |x| on real activations, per round (docs/act_bits_2026-08-05/scripts/probe_warmup.py):
        #:     A8   0.0197 -> 0.00008 -> 0.00000
        #:     A4   0.4006 -> 0.0263  -> 0.0018 -> 0.00013 -> 0.00001
        #: A8 converges in one round (paper Appendix D.5; measured |a_hat-x|/|x|
        #: 0.0197 -> 0.00008). Extra rounds at t=T cost ~11 ms/step amortized and
        #: do not move quality (relL2 0.109 vs 0.108). INT4 keeps default 5.
        self.warmup_steps = max(1, int(os.environ.get("MODIFF_WARMUP_STEPS", "1")))

        # --- Calibration state ---
        self.calibrating = False
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self.register_buffer('static_input_scale', torch.tensor(1.0, dtype=torch.float32))
        # --- MoDiff per-step DELTA scale table (paper Theorem 4.3) ---
        # The whole benefit of MoDiff is that the quantizer step s comes from the range of the
        # temporal delta a_t - a_hat_{t+1}, not from the range of a_t: the error bound is
        # ||x - Q(x)||^2 <= s^2 d, so quantizing the delta on the activation's grid leaves the
        # error *unchanged* and buys nothing but error feedback. Until 2026-08-03 that is exactly
        # what happened -- static_input_scale (calibrated on the full activation) was passed
        # straight into the delta-quantize kernels.
        #
        # Why a table and not one scalar: the delta at the second step is a_{T-1} - Q(a_T), which
        # carries the whole t=T quantization error (range ~ full_range/(2(2^b-1))) on top of the
        # true temporal delta, so it is several times a steady-state delta. One scalar either clips
        # catastrophically at step 2 or wastes half the codes for the remaining ~199 steps.
        # 2 x 256 x 4B = 2 KB per layer, and indexing is a tensor slice -- no host sync.
        self.register_buffer('static_delta_scale',
                             torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32))
        self.register_buffer('static_delta_alpha',      # 1/scale, the CUTLASS epilogue alpha
                             torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32))
        self.register_buffer('is_delta_calibrated', torch.tensor(False))
        #: MODIFF_CONV_BLOCKK scratch (see _forward_conv_blockk); None until first use.
        self._blockk_w = None
        self._blockk_ws = None
        self._blockk_bias = None
        self._blockk_empty = None
        #: Host-side mirror of is_delta_calibrated. The hot path must not read the device buffer --
        #: see _delta_scale_args for the measured cost of that mistake.
        self._delta_cal = False
        #: Delta-quantizer mode. Two ways to pick the step size for Q(a_t - a_hat_{t+1}):
        #:   static  -- a per-step scale from `static_delta_scale`, calibrated offline. Cheapest
        #:              (no extra pass) but it can clip, and clipping is unrecoverable error that
        #:              MoDiff's feedback term then propagates. Measured 2026-08-04 on the real
        #:              LSUN-churches checkpoint: 49 of 70 conv layers clip.
        #:   dynamic -- Q/max|delta| computed per call on device. Cannot clip by construction, and
        #:              needs no calibration file. Costs one extra read pass over x and a_hat.
        #: The paper's Theorem 4.3 bound assumes the dynamic form ("to avoid clipping error"), so
        #: static is the engineering shortcut and dynamic is the faithful implementation.
        #:
        #: STATIC is the default since 2026-08-12, and it is NOT the better setting. It is the
        #: paper's: README:96 reproduces MoDiff with --modulate --quant_mode qdiff --cali_min_max,
        #: i.e. a calibrated static per-step delta table. Fidelity to that is the reason; the
        #: numbers below are the price, and they are large. MODIFF_DELTA_MODE=dynamic restores the
        #: better-measuring path in one env var and changes nothing else.
        #:
        #: Dynamic won decisively when it was the default. Real checkpoint, S=50, batch 8,
        #: latent relL2 vs fp16, measured at steady state (2026-08-04):
        #:     W8A8  baseline 0.2378 | MoDiff static 0.1878 | MoDiff dynamic 0.0393  (6.05x
        #:           better than baseline, 4.78x better than static)
        #:     W4A4  baseline 0.7837 | MoDiff static 0.7770 | MoDiff dynamic 0.4199  (1.87x
        #:           better than baseline; with a static scale MoDiff bought almost nothing)
        #: Those static numbers predate the delta table being LOADABLE at all: until 2026-08-12
        #: apply_int8_delta_scales had zero call sites, so "static" meant an uncalibrated grid.
        #: benchmark_ldm.py:_load_delta_table now loads it, and the honest static-vs-dynamic
        #: comparison is the one in docs/static_qdiff_2026-08-12/FINDINGS.md.
        #:
        #: "At steady state" is load-bearing. The quantized attention blocks self-calibrate over
        #: their first MODIFF_ATTN_CALIB_STEPS forwards, so the first sampling run after model
        #: construction is several x worse than the second (int8 dynamic: 0.2107 then 0.0393).
        #: A first-run measurement reverses this ranking and is how an earlier version of this
        #: comment came to claim, wrongly, that static beat dynamic at W8A8.
        self.delta_dynamic = os.environ.get("MODIFF_DELTA_MODE", "static").lower() != "static"
        #: `MODIFF_DELTA_CLIP` (deliberate-clipping ratio: scale = Q/(ratio*max|delta|), so
        #: ratio < 1 traded a finer grid for clipping the top of the observed range) is RETIRED.
        #: Q_level is now Q_b, full stop. Refused rather than ignored, because an archived script
        #: that sets it was measuring something this code no longer does, and silently returning
        #: absmax numbers under a clip label is the kind of defect this tree has paid for twice.
        #:
        #: What it measured, kept here so the retirement is not a loss of the result. Steady state,
        #: real checkpoint, latent relL2 vs fp16 (2026-08-04):
        #:     ratio    1.00    0.70    0.50    0.35    0.25    0.15    0.10
        #:     W8A8   0.0393  0.0490  0.0556  0.0616  0.0924  0.1504  0.1973
        #:     W4A4   0.4571  0.4275  0.4501  0.4199  0.4459  0.4829  0.5307
        #: W8A8 is cleanly monotone -- any deliberate clipping hurts, so the default 1.0 was already
        #: optimal there and nothing is lost. W4A4 is flat within noise from 0.35 to 1.0. The one
        #: place it genuinely bought something was W8A4, where docs/act_bits_2026-08-05 measured
        #: r=0.40 at 0.086 against r=1.0's 0.183; that option goes away with this knob.
        #: Full sweeps and figures: docs/delta_clip_2026-08-06/, docs/act_bits_2026-08-05/.
        _clip = os.environ.get("MODIFF_DELTA_CLIP")
        if _clip is not None and float(_clip) != 1.0:
            raise ValueError(
                f"MODIFF_DELTA_CLIP={_clip}: the clip ratio was retired. Q_level is Q_b now, so "
                f"this would be silently ignored rather than applied. The sweeps it produced are "
                f"in docs/delta_clip_2026-08-06/ and docs/act_bits_2026-08-05/.")
        #: Activation precision for this W8 datapath: 8 (W8A8, the shipped mode) or 4 (W8A4, the
        #: paper's own configuration). Those are the only two this class supports; W4A4 is Int4Conv.
        #:
        #: This used to be `MODIFF_ACT_Q`, a free symmetric ceiling that also accepted 63/31/15/3/1
        #: (A7/A6/A5/A3/A2) for the 2026-08-05 sweep. Those intermediate widths are gone: they were a
        #: research instrument for one report, they are not configurations anything ships, and every
        #: one of them was a distinct value that a quantize call site could pass wrongly. `act_bits`
        #: is the bit-width; `act_q` below is derived from it, in one place.
        #: The sweep's data and conclusions are unaffected and still in docs/act_bits_2026-08-05/.
        #:
        #: This is a QUALITY instrument, not a speed one. Activations keep their int8 container and
        #: the GEMM stays W8A8, so a low setting costs nothing and saves nothing -- a real A4
        #: datapath needs int4 tensor cores, which require BOTH operands at 4 bits. What it does
        #: buy is the paper's actual configuration: MoDiff's claim is W8A4/W8A3, and until this knob
        #: existed the only way to measure it was to abuse the (now retired) MODIFF_DELTA_CLIP
        #: as Q_level = 127/ratio, which moved the delta grid and left the t=T grid at A8.
        #:
        #: Applied in two places, which together cover every activation the conv path quantizes:
        #:   * the dynamic delta quantizer's Q_level (Q/ratio), so Q(a_t - a_hat) gets b bits;
        #:   * set_static_scale, which rescales the calibrated per-tensor scale by Q/127, so the
        #:     baseline (MoDiff off) path and MoDiff's t=T warm-up both get b bits too.
        #: NOT applied to the attention blocks (quantized_std_attention hardcodes lvl=127/7) or to
        #: the Linear layers, so both arms of a sweep keep A8 there. The comparison stays symmetric;
        #: it is just not a whole-network A_b.
        #:
        #: One asymmetry to keep in mind when reading a sweep. The delta quantizer is dynamic
        #: (Q/max|delta| per call) so it cannot clip and is an exact b-bit quantizer. The static
        #: baseline scale is Q/calibrated_range, and the quantize kernels clamp codes at +-127, not
        #: +-Q -- so a baseline activation ABOVE its calibrated range keeps resolution where a true
        #: b-bit quantizer would saturate. That makes the baseline arm slightly optimistic, i.e. it
        #: understates rather than overstates MoDiff's advantage.
        #:
        #: Controls: A8 reproduces the shipped numbers exactly (relL2 0.039 MoDiff / 0.238
        #: baseline at batch 8, DDIM 50, real ckpt, steady state), which is how the plumbing is
        #: checked -- see docs/act_bits_2026-08-05/.
        _ab = int(os.environ.get("MODIFF_ACT_BITS", "8"))
        if _ab not in (4, 8):
            raise ValueError(
                f"MODIFF_ACT_BITS={_ab}: this datapath supports 8 (W8A8) or 4 (W8A4). The "
                f"intermediate widths that MODIFF_ACT_Q used to accept were retired -- see the "
                f"comment above and docs/act_bits_2026-08-05/ for their measurements.")
        self.act_bits = _ab
        #: Q_b = 2^(b-1)-1, derived from act_bits and never set independently, so the scale the
        #: reduction builds and the limit the quantize kernel saturates at cannot disagree.
        self.act_q = 127.0 if _ab == 8 else 7.0
        #: Recompute the dynamic delta scale every Nth modulated step, reusing it in between.
        #: 1 = exact (recompute every step). See _delta_should_refresh for the mechanism.
        #:
        #: Default 4, from the measured staleness tolerance (real ckpt, DDIM 50, latent relL2 as a
        #: ratio to K=1 at the same clip):
        #:     K            1      2      4      8     25
        #:     W8A8      1.00x  1.04x  1.09x  0.97x  3.15x
        #:     W4A4      1.00x  1.02x  1.01x  1.06x  1.15x
        #: Staleness is free out to K=8 (K=8 measured marginally BETTER than K=1 at int8, i.e.
        #: inside the metric's noise) and only collapses at K=25, where 50 steps get just two
        #: refreshes. 4 is the conservative pick inside the free region; it cuts the reduction pass
        #: to a quarter of its cost. Longer runs are safer at a given K, not riskier -- more steps
        #: means both more refreshes and less change per step.
        #:
        #: This is why the delta-quantize kernels were NOT modified to report the absmax they
        #: already compute (which would give a one-step-stale scale for free): K=4 captures most of
        #: the same win with no kernel changes at all.
        #: DEFAULT 4, and this survived a challenge. K=1 recomputes every step, which is what the
        #: paper's dynamic quantizer specifies, so 2026-08-06 this was changed to 1 on fidelity
        #: grounds -- and the measurement said no. Paired over 3 seeds, MoDiff latent relL2, warm-up
        #: fix in place, K=4 vs K=1:
        #:     A8  0.0595 / 0.0613 (K=1 loses 3/3 seeds)   A5  0.0768 / 0.0688 (wash)
        #:     A6  0.0590 / 0.0630 (2/3)                   A4  0.1553 / 0.1529 (wash)
        #:     A3  0.3595 / 0.4302 (3/3, by 0.055-0.087)   A2  0.6058 / 0.7063 (3/3, by 0.079-0.121)
        #: K=1 never wins and loses badly at 3 and 2 bits, on top of costing ~8% of ms/step. The
        #: mechanism is that a per-step absmax sets the grid from THIS step's single worst outlier,
        #: while holding it for K steps smooths that estimate -- and with 3-7 levels, one outlier
        #: eats most of the grid. Fidelity to the paper's formulation lost to the measurement.
        self.delta_refresh = max(1, int(os.environ.get("MODIFF_DELTA_REFRESH", "4")))
        #: Free absmax reporting: let the delta-quantize kernel record the range it already
        #: computes and publish the NEXT step's scale in its own retirement election, instead of
        #: running a separate reduction pass (gn_report_delta_absmax in group_norm_silu.cu).
        #:
        #: DEFAULT OFF -- it is a QUALITY REGRESSION, measured 2026-08-04, latent relL2:
        #:                      report=0    report=1
        #:     W8A8              0.0389      0.0507      (30% worse)
        #:     W4A4              0.4746     11.6553      (diverges)
        #: It saves only 0.8-1.1 ms/step, so the trade is bad at int8 and unusable at int4.
        #:
        #: WHY, and it was a flaw in my reasoning rather than in the kernel: I assumed the scale
        #: would be "one step stale". It is not. Reporting happens on a REFRESH step and the value
        #: is consumed across the FOLLOWING window, so by the end of that window the scale is up to
        #: 2*delta_refresh steps old -- while the separate pass measures the current step's range and
        #: uses it immediately. The staleness sweep that blessed K=4 assumed the latter. At W4A4's 15
        #: levels that extra lag clips, and clipping compounds through the error-feedback term.
        #: Making this viable would need the publication and the consumption in the same window
        #: (e.g. report every step, which measured SLOWER than the separate pass -- 98k blocks
        #: contending on one atomicCAS), so it is kept only as the recorded result.
        #: DEFAULT OFF, on measured grounds. Reporting is correct and is a clear win when a fresh
        #: scale is wanted every step (K=1: 83.55 -> 79.98 ms/step, beating a separate pass 2:1).
        #: But at the shipped K=4 it loses: gating the report to refresh steps means those steps
        #: quantize with a 4-step-old scale instead of a freshly measured one, and the measured
        #: price is relL2 0.0395 -> 0.0511 (+29%) for 77.06 -> 76.23 ms/step (-1.1%). Accuracy is
        #: what MoDiff exists to buy, so that trade is refused by default.
        #: Kept because it is verified, documented, and the right choice at K=1.
        self.delta_report = os.environ.get("MODIFF_DELTA_REPORT", "0") == "1"
        #: Headroom on the reported range, for it growing between steps.
        self.delta_report_safety = float(os.environ.get("MODIFF_DELTA_SAFETY", "1.15"))
        self._delta_seeded = False
        #: DOUBLE BUFFERING for the published scale, and it is load-bearing.
        #: On a reporting step the quantize kernel must quantize with the CURRENT scale while
        #: publishing the NEXT one -- and the conv that follows reads the current alpha to dequantize
        #: its accumulator. Publishing into the same buffer overwrites that alpha before the conv
        #: runs, so o_hat accumulates on a scale that was never used to quantize. Measured cost of
        #: getting this wrong: latent relL2 went 0.0389 -> 10.3230, i.e. total divergence, while
        #: every kernel-level unit test still passed (they exercise one launch, and the bug is a
        #: cross-kernel ordering hazard within a step).
        #: So the kernel publishes into pair B while pair A is in use, then Python flips which pair
        #: is current -- a reference swap, no copy, no launch.
        self._scale_pair_b = False
        self._scale_buf_b: Optional[torch.Tensor] = None
        self._inv_scale_buf_b: Optional[torch.Tensor] = None
        # Delta-range observation (calibration only). The trick that makes this rebuild-free:
        # starting from static_input_scale the delta CANNOT clip, because |delta| <= activation
        # range by construction. So max|q| over the emitted codes recovers the delta's range as
        # max|q| / scale_used, with no kernel change and no extra pass over the activation.
        self._delta_calib = False
        self._delta_code_max: Optional[torch.Tensor] = None
        self._act_channel_max: Optional[torch.Tensor] = None
        self._cached_scale_float: Optional[float] = None
        self._cached_alpha_tensor: Optional[torch.Tensor] = None
        self._cached_scale_tensor: Optional[torch.Tensor] = None  # for _forward_standard fused path
        self.standard_output_fp16 = False
        self._standard_output_buf: Optional[torch.Tensor] = None
        # INT8 conv->conv chaining (TensorRT-style): when output_requant_scale is
        # set (to the NEXT conv's static_input_scale), forward_to_int8 emits int8
        # requantized by it (+ optional fused ReLU), so the next conv reads int8
        # directly with no fp16 round-trip. See integration/fused_ops/chained_bottleneck.py.
        self.output_requant_scale: Optional[torch.Tensor] = None
        self.fuse_output_relu: bool = False
        self._int8_output_buf: Optional[torch.Tensor] = None
        # Autotuned CUTLASS tile config id for the deep-fuse int8 conv (lazy, per
        # this conv's shape). None = not yet tuned; -1 = fixed default tile.
        self._tuned_config_id: Optional[int] = None
        # Persistent scratch for the cast-free fp16 quantize in _forward_standard
        # (see there): a zeroed a_hat lets the fused step1 kernel consume fp16
        # activations directly, avoiding a per-layer fp16->fp32 cast.
        self._zero_ahat_buf: Optional[torch.Tensor] = None
        self._empty_smooth: Optional[torch.Tensor] = None

        # --- SmoothQuant identity flag for fast path ---
        self._smooth_is_identity = True

        # --- SiLU fusion: set by fused_resblock.py's wire_silu_fusion() when
        # this layer directly follows a ResBlock's GroupNorm (i.e. it's a
        # ResBlock in_conv/out_conv). When True, callers pass the *pre-SiLU*
        # activation and this layer applies SiLU itself -- either fused into
        # the quantize kernel (fast path) or via a plain F.silu(x) call
        # (first-step/uncalibrated fallback) -- see forward().
        self.fuse_input_silu = False

        # --- Fused kernel persistent buffers (lazy-initialized) ---
        self._residual_buf: Optional[torch.Tensor] = None
        self._scale_buf: Optional[torch.Tensor] = None
        self._inv_scale_buf: Optional[torch.Tensor] = None
        self._absmax_buf: Optional[torch.Tensor] = None
        self._retire_count: Optional[torch.Tensor] = None

        # --- Dynamic (uncalibrated) baseline buffers: no cache, so these are
        # smaller/separate from the MoDiff buffers above (no _residual_buf,
        # no a_hat/o_hat needed) ---
        self._dyn_scale_buf: Optional[torch.Tensor] = None
        self._dyn_inv_scale_buf: Optional[torch.Tensor] = None
        self._dyn_absmax_buf: Optional[torch.Tensor] = None
        self._dyn_retire_count: Optional[torch.Tensor] = None
        self._ahat_qscale: Optional[torch.Tensor] = None

    @staticmethod
    def _ahat_bits() -> int:
        try:
            return int(os.environ.get("MODIFF_AHAT_BITS", "16"))
        except (TypeError, ValueError):
            return 16

    @staticmethod
    def _imode() -> bool:
        """I-MoDiff: integer a_hat math, frozen s*, no dequant. MODIFF_IMODE=1.

        Orthogonal to held-int8 (IMODE=0, AHAT_BITS=8): that path still dequants.
        """
        return os.environ.get("MODIFF_IMODE", "0") == "1"

    def _ahat_want_int8(self) -> bool:
        """True when a_hat is stored as int8 codes + dequant scale, not I-MoDiff.

        Per-tensor: MODIFF_AHAT_BITS=8 or 4 (qmax 127 / 7). Along-C block:
        MODIFF_AHAT_BLOCK>0 (qmax 127, 4D scales [N,H,W,C/B]). Both skip
        Python fake-quant; the kernel dequants on load and snaps on store.
        """
        if self._imode():
            return False
        if not self.is_calibrated:
            return False
        if self._ahat_block() > 0:
            return True
        return 0 < self._ahat_bits() < 16

    def _ahat_qmax(self) -> float:
        bits = self._ahat_bits()
        if self._imode():
            if bits >= 16:
                return 32767.0
            return 127.0 if bits >= 8 else 7.0
        return 127.0 if bits >= 8 else 7.0

    def _imode_dtype(self):
        return torch.int16 if self._ahat_bits() >= 16 else torch.int8

    @staticmethod
    def _ahat_refresh() -> bool:
        return os.environ.get("MODIFF_AHAT_REFRESH", "0") == "1"

    def _ahat_block(self) -> int:
        """Along-C group size for int8 a_hat. 32 consecutive channels at each (n,h,w) share a
        scale; storage is int8 NHWC + fp32 scales [N,H,W,C/B], refreshed in-kernel on every write
        from the new per-block amax. 0 is off.

        DEFAULT 32, and 32 specifically -- it is the only block size where this pays. Measured
        end to end (batch 128, 50 DDIM, seed 1234, against the same arm with fp16 a_hat):

            B    W8A8 ms   vs fp16 a_hat   peak delta   cache    eta_cum
            16    87.47       0.936x         +187 MB    877 MB    0.0432
            32    79.82       1.026x         -608 MB    789 MB    0.0531
            64    90.20       0.908x          +24 MB    745 MB    0.0625

        B=16 and B=64 have a SMALLER cache and yet a peak equal to or worse than fp16 a_hat,
        because the compile-time fast path exists only at 32: B=16 misses ahat_is_b32 and takes
        the generic c/B divide, and at B=64 ahat_block_shuffle_ok fails so the host disables the
        in-kernel write and runs a separate ahat_commit_block pass, which keeps the delta codes
        live for an extra allocation plus a launch. Accuracy does not discriminate -- all three
        sit 5-7x inside the sample-anchored eta_cum threshold of 0.30 and all three decode to
        within the run-to-run image-MSE floor. See docs/ahat_conv_report_2026-09-02.

        Instance method, not static, for the two guards below: as a DEFAULT this has to degrade
        rather than raise.
        """
        try:
            b = int(os.environ.get("MODIFF_AHAT_BLOCK", "32"))
        except (TypeError, ValueError):
            return 0
        if b <= 0:
            return 0
        # Mutually exclusive with the blockwise CONV. Both of its routes keep a_hat in fp16
        # (_blockk_dequant returns fp16, `d = x - a_hat` is an fp16 subtract) and both decide
        # is_first_step from `a_hat_cache.dtype != torch.float16`, so an int8 a_hat would force
        # the first-step branch on EVERY step and silently disable the temporal accumulation.
        # Measured symptom: with MODIFF_CONV_BLOCKK=64 the AHAT_BLOCK=32 arm had byte-identical
        # peak allocation to AHAT_BLOCK=0 (7734 MB both) -- the setting did nothing. Return 0 so
        # that is explicit instead of silent.
        if self._conv_blockk() != 0:
            return 0
        # Degrade for a channel count 32 does not divide, instead of raising from
        # _pack_ahat_along_c. As an opt-in that only affected whoever set the env; as the default
        # it would break any model with a channel count off the 32-grid.
        a = getattr(self, "a_hat_cache", None)
        c = a.shape[1] if a is not None and a.dim() == 4 else getattr(self, "in_channels", None)
        return b if (c is not None and c % b == 0) else 0

    @staticmethod
    def _ahat_block_fake() -> bool:
        """Also along-C fake-quant the fp16 a_hat of UNcalibrated layers.

        Off by default: those layers have no int8 datapath, so the snap is pure
        eager-kernel cost. See _maybe_quantize_ahat.
        """
        return os.environ.get("MODIFF_AHAT_BLOCK_FAKE", "0") == "1"

    @staticmethod
    def _snap_ahat_along_c_(a: torch.Tensor, block: int, qmax: float = 127.0) -> None:
        """In-place int8 fake-quant of `a` (NCHW): groups of `block` channels / pixel."""
        x = a.permute(0, 2, 3, 1).contiguous().float()
        n, h, w, c = x.shape
        bsz = min(int(block), c)
        if bsz <= 0:
            return
        pad = (bsz - c % bsz) % bsz
        xp = F.pad(x, (0, pad)) if pad else x
        g = xp.shape[-1] // bsz
        blk = xp.reshape(n, h, w, g, bsz)
        amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        s = amax / qmax
        q = (blk / s).round().clamp_(-qmax, qmax)
        recon = (q * s).reshape(n, h, w, -1)[..., :c]
        a.copy_(recon.permute(0, 3, 1, 2).to(a.dtype))

    # ---- blockwise activation-quantizer simulation (MODIFF_ACT_BLOCK) -------------
    #
    # The conv INPUT quantizer -- Q(a_t) in the baseline arm, Q(a_t - a_hat_{t+1}) in
    # MoDiff -- is per-tensor today because its scale is the CUTLASS epilogue's scalar
    # `alpha` (conv2d_evt.cu: E_MulA = Sm80EVT<Mul, Accum, Alpha>). A blockwise scale
    # along C is a scale along the conv's REDUCTION axis, and by the time the epilogue
    # sees the int32 accumulator the per-block structure has already been summed away.
    # Expressing it for real needs a mainloop that promotes to fp32 every 32 K, which
    # this CUTLASS 2.x sm80 int8 conv has no support for.
    #
    # So this path exists to price the accuracy first: it fake-quantizes (quantize ->
    # dequantize) the same tensor the real kernels quantize and runs the conv in fp32
    # on dequantized int8 weights. Weights stay W8 per-output-channel, both arms share
    # this code, and the only variable is the activation quantizer's granularity.
    # It is a measurement harness, NOT a fast path -- it bypasses every fused kernel.
    #
    #   MODIFF_ACT_BLOCK = 0   off (default): the real fused int8 kernels
    #                     -3   sim, activations EXACT (no activation quantizer at all)
    #                          -- the arm that isolates the OTHER error sources
    #                     -2   sim, per-tensor STATIC (calibrated scale / delta table)
    #                          -- the control that should reproduce the shipped relL2
    #                     -1   sim, per-tensor DYNAMIC absmax
    #                      N   sim, DYNAMIC blockwise, N consecutive channels per pixel
    #
    # Two companion knobs, both for the error budget in docs/act_budget_2026-09-02:
    #   MODIFF_ACT_SIM_EXACT_W=1  exact instead of W8 conv weights (must be set pre-build)
    #   MODIFF_ACT_SIM_QMAX=7     coarser activation grid, as a needle control
    #
    # -1 vs N is the honest granularity comparison (both dynamic); -2 validates the
    # harness against the numbers the real kernels produce.
    @staticmethod
    def _act_block() -> int:
        try:
            return int(os.environ.get("MODIFF_ACT_BLOCK", "0"))
        except (TypeError, ValueError):
            return 0

    def _sim_guard(self, where: str) -> None:
        """Fused entry points must not run in sim mode -- they would quantize with the
        real per-tensor kernels and silently report the wrong granularity.

        MODIFF_CONV_BLOCKK has the identical failure mode: the fused GN->quantize kernels
        emit per-tensor codes and call the CUTLASS conv directly, so a fused entry point
        firing under a blockwise label would report the SHIPPED path's number. Same hard
        error rather than a silently wrong measurement."""
        if self._conv_blockk() != 0:
            raise RuntimeError(
                f"MODIFF_CONV_BLOCKK reached fused entry point {where} on layer "
                f"{getattr(self, 'layer_name', '?')}. All five fusion kill switches must be "
                f"set so every conv goes through forward(): "
                f"MODIFF_DISABLE_GN_MODIFF_FUSION=1 MODIFF_DISABLE_GN_INT8_FUSION=1 "
                f"MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION=1 "
                f"MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION=1 "
                f"MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION=1 (read at fused_resblock import).")
        if self._act_block() != 0:
            raise RuntimeError(
                f"MODIFF_ACT_BLOCK sim reached fused entry point {where} on layer "
                f"{getattr(self, 'layer_name', '?')}. ALL FIVE fusion kill switches must be "
                f"set, not just the GN pair: MODIFF_DISABLE_GN_MODIFF_FUSION=1 "
                f"MODIFF_DISABLE_GN_INT8_FUSION=1 MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION=1 "
                f"MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION=1 "
                f"MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION=1 -- so every conv goes through "
                f"forward(). They are read at fused_resblock import time, so they must be "
                f"set before it is imported.")

    @staticmethod
    def _fq_along_c(x: torch.Tensor, block: int, qmax: float = 127.0) -> torch.Tensor:
        """Fake-quantize NCHW `x`: `block` consecutive channels per pixel share a
        dynamic absmax scale. block<=0 means one scale for the whole tensor. A C not
        divisible by `block` leaves a short final group with its own scale."""
        if block <= 0:
            s = x.abs().amax().clamp_min(1e-12).float() / qmax
            return ((x.float() / s).round_().clamp_(-qmax, qmax) * s).to(x.dtype)
        t = x.permute(0, 2, 3, 1).float()
        n, h, w, c = t.shape
        bsz = min(block, c)
        pad = (bsz - c % bsz) % bsz
        tp = F.pad(t, (0, pad)) if pad else t
        blk = tp.reshape(n, h, w, tp.shape[-1] // bsz, bsz)
        s = blk.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        recon = ((blk / s).round_().clamp_(-qmax, qmax) * s).reshape(n, h, w, -1)[..., :c]
        return recon.permute(0, 3, 1, 2).to(x.dtype).contiguous(
            memory_format=torch.channels_last)

    @staticmethod
    def _sim_qmax() -> float:
        """MODIFF_ACT_SIM_QMAX lowers the activation grid below int8. Its purpose is a
        NEEDLE CONTROL: an error-budget sweep that comes out flat is only interpretable if
        the same metric demonstrably responds to a coarser grid. qmax=7 is int4."""
        try:
            return float(os.environ.get("MODIFF_ACT_SIM_QMAX", "127"))
        except (TypeError, ValueError):
            return 127.0

    def _sim_fq(self, x: torch.Tensor, static_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply the sim-mode quantizer selected by MODIFF_ACT_BLOCK."""
        b = self._act_block()
        if b == -3:
            return x                      # activations EXACT: the weight/attention-only arm
        qmax = self._sim_qmax()
        if b == -2 and static_scale is not None:
            # The static scale is calibrated for the int8 grid, so a lowered qmax must clamp
            # to the coarser range while keeping that scale -- this is clipping, which is
            # exactly what the shipped static path already does on 49 of 70 layers.
            s = static_scale.detach().float().reshape(())
            return ((x.float() * s).round_().clamp_(-qmax, qmax) / s).to(x.dtype)
        return self._fq_along_c(x, b if b > 0 else 0, qmax=qmax)

    @staticmethod
    def _sim_wcfg():
        """(wbits, wblock) for the sim's WEIGHT quantizer.

        MODIFF_ACT_SIM_WBITS   0  the shipped path: dequantized W8, per-output-channel (default)
                              -1  exact weights (with MODIFF_ACT_SIM_EXACT_W=1, same thing)
                               b  quantize the ORIGINAL weight to b bits
        MODIFF_ACT_SIM_WBLOCK  0  one scale per output channel, over all (C,R,S)  [the free axis]
                               N  one scale per (output channel, N consecutive C)  [along K -- costs
                                  a mainloop flush, which is the thing being priced]

        Weight granularity is the axis DeepSeek-V3 blocks at 128x128 and we currently do not block
        at all. Per-output-channel is the free axis (it factors out of the reduction); blocking
        along C does not, so it only makes sense if it buys measurable error.
        """
        def _i(k, d):
            try:
                return int(os.environ.get(k, d))
            except (TypeError, ValueError):
                return int(d)
        return _i("MODIFF_ACT_SIM_WBITS", "0"), _i("MODIFF_ACT_SIM_WBLOCK", "0")

    def _sim_weight(self) -> torch.Tensor:
        """Weight as [K,C,R,S] fp32 for the sim conv, at the configured bit width/granularity.

        Default (wbits=0) is the dequantized per-output-channel int8 the CUTLASS path uses, so the
        default sim is W8 and only the activation quantizer varies.
        """
        w = getattr(self, "_sim_w", None)
        if w is not None and w.device == self.weight_int8.device:
            return w
        wbits, wblk = self._sim_wcfg()
        exact_env = os.environ.get("MODIFF_ACT_SIM_EXACT_W") == "1"
        if wbits == 0 and not exact_env:
            K = self.out_channels
            wq = self.weight_int8.permute(0, 3, 1, 2).float()   # [K,R,S,C] -> [K,C,R,S]
            w = (wq * self.weight_scale_channel.reshape(K, 1, 1, 1).float()).contiguous()
            self._sim_w = w
            return w
        ow = getattr(self, "_orig_weight", None)
        if ow is None:
            raise RuntimeError(
                f"the weight sim needs _orig_weight on layer {getattr(self, 'layer_name', '?')}, "
                f"but apply_static_scales already freed it. MODIFF_ACT_SIM_WBITS / "
                f"MODIFF_ACT_SIM_WBLOCK / MODIFF_ACT_SIM_EXACT_W must be set BEFORE the model "
                f"is built.")
        ow = ow.detach().float()
        if exact_env or wbits < 0:
            w = ow.contiguous()
        else:
            qmax = float(2 ** (wbits - 1) - 1)          # 8 -> 127, 4 -> 7
            K, C = ow.shape[0], ow.shape[1]
            if wblk <= 0:
                sc = ow.reshape(K, -1).abs().amax(1).clamp_min(1e-12) / qmax
                w = ((ow.reshape(K, -1) / sc[:, None]).round_().clamp_(-qmax, qmax)
                     * sc[:, None]).reshape_as(ow).contiguous()
            else:
                b = min(wblk, C)
                pad = (b - C % b) % b
                t = F.pad(ow, (0, 0, 0, 0, 0, pad)) if pad else ow      # pad the C axis
                g = t.shape[1] // b
                t = t.reshape(K, g, b, *ow.shape[2:])
                sc = t.abs().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-12) / qmax
                t = (t / sc).round_().clamp_(-qmax, qmax) * sc
                w = t.reshape(K, t.shape[1] * b, *ow.shape[2:])[:, :C].contiguous()
        self._sim_w = w
        return w

    def _sim_conv(self, x: torch.Tensor, with_bias: bool) -> torch.Tensor:
        b = (self.bias.reshape(-1).float() if (with_bias and self.bias is not None) else None)
        return F.conv2d(x.float(), self._sim_weight(), b, self.stride, self.padding,
                        self.dilation, self.groups)

    def _forward_blockwise_sim(self, x: torch.Tensor) -> torch.Tensor:
        """Measurement-only forward for MODIFF_ACT_BLOCK. Mirrors the real semantics:
        baseline  out    = A(Q(a_t)) + bias
        MoDiff  t=T      a_hat = Q(a_t),  o_hat = A(a_hat) + bias
                t<T      a_hat += Q(a_t - a_hat),  o_hat += A(Q(a_t - a_hat))
        """
        if not self._smooth_is_identity:
            raise RuntimeError("MODIFF_ACT_BLOCK sim does not model SmoothQuant "
                               "(int8 churches calibration has none).")
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        static = self.static_input_scale if self.is_calibrated else None

        if not self.modiff_enabled:
            return self._sim_conv(self._sim_fq(x, static), with_bias=True).half()

        if (self.a_hat_cache is None or self.a_hat_cache.shape != x.shape
                or self.a_hat_cache.dtype != torch.float16):
            self.is_first_step = True

        if self.is_first_step:
            a_hat = self._sim_fq(x, static)
            o_hat = self._sim_conv(a_hat, with_bias=True)
            self.a_hat_cache = a_hat.half().contiguous(memory_format=torch.channels_last)
            self.o_hat_cache = o_hat.half().contiguous(memory_format=torch.channels_last)
            self.is_first_step = False
            self.step_count = 0
        else:
            self.step_count += 1
            # MoDiff's modulated steps quantize the delta on the per-step table entry;
            # -2 reproduces that, the dynamic modes derive their own scale from it.
            d_static = (self.static_delta_scale[min(self.step_count - 1,
                                                    self.static_delta_scale.numel() - 1)]
                        if bool(self.is_delta_calibrated) else None)
            dq = self._sim_fq(x - self.a_hat_cache, d_static)
            self.a_hat_cache += dq.half()
            self.o_hat_cache += self._sim_conv(dq, with_bias=False).half()

        if self._ahat_block() > 0:
            self._snap_ahat_along_c_(self.a_hat_cache, self._ahat_block(), 127.0)
        return self.o_hat_cache

    def _ahat_scale_arg(self) -> torch.Tensor:
        s = getattr(self, "_ahat_qscale", None)
        if s is None:
            dev = self.a_hat_cache.device if self.a_hat_cache is not None else torch.device("cpu")
            return self._empty_f32_arg(dev)
        if not s.is_contiguous():
            s = s.contiguous()
            self._ahat_qscale = s
        return s

    def _ensure_block_qscale(self, n: int, c: int, h: int, w: int, device) -> None:
        bsz = self._ahat_block()
        if bsz <= 0 or c % bsz != 0:
            return
        g = c // bsz
        shape = (n, h, w, g)
        s = getattr(self, "_ahat_qscale", None)
        if s is None or tuple(s.shape) != shape or s.device != device:
            self._ahat_qscale = torch.ones(shape, device=device, dtype=torch.float32)

    def _ensure_ahat_qscale(self, device) -> None:
        if self._ahat_block() > 0 and not self._imode():
            return
        qmax = self._ahat_qmax()
        s = getattr(self, "_ahat_qscale", None)
        if s is not None and s.dim() == 4:
            return
        if s is None or s.numel() < 2 or s.device != device:
            scale0 = 0.0 if self._imode() else 1.0
            self._ahat_qscale = torch.tensor([scale0, qmax], device=device, dtype=torch.float32)
        else:
            if self._imode():
                self._ahat_qscale[0] = 0.0
            self._ahat_qscale[1] = qmax

    def _pack_ahat_along_c(self) -> None:
        """Quantize a floating a_hat into int8 codes + per-block scales [N,H,W,C/B].

        t=T pack only. Later writes refresh those scales in-kernel from the
        new_c amax (same as `_snap_ahat_along_c_`).
        """
        a = self.a_hat_cache
        if a is None or a.dtype == torch.int8:
            return
        bsz = self._ahat_block()
        n, c, h, w = a.shape
        if bsz <= 0 or c % bsz != 0:
            raise RuntimeError(
                f"MODIFF_AHAT_BLOCK={bsz} must divide C={c} for layer {getattr(self, 'layer_name', '?')}")
        qmax = 127.0
        g = c // bsz
        # Fused packs, so the t=T conversion never materializes an fp32 copy of the whole
        # cache. B=32 has its own one-shot kernel; B>=64 reuses conv_quantize_block_nhwc, which
        # is the same math with the same [N,H,W,C/B] fp32 scale layout -- it exists because the
        # conv-input quantizer needed B=64, and nothing in it is specific to a live input.
        # Without this, B != 32 paid a .float() copy of every layer's cache plus a same-sized
        # intermediate, which is what made B=16/64 peak HIGHER than B=32 end to end despite a
        # smaller steady-state cache (+793 / +634 MB measured).
        fused = None
        if bsz == 32 and hasattr(modiff_cutlass, "ahat_pack_block_nhwc"):
            fused = modiff_cutlass.ahat_pack_block_nhwc(a, 32)
        elif bsz in (64, 128, 256) and hasattr(modiff_cutlass, "conv_quantize_block_nhwc"):
            fused = modiff_cutlass.conv_quantize_block_nhwc(a, bsz)
        if fused is not None:
            q, scale = fused
            s = getattr(self, "_ahat_qscale", None)
            if (s is None or s.shape != scale.shape or s.dtype != torch.float32
                    or s.device != scale.device):
                self._ahat_qscale = scale
            else:
                s.copy_(scale)
            self.a_hat_cache = q
            return
        # B < 32 has no fused pack. Chunk over N and keep the reduction in fp16 so the transient
        # is one sample's worth, not the whole cache's.
        q = torch.empty((n, h, w, c), device=a.device, dtype=torch.int8)
        scale = torch.empty((n, h, w, g), device=a.device, dtype=torch.float32)
        # 8 chunks: caps the fp32 transient at an eighth of the cache while keeping the launch
        # count low -- one chunk per sample was 1.14x slower end to end than the unchunked copy.
        step = max(1, (n + 7) // 8)
        for i in range(0, n, step):
            j = min(n, i + step)
            xi = a[i:j].permute(0, 2, 3, 1).contiguous().view(j - i, h, w, g, bsz)
            si = (xi.abs().amax(-1).float().clamp_min(1e-12)) / qmax
            scale[i:j] = si
            q[i:j] = ((xi.float() / si.unsqueeze(-1)).round()
                      .clamp_(-qmax, qmax).to(torch.int8).view(j - i, h, w, c))
        scale = scale.contiguous()
        q = q.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        s = getattr(self, "_ahat_qscale", None)
        if (s is None or s.shape != scale.shape or s.dtype != torch.float32
                or s.device != scale.device):
            self._ahat_qscale = scale
        else:
            s.copy_(scale)
        self.a_hat_cache = q

    def _pack_ahat_int8(self) -> None:
        """Quantize a floating a_hat into int8 codes + scale (per-tensor or along-C)."""
        if self._ahat_block() > 0:
            self._pack_ahat_along_c()
            return
        a = self.a_hat_cache
        if a is None or a.dtype == torch.int8:
            return
        qmax = self._ahat_qmax()
        a_f = a.float()
        amax = a_f.abs().amax().clamp_min(1e-6)
        self._ensure_ahat_qscale(a.device)
        self._ahat_qscale[0] = amax / qmax
        q = a_f.mul(qmax / amax).round_().clamp_(-qmax, qmax).to(torch.int8)
        self.a_hat_cache = q.contiguous(memory_format=torch.channels_last)

    def _unpack_ahat_to_fp16(self) -> None:
        a = self.a_hat_cache
        if a is None or a.dtype != torch.int8:
            return
        s = self._ahat_qscale
        if s is not None and s.dim() == 4:
            n, c, h, w = a.shape
            g = s.shape[-1]
            bsz = c // g
            q = a.permute(0, 2, 3, 1).reshape(n, h, w, g, bsz).float()
            recon = (q * s.unsqueeze(-1)).reshape(n, h, w, c).permute(0, 3, 1, 2)
            self.a_hat_cache = recon.to(torch.float16).contiguous(
                memory_format=torch.channels_last)
            return
        scale0 = s[0] if s is not None else 1.0
        self.a_hat_cache = (a.float() * scale0).to(torch.float16).contiguous(
            memory_format=torch.channels_last)

    def _begin_ahat_kernel(self, write_ahat: bool) -> None:
        """On a refresh commit, dequant to fp16 so the kernel writes true new_c, then pack."""
        if self._imode():
            return
        if write_ahat and self._ahat_refresh() and self.a_hat_cache is not None \
                and self.a_hat_cache.dtype == torch.int8:
            self._unpack_ahat_to_fp16()

    def _write_ahat_now(self) -> bool:
        w = not self._skip_cache_store()
        self._begin_ahat_kernel(w)
        return w

    def _ahat_dtype_ok(self, t: Optional[torch.Tensor] = None) -> bool:
        a = self.a_hat_cache if t is None else t
        return a is not None and a.dtype in (torch.float16, torch.int8, torch.int16)

    def _ensure_state_buffers(self, x: torch.Tensor):
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        h_out = ((x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x.shape[0], self.out_channels, h_out, w_out)
        o_dtype = torch.float16 if self.is_calibrated else torch.float32
        if self._imode() and self.is_calibrated:
            a_dtype = self._imode_dtype()
        else:
            a_dtype = torch.int8 if self._ahat_want_int8() else o_dtype

        if (self.a_hat_cache is None or self.a_hat_cache.shape != x.shape):
            self.a_hat_cache = torch.zeros(
                x.shape, device=x.device, dtype=a_dtype
            ).contiguous(memory_format=torch.channels_last)
            if a_dtype in (torch.int8, torch.int16):
                self._ensure_ahat_qscale(x.device)
                if a_dtype == torch.int8 and self._ahat_block() > 0:
                    n, c, h, w = x.shape
                    self._ensure_block_qscale(n, c, h, w, x.device)
        elif self.a_hat_cache.dtype != a_dtype:
            if (not self._imode() and a_dtype == torch.int8
                    and self.a_hat_cache.dtype in (torch.float16, torch.float32)):
                self._pack_ahat_int8()
            else:
                self.a_hat_cache = torch.zeros(
                    x.shape, device=x.device, dtype=a_dtype
                ).contiguous(memory_format=torch.channels_last)
                if a_dtype in (torch.int8, torch.int16):
                    self._ensure_ahat_qscale(x.device)
                    if a_dtype == torch.int8 and self._ahat_block() > 0:
                        n, c, h, w = x.shape
                        self._ensure_block_qscale(n, c, h, w, x.device)
        if (self.o_hat_cache is None or self.o_hat_cache.shape != output_shape
                or self.o_hat_cache.dtype != o_dtype):
            self.o_hat_cache = torch.zeros(
                output_shape, device=x.device, dtype=o_dtype
            ).contiguous(memory_format=torch.channels_last)

        if self.is_calibrated:
            self._residual_buf = None
        elif self._residual_buf is None or self._residual_buf.shape != x.shape:
            self._residual_buf = torch.empty_like(x)
        if self._scale_buf is None or self._scale_buf.device != x.device:
            self._scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        if self._inv_scale_buf is None or self._inv_scale_buf.device != x.device:
            self._inv_scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        if self._absmax_buf is None or self._absmax_buf.device != x.device:
            self._absmax_buf = torch.zeros(1, device=x.device, dtype=torch.float32)
        if self._retire_count is None or self._retire_count.device != x.device:
            self._retire_count = torch.zeros(1, device=x.device, dtype=torch.int32)

        if not hasattr(self, '_smooth_inv_flat') or self._smooth_inv_flat.device != x.device:
            if not self._smooth_is_identity:
                self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
            else:
                self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # MoDiff delta-scale table
    # ------------------------------------------------------------------

    def _delta_step_index(self) -> int:
        """Table index for the modulated step about to run.

        `step_count` is incremented at the top of every modulated forward and is 0 after
        `_forward_first_step` (which does not increment), so the first modulated step sees
        step_count == 1 -> index 0. Runs longer than the table clamp to its last entry.
        """
        return min(max(self.step_count - 1, 0), MODIFF_MAX_STEPS - 1)

    def _delta_should_refresh(self) -> bool:
        """Whether this modulated step recomputes the dynamic delta scale, or reuses the last one.

        The exact per-call scale costs one extra read pass over `x` and `a_hat` (+4.93 ms/step at
        int8 / +7.38 at int4, batch 128 -- the largest remaining MoDiff overhead). But the delta's
        range evolves smoothly along a DDIM trajectory, so a scale measured K steps ago may still be
        usable. `MODIFF_DELTA_REFRESH=K` recomputes every Kth step and reuses `_scale_buf` in
        between, cutting the reduction cost to 1/K.

        The reused scale can clip when the range grows between refreshes; pair it with
        the `a4` datapath limit, which saturates those codes at Q_b instead of letting them
        through at up to 127. (The `MODIFF_DELTA_CLIP` that used to leave headroom here is retired.)

        step_count is 1 on the first modulated step, so `(step_count - 1) % K == 0` always refreshes
        there -- required, since `_scale_buf` holds nothing valid before the first refresh.
        """
        k = self.delta_refresh
        return k <= 1 or ((self.step_count - 1) % k) == 0

    def _ensure_delta_dyn_bufs(self, device):
        """Allocate the 4 reduction buffers the dynamic delta scale needs.

        _ensure_state_buffers already does this, but it is keyed on the conv's own input tensor and
        the updown fusion never materializes one -- it hands the kernel the PRE-resize activation.
        So the resize path needs a device-only init. Same name and contract as Int4Conv's.
        """
        if self._scale_buf is None or self._scale_buf.device != device:
            self._scale_buf = torch.empty(1, device=device, dtype=torch.float32)
            self._inv_scale_buf = torch.empty(1, device=device, dtype=torch.float32)
            self._absmax_buf = torch.zeros(1, device=device, dtype=torch.float32)
            self._retire_count = torch.zeros(1, device=device, dtype=torch.int32)

    def _empty_f32_arg(self, device):
        """Cached 0-element fp32 tensor, the "not supplied" sentinel for the GN kernels'
        optional dynamic-scale arguments. Cached so the static path allocates nothing."""
        if getattr(self, "_empty_f32", None) is None or self._empty_f32.device != device:
            self._empty_f32 = torch.empty(0, device=device, dtype=torch.float32)
        return self._empty_f32

    def _delta_dummy_alpha(self, device):
        """Placeholder 1-element alpha for dynamic-mode GN-fused callers, which overwrite it
        with `_inv_scale_buf` before use. Cached so the hot path allocates nothing."""
        if getattr(self, "_dummy_alpha", None) is None or self._dummy_alpha.device != device:
            self._dummy_alpha = torch.ones(1, device=device, dtype=torch.float32)
        return self._dummy_alpha

    def _cur_scale_pair(self):
        """(scale, inv_scale) currently in force. See _scale_pair_b for why there are two."""
        if self._scale_pair_b:
            return self._scale_buf_b, self._inv_scale_buf_b
        return self._scale_buf, self._inv_scale_buf

    def _pub_scale_pair(self, device):
        """The pair a reporting kernel should publish INTO -- always the one not in force."""
        if self._scale_buf_b is None or self._scale_buf_b.device != device:
            self._scale_buf_b = torch.empty(1, device=device, dtype=torch.float32)
            self._inv_scale_buf_b = torch.empty(1, device=device, dtype=torch.float32)
        if self._scale_pair_b:
            return self._scale_buf, self._inv_scale_buf
        return self._scale_buf_b, self._inv_scale_buf_b

    def _delta_report_on(self) -> bool:
        """Whether this step uses free reporting rather than a separate absmax pass.

        Requires a seed: the first modulated step has no previously-published scale, so it runs one
        real absmax pass and every later step rides on the reports.

        Gated on the SAME K-step schedule as the separate pass, not on every step. Measured
        (batch 128, 2026-08-04): reporting on every step took int8 dynamic K=1 from 83.55 to
        79.98 ms/step -- much cheaper than a separate pass -- but took K=4 from 77.06 to 78.30,
        i.e. it LOST. Reporting is cheaper per occurrence but it is not free: the grid here is
        num_elements/2/256 blocks (~98k at batch 128), all contending on one atomicCAS address.
        Paying that on 100% of steps costs more than a separate pass on 25% of them. Gating it to
        refresh steps gets the cheap-per-occurrence win at the K=4 frequency.
        """
        return (self.delta_dynamic and self.delta_report and self._delta_seeded
                and self._delta_should_refresh())

    @property
    def _delta_a4(self) -> bool:
        """Whether this layer's delta quantizer is on the 4-bit datapath (codes saturate at 7).

        THE ONE PLACE that answers "how many bits is this activation". It replaced a
        `_delta_code_ceiling` returning a magnitude (`act_q`, or -1 for "use the kernel's literal
        127"), which every quantize call site had to remember to pass -- and the updown resize
        fusion did not, so those eight layers ran at 127 while the other 62 ran at act_q for
        months. A bool derived here, threaded from here, has no value to get wrong.

        Only `dynamic` mode is a4. The asymmetry is deliberate and unchanged: in static mode the
        scale is the calibrated Q_b/range rather than this call's absmax, and letting a delta above
        the calibrated range keep resolution instead of saturating is load-bearing for the published
        baseline comparison (it favours the baseline arm). Changing that belongs to its own
        measurement.

        In dynamic mode at clip 1.0 the scale is already `Q_b/absmax`, so no code can exceed Q_b and
        this changes nothing. It starts to matter in exactly two places, both of them the quantizer
        failing to honour its own bit width:
          * clip < 1, whose whole purpose is to make the top (1-clip) of the range saturate. Without
            it there was no saturation at all below A8 -- the knob was a finer grid, not a clip.
          * MODIFF_DELTA_REFRESH > 1, where a scale measured up to K-1 steps ago is reused and the
            delta's range may have grown since. Those codes are supposed to clip at Q_b; clamped at
            127 instead, an "A4" layer can emit a code of 100 on 3 of every 4 steps.
        """
        return bool(self.delta_dynamic and self.act_bits == 4)

    def _delta_gn_dynamic_args(self, device):
        """The six trailing arguments of group_norm_silu_delta_quantize[_pack]_nhwc.

        Static mode passes empty tensors, so the kernel keeps using the scale it is given.
        Dynamic mode passes the real reduction buffers, and the kernel discovers the scale from
        this call's own delta between its statistics pass and its quantize pass -- see
        gn_delta_absmax_flat_kernel. In dynamic mode the conv alpha must come from
        `_inv_scale_buf` (which that kernel writes), not from the table.

        The trailing element is the datapath's bit-width flag (see `_delta_a4`). Note the Q_level in
        these tuples and that flag are different quantities: Q_level/absmax is the SCALE the
        kernel will publish for a later step, a4 fixes where THIS call's codes saturate, and a clip
        ratio is exactly the case where they stop agreeing.
        """
        if not self.delta_dynamic:
            e = self._empty_f32_arg(device)
            return e, e, e, e, 127.0, False, 1.0, False
        a4 = self._delta_a4
        if self._delta_report_on():
            # Free reporting: the quantize kernel quantizes with the pair currently in force (the
            # caller passes it as `scale`, and the conv reads its alpha) while publishing the next
            # pair into the OTHER buffers. Flip afterwards so later steps use the new one.
            pub_s, pub_i = self._pub_scale_pair(device)
            self._scale_pair_b = not self._scale_pair_b
            return (self._absmax_buf, pub_s, pub_i,
                    self._retire_count, self.act_q,
                    True, self.delta_report_safety, a4)
        if not self._delta_should_refresh():
            # Reuse the last measured scale: pass empty reduction buffers so the kernel skips its
            # absmax pass entirely. The caller hands it `_scale_buf` as the scale to quantize with,
            # which still holds the value the last refresh wrote there. This is the case the ceiling
            # matters most for: that scale is up to K-1 steps old, so the delta may have outgrown it.
            e = self._empty_f32_arg(device)
            return e, e, e, e, 127.0, False, 1.0, a4
        self._delta_seeded = True      # this pass publishes a scale the next step can ride on
        return (self._absmax_buf, self._scale_buf, self._inv_scale_buf,
                self._retire_count, self.act_q, False, 1.0, a4)

    def _delta_scale_args(self, device, x=None, fused_silu=False):
        """(quantize_scale, conv_alpha) as 1-element device views for the current step.

        In dynamic mode (`delta_dynamic`), `x` must be the tensor the paired quantize kernel
        will consume and `fused_silu` must say whether that kernel applies SiLU itself, so the
        reduction and the quantization evaluate the same expression. The scale then comes from
        this call's own delta absmax and cannot clip, at the cost of one extra read pass. This
        is the regime the paper's Theorem 4.3 assumes; the static alternative was measured
        clipping on 49 of 70 conv layers on the real checkpoint (2026-08-04).

        Dynamic mode needs no delta calibration at all -- the table and the warning below apply
        only to static mode.

        In static mode both returns are slices of a resident table, so this costs no host sync and
        no allocation -- replacing the `float(self.static_input_scale.item())` sync the four
        modulated paths used to do on every call.

        Static mode falls back to the full-activation scale when the delta table has not been
        calibrated, which reproduces the pre-2026-08-03 (paper-incorrect) behaviour rather than
        silently emitting zeros. Warns once per layer so a missing delta-calibration file cannot
        pass unnoticed.
        """
        if self.delta_dynamic:
            if x is not None:
                # Once the GN-fused layers are publishing scales for free, this path still needs
                # its own reduction (its quantize kernel does not report yet), so keep the K-step
                # refresh here rather than tying it to _delta_report_on.
                if self._delta_should_refresh():
                    modiff_cutlass.delta_absmax_fp16(
                        x, self.a_hat_cache, self._absmax_buf, self._scale_buf,
                        self._inv_scale_buf, self._retire_count,
                        self.act_q, self._smooth_inv_flat, fused_silu)
                return self._scale_buf.view(1), self._inv_scale_buf.view(1)
            # GN-fused caller: the kernel discovers the scale itself, so return a valid but
            # unused pair. Deliberately skips the missing-calibration warning below -- dynamic
            # mode has no table to be missing.
            return self.static_input_scale.view(1), self._delta_dummy_alpha(device)
        # Read a PYTHON bool, never the device buffer. `bool(self.is_delta_calibrated)` on a
        # registered CUDA buffer forces a GPU->CPU sync, and this runs once per modulated conv per
        # step: 70 layers x 200 steps = 14000 syncs per sample. Measured cost of getting this wrong
        # -- MoDiff's overhead went from +7.45 to +12.53 ms/step, i.e. the sync alone was ~5 ms/step,
        # larger than any fusion Stage 3 proposes to win back.
        #
        # The buffer stays for serialization. Latch it into the mirror once if a state_dict load set
        # it behind our back; that costs one sync per layer per process, not per call.
        if not self._delta_cal and bool(self.is_delta_calibrated):
            self._delta_cal = True
        if not self._delta_cal:
            if not getattr(self, "_warned_no_delta_calib", False):
                self._warned_no_delta_calib = True
                print(f"⚠ {self.layer_name or type(self).__name__}: no MoDiff delta calibration; quantizing the temporal "
                      f"delta on the FULL-ACTIVATION grid. Per paper Theorem 4.3 this leaves the "
                      f"quantization error unchanged -- MoDiff buys only error feedback. Run the "
                      f"delta calibration pass.")
            if self._cached_alpha_tensor is None or self._cached_alpha_tensor.device != device:
                scale = float(self.static_input_scale.item())
                self._cached_scale_float = scale
                self._cached_alpha_tensor = torch.tensor([1.0 / scale], device=device,
                                                         dtype=torch.float32)
            return self.static_input_scale.view(1), self._cached_alpha_tensor.view(1)
        if self._imode() or os.environ.get("MODIFF_DELTA_FREEZE", "0") == "1":
            return self.static_delta_scale[0:1], self.static_delta_alpha[0:1]
        i = self._delta_step_index()
        return self.static_delta_scale[i:i + 1], self.static_delta_alpha[i:i + 1]

    def _observe_delta_codes(self, x_int8: torch.Tensor):
        """Record max|code| for this step, so the delta's true range can be recovered.

        Called only while `_delta_calib` is set. `delta_absmax = max|q| / scale_used`, exact up to
        the integer granularity of max|q| -- which is why the calibration runs twice: round 0 uses
        the (small) activation scale and lands max|q| around 10-15, round 1 uses round 0's result
        and lands it near Q, giving ~1% resolution.
        """
        if self._delta_code_max is None:
            self._delta_code_max = torch.zeros(MODIFF_MAX_STEPS, dtype=torch.float32,
                                               device=x_int8.device)
        i = self._delta_step_index()
        m = x_int8.abs().max().to(torch.float32)
        self._delta_code_max[i] = torch.maximum(self._delta_code_max[i], m)

    def effective_code_utilisation(self, x: torch.Tensor, fused_silu: bool = False) -> float:
        """max |value| this layer's static quantizer will see, in CODE units. Q (=127) is full scale.

        The one correct way to ask "is the static activation quantizer matched to what this layer
        actually sees". >Q means it is clipping. Two wrong ways were used before this existed, both
        of which produced confidently wrong numbers:

        * `127 / static_input_scale` treated as "the calibrated range" -- wrong whenever SmoothQuant
          is active, because `end_calibration` derives the scale from the SMOOTHED range,
          `127 / max_c(act_max_c / s_c)`, while the kernel quantizes `x * smooth_inv`. On this model
          it reported activations as 41000x out of range when the real figure was ~5x.
        * `x.abs().max()` straight out of a forward hook -- wrong because several entry points
          (`forward_from_int8` and friends) receive int8 CODES, so the answer is a constant 127.

        Pass the tensor as the KERNEL receives it. `fused_silu=True` when this layer applies SiLU
        itself (`fuse_input_silu`), since the fused kernels do `silu(x)` before `*= smooth_inv`.
        """
        xs = x.detach().float()
        if fused_silu:
            xs = F.silu(xs)
        if not self._smooth_is_identity:
            xs = xs * self._smooth_inv.to(xs.device, torch.float32)
        return float(xs.abs().max().item() * float(self.static_input_scale.item()))

    def dequantized_weight(self) -> torch.Tensor:
        """The fp32 weight the int8 conv effectively applies, as [K, C, kh, kw].

        `weight_int8` is stored NHWC-permuted and the per-output-channel scale is separate, so this
        is the reference any "does o_hat still equal A(a_hat)" check needs. Note it is the SMOOTHED
        weight (SmoothQuant is folded in), which pairs with the smoothed activation the kernel
        quantizes -- so conv(a_hat_cache, this) is the right reference, not conv(x, original_weight).
        """
        w = self.weight_int8.permute(0, 3, 1, 2).float()          # [K,kh,kw,C] -> [K,C,kh,kw]
        return w * self.weight_scale_channel.view(-1, 1, 1, 1).float()

    def _module_output(self) -> torch.Tensor:
        # Used to force-cast a fp16 o_hat_cache up to fp32 here, which meant every
        # calibrated MoDiff conv call materialized a full extra fp32 copy of its
        # output on the way out -- only for the very next op (autocast-managed
        # conv/linear, or our own autocast-disabled GroupNorm+SiLU) to want fp16
        # again anyway. The rest of the fp16-autocast pipeline already tolerates
        # fp16 activations natively, so just return the cache as-is.
        #
        # Deliberately NOT casting fp32 -> fp16 here. While the layer is uncalibrated the cache is
        # fp32, and forcing it to fp16 was tried: it fixes the quantized attention proj's
        # fp16-residual contract but then feeds fp16 into non-quantized convs that still hold fp32
        # biases ("Input type (c10::Half) and bias type (float) should be the same"). The
        # uncalibrated window is an fp32-flavoured pipeline and should stay one; the single consumer
        # that genuinely requires fp16 enforces it itself, in
        # quantized_std_attention._proj_with_residual.
        return self.o_hat_cache

    def _skip_cache_store(self) -> bool:
        """Naive freeze: skip in-place a_hat/o_hat stores on K-1 of every K modulated steps.

        Cadence matches the quality grid: after step_count increment, commit iff
        step_count % K == 0. t=T never increments and always commits. K=1 (the
        default until the skip-K bench lands) is today's store-every-step path.
        """
        try:
            k = int(os.environ.get("MODIFF_CACHE_SKIP_K", "1"))
        except (TypeError, ValueError):
            k = 1
        return k > 1 and self.step_count > 0 and (self.step_count % k) != 0

    def _maybe_quantize_ahat(self) -> None:
        """Snap a_hat onto an N-bit dynamic-absmax grid after a real cache write.

        Real int8 storage (AHAT_BITS=8/4 or AHAT_BLOCK>0) already quantizes
        in-kernel; this only fake-quants a still-fp16 buffer. Skip-K skip steps
        do not write.

        An AHAT_BLOCK layer only reaches the fp16 branch when it is UNcalibrated,
        i.e. it has no int8 datapath at all -- blockwise a_hat is not a storage
        change there, just eight eager kernels (abs/amax/div/round/clamp/copy)
        per layer per step. That cost 3.3 ms/step of the 3.8 ms the B=32 arm was
        losing to the fp16-a_hat arm, on layers the scheme does not even claim.
        MODIFF_AHAT_BLOCK_FAKE=1 restores it, for reproducing the pre-kernel
        fake-quant numbers in docs/ahat_blockwise_2026-09-01.
        """
        if self.calibrating or self.a_hat_cache is None:
            return
        # Real int8/int16 storage already quantizes in-kernel; do not snap the fp16 buffer.
        if self.a_hat_cache.dtype in (torch.int8, torch.int16):
            return
        bits = self._ahat_bits()
        block = self._ahat_block() if self._ahat_block_fake() else 0
        if bits >= 16 and block <= 0:
            return
        if self.step_count > 0 and self._skip_cache_store():
            return
        a = self.a_hat_cache
        if block > 0:
            self._snap_ahat_along_c_(a, block, 127.0)
            return
        qmax = 127.0 if bits >= 8 else 7.0
        scale = qmax / a.abs().amax().float().clamp_min(1e-6)
        a.mul_(scale).round_().clamp_(-qmax, qmax).div_(scale)


    def _snap_ohat_(self) -> None:
        """SIM ONLY: fake-quantize o_hat in place, blockwise along K, to price compressing it.

        o_hat is the LARGER cache (1131 MB vs a_hat's 702 at batch 128, measured) and has no
        low-precision path at all -- o_dtype is hard-wired fp16. Unlike a_hat, its error has no
        self-correction: the delta references a_hat, so a_hat's rounding is absorbed exactly by the
        next step, while nothing ever recomputes the true conv output. Its failure mode is also
        different -- the per-step increment is only 1.6-9.2% of the accumulator (measured), so at
        int8 B=32 the quantization step is 0.43-2.71 LSB of the update and plain rounding DROPS it.
        Hence MODIFF_OHAT_SIM_SR, on by default here: stochastic rounding preserves the update in
        expectation, and in the real epilogue it is free (the exact fp32 value is already in
        registers).

        Snapping the cache in place is a faithful simulation because the next step's read-modify-
        write reads exactly this value. What it does NOT simulate is memory or speed: this path
        allocates fp32 temporaries, so peak here is meaningless -- the point is Sum(zeta), the one
        quantity the kernel-1 harness cannot reach because it has no conv in the loop.

        MODIFF_OHAT_SIM_BITS  0 = off (default).  MODIFF_OHAT_SIM_BLOCK  along-K block, default 32
        -- and 32 specifically: layer L's o_hat channel axis IS layer L+1's a_hat axis, so 32
        aligns them.  MODIFF_OHAT_SIM_SR  1 = stochastic rounding (default), 0 = round-to-nearest.
        """
        try:
            bits = int(os.environ.get("MODIFF_OHAT_SIM_BITS", "0"))
        except (TypeError, ValueError):
            return
        if bits <= 0:
            return
        oh = self.o_hat_cache
        if oh is None or oh.dim() != 4:
            return
        try:
            blk = int(os.environ.get("MODIFF_OHAT_SIM_BLOCK", "32"))
        except (TypeError, ValueError):
            blk = 32
        n, k, h, w = oh.shape
        if blk <= 0 or k % blk:
            return
        lim = float(2 ** (bits - 1) - 1)
        v = oh.permute(0, 2, 3, 1).reshape(n, h, w, k // blk, blk).float()
        s = v.abs().amax(-1, keepdim=True).clamp_min(1e-12) / lim
        q = v / s
        if os.environ.get("MODIFF_OHAT_SIM_SR", "1") != "0":
            q = torch.floor(q + torch.rand_like(q))      # unbiased: E[q] is the exact value
        else:
            q = torch.round(q)
        v = (q.clamp_(-lim, lim) * s).reshape(n, h, w, k).permute(0, 3, 1, 2)
        oh.copy_(v)

    def _after_ahat_write(self, out):
        if self._imode():
            self._snap_ohat_()
            return out
        if (self._ahat_want_int8() and self.a_hat_cache is not None
                and self.a_hat_cache.dtype != torch.int8):
            self._pack_ahat_int8()
        else:
            self._maybe_quantize_ahat()
        self._snap_ohat_()
        return out

    def ahat_sat_frac(self) -> float:
        """max(|a_hat|) / qmax. I-MoDiff overflow telemetry;  >1 means the last store saturated."""
        a = self.a_hat_cache
        if a is None or a.dtype not in (torch.int8, torch.int16):
            return 0.0
        return float(a.abs().max().float().item()) / max(self._ahat_qmax(), 1.0)

    def _skip_out_buf(self) -> torch.Tensor:
        buf = getattr(self, "_skip_ohat_out", None)
        oh = self.o_hat_cache
        if (buf is None or buf.shape != oh.shape or buf.dtype != oh.dtype
                or buf.device != oh.device):
            self._skip_ohat_out = torch.empty_like(oh)
        return self._skip_ohat_out

    def _layer_out_buf(self) -> torch.Tensor:
        """Persistent fp16 output for residual EVT (commit and skip). Avoids empty_like per step."""
        oh = self.o_hat_cache
        buf = getattr(self, "_fused_out_buf", None)
        if (buf is None or buf.shape != oh.shape or buf.dtype != oh.dtype
                or buf.device != oh.device):
            self._fused_out_buf = torch.empty_like(oh)
        return self._fused_out_buf

    def _evt_ohat(self, x_q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """MoDiff o_hat conv. Skip-K writes `out = o_hat_old + conv` and leaves the cache."""
        strides = (self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                   self.dilation[0], self.dilation[1])
        wscale = self.weight_scale_channel.view(-1)
        if self._skip_cache_store():
            out = self._skip_out_buf()
            if (self.o_hat_cache.dtype == torch.float16
                    and hasattr(modiff_cutlass, "conv2d_int8_evt_o_hat_skip")):
                modiff_cutlass.conv2d_int8_evt_o_hat_skip(
                    x_q, self.weight_int8, alpha, wscale, self.o_hat_cache, out, *strides)
                return out
            # fp32 (uncalibrated) or no skip EVT: accumulate into a copy, leave the cache.
            out.copy_(self.o_hat_cache)
            (modiff_cutlass.conv2d_int8_evt_o_hat if out.dtype == torch.float16
             else modiff_cutlass.conv2d_int8_fprop_o_hat)(
                x_q, self.weight_int8, alpha, wscale, out, *strides)
            return out
        (modiff_cutlass.conv2d_int8_evt_o_hat if self.o_hat_cache.dtype == torch.float16
         else modiff_cutlass.conv2d_int8_fprop_o_hat)(
            x_q, self.weight_int8, alpha, wscale, self.o_hat_cache, *strides)
        return self._module_output()

    def _evt_ohat_residual(self, x_q: torch.Tensor, alpha: torch.Tensor,
                           residual: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        strides = (self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                   self.dilation[0], self.dilation[1])
        skip = (self._skip_cache_store()
                and hasattr(modiff_cutlass, "conv2d_int8_evt_o_hat_residual_skip"))
        fn = (modiff_cutlass.conv2d_int8_evt_o_hat_residual_skip if skip
              else modiff_cutlass.conv2d_int8_evt_o_hat_residual)
        fn(x_q, self.weight_int8, alpha, self.weight_scale_channel.view(-1),
           self.o_hat_cache, residual, out, *strides)
        return out

    # ==================================================================
    # Quantization helpers
    # ==================================================================

    def _compute_activation_scale(self, x: torch.Tensor, is_residual: bool = False) -> float:
        """Per-tensor symmetric activation scale: 127 / max(|x|).
        Used during calibration and first-step only (slow path with .item() sync).
        """
        if self.calibrating:
            abs_max = x.abs().max().item()
            scale = 127.0 / max(abs_max, 1e-6)
            if not is_residual:
                self._scale_sum += scale
                self._scale_count += 1
                with torch.no_grad():
                    ch_max = x.abs().amax(dim=(0, 2, 3))
                    if self._act_channel_max is None:
                        self._act_channel_max = ch_max.clone()
                    else:
                        torch.max(self._act_channel_max, ch_max, out=self._act_channel_max)
            return scale

        if is_residual or not self.is_calibrated:
            abs_max = x.abs().max().item()
            return 127.0 / max(abs_max, 1e-6)

        if self._cached_scale_float is None:
            self._cached_scale_float = float(self.static_input_scale.item())
        return self._cached_scale_float

    def _compute_scale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """GPU-only per-tensor scale computation. No .item() sync.
        Returns 1-element GPU tensor = 127.0 / max(|x|, 1e-6).
        Used on the modulated hot path to avoid CPU-GPU synchronization.
        """
        abs_max = x.abs().amax()
        return 127.0 / torch.clamp(abs_max, min=1e-6)

    def _dequantize_activation(self, x: torch.Tensor, input_scale) -> torch.Tensor:
        """Simulate quantize-then-dequantize: a_hat = Q(x) in FP32.
        input_scale can be float or 1-element tensor.
        """
        return (x * input_scale).round().clamp(-127, 127) / input_scale

    def _int8_conv(self, x: torch.Tensor, input_scale, with_bias: bool = True) -> torch.Tensor:
        """INT8 x INT8 convolution via CUTLASS tensor core kernel.
        input_scale can be float or 1-element GPU tensor.
        """
        if self.use_cutlass:
            if isinstance(input_scale, (int, float)):
                alpha = 1.0 / input_scale
                if (self._cached_alpha_tensor is not None
                        and self._cached_scale_float is not None
                        and input_scale == self._cached_scale_float):
                    scale_tensor = self._cached_alpha_tensor
                else:
                    scale_tensor = torch.tensor([alpha], device=x.device, dtype=torch.float32)
                x_scaled = x * input_scale
                if not x_scaled.is_contiguous(memory_format=torch.channels_last):
                    x_scaled = x_scaled.contiguous(memory_format=torch.channels_last)
                x_int8 = x_scaled.round().clamp(-127, 127).to(torch.int8)
            else:
                # Tensor path: use fused scale+quantize kernel (no CPU sync)
                scale_tensor = (1.0 / input_scale).view(1)
                if not x.is_contiguous(memory_format=torch.channels_last):
                    x = x.contiguous(memory_format=torch.channels_last)
                x_int8 = modiff_cutlass.scale_quantize_int8(x, input_scale)

            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, device=x.device)

            out_raw = modiff_cutlass.conv2d_int8_fprop(
                x_int8,
                self.weight_int8,
                scale_tensor,
                self._empty_bias,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1]
            )
            # Dequantize per-channel
            out = out_raw * self.weight_scale_channel
        else:
            raise RuntimeError(
                f"CUTLASS INT8 kernel unavailable for layer {self.layer_name} "
                f"(groups={self.groups}). Build modiff_cutlass extension."
            )

        if with_bias and self.bias is not None:
            out = out + self.bias
        return out

    def _ensure_dynamic_buffers(self, x: torch.Tensor):
        if self._dyn_scale_buf is None or self._dyn_scale_buf.device != x.device:
            self._dyn_scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
            self._dyn_inv_scale_buf = torch.empty(1, device=x.device, dtype=torch.float32)
            self._dyn_absmax_buf = torch.zeros(1, device=x.device, dtype=torch.float32)
            self._dyn_retire_count = torch.zeros(1, device=x.device, dtype=torch.int32)

    def _int8_conv_dynamic_fused(self, x: torch.Tensor, with_bias: bool = True) -> torch.Tensor:
        """Cache-free dynamic (uncalibrated) INT8 conv: fuses the absmax
        reduction + scale/inv_scale computation into one kernel
        (dynamic_quantize_int8_fprop -> compute_dynamic_scale), instead of
        the generic _int8_conv's tensor-scale path, which does the reduction
        via a plain `.abs().amax()` PyTorch call and a separate `1.0/scale`
        reciprocal — this collapses those into the same fused kernel already
        used elsewhere, and avoids ever materializing a residual buffer
        (there is no cache here, so unlike the MoDiff dynamic path there is
        nothing to subtract).
        """
        # dynamic_quantize_int8_fprop's kernels read x via data_ptr<float>(); unlike
        # _int8_conv's tensor-scale branch (which happens to promote to fp32 through
        # `x * input_scale` in its scalar branch, or would hard-error via data_ptr<float>()
        # in its tensor branch), cast explicitly so this path is correct regardless of
        # what dtype the previous layer produced.
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_dynamic_buffers(x)

        x_int8 = modiff_cutlass.dynamic_quantize_int8_fprop(
            x, self._dyn_absmax_buf, self._dyn_scale_buf,
            self._dyn_inv_scale_buf, self._dyn_retire_count
        )

        if self._empty_bias is None or self._empty_bias.device != x.device:
            self._empty_bias = torch.empty(0, device=x.device)

        out_raw = modiff_cutlass.conv2d_int8_fprop(
            x_int8,
            self.weight_int8,
            self._dyn_inv_scale_buf.view(1),
            self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        out = out_raw * self.weight_scale_channel
        if with_bias and self.bias is not None:
            out = out + self.bias
        return out

    # ==================================================================
    # Forward paths
    # ==================================================================

    def _can_fuse_input_silu(self, x: torch.Tensor) -> bool:
        """True when this call can take the fused SiLU+quantize kernel path:
        `x` must be the pre-activation ResBlock GroupNorm output (see
        fuse_input_silu / fused_resblock.py's wire_silu_fusion), not yet
        SiLU'd, and everything the fused kernel requires (calibrated, FP16
        cache, matching shape/dtype) must already hold.
        """
        return (self.fuse_input_silu and self.modiff_enabled and not self.is_first_step
                and self.is_calibrated and HAS_CUTLASS and self.use_cutlass
                and self.a_hat_cache is not None
                and self._ahat_dtype_ok()
                and self.a_hat_cache.shape == x.shape
                and x.dtype == torch.float16)

    def _forward_modulated_static_fused_silu(self, x: torch.Tensor) -> torch.Tensor:
        """Same as _forward_modulated's calibrated CUTLASS branch, but `x` is
        the pre-activation input -- SiLU is applied inline inside
        step1_static_quantize_fprop_silu's CUDA kernel instead of a separate
        F.silu(x) Python call over the whole activation tensor beforehand.
        """
        self._sim_guard('_forward_modulated_static_fused_silu')
        self.step_count += 1
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_state_buffers(x)

        d_scale, d_alpha = self._delta_scale_args(x.device, x, fused_silu=True)

        if not hasattr(self, '_smooth_inv_flat') or self._smooth_inv_flat.device != x.device:
            if not self._smooth_is_identity:
                self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
            else:
                self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

        p_step1 = profiler.start("MoDiff INT8 Static Step1 (fused SiLU)")
        x_int8 = modiff_cutlass.step1_static_quantize_fprop_silu(
            x,
            self.a_hat_cache,
            d_scale,
            self._smooth_inv_flat,
            self._delta_a4,
            self._write_ahat_now(),
            self._ahat_scale_arg(),
        )
        profiler.stop("MoDiff INT8 Static Step1 (fused SiLU)", p_step1)
        if self._delta_calib:
            self._observe_delta_codes(x_int8)

        p_conv = profiler.start("MoDiff INT8 Static Conv2d")
        # alpha MUST be the reciprocal of the scale the quantize above used: the GEMM computes
        # acc * alpha * w_scale[k], so a mismatched pair silently rescales the whole increment.
        out = self._evt_ohat(x_int8, d_alpha)
        profiler.stop("MoDiff INT8 Static Conv2d", p_conv)
        return self._after_ahat_write(out)

    def can_gn_fuse_modiff(self, x: torch.Tensor) -> bool:
        """Eligibility for the fused GroupNorm+SiLU+delta-quantize modiff path
        (group_norm_silu_delta_quantize_nhwc). Same gate as _can_fuse_input_silu
        (fuse_input_silu + calibrated + fp16 cache present + shape/dtype match +
        not first step) plus groups==1 and cuda. The caller (_prequant_gn_conv)
        additionally checks the GN-native conditions (channels_last, C%ng==0)
        and that the kernel is available."""
        return (self._can_fuse_input_silu(x)
                and getattr(self, 'groups', 1) == 1
                and x.is_cuda)

    def forward_gn_fused_modiff(self, x, gn_weight, gn_bias, num_groups, eps,
                                mod_scale2d, mod_shift2d, residual=None):
        """Fused GroupNorm(+scale-shift mod)+SiLU + INT8 temporal-delta quantize
        + o_hat conv, in one GN-quantize kernel + one conv. Replaces the
        standalone GroupNorm kernel + step1_static_quantize_fprop_silu that
        _forward_modulated_static_fused_silu runs back-to-back, removing the fp16
        `normed` round-trip between them. Bit-identical to that two-kernel path
        (the kernel replicates the fp16 rounding of `normed` before SiLU, and the
        a_hat update `cache += q/scale` is unchanged). Caller must have verified
        can_gn_fuse_modiff(x). `mod_scale2d`/`mod_shift2d` are [N,C] (or empty)
        matching x.dtype; `residual` (fp16, or None) is added to the output."""
        self._sim_guard('forward_gn_fused_modiff')
        self.step_count += 1
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_state_buffers(x)
        # The quantity being quantized here is silu(gn(x)), which only exists inside the fused
        # kernel -- so dynamic mode cannot pre-reduce it from `x`. Instead the kernel takes the
        # reduction buffers and discovers the scale internally, reusing its own GN statistics
        # (gn_delta_absmax_flat_kernel). In dynamic mode the conv alpha is what that kernel
        # writes to _inv_scale_buf, not the table entry.
        d_scale, d_alpha = self._delta_scale_args(x.device)
        # Read the pair in force BEFORE _delta_gn_dynamic_args, which flips it on a reporting step.
        cur_s, cur_i = self._cur_scale_pair()
        gn_dyn = self._delta_gn_dynamic_args(x.device)
        if self.delta_dynamic:
            # In dynamic mode the scale always lives in _scale_buf: on a refresh step the kernel
            # writes it there before quantizing, on a reuse step it is already there from the last
            # refresh. Either way the quantize reads that pointer and alpha is its reciprocal.
            d_scale = cur_s.view(1)
            d_alpha = cur_i.view(1)

        p_step1 = profiler.start("MoDiff INT8 GN-fused Step1 (GN+SiLU+delta)")
        # EXPERIMENTAL A/B, default off: the baseline-style group-major single kernel (one block
        # per group, a_hat added on -- see csrc/modiff/norm/group_norm_silu.cu's
        # group_norm_silu_delta_quantize_nhwc_fused, kept there as measured-regression dead code)
        # wired into the real dispatch path so it can be timed on real activations, not just
        # synthetic tensors. Static delta mode + even CPG only, same constraints the kernel
        # TORCH_CHECKs; falls through to the shipped channel-major path otherwise.
        write_ahat = self._write_ahat_now()
        if (_GN_GROUPMAJOR and not self.delta_dynamic
                and self.a_hat_cache.dtype == torch.float16
                and (self.a_hat_cache.shape[1] // num_groups) % 2 == 0):
            x_int8 = modiff_cutlass.group_norm_silu_delta_quantize_nhwc_fused(
                x, gn_weight, gn_bias, self.a_hat_cache, num_groups, eps, True,
                d_scale, self._smooth_inv_flat, mod_scale2d, mod_shift2d, False, write_ahat)
        else:
            x_int8 = modiff_cutlass.group_norm_silu_delta_quantize_nhwc(
                x, gn_weight, gn_bias, self.a_hat_cache, num_groups, eps, True,
                d_scale, self._smooth_inv_flat,
                mod_scale2d, mod_shift2d, *gn_dyn, write_ahat, self._ahat_scale_arg())
        profiler.stop("MoDiff INT8 GN-fused Step1 (GN+SiLU+delta)", p_step1)
        if self._delta_calib:
            self._observe_delta_codes(x_int8)

        if residual is not None:
            # EVT dual-store (same conv2d_int8_evt_o_hat_residual kernel
            # forward_modiff_fused_silu_residual already uses): fold the ResBlock
            # skip-add into the o_hat conv's accumulate epilogue instead of the
            # separate eager `out + residual` below -- removes an elementwise-add
            # kernel AND (for the GN-fusion out_conv->next-block-in_conv edge)
            # restores a direct producer/consumer relationship between this
            # conv's output and the next GroupNorm's input, with no intervening
            # op. Caller's can_gn_fuse_modiff(x) precondition (is_calibrated=True)
            # already guarantees a_hat/o_hat_cache are fp16 (see
            # _ensure_state_buffers), matching forward_modiff_fused_silu_residual's
            # own (unconditional) use of the EVT kernel.
            residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
            out = self._layer_out_buf()
            p_conv = profiler.start("MoDiff INT8 Static Conv2d (o_hat+residual)")
            self._evt_ohat_residual(x_int8, d_alpha, residual, out)
            profiler.stop("MoDiff INT8 Static Conv2d (o_hat+residual)", p_conv)
            return self._after_ahat_write(out)

        p_conv = profiler.start("MoDiff INT8 Static Conv2d")
        out = self._evt_ohat(x_int8, d_alpha)
        profiler.stop("MoDiff INT8 Static Conv2d", p_conv)
        return self._after_ahat_write(out)

    def forward_modiff_fused_silu_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Same as _forward_modulated_static_fused_silu, but fuses the ResBlock
        skip-add into the o_hat conv's accumulate epilogue
        (conv2d_int8_fprop_o_hat_residual): returns o_hat + residual in one pass,
        with NO trailing aten::add and WITHOUT polluting the o_hat cache (the cache
        write is byte-identical to the non-residual path). Caller must have verified
        _can_fuse_input_silu(x). `residual` is the ResBlock skip (cast to fp16
        channels_last here)."""
        self._sim_guard('forward_modiff_fused_silu_residual')
        self.step_count += 1
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        self._ensure_state_buffers(x)
        d_scale, d_alpha = self._delta_scale_args(x.device, x, fused_silu=True)

        p_step1 = profiler.start("MoDiff INT8 Static Step1 (fused SiLU)")
        x_int8 = modiff_cutlass.step1_static_quantize_fprop_silu(
            x, self.a_hat_cache, d_scale, self._smooth_inv_flat, self._delta_a4,
            self._write_ahat_now(), self._ahat_scale_arg())
        profiler.stop("MoDiff INT8 Static Step1 (fused SiLU)", p_step1)
        if self._delta_calib:
            self._observe_delta_codes(x_int8)

        residual = residual.to(torch.float16).contiguous(memory_format=torch.channels_last)
        out = self._layer_out_buf()
        p_conv = profiler.start("MoDiff INT8 Static Conv2d (o_hat+residual)")
        # EVT dual-store: o_hat += acc*alpha*weight_scale (in place) and out = o_hat_new +
        # residual, in ONE conv pass -- removes the fp32 conv_out round-trip of the old
        # conv2d_int8_fprop_o_hat_residual (verified bit-exact o_hat + out; ~1.4-1.8x faster b128).
        # Skip-K uses the out-only twin so o_hat is not committed.
        self._evt_ohat_residual(x_int8, d_alpha, residual, out)
        profiler.stop("MoDiff INT8 Static Conv2d (o_hat+residual)", p_conv)
        return self._after_ahat_write(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_start = profiler.start("Layer: OptimizedInt8Conv2d.forward")

        if self._act_block() != 0:
            # Measurement harness, not a fast path -- see _forward_blockwise_sim.
            out = self._forward_blockwise_sim(F.silu(x) if self.fuse_input_silu else x)
            profiler.stop("Layer: OptimizedInt8Conv2d.forward", fwd_start)
            return out

        if self._conv_blockk() != 0:
            xb = F.silu(x) if self.fuse_input_silu else x
            if self._blockk_eligible(xb):
                out = self._forward_conv_blockk(xb)
                profiler.stop("Layer: OptimizedInt8Conv2d.forward", fwd_start)
                return out
            # Ineligible (C%64, odd Kout, uncalibrated, grouped/dilated): fall through to the
            # shipped path. x, not xb -- the shipped path applies its own SiLU.

        if self.fuse_input_silu:
            if self._can_fuse_input_silu(x):
                output = self._forward_modulated_static_fused_silu(x)
                profiler.stop("Layer: OptimizedInt8Conv2d.forward", fwd_start)
                return output
            # Fast path not applicable this call (first step / uncalibrated /
            # dtype mismatch) -- the caller passed the pre-activation input
            # expecting this layer to apply SiLU itself, so do it explicitly.
            x = F.silu(x)

        # The calibrated MoDiff modulated path's CUDA kernel (step1_static_quantize_fprop)
        # reads fp16 x directly, so skip this cast there -- it used to force a full-tensor
        # fp16->fp32 copy of every activation before every quantized conv call, which cost
        # more GPU time than the quantized conv itself (see FusedGroupNormSiLU's sibling
        # analysis; profiling showed aten::copy_ as the single largest kernel-time bucket
        # in int8/int4 mode). The other paths (calibration, uncalibrated dynamic MoDiff)
        # still use fp32-only kernels, so they keep the upfront cast.
        if x.dtype != torch.float32 and (self.calibrating or (self.modiff_enabled and not self.is_calibrated)):
            x = x.float()
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        # SmoothQuant: equalize per-channel activation ranges
        # For modulated path, SmoothQuant is fused into sub_absmax_scale kernel
        if not self._smooth_is_identity and (not self.modiff_enabled or self.is_first_step):
            x = x * self._smooth_inv

        if not self.modiff_enabled:
            output = self._forward_standard(x)
        elif self.is_first_step:
            output = self._forward_first_step(x)
            self.is_first_step = False
        else:
            output = self._forward_modulated(x)

        profiler.stop("Layer: OptimizedInt8Conv2d.forward", fwd_start)
        return output

    # ---- blockwise-along-C conv input, REAL kernels (MODIFF_CONV_BLOCKK) ---------
    #
    # Wires csrc/modiff/conv/conv2d_int8_blockk.cu into the model. Unlike MODIFF_ACT_BLOCK
    # (a fp32 simulation harness), this is the real datapath: blockwise int8 codes from
    # conv_quantize_block_nhwc, dequantized per K-block inside the conv mainloop.
    #
    #   MODIFF_CONV_BLOCKK = 0       off (default)
    #                        32|64   block size along C
    #   MODIFF_CONV_BLOCKK_CTRL=1    matched scalar-alpha control at the SAME tile config
    #                                (same hand-written kernel, per-tensor static scale), which
    #                                is what separates "hand tile vs CUTLASS" from "blockwise".
    #
    # READ THE COST HONESTLY. Two things here are NOT blockwise costs and must not be
    # attributed to it:
    #   1. The quantize is a SEPARATE pass. The shipped path fuses GN+SiLU+quantize into one
    #      kernel; there is no blockwise-emitting variant of that kernel, so this path pays a
    #      full extra read+write of the activation. docs/... fused-vs-separate measured that
    #      fusion at 4.85x on the quantize step alone.
    #   2. The MoDiff arm materializes the dequantized delta to update a_hat, which the shipped
    #      path folds into its quantize kernel.
    # The CTRL arm pays both of those too, so CTRL->blockwise is the clean blockwise delta.
    @staticmethod
    def _conv_blockk() -> int:
        try:
            return int(os.environ.get("MODIFF_CONV_BLOCKK", "0"))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _conv_blockk_ctrl() -> bool:
        return os.environ.get("MODIFF_CONV_BLOCKK_CTRL") == "1"

    def _blockk_eligible(self, x: torch.Tensor) -> bool:
        """The kernel's hard constraints. C%64 is BKC_CTA_K, not the block size -- so B=32 is
        ALSO gated on C%64==0. Kout%2 is the epilogue's __half2 column pair. Everything that
        fails here falls back to the shipped path, which is why the first conv (C=4) is fine."""
        blk = self._conv_blockk()
        if blk not in (32, 64, 128, 256):
            return False
        c = x.shape[1]
        return (c % 64 == 0 and c % blk == 0 and self.out_channels % 2 == 0
                and self.is_calibrated and self._smooth_is_identity
                and self.dilation[0] == 1 and self.dilation[1] == 1
                and self.groups == 1
                and self.stride[0] == self.stride[1] and self.padding[0] == self.padding[1]
                and x.dtype in (torch.float16, torch.float32))

    def _blockk_args(self):
        w = getattr(self, "_blockk_w", None)
        if w is None:
            self._blockk_w = self.weight_int8.contiguous()
            self._blockk_ws = self.weight_scale_channel.reshape(-1).float().contiguous()
            self._blockk_bias = (self.bias.reshape(-1).half().contiguous()
                                 if self.bias is not None else None)
        return self._blockk_w, self._blockk_ws, self._blockk_bias

    def _blockk_quant(self, v: torch.Tensor, blk: int):
        """(codes, block_scales_or_empty, scalar_dequant_scale). CTRL quantizes per-tensor on the
        calibrated static scale -- the same grid the shipped path uses -- so the control differs
        from the shipped arm only in which conv kernel runs."""
        if not v.is_contiguous(memory_format=torch.channels_last):
            v = v.contiguous(memory_format=torch.channels_last)
        if self._conv_blockk_ctrl():
            if self._empty_smooth is None or self._empty_smooth.device != v.device:
                self._empty_smooth = torch.empty(0, device=v.device, dtype=torch.float32)
            vh = v if v.dtype == torch.float16 else v.half()
            q = modiff_cutlass.step1_static_quantize_noahat_fprop(
                vh, self.static_input_scale.view(1), self._empty_smooth)
            if self._blockk_empty is None or self._blockk_empty.device != v.device:
                self._blockk_empty = torch.empty(0, device=v.device, dtype=torch.float32)
            return q, self._blockk_empty, 1.0 / float(self.static_input_scale.item())
        q, sb = modiff_cutlass.conv_quantize_block_nhwc(v, blk)
        return q, sb, 0.0

    @staticmethod
    def _blockk_dequant(q: torch.Tensor, sb: torch.Tensor, blk: int) -> torch.Tensor:
        """codes * per-(pixel, C-block) scale -> fp16 NHWC. Only the MoDiff arm needs this, to
        keep a_hat equal to the value the conv actually consumed."""
        n, c, h, w = q.shape
        t = q.permute(0, 2, 3, 1).reshape(n, h, w, c // blk, blk).float()
        out = (t * sb.reshape(n, h, w, c // blk, 1)).reshape(n, h, w, c)
        return out.permute(0, 3, 1, 2).half().contiguous(memory_format=torch.channels_last)

    def blockk_gn_fused(self, x, gn_w, gn_b, ng, eps, apply_silu, mod_scale, mod_shift,
                        residual):
        """FUSED GN(+mod)(+SiLU) -> blockwise B=32 quantize -> blockk conv. B=32 only: the fused
        kernel's 16-lane x 2-channel group is exactly one B=32 block.

        This is the path that removes the +21/+16 ms fusion loss the first wiring paid
        (docs/conv_blockk_e2e_2026-09-02): one kernel for GN+SiLU+blockwise-quantize+a_hat,
        then the conv. Returns None when not eligible so the caller keeps its own path.
        """
        blk = self._conv_blockk()
        if blk not in (32, 64) or not self._blockk_eligible(x):
            return None
        c = x.shape[1]
        if c % ng != 0 or (c // ng) % 2 != 0:
            return None
        wq, ws, bias = self._blockk_args()
        st, pd = int(self.stride[0]), int(self.padding[0])
        if self._blockk_empty is None or self._blockk_empty.device != x.device:
            self._blockk_empty = torch.empty(0, device=x.device, dtype=torch.float32)
        em = self._blockk_empty
        eh = x.new_empty(0)
        ms = mod_scale.reshape(x.shape[0], c).contiguous() if mod_scale is not None else eh
        sh = mod_shift.reshape(x.shape[0], c).contiguous() if mod_shift is not None else eh
        smooth = em if self._smooth_is_identity else self._smooth_inv.view(-1).float().contiguous()

        if not self.modiff_enabled:
            q, sb = modiff_cutlass.gn_silu_blockk_quantize_b32(
                x, gn_w, gn_b, eh, ng, eps, apply_silu, smooth, ms, sh, blk)
            rz = None if residual is None else residual.to(torch.float16).contiguous(
                memory_format=torch.channels_last)
            return modiff_cutlass.conv2d_int8_blockk(q, wq, ws, sb, 0.0, blk, st, pd,
                                                     bias, None, rz)

        # `is_first_step` is the authority, NOT the cache being None: reset_state() ZEROES
        # a_hat/o_hat in place and sets is_first_step=True, it does not free them. An
        # allocation-shaped check therefore reports "not first" on the first step of every
        # sample after the first, which silently skips re-seeding o_hat and accumulates the
        # whole sample with bias=None. That measured relL2 0.4888 instead of 0.0641.
        if (self.a_hat_cache is None or self.a_hat_cache.shape != x.shape
                or self.a_hat_cache.dtype != torch.float16):
            self.is_first_step = True
        first = self.is_first_step
        if first:
            # a_hat = 0 makes the MoDiff kernel compute delta = GN(x) - 0 = GN(x) and then
            # a_hat += dequant(codes), which IS the t=T semantics -- no separate eager pass.
            if (self.a_hat_cache is None or self.a_hat_cache.shape != x.shape
                    or self.a_hat_cache.dtype != torch.float16):
                self.a_hat_cache = torch.zeros_like(x, dtype=torch.float16,
                                                    memory_format=torch.channels_last)
            else:
                self.a_hat_cache.zero_()
            self.step_count = 0
            self.is_first_step = False
        q, sb = modiff_cutlass.gn_silu_blockk_quantize_b32(
            x, gn_w, gn_b, self.a_hat_cache, ng, eps, apply_silu, smooth, ms, sh, blk)
        if first:
            # already channels_last out of the kernel; no re-contiguous
            self.o_hat_cache = modiff_cutlass.conv2d_int8_blockk(
                q, wq, ws, sb, 0.0, blk, st, pd, bias, None)
        else:
            self.step_count += 1
            modiff_cutlass.conv2d_int8_blockk(q, wq, ws, sb, 0.0, blk, st, pd, None,
                                              self.o_hat_cache)
        out = self.o_hat_cache
        return out if residual is None else out + residual

    def _forward_conv_blockk(self, x: torch.Tensor) -> torch.Tensor:
        blk = self._conv_blockk()
        wq, ws, bias = self._blockk_args()
        st, pd = int(self.stride[0]), int(self.padding[0])
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        if not self.modiff_enabled:
            q, sb, a_s = self._blockk_quant(x, blk)
            return modiff_cutlass.conv2d_int8_blockk(q, wq, ws, sb, a_s, blk, st, pd, bias, None)

        if (self.a_hat_cache is None or self.a_hat_cache.shape != x.shape
                or self.a_hat_cache.dtype != torch.float16):
            self.is_first_step = True

        if self.is_first_step:
            q, sb, a_s = self._blockk_quant(x, blk)
            o_hat = modiff_cutlass.conv2d_int8_blockk(q, wq, ws, sb, a_s, blk, st, pd, bias, None)
            if self._conv_blockk_ctrl():
                a_hat = (q.float() * a_s).half().contiguous(memory_format=torch.channels_last)
            else:
                a_hat = self._blockk_dequant(q, sb, blk)
            self.a_hat_cache = a_hat
            self.o_hat_cache = o_hat.contiguous(memory_format=torch.channels_last)
            self.is_first_step = False
            self.step_count = 0
            return self.o_hat_cache

        self.step_count += 1
        d = x - self.a_hat_cache
        q, sb, a_s = self._blockk_quant(d, blk)
        if self._conv_blockk_ctrl():
            dq = (q.float() * a_s).half().contiguous(memory_format=torch.channels_last)
        else:
            dq = self._blockk_dequant(q, sb, blk)
        self.a_hat_cache = (self.a_hat_cache + dq).contiguous(memory_format=torch.channels_last)
        # bias=None: the MoDiff modulated step accumulates the DELTA's contribution only; the
        # bias is already in o_hat from t=T. o_hat_opt makes this an in-kernel RMW.
        modiff_cutlass.conv2d_int8_blockk(q, wq, ws, sb, a_s, blk, st, pd, None,
                                          self.o_hat_cache)
        return self.o_hat_cache

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard INT8 forward without MoDiff modulation.

        When static scales are available (is_calibrated=True), uses the same
        fused CUDA kernels as the MoDiff modulated path:
            scale_quantize_int8 → conv2d_int8_fprop
        This avoids separate PyTorch round/clamp/cast kernels and is the
        only fair baseline against which to measure temporal caching overhead.

        When not calibrated, falls back to the naive PyTorch path (which
        includes a CPU-GPU sync via .item() in _compute_activation_scale).
        """
        if self.is_calibrated and HAS_CUTLASS and self.use_cutlass:
            self._ensure_conv_caches(x.device)
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
            if x.dtype == torch.float16:
                # Cache-free static quantize (baseline: no temporal cache). Reads fp16 x directly
                # (no fp32 cast) and does NOT touch a_hat — dropping the per-call a_hat zero-fill +
                # a_hat read+write that step1_static_quantize_fprop(x, a_hat=0) wasted. Output is
                # bit-identical (residual=x-0=x). SmoothQuant already applied upstream -> smooth empty.
                if self._empty_smooth is None or self._empty_smooth.device != x.device:
                    self._empty_smooth = torch.empty(0, device=x.device, dtype=torch.float32)
                x_int8 = modiff_cutlass.step1_static_quantize_noahat_fprop(
                    x, self.static_input_scale.view(1), self._empty_smooth
                )
            else:
                x_for_quant = x if x.dtype == torch.float32 else x.float()
                x_int8 = modiff_cutlass.scale_quantize_int8(x_for_quant, self._cached_scale_tensor)
            return self._conv_from_int8(x_int8)
        # Fallback: during calibration we need the host-visible scale path so the
        # module can accumulate static activation statistics. Outside calibration
        # we use the fully-fused cache-free dynamic-scale kernel (no cache, no
        # residual -- see _int8_conv_dynamic_fused) instead of _compute_scale_tensor's
        # separate .amax() + reciprocal + _int8_conv's tensor-scale branch, which
        # needlessly cost 2 extra small kernel launches on every uncalibrated call.
        if self.calibrating:
            input_scale = self._compute_activation_scale(x)
            return self._int8_conv(x, input_scale, with_bias=True)
        return self._int8_conv_dynamic_fused(x, with_bias=True)

    def _apply(self, *args, **kwargs):
        """Keep the packed INT8 weight standard-contiguous through any tensor
        transform. `model.to(memory_format=torch.channels_last)` (applied to make
        activations channels_last) also reformats the [K,R,S,C] `weight_int8` buffer
        to a channels_last stride -- which for R,S>1 (3x3 convs) silently transposes
        the physical layout the CUTLASS conv kernel reads, producing garbage output
        (1x1 convs are unaffected: channels_last == contiguous there). This was
        invisible to random-weight consistency and speed checks; only real accuracy
        vs fp16 exposed it. Re-contiguating after the transform costs one small copy."""
        out = super()._apply(*args, **kwargs)
        wi = getattr(self, "weight_int8", None)
        if wi is not None and wi.dim() == 4 and not wi.is_contiguous():
            self.weight_int8 = wi.contiguous()
        return out

    def _ensure_conv_caches(self, device):
        """Lazy-init the per-tensor scale caches reused by the calibrated conv path."""
        if self._cached_scale_float is None:
            self._cached_scale_float = float(self.static_input_scale.item())
        if self._cached_alpha_tensor is None:
            self._cached_alpha_tensor = torch.tensor(
                [1.0 / self._cached_scale_float], device=device, dtype=torch.float32)
        if self._cached_scale_tensor is None:
            self._cached_scale_tensor = torch.tensor(
                [self._cached_scale_float], device=device, dtype=torch.float32)
        if self._empty_bias is None or self._empty_bias.device != device:
            self._empty_bias = torch.empty(0, device=device)
        if self._empty_smooth is None or self._empty_smooth.device != device:
            self._empty_smooth = torch.empty(0, device=device, dtype=torch.float32)

    def _ensure_tuned_config(self, x_int8: torch.Tensor, output_shape) -> Optional[int]:
        """Lazily pick the fastest CUTLASS tile config for this conv's shape by
        timing all configs on the actual int8 input (cuDNN-style per-shape select).
        Cached in _tuned_config_id. Returns -1 (use fixed default tile) when
        autotuning is disabled or the deep-fuse tuned kernel is unavailable."""
        if self._tuned_config_id is not None:
            return self._tuned_config_id
        if not _CONV_AUTOTUNE:
            self._tuned_config_id = -1
            return -1
        ncfg = modiff_cutlass.conv2d_int8_num_tuned_configs()
        buf = torch.empty(output_shape, device=x_int8.device, dtype=torch.float16
                          ).contiguous(memory_format=torch.channels_last)
        wscale_h = self.weight_scale_channel_half.view(-1)
        args = (x_int8, self.weight_int8, self._cached_alpha_tensor, wscale_h, buf)
        strides = (self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                   self.dilation[0], self.dilation[1])
        best_t, best_id = float("inf"), -1
        for cid in range(ncfg):
            try:
                for _ in range(3):
                    modiff_cutlass.conv2d_int8_dequant_fp16_tuned(*args, cid, *strides)
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(10):
                    modiff_cutlass.conv2d_int8_dequant_fp16_tuned(*args, cid, *strides)
                e.record(); torch.cuda.synchronize()
                t = s.elapsed_time(e)
            except Exception:
                continue
            if t < best_t:
                best_t, best_id = t, cid
        self._tuned_config_id = best_id  # -1 if every config failed -> fixed default
        return self._tuned_config_id

    def _conv_from_int8_o_hat(self, x_int8: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """MoDiff conv from PRE-QUANTIZED delta codes: the EVT in-place o_hat accumulate.

        Exists for the updown resize fusion (fused_resblock._prequant_gn_resize_conv_modiff), which
        produces the delta codes itself in the GN+resize kernel and so needs the conv half of the
        modulated step on its own. `alpha` must be the reciprocal of the scale that quantized
        x_int8, or o_hat is accumulated on the wrong scale for every remaining timestep.
        """
        self._sim_guard('_conv_from_int8_o_hat')
        self._ensure_state_buffers_from_codes(x_int8)
        p = profiler.start("MoDiff INT8 Static Conv2d (from codes)")
        out = self._evt_ohat(x_int8, alpha)
        profiler.stop("MoDiff INT8 Static Conv2d (from codes)", p)
        return out

    def _ensure_state_buffers_from_codes(self, x_int8: torch.Tensor):
        """o_hat sizing for the from-codes path. a_hat is already allocated (the caller checked its
        shape before fusing); only o_hat needs to exist at the conv's output shape."""
        n, c, h, w = x_int8.shape
        h_out = ((h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                 // self.stride[0]) + 1
        w_out = ((w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                 // self.stride[1]) + 1
        shape = (n, self.out_channels, h_out, w_out)
        dtype = torch.float16 if self.is_calibrated else torch.float32
        if (self.o_hat_cache is None or self.o_hat_cache.shape != shape
                or self.o_hat_cache.dtype != dtype):
            self.o_hat_cache = torch.zeros(shape, device=x_int8.device, dtype=dtype
                                           ).contiguous(memory_format=torch.channels_last)

    def _conv_from_int8(self, x_int8: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the calibrated INT8 conv (dequant/bias/store dispatch) on an already
        -quantized channels_last int8 activation. Shared by _forward_standard (which
        quantizes first) and forward_from_int8 (which skips the quantize).

        If `residual` (fp16 channels_last, same shape as the conv output) is given,
        it is added in the store epilogue (fusing a ResBlock skip-add) rather than
        as a separate aten::add."""
        self._ensure_conv_caches(x_int8.device)
        h_out = ((x_int8.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x_int8.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x_int8.shape[0], self.out_channels, h_out, w_out)
        bias_fused = False
        residual_fused = False
        if self.standard_output_fp16:
            if (self._standard_output_buf is None
                    or self._standard_output_buf.shape != output_shape
                    or self._standard_output_buf.device != x_int8.device
                    or self._standard_output_buf.dtype != torch.float16):
                self._standard_output_buf = torch.empty(
                    output_shape, device=x_int8.device, dtype=torch.float16
                ).contiguous(memory_format=torch.channels_last)
            deep_ok = (self.out_channels % 8 == 0
                       # Probe the symbol this path actually uses. The old probe named the deepfuse
                       # variant, which is call-site-free on the int8 side.
                       and hasattr(modiff_cutlass, "conv2d_int8_dequant_fp16_tuned"))
            if deep_ok:
                cid = self._ensure_tuned_config(x_int8, output_shape)
                cid = cid if cid is not None else -1
                if self.bias is None and residual is None:
                    # No bias/residual -> deep-fuse writes final fp16 directly (no store pass).
                    if cid >= 0:
                        return modiff_cutlass.conv2d_int8_dequant_fp16_tuned(
                            x_int8, self.weight_int8, self._cached_alpha_tensor,
                            self.weight_scale_channel_half.view(-1), self._standard_output_buf, cid,
                            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                            self.dilation[0], self.dilation[1])
                    return modiff_cutlass.conv2d_int8_fprop_dequant_fp16_prealloc(
                        x_int8, self.weight_int8, self._cached_alpha_tensor,
                        self.weight_scale_channel_half.view(-1), self._standard_output_buf,
                        self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                        self.dilation[0], self.dilation[1])
                # EVT single-pass: acc*alpha*weight_scale[k] + bias[k] + residual[elem] -> fp16,
                # no fp32/fp16 scratch (replaces the deep-fuse scratch + bias_residual_store pass).
                # Beats even the best autotuned deep-fuse cid at b128 (scratch IO dominates the
                # fixed-tile cost); ~fp16-ulp vs the old 2x-rounded path (single-round -> more
                # accurate). weight_scale/bias are read FP32 in the visitor tree.
                if self.bias is not None:
                    if getattr(self, '_evt_bias_f32', None) is None or self._evt_bias_f32.numel() != self.bias.numel():
                        self._evt_bias_f32 = self.bias.view(-1).float().contiguous()
                    bias_arg = self._evt_bias_f32
                else:
                    bias_arg = self._empty_bias
                res_arg = (residual if residual is not None
                           else torch.empty(0, device=x_int8.device, dtype=torch.float16))
                return modiff_cutlass.conv2d_int8_evt_bias_residual_fp16(
                    x_int8, self.weight_int8, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), bias_arg, res_arg,
                    self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
            # Fallback (out_channels % 8 != 0 or kernel unavailable): fp32-temp store paths.
            if residual is not None:
                bias_arg = (self.bias.view(-1).contiguous()
                            if self.bias is not None else self._empty_bias)
                return modiff_cutlass.conv2d_int8_fprop_no_ohat_prealloc_bias_residual(
                    x_int8, self.weight_int8, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), bias_arg, residual,
                    self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
            if self.bias is not None and hasattr(modiff_cutlass, "conv2d_int8_fprop_no_ohat_prealloc_bias"):
                out = modiff_cutlass.conv2d_int8_fprop_no_ohat_prealloc_bias(
                    x_int8, self.weight_int8, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), self.bias.view(-1).contiguous(),
                    self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
                bias_fused = True
            else:
                out = modiff_cutlass.conv2d_int8_fprop_no_ohat_prealloc(
                    x_int8, self.weight_int8, self._cached_alpha_tensor,
                    self.weight_scale_channel.view(-1), self._standard_output_buf,
                    self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                    self.dilation[0], self.dilation[1])
        else:
            out_raw = modiff_cutlass.conv2d_int8_fprop(
                x_int8, self.weight_int8, self._cached_alpha_tensor, self._empty_bias,
                self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1])
            out = out_raw * self.weight_scale_channel
        if self.bias is not None and not bias_fused:
            bias = self.bias.to(out.dtype) if out.dtype != self.bias.dtype else self.bias
            out = out + bias
        # Residual not fused in the epilogue (non-fp16 output / kernel unavailable):
        # add it here so behaviour is identical, just unfused.
        if residual is not None and not residual_fused:
            out = out + (residual.to(out.dtype) if out.dtype != residual.dtype else residual)
        return out

    def forward_from_int8(self, x_int8: torch.Tensor,
                          residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Baseline fast path: the activation is already quantized to int8 (with SiLU
        applied upstream, e.g. by the GN->int8 fusion in fused_resblock.py), so skip
        the per-layer quantize (K1) and go straight to the conv. Only valid when
        calibrated + not modiff_enabled. Optional `residual` (fp16 channels_last) is
        fused into the store epilogue as the ResBlock skip-add."""
        self._sim_guard('forward_from_int8')
        if not x_int8.is_contiguous(memory_format=torch.channels_last):
            x_int8 = x_int8.contiguous(memory_format=torch.channels_last)
        return self._conv_from_int8(x_int8, residual=residual)

    def forward_from_int8_dual(self, x_int8: torch.Tensor, residual: torch.Tensor,
                               requant_scale: torch.Tensor, apply_relu: bool = True):
        """Cross-block-chaining conv3: dequant + bias + fp16 skip-residual + ReLU, and
        emit BOTH the fp16 block output (x_{N+1}, for the next block's identity) AND
        that output requantized to int8 by `requant_scale` (= the next block conv1's
        static_input_scale) -- in one fused store. This folds the block-entry quantize
        (the standalone per-block K1) into this conv3's epilogue. Returns
        (out_fp16, out_int8), both channels_last. Requires the deep-fuse dual kernel,
        out_channels%8==0, calibrated + standard_output_fp16."""
        self._sim_guard('forward_from_int8_dual')
        assert self.standard_output_fp16, "dual store requires standard_output_fp16"
        assert hasattr(modiff_cutlass, "conv2d_int8_fprop_deepfuse_bias_residual_dual"), \
            "dual-store kernel unavailable (rebuild the extension)"
        assert self.out_channels % 8 == 0, "dual store requires out_channels%8==0"
        self._ensure_conv_caches(x_int8.device)
        if not x_int8.is_contiguous(memory_format=torch.channels_last):
            x_int8 = x_int8.contiguous(memory_format=torch.channels_last)
        if not residual.is_contiguous(memory_format=torch.channels_last):
            residual = residual.contiguous(memory_format=torch.channels_last)
        h_out = ((x_int8.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x_int8.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x_int8.shape[0], self.out_channels, h_out, w_out)
        if (self._standard_output_buf is None
                or self._standard_output_buf.shape != output_shape
                or self._standard_output_buf.device != x_int8.device
                or self._standard_output_buf.dtype != torch.float16):
            self._standard_output_buf = torch.empty(
                output_shape, device=x_int8.device, dtype=torch.float16
            ).contiguous(memory_format=torch.channels_last)
        if (self._int8_output_buf is None
                or self._int8_output_buf.shape != output_shape
                or self._int8_output_buf.device != x_int8.device):
            self._int8_output_buf = torch.empty(
                output_shape, device=x_int8.device, dtype=torch.int8
            ).contiguous(memory_format=torch.channels_last)
        bias_arg = (self.bias.view(-1).contiguous() if self.bias is not None else self._empty_bias)
        cid = self._ensure_tuned_config(x_int8, output_shape)
        modiff_cutlass.conv2d_int8_fprop_deepfuse_bias_residual_dual(
            x_int8, self.weight_int8, self._cached_alpha_tensor,
            self.weight_scale_channel_half.view(-1), bias_arg, residual.half(),
            requant_scale.view(1), self._standard_output_buf, self._int8_output_buf,
            apply_relu, cid if cid is not None else -1,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        return self._standard_output_buf, self._int8_output_buf

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize an fp16/fp32 activation to channels_last int8 using this conv's
        calibrated static_input_scale -- the block-entry K1 for int8 chaining. Reuses
        the cast-free fp16 path from _forward_standard (step1 with a zeroed a_hat)."""
        self._ensure_conv_caches(x.device)
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        if x.dtype == torch.float16:
            if (self._zero_ahat_buf is None
                    or self._zero_ahat_buf.shape != x.shape
                    or self._zero_ahat_buf.device != x.device):
                self._zero_ahat_buf = torch.zeros_like(x)
            else:
                self._zero_ahat_buf.zero_()
            return modiff_cutlass.step1_static_quantize_fprop(
                x, self._zero_ahat_buf, self.static_input_scale.view(1), self._empty_smooth)
        x_for_quant = x if x.dtype == torch.float32 else x.float()
        return modiff_cutlass.scale_quantize_int8(x_for_quant, self._cached_scale_tensor)

    def forward_to_int8(self, x_int8: torch.Tensor, apply_relu: bool = True) -> torch.Tensor:
        """INT8-in, INT8-out conv for chaining: dequant + optional ReLU + requantize
        by output_requant_scale (the next conv's input scale), in one fused kernel,
        so the next conv reads int8 directly. Requires output_requant_scale set,
        calibrated, use_cutlass. Returns a channels_last int8 tensor."""
        self._sim_guard('forward_to_int8')
        self._ensure_conv_caches(x_int8.device)
        assert self.output_requant_scale is not None, "output_requant_scale not wired"
        if not x_int8.is_contiguous(memory_format=torch.channels_last):
            x_int8 = x_int8.contiguous(memory_format=torch.channels_last)
        h_out = ((x_int8.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        w_out = ((x_int8.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        output_shape = (x_int8.shape[0], self.out_channels, h_out, w_out)
        if (self._int8_output_buf is None
                or self._int8_output_buf.shape != output_shape
                or self._int8_output_buf.device != x_int8.device):
            self._int8_output_buf = torch.empty(
                output_shape, device=x_int8.device, dtype=torch.int8
            ).contiguous(memory_format=torch.channels_last)
        bias_arg = (self.bias.view(-1).contiguous()
                    if self.bias is not None else self._empty_bias)
        # Deep-fuse path: fold per-channel weight_scale into the CUTLASS GEMM
        # epilogue (fp16, no fp32 temporary), then bias+ReLU+requant->int8. Removes
        # the fp32 intermediate the plain path pays. Requires out_channels%8==0.
        if (self.out_channels % 8 == 0
                and hasattr(modiff_cutlass, "conv2d_int8_fprop_deepfuse_relu_requant_int8")):
            cid = self._ensure_tuned_config(x_int8, tuple(self._int8_output_buf.shape))
            return modiff_cutlass.conv2d_int8_fprop_deepfuse_relu_requant_int8(
                x_int8, self.weight_int8, self._cached_alpha_tensor,
                self.weight_scale_channel_half.view(-1), bias_arg,
                self.output_requant_scale.view(1), self._int8_output_buf, apply_relu,
                cid if cid is not None else -1,
                self.stride[0], self.stride[1], self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1])
        return modiff_cutlass.conv2d_int8_fprop_relu_requant_int8(
            x_int8, self.weight_int8, self._cached_alpha_tensor,
            self.weight_scale_channel.view(-1), bias_arg,
            self.output_requant_scale.view(1), self._int8_output_buf, apply_relu,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])

    def _forward_first_step_imode(self, x: torch.Tensor) -> torch.Tensor:
        """t=T for I-MoDiff: q0 = sat_i8(round(x/s*)), a_hat = q0, o_hat = conv(q0)*s* + bias.

        x is already SmoothQuant-smoothed (caller of _forward_first_step). Same s* as later
        steps (static_delta_scale[0]); the paper warmup is a no-op on this grid so we skip it.
        """
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        d_scale, d_alpha = self._delta_scale_args(x.device)
        q0 = (x.float() * d_scale).round_().clamp_(-127, 127).to(torch.int8)
        q0 = q0.contiguous(memory_format=torch.channels_last)
        qmax = int(self._ahat_qmax())
        a = q0.to(torch.int32).clamp_(-qmax, qmax)
        self.a_hat_cache = a.to(self._imode_dtype()).contiguous(
            memory_format=torch.channels_last)
        self._ensure_ahat_qscale(x.device)

        if self._empty_bias is None or self._empty_bias.device != x.device:
            self._empty_bias = torch.empty(0, device=x.device)
        out_raw = modiff_cutlass.conv2d_int8_fprop(
            q0, self.weight_int8, d_alpha, self._empty_bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1])
        o_hat = out_raw * self.weight_scale_channel
        if self.bias is not None:
            o_hat = o_hat + self.bias
        self._adopt_first_step_caches(None, o_hat, torch.float16)
        return self._module_output()

    def _adopt_first_step_caches(self, a_hat, o_hat, cache_dtype):
        """Write t=T a_hat/o_hat into existing cache buffers when shapes match.

        CUDA-graph capture of the first phase pre-allocates these buffers (warmup
        or `_ensure_state_buffers`). Replacing them with a new `.to().contiguous()`
        tensor is an illegal allocation during capture and also breaks the
        modulated-phase graph, which captured the old pointers. Eager first
        sample still assigns when no buffer exists.
        """
        if a_hat is not None:
            ah = getattr(self, "a_hat_cache", None)
            if (ah is not None and ah.shape == a_hat.shape and ah.device == a_hat.device
                    and ah.dtype == cache_dtype):
                ah.copy_(a_hat)
            else:
                self.a_hat_cache = a_hat.to(cache_dtype).contiguous(
                    memory_format=torch.channels_last)
        if o_hat is not None:
            oh = getattr(self, "o_hat_cache", None)
            if (oh is not None and oh.shape == o_hat.shape and oh.device == o_hat.device
                    and oh.dtype == cache_dtype):
                oh.copy_(o_hat)
            else:
                self.o_hat_cache = o_hat.to(cache_dtype).contiguous(
                    memory_format=torch.channels_last)

    def _forward_first_step(self, x: torch.Tensor) -> torch.Tensor:
        """First timestep (t=T): warm-up with repeated quantisation.

        Adopts a_hat/o_hat via `_adopt_first_step_caches` (copy into a resident
        buffer when CUDA-graph warmup already allocated one; otherwise assign).
        Scratch buffers `_ensure_state_buffers` sets up (_residual_buf,
        _scale_buf, ...) aren't used here — they're allocated by
        `_forward_modulated`'s own `_ensure_state_buffers` call on step 2.

        Unlike _forward_modulated's hot path, this one still needs fp32 x:
        the tensor-scale branch of _int8_conv calls the vectorized
        scale_quantize_int8 kernel, which reinterpret_casts its input as
        float4 and would read garbage (wrong stride, half the needed bytes)
        given fp16 memory. This only runs once per layer per sample() call
        (t=T warm-up), so the cast's cost is negligible next to the N-1
        _forward_modulated calls that now skip it.
        """
        if x.dtype != torch.float32:
            x = x.float()
        if self._imode() and self.is_calibrated:
            return self._forward_first_step_imode(x)
        if self.is_calibrated:
            input_scale = self.static_input_scale
            if input_scale.device != x.device:
                input_scale = input_scale.to(x.device)
        elif self.calibrating:
            input_scale = self._compute_activation_scale(x)
        else:
            input_scale = self._compute_scale_tensor(x)

        a_hat = self._dequantize_activation(x, input_scale)
        o_hat = self._int8_conv(x, input_scale, with_bias=True)

        for _ in range(self.warmup_steps - 1):
            residual = x - a_hat
            if self.is_calibrated:
                # A DYNAMIC scale, not the static activation grid. The warm-up converges only if the
                # grid shrinks with the residual: after round 1 the residual is under half an LSB of
                # the full-activation grid, so quantizing it on that same grid rounds to zero and the
                # loop does nothing. Measured on real activations (probe_warmup.py): with the static
                # scale |a_hat - x|/|x| is bit-identical across all 5 rounds -- 0.0197 at A8, 0.4006
                # at A4 -- while a per-round absmax contracts to 0.00008 and 0.00001 respectively.
                # This is the paper's Appendix D.5 warm-up; the static version was not.
                r_scale = self._compute_scale_tensor(residual)
            elif self.calibrating:
                r_scale = self._compute_activation_scale(residual, is_residual=True)
            else:
                r_scale = self._compute_scale_tensor(residual)
            conv_r = self._int8_conv(residual, r_scale, with_bias=False)
            r_dq = self._dequantize_activation(residual, r_scale)
            a_hat = a_hat + r_dq
            o_hat = o_hat + conv_r

        cache_dtype = torch.float16 if self.is_calibrated else torch.float32
        self._adopt_first_step_caches(a_hat, o_hat, cache_dtype)
        if self._ahat_want_int8():
            self._pack_ahat_int8()
        return self._after_ahat_write(self._module_output())

    def _forward_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """MoDiff modulated step (t<T). No periodic reset per paper.
        Uses fused sub+absmax+scale kernel and device pointer alpha to minimize
        kernel launches and avoid CPU-GPU synchronization.
        SmoothQuant multiply is fused into sub_absmax_scale when applicable.
        """
        self.step_count += 1

        if self.a_hat_cache is None or self.a_hat_cache.shape != x.shape:
            self.is_first_step = True
            if not self._smooth_is_identity:
                x = x * self._smooth_inv
            out = self._forward_first_step(x)
            self.is_first_step = False
            return out

        self._ensure_state_buffers(x)

        # Observe the activation range on the MODULATED steps too, not just at t=T.
        #
        # _compute_activation_scale had three call sites and none of them covered t < T in MoDiff
        # mode: :684 sits in _forward_standard (unreachable when modiff_enabled), :993 is
        # _forward_first_step (t=T only), and the warm-up residuals pass is_residual=True, which the
        # accumulator deliberately skips. The modulated steps took the dynamic branch below, which
        # derives its scale on-device via sub_absmax_scale and never reports it back. So
        # static_input_scale described the FIRST diffusion step and nothing else.
        #
        # Measured 2026-08-03 over 20 steps, calibrated absmax vs what actually runs:
        #   in_conv  0.244    -> 5.61   (23x out of range)
        #   out_conv 0.000446 -> 18.1   (41000x out of range)
        # out_conv is worst because at t=T the ResBlock's internal activation has barely developed,
        # so it locks in an absurdly small range for every later step. Downstream, every quantizer
        # then clips -- which is also why MoDiff's own delta calibration could not converge on those
        # layers: its denominator was this scale.
        #
        # This is the documented purpose of _compute_activation_scale ("used during calibration and
        # first-step only (slow path with .item() sync)"), so the per-step host sync is expected and
        # calibration-scoped. Only the plain _forward_modulated needs it: the three fused variants
        # all gate on _can_fuse_input_silu, which requires is_calibrated, so during calibration this
        # is the only modulated path that runs.
        if self.calibrating:
            self._compute_activation_scale(x)

        if self.is_calibrated and HAS_CUTLASS and self.use_cutlass:
            # No SiLU here: step1_static_quantize_fprop below quantizes x itself.
            d_scale, d_alpha = self._delta_scale_args(x.device, x, fused_silu=False)

            if not hasattr(self, '_smooth_inv_flat') or self._smooth_inv_flat.device != x.device:
                if not self._smooth_is_identity:
                    self._smooth_inv_flat = self._smooth_inv.view(-1).contiguous()
                else:
                    self._smooth_inv_flat = torch.empty(0, device=x.device, dtype=torch.float32)

            p_step1 = profiler.start("MoDiff INT8 Static Step1")
            x_int8 = modiff_cutlass.step1_static_quantize_fprop(
                x,
                self.a_hat_cache,
                d_scale,
                self._smooth_inv_flat,
                self._delta_a4,
                self._write_ahat_now(),
                self._ahat_scale_arg(),
            )
            profiler.stop("MoDiff INT8 Static Step1", p_step1)
            if self._delta_calib:
                self._observe_delta_codes(x_int8)

            p_conv = profiler.start("MoDiff INT8 Static Conv2d")
            out = self._evt_ohat(x_int8, d_alpha)
            profiler.stop("MoDiff INT8 Static Conv2d", p_conv)
            return self._after_ahat_write(out)

        # Kernel 1 Fused C++ Backend Call:
        # Fuses sub_absmax_scale, dequant_accumulate, and scale_quantize into 1 python launch.
        p_step1 = profiler.start("MoDiff INT8 Fused Step1")
        # Q_level, not a literal 127: this path is the uncalibrated / non-GN-fusable sibling of
        # _delta_gn_dynamic_args, and it hardcoded 127 while those honoured the knobs. That made
        # MODIFF_ACT_BITS silently partial -- whichever conv layers fell
        # through to the plain modulated path kept an 8-bit delta grid while the rest changed.
        step1 = (modiff_cutlass.step1_quantize_no_ahat_fprop if self._skip_cache_store()
                 else modiff_cutlass.step1_quantize_fprop)
        x_int8 = step1(
            x, self.a_hat_cache, self._residual_buf,
            self._absmax_buf, self._scale_buf, self._inv_scale_buf,
            self._retire_count, self.act_q, self._smooth_inv_flat
        )
        profiler.stop("MoDiff INT8 Fused Step1", p_step1)

        p_conv = profiler.start("MoDiff INT8 Fused Conv2d")
        out = self._evt_ohat(x_int8, self._inv_scale_buf.view(1))
        profiler.stop("MoDiff INT8 Fused Conv2d", p_conv)
        return self._after_ahat_write(out)

    # ==================================================================
    # MoDiff controls
    # ==================================================================

    def enable_modiff(self, enabled: bool = True):
        self.modiff_enabled = enabled
        if not enabled:
            self.reset_state()

    def set_standard_output_fp16(self, enabled: bool = True):
        self.standard_output_fp16 = enabled
        if not enabled:
            self._standard_output_buf = None

    def reset_state(self):
        self.is_first_step = True
        with torch.inference_mode():
            if self.a_hat_cache is not None:
                self.a_hat_cache.zero_()
            if self.o_hat_cache is not None:
                self.o_hat_cache.zero_()
        self.step_count = 0

    # ==================================================================
    # Calibration + SmoothQuant
    # ==================================================================

    def begin_calibration(self):
        self.calibrating = True
        self.is_calibrated = False
        self._scale_sum = 0.0
        self._scale_count = 0
        self._act_channel_max = None

    def end_calibration(self):
        self.calibrating = False
        if self._scale_count == 0:
            return

        if self._act_channel_max is not None and self._orig_weight is not None:
            self._apply_smoothquant()

        if self._act_channel_max is not None:
            s = self.smooth_scale.view(-1)
            smoothed_ch_max = self._act_channel_max / s
            smoothed_global_max = smoothed_ch_max.max().item()
            static_scale = 127.0 / max(smoothed_global_max, 1e-6)
        else:
            static_scale = self._scale_sum / self._scale_count

        # Same Q_b/127 rescale set_static_scale applies, so a live-calibrated model and one restored
        # from a calibration file agree on the activation precision. (This method fills the buffer
        # itself rather than delegating, because it also has the smooth-scale bookkeeping below.)
        static_scale = float(static_scale) * (self.act_q / 127.0)
        if not math.isfinite(static_scale) or static_scale <= 0:
            # A NaN/Inf scale poisons every later step. Observed on SD1.5 C=320
            # in_conv during SmoothQuant. Leave the layer on the dynamic path.
            self.is_calibrated = False
            self._cached_scale_float = None
            return
        self.static_input_scale.fill_(float(static_scale))
        self.is_calibrated = True
        self._cached_scale_float = float(static_scale)
        alpha = 1.0 / float(static_scale)
        self._cached_alpha_tensor = torch.tensor(
            [alpha], device=self.static_input_scale.device, dtype=torch.float32
        )
        self._smooth_inv.copy_(1.0 / self.smooth_scale)
        self._smooth_is_identity = bool(torch.allclose(
            self._smooth_inv,
            torch.ones_like(self._smooth_inv),
            atol=1e-6
        ))
        # MODIFF_ACT_SIM_EXACT_W=1 keeps the original fp16 weight alive so the
        # MODIFF_ACT_BLOCK sim can run an EXACT-weight conv, which is what isolates the
        # activation quantizer from the W8 weight quantizer in an error budget. Off by
        # default because this buffer is otherwise pure waste for the life of the model
        # (see the note at its registration).
        if not (os.environ.get("MODIFF_ACT_SIM_EXACT_W") == "1"
                or os.environ.get("MODIFF_ACT_SIM_WBITS")
                or os.environ.get("MODIFF_ACT_SIM_WBLOCK")):
            self._orig_weight = None

    # ------------------------------------------------------------------
    # MoDiff delta calibration (two self-consistent rounds)
    # ------------------------------------------------------------------

    def begin_delta_calibration(self, reset: bool = False):
        """Arm delta-range observation, and seed the table at a provably non-clipping scale.

        Observation happens at `act_scale / OBS_DIV`. Since |a_t - a_hat_{t+1}| <= 2*act_absmax, a
        divisor of 2 already guarantees no clipping; 4 leaves headroom for the fp16 rounding in
        a_hat. That makes ONE pass exact, so `reset` is vestigial and no search is needed -- see
        end_delta_calibration for what the previous iterative version did wrong.
        """
        self._delta_calib = True
        self._delta_code_max = None
        OBS_DIV = 4.0
        obs = float(self.static_input_scale.item()) / OBS_DIV
        self.static_delta_scale.fill_(obs)
        self.static_delta_alpha.fill_(1.0 / obs)
        self.is_delta_calibrated.fill_(True)
        self._delta_cal = True

    def end_delta_calibration(self, safety: float = 1.02, smooth: bool = True) -> bool:
        """Turn the observed code maxima into the per-step scale table. Returns True if set.

        `delta_absmax[i] = code_max[i] / scale_used[i]`, then `scale[i] = Q / (safety*absmax[i])`.
        Steps never reached keep the previous entry (forward-fill), so a short calibration run still
        yields a usable table for a longer production run -- the delta range is flat in the tail.
        """
        self._delta_calib = False
        if self._delta_code_max is None or not self.is_calibrated:
            return False
        code_max = self._delta_code_max.detach().to("cpu", torch.float64)
        # Which scale produced each observation: the table if it was already calibrated
        # (round >= 1), else the flat activation scale (round 0).
        if bool(self.is_delta_calibrated):
            used = self.static_delta_scale.detach().to("cpu", torch.float64)
        else:
            used = torch.full_like(code_max, float(self.static_input_scale.item()))

        seen = code_max > 0
        if not bool(seen.any()):
            return False
        self._delta_obs_code_max = float(code_max[seen].max())
        self._delta_obs_clipped_frac = float((code_max[seen] >= 127.0).to(torch.float64).mean())
        absmax = torch.zeros_like(code_max)
        absmax[seen] = code_max[seen] / used[seen].clamp_min(1e-12)

        # SINGLE-SHOT, no search. |delta| = |a_t - a_hat_{t+1}| <= 2*act_absmax always, because both
        # terms lie inside the activation range. So observing at scale = act_scale/2 provably cannot
        # clip, and code_max/used is then the delta's true range in one exact pass.
        #
        # This replaces an iterative scheme (geometric backoff on clip + monotone running max) that
        # was actively harmful: both operations only ever shrink the scale, so extra rounds could only
        # make the quantizer COARSER. Measured on the real checkpoint, the median step gain decayed
        # 1.5x -> 1.0x -> 0.5x -> 0.4x across 8 rounds -- i.e. it ratcheted itself below the
        # activation grid it was supposed to improve on -- and latent error got 14.6% worse than
        # leaving the table off. A monotone max cannot converge on a fixed point that it moves.
        if bool((code_max[seen] >= 127.0).any()):
            # Should not happen at act_scale/2; if it does the observation scale was wrong, so say so
            # rather than silently returning a lower bound.
            print(f"⚠ {self.layer_name or type(self).__name__}: delta clipped during observation "
                  f"({self._delta_obs_clipped_frac:.0%} of steps) -- the observation scale was too "
                  f"coarse, so this table is a lower bound. Observe at act_scale/2.")

        # Forward-fill unobserved steps, then back-fill the head if step 0 was never seen.
        last = 0.0
        for i in range(absmax.numel()):
            if seen[i]:
                last = float(absmax[i])
            elif last > 0.0:
                absmax[i] = last
        first = next((float(absmax[i]) for i in range(absmax.numel()) if absmax[i] > 0), 0.0)
        absmax[absmax <= 0] = first
        if first <= 0.0:
            return False

        if smooth:
            # A single noisy batch must not shrink a neighbour's scale: take a running max over a
            # 3-wide window. Monotone-safe and cheap; the table is 256 entries on CPU.
            a = absmax.clone()
            for i in range(absmax.numel()):
                lo, hi = max(0, i - 1), min(absmax.numel(), i + 2)
                absmax[i] = float(a[lo:hi].max())

        q = 127.0
        scale = q / (absmax * safety).clamp_min(1e-12)
        self.static_delta_scale.copy_(scale.to(torch.float32))
        self.static_delta_alpha.copy_((1.0 / scale).to(torch.float32))
        self.is_delta_calibrated.fill_(True)
        self._delta_cal = True
        self._delta_code_max = None
        return True

    def delta_calibration_report(self) -> Dict[str, object]:
        """Diagnostics for the acceptance test: how much of the code range is actually used."""
        if not bool(self.is_delta_calibrated):
            return {"layer": self.layer_name, "calibrated": False}
        s = self.static_delta_scale.detach().to("cpu", torch.float64)
        act = float(self.static_input_scale.item())
        return {"layer": self.layer_name, "calibrated": True,
                "activation_scale": act,
                "obs_code_max": getattr(self, "_delta_obs_code_max", None),
                "obs_clipped_frac": getattr(self, "_delta_obs_clipped_frac", None),
                "delta_scale_step0": float(s[0]), "delta_scale_step1": float(s[1]),
                "delta_scale_tail": float(s[-1]),
                # How much finer the delta grid is than the activation grid. This is the number the
                # paper's Theorem 4.3 turns into an error reduction: error ~ s^2, so a gain of g
                # means the squared error falls by g^2.
                "step_gain_tail": float(s[-1]) / act if act > 0 else None}

    def _apply_smoothquant(self):
        """SmoothQuant: derive the per-in-channel smooth scale from the calibrated
        activation range and fold it into the weights."""
        act_max = self._act_channel_max
        w = self._orig_weight
        K = self.out_channels

        w_dev = w.to(act_max.device)
        w_by_cin = w_dev.reshape(K, self.in_channels, -1)
        w_max = w_by_cin.abs().amax(dim=(0, 2))

        ratio = act_max / torch.clamp(w_max, min=1e-8)
        s = ratio.sqrt().clamp(min=1e-4, max=1e4)

        # SmoothQuant is meaningless where the weight range is zero: it exists to move dynamic range
        # from the activations into the weights, and a zero weight has none to receive. Left alone,
        # `ratio = act_max/1e-8` sends `s` straight to its 1e4 ceiling. LDM wraps every ResBlock
        # output conv in zero_module, so this is 35 of the 70 quantized convs here.
        #
        # HYGIENE, NOT NUMERICS -- be clear about what this does and does not buy. A uniform `s`
        # cancels exactly: the kernel quantizes x/s and the scale is Q*s/act_max, so the emitted code
        # is Q*x/act_max either way. Measured: utilisation is bit-identical before and after this
        # change (521.7 both). What it fixes is that `static_input_scale` was ~1e4x its meaningful
        # value (2.4e5 vs 23.85) and `smooth_scale` sat pinned at a clamp ceiling, which makes every
        # diagnostic on those fields unreadable and makes the layer's behaviour depend on a clamp
        # bound rather than on anything measured. It is not the cause of those layers' clipping;
        # that is genuine under-observation of the activation range (see FINDINGS 2026-08-03).
        dead = w_max <= 1e-12
        if bool(dead.any()):
            s = torch.where(dead, torch.ones_like(s), s)

        self._fold_weights_with_smooth(s)

    def _fold_weights_with_smooth(self, s: torch.Tensor):
        """Fold a given per-in-channel SmoothQuant scale `s` ([C_in]) into the weights:
        set smooth_scale, then requantize the *original* fp weights against their smoothed
        per-output-channel range. Shared by _apply_smoothquant (live calibration, `s` from
        activation stats) and set_static_calibration (`s` restored from a checkpoint).
        Requires _orig_weight to still be present."""
        w = self._orig_weight
        K = self.out_channels

        w_dev = w.to(s.device)
        self.smooth_scale.copy_(s.view(1, -1, 1, 1))

        w_smoothed = w_dev * s.view(1, -1, 1, 1)
        w_flat = w_smoothed.reshape(K, -1)
        ch_max = w_flat.abs().max(dim=1).values
        ch_scale = torch.clamp(ch_max / 127.0, min=1e-8)

        self.weight_scale_channel.copy_(ch_scale.view(1, K, 1, 1))
        self.weight_scale_channel_half.copy_(ch_scale.half().to(self.weight_scale_channel_half.device))

        w_quant = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        w_quant = w_quant.reshape(K, self.in_channels, *self.kernel_size)
        w_nhwc = w_quant.permute(0, 2, 3, 1).contiguous()
        self.weight_int8.data = w_nhwc.to(self.weight_int8.device)

    def set_calibrating(self, calibrating: bool):
        if calibrating:
            self.begin_calibration()
        else:
            self.end_calibration()

    def set_static_scale(self, scale: float):
        # The one funnel for the calibrated activation grid: end_calibration, the checkpoint
        # restore path and set_static_calibration all land here, and every consumer (the cached
        # alpha/scale tensors, _forward_standard's static quantize, _forward_first_step) derives
        # from it. So MODIFF_ACT_Q is applied here, once, at load time: a calibrated scale means
        # Q/range, and dropping to b bits is exactly rescaling it by Q_b/127. No recalibration and
        # no hot-path cost. Q=127 leaves the value bit-identical to the shipped default.
        scale = float(scale) * (self.act_q / 127.0)
        self.static_input_scale.fill_(float(scale))
        self.is_calibrated = True
        self._cached_scale_float = float(scale)
        alpha = 1.0 / float(scale)
        self._cached_alpha_tensor = torch.tensor(
            [alpha], device=self.static_input_scale.device, dtype=torch.float32
        )
        self._cached_scale_tensor = torch.tensor(
            [float(scale)], device=self.static_input_scale.device, dtype=torch.float32
        )

    def set_static_calibration(self, scale: float, smooth_scale: Optional[torch.Tensor] = None):
        """Restore a full static calibration (per-tensor `scale` + optional per-in-channel
        `smooth_scale`) from an exported checkpoint. The smooth scale is re-folded into the
        freshly converted (unsmoothed) weights so smoothed activations meet smoothed
        weights at inference, keeping the SmoothQuant-derived static scale correct. See
        OptimizedInt4Conv2d.set_static_calibration for the full rationale; int8's 8-bit
        range hides the mismatch that the int4 path suffers, but the treatment is mirrored
        for consistency. Falls back to scale-only if the weights can't be re-folded
        (_orig_weight already released)."""
        if smooth_scale is not None and self._orig_weight is not None:
            s = torch.as_tensor(smooth_scale, dtype=torch.float32,
                                device=self.smooth_scale.device).reshape(-1)
            self._fold_weights_with_smooth(s)
            self._smooth_inv.copy_(1.0 / self.smooth_scale)
            self._smooth_is_identity = bool(torch.allclose(
                self._smooth_inv, torch.ones_like(self._smooth_inv), atol=1e-6))
            if hasattr(self, '_smooth_inv_flat'):
                del self._smooth_inv_flat
        self.set_static_scale(scale)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

def convert_model_to_optimized_int8(model: nn.Module, prefix: str = "", use_compile: bool = False,
                                     skip_pointwise: bool = True,
                                     _memo: dict = None) -> nn.Module:
    """Wrap every eligible nn.Conv2d in OptimizedInt8Conv2d.

    DEDUPLICATED BY OBJECT IDENTITY (`_memo`), added 2026-08-13 to fix a 1014.6 MiB leak.

    FusedResBlock aliases one conv under two attributes -- `fused.in_conv` IS
    `fused.original.in_layers[-1]` (fused_resblock.py:756), and likewise for out_conv. This walk
    recurses over named_children(), so it reached the SAME nn.Conv2d down two paths and wrapped it
    TWICE, into two independent modules each holding its own packed int4 weights. Only the one
    `forward` uses was ever called or calibrated; the other 70 sat inert with modiff_enabled=True.
    Measured: 70 live + 70 orphans, 1014.6 MiB of duplicated weights out of 2762 MiB allocated -- 37%
    of the model's memory, for modules that never ran.

    The memo makes the second path reuse the first wrapper, so one object is referenced from both.

    THE NAME MATTERS AND IS NOT COSMETIC. apply_int{{4,8}}_static_scales matches on
    `module.layer_name`, and the calibration files key on the NON-`.original.` path, which is why 70
    of 140 wrappers loaded a scale before this fix. named_children() registers `original` before
    `in_conv`, so the first wrapper created carries the `.original.` name -- keeping it would have
    silently dropped calibration to 0 layers while looking like a pure memory win. The memo therefore
    upgrades layer_name to the non-`.original.` path when it sees it.
    """
    if _memo is None:
        _memo = {}
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and not isinstance(child, OptimizedInt8Conv2d):
            if child.in_channels < 32:
                continue
            is_skip = 'skip' in name
            is_final_out = full_name.startswith('out.')
            is_pointwise = child.kernel_size == (1, 1)
            is_grouped = child.groups != 1
            quant_skip_out = os.environ.get("MODIFF_QUANT_SKIP_OUT", "0") == "1"

            if is_grouped:
                continue
            if (is_skip or is_final_out) and not quant_skip_out:
                continue
            if is_pointwise and skip_pointwise and not quant_skip_out:
                continue

            hit = _memo.get(id(child))
            if hit is not None:
                # Same underlying conv, reached by a second alias. Reuse the one wrapper, and prefer
                # the name the calibration files key on (see the docstring).
                if ".original." in (getattr(hit, "layer_name", "") or "") \
                        and ".original." not in full_name:
                    hit.layer_name = full_name
                setattr(model, name, hit)
                continue
            optimized_conv = OptimizedInt8Conv2d(child, layer_name=full_name, use_compile=use_compile)
            _memo[id(child)] = optimized_conv
            target_device = child.weight.device
            if target_device.type != 'cpu':
                optimized_conv = optimized_conv.to(target_device)
            setattr(model, name, optimized_conv)
        else:
            convert_model_to_optimized_int8(child, prefix=full_name, use_compile=use_compile,
                                             skip_pointwise=skip_pointwise,
                                             _memo=_memo)

    # Convert to channels_last for PyTorch perf, then restore weight_int8.
    # Only at the top-level call: this function recurses, and re-running the
    # whole-subtree conversion at every nesting level would re-scramble and
    # re-fix every already-fixed descendant's weight_int8 once per level of
    # nesting instead of once overall.
    if not prefix:
        model = model.to(memory_format=torch.channels_last)
        for m in model.modules():
            if isinstance(m, OptimizedInt8Conv2d):
                m.weight_int8.data = m.weight_int8.data.contiguous()
    return model


# ---------------------------------------------------------------------------
# Global calibration helpers
# ---------------------------------------------------------------------------

class CalibrationConfig:
    def __init__(self):
        self.is_calibrated = False
        self.scales = {}

    def update(self, layer_name: str, scale: float):
        self.scales[layer_name] = float(scale)

    def load(self, path):
        self.scales = torch.load(path, weights_only=True)
        self.is_calibrated = True

    def save(self, path):
        torch.save(self.scales, path)

    def finalize(self):
        self.is_calibrated = True


_calib_config = CalibrationConfig()


def get_calibration_config():
    return _calib_config


def reset_calibration():
    _calib_config.scales = {}
    _calib_config.is_calibrated = False


def enable_modiff_mode(model, enabled=True):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.enable_modiff(enabled)


def reset_modiff_state(model):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.reset_state()
    manager = getattr(model, '_cuda_graph_manager', None)
    if manager is not None:
        manager.reset_sequence()


def set_standard_output_fp16(model, enabled: bool = True):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.set_standard_output_fp16(enabled)


def set_calibrating(model, calibrating):
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d):
            module.set_calibrating(calibrating)
            if not calibrating and module.is_calibrated:
                _calib_config.update(module.layer_name, float(module.static_input_scale.item()))
    if not calibrating:
        _calib_config.finalize()


def begin_delta_calibration_int8(model: nn.Module, reset: bool = False) -> int:
    """Arm delta-range observation on every calibrated int8 conv. Returns how many."""
    n = 0
    for m in model.modules():
        if isinstance(m, OptimizedInt8Conv2d) and m.is_calibrated:
            m.begin_delta_calibration(reset=reset)
            n += 1
    return n


def end_delta_calibration_int8(model: nn.Module, safety: float = 1.02) -> int:
    """Convert observations into the per-step delta-scale table. Returns how many were set."""
    n = 0
    for m in model.modules():
        if isinstance(m, OptimizedInt8Conv2d):
            if m.end_delta_calibration(safety=safety):
                n += 1
    return n


def export_int8_delta_scales(model: nn.Module) -> Dict[str, object]:
    """Export the MoDiff per-step delta-scale table, keyed by layer name.

    Kept in a SEPARATE artifact from export_int8_static_scales: the two describe different
    quantities (activation range vs temporal-delta range) and are valid for different modes.
    Sharing one file, as the tree used to, is how int8_baseline and int8 ended up reading a
    calibration whose semantics differed from what they needed.
    """
    out = {}
    for m in model.modules():
        if isinstance(m, OptimizedInt8Conv2d) and bool(m.is_delta_calibrated):
            out[m.layer_name] = m.static_delta_scale.detach().to("cpu", torch.float32).clone()
    return out


def apply_int8_delta_scales(model: nn.Module, table: Dict[str, object]) -> int:
    """Load a table produced by export_int8_delta_scales. Returns how many layers were filled."""
    if not table:
        return 0
    loaded = 0
    for m in model.modules():
        if not isinstance(m, OptimizedInt8Conv2d) or m.layer_name not in table:
            continue
        s = table[m.layer_name].to(m.static_delta_scale.device, torch.float32)
        n = min(s.numel(), m.static_delta_scale.numel())
        m.static_delta_scale[:n].copy_(s[:n])
        if n < m.static_delta_scale.numel():          # forward-fill a shorter saved table
            m.static_delta_scale[n:].fill_(float(s[n - 1]))
        m.static_delta_alpha.copy_(1.0 / m.static_delta_scale.clamp_min(1e-12))
        m.is_delta_calibrated.fill_(True)
        m._delta_cal = True
        loaded += 1
    return loaded


def delta_calibration_report_int8(model: nn.Module):
    return [m.delta_calibration_report() for m in model.modules()
            if isinstance(m, OptimizedInt8Conv2d) and m.is_calibrated]


def export_int8_static_scales(model: nn.Module) -> Dict[str, object]:
    """Export the static calibration per int8 conv. Identity-SmoothQuant layers export as
    a bare float (legacy format); SmoothQuant layers export as
    ``{"static_scale": float, "smooth_scale": cpu fp32 tensor [C_in]}`` so smoothing can be
    restored on apply. Mirrors export_int4_static_scales (int8's wider range hides the
    quality loss the int4 path suffers, but the round-trip is kept faithful)."""
    scales = {}
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d) and module.is_calibrated:
            if module._smooth_is_identity:
                scales[module.layer_name] = float(module.static_input_scale.item())
            else:
                scales[module.layer_name] = {
                    "static_scale": float(module.static_input_scale.item()),
                    "smooth_scale": module.smooth_scale.detach().to("cpu", torch.float32).reshape(-1).clone(),
                }
    return scales


def apply_static_scales(model, *args, **kwargs):
    """Load static calibration produced by export_int8_static_scales. Accepts both the
    legacy flat ``{name: float}`` format and the richer ``{name: {"static_scale":...,
    "smooth_scale":...}}`` format (mixed is fine)."""
    scales = kwargs.get('scales', None)
    if scales is None and len(args) > 0 and isinstance(args[0], dict):
        scales = args[0]
    if scales is None:
        return 0

    loaded = 0
    for module in model.modules():
        if isinstance(module, OptimizedInt8Conv2d) and module.layer_name in scales:
            entry = scales[module.layer_name]
            if isinstance(entry, dict):
                module.set_static_calibration(entry["static_scale"], entry.get("smooth_scale"))
            else:
                module.set_static_scale(float(entry))
            loaded += 1
    # Keep _calib_config a plain {name: float} map (its historical contract): downstream
    # consumers (benchmark_ldm.py) treat these values as scalars.
    _calib_config.scales = {
        k: (v["static_scale"] if isinstance(v, dict) else v) for k, v in scales.items()
    }
    _calib_config.is_calibrated = True
    return loaded

