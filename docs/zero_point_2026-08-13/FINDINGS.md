# Activation zero point: the mechanism is built and proven; the answer needs 4 more kernels

**Fix #2 is still unanswered, and this file exists so the next attempt starts from what is true.** The
zero point's arithmetic, module state, bias fold, kernel and routing are implemented and each gated on
its own. What is missing is coverage: **1 of ~6 quantize entry points honours `z`**, so an end-to-end
measurement is contaminated on both arms and no accuracy claim is made here.

## What is built and independently verified

| piece | gate |
|---|---|
`weight_sum_q`, `static_input_zp`, `_orig_bias`, `_refold_zp_bias` | [`test_int4_zero_point.py`](../../integration/tests/test_int4_zero_point.py): folded bias **exactly** equals an fp64 reference rounded to the storage dtype; z=0 **bit-identical** to the symmetric path (bias *and* conv output); idempotent under re-calibration; bias-free conv gains one and gives it back |
`group_norm_silu_quantize_pack_nhwc{,_fast}_zp` applying `+ z` before the clamp | z=0 **bit-identical** to the symmetric kernel on C=192/768/1536; z≠0 differs |
`apply_int4_static_scales` reading `entry["zero_point"]` | every existing bare-float and dict file behaves exactly as before (default 0.0) |
Python routing to the `_zp` entry | refuses rather than guessing when a non-zero `z` meets a path that cannot honour it |

The arithmetic: `a_q = clamp(round(a·s) + z, −7, 7)`, so `a = (a_q − z)/s` and
`Σ w_q·a = (Σ w_q·a_q − z·Σ w_q)·ws/s`. The second term is **constant per output channel**, so it folds
into the bias at calibration time and neither the GEMM nor the EVT epilogue ever sees `z`.

## Why there is no answer yet: the coverage census

With a real asymmetric table loaded, these layers are quantized by entry points that **ignore `z`**
while their bias already carries the `−z·Σw_q` correction — symmetric codes against a corrected bias,
which is worse than either choice alone:

| arm | pairs ignoring z | via |
|---|--:|---|
| W4A4 PTQ | **8** | `group_norm_silu_quantize_resize_nhwc` / `upsample2x_quantize_pack_noahat_fprop` (updown blocks) |
| W4A4 MoDiff | **70** | 62 × `step1_static_quantize_pack_int4_fprop` (the t=T activation grid), 8 × resize/upsample |

So both arms are invalid, and the measured numbers (PTQ 0.5267 → 0.8539, MoDiff 0.3118 → 7.3057) are
**artifacts of incomplete coverage, not results.** They are recorded in `data/zp_measured.json` only as
evidence of the contamination.

**MoDiff needs exactly one more kernel to be testable**: `step1_static_quantize_pack_int4_fprop{,_silu}`.
Its delta steps are correctly `z`-free — the delta is a difference of activations and `a_hat` is cached
*dequantized*, so `z` cancels; the earlier finding that symmetric slightly beat asymmetric on the delta
(residual zero-mean, zp median 7.50 of 0..15) agrees. The bias correction enters `o_hat` exactly once,
at t=T, which is where the activation grid is actually read.

## The guard, and why it is strict by default

`MODIFF_ZP_STRICT` defaults to **1**: a conv whose zero point would be ignored **raises**, naming the
layer and the entry point. Set `MODIFF_ZP_STRICT=0` only to *collect* the census above.

This exists because the alternative is what actually happened: a partial build quantized symmetrically
against corrected biases and produced relL2 7–22, which a script then reported as *"fix #2 is answered
NEGATIVELY."*

## Four mistakes, and what each cost

**1. I nearly published a bug as a finding.** The first run printed a confident, quotable conclusion
from a table describing the wrong tensor. What stopped it was magnitude: relL2 **15.8** is divergence,
not a suboptimal 4-bit grid — a bad grid costs tens of percent. The verdict logic now calls anything
past 3× "bug magnitudes, not a result" and names the likely cause.

**2. The diagnostic was on screen and I read past it.** That run reported
`median |max|/|min| = 0.79×` for a quantity documented at **19.91×** in the same file's own docstring —
a factor of 25, printed before any measurement. The collection now refuses to continue below 5×.

**3. I reinvented a collection that already worked.** I built the **int4** model and hooked
`OptimizedInt4Conv2d`, which looks *more* faithful and is wrong: on the fused path the conv is entered
via `forward_from_int4(packed, …)`, a direct method call, so a `forward_pre_hook` never observes
`silu(gn(x))`. `probe_int4_code_use.py` got 19.91× because it hooks the **fp16** model, where
`self.in_conv(h)` is an ordinary call.

**4. "0 hits" was incomplete coverage, not a clean bill of health.** With four of the ~five unpatched
sites guarded, PTQ reported **0** contaminated layers and I was one step from concluding its +62% was a
real result. Completing the coverage showed **8**. Absence of evidence from an instrument with known
gaps is not evidence.

## What the next attempt should do

1. Teach `step1_static_quantize_pack_int4_fprop{,_silu}` the zero point → **MoDiff becomes testable**.
2. Teach `group_norm_silu_quantize_resize_nhwc` and `upsample2x_quantize_pack_noahat_fprop` → PTQ too.
3. Re-run `scripts/export_and_measure_zp.py`. Its guards now refuse rather than conclude when the
   instrument is wrong, and the prediction is recorded in its docstring **before** the run: this should
   help PTQ and barely move MoDiff, because MoDiff reads the activation grid essentially only at t=T.

Note that each header change forces a full 27-source rebuild (~30 min), so batch all of it into one.

## Reproducing

```bash
python integration/tests/test_int4_zero_point.py                       # the mechanism, no GPU kernels
MODIFF_ZP_STRICT=0 python docs/zero_point_2026-08-13/scripts/export_and_measure_zp.py
```
