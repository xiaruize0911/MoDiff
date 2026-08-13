# Fix #2: the zero point helps PTQ by 7.1%. The +82%/+204% was a padding defect, not the lever

**CORRECTED 2026-08-13, and the correction reverses the headline.** An earlier version of this file
concluded "fix #2 is answered NO … the ceiling is 1.06× against a 1.15× bar". That was wrong: it
measured a datapath whose padding was broken for an asymmetric grid and attributed the result to the
zero point. With the padding **correct**, on the same protocol, same seeds, same pinned references:

| arm | symmetric | asym, CUTLASS zero-fill | asym, code-z padding |
|---|--:|--:|--:|
| W4A4 PTQ | 0.5023 | 0.9125 (**+81.7%**) | **0.4665 (−7.1%)** |
| W4A4 MoDiff | 0.3095 | 0.9393 (**+204.0%**) | **0.3083 (−0.4%)** |

−7.1% on the PTQ axis is **12× the measured W4A4 cross-process floor** of 0.6%, and it is exactly what
was predicted on the record before any of this ran: *"this should help the PTQ axis and barely move
MoDiff"*. MoDiff's −0.4% sits at the floor, which is what reading the activation grid only at t=T
predicts.

**The 1.06× ceiling was accurate; using it to close the question was the error.** It predicted ~6% and
PTQ delivered 7.1%. The mistake was scoring 1.06× against `zp_headroom.py`'s 1.15× bar — a bar set when
fix #2's cost was *"15 CUDA entry points plus a Σ(w_q) fold"*. Those kernels are built and gated now, so
that bar no longer prices anything.

## What it costs: three implementations, measured

Correct padding is implemented in production kernels, selected by `MODIFF_ZP_PAD_MODE`:

| mode | mechanism | quantize+conv cost | end-to-end PTQ | end-to-end MoDiff |
|---|---|--:|--:|--:|
| `none` | CUTLASS zero-fill — **the defect** | baseline | +81.7% | +204.0% |
| **`halo`** (default) | quantize kernel emits a `z`-valued spatial halo; conv runs `padding=0` | **+8.9%** | **−7.1%** | **−0.4%** |
| `border` | keep zero-fill, add `(z/s)·ws[k]·Σ_missing w_q[k]` to the output's border pixels | **+5.3%** | −1.6% | −0.9% |

(`border`'s deficit is diagnosed below: it corrects an already-rounded value, so it inherits an ulp the
halo never incurs.)

Both corrections are exact in principle and agree to **1.000×** on an isolated fp32 conv
([`test_int4_zp_prepad.py`](../../integration/tests/test_int4_zp_prepad.py)). New entry points:
`group_norm_silu_quantize_pack_nhwc_zp_pad` (halo emitted in-kernel, so no extra traversal),
`pad_packed_int4_code` (one-pass code-padding of a packed tensor), `add_zp_border_correction`
(border-only, fp16/fp32).

**`halo` is the default, and the reason is now diagnosed rather than suspected**
([`border_vs_halo_diagnosis.py`](scripts/border_vs_halo_diagnosis.py),
[`data/border_vs_halo_diagnosis.json`](data/border_vs_halo_diagnosis.json)).

Both corrections are exact in exact arithmetic, and they agree to 1.000× on an isolated conv. Measured
against an **exact integer reference** — the conv recomputed in float64 from the integer codes, so the
reference contains no fp16 and no epilogue rounding — and split into border and interior pixels:

| mode | relL2 all | border | interior | border excess (quadrature) |
|---|--:|--:|--:|--:|
| `none` (defect) | 0.2729 | 0.6679 | 0.00754 | 0.6679 |
| `halo` | 0.00755 | 0.00762 | 0.00754 | **0.0011** |
| `border` | 0.00770 | 0.00844 | 0.00754 | **0.0038** |

The interior error is the epilogue's own rounding and is identical in all three modes, which is what makes
the border column readable. **The halo's border is as accurate as its interior; the post-hoc correction
adds 1.8–∞× more** (across four shapes). It corrects a value the epilogue has *already rounded* at ~2.7×
the true magnitude — one ulp at 3.3 is 2^-8 = 0.00391, which is exactly the observed max difference — and
that ulp is unrecoverable. Per conv the excess is invisible (0.0077 vs 0.0076 against a reference, or
0.4303 vs 0.4303 once 4-bit noise is included, which is why the isolated gate could not see it); over 70
layers it compounds into the end-to-end gap.

Two hypotheses were refuted on the way, and both are recorded in the script so they are not re-run:

* **coverage** — 70 correction calls per step against 70 padded convs, i.e. every one;
* **the fp16 store** — real and measurable as that one ulp, but not the driver: forcing an fp32 store
  leaves the gap intact (border 0.5101, halo 0.4778, symmetric 0.5212).

**So a post-hoc correction cannot match the halo at any store dtype**, because the rounding it inherits
happens inside the epilogue. "Make `border` equivalent" is not a tuning task — the cheap route is only
viable **fused into the epilogue**, where the accumulator is still exact, and there it would be both
cheaper than the halo *and* as accurate. That is the one remaining piece of work on this lever, and it is
now a specified change rather than an open question.

## What stands unchanged

Everything about **coverage**, which was P1's actual blocker, and everything about the **mechanism**:

* coverage is complete, gated and unconditional on both arms — four kernels honour `z`, and the three
  remaining `_zp_unsupported` sites are documented non-holes (§2);
* the padding defect's arithmetic, `−z·Σ(missing w_q)·ws/s` per output pixel, is confirmed to 0.9–2.6%
  against the closed form (§3), and it is what made the zero-filled numbers what they were;
* the fp16-bias hypothesis is still refuted (§4);
* `_refold_zp_bias` still refuses `z ≠ 0` on a padded conv by default — which remains right, because the
  shipped path still zero-fills. `MODIFF_ZP_ALLOW_PADDED=1` plus `MODIFF_ZP_PREPAD=1` is the correct
  combination, and the refusal is what stops a silently-defective asymmetric run.

---

## 1. The old census was wrong in both directions

[docs/zero_point_2026-08-13/FINDINGS.md](../zero_point_2026-08-13/FINDINGS.md) counted **70**
contaminated pairs on the MoDiff arm, 62 of them via `step1_static_quantize_pack_int4_fprop`,
described as *"the t=T activation grid"*, and prescribed teaching those kernels `z`.

Measured per call at the kernel boundary ([`data/site_census.json`](data/site_census.json)) — is
`a_hat` subtracted, is the scale the activation scale, does the consumer conv add a bias:

> **Which artifact shows what.** A fresh census run now reports **0** guard hits on both arms, because
> the delta sites declare `grid="delta"` and the activation sites route to a `_zp` kernel — that zero
> *is* the post-fix confirmation. So the historical counts are reproduced deliberately rather than
> implied: `CENSUS_COUNT_DELTA_AS_GAPS=1` restores the old name-based classification and writes
> [`data/site_census_name_based.json`](data/site_census_name_based.json), which shows the MoDiff arm's
> **62 via `step1_static_quantize_pack_int4_fprop` + 8 via `group_norm_silu_delta_quantize_resize_nhwc`
> = 70**. The PTQ arm's historical 8 do not come back there, because those sites now call a `_zp`
> kernel and never reach the guard at all; they are visible in
> [`data/coverage_gate.json`](data/coverage_gate.json) as 96 `group_norm_silu_quantize_resize_nhwc_zp`
> calls instead. [`data/site_census_after_coverage.json`](data/site_census_after_coverage.json) is the
> post-fix run kept alongside for the comparison.

| site | a_hat | grid | consumer | consumer bias | verdict |
|---|---|---|---|---|---|
| `step1_static_quantize_pack_int4_fprop` | **yes** | delta | `conv2d_int4_evt_o_hat` | **no** | delta → z inapplicable |
| `group_norm_silu_delta_quantize_resize_nhwc` | **yes** | delta | `conv2d_int4_evt_o_hat` | **no** | delta → z inapplicable |
| `scale_quantize_and_pack` (t=T) | no | activation | `conv2d_int4_fprop` | yes (python) | **real gap, and it had no guard** |
| `group_norm_silu_quantize_resize_nhwc` | no | activation | — | yes | real gap (8 layers) |

**62 of the 70 were false positives.** A difference of activations has no zero point — it cancels —
and those convs add no bias for one to pair with. Following the documented next step would have
corrupted the `a_hat` update, which dequantizes as `q/s`.

**The census simultaneously missed the site that was actually broken**, because that site had no
guard to be counted by: `_int4_conv`'s tensor branch, MoDiff's t=T entry point. It is the one
quantize per conv per sample whose conv adds the corrected bias, and it seeds the `o_hat` that every
later step accumulates into. That is why the MoDiff arm previously measured relL2 **7.3057** — a
divergence, not a grid.

The lesson generalises past this fix: **the old census classified by entry-point name, which cannot
distinguish two roles of one kernel.** `upsample2x_quantize_pack_noahat_fprop` is literally both — it
quantizes an activation when its `a_hat` argument is empty and a delta when it is not.

## 2. What changed

Four kernels learned `z`, each as a second arity (pybind11 does not inherit C++ defaults, so the
existing callers are untouched):

| kernel | role | constraint |
|---|---|---|
| `scale_quantize_and_pack_zp` | MoDiff t=T | — |
| `group_norm_silu_quantize_resize_nhwc_zp` | PTQ updown | packed int4 only; the int8 output has no matching bias correction and the wrapper `TORCH_CHECK`s it |
| `upsample2x_quantize_pack_noahat_fprop_zp` | PTQ Upsample | empty `a_hat` only; with a cache it quantizes a delta, and the wrapper refuses |
| `step1_static_quantize_pack_int4_noahat_fprop_zp` | `_forward_standard`'s fp16 branch | none — no `a_hat` parameter at all, so it has only the one role |

**The fourth was added to make the coverage claim unconditional.** It was initially left *guarded*
rather than taught, on the correct grounds that the census found it unreachable on both shipped W4A4
arms. But "unreachable in this configuration" makes the claim config-dependent: a configuration that
did reach it got a hard error instead of a result. It is the same one-line `+ z`, both its scalar and
vec2 instantiations are gated, and the odd-channel path (which selects the scalar kernel) is exercised
explicitly.

**Every activation-grid quantize on the int4 path now honours `z`.** The three `_zp_unsupported` calls
that remain are not coverage holes:

* `quantize_and_pack (float-scale branch)` — reachable only with a python-float `input_scale`, i.e. an
  *uncalibrated* layer, and a layer with a zero point is calibrated by construction (the zp arrives
  with the scale in `set_static_calibration`). Unreachable by construction, not by configuration.
* `group_norm_silu_quantize_resize_nhwc` — now the `else` of the `_zp` routing, so it fires only on the
  **int8** path, whose bias carries no `-z·Σw_q` correction. Honouring `z` there would *be* the defect.
* `upsample2x_quantize_pack_noahat_fprop` — its `else` is the delta role (exempt, verified) or `z = 0`.

`_zp_unsupported` gained `grid=`, so a site **declares** what it quantizes instead of being
classified by name. `"delta"` is exempt and the declaration is **verified** — a delta site with no
`a_hat` cache raises. The `z == 0` early-out still comes first, so the shipped symmetric path pays one
attribute read and one float compare, which is the cost that mattered in `2a2b1c3`.

`_refold_zp_bias` now **refuses** a non-zero `z` on a padded conv (`MODIFF_ZP_ALLOW_PADDED=1` to
reproduce the measurement). Given the negative answer, leaving an asymmetric table silently runnable
is a trap worth +82%/+204%.

### A bug the ratio caught, which is this fix pointing the wrong way

The first routing applied `z` in `_int4_conv` unconditionally, so **all five** t=T quantizes got it:
the activation *and* the four warm-up residuals, which run on a dynamic per-round scale through a
**bias-free** conv. 350 zp calls per run where 70 was correct — codes and bias disagreeing about `z`,
the same class of defect this fix exists to remove.

A `> 0` check passed that happily. What caught it was asserting the **ratio**:
`verify_zp_coverage.py` requires symmetric == 4× asymmetric on the MoDiff arm, because
`_forward_first_step` quantizes one activation with `z` and `warmup_steps-1 = 4` residuals without it.
It failed 350 : 0 and now reads 560 : 140.

## 3. Why ZERO-FILLED padding and the zero point cannot both hold

For output pixel `p`, the fold assumes every tap carries the zero point:

```
sum_i w_q[k,i] * a[i] = (sum_i w_q[k,i]*a_q[i] - z*sum_i w_q[k,i]) * ws[k]/s
```

`z*sum_i w_q[k,i]` sums over **all** taps, i.e. it assumes every tap carries `z`. A tap padded with the
byte 0 does not, so a border pixel needs the sum over only the taps it actually has. The residual is

```
-z * sum_{missing} w_q[k] * ws[k] / s
```

**per output PIXEL**, which is why it cannot be *corrected* by a per-channel bias. But it does not have
to be corrected: the term exists only because the padded tap was encoded wrongly, and padding with the
code `z` removes it at the source rather than compensating for it (§6). That is the difference between
this and fix #4's `z_w · Σa`, which is per-output-pixel in the DATA and therefore genuinely needs a
reduction.

Confirmed three ways:

* **Exact prediction.** [`test_int4_zp_padding.py`](../../integration/tests/test_int4_zp_padding.py):
  the corner-pixel error equals the formula above to **0.9–2.6%** across `(z, s)` in
  `{-2,-5,-7}×{8}` and `{-5}×{2}`. Border/interior concentration **95×**; `padding=0` asymmetric sits
  *at* the harness floor (0.0066 vs 0.0069) while `padding=1` is 36× it.
* **Localisation and recovery.** [`zp_padding_probe.py`](scripts/zp_padding_probe.py): asymmetric error
  is border-localised (1.47× interior over the whole ring) while symmetric is flat (1.00×), and
  pre-padding the activation with real zeros — which the quantizer encodes as code `z`, the correct
  padding value — recovers it. With no padded taps at all, asymmetric is harmless (1.00×, 0.97×).
* **Scale.** The border ring is 23% of pixels at 16×16, 44% at 8×8, **75% at 4×4**, and
  **70 of 70** calibrated convs are 3×3 `padding=1`. There is no subset of this model on which the
  zero point is implementable.

MoDiff is hit *hardest* (+204% vs PTQ's +82%) despite reading the activation grid only at t=T,
because the resulting `o_hat` is accumulated over every remaining step: a border error at t=T never
washes out.

## 4. Hypotheses tested and refuted

* **fp16 bias storage.** Two large terms cancel in the epilogue (|corr|/|o| median **7×**), so fp16
  rounding on the folded bias was a plausible culprit. Refuted: the bias is stored **fp32** and
  injects **0.00%** of the output scale ([`zp_bias_precision.py`](scripts/zp_bias_precision.py)).
* **Missing coverage.** Refuted by construction — [`data/coverage_gate.json`](data/coverage_gate.json)
  runs both arms under `MODIFF_ZP_STRICT=1` with the `_zp` entry points exercised at predicted counts,
  and the symmetric twins called **zero** times.

## 5. What is gated

| gate | what it holds |
|---|---|
[`test_int4_zero_point.py`](../../integration/tests/test_int4_zero_point.py) | the host-side fold: exact against fp64→storage-dtype, idempotent, bit-identical at z=0, bias-free conv gains one and gives it back |
[`test_int4_zero_point_kernels.py`](../../integration/tests/test_int4_zero_point_kernels.py) | per kernel: z=0 **bit-identical** (`torch.equal`), z≠0 differs, and the unclamped codes shift by **exactly +z** — checked by unpacking nibbles rather than reimplementing GroupNorm/SiLU/resize, so the gate cannot be wrong in a second place. Plus refusals: z on a delta quantize, z on the int8 resize output, 2D input |
[`test_int4_zp_padding.py`](../../integration/tests/test_int4_zp_padding.py) | the padding defect, quantitatively, against the closed form |
[`test_int4_zp_prepad.py`](../../integration/tests/test_int4_zp_prepad.py) | the code-`z` padding path: bit-identical to the normal path at z=0 (so a wrong pad byte, slice offset or padding/stride mismatch shows up on a case whose answer is known), the pad byte decodes to `z` in both nibbles for negative `z`, and it reduces a padded conv's error at z≠0 |
[`verify_zp_coverage.py`](scripts/verify_zp_coverage.py) | both arms clean under strict mode, with per-kernel counts and the 4× warm-up invariant |

`test_int4_zero_point.py` sets `MODIFF_ZP_ALLOW_PADDED=1` deliberately rather than moving its fixture
to `padding=0`: every calibrated conv in the model is 3×3 `padding=1`, so a `padding=0` fixture would
gate the fold on a shape the tree does not contain. That file gates the fold's *arithmetic*, which is
exact per output channel; the padding defect is a separate per-output-pixel term with its own file.

## 6. Recommendation

**Build the z-valued halo in the quantize kernels, and re-enable the zero point on the PTQ axis.**
The measured prize is −7.1% relL2 on W4A4 PTQ, 12× the cross-process floor. The mechanism is already
built, gated and bit-exact at `z = 0`; what is missing is only that the padded halo carries `z` instead
of 0.

The cheap route is closed and the reason is measured: the `MODIFF_ZP_PREPAD=1` emulation buys the 7.1%
accuracy for +7.1% latency, a 1:1 trade. The kernels that produce the packed activation
(`group_norm_silu_quantize_pack_nhwc_zp`, `group_norm_silu_quantize_resize_nhwc_zp`,
`scale_quantize_and_pack_zp`) already write every output byte and already know `z`; writing a halo of
`z` bytes around their output is a bounds change on an existing write, with no extra pass over the
activation and no reduction. The conv then runs with `padding=0`.

**Do not enable it on the MoDiff axis.** −0.4% is at the floor, and MoDiff's delta steps must keep
zero-fill: they quantize a difference on a symmetric delta grid where code 0 *is* delta 0, so a `z`
halo there would be the defect rather than the fix.

**Leave `_refold_zp_bias`'s refusal in place until that halo exists.** The shipped path still
zero-fills, where an asymmetric table costs +82%/+204%; the refusal is what prevents that being run by
accident, and `MODIFF_ZP_ALLOW_PADDED=1 MODIFF_ZP_PREPAD=1` is the deliberate combination.

### And the capability fix #4 needs is a DIFFERENT one

An earlier version of this section claimed fix #2 and fix #4 were blocked on one shared capability, a
per-output-pixel correction, and that only fix #4 justified it. Half of that is now wrong: **fix #2 does
not need it at all** — padding with `z` is a value change, not a reduction. Fix #4 still does:

```
sum_i (w_q[k,i] - z_w[k]) * a[i] = sum_i w_q[k,i]*a[i] - z_w[k] * sum_i a[i]
```

`Σ a[i]` runs over the conv window, so it genuinely varies per output pixel and cannot fold into a
per-channel bias. So the two are independent tasks, and both are now justified:

| lever | measured worth | what it needs |
|---|--:|---|
| fix #2, activation zero point | **−7.1%** W4A4 PTQ relL2 | a `z`-valued halo in three existing quantize kernels |
| fix #4, weight zero point + AdaRound | **1.58×** end-to-end, weight-only | a windowed `Σa` reduction the epilogue can consume |

## 7. The measurement-reliability question this raised — now answered separately

While measuring the above, the PTQ symmetric arm read 0.5267 → 0.4901 → 0.5022 in three runs on
identical inputs and the same cached references. That is chased to a conclusion in
**[FINDINGS_NOISE_FLOOR.md](FINDINGS_NOISE_FLOOR.md)**, and the short version is:

* the recorded floors **hold** (W4A4 0.09%/0.13%, W8A8 1.97%/0.91%, one arm per process, idle GPU);
* what actually breaks comparability is **arm order** — the W4A4 MoDiff arm reads 0.3954 measured
  first and 0.3095 measured after `int4_baseline`, **+27.8%**, 200× the floor. That retracts
  `arm_position_effect.py`'s `position_irrelevant` verdict and explains the 6.9% it could not.

**Nothing above is affected**, and the reason is structural rather than lucky: the fix #2 arms are all
measured in one run of one script in a fixed order, and compared against the symmetric baseline *from
that same run* (0.5022 / 0.3090) rather than against a committed number. The margins are +82% and
+204%, and the 1.06× ceiling uses no sampler at all.
