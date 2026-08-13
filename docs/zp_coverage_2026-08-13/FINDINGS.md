# Fix #2 is answered: NO. Coverage is complete, the obstacle is zero-padding, and the ceiling is 1.06×

**The activation zero point does not help W4A4 in this tree, and the reason is not this
implementation.** Coverage — the thing
[docs/zero_point_2026-08-13/FINDINGS.md](../zero_point_2026-08-13/FINDINGS.md) left open — is now
complete and gated on both arms. With it complete the asymmetric grid measures **worse**, and two
independent measurements say why and say that fixing it would not be worth it:

| | symmetric | asym r=1 | asym r=4.5 | best change |
|---|--:|--:|--:|--:|
| W4A4 PTQ | 0.5022 | 5.1985 | 0.9125 | **+81.7%** |
| W4A4 MoDiff | 0.3090 | 5.0680 | 0.9393 | **+204.0%** |

`docs/zero_point_2026-08-13/data/zp_measured.json`, 3 seeds, 50 steps, batch 8, pinned fp16
references. MoDiff's symmetric arm reproduces the committed shipped 0.3090 exactly.

**Those are defect magnitudes, and the defect is zero-padding — not missing coverage.** CUTLASS's
implicit GEMM zero-fills padded taps, so a padded tap reads code `0`, which an asymmetric grid
dequantizes to `-z/s` rather than `0`, while the folded bias subtracts a per-output-**channel**
correction for a sample that was never taken.

**And the ceiling says do not fix it.** On the captured `silu(gn(x))` tensors — no kernel, no conv, no
padding, no sampler — the best asymmetric reconstruction beats the best symmetric one by **1.06×**
(61 of 70 convs). `zp_headroom.py` set fix #2's bar at 1.15×. Fix #3's clip ratio already took the
slack the zero point was after, which is also why the r=1 arms above are catastrophic and the r=4.5
arms are merely bad.

---

## 1. The old census was wrong in both directions

[docs/zero_point_2026-08-13/FINDINGS.md](../zero_point_2026-08-13/FINDINGS.md) counted **70**
contaminated pairs on the MoDiff arm, 62 of them via `step1_static_quantize_pack_int4_fprop`,
described as *"the t=T activation grid"*, and prescribed teaching those kernels `z`.

Measured per call at the kernel boundary ([`data/site_census.json`](data/site_census.json)) — is
`a_hat` subtracted, is the scale the activation scale, does the consumer conv add a bias:

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

Three kernels learned `z`, each as a second arity (pybind11 does not inherit C++ defaults, so the
existing callers are untouched):

| kernel | role | constraint |
|---|---|---|
| `scale_quantize_and_pack_zp` | MoDiff t=T | — |
| `group_norm_silu_quantize_resize_nhwc_zp` | PTQ updown | packed int4 only; the int8 output has no matching bias correction and the wrapper `TORCH_CHECK`s it |
| `upsample2x_quantize_pack_noahat_fprop_zp` | PTQ Upsample | empty `a_hat` only; with a cache it quantizes a delta, and the wrapper refuses |

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

## 3. Why padding and the zero point cannot both hold

For output pixel `p`, the fold assumes every tap carries the zero point:

```
sum_i w_q[k,i] * a[i] = (sum_i w_q[k,i]*a_q[i] - z*sum_i w_q[k,i]) * ws[k]/s
```

`z*sum_i w_q[k,i]` sums over **all** taps. A padded tap reads code 0, not code `z`, so a border pixel
needs the sum over only the taps it actually has. The residual is

```
-z * sum_{missing} w_q[k] * ws[k] / s
```

**per output PIXEL**, which is why it cannot fold into a per-channel bias — the same obstruction as
the weight zero point in fix #4.

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
[`verify_zp_coverage.py`](scripts/verify_zp_coverage.py) | both arms clean under strict mode, with per-kernel counts and the 4× warm-up invariant |

`test_int4_zero_point.py` sets `MODIFF_ZP_ALLOW_PADDED=1` deliberately rather than moving its fixture
to `padding=0`: every calibrated conv in the model is 3×3 `padding=1`, so a `padding=0` fixture would
gate the fold on a shape the tree does not contain. That file gates the fold's *arithmetic*, which is
exact per output channel; the padding defect is a separate per-output-pixel term with its own file.

## 6. Recommendation

**Close fix #2 as answered negatively.** Keep the mechanism: it is bit-exact at `z = 0`, costs nothing
in the shipped configuration, is gated, and now refuses the configuration that is wrong. Do not build
the position-aware epilogue correction that a padding fix would need — its ceiling is 1.06× against a
1.15× bar.

The `MODIFF_ZP_STRICT` / `MODIFF_ZP_ALLOW_PADDED` pair is what makes this a closed question rather
than a landmine: an asymmetric table cannot be run by accident, and can still be run on purpose.

## 7. Open, and NOT closed by this file

**The W4A4 relL2 noise floor of 0.05–0.6% does not hold across processes.** The PTQ symmetric arm,
same protocol and the same cached fp16 references, measured 0.5267 → 0.4901 → 0.5022 in three runs
today — a 7% spread — after having reproduced 0.5266851782798767 to 16 significant digits earlier.
At least one of those runs overlapped another CUDA process on the same GPU, which
`docs/bench_report_2026-08-13/scripts/run_all.sh` already documents as able to turn CV 0.23% into 38%.

This does not touch anything above: +82% and +204% are two orders of magnitude past that spread, and
the 1.06× ceiling is measured without a sampler at all. But it does mean **any past conclusion in this
tree resting on a few percent of W4A4 relL2 needs re-checking against a floor measured with N repeats
in N separate processes on an otherwise-idle GPU** — which is P5's subject, not this file's.
