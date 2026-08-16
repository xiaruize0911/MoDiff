# FID, and why W4A4 is the wrong configuration for MoDiff

**2026-08-05 · A40 · LSUN-churches LDM, real checkpoint · 10,000 images/mode · DDIM 50 · batch 128**

First perceptual measurement in this project. Every accuracy figure before this was latent relative
L2, which orders the modes correctly but is not perceptual and cannot be compared to the paper.

## Result

Reference: 10,000 real LSUN church_outdoor images, center-cropped to the short side and resized to
256×256 bicubic, saved lossless. Generated samples use identical preprocessing — FID compares two
Inception feature distributions, so any asymmetry between the two sides is measured as distance.

| mode | FID vs real | FID vs fp16 | quantization error removed |
|---|---:|---:|---:|
| FP16 (reference model) | 7.803 | 0.000 | — |
| W8A8 baseline (MoDiff off) | 16.366 | 6.394 | — |
| **W8A8 + MoDiff** | **7.802** | **0.175** | **97.3%** |
| W8A4 baseline (MoDiff off) | 311.453 | 312.530 | — |
| **W8A4 + MoDiff** — *the paper's configuration* | **35.303** | **18.647** | **94.0%** |
| W4A4 baseline (MoDiff off) | 277.963 | 277.981 | — |
| W4A4 + MoDiff | 200.139 | 191.092 | 31.3% |

The W8A4 rows were **added 2026-08-16** — the images had been on disk since 04:36 that morning and the
FID was never computed, so the paper's own configuration had no row in the table measuring it. Same
protocol, same 10k reference, same script; `eigh` agrees with pytorch_fid to 1.71e-12.

**Read the last two columns together, because they say different things.** MoDiff removes **94.0%** of
A4 quantization error — the largest figure in this table, and the paper's method working in the paper's
configuration. But 35.303 does **not** beat the W8A8 baseline's 16.366, and the strong form of the
paper's claim (4-bit activations plus MoDiff beating 8-bit PTQ) therefore does not hold on FID, where
latent relL2 said it did by 1.65×. That reversal is the third of this session and is written up above.

Two further rows, from the CURRENT tree rather than this document's, so they are listed apart rather
than mixed into the table above:

| mode (current tree, 2026-08-16) | FID vs real | FID vs fp16 |
|---|---:|---:|
| W4A4 + MoDiff, MSE weight scale (shipped default) | 181.514 | 171.276 |
| **W4A4 + MoDiff + AdaRound rounding** | **140.187** | **125.138** |

No "error removed" column for these two: it needs a W4A4 baseline generated from the same tree, and the
one on disk is this document's (absmax weight scale, 2026-08-05). Quoting 277.981 against them would be
mixing trees, which is the same mistake that made the B5 arms worth regenerating.

**W8A8 + MoDiff is indistinguishable from the fp16 model** (7.802 vs 7.803) while running 1.36× faster
end to end. Quantization costs +8.56 FID and MoDiff returns essentially all of it.

Sanity of the anchor: the LDM paper reports 4.02 for LSUN-Churches at 200 steps / 50k samples. 7.803
at 50 steps / 10k is where that should land — both the lower step count and the smaller N raise FID.
fp16 and int8_baseline reproduced to 3 decimals across two independent processes.

**Caveat on N.** FID is biased upward at 10k versus the standard 50k, so these are not comparable to
published absolute numbers. Cross-mode comparison at fixed N is what they are for.

## Latent relL2 is a badly nonlinear proxy for perceptual quality

| latent relL2 | FID vs real |
|---:|---:|
| 0.000 (fp16) | 7.80 |
| 0.039 (W8A8+MoDiff) | 7.80 — parity |
| 0.238 (W8A8 baseline) | 16.37 |
| 0.456 (W4A4+MoDiff) | 200.14 |

relL2 **understated** MoDiff at W8A8 (it said 6× better; FID says the distributions match) and
**overstated** it at W4A4 (1.7× better in relL2 is almost no perceptual recovery). Every W4A4 quality
claim made from relL2 in this project should be read with that in mind.

## W4A4 is weight-limited, and MoDiff is an activation method

Three checks, in order.

**1. Not a misconfiguration.** Reproducing the FID generator's exact build path gives latent relL2
0.4564 for W4A4+MoDiff, against the shipped 0.4513–0.4979, and 0.7297 for the baseline against
~0.781. The generator is configured correctly; FID 200 is genuinely what relL2 0.46 looks like.

**2. The paper's configuration is W8A4, not W4A4.** MoDiff replaces the quantized tensor with a
temporal delta, which shrinks the quantizer's dynamic range. That addresses ACTIVATION quantization
and does nothing about weight error. `int8_optimized.py:170` documents that the delta clip ratio is
applied as `Q_level = 127/ratio`, so setting the ratio to 127/7 restricts activations to 15 levels
while the GEMM keeps int8 weights — quality-exact W8A4 (at W8A8 speed; a real A4 datapath needs int4
tensor cores, which require both operands at 4 bits, which is why W8A4 is not a shipped mode).

| config | activation levels | latent relL2 vs fp16 |
|---|---:|---:|
| W8A8 + MoDiff | 255 | 0.068 |
| **W8A4 + MoDiff** — the paper's config | 15 | **0.127** |
| W8A3 + MoDiff | 7 | 0.383 |
| W4A4 + MoDiff — our shipped int4 | 15 | 0.444 |

Same activation precision, 3.5× different outcome: the gap is entirely the weights. So
`277.96 → 200.14` is not a failure to reproduce the paper; it is a strictly harder configuration the
paper never claimed.

> **THE FID ROW FOR THE PAPER'S CONFIGURATION, MEASURED 2026-08-16 — and it reverses the sentence that
> used to end this paragraph.** This section previously concluded that W8A4+MoDiff "beats the W8A8
> **baseline**  — the paper's claim in substance". That was a relL2 comparison (0.1553 vs 0.256). The
> images had been on disk since 04:36 that morning, 10000 per arm, generated after the warm-up fix; the
> FID was never computed. It is:
>
> | mode | FID vs real |
> |---|--:|
> | fp16 | 7.803 |
> | W8A8 + MoDiff | 7.802 |
> | **W8A8 PTQ (the baseline in question)** | **16.365** |
> | W8A4 PTQ | 311.453 |
> | **W8A4 + MoDiff — the paper's configuration** | **35.303** |
>
> **MoDiff works, and works enormously, in the paper's own configuration**: −276.151 FID, −88.67%,
> 40× paired bootstrap 95% CI [−276.975, −275.028]. A4 PTQ is unusable at 311.5 and MoDiff makes it 35.3.
>
> **But it does not beat the W8A8 baseline on FID: 35.303 against 16.365, a 2.16× deficit**, where relL2
> gave it a 1.65× advantage. THIRD sign reversal between relL2 and FID this session (B3, C7, this), and
> the most consequential: the first two had relL2 overrating a *change*, this has it overrating the
> *method*. Per the rule those two established, the FID row is the verdict, so the strong form of the
> paper's claim — 4-bit activations plus MoDiff beating 8-bit PTQ — **does not hold here**. The weak form
> does: MoDiff removes 88.67% of A4 quantization error.
>
> Anchors recomputed with the same script and the same reference to make the comparison sound: fp16
> 7.803 and W8A8 PTQ 16.365, reproducing this document's 7.803 / 16.366 to 0.001. `eigh` agrees with
> pytorch_fid to 4.86e-12.

> **NOTE, 2026-08-06 (docs/act_bits_2026-08-05).** The rows above were produced by abusing
> `MODIFF_DELTA_CLIP`, which moves only the delta quantizer and leaves the static grid that quantizes
> t=T at A8. That mattered for a reason nobody had spotted: the t=T warm-up loop was a **no-op** on the
> calibrated path — it passed the static grid to every residual round, so round 2 rounded to zero —
> leaving the A4 anchor with 40% relative error where the paper's converged warm-up carries 1e-5.
> Leaving t=T on the A8 grid accidentally approximated a working warm-up.
>
> With the warm-up fixed, W8A4+MoDiff with **every** conv activation site at A4 — the paper's protocol
> — measures **0.1553 ± 0.015**, so this table's conclusion stands: it beats the W8A8 baseline (0.256).
> The 0.127 itself was a single seed at the bottom of a 0.130–0.196 spread. Full sweep, before and
> after the fix, in docs/act_bits_2026-08-05/FINDINGS.md.

**3. A real defect in the int4 weight quantizer, now fixed.** It used one symmetric absmax scale per
output channel over the whole flattened kernel — 15 levels shared by up to ~1700 weights, with a
median max/median weight ratio of 6.5 and up to 24.9. Measured over the 87 quantized convs of the
real checkpoint, relative Frobenius error of the reconstructed weight:

| scale rule | median | worst | inference cost |
|---|---:|---:|---|
| per-channel absmax (was shipped) | 0.1825 | 0.4493 | — |
| per-channel p99.9 | 0.1498 | 0.3203 | none, load-time only |
| **per-channel MSE search (now default)** | **0.1254** | **0.2609** | **none, load-time only** |
| group-128 absmax | 0.1226 | 0.2206 | needs the scale inside the K-loop |

The MSE search recovers 96% of what group-wise quantization would buy while keeping the one-fp16-
scale-per-output-channel layout the CUTLASS int4 epilogue folds, so it is free at inference.
`integration/kernels/int4_optimized.py::_int4_weight_scale`, `MODIFF_INT4_WSCALE=absmax` reverts.

End to end, paired A/B (same 4 seeds per arm, batch 16, DDIM 50, warm-up discarded):

| | mean relL2 | per-seed |
|---|---:|---|
| absmax | 0.5067 ± 0.0195 | 0.5248, 0.4983, 0.5206, 0.4833 |
| MSE | 0.4689 ± 0.0093 | 0.4737, 0.4688, 0.4772, 0.4559 |

4/4 seeds improve, mean −7.5%, effect ≈2× the spread. Note the end-to-end gain (−7.5%) is far
smaller than the weight-reconstruction gain (−31%): minimising ‖W−Q(W)‖ is not the same objective as
minimising output error, and clipping outliers trades away some of the salient weights AWQ exists to
protect.

**A correction on that last point.** A first, unpaired single-seed check reported the MSE scale as
making things *worse* (0.4437 → 0.4981). In that same comparison the int8 rows — which this code
cannot affect — moved by 10–30%, which is what exposed it as noise rather than signal. The paired
design above is the measurement to trust.

## Reproducing

```
python docs/fid_2026-08-05/scripts/export_lsun_reference.py --n 10000
python docs/fid_2026-08-05/scripts/generate_fid_samples.py --n 10000 --batch 128 --steps 50
python docs/fid_2026-08-05/scripts/compute_fid.py
```

Do not run `compute_fid.py` while generation is in flight. They share the GPU; a concurrent FID
process holding 1.8 GiB OOM'd the VAE decode mid-run and cost a mode's worth of generation. The
decode is now chunked (`--decode-chunk`, default 32) which bounds the peak, but the contention is
still real.

Four things the generator makes load-bearing, each of which would silently distort the result:
real-checkpoint calibration (`*_realckpt.pt`); one warm-up sampling run discarded per mode, because
quantized attention self-calibrates over its first forwards; a different seed per batch, or FID sees
79 copies of the same 128 images; the same seed sequence across modes, which makes the comparison
paired.

## Open

- **FID for W8A4+MoDiff.** The one row directly comparable to the paper's table. ~30 min.
- **FID for W4A4+MoDiff with the new weight scale.** −7.5% relL2 at a point where the relL2→FID
  curve is very steep; the effect on FID is unknown and could be either sign.
- FID at 50k for publication-grade absolute numbers.
