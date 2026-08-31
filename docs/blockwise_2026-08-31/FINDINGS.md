# Everything blockwise: which tensor actually pays, and at what block size

**Date** 2026-08-31 · **GPU** NVIDIA A40 · **Model** LSUN-Churches LDM-8, 50 DDIM steps

The ask was to make every quantized tensor blockwise, pick a block size, and report. Both halves
were done. The short answer:

> **G = 32 input channels** is the right block size, and **only the weights should use it.**
> Blockwise *activations* are worthless at every block size tested — the curve is flat in G — and
> at the shipped refresh cadence they are actively harmful. Blockwise weights are worth ~1.9x at
> W8A8. But no blockwise variant is implementable on the current epilogue: the only working
> implementation costs **4.7x** the conv path at G=32, so this is a mainloop project, not a knob.

Throughout, **G counts input channels**. A weight block is G channels x all R*S taps; an
activation block is G channels at one (n,h,w). Weights and activations share the same C-block
boundaries, which is the alignment any real implementation needs (section 2).

Raw data in [`data/`](data/); every figure regenerates from
[`scripts/make_plots.py`](scripts/make_plots.py).

---

## 0. Noise floor, first

Four independent runs happened to share an identical baseline arm. Same config, same seeds,
different process:

| arm | replicates | range |
|---|---|--:|
| W8A8 shipped, refresh=4 | 0.0273, 0.0266, 0.0258, 0.0240 | **0.0033** |
| W8A8 shipped, refresh=1 | 0.0240, 0.0233, 0.0226 | 0.0014 |
| W4A4 shipped, refresh=4 | 0.2788, 0.2794, 0.2801, 0.2791 | 0.0013 |
| W4A4 shipped, refresh=1 | 0.2782, 0.2795, 0.2784 | 0.0013 |

So the pipeline is not bit-deterministic across processes, and **an effect below ~0.003 relL2 is
not readable at W8A8.** Every claim below clears that bar or is explicitly labelled as not
clearing it. Quoted `+-` values are half the max-min over 3 seeds at n=24.

---

## 1. Why the shipped path is not blockwise (the constraint)

The MoDiff EVT epilogue ([`csrc/modiff/conv/conv2d_evt.cu`](../../csrc/modiff/conv/conv2d_evt.cu)) is

```
o_hat[elem] += acc * alpha * weight_scale[k]
```

with `alpha` a `VisitorScalarBroadcast<_0,_0,_0>` — one scalar — and `weight_scale` a
`VisitorRowBroadcast` over **output** channels. That is the whole story:

* **Output channel is the GEMM N dim**, so a per-output-channel weight scale factors out of the
  reduction and the epilogue can apply it. This is why the shipped weight quantization is
  per-output-channel and always has been.
* **A block scale varies along the reduction axis** `K = Cin*R*S`. The epilogue only ever sees the
  finished accumulator, so no epilogue node can undo a scale that was applied per-K-block. This is
  not an implementation gap; it is arithmetic.
* **Per-token (one scale per input pixel) does not rescue it either**, for R,S>1: a 3x3 conv output
  row draws from 9 input pixels with 9 different scales, so the scale is not a function of the GEMM
  M index. Only 1x1 convs could take a per-M scale in the epilogue.

Consistent with that, `grep -rE 'group_scale|block_scale|blockwise' csrc/` returns **0 hits**. The
delta-quantize kernel dereferences one float (`float scale = *scale_ptr`,
[`delta_quantize.cu:91`](../../csrc/modiff/quantize/delta_quantize.cu:91)).

Corollary that matters later: **`a_hat` is exempt.** It is a cache touched by elementwise kernels,
never a GEMM operand, so blockwise `a_hat` would be cheap. It is also not where the error is.

## 2. An exact blockwise implementation, on today's kernels

The D2 epilogue is a read-modify-write into `o_hat`. So calling the conv once per channel block —
each call with that block's own `alpha` and its own per-(block, out-channel) `weight_scales` —
accumulates exactly the blockwise-dequantized sum. No approximation.

Verified against an fp32 per-block reference with distinct scales per block:

```
split-K blockwise vs reference: relerr 3.7e-04   (residual is fp16 o_hat accumulation)
```

This is what section 4 measures the cost of, and it delivers per-block activation scales *and*
per-(block, out-channel) weight scales simultaneously.

## 3. Weight-only reconstruction error (analytic, 72 3x3 convs)

No sampling: read straight from the checkpoint.
[`scripts/weight_granularity.py`](scripts/weight_granularity.py). Metric is relative Frobenius
error of the dequantized weight, median and worst conv.

Two block axes, and the difference between them is the whole point:

* **chan** — G channels x all 9 taps (9G elements). C-aligned, implementable.
* **flat** — G contiguous elements of the flattened `[Cout, Cin*R*S]`. Finer at equal G, but the
  blocks straddle channels, so no channel-block split can produce it.

| W4 rule | median | worst | scale bytes |
|---|--:|--:|--:|
| per-channel absmax | 0.1956 | 0.4493 | 0% |
| **per-channel mse (shipped)** | **0.1299** | 0.2608 | 0% |
| chan-256 mse | 0.1293 | 0.2556 | 0.2% |
| chan-32 mse | 0.1238 | 0.2286 | 1.4% |
| chan-16 mse | 0.1185 | 0.2058 | 2.8% |
| chan-16 absmax | 0.1319 | 0.2278 | 2.8% |
| flat-16 mse | 0.0832 | 0.1178 | 25% |

At W8 the same sweep runs 0.0108 (per-channel) down to 0.0073 (chan-16); weight error at 8 bits is
already an order of magnitude below the activation error, so it barely matters.

**This reproduces the committed measurement.** `flat-128 absmax` at W4 gives 0.1295 median /
**0.2206** worst against the table in
[`_int4_weight_scale`](../../integration/kernels/int4_optimized.py:59)'s docstring
("group-128 absmax 0.1226 median, 0.2206 worst") — worst matches exactly; the median differs
because that table covered 87 convs including 1x1s and this one covers the 72 3x3s.

Two things follow:

1. **The committed decision to prefer per-channel MSE over group-128 still holds, and is stronger
   than the docstring claims.** Against the *implementable* C-aligned axis, per-channel MSE
   (0.1299) beats chan-256 MSE (0.1293) to within nothing and loses to chan-16 MSE by only 9%.
   The docstring's "recovers 96% of what group-wise would buy" was measured against the flat axis,
   which is the more favourable comparison for group-wise.
2. **The clip rule matters more than granularity at W4.** chan-16 *absmax* (0.1319) is worse than
   per-channel *mse* (0.1299). Granularity does not substitute for a good clip.

![weight granularity](plots/fig1_weight_granularity.png)

## 4. Cost of blockwise on the real kernel

Channel-block split-K, A40, B=128, on the high-frequency UNet ResBlock conv shapes.
[`scripts/blockwise_cost.py`](scripts/blockwise_cost.py).

Freq-weighted over the sampled shapes, conv path only:

| | ms/step | vs fused |
|---|--:|--:|
| fused, per-tensor (shipped) | 10.44 | 1.000x |
| G=64 | 25.19 | **0.415x** |
| G=32 | 49.17 | **0.212x** |
| G=16 | 97.41 | **0.107x** |

Dead linear in `Cin/G`, which is the prediction: the MAC work is unchanged (K is merely
partitioned) but every block re-runs the full `N*P*Q*K` epilogue, so accumulator and `o_hat`
traffic multiply by the block count. **These are a floor** — the block-slicing copies are hoisted
out of the timed region.

For scale: W8A8 full is 32.47 ms/step and W4A4 full is 21.47 ms/step on the committed conv-set
benchmark. Blockwise W4A4 at G=32 would land near 100 ms/step — three times the cost of just
using 8 bits.

![cost](plots/fig4_cost.png)

## 5. End-to-end quality: activations vs weights

n=24, 50 steps, 3 seeds, relL2 of the sampled latent against fp16. Only granularity varies between
arms — warmup rounds (5) and refresh cadence are held fixed, which the committed group-quant run
did not do (it moved granularity and cadence together, so it could not say which one paid).
[`scripts/blockwise_e2e.py`](scripts/blockwise_e2e.py).

**The result that decides everything:**

| G (channels) | 256 | 128 | 64 | 32 | 16 | shipped |
|---|--:|--:|--:|--:|--:|--:|
| W8A8 **activations only**, r=4 | .0431 | .0452 | .0485 | .0510 | .0532 | **.0240** |
| W8A8 **activations only**, r=1 | .0236 | .0230 | .0228 | .0227 | .0232 | **.0226** |
| W8A8 **weights only**, r=4 | .0268 | .0192 | .0228 | **.0139** | .0133 | .0258 |
| W4A4 **activations only**, r=4 | .2391 | .2390 | .2399 | .2403 | .2415 | **.2791** |
| W4A4 **weights only**, r=4 | .2760 | .2725 | .2608 | **.2419** | .2416 | .2801 |

Read the activation rows across: **they are flat.** At W8A8/r=1 every block size lands on
0.0227-0.0236 against a 0.0226 baseline — a 0.001 span against a 0.0014 noise floor, i.e. nothing.
At W4A4 the activation rows are also flat (0.2391-0.2415) but sit 0.040 *below* the baseline, and
that entire gain is already delivered by the coarsest arm — and equally by the `token act` arm
(0.2383), which is one scale per pixel and no channel blocking at all.

> **Blockwise activation quantization buys nothing from the blocks.** What little it buys at W4A4
> is buying its way off the per-tensor scale, and per-token gets you there.

The weight rows do slope, and clear the noise floor: W8A8 0.0258 -> 0.0139 at G=32 (1.9x), W4A4
0.2801 -> 0.2419 (1.16x). G=16 adds 0.0006 and 0.0003 respectively — inside noise.

![attribution](plots/fig5_attribution.png)

### Both together, and an unexplained inversion

| | shipped | G=256 | G=128 | G=64 | G=32 | G=16 |
|---|--:|--:|--:|--:|--:|--:|
| W8A8 both, r=4 | .0266 | .0434 | .0476 | .0506 | .0502 | .0541 |
| W8A8 both, r=1 | .0233 | .0299 | .0215 | .0230 | **.0134** | .0139 |
| W4A4 both, r=4 (mse W) | .2794 | .2341 | .2243 | .2149 | .1942 | **.1827** |
| W4A4 both, r=4 (absmax W) | .2788 | .1968 | .1845 | .1731 | .1668 | **.1495** |

Note `W8A8 both G=32 r=1` (.0134) lands on `W8A8 weights only G=32` (.0139) — the combined win is
the weight win, again.

At W4A4 the two effects are super-additive (weights alone 1.16x, activations alone 1.17x, together
1.53x), which is real but I did not isolate the mechanism.

**One inversion I cannot explain and am flagging rather than burying:** at W4A4, per-block
**absmax** weights beat per-block **mse** weights end-to-end (0.1495 vs 0.1827 at G=16) even though
MSE has the lower weight Frobenius error (0.1185 vs 0.1319, section 3). Lower reconstruction error
producing worse end-to-end output is the opposite of the assumption the shipped int4 clip search
rests on. This deserves its own investigation; it is not settled here.

![e2e](plots/fig2_e2e_relL2.png)

## 6. Mechanism: a tight block scale cannot absorb delta growth

Why blockwise activations *hurt* at the shipped cadence. The delta scale is computed from
`|delta|max` and then **held for 4 steps** (`DELTA_REFRESH=4`). Fraction of delta codes that clip:

| | shipped | G=256 | G=128 | G=64 | G=32 | G=16 |
|---|--:|--:|--:|--:|--:|--:|
| W8A8, refresh=4 | 0.000% | 0.21% | 0.35% | 0.59% | 1.00% | 1.72% |
| W8A8, refresh=1 | 0% | 0% | 0% | 0% | 0% | 0% |

A per-tensor scale is set by the globally worst block, so it is loose for every other block and has
headroom when a delta grows mid-window. A per-block scale is tight by construction, so growth
clips — and the finer the block, the tighter the scale and the more it clips. At refresh=1 the
scale comes from the current delta's own absmax and clipping is identically zero, which is why the
sign of the whole effect flips with cadence.

At W4A4 the grid is coarse enough that granularity dominates clipping, so blockwise wins there
despite clipping just as much.

![clip fraction](plots/fig3_clip_frac.png)

## 7. Recommendation

**Block size: G = 32 input channels.** It is the knee of the only curve that slopes (weights), and:

* G=16 adds 0.0006 relL2 at W8A8 and 0.0003 at W4A4 — both inside the noise floor — while doubling
  the cost again (0.212x -> 0.107x).
* 32 divides every UNet channel count exactly (192, 384, 576, 768, 1152, 1536), so no block ever
  needs padding.
* 32 int8 channels = 32 B = two 16-B vectorized NHWC accesses, so a block boundary never splits a
  `uint4` load.
* Scale metadata is 0.7% of int8 weight bytes, 1.4% at int4.

**Scope: weights only.** Sections 5 and 6 say activation blocks contribute nothing and, at the
shipped cadence, cost you. This is a useful narrowing rather than a disappointment: a mainloop that
only needs a per-K-block *weight* scale is a materially smaller change than full blockwise, because
weight scales are static, known at load time, and can be laid out for coalesced access — while an
activation block scale would have to be produced every step by the GN/delta-quantize kernel and
consumed in the mainloop.

**Do not ship the split-K version.** 4.7x on the conv path to buy 1.9x on a W8A8 error term that is
already 10x smaller than the W4A4 error is not a trade worth making. The honest options are:

1. **Leave it, and look at refresh cadence instead.** Paired within-run (same process, same seeds),
   refresh=1 beat refresh=4 on the shipped W8A8 path in all three runs that measured both, by
   +0.0033 / +0.0033 / +0.0014 relL2. The sign is consistent but two of the three margins only just
   clear the 0.0033 cross-run floor, so treat this as *promising and unconfirmed* rather than
   measured — it wants its own paired run before anyone acts on it. It is attractive because it is
   an env knob rather than a kernel, and because this tree already built free absmax reporting to
   make per-step refresh cheap (see the note at
   [`int8_optimized.py:279`](../../integration/kernels/int8_optimized.py:279)). At W4A4 the same
   comparison is +0.0006 / -0.0001 / +0.0007 — cadence does nothing there.
2. **Build a fused blockwise-weight mainloop.** One epilogue pass, a weight scale folded per K-block
   inside the reduction. Expected overhead is a scale load per K-tile rather than a 4.7x epilogue
   multiplier. Scope is comparable to the existing hand-assembled `ImplicitGemmConvolutionEVT`
   (CUTLASS 4.6.1 has no EVT-on-conv path, so there is no library shortcut here either).

**Blockwise does not rescue W4A4.** Best blockwise W4A4 is 0.1495 relL2 against W8A8's 0.0259 —
still 5.8x worse, at 3x the cost of just using 8 bits. If the goal is low-bit activations, the
lever is not granularity.

---

## Open

* The W4A4 absmax-beats-mse inversion (section 5). Lower weight reconstruction error giving worse
  end-to-end output contradicts the premise of the shipped int4 clip search.
* The W4A4 weight/activation super-additivity (1.16x x 1.17x -> 1.53x) is measured, not explained.
* Everything here is relL2 on 72 latents. No FID was run: at W8A8 the effects are at or near the
  0.003 noise floor and FID at this sample count would not resolve them. The W4A4 blockwise gain
  (0.279 -> 0.150) is large enough to be worth an FID confirmation, which was not done.
* `a_hat` blockwise is cheap (section 1) and untested here. It is very likely irrelevant given that
  activation granularity is flat, but it is the one blockwise variant that would cost nothing.
