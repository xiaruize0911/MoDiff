# `o_hat` compression, and whether the a_hat/o_hat error can be "aligned"

A40, LSUN-churches LDM-KL-8. Sections 1–2 are measured on the live model; section 3 is the
PyTorch model of the kernel-1 recurrence, which `REPORT.md` §3 validated against the real CUDA
kernel to 1e-4 (the real kernel does not implement any of the proposed recursions, so a model is
the only way to price them before building).

## 1. `o_hat` is the larger cache

| | batch 32 measured | scaled to batch 128 |
|---|---|---|
| `a_hat` (int8 B=32 + fp32 block scales) | 175.4 MB | 702 MB |
| **`o_hat` (fp16)** | **282.8 MB** | **1131 MB** |
| ratio | **1.61x** | |

`o_hat` is fp16 `[N,K,H,W]` channels_last, allocated at `int8_optimized.py:889`; `o_dtype` is
`float16` when calibrated and `float32` during calibration — there is no low-precision path.
It is also the conv's output tensor (epilogue RMW, `out` aliases `o_hat`).

Note the asymmetry that makes compressing it attractive: the **output** is transient (one layer
live at a time) while the **cache** is persistent (70 layers live). Un-aliasing them costs one
extra fp16 write and takes epilogue traffic from 4.0 to 4.25 B/elem — near neutral — while making
the persistent 1131 MB compressible.

## 2. `o_hat` is an accumulator whose increment is about one LSB

`r_t = ‖o_hat_t − o_hat_{t−1}‖ / ‖o_hat_t‖`, 50 DDIM steps, batch 8, 10 layers sampled
(`scripts/ratio.py` — forward hooks do not fire, the fused ResBlock calls
`forward_gn_fused_modiff` directly, so this snapshots per UNet call):

| layer | r med | r min | amax/rms | **step ÷ min increment**: i8 per-tensor / i8 B=32 / fp16 |
|---|---|---|---|---|
| input_blocks.3.0.out_layers.3 | 0.0607 | 0.0360 | 5.64 | 1.23 / 0.59 / 0.014 |
| input_blocks.7.0.in_layers.2 | 0.0275 | 0.0181 | 4.37 | 1.91 / 0.92 / 0.027 |
| input_blocks.13.0.out_layers.3 | 0.0920 | 0.0481 | 5.41 | 0.89 / 0.43 / 0.010 |
| **output_blocks.0.0.in_layers.2** | 0.0164 | **0.0077** | 5.50 | **5.64 / 2.71** / 0.064 |
| output_blocks.5.0.out_layers.3 | 0.0870 | 0.0417 | 5.99 | 1.13 / 0.54 / 0.012 |
| output_blocks.8.0.in_layers.2 | 0.0595 | 0.0283 | 5.80 | 1.61 / 0.78 / 0.017 |

Last column > 1 means one step's update is smaller than one quantization step and gets rounded
away. At int8 per-tensor every layer is at or above 1. Blockwise B=32 (scaled by the 2.08x error
reduction measured for `a_hat`) puts the median near 0.7 but leaves `output_blocks.0.0` at 2.71.
fp16 has 15–100 LSB of headroom, which is why the shipped path is fine.

**`o_hat` has no self-correction.** `a_hat`'s error is absorbed exactly by the next delta
(`d_t = o_t − â_{t−1}`), which is why 8-bit `a_hat` keeps a 5.9x threshold margin. Nothing ever
recomputes the true conv output, so `o_hat`'s error is an unanchored sum. Stochastic rounding or
error feedback is therefore a **requirement**, not an optimization.

## 3. Can the two be "aligned"? Two schemes, simulated

The `Σηₖ` term exists because of a write-side inconsistency: `o_hat` accumulates the **codes**
while `a_hat` stores the **rounded** value. Feeding the conv `â_t − â_{t−1}` instead telescopes:

```
now:      Σ_t conv(q_t/s_t)      = conv(consumed_T) − conv(Σ_{k<T} η_k)     ← grows
aligned:  Σ_t conv(â_t − â_{t−1}) = conv(â_T) = conv(consumed_T) + conv(η_T) ← one term
```

Measured accumulation factor is 7x (`REPORT.md` §4, √48), so the ceiling on this idea is 7x.

![sim](plots/sim_aligned.png)

### 3a. Window-frozen block scale — **does not deliver, dropped**

Telescoping needs a constant scale, so freeze the `a_hat` block scale for K steps. `acc_err` at
t=48 (5-layer, 8-bit B=32; `scripts/sim_aligned.py`, `sim_aligned2.py`):

| layer | current | K=2 | K=5 | K=10 | K=49 | K=5 m=1.5 | K=10 m=1.5 | K=10 m=2.0 |
|---|---|---|---|---|---|---|---|---|
| 192x32x32 | 0.0430 | 0.0347 | 0.1045 | 0.2410 | 0.5355 | 0.0265 | 0.0470 | **0.0239** |
| 384x16x16 | 0.0399 | **0.0346** | 0.2173 | 0.2781 | 0.2648 | 0.1176 | 0.1882 | 0.1333 |
| 1536x4x4 | 0.0526 | 0.0412 | 0.0738 | 0.0981 | 0.1533 | 0.0360 | **0.0313** | 0.0335 |
| 768x16x16 | 0.0542 | 0.0416 | 0.0836 | 0.1209 | 0.2168 | 0.0398 | 0.0507 | **0.0393** |
| 576x32x32 | 0.0615 | 0.0455 | 0.0929 | 0.1271 | 0.1624 | 0.0483 | 0.0598 | **0.0454** |

Best case is **1.15–1.80x** better than current, and only with a headroom factor `m` tuned jointly
with K; at `m=1` and K≥5 it is 2–12x **worse**. Mechanism, from the clip diagnostics: `clip_i = 0`
everywhere (the increment transport is fine, never overflows int8) but `clip_a` runs
10.9% (K=1) → 17% (K=2) → 34% (K=5) → 87% (K=49) — the frozen grid clips, and because the aligned
scheme removes `a_hat`'s self-correction, that clipping now damages the output directly. On top of
that, re-gridding the state at a window boundary (`round(a_code·S_old/S_new)`) costs a full `η`
each time, so K=1 is 1.66x worse than current even though it re-derives the scale every step.
Two tuned parameters for ≤1.8x on 4 of 5 layers: not worth it.

### 3b. σ-Δ drift correction — **delivers the full 7x, with 2 bits of state**

Do not align the grids; correct the codes. Track the drift between what the conv accumulated and
what is stored, and fold it into the next step's codes:

```
q_t  = clamp(round((O_t − â_{t−1})·s_t))          unchanged
q'_t = clamp(round(q_t − D̂_{t−1}·s_t))            what the conv is fed   ← the only change
â_t  = Q_ahat(cons_t)                             unchanged, 8-bit blockwise
D_t  = (Σ q'/s) − â_t                             stored at R bits of one a_hat LSB
     = −e_{t−1} + ρ/s_t − η_t                     bounded: no sum
```

`acc_err` at t=48, and the same quantity for simply buying `a_hat` bits at the same B/elem
(`scripts/sim_sd.py`):

| layer | 8b (1.125 B) | **σ-Δ R=2 (1.375 B)** | 10b (1.375 B) | 11b (1.500 B) | 12b (1.625 B) |
|---|---|---|---|---|---|
| 192x32x32 | 0.0430 | **0.0027** | 0.0106 | 0.0056 | 0.0026 |
| 384x16x16 | 0.0399 | 0.0115 | **0.0100** | 0.0049 | 0.0025 |
| 1536x4x4 | 0.0526 | **0.0049** | 0.0129 | 0.0065 | 0.0032 |
| 768x16x16 | 0.0542 | **0.0083** | 0.0133 | 0.0066 | 0.0033 |
| 576x32x32 | 0.0615 | **0.0108** | 0.0148 | 0.0073 | 0.0036 |

- **It works as predicted**: 0.043–0.062 → 0.0023–0.0115, a **5.3–16x** reduction, i.e. the full
  accumulation factor and then some.
- **2 bits of drift is enough.** R=2/3/4/8 differ in the 4th decimal — the residual is dominated
  by `η` (a_hat's own 8-bit rounding), so more drift bits buy nothing.
- **But at equal memory it only beats buying bits by ~1.6x median** (1.4–3.9x on four layers,
  0.87x on one), and because it saturates while the buy-bits line keeps descending, **the two
  cross at ≈1.5–1.6 B/elem** — above that budget, plain higher-precision `a_hat` wins outright.

## 4. Conclusions

1. **The window-frozen scheme is dropped.** ≤1.8x, two coupled tuned parameters, fragile.
2. **σ-Δ drift correction is the correct "aligned" scheme** and gets the whole 7x for 2 bits. It
   aligns by correcting the codes, not by aligning the grids.
3. **On `a_hat` there is no large free lunch left.** The accumulation factor (7x ≈ 2.8 bits) costs
   about what 2 bits of drift state costs, so 8-bit B=32 sits close to the efficient frontier —
   and with a 5.9x threshold margin we do not currently need the accuracy.
4. **The place this trick pays is `o_hat`, because there it is free.** `o_hat`'s problem is
   swallowing, not accumulated rounding, and stochastic rounding needs no state at all (the
   epilogue already holds the exact value in registers). Recommendation stands: **int8 B=32 along
   K + stochastic rounding, B pinned to 32 to match the next layer's `a_hat` blocking**, for
   −495 MB at near-neutral epilogue traffic.

**Still a model, not a measurement:** `Σζ` (o_hat's own quantization error) cannot be measured in
this harness — it needs the conv in the loop. The estimate is 0.043 at int8 B=32 with stochastic
rounding (δ/rms = 5.6/127/2.08, rms rounding δ/√12, √50 random walk), which combines with a_hat's
0.053 in quadrature to 0.068 — 4.4x inside the 0.30 threshold. That is the next thing to measure.

## Reproduction

```bash
python docs/ohat_compress_2026-09-03/scripts/probe.py        # cache sizes
python docs/ohat_compress_2026-09-03/scripts/ratio.py        # increment / accumulator
python docs/ohat_compress_2026-09-03/scripts/sim_aligned.py  # window-frozen, v1
python docs/ohat_compress_2026-09-03/scripts/sim_aligned2.py # + carry + headroom
python docs/ohat_compress_2026-09-03/scripts/sim_sd.py       # sigma-delta vs buying bits
python docs/ohat_compress_2026-09-03/scripts/plot_sim.py
```
