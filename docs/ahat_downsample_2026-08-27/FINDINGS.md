# Spatially downsampled a_hat: refuted by the accumulator invariant, not by lost detail

**Status: measured, refuted. No CUDA code written.** Prompted by the project owner pushing back on
the a_hat line being closed — *"我觉得还是应该考虑减少 a_hat 的读写"* — which was a fair objection: every
refutation on record attacked a_hat's **bytes per element** (int8/fp8/companding) or its **number of
writes** (skip-K, deferred-write). Nothing had attacked the **number of elements**, which is the one
framing that reduces a_hat's read and its write together.

## The idea, and why it was not covered by the existing refutations

Store a_hat at 1/f spatial resolution; on read, upsample; quantize
`delta = silu(gn(x)) - upsample(a_hat_small)` as usual.

* The **bit-budget** argument that killed int8 storage does not apply: a_hat's dynamic range is
  untouched, so there is no 11.6-bits-needed-vs-8-available shortfall
  ([int8_ahat_cache](../int8_ahat_cache_2026-08-26/FINDINGS.md)).
* The **2.024 / 1.742 ms/step ceiling** does not bound it either. That ceiling is for schemes
  removing the a_hat *write*; a_hat read+write is **8 of the apply kernel's 10 sectors/warp**
  ([C17](../OPEN_ITEMS.md)), so f=2 cuts a_hat traffic 4× → 10 sectors → 4, about **1.5× what
  write-elision alone can remove**.
* The **structural** objection that killed skip-K (frozen anchor ⇒ codes are not increments) does
  not obviously apply: every step still computes a fresh code against a fresh reference.

So it needed measuring, not arguing.

## Step 1: the entry price, on real tensors

[`scripts/ahat_downsample_screen.py`](scripts/ahat_downsample_screen.py) hooks
`forward_gn_fused_modiff`, reconstructs the real quantization target `silu(gn(x))` (+mod, +smooth)
in fp32, and compares `absmax(delta)` against the real full-resolution a_hat versus a
downsample→upsample copy of it. 15 real shapes, W8A8+MoDiff, batch 4, 10 steps:

| reference | median delta-absmax inflation | median MoDiff gain (act/delta) |
|---|--:|--:|
| full resolution (production) | 1.00× | 2.73 |
| **f=2, nearest** | **1.82×** | 1.40 |
| f=2, bilinear | 2.05× | — |
| f=4, nearest | 2.36× | 1.15 |

Per-shape inflation ranged 1.43–2.75× at f=2. Read on its own this looks **affordable**: 1.82×
inflation is ~0.86 bits of delta precision for −60% of the apply kernel's bytes. Bilinear
upsampling is *worse* than nearest, which is the first hint that the lost content is not smooth.

(Note the full-res gain reads 2.73 here, not the calibration table's 12.45× median `step_gain_tail`.
They are different quantities — this is an instantaneous act/delta ratio at 10 steps, that is
tail-scale ÷ step0-scale at 200. Only the *ratios between arms* in this table are the measurement.)

## Step 2: the recursion, which is what actually decides it

The screen is single-step and therefore optimistic: the real scheme rebuilds a_hat **from the
downsampled copy** every step. [`scripts/ahat_ds_recursion.py`](scripts/ahat_ds_recursion.py)
captures the real per-step target for one layer (`input_blocks.4.0.in_conv`, 192×16×16, 30 steps of
a real generation) and replays both schemes on it, tracking the invariant residual

```
resid_t = max | a_hat_t  -  (a_hat_0 + Σ_i dequant(code_i)) |
```

which is exactly what `conv(a_hat) − o_hat` sees, since `o_hat` only ever adds
`conv(code) · dequant` and never sees a_hat's own storage error. Conv is linear, so the elementwise
form transfers (the argument `simulate_drift.py` already established).

| step | full (production) | **f=2** | f=4 |
|--:|--:|--:|--:|
| 1 | 0.00093 | 0.00093 | 0.00093 |
| 2 | 0.0018 | **2.14** | 3.00 |
| 5 | 0.0029 | **8.41** | 11.29 |
| 16 | 0.0060 | **24.02** | 27.93 |
| 30 | **0.011** | **44.25** | 48.73 |

**Production stays bounded at 0.011 (fp16 rounding). The downsampled arms grow LINEARLY, ~+1.5 per
step, reaching 44 by step 30** — against an activation whose own magnitude is O(1). Linear, not
√t: this is a systematic bias, not a random walk, so 200 steps extrapolates to ~300.

Two corollaries from the same run:

* `|target − o_hat's view|` tracks the residual exactly (43.8 at step 30 vs **0.0156** for
  production), i.e. o_hat's reconstruction of the activation is destroyed, not merely degraded.
* The delta's absmax inflation under recursion is **9.2×** (0.366 → 3.372), not the 1.82× the
  single-step screen measured. The reference is persistently wrong, so the delta must also encode
  the accumulated error.

**Mechanism.** `avg_pool → nearest-upsample` is a biased operator: each 2×2 block loses its
deviation from the block mean. The delta re-encodes that deviation, the accumulator keeps it, and
then a_hat is re-coarsened immediately and loses it again — every step, with the same sign. So the
accumulator and the stored reference separate at a constant rate.

## What this generalises to, and it predicts future attempts

a_hat is **not a free-standing cache**. It is one half of an accumulator pair with o_hat, and o_hat's
update only ever sees dequantized codes. **Any lossy transform of a_hat desynchronises the pair, and
the error accumulates rather than cancelling.** That single principle now covers every attempt on
record:

| transform of a_hat | lossy? | outcome |
|---|---|---|
| int8 / fp8 fixed-point storage | yes | refuted — FID 182–305 vs baseline 8.2 ([int8_ahat_cache](../int8_ahat_cache_2026-08-26/FINDINGS.md)) |
| companding / non-uniform 8-bit | yes | refuted — 8.56 bits needed under the most favourable assumptions |
| sparse encoding of zero codes | n/a (exact) | refuted on density, not correctness — 4.5% ceiling at any granularity ([ahat_zero_skip](../ahat_zero_skip_2026-08-26/FINDINGS.md)) |
| **spatial downsampling** | **yes** | **refuted here — linear divergence, 44 by step 30** |
| deferred write + exact reconstruction | **no (bit-exact)** | the only one that reached the timing stage — and lost there ([ahat_skip2_exact](../ahat_skip2_exact_2026-08-26/FINDINGS.md)) |

So the line is closed for a reason, not by bad luck: **only lossless transforms can preserve the
invariant, and the one lossless transform available is measured negative on time.**

One obvious rescue, blocked in advance: keeping the pair in sync by also adding `conv(e_t)` to o_hat
each step would restore the invariant, but computing it costs a full extra conv per layer per step —
far more than the traffic it buys back.

## What this does NOT close

**Selective per-layer MoDiff.** Dropping MoDiff on a layer removes the a_hat read, the a_hat write
*and* o_hat together, and it does not transform a_hat at all, so this principle does not apply to it.
[`ahat_overlap`](../ahat_overlap_2026-08-26/FINDINGS.md) already sized it at **3.551 ms/step** on the
five dominant shapes (W8A8) — 1.75× the write-only ceiling — and explicitly noted it is not bounded
by that ceiling. What it lacked was a criterion for *which* layers. The int4 delta table generated on
2026-08-27 supplies one:

| | layers with `step_gain_tail < 1` (MoDiff's delta needs a COARSER grid ⇒ buys no precision) | `< 2` | median gain |
|---|--:|--:|--:|
| int8 (committed table) | 2/70 = **3%** | 6% | 12.45× |
| int4 (generated 2026-08-27) | **57/70 = 81%** | 94% | **0.71×** |

At W4A4, MoDiff provably buys nothing on 81% of layers (Theorem 4.3: error is unchanged at gain 1
and strictly worse below it, leaving only error feedback) while paying a_hat read + a_hat write +
o_hat on all of them. That is the live lever in this direction, and it is a *deletion*, not a
transform.

## Scope and limitations

- **One layer, 30 steps, batch 2** for the recursion; 15 shapes, 10 steps, batch 4 for the screen.
  The divergence is linear and 4000× production's residual by step 30, so more layers would sharpen
  the number rather than change the verdict — but the exact rate is that layer's.
- **`avg_pool` + `nearest`/`bilinear` only.** A smarter reconstruction (learned upsampler, stored
  high-frequency residual) would reduce `e_t` but not make it zero, and anything that stores the
  residual is no longer a downsampled a_hat — it is a different, larger representation.
- **The screen's fp32 reconstruction of `silu(gn(x))`** is not bit-identical to the fused kernel's
  fp16 path. It cannot be: the target only exists inside the kernel. Both arms use the same
  reconstruction, so the inflation *ratios* are unaffected.
- **The int4 gain table backing the selective-MoDiff sizing is live-absmax calibration**
  (20 steps / batch 4 / 6 rounds), not the paper's Q-Diffusion protocol. The 81% figure should be
  re-checked if a real `int4_delta_qdiff.pt` is ever produced, though median 0.71 is a large enough
  signal that the protocol is unlikely to explain it.

## Files

- [`scripts/ahat_downsample_screen.py`](scripts/ahat_downsample_screen.py) — single-step inflation on real tensors, 15 shapes
- [`scripts/ahat_ds_recursion.py`](scripts/ahat_ds_recursion.py) — the recursion replay that decides it
