# Activation precision at fixed W8: A8 → A2, MoDiff on and off, against the paper

**2026-08-05 · A40 · LSUN-Churches LDM-8, real checkpoint · DDIM 50 · batch 8 · 3 seeds, paired ·
latent relL2 vs fp16**

The first sweep of activation precision in this project, and the first measurement of the
configuration the paper actually claims. Every shipped mode pairs the two precisions (W8A8, W4A4),
so every earlier quality number confounds weight error with activation error — and MoDiff only
addresses the second. Data: `data/act_bit_sweep.json`, `logs/sweep.log`.

## Result

Q is the symmetric code ceiling 2^(b-1)−1. Both columns move together: baseline = MoDiff off with the
calibrated per-tensor activation grid rescaled to b bits; MoDiff = dynamic Q/max|delta| delta grid at
b bits, t=T warm-up also at b bits.

| A bits | levels | baseline (PTQ) relL2 | MoDiff relL2 | baseline/MoDiff |
|---|---:|---:|---:|---:|
| A8 | 255 | 0.2565 ± 0.0172 | **0.0627 ± 0.0282** | 4.09× |
| A7 | 127 | 0.2578 ± 0.0491 | **0.0669 ± 0.0276** | 3.85× |
| A6 | 63 | 0.3184 ± 0.0429 | **0.0795 ± 0.0252** | 4.01× |
| A5 | 31 | 0.7356 ± 0.0197 | **0.1563 ± 0.0177** | 4.71× |
| A4 | 15 | 0.8071 ± 0.0378 | **0.3367 ± 0.0664** | 2.40× |
| A3 | 7 | 0.9313 ± 0.0341 | 0.8206 ± 0.0211 | 1.13× |
| A2 | 3 | **0.9366 ± 0.0611** | 1.3308 ± 0.0765 | 0.70× |

Read against the relL2 → FID anchors measured in `docs/fid_2026-08-05` (0.039 → FID 7.80, i.e. parity
with fp16; 0.238 → 16.4; 0.456 → 200; 0.784 → 278):

* **MoDiff holds to A6** — 0.063 → 0.080 from A8 to A6 is flat inside the seed spread.
* **The baseline is already broken at A8** (0.257 ≈ FID 16) and collapses between A6 and A5
  (0.318 → 0.736, and 0.784 is FID 278 territory).
* **MoDiff collapses between A5 and A4**, one to two bits below the baseline's cliff.
* **At A2 MoDiff is worse than not using it** (1.33 vs 0.94, 0.70×). This is the paper's own bound
  failing on its own terms: Theorem 4.4 assumes the quantization error is bounded by the input
  magnitude with a coefficient below 1/2. Three levels violates that, and the error-feedback term
  then amplifies instead of compensating.

ms/step is flat across every row (baseline 12.4–14.7, MoDiff 17.3–19.0, fp16 18.14) — the control
that this is a quality instrument only. A low A_b costs nothing and saves nothing: activations keep
their int8 container and the GEMM stays W8A8. MoDiff at 0.75× the baseline's speed at batch 8 also
reproduces the known batch-8 regime (0.69–0.73×, `docs/SESSION_2026-08-05.md`).

## The 0.127 in docs/fid_2026-08-05/FINDINGS.md was measuring A4 deltas with an A8 anchor

That report's W8A4+MoDiff row (relL2 0.127, "beats the W8A8 baseline 0.238 — the paper's claim in
substance") does not survive. Reproduced and attributed in `data/verify_vs_old_w8a4.json`:

| configuration | relL2 | per seed |
|---|---:|---|
| old: `MODIFF_DELTA_CLIP=127/7`, static grid left at A8 | 0.1626 ± 0.0330 | 0.162, 0.196, 0.130 |
| new: `MODIFF_ACT_Q=7`, every conv activation site at A4 | 0.3581 ± 0.0665 | 0.286, 0.372, 0.416 |

The old number reproduces (0.127 is that arm's bottom seed; it was a single-seed measurement), so
nothing was mismeasured — it measured a different configuration than its label. **The two arms differ
in exactly one thing: the static activation grid that quantizes t=T.** The delta grid is at 4 bits in
both, including in the plain `_forward_modulated` path, whose hardcoded `127.0` was fixed as part of
this work. So the whole 2.2× is the t=T anchor:

    at 4 activation bits, quantizing the t=T warm-up at A4 instead of A8 doubles the end-to-end
    latent error, even though it is 1 step in 50.

Which follows from what MoDiff is. t=T seeds â, every later step quantizes a difference *against* â,
and the error-feedback term carries the anchor's error forward instead of averaging it out. A faithful
A_b network quantizes t=T at b bits too, so the sweep above is the correct reading and the earlier row
was optimistic. Path instrumentation over the measured runs, for scale: `forward_gn_fused_modiff`
9114 calls / 62 layers, `_forward_modulated` 288 calls / 8 layers, `_forward_first_step` 210 calls /
70 layers.

**Consequence for the earlier conclusion:** W8A4+MoDiff (0.337–0.358) does *not* beat the W8A8
baseline (0.257) in this implementation. The claim that this project reproduces the paper's headline
"in substance" is withdrawn — see the next section for what does hold and why the gap is not MoDiff.

## Against the paper

The paper is arXiv 2506.22463 (ICML 2025). Its Table 2 is our exact model: LSUN-Church, LDM-8, W8,
200 sampling steps, FID over 50,000 images. `Q-Diff` is Q-Diffusion (static, MSE-calibrated),
`LCQ` is dynamic **channel-wise** min-max.

| A bits | Q-Diff | Q-Diff+MoDiff | LCQ | LCQ+MoDiff |
|---|---:|---:|---:|---:|
| 32 (fp act) | 4.03 | — | — | — |
| 8 | 4.24 | 3.85 | 4.02 | 3.99 |
| 6 | 55.13 | 5.43 | 4.50 | 3.89 |
| 4 | 355.85 | 3.97 | 198.37 | 34.02 |
| 3 | 367.51 | 5.40 | 341.62 | 12.05 |

Our kernels quantize activations **per-tensor**, not per-channel. The paper's per-tensor variant is
`LTQ` in Appendix D.2, reported on CIFAR-10 only (Table 14, W8):

| A bits | LTQ | LTQ+MoDiff |
|---|---:|---:|
| 8 | 4.19 | 4.21 |
| 6 | 9.93 | 4.00 |
| 4 | 306.06 | 28.19 |
| 2 | 457.25 | 186.04 |

with the paper's own explanation: *"the minimum activation bit-width achievable with LTQ is higher
than that of LCQ. This is because tensor-wise quantization operates on higher-dimensional data,
making accurate quantization more challenging."*

**Our sweep reproduces the shape of LTQ, not of LCQ.** Same three features, in the same places:
MoDiff at parity through A6; the baseline collapsing first and by a much larger factor; MoDiff itself
degrading badly at A4 (their FID 28.19 against a 4.24 reference is "far better than baseline, not
usable") and worse than useless at A2 (their 186.04; our 1.33, worse than the baseline).

So the honest summary is: **at per-tensor granularity MoDiff buys about two activation bits, and the
paper's headline A4/A3-on-Churches numbers are a property of per-channel dynamic quantization that
this implementation does not have.** The gap is quantizer granularity, not the method.

Two caveats on the comparison, both real:

* **Metric.** They report FID over 50k images at 200 steps; this is latent relL2 at 50 steps. relL2
  is badly nonlinear against FID (`docs/fid_2026-08-05`), so it supports "which is better and roughly
  where the cliff is", not "our A4 equals their FID 28".
* **Granularity, again.** Ours is per-tensor *symmetric* with SmoothQuant; their LTQ is per-tensor
  dynamic min-max (asymmetric). At low bit-widths asymmetric buys roughly half a bit on activations
  that are one-sided — and post-SiLU activations are.

## Scope of MODIFF_ACT_Q, which bounds every number above

`MODIFF_ACT_Q` reaches the quantized **conv** path (70 calibrated layers of 89 converted). It does
**not** reach:

* the 21 quantized attention blocks (`quantized_std_attention.py` hardcodes `lvl = 127.0` / `7.0`),
* the 42 quantized Linear layers (`wxax_linear.py`),
* `int4_optimized.py`'s plain modulated path, which still hardcodes `7.0` (irrelevant at W8, listed
  so the next person does not assume symmetry with int8).

Both arms therefore keep A8 attention and Linears, so the *comparison* is symmetric — but a row
labelled A4 is "A4 in the conv path", not a whole-network A4. A whole-network sweep would be strictly
worse than what is reported here.

One asymmetry favours the baseline: its static scale is Q_b/calibrated_range and the quantize kernels
clamp codes at ±127 rather than ±Q_b, so a baseline activation above its calibrated range keeps
resolution where a true b-bit quantizer saturates. The MoDiff arm's dynamic delta quantizer cannot
clip by construction. Both effects understate MoDiff's advantage, so the gains in the table are lower
bounds.

## Code changes

| file | change |
|---|---|
| `integration/kernels/int8_optimized.py` | new `MODIFF_ACT_Q` (default 127 = shipped A8, bit-identical). Applied at the delta quantizer's `Q_level` and in `set_static_scale` / `end_calibration` as a Q_b/127 rescale of the calibrated grid, so both the baseline path and MoDiff's t=T warm-up move together. Load-time only, no hot-path cost |
| `integration/kernels/int8_optimized.py` | `_forward_modulated`'s `step1_quantize_fprop` passed a literal `127.0` while the GN-fused paths honoured the knobs, so `MODIFF_DELTA_CLIP` was silently partial — 8 of 70 layers kept an 8-bit delta grid whatever it was set to |
| `docs/act_bits_2026-08-05/scripts/act_bit_sweep.py` | the sweep. `SWEEP_BITS` / `SWEEP_SEEDS` subset it |
| `docs/act_bits_2026-08-05/scripts/verify_vs_old_w8a4.py` | the old-vs-new attribution and the forward-path instrumentation |

Controls: A8 reproduces the shipped numbers (baseline 0.2565 against the documented 0.2378, MoDiff
0.0627 against 0.0393–0.068 — both within the seed spread), and `MODIFF_ACT_Q=127` leaves the scale
bit-identical.

## Open

1. **Per-channel dynamic activation quantization.** This is now the single highest-value accuracy
   item: it is what stands between this implementation and the paper's A4/A3 claims, and the paper
   measures ~6× FID between LCQ and LTQ at A4. Cost is real — a per-channel scale has to be folded
   into the CUTLASS epilogue rather than passed as one alpha.
2. **Keep t=T at A8 as a shipped policy.** It is 1 step in 50 and it is worth a factor of 2 at A4.
   The knob to express that (a separate anchor precision) does not exist yet; today `MODIFF_ACT_Q`
   moves both.
3. **FID for W8A5+MoDiff** (relL2 0.156) — the lowest activation precision that still looks usable
   here, and the only sweep row whose FID is not guessable from the anchors. `fid/fp16` still holds
   its 10k samples, so FID-vs-fp16 costs one 10k generation; FID-vs-real additionally needs the LSUN
   LMDB re-downloaded (`/workspace/lsun_dl` is empty) and `fid/real` re-exported.
4. Thread `MODIFF_ACT_Q` through attention and Linear for a whole-network sweep, if a whole-network
   claim is ever needed.
