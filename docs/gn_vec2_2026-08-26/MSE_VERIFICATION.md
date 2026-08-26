# MSE verification against an fp64 reference

**Date** 2026-08-26 · **GPU** NVIDIA A40 · **Batch** 8

Every gate committed so far under `gn_vec2_2026-08-26` is a **bit-identity** check against the
OLD kernel a change replaces (`torch.equal`). That proves vec2 changed nothing relative to what
shipped before -- it never checks that the thing being preserved was numerically right in the
first place. [`verify_mse.py`](scripts/verify_mse.py) closes that gap: it recomputes GroupNorm +
SiLU + delta-quantize + the a_hat update in float64 pure PyTorch and scores the real CUDA
kernels (scalar and vec2, wherever both exist) against it with MSE and max-abs-error -- the same
"vs fp64" convention [`test_cat2_gn_fold.py`](../../integration/tests/test_cat2_gn_fold.py)
already uses in this repo.

## Part A -- GN stats (mean, inv_std)

Both the scalar and vec2 chanmajor kernels agree with an independent float64 group mean/variance
to **machine precision for float32**: MSE 3e-19 to 1e-18 on the mean, 4e-16 to 1e-15 on
`inv_std`, max-abs-error 2e-9 to 1e-8. Scalar and vec2 read identically on every shape (expected
-- this is what the bit-identity gate already established); the new information here is that
*both* are correct against ground truth, not merely consistent with each other.

## Part B -- the full flat delta-quantize pipeline (production kernel)

On 4 of 6 shapes, codes match an fp64 replay exactly. On the other 2, a handful of elements
(1-4 out of 0.6-4.7 million, ~0.0001-0.0007%) disagree by **exactly one quantization code** --
and every one of those elements has its raw pre-round value within **0.015** of an exact `.5`
boundary. This is floating-point rounding-boundary sensitivity: CUDA's single-precision
`expf`/`silu` and the host's double-precision `sigmoid` differ at the ~1e-4-to-1e-2 level right
at the input to `round()`, which is exactly where a decision this close to a coin-flip can land
on either side. Confirmed directly: `max |code diff|` is exactly **1** everywhere it is nonzero,
and `max dist to .5` never exceeds 0.016 -- a real bug would show larger, boundary-independent
disagreements instead. Scalar and vec2 disagree with the reference at the **identical elements,
identical count, identical direction** on every shape -- proof that vec2 changed nothing here
either, on top of the direct bit-identity gate.

## Part C -- the resize kernel's a_hat (UP + DOWN)

Same story, higher rate: 0.1-0.15% of codes disagree by exactly 1, all within 0.032 of a `.5`
boundary. The higher rate than Part B is expected, not investigated further as a discrepancy:
diagnosed directly (see `diag_resize.py` in the session transcript) by dumping the raw
pre-round values at every disagreeing element -- e.g. 16.4999, 12.5013, 28.5063, -2.5003,
6.5016, 120.5166 -- confirming every single one sits within 3% of an exact tie, none are a
wholesale formula error. The DOWN path averages four `compute_pair` results in float32 before
quantizing, one more accumulation step than the flat kernel's single value, which plausibly
explains a higher rate of near-tie deltas; ruled out as the cause of the discrepancy ITSELF
(not just its rate) by re-deriving the reference's group stats from the kernel's own float32
computation (via the Part A probe) rather than an idealized fp64 mean/var -- this changed the
mismatch count by at most 1 element out of millions, confirming the residual is not a stats-
precision artifact either.

## What this rules out, and what it does not

- **Rules out**: a structural formula error in either kernel (scalar or vec2), a rounding-
  convention bug in the reference (`torch.round`'s round-half-to-even vs CUDA `roundf`'s
  round-half-away-from-zero was tested directly as a hypothesis and made no measurable
  difference -- the residual predates and is independent of that distinction), and any
  divergence between the scalar and vec2 arms (they disagree with fp64 at the identical
  elements, which is only possible if they compute the identical value).
- **Does not, and cannot, rule out**: this script's own reference introducing a DIFFERENT tiny
  discrepancy than the kernel's true internal single-precision `expf` (host `torch.sigmoid` on a
  float64 tensor is not guaranteed to use bit-identical transcendental-function evaluation to
  CUDA's device `expf`). That is precisely the source of the near-tie sensitivity described
  above, and is inherent to comparing any two independent floating-point implementations near a
  quantization decision boundary -- not something a better reference eliminates, only relocates.

## Files

- [`scripts/verify_mse.py`](scripts/verify_mse.py) -- self-contained, prints all three parts

## Scope and limitations

- Batch 8 (not 128) and 4-6 representative shapes, not all 20/18 -- this is a numerics check,
  not a timing one, and batch/shape count do not change what is being verified.
- Part B's scalar arm uses `MODIFF_GN_STATS_VEC2=0` at runtime; Part C's fix has no such toggle
  (see [`FINDINGS.md`](FINDINGS.md)), so only the vec2 (current) build is checked there --
  already covered for bit-identity against the pre-fix build by
  [`test_gn_resize_ahat_vec2.py`](../../integration/tests/test_gn_resize_ahat_vec2.py).
- `mod_scale`/`mod_shift`/`smooth_inv` are all empty (untested) in this script, matching the
  static, unmodulated configuration the GN vec2 gates already cover.

