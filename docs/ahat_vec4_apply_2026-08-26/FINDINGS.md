# A genuine vec4 test for the apply kernel: small but real, bit-identical, NOT landed

**Status: measured, positive, deliberately not built into csrc/.** The project owner decided the
win is too small to be worth the fallback-path engineering it would need. Recorded here so it is
not silently lost or re-discovered from scratch later.

## Why the earlier "ILP" test did not answer this question

An earlier test in this session (`gn_vec2_2026-08-26`'s apply-kernel ILP check, U ∈ {1,2,4,8}) gave
each thread `U` **grid-strided** pairs: `idx[u] = tid + u*stride` — separated by the full grid
stride, i.e. not adjacent in memory. That measured "does more independent, far-apart work per
thread help" (answer: no — U=1 won). It never tested a genuinely **wider, contiguous** load/store
instruction (e.g. one 64-bit `__half4`-equivalent load covering 4 adjacent elements, vs vec2's
32-bit `__half2`). This closes that gap.

## Structural precondition

A group of 4 contiguous elements must not straddle a GroupNorm group boundary, i.e. `CPG = C/G`
must be a multiple of 4. Checked against every real Cin this UNet uses (`G = 32`):

| C | 192 | 384 | 576 | 768 | 1152 | 1536 |
|---|--:|--:|--:|--:|--:|--:|
| CPG | 6 | 12 | 18 | 24 | 36 | 48 |
| vec4-safe | **no** | yes | **no** | yes | yes | yes |

C=192 and C=576 together are only 10 of ~62 real calls/step — a minority, but a fallback to vec2
would be needed for them, and none was built here.

## Correctness

[`scripts/probe_vec4.cu`](scripts/probe_vec4.cu) is bit-identical to the shipped vec2 kernel on
every checked shape — confirmed via `torch.equal` on both `a_hat` and the output codes, on 4
representative shapes.

## Result

[`scripts/bench_vec4_vs_vec2.py`](scripts/bench_vec4_vs_vec2.py), all 14 real vec4-eligible
shapes (50 of ~62 calls/step), 5 trials with rotated order:

| shape | freq | vec2 ms | vec4 ms | speedup |
|---|--:|--:|--:|--:|
| `768,2x2` | 12 | 0.0056 | 0.0045 | **1.24×** |
| `384,8x8` | 8 | 0.0419 | 0.0414 | 1.01× |
| `384,16x16` | 7 | 0.1583 | 0.1580 | 1.00× |
| `768,4x4` | 7 | 0.0208 | 0.0181 | **1.15×** |
| `1536,2x2` | 3 | 0.0082 | 0.0066 | **1.25×** |
| `768,8x8` | 2 | 0.0806 | 0.0803 | 1.00× |
| `1536,4x4` | 2 | 0.0421 | 0.0415 | 1.01× |
| `384,32x32` | 2 | 0.6235 | 0.6243 | 1.00× |
| `768,16x16` | 2 | 0.3118 | 0.3130 | 1.00× |
| `384,4x4` | 1 | 0.0083 | 0.0063 | **1.32×** |
| `1152,8x8` | 1 | 0.1202 | 0.1192 | 1.01× |
| `1152,4x4` | 1 | 0.0324 | 0.0318 | 1.02× |
| `768,8x8` | 1 | 0.0806 | 0.0804 | 1.00× |
| `384,32x32` | 1 | 0.6231 | 0.6234 | 1.00× |

**freq-weighted total: 4.662 → 4.617 ms/step, −1.0%, saving 0.045 ms/step.**

The pattern is clean: small, low-element-count, high-frequency shapes (`768,2x2`, `1536,2x2`,
`768,4x4`, `384,4x4`) gain 15–32% — these are launch/occupancy-bound, not bandwidth-bound, so a
wider single instruction (fewer, bigger transactions) genuinely helps. The large shapes
(`384,32x32`, `768,16x16`, `384,16x16`) show **zero** measurable change — they are already
bandwidth-bound near the kernel's known 84.5%-of-peak ceiling, and freq-weighting is dominated by
them, which is why the aggregate saving is small (0.045 ms/step ≈ 0.06% of the 77 ms steady step)
despite some individual shapes moving by a real double-digit percentage.

## Why this was not landed

- The saving is real and bit-identical, but two orders of magnitude smaller than the GN
  stats/resize vectorisation already landed this session (−1.29 ms/step).
- Shipping it needs a fallback path for C=192/576 (vec2, unchanged) plus the same gate/MSE
  discipline every other change in this session got — engineering cost disproportionate to a
  0.045 ms/step return.
- Explicitly deferred at the project owner's request, not abandoned for a technical reason — the
  probe and benchmark are complete and reusable if priorities change.

## Files

- [`scripts/probe_vec4.cu`](scripts/probe_vec4.cu) — the vec4 kernel, structurally restricted to
  `CPG % 4 == 0`
- [`scripts/build_probe_vec4.py`](scripts/build_probe_vec4.py) — builds it as a standalone
  extension
- [`scripts/bench_vec4_vs_vec2.py`](scripts/bench_vec4_vs_vec2.py) — bit-identity check + timing,
  depends on the vec2 probe built in `ahat_overlap_2026-08-26/scripts/`
