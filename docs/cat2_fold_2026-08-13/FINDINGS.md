# Folding the decoder skip-concat into the conv prologue: feasible, worth 1.5–2%

`cat2_channels_last_fp16_kernel` is **392 ms of the 12106 ms W4A4 window (3.2%)** over 3000 launches,
and it is pure data movement. Unlike the GroupNorm stats pass — which turned out to run at 57–70% of
peak and therefore had no headroom at all — a fold here **removes the traffic** rather than doing it
faster. So the ceiling is the whole 3.2%.

| | end to end |
|---|--:|
| ceiling, cat2 removed entirely | **3.24%** |
| realistic, after the split's cost | **2.01%** |
| realistic, dropping non-credible rows | **1.45%** |

**Verdict: worth scoping.** The first item in this session's sweep that is neither a dead end nor
already optimal.

## The obstacle: two consumers, not one

Read out of the code before measuring anything. Inside the fused ResBlock the concatenated tensor is
consumed **twice**:

| site | consumer |
|---|---|
| [`fused_resblock.py:844`](../../integration/fused_ops/fused_resblock.py) | `_prequant_gn_conv(x, fused_in_norm_silu, in_conv)` — the GN+SiLU+quantize prologue |
| [`fused_resblock.py:863`](../../integration/fused_ops/fused_resblock.py) | `self.skip_connection(x)` — a 1×1 conv, or `Identity` whose output *is* x and is still read as the out-conv's residual |

So **folding into the prologue alone removes nothing** — the skip path would still need the
materialised tensor. Both consumers have to accept a split input. Both splits are exact:

* **GN prologue — no arithmetic risk.** GroupNorm over a channel range does not care which buffer a
  channel lives in, and in the chan-major layout a thread owns one channel for the entire kernel, so
  "which pointer" is decided **once per thread**. Of the 9 real shapes, 5 are group-aligned
  (`C1 % CPG == 0`); the other 4 have one group straddling the boundary, which is fiddlier index math
  but nothing more.
* **Skip conv** — `W·[a;b] = W1·a + W2·b`, exactly. This is the half that costs something.

## The real shapes, probed rather than derived

Recorded by wrapping `_skip_concat` on a live sampling run:

| C1 | C2 | C | H×W | calls | CPG | straddles? | cat2 µs | GB/s | % peak |
|--:|--:|--:|---|--:|--:|---|--:|--:|--:|
| 768 | 768 | 1536 | 2×2 | 21 | 48 | no | 17.9 | 180 | 26% |
| 768 | 768 | 1536 | 4×4 | 14 | 48 | no | 30.5 | 414 | 60% |
| 384 | 384 | 768 | 8×8 | 14 | 24 | no | 51.3 | 493 | 71% |
| 384 | 384 | 768 | 16×16 | 14 | 24 | no | 183.5 | 549 | 79% |
| 192 | 192 | 384 | 32×32 | 14 | 12 | no | 357.8 | 561 | **81%** |
| 768 | 384 | 1152 | 4×4 | 7 | 36 | YES | 25.4 | 375 | 54% |
| 768 | 384 | 1152 | 8×8 | 7 | 36 | YES | 74.0 | 510 | 73% |
| 384 | 192 | 576 | 16×16 | 7 | 18 | YES | 139.8 | 540 | 78% |
| 384 | 192 | 576 | 32×32 | 7 | 18 | YES | 533.0 | 567 | **81%** |

At 81% of peak on the shapes that dominate, cat2 is already as fast as a copy can be — which is
exactly why the only way to win here is to not do it.

## Three corrections to my own measurement

Each changed the answer, and two of them would have produced a confidently wrong verdict.

**1. The split was priced 6× too pessimistically.** The first version timed it as
`f(a,wa) + f(b,wb)`, which materialises **two** full outputs and reads both back to add them. A real
implementation has the second half **accumulate in place** (β = 1) — which this codebase already does
for `o_hat`. A 1×1 conv is a GEMM, so both are timed as GEMMs where β = 1 is expressible (`addmm_`):

| | net saving |
|---|--:|
| naive `+` | 12% of cat2 → 0.38% end to end |
| **in-place accumulate** | **62% of cat2 → 2.01% end to end** |

The naive number would have killed the idea on an artifact of how I wrote the benchmark.

**2. `Cout` was assumed.** I used `Cout = C2`. Probed off `output_blocks.*.skip_connection.weight`,
that is right for the symmetric shapes and wrong for the asymmetric ones — C=1152 also maps to 768,
C=576 also maps to 384. Larger `Cout` makes the split *more* expensive, so the assumption was
optimistic. Both variants are now measured, with the concat count split between them.

**3. The verdict threshold could not fail.** It compared a *fraction* (0.0145) against `1.0` instead
of `0.01`, so the "below 1% end to end" branch was reachable for every possible input — it printed
that verdict even at 1.45%. Same shape as the divide-by-zero `"CONSISTENT"` in
`bench_gn_stats_roofline.py`: **a gate whose passing condition is unfalsifiable is not a gate.** That
is twice in one day, in two different scripts, both written by me.

## What is still soft

Two rows time **one GEMM slower than two** (`768|384|Cout=768|8×8` and `384|192|Cout=192|32×32`), which
cannot be a property of the split — the same math in fewer launches. It is a cuBLAS heuristic artifact;
the conv-based timing of one of those shapes disagreed with the GEMM timing by nearly 2×. They carry
**28%** of the saving, which is why the answer is quoted as a range (1.45–2.01%) rather than as 2.01%.

The counts come from one sampling pass and are per-pass, not per-step, so they weight the shapes
relative to each other correctly but are not a per-step figure on their own. The end-to-end percentages
are anchored to the profiled 392 ms instead, using the microbenchmark only as a **ratio**, so a
systematic offset between CUDA-event timing here and profiler self-time there cannot inflate the
result.

## If it gets built

Sequence matters: **start with the two-input GN prologue.** It carries no arithmetic risk and is
independently testable. But the saving only materialises once *both* halves land — the prologue alone
removes nothing, because the skip conv still forces the concat. Anyone starting this should know they
are committing to both kernels, for ~1.5–2%.

## Reproducing

```bash
python integration/tests/bench_cat2_fold.py
```
