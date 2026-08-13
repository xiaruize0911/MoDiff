# GN stats in the conv epilogue: the prototype fails the speed gate by 3.6×, and the shipped kernel has no headroom

**This file retracts and replaces an earlier version of itself committed the same day
(`a67d833`).** That version claimed the shipped stats pass runs at 15–19% of the A40's bandwidth and
is 6.2× off its roofline, so ~3.8% of end-to-end was available by tuning it. **Both claims were
wrong**, and they were wrong because they were built on an inherited number instead of a measured one.
The correction is below, and the mistake is described at the end because it is the more useful part.

## 1. The shipped stats pass is already near memory-bound optimal

Measured 2026-08-13 with `integration/tests/bench_gn_stats_roofline.py --profile`, driving the real
two-pass MoDiff entry point (`group_norm_silu_delta_quantize_pack_nhwc`) and isolating each kernel by
CUPTI self-time. The stats pass reads X exactly once and writes a tiny `[N,G,nblocks]` buffer, so one
read is its whole traffic and the percentage is honest rather than an upper bound:

| C | H×W | n | stats µs | combine µs | apply µs | stats GB/s | % of 696 GB/s |
|---|---|--:|--:|--:|--:|--:|--:|
| 192 | 32×32 | 14 | 113.3 | 8.2 | 285.2 | 444 | **64%** |
| 384 | 32×32 | 4 | 206.8 | 8.3 | 568.3 | 487 | **70%** |
| 768 | 16×16 | 4 | 126.8 | 8.1 | 284.1 | 397 | **57%** |
| 768 | 4×4 | 10 | 21.9 | 3.2 | 18.9 | 144 | 21% |

**57–70% of peak on every shape that carries weight.** For a kernel doing a strided reduction that is
close to what the hardware allows; the only poor shape is 768×4×4, a 3 MiB tensor where the launch
dominates. Count-weighted the stats pass is **3.14 ms** and the full GN op **10.94 ms**, so stats is
**29%** of it.

So there is **no meaningful headroom in this kernel**. Its comment block claims it is coalesced,
atomic-free and deterministic, and the bandwidth number now confirms the claim rather than merely
restating it.

## 2. Therefore the prototype fails the speed gate — by 3.6×, not 0.96×

`bench_gn_stats_tiles.py` grades `gn_stats_from_tiles` against a "shipped µs" column **inherited from
the 2026-08-11 report**, because the stats kernel has no pybind entry of its own. That column reads
**11.94 ms** weighted. The real stats pass is **3.14 ms** — the inherited column is **3.8× too
large**, and it is not the stats pass at all (it is close to this measurement's *full* GN op, 10.94 ms,
which suggests it was the whole fused operation).

Correcting the baseline:

| | weighted |
|---|--:|
| shipped stats pass, measured | **3.14 ms** |
| `gn_stats_from_tiles` warp-tree prototype | 11.43 ms → **3.64× SLOWER** |
| the 2026-08-11 atomics prototype | 20.83 ms → 6.63× slower |

**The prototype does not pass its speed gate. It loses by 3.6×.** What survives from the earlier
re-measurement is only the other two gates, which are real and were the point of the rewrite:
correctness (max rel err 5.7e-07 against an fp32 reference) and **determinism on every shape**, which
the atomics version failed outright. The rewrite fixed what it set out to fix. It just does not make
the idea viable.

And the direction is dead for a structural reason, not a tuning one: the pass being replaced already
moves its bytes at 57–70% of peak, so there is nothing for a cleverer reduction to recover. Folding the
stats into the conv epilogue would remove a read that costs 3.14 ms weighted (**about 6.7% of the W4A4
window**, since stats+combine is 810 of 12106 ms) — but only if the epilogue's own reduction were free,
and the standalone measurement says that reduction alone costs 11.43 ms.

## 3. Why GroupNorm is only 1.19× faster than fp16, and why that is not a bug

The GN family is 29.8% of the W4A4 run at 1.19× vs fp16, which looks like the largest remaining
Amdahl term and was described that way earlier in the session. It is not an inefficiency. GroupNorm's
input is **fp16 in every mode** — quantization shrinks the GEMM operands, not the normalisation's
traffic — so a memory-bound pass over X costs the same at W4A4 as at fp16 by construction. Running at
57–70% of peak *is* the answer. **There is no double-digit win here, and there is no small one either.**

## 4. The mistake, which is the transferable part

The earlier version of this file computed a roofline from a number it did not measure. Three separate
guards should have caught it and each was skipped:

* **The inherited column was flagged as inherited and then used anyway.** That version's own "What this
  does not say" section states that the shipped µs column is quoted rather than re-measured, and calls
  it "unlikely to be wrong by the 6× that would change the conclusion". It was wrong by 3.8×, which
  was enough. Naming a caveat is not the same as respecting it.
* **An arithmetic cross-check was available and free.** The full GN op measured 12.52 ms against an
  inherited stats-only 11.94 ms, which would leave apply+combine at 0.58 ms while moving *more* bytes
  than stats. That is impossible on its face, and it was visible in the first table printed.
* **The first driver measured the wrong kernel entirely and reported a passing verdict.** It drove
  `group_norm_silu_nhwc`, a single fused kernel that never launches the stats kernel, so stats came out
  at 0.0 µs — and the comparison then divided 11.94 by zero and printed **"CONSISTENT"**. An instrument
  that reports agreement when it has no data is worse than no instrument. It now refuses.

The pattern across all three: a plausible chain of reasoning was allowed to stand in for a
measurement, and every check that would have broken the chain was itself reasoned about rather than
run.

## Reproducing

```bash
# the correction: isolates stats / combine / apply on the real two-pass path
python integration/tests/bench_gn_stats_roofline.py --batch 128 --profile

# the prototype's three gates (its "shipped us" column is the wrong baseline -- see above)
python integration/tests/bench_gn_stats_tiles.py --batch 128
```

Nsight Compute would have isolated the kernel directly, which is what this should have used from the
start; `ncu` is installed but returns `ERR_NVGPUCTRPERM` on this host (the driver restricts performance
counters to admin), so the isolation is done with CUPTI activity tracing through `torch.profiler`
instead, which needs no special permission.
