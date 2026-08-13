# GN stats in the conv epilogue: the rewrite passes all three gates — and targets the wrong cost

Two results, and the second one matters more than the first.

## 1. The warp-tree rewrite passes. All three gates.

The 2026-08-11 prototype failed on two of three: 6.5× too slow and non-deterministic on every shape,
both from 23–56 shared-memory slots with 256 threads contending on `atomicAdd`. The kernel was
rewritten on 2026-08-12 to a segmented warp reduction (`__match_any_sync` groups the lanes sharing a
slot, a masked butterfly sums them, one leader per group writes to its warp's private slots, warps
combined in a fixed `w = 0..WARPS-1` order — no atomics anywhere). **That rewrite was never
re-measured.** Measured 2026-08-13, A40, batch 128
([`data/tree_prototype_2026-08-13.json`](data/tree_prototype_2026-08-13.json)):

| C | H×W | n | tree µs | shipped µs | atomics µs | tree/shipped | max rel err | det |
|---|---|--:|--:|--:|--:|--:|--:|---|
| 192 | 32×32 | 14 | 423.9 | 476.2 | 542.7 | **0.89×** | 5.6e-07 | ok |
| 384 | 32×32 | 4 | 768.9 | 771.5 | 1439.0 | 1.00× | 4.0e-07 | ok |
| 768 | 16×16 | 4 | 404.4 | 416.6 | 1277.3 | 0.97× | 5.7e-07 | ok |
| 768 | 4×4 | 10 | 80.2 | 52.5 | 237.0 | 1.53× | 3.0e-07 | ok |

Count-weighted: **tree 11.43 ms, shipped 11.94 ms — 0.96×**, and 1.82× better than the atomics
version. Determinism holds on every shape, which is the gate the rewrite was for. The predicted
failure mode survives only where it was predicted to: 768×4×4 is 1.53× worse, the shape with the
fewest slots and therefore the most contention per slot.

So the engineering worked. **But 0.96× is a warning, not a green light** — a reduction that merely
ties the entire pass it is supposed to be folded into for free is not cheap.

## 2. The pass is not bandwidth bound, so fusing it away saves ~16%, not 100%

The whole premise of putting stats in the epilogue is to avoid reading X a second time. That is only
worth something if the read is the cost. It is not:

| shape | X | shipped µs | achieved | % of A40 700 GB/s |
|---|--:|--:|--:|--:|
| C192 32×32 | 48 MiB | 476.2 | 106 GB/s | **15%** |
| C384 32×32 | 96 MiB | 771.5 | 130 GB/s | **19%** |
| C768 16×16 | 48 MiB | 416.6 | 121 GB/s | **17%** |
| C768 4×4 | 3 MiB | 52.5 | 60 GB/s | **9%** |

At the roofline the weighted pass would cost **1.91 ms**; it costs **11.94 ms** — **6.2× off**. So at
most ~16% of it is the read itself, and **~84% is the reduction and per-launch structure**.

That reframes the whole direction:

* **Fusing the stats into the conv epilogue removes only the read.** Ceiling ≈ 16% of the pass. In the
  full W4A4 run the stats pass is 689 ms of a 12106 ms window (5.7%), so the fusion is worth about
  **0.9% of end-to-end** — for an EVT epilogue change, a new auxiliary output, and a reduction that
  already loses 1.53× on one of the four shapes.
* **A fused stats+apply GroupNorm** (one CTA per (n, group), keep X in shared memory) has the same
  ceiling for the same reason, though it is feasible on smem grounds: CPG×HW×2 B is 12 KB at C=192,
  24 KB at C=384, and 96 KB at the decoder's concatenated C=1536 — the last one right at the A40's
  100 KB opt-in limit.
* **The actual prize is that the existing kernel is 6.2× off its own roofline.** Getting the stats pass
  to even half of roofline would take 11.94 → ~4 ms, i.e. 689 → ~230 ms of window, about **3.8% of
  end-to-end** — W4A4 from 1.749× to roughly 1.82×. Four times the fusion's ceiling, and it needs no
  EVT work, no new auxiliary output, and no epilogue contract.

**Recommendation: do not wire the epilogue prototype. Profile why `gn_stats_partials_chanmajor` runs at
15–19% of roofline first.** The kernel is already coalesced, atomic-free and deterministic — the
comment block documents all of that carefully — so the deficit is somewhere else (block shape is
C/K threads, i.e. only 6 warps at C=192; the partials round-trip through global memory; and 11 000
launches at these small shapes). That is a measurement to make, not a design to argue about.

## What this does not say

The per-shape "shipped µs" column is quoted from the 2026-08-11 report rather than re-measured here —
`gn_stats_partials_chanmajor_kernel` has no pybind entry, so the harness cannot time it directly. The
roofline arithmetic above therefore inherits that column. It is unlikely to be wrong by the 6× that
would change the conclusion, but the honest statement is that the ratio is apples-to-apples (both
columns weighted the same way) while the absolute shipped numbers are inherited.

Whether the tree kernel's own 11.43 ms would shrink inside a real epilogue is also not measured: there
its fragment is already in registers, so it would skip the load but still pay the reduction. That is
the ~84% term, which is exactly why the fusion's ceiling is low.

## Reproducing

```bash
python integration/tests/bench_gn_stats_tiles.py --batch 128 \
  --json docs/gn_stats_in_epilogue_2026-08-11/data/tree_prototype_2026-08-13.json
```
