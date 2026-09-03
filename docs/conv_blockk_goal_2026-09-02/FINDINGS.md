# Blockwise conv at >=80% of shipped speed: per-configuration benchmark

GOAL: under blockwise activation quantization, W8A8 and W4A4 at >=80% of the shipped per-tensor
speed; sweep K and the other tile parameters; report every configuration.

A40, batch 128, frequency-weighted over the 20 churches-UNet conv shapes (total frequency 62),
CUDA events. 24 configurations in `conv2d_blockk_tune.cu`'s table, both precisions.
`tune_check.py` (correctness gate) then `tune_sweep.py` -> `data/tune_sweep.json`.

## Correctness gate first

6 of 24 configs FAIL for int8 and 15 of 24 for int4, so they are excluded rather than reported
as fast:

- **Every STAGES>=3 config fails** (cfg 1, 10, 14, 21, 22, 23): relL2 0.10-0.24 against a
  2.1e-04 floor. STAGES>=3 is a correctness bug in the tune kernel, not merely slow -- which is
  consistent with, and stronger than, the "STAGES=3 rejected" note in the production kernel.
- int4 additionally rejects every TK=128B config on C=384 (`C must be a multiple of 256`) and
  cfg9 (`int4 needs blk >= 64`).

## The metric that matters: deployable = blockwise where eligible + shipped fallback elsewhere

A config with large BLK/TK is eligible on fewer shapes, so its own weighted total is not
comparable. The deployable time is `blockk(covered) + shipped(uncovered)`.

### W8A8 (shipped total over all shapes = 21.780 ms)

| cfg | config | cov | blockk(cov) | mixed total | % of shipped | |
|---|---|---|---|---|---|---|
| **8** | **M64N128K128_W64x16_S2_B256** | 47% | 8.769 | 25.151 | **86.6%** | ✓ |
| **7** | **M128N64K128_W64x16_S2_B256** | 47% | 9.981 | 26.363 | **82.6%** | ✓ |
| 5 | M128N64K128_W32x32_S2_B128 | 82% | 21.895 | 29.896 | 72.9% | |
| 4 | M64N128K128_W64x16_S2_B128 | 82% | 23.962 | 31.963 | 68.1% | |
| 3 | M128N64K128_W64x16_S2_B128 | 82% | 26.394 | 34.395 | 63.3% | |
| 0 | M128N128K64_W128x16_S2_B64 | 100% | 34.471 | 34.471 | 63.2% | |
| 11 | M64N64K128_W32x16_S2_B128 | 82% | 28.525 | 36.527 | 59.6% | |
| 15 | M128N128K128_W64x32_S2_B256 | 47% | 21.512 | 37.893 | 57.5% | |
| 2 | M128N128K64_W64x32_S2_B64 | 100% | 38.483 | 38.483 | 56.6% | |
| 9 | M128N128K64_W128x16_S2_B32 | 100% | 44.448 | 44.448 | 49.0% | |
| 12 | M128N128K128_W64x32_S2_B128 | 82% | 37.897 | 45.898 | 47.5% | |
| 18 | M128N128K128_W64x32_SA2SB1_B256 | 47% | 29.117 | 45.499 | 47.9% | |
| 17 | M128N128K128_W64x32_SA2SB1_B128 | 82% | 121.328 | 129.329 | 16.8% | |
| 16 | M128N128K128_W128x16_SA2SB1_B128 | 82% | 125.976 | 133.977 | 16.3% | |
| 13 | M128N128K128_W128x16_S1_B128 | 82% | 129.799 | 137.800 | 15.8% | |
| 19 | M128N128K32_W128x16_S2_B64 | 100% | 142.485 | 142.485 | 15.3% | |
| 20 | M128N128K32_W64x32_S2_B64 | 100% | 160.175 | 160.175 | 13.6% | |
| 6 | M128N128K64_W128x16_S2_B128 | 82% | 158.101 | 166.102 | 13.1% | |

### W4A4 (shipped total = 12.188 ms)

| cfg | config | cov | blockk(cov) | mixed total | % of shipped | |
|---|---|---|---|---|---|---|
| **8** | **M64N128K128_W64x16_S2_B256** | 47% | 4.695 | 13.840 | **88.1%** | ✓ |
| **7** | **M128N64K128_W64x16_S2_B256** | 47% | 4.844 | 13.989 | **87.1%** | ✓ |
| 6 | M128N128K64_W128x16_S2_B128 | 82% | 12.870 | 17.212 | 70.8% | |
| 15 | M128N128K128_W64x32_S2_B256 | 47% | 10.512 | 19.657 | 62.0% | |
| 0 | M128N128K64_W128x16_S2_B64 | 82% | 16.066 | 20.408 | 59.7% | |
| 2 | M128N128K64_W64x32_S2_B64 | 82% | 17.894 | 22.236 | 54.8% | |
| 20 | M128N128K32_W64x32_S2_B64 | **100%** | 25.876 | 25.876 | 47.1% | |
| 19 | M128N128K32_W128x16_S2_B64 | **100%** | 27.166 | 27.166 | 44.9% | |
| 18 | M128N128K128_W64x32_SA2SB1_B256 | 47% | 17.905 | 27.050 | 45.1% | |

## Result: the goal is met at both precisions, by the same configuration

**cfg8 `M64N128K128_W64x16_S2_B256`: W8A8 86.6%, W4A4 88.1% of shipped.** cfg7 also clears at
82.6% / 87.1%. Two of 18 (int8) and two of 9 (int4) surviving configs clear 80%.

## The TK=32 route is answered, and it is the wrong direction

TK=32 was added specifically to remove the W4A4 fallback: at 32 bytes/row an int4 tile is 64
elements, needing only `C % 64 == 0`, and all six churches channel counts (192/384/576/768/1152/
1536) satisfy that. It works -- **cfg19/20 are the only int4 configs with 100% shape coverage.**

But it costs more than the fallback it removes: 44.9-47.1% versus 88.1% for cfg8 with fallback.
Universal blockwise is a **1.9x net loss**. Before the sweep I expected TK=32 to unlock the goal;
measured, it unlocks coverage and loses the speed twice over. Closed.

The pattern is monotone in BLK: B=256 configs are the fast ones, B=128 middling, B=64/B=32 poor,
because #flush = K_g/BLK per accumulator -- coarser blocks flush less. Coverage moves the opposite
way (larger BLK needs a larger C multiple), so **coverage and speed trade off directly along BLK,
and the optimum is coarse BLK plus fallback.**

## What is actually binding: the tile, not blockwise

Splitting cfg8's gap with the scalar-alpha control (same kernel, same tile, dequant off):

| | W8A8 | W4A4 |
|---|---|---|
| blockwise vs shipped | 1.624x | 1.543x |
| our tile at scalar alpha vs shipped | 1.546x | 1.414x |
| **blockwise tax over our own tile** | **1.051x** | **1.091x** |

The blockwise dequant costs **5-9%**. The other 1.41-1.55x is our tile losing to CUTLASS. So the
blockwise scheme itself is essentially free at B=256; every remaining point of the gap is tile
engineering, which is where further work belongs.

## Caveats

- These are conv-kernel-only numbers. The E2E figures recorded earlier (W8A8 82.2%, W4A4 86.8%
  with fallback / 75.4% forced-blockwise) use ms/step as the denominator, where the conv is one
  part of the step -- consistent in direction with the table above, different denominator.
- Fallback means the ineligible layers keep the **per-tensor** activation quantizer, so the
  accuracy benefit of blockwise is partial, not model-wide. 47% coverage by frequency is a real
  quality/speed trade, not a free lunch, and the accuracy work in
  `docs/ahat_accuracy_2026-09-02` does not cover the mixed configuration.
- `conv2d_blockk_tune.cu` is the sweep kernel, not the shipped one. Adopting cfg8 means porting
  its tile (M64 N128 K128, warp 64x16, 2 stages, BLK=256) into `conv2d_int8_blockk.cu` /
  `conv2d_int4_blockk.cu`, which currently run a different tile.
