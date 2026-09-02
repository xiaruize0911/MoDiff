# Tile x B sweep: 80% of shipped is NOT reached. Best is 65% for both precisions, and the
# ceiling is a shared-memory trade-off between tile quality and the blockwise tax.

Goal: blockwise conv within 1.25x of the shipped per-tensor conv (>= 80% of its speed), for
W8A8 and W4A4, by sweeping K and the tile parameters. A40, batch 128, CUDA events, median of 20,
frequency-weighted over the 20 churches-UNet conv shapes.

`csrc/modiff/conv/conv2d_blockk_tune.cu` (16 configs), `scripts/tune_sweep.py`,
`scripts/tune_check.py` (correctness gate), `data/tune_sweep.json`, `plots/tune_tradeoff.png`.
**Every config in the tables below passed the correctness gate first** -- relL2 ~3.6e-04 against
an exact fp32 per-block reference whose fp16 store floor is 2.07e-04.

## Result: not met

| | best config | blockwise vs fp16 | **% of shipped** | tax | tile alone |
|---|---|---:|---:|---:|---:|
| **W8A8** | cfg5 `M128N64K128_W32x32_S2_B128` | 1.152x | **65.2%** | 1.09x | 71.0% |
| **W4A4** | cfg7 `M128N64K128_W64x16_S2_B256` | 2.052x | **65.1%** | 1.13x | 73.7% |

Starting point was 1.72x / 1.84x tax, i.e. 58% / 54% of shipped, so the sweep gained ~7-11
points. It did not reach 80%.

## What the sweep established

**The tax prediction was right and the tax target is MET.** Predicted 1 + (3/B)*(tensor_rate/
fp32_rate): B=128 -> 1.19x (int8), B=256 -> 1.19x (int4). Measured, with `CTA_K` set so a block
does not span tiles: **1.03-1.14x**. Large B works as soon as the cross-tile carry is removed --
the earlier "large B is slower" result was entirely the carry, not the ALU.

**The binding constraint moved to the tile.** No config gets both. Solid evidence:

| | best tile | its tax | best tax | its tile | product if they co-existed |
|---|---:|---:|---:|---:|---:|
| int8 | 84.4% (cfg14, K=64) | 1.60x | 1.03x (cfg8, K=128) | 65.8% | **81.7%** |
| int4 | 82.5% (cfg1, K=64) | 1.98x | 1.10x (cfg8, K=128) | 71.8% | **74.7%** |

**The trade is mediated by shared memory, which is why it cannot be broken by parameters.**
Removing the carry needs `CTA_K*EPB == BLK`, so K=128 bytes. smem is
`STAGES*(CTA_M+CTA_N)*CTA_K`, and 2 CTA/SM on an A40 needs <= 50688 B, so K=128 forces
`CTA_M+CTA_N <= 192` -- one of the two must drop to 64. That smaller tile is exactly what costs
tile efficiency (more CTAs, more A re-reads, worse register blocking). Keeping the full 128x128
tile at K=128 is cfg12: smem 66560 B -> **1 CTA/SM**, tile 59.5%. Buying the occupancy back with
STAGES=1 is cfg13: no pipelining, tax 5.40x.

So for **int4 the goal is unreachable in this design space** -- even the impossible combination
of the best tile and the lowest tax is 74.7%. For **int8 it is marginally reachable in principle
(81.7%)** but only by a config that does not exist: one with K=64's tile and K=128's tax.

## Every config

W8A8 (`cov` = share of conv frequency the config is eligible for; `tax` = blockwise / same-tile
scalar; `tile alone` = the scalar control as a % of shipped speed):

| cfg | config | cov | blockwise vs fp16 | % of shipped | tax | tile alone |
|---:|---|---:|---:|---:|---:|---:|
| 5 | M128N64K128_W32x32_S2_B128 | 82% | 1.152x | **65.2%** | 1.09x | 71.0% |
| 0 | M128N128K64_W128x16_S2_B64 | 100% | 1.015x | 64.6% | 1.26x | 81.4% |
| 8 | M64N128K128_W64x16_S2_B256 | 47% | 1.103x | 63.7% | 1.03x | 65.8% |
| 2 | M128N128K64_W64x32_S2_B64 | 100% | 0.947x | 60.3% | 1.35x | 81.4% |
| 7 | M128N64K128_W64x16_S2_B256 | 47% | 1.021x | 59.0% | 1.14x | 67.1% |
| 4 | M64N128K128_W64x16_S2_B128 | 82% | 1.028x | 58.2% | 1.11x | 64.6% |
| 3 | M128N64K128_W64x16_S2_B128 | 82% | 0.946x | 53.6% | 1.27x | 67.8% |
| 14 | M128N128K64_W64x32_S3_B64 | 100% | 0.826x | 52.6% | 1.60x | **84.4%** |
| 9 | M128N128K64_W128x16_S2_B32 | 100% | 0.812x | 51.7% | 1.57x | 81.2% |
| 1 | M128N128K64_W128x16_S3_B64 | 100% | 0.811x | 51.6% | 1.63x | 83.9% |
| 11 | M64N64K128_W32x16_S2_B128 | 82% | 0.886x | 50.2% | 1.22x | 61.2% |
| 10 | M128N64K128_W64x16_S3_B128 | 82% | 0.881x | 49.9% | 1.24x | 61.8% |
| 12 | M128N128K128_W64x32_S2_B128 | 82% | 0.619x | 35.1% | 1.70x | 59.5% |
| 15 | M128N128K128_W64x32_S2_B256 | 47% | 0.421x | 24.3% | 2.38x | 57.7% |
| 13 | M128N128K128_W128x16_S1_B128 | 82% | 0.185x | 10.5% | 5.40x | 56.5% |
| 6 | M128N128K64_W128x16_S2_B128 | 82% | 0.151x | 8.5% | 8.81x | 75.2% |

W4A4:

| cfg | config | cov | blockwise vs fp16 | % of shipped | tax | tile alone |
|---:|---|---:|---:|---:|---:|---:|
| 7 | M128N64K128_W64x16_S2_B256 | 47% | 2.052x | **65.1%** | 1.13x | 73.7% |
| 8 | M64N128K128_W64x16_S2_B256 | 47% | 2.052x | 65.1% | 1.10x | 71.8% |
| 6 | M128N128K64_W128x16_S2_B128 | 82% | 1.993x | 63.6% | 1.26x | 80.1% |
| 4 | M64N128K128_W64x16_S2_B128 | 47% | 1.952x | 61.9% | 1.16x | 71.8% |
| 10 | M128N64K128_W64x16_S3_B128 | 47% | 1.673x | 53.1% | 1.24x | 65.6% |
| 11 | M64N64K128_W32x16_S2_B128 | 47% | 1.659x | 52.6% | 1.34x | 70.4% |
| 5 | M128N64K128_W32x32_S2_B128 | 47% | 1.640x | 52.0% | 1.50x | 77.8% |
| 2 | M128N128K64_W64x32_S2_B64 | 82% | 1.624x | 51.9% | 1.54x | 79.6% |
| 0 | M128N128K64_W128x16_S2_B64 | 82% | 1.529x | 48.8% | 1.61x | 78.4% |
| 3 | M128N64K128_W64x16_S2_B128 | 47% | 1.503x | 47.7% | 1.55x | 74.1% |
| 14 | M128N128K64_W64x32_S3_B64 | 82% | 1.397x | 44.6% | 1.84x | 81.9% |
| 12 | M128N128K128_W64x32_S2_B128 | 47% | 1.315x | 41.7% | 1.53x | 63.6% |
| 1 | M128N128K64_W128x16_S3_B64 | 82% | 1.306x | 41.7% | 1.98x | 82.5% |
| 15 | M128N128K128_W64x32_S2_B256 | 47% | 1.279x | 40.5% | 1.57x | 63.6% |
| 13 | M128N128K128_W128x16_S1_B128 | 47% | 0.397x | 12.6% | 4.92x | 62.0% |

**Coverage is not free.** K=128 needs `C % 128 == 0` for int8 (82% of conv frequency) and
`C % 256 == 0` for int4 (47%). A config at 47% coverage would leave more than half the conv
frequency on the shipped per-tensor path, so its 65% is measured on a minority of the model. The
best config with full int8 coverage is cfg0 at 64.6%.

## What would actually be needed

The remaining gap is not a parameter. Both precisions need the hand-written tile to reach
CUTLASS's steady-state efficiency at K=128 -- deeper pipelining without the smem cost (async
barriers / a 3-stage A ring with a 2-stage B ring), better ldmatrix scheduling, and per-shape
tile selection. That is "match CUTLASS at implicit GEMM", which is open-ended rather than a
sweep. Two smaller, bounded items also remain: FFMA interleaving (DeepGEMM reports 10%+, and
CUDA 12.4 here does not do it automatically) and split-K for the low-occupancy 2x2/4x4 shapes
(measured ceiling +4%).
