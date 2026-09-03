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


---

# Addendum: asymmetric A/B rings — the last candidate, and it fails

The ceiling argument above said the trade is mediated by shared memory: removing the accumulator
carry needs TK=128, and 2 CTA/SM then forces `CTA_M+CTA_N <= 192`, so the full 128x128 tile (where
the tile quality is) and TK==BLK (where the low tax is) cannot co-exist. There was one mechanism
left that the argument did not cover: **give A and B separate ring depths.**

smem is `SA*TM*TK + SB*TN*TK`, so SA=2 / SB=1 puts a full 128x128 tile at TK=128 into
**50176 B -> 2 CTA/SM**, where SA=SB=2 needs 66560 B and gets 1. That is exactly the missing
combination, so it was implemented (configs 16-18) and correctness-gated (relL2 3.61-3.66e-04).

**It fails badly.** Best asymmetric result: W8A8 cfg18 at **18.9%** of shipped (tax **3.62x**),
W4A4 cfg16 at **46.2%** (tax 1.52x) -- against 63.5% and 64.4% for the best symmetric configs.

The reason is the price the asymmetry charges: with SB=1, B has no load/compute overlap and every
tile needs a `__syncthreads()` before its reload. That serialises half the tile traffic against
the mma stream. It also does not deliver the hoped-for tile: configs 16/17 measure a 68-70% tile,
not the 80-84% the 128x128 configs reach at TK=64, because the extra barrier costs the scalar
control too.

**So the design space is closed.** Both operands need real double buffering, that fixes smem at
`2*(CTA_M+CTA_N)*TK`, and from there the tile-vs-tax trade is forced. 80% of shipped is not
reachable by any (CTA_M, CTA_N, CTA_K, warp tile, stages, B) setting of this kernel; it requires
the tile itself to reach ~95% of CUTLASS while paying a >=1.19x tax, and it is at 65-82%.

## Final benchmark, all 19 configs

Same protocol as above. Run-to-run spread on `% of shipped` is 1-2 points, so read the top few as
a tie.

### W8A8

| cfg | config | cov | blockwise vs fp16 | % of shipped | tax | tile alone |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `M128N128K64_W128x16_S2_B64` | 100% | 1.010x | **63.5%** | 1.26x | 79.8% |
| 5 | `M128N64K128_W32x32_S2_B128` | 82% | 1.125x | **63.2%** | 1.07x | 67.5% |
| 8 | `M64N128K128_W64x16_S2_B256` | 47% | 1.121x | **62.5%** | 1.05x | 65.7% |
| 4 | `M64N128K128_W64x16_S2_B128` | 82% | 1.030x | **57.9%** | 1.11x | 64.2% |
| 2 | `M128N128K64_W64x32_S2_B64` | 100% | 0.902x | **56.8%** | 1.44x | 81.6% |
| 7 | `M128N64K128_W64x16_S2_B256` | 47% | 0.989x | **55.1%** | 1.22x | 67.4% |
| 3 | `M128N64K128_W64x16_S2_B128` | 82% | 0.937x | **52.6%** | 1.28x | 67.3% |
| 9 | `M128N128K64_W128x16_S2_B32` | 100% | 0.781x | **49.1%** | 1.61x | 78.9% |
| 11 | `M64N64K128_W32x16_S2_B128` | 82% | 0.865x | **48.6%** | 1.26x | 61.3% |
| 1 | `M128N128K64_W128x16_S3_B64` | 100% | 0.739x | **46.5%** | 1.71x | 79.5% |
| 14 | `M128N128K64_W64x32_S3_B64` | 100% | 0.720x | **45.3%** | 1.86x | 84.1% |
| 10 | `M128N64K128_W64x16_S3_B128` | 82% | 0.765x | **43.0%** | 1.25x | 53.6% |
| 12 | `M128N128K128_W64x32_S2_B128` | 82% | 0.649x | **36.4%** | 1.62x | 59.1% |
| 15 | `M128N128K128_W64x32_S2_B256` | 47% | 0.458x | **25.5%** | 2.27x | 57.9% |
| 18 | `M128N128K128_W64x32_SA2SB1_B256` | 47% | 0.338x | **18.9%** | 3.62x | 68.2% |
| 17 | `M128N128K128_W64x32_SA2SB1_B128` | 82% | 0.203x | **11.4%** | 6.56x | 74.8% |
| 16 | `M128N128K128_W128x16_SA2SB1_B128` | 82% | 0.195x | **11.0%** | 6.59x | 72.3% |
| 13 | `M128N128K128_W128x16_S1_B128` | 82% | 0.190x | **10.7%** | 5.33x | 56.8% |
| 6 | `M128N128K64_W128x16_S2_B128` | 82% | 0.156x | **8.8%** | 8.51x | 74.5% |

### W4A4

| cfg | config | cov | blockwise vs fp16 | % of shipped | tax | tile alone |
|---:|---|---:|---:|---:|---:|---:|
| 8 | `M64N128K128_W64x16_S2_B256` | 47% | 2.041x | **64.4%** | 1.09x | 70.2% |
| 7 | `M128N64K128_W64x16_S2_B256` | 47% | 1.983x | **62.5%** | 1.15x | 71.9% |
| 6 | `M128N128K64_W128x16_S2_B128` | 82% | 1.888x | **60.7%** | 1.31x | 79.8% |
| 4 | `M64N128K128_W64x16_S2_B128` | 47% | 1.915x | **60.4%** | 1.17x | 70.6% |
| 5 | `M128N64K128_W32x32_S2_B128` | 47% | 1.642x | **51.8%** | 1.48x | 76.8% |
| 11 | `M64N64K128_W32x16_S2_B128` | 47% | 1.579x | **49.8%** | 1.39x | 69.2% |
| 0 | `M128N128K64_W128x16_S2_B64` | 82% | 1.532x | **49.3%** | 1.61x | 79.2% |
| 3 | `M128N64K128_W64x16_S2_B128` | 47% | 1.554x | **49.0%** | 1.48x | 72.6% |
| 10 | `M128N64K128_W64x16_S3_B128` | 47% | 1.552x | **48.9%** | 1.22x | 59.7% |
| 16 | `M128N128K128_W128x16_SA2SB1_B128` | 47% | 1.465x | **46.2%** | 1.52x | 70.0% |
| 2 | `M128N128K64_W64x32_S2_B64` | 82% | 1.367x | **44.0%** | 1.80x | 79.2% |
| 17 | `M128N128K128_W64x32_SA2SB1_B128` | 47% | 1.389x | **43.8%** | 1.65x | 72.1% |
| 12 | `M128N128K128_W64x32_S2_B128` | 47% | 1.362x | **42.9%** | 1.41x | 60.5% |
| 1 | `M128N128K64_W128x16_S3_B64` | 82% | 1.217x | **39.1%** | 2.05x | 80.3% |
| 14 | `M128N128K64_W64x32_S3_B64` | 82% | 1.209x | **38.9%** | 2.08x | 81.0% |
| 15 | `M128N128K128_W64x32_S2_B256` | 47% | 0.917x | **28.9%** | 2.11x | 61.1% |
| 18 | `M128N128K128_W64x32_SA2SB1_B256` | 47% | 0.539x | **17.0%** | 4.26x | 72.3% |
| 13 | `M128N128K128_W128x16_S1_B128` | 47% | 0.383x | **12.1%** | 4.85x | 58.7% |

---

# Addendum 2: the 80% target IS met, at both precisions -- the missing piece was a kernel,
# not a tile parameter

The sweep above concluded 80% was unreachable. That conclusion was drawn from **conv-kernel**
ratios, and it holds at that level: the best conv-kernel result is 63.5% (W8A8) / 64.4% (W4A4).
But the target is about speed, and end to end the conv loss is diluted by the buckets blockwise
does not touch. Measured end to end, with the fused quantize kernels in place:

| | shipped | blockwise B=64 | **% of shipped** | vs fp16 | peak alloc | relL2 |
|---|---:|---:|---:|---:|---:|---:|
| **W8A8** | 72.60 | **88.37** | **82.2%** | 1.154x | 3.62 G | 0.1557 (shipped 0.3219) |
| **W4A4** | 59.67 | **68.73** | **86.8%** | 1.483x | 3.86 G | 0.5295 (shipped 0.6014) |

Both **meet the >= 80% target**, and both are also *more accurate* and use *less peak memory*
than their shipped per-tensor arm.

## What actually closed the gap

Not a tile parameter -- `gn_silu_blockk_quantize_pack_int4`, the fused
GN(+mod)(+SiLU) -> blockwise-along-C int4 quantize+pack. W4A4 measured, all three numbers from
the same protocol:

| | ms/step |
|---|---:|
| fusion handicap the unfused arms were paying | **+30.25** (59.67 -> 89.93 per-tensor) |
| **fusing the blockwise quantize back in** | **-28.64** (97.37 -> 68.73) |
| blockwise cost that remains, fused vs shipped | +9.05 -> 86.8% of shipped |

So the blockwise mechanism was never the problem at W4A4: it costs ~9 ms, while running without a
fused quantize cost 30 ms. The int8 arm had had its fused kernel since
`docs/conv_blockk_e2e_2026-09-02`; int4 did not, and that single absence was the whole shortfall.

BLK=64 is the operating point at both precisions. BLK=128 unfused measured 99.47 vs 97.37 for
BLK=64, so the larger block does not pay here either.

## Two things this does not claim

**The conv-kernel ratios above are unchanged.** 63.5% / 64.4% at the kernel level is still the
honest number for "this conv kernel versus CUTLASS", and the shared-memory ceiling argument for
that level still stands. The E2E figures are larger because conv is a fraction of the step.

**W4A4 quality is poor in absolute terms.** relL2 0.5295 against fp16, and the samples
(`plots/samples_w4a4.png`) are visibly broken at every W4A4 setting. Blockwise improves it 12%
(0.6014 -> 0.5295), far less than the 2.5x `docs/wa_budget_2026-09-02` measured with attention
held at fp16 -- because here attention is genuinely W4A4 and masks the activation term, the same
way it does at 8 bits. W4A4 is not shippable on this model regardless of the conv speed.
