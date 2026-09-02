# Where int8's conv speedup actually comes from, and where blockwise spends it

A40, batch 128, 3x3 stride 1 pad 1, CUDA events, median of 25 after 8 warmup.
`scripts/shape_sweep.py`, `data/shape_sweep.json`, `plots/`.

## Why this exists, and a correction it forced

`docs/conv_blockk_e2e_2026-09-02` compared the two largest conv kernels of each arm in a model
profile and read 24.64 (int8) vs 23.99 ms (fp16), concluding int8's tensor-core advantage does
not show up at these shapes. **That was wrong.** fp16 spreads its conv over **15** kernel
instances (different tile configs, sm80/sm86 variants) while int8 concentrates in **2** fused EVT
kernels, so the top-2 of one is 56% of its conv time and the top-2 of the other is 77%.
Comparing top-2 to top-2 is not like-for-like. The correct in-model comparison is the totals:
**42.72 vs 32.09 ms = 1.33x**, and isolating just the quantized convs gives ~1.43x.

## Frequency-weighted over the 20 real UNet conv shapes

| | ms | vs fp16 |
|---|---:|---:|
| fp16 | 35.31 | 1.000x |
| **int8 EVT (shipped)** | **22.42** | **1.575x** |
| our tile, scalar alpha | 27.17 | 1.300x |
| **our tile, blockwise B=64** | **35.06** | **1.007x** |

**Blockwise B=64 gives back the whole of int8's conv win** -- it lands on fp16. The E2E number is
still 1.155x (`conv_blockk_e2e` Addendum 3) only because the attention and elementwise buckets
keep their int8 advantage; the conv segment breaks even.

The 1.575x -> 1.007x gap splits roughly evenly: ~0.27x lost to our tile, ~0.29x to the blockwise
dequant. That corrects the impression from the in-model profile that the tile was already at
CUTLASS parity (1.06x there); shape-weighted it is 1.21x.

## What the sweep shows about shape dependence (`plots/speedup_vs_shape.png`)

**int8 EVT is nearly shape-independent**, flat at 1.75-1.85x across B, N, H and W. Two exceptions:

- **C <= 192 drops it to 1.23-1.46x** -- the reduction is too shallow to fill the tensor cores.
  This matters: 10 of the 70 quantized convs have C=192.
- B=8/256 and W=64 spike to 2.3-2.5x, which is the fp16 side picking a worse kernel, not int8
  improving.

**Blockwise B=64 is also nearly shape-independent, at 1.03-1.15x.** The one exception is C=64
(1.72x), where each pixel holds a single block and the flush count is minimal; the cost rises
with C, reaching **0.99x -- slower than fp16 -- at C=1536**.

Per shape (`plots/unet_shapes.png`), blockwise is worst on `C1536->N768 2x2` (0.36x) and
`C1536->N768 4x4` (0.83x): deep reduction, tiny spatial extent, so few CTAs and many flushes.
`C1536->N768 2x2` launches **24 CTAs onto 84 SMs**.

## Occupancy is not the lever it looks like

Shapes with a grid under two full waves are only **20% of the frequency-weighted time**, and the
well-occupied shapes only reach a **1.04x** median. So even if split-K brought every
under-occupied shape up to that median, the total moves 1.007x -> **1.049x, +4%**. Split-K was
therefore not implemented. The real headroom is the tile and the dequant, which is what
`docs/conv_int4_blockk_2026-09-02/TUNE_FINDINGS.md` sweeps.

## Two rejected micro-optimisations (see the kernel header)

- **STAGES=3**: helps the scalar control (1.300 -> 1.334x) and **hurts blockwise**
  (1.012 -> 0.829x). At STAGES=3/B=64 the blockwise smem is 50688 B and 2x that is exactly the
  opt-in per-SM limit, so blockwise likely falls to 1 CTA/SM where the control keeps 2. Reverted.
- **RZ-addend mma** (`modiff_mma_m16n8k32_zero`) to fold the per-flush `a[k]=0` into the mma that
  starts a block, saving 64 MOVs per flush per thread: correctness fine, **speed neutral**. Two
  mma variants in the inner loop cost about what the MOVs saved. Helper kept, not used.
