# `conv2d_int4_blockk`: the blockwise conv kernel at the precision where it actually pays

A40, batch 128, median of 50 after 10 warmup. `csrc/modiff/conv/conv2d_int4_blockk.cu`.

## Why int4

`docs/wa_budget_2026-09-02` measured the activation-granularity term at both precisions:

| | per-tensor | blockwise B=64 | gain |
|---|---:|---:|---:|
| 8-bit | 0.0382 | 0.0097 (at the 0.0113 floor) | 3.9x, but the term was already small |
| **4-bit** | **0.5181** | **0.0415** | **12.5x** |

At 8 bits blockwise removes a term that quantized attention (0.1034) dwarfs. At 4 bits the
activation quantizer **is** the budget, and blockwise is the only thing measured that touches it.
Full stack W4A4 goes 0.5051 -> 0.2034, a **2.5x** end-to-end reduction.

## The port

A close port of `conv2d_int8_blockk.cu`. Three changes; everything about the gather, the smem
swizzle, the zero-filled `cp.async` and the scale staging is unchanged.

1. `mma.m16n8k64.s4` (`modiff_mma_m16n8k64_s4`, already in the tree) instead of
   `mma.m16n8k32.s8`. One mma reduces 64 K, not 32.
2. Operands are packed 2 codes per byte, so a **64-byte smem row holds 128 elements**. The loaders
   are byte-indexed and therefore identical; only the host's element<->byte accounting and the A
   gather address change (`X + pix*(C/2) + c0/2`). `BK4_INTRIN_KB = 32` keeps `ld_col` in bytes.
3. Overflow stops mattering. A block peaks at `BLK*7*7 = 49*BLK`, so `bk4_i2f`'s 2^22 limit is not
   reached until BLK ~ 85000. Compare int8, where BLK=256 already sits at a 1.5% margin.

The `ACCUM` (`o_hat` RMW) and `RESID` (skip-add fold) epilogues come across for free.

**Correctness** (`scripts/check_int4.py`): relL2 **3.6e-04** against an exact fp32 per-block
reference whose own fp16 storage floor is 2.07e-04 -- the residual is fp32 accumulation order, the
same signature the int8 kernel has. With all block scales set equal, blockwise reproduces the
scalar-alpha control **99.92-99.99% bitwise**. Verified at B=64/128/256, 3x3 and 1x1.

## Coverage: the C%128 constraint costs almost nothing

A 128-element K tile must not straddle two (r,s) taps, so `C % 128 == 0`:

| C | layers | weight-MAC share | |
|---:|---:|---:|---|
| 192 | 10 | 1.5% | falls back |
| 384 | 24 | 13.5% | OK |
| 576 | 2 | 1.3% | falls back |
| 768 | 27 | 56.2% | OK |
| 1152 | 2 | 5.1% | OK |
| 1536 | 5 | 22.5% | OK |

**58/70 layers, 97% of weight-MACs.** The 12 ineligible layers are the two smallest channel counts
and together are 2.8% of the arithmetic. A 32-byte-row variant (1-bit swizzle, reworked loaders)
would close it but is not worth writing for 2.8%.

## Cost

| shape (batch 128) | CUTLASS int4 | ctrl (our tile, scalar) | **B=64** | B=128 | B=256 |
|---|---:|---:|---:|---:|---:|
| C384 16x16 K384 3x3 | 0.259 | 0.278 (1.07x) | **0.401 (1.55x)** | 0.498 (1.92x) | — |
| C768 8x8 K768 3x3 | 0.218 | 0.289 (1.32x) | **0.411 (1.88x)** | 0.488 (2.23x) | 0.483 (2.21x) |
| C384 32x32 K384 3x3 | 0.846 | 1.075 (1.27x) | **1.565 (1.85x)** | 1.821 (2.15x) | — |

Against our own matched control the blockwise cost is **1.42-1.46x** at B=64 -- i.e. the dequant
alone is +44%, and the rest of the gap to CUTLASS is our tile (1.07-1.32x, worse than the int8
port's 1.06x).

**B=64 is the operating point and larger B is worse**, same as int8 but for a different reason.
Registers: B=64 takes **128** (2 CTA/SM); B=128 takes **197** (1 CTA/SM), because the int32
accumulator must survive both `k01` steps instead of being flushed each one. Halving the flush
rate does not pay for halving occupancy. (The `__launch_bounds__` min-blocks predicate reads
`BLK <= BK4_CTA_K`, comparing elements against a byte constant; for int4 that happens to split
64/128 correctly, but it is semantically wrong and should read `BK4_CTA_KE`. Given B=128 is slower
either way, the accidental behaviour is the better one.)

## The trade, finally positive

| | accuracy gain | conv cost vs matched control |
|---|---:|---:|
| int8 B=64 | term already at the floor | 1.25x |
| **int4 B=64** | **12.5x** | **1.44x** |

This is the first configuration in this line of work where blockwise clearly pays: 12.5x on the
dominant error term for +44% on the conv kernel. At int8 the same mechanism bought something
unmeasurable for +25%.

## Kernel-level speed on the real UNet shapes -- the point of the whole exercise

`scripts/int4_shape_sweep.py`, `data/int4_shape_sweep.json`. Conv 3x3 only, batch 128,
frequency-weighted over the **14 of 20** churches-UNet shapes with `C % 128 == 0`.

| | ms | vs fp16 |
|---|---:|---:|
| fp16 | 24.587 | 1.000x |
| CUTLASS int4 (shipped) | 7.888 | **3.117x** |
| our tile, scalar alpha | 10.013 | 2.456x |
| **our tile, blockwise B=64** | **14.486** | **1.697x** |
| blockwise quantize+pack (separate pass) | +2.367 | 16% on top of the conv |

**The blockwise tax is multiplicative, which is why it only pays at 4 bits.** Same mechanism,
same relative cost (~1.5x over the matched scalar control at both precisions), completely
different outcome, because it divides a different baseline:

| | shipped baseline | blockwise tax | what is left |
|---|---:|---:|---:|
| W8A8 | 1.76x | /1.72x | **1.02x -- nothing** |
| W4A4 | 3.12x | /1.84x | **1.70x** |

At 8 bits the tax consumes the entire quantization win. At 4 bits there is enough headroom that
blockwise still lands at 1.70x over fp16 -- and per `docs/wa_budget_2026-09-02` it buys **12.5x**
on the dominant error term there, against something at the measurement floor at 8 bits.
`plots/int8_vs_int4.png`.

Including the separate quantize pass, 14.486 + 2.367 = 16.85 ms = **1.459x** fp16. A fused
GN->int4-blockwise-quantize kernel would recover most of that 2.367 (the int8 analogue did).

### Correction to the coverage note above

The `C % 128` constraint costs **29% of conv TIME**, not the 2.8% of weight-MACs stated earlier.
The ineligible shapes (C=192 and 576) are the high-resolution layers -- MAC-light but
time-heavy, since conv time at these shapes is not MAC-bound. The 32-byte-row variant is
therefore considerably more valuable than the MAC share suggested.

## Not done

- **No int4 blockwise activation quantizer.** `conv_quantize_block_nhwc` emits unpacked int8
  codes; int4 needs quantize + pack-2-per-byte + per-(pixel, C-block) scales. Required for E2E.
- **Not wired into the model.** `int4_optimized.py` has no `MODIFF_CONV_BLOCKK` path and no sim
  harness at all; the int8 wiring does not carry over.
- **No E2E number, and no fused GN->int4-blockwise-quantize kernel** (the int8 one recovered
  ~18 ms/step, so the analogous gap exists here).
- Zero-point: the shipped int4 path carries `weight_zp` / `zpw_window_sum`. This kernel is
  symmetric (no zero point), matching `conv2d_int4_fprop`'s symmetric mode. If the model path
  needs asymmetric activations, that is additional work.
