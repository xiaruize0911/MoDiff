# Blockwise B=32 for the conv-input quantizer (baseline + MoDiff)

The `a_hat` work (`docs/ahat_blockwise_2026-09-01`) made the *cache* blockwise int8 and
faster than fp16. This asks the same of the quantizer that feeds the conv: `Q(a_t)` in
the baseline arm, `Q(a_t - a_hat_{t+1})` in MoDiff.

## The structural difference from a_hat

`a_hat` is stored state. Both its quantize and its dequantize happen inside kernels we
own, so the scale can be as fine as we like.

The conv-input quantizer is a **GEMM operand**, and its scale is the CUTLASS epilogue's
scalar `alpha`:

```74:74:/workspace/MoDiff/csrc/modiff/conv/conv2d_evt.cu
  using E_MulA  = ct::Sm80EVT<Mul, Accum, Alpha>;      // acc*alpha
```

A blockwise scale along C is a scale along the conv's **reduction** axis. By the time
the epilogue sees the int32 accumulator, the per-block structure has been summed away.
Expressing it needs a mainloop that promotes to fp32 every 32 K -- the DeepSeek-V3
blockwise-fp8 shape -- which this CUTLASS 2.x sm80 int8 conv has no support for. Weight
scales are per-output-channel, i.e. along N, which is exactly the axis that *is* free.

So the accuracy had to be priced before deciding whether that mainloop is worth writing.

## Measured accuracy of each granularity

`scripts/act_quant_error.py` captures the exact tensor each int8 conv quantizes, on a
real 50-step run, and reports `||dequant(Q(v)) - v|| / ||v||` averaged over layers and
steps. This measures the quantizer, not the sampler -- see "why not relL2" below.

| granularity | baseline | MoDiff | vs shipped | epilogue-expressible? |
|---|---:|---:|---:|---|
| `static` per-tensor, calibrated / delta table (**ships today**) | 0.1838 | 0.1537 | 1.0x | yes (scalar alpha) |
| `dyn` per-tensor, dynamic absmax | 0.0451 | 0.0506 | ~3.5x | yes (scalar alpha) |
| `perpix` one scale per pixel, all C | 0.0189 | 0.0186 | ~9x | 1x1 convs only (see below) |
| `B=128` along C | 0.0126 | 0.0124 | ~13x | no -- along K |
| `B=64` along C | 0.0103 | 0.0102 | ~16x | no -- along K |
| `B=32` along C | 0.0082 | 0.0082 | ~20x | no -- along K |
| `B=16` along C | 0.0064 | 0.0065 | ~26x | no -- along K |

Both arms agree closely, which is expected: the granularity question is about the shape
of the tensor being quantized, and MoDiff's delta has the same per-pixel/per-channel
spread structure as the activation it came from.

Three things fall out:

**The shipped static per-tensor scale is the single largest quantization error in the
conv path**, 3-4x worse than simply taking a dynamic absmax. That matches the existing
note in `_delta_scale_args` that static was measured clipping on 49 of 70 conv layers.

**B=32 is a real 6x win over per-tensor dynamic and ~20x over what ships.** The request
is well motivated on accuracy; the only obstacle is the mainloop.

**Per-pixel is NOT the free lunch it looks like.** A scale that varies purely along M is a
row broadcast that Sm80 EVT already expresses. But the activation scale lives on the *input*
pixel, and a 3x3 conv reads 9 different input pixels per output pixel -- so in implicit-GEMM
terms the scale depends on the reduction index (r,s) as well as on m, and it is not an M
broadcast at all. It only reduces to one for 1x1 convs, where input and output pixels
coincide. (An earlier revision of this file claimed otherwise; it was wrong.) The same
argument kills any activation granularity finer than per-tensor in the epilogue of an R,S>1
conv. So the mainloop is the only route, and B is a free parameter -- hence the sweep above.

**Accuracy is flat in B.** B=32 to B=128 is only 1.5x, while the mainloop dequant cost scales
as 1/B. That makes B a real tuning knob rather than a foregone 32.

## Why not relL2 of the final latent

The first attempt measured final-latent relL2 per arm. It does not work here. A single
conv's quantization error starts around 3e-4; 70 convs and 50 steps amplify it, so the
number is dominated by trajectory chaos. Verified directly: the simulation path
reproduces the real int8 kernel to 3.7e-4 relL2 on the first layer at step 0
(`out relL2 0.00037`, inputs bit-identical) yet the two runs' final latents differ by
0.6 after only 5 steps. This project already knows relL2 is not reproducible to better
than +-0.03. Per-layer quantizer error has none of that variance.

## What is in the tree

- `MODIFF_ACT_BLOCK` in `integration/kernels/int8_optimized.py`: a simulation forward
  (`_forward_blockwise_sim`) that fake-quantizes the conv input at a chosen granularity
  and runs the conv in fp32 on dequantized W8 weights. Both arms share it, so
  granularity is the only variable. `0` off, `-2` per-tensor static, `-1` per-tensor
  dynamic, `N` blockwise. **Measurement harness only -- it bypasses every fused kernel**,
  and `_sim_guard` makes any fused entry point that still fires a hard error rather than
  a silently per-tensor measurement.
- `scripts/act_quant_error.py` -- the table above.
- `scripts/act_block32_quality.py` -- end-to-end sim arms + image grid. Kept, but read
  it with the chaos caveat above.

## The blockwise-along-K mainloop, built and measured

`csrc/modiff/linear/gemm_blockk.cu` -- `gemm_w8a8_blockk`. An int8 GEMM that applies the
activation scale *inside* the mainloop, once per K-block, plus a matched scalar-alpha control
(`BLOCKWISE=false`) at the identical tile config so the delta is the dequant and not a tiling
change. `mma.m16n8k32.s8` reduces exactly 32 K, so one mma is one B=32 block and its int32
result is already the block's partial sum.

Two structural costs, both independent of B:

*Registers.* A blockwise mainloop must hold an int32 block partial and an fp32 running sum at
once, so accumulator registers double. The shipped 4-warp `gemm_w8a8_kernel_awq` carries
`acc[8][2][8]` = 128 int32/thread, which would become 256 and spill. This kernel runs 8 warps
at `WARP_N=16` (`NJ=1`) instead: 64 int32 + 64 fp32, same 128x128 CTA tile, same 48 KiB smem.
Measured: control 121 regs, B=64 128 regs (16 B spill), B=32 144 regs, none of it serious.

*ALU.* Per accumulator per flush, one IADD + one FADD (int32->float) + one FFMA. The
conversion avoids I2F, which is quarter-rate on the XU pipe on GA10x -- adding the integer to
the mantissa of 1.5*2^23 and subtracting it back is exact and full rate, the same trick
`ahat_cache.cuh` uses for int8. Safe to |v| < 2^22; a B=128 block of int8 products peaks at
128*127*127 = 2064512.

**Correctness** (`scripts/blockk_gemm_check.py`): against an exact fp32 per-block reference,
relL2 = 2.071e-04 -- which is precisely the fp16 output rounding floor (rounding the reference
itself to fp16 gives the same 2.071e-04), so the mainloop is bit-exact up to the store. With
all block scales set equal, blockwise reproduces the scalar-alpha control with 99.94% of
elements bitwise identical, the rest fp32 accumulation order.

**Cost**, A40, median of 50:

| M | K | N | control | CUTLASS `gemm_w8a8_awq` | B=64 | B=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1024 | 1024 | 0.074 | 0.076 | 0.094 (1.27x) | 0.141 (1.89x) |
| 8192 | 1152 | 1152 | 0.143 | 0.159 | 0.192 (1.33x) | 0.276 (1.94x) |
| 16384 | 1024 | 512 | 0.128 | 0.138 | 0.171 (1.35x) | 0.258 (2.02x) |
| 32768 | 512 | 512 | 0.136 | 0.151 | 0.181 (1.34x) | 0.265 (1.95x) |

The control matches or beats the shipped AWQ GEMM, so the slowdowns are measured against a
real baseline rather than a strawman.

**B=64 is the operating point.** It keeps 85% of B=32's accuracy gain (16x vs 20x against the
shipped static scale) for 1.33x instead of 1.95x. It also divides every channel count in this
UNet (192, 384, 576) and equals `BK_CTA_K`, so it flushes exactly once per CTA-K tile.

## The conv

`csrc/modiff/conv/conv2d_int8_blockk.cu` -- `conv2d_int8_blockk`. Same mainloop, implicit GEMM
(M = N*P*Q, N_gemm = K_out, K_gemm = R*S*C). The weight is already [K,R,S,C] contiguous so B
loads exactly as in the GEMM; two things are new.

*The A loader gathers, and pads with zeros.* An out-of-bounds tap is part of the sum, not a row
to be skipped, so it uses cp.async's src-size form (src-size 0 zero-fills the destination)
instead of the predicated form the GEMM uses for its M tail.

*The block scale is indexed by the input pixel.* Recomputing that gather at every flush costs
about 7 integer ops and an LDG per accumulator row, on top of the 3 ops per accumulator the
flush already does -- roughly +60%. Instead the CTA's 128 row scales are staged into smem on
the same pipeline stage as the A/B cp.async, so the existing per-tile `__syncthreads` covers
them and the flush is just LDS. Each thread always stages the same CTA row, so the two integer
divisions of the (n,p,q) decomposition are hoisted out of the k-loop entirely. Registers stay
at 125-130 with no spills; smem goes slightly over the 48 KiB static limit, hence the dynamic
allocation and `cudaFuncSetAttribute`.

**Correctness** (`scripts/blockk_conv_check.py`): relL2 2.93e-04 against an exact fp32
reference whose own fp16 floor is 2.07e-04, the residual being fp32 accumulation order. Edge
and interior pixels agree (2.92e-04 vs 2.95e-04) including non-square odd inputs and stride 2,
so the zero-fill padding is right. With all block scales equal, blockwise reproduces the
scalar-alpha control with 99.9%+ of elements bitwise identical, for 3x3, 1x1 and stride 2.

**Cost**, A40, median of 30, batch 32 at the churches UNet's real conv shapes:

| C | HxW | K | R | control | CUTLASS `conv2d_int8_fprop` | B=64 | B=32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 32x32 | 192 | 3 | 0.266 | 0.241 | 0.367 (1.39x) | 0.408 (1.53x) |
| 384 | 16x16 | 384 | 3 | 0.217 | 0.144 | 0.298 (1.38x) | 0.331 (1.53x) |
| 576 | 8x8 | 576 | 3 | 0.124 | 0.079 | 0.150 (1.21x) | 0.167 (1.36x) |
| 192 | 32x32 | 192 | 1 | 0.126 | 0.098 | 0.138 (1.10x) | 0.142 (1.12x) |

The parenthesised factors are against the matched control, i.e. the dequant alone. **Against
the shipped CUTLASS conv the honest factor is worse: 1.5x, 2.1x, 1.9x, 1.4x for B=64**, because
this hand-written implicit-GEMM tile is itself 1.1-1.5x behind CUTLASS before any blockwise
work. That gap is tile tuning, not the mainloop, and it has not been chased.

1x1 convs are nearly free (1.10x) -- they are memory bound at these shapes, so the extra ALU
hides. The 3x3 convs are where the cost is.

## Where this leaves it

Accuracy and cost, both measured, for the conv-input quantizer:

| | quantization error | conv cost vs shipped |
|---|---:|---:|
| per-tensor static (ships) | 0.154 | 1.0x |
| blockwise B=64 | 0.0102 (15x better) | ~1.5-2.1x |
| blockwise B=32 | 0.0082 (19x better) | ~1.7-2.3x |

B=64 is the operating point: 85% of B=32's accuracy gain at a noticeably lower cost, and it
divides every channel count here while equalling the CTA-K tile.

Not done: wiring either kernel into the model behind a flag, and closing the 1.1-1.5x gap
between the hand-written tile and CUTLASS (which would take the B=64 conv to roughly 1.3-1.4x
of shipped rather than 1.5-2.1x).
