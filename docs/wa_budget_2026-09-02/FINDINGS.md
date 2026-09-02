# Weight vs activation granularity, at 8 and 4 bits: block the ACTIVATIONS, not the weights --
# and the blockwise conv kernel's real use case is W4A4

LSUN-churches LDM-KL-8, A40, n=6, seed 20260805, 50 DDIM, `MODIFF_LINEAR=0`, static delta.
**Attention fp16 in every arm** (`MODIFF_STD_ATTN_BITS=0`) so it cannot mask the conv terms.
Latent relL2 vs the fp16 arm. `scripts/wa_budget.sh`, `data/arms.jsonl`.

Instrument: the `MODIFF_ACT_BLOCK` simulation forward, so bit width and granularity are qmax and
grouping changes rather than new kernels. New knobs `MODIFF_ACT_SIM_WBITS` /
`MODIFF_ACT_SIM_WBLOCK` quantize the ORIGINAL weight at a chosen bit width, either per-output-
channel (the free axis) or blockwise along C (the axis that costs a mainloop flush). The weight
quantizer was verified to match an explicit per-group loop exactly.

**This measures quantizer error propagation, not the shipped int4 kernel path** (no zero-point, no
packing). It decides granularity; it is not an int4 speed or fidelity number.

## Floor

**A exact, W exact, attention fp16 = 0.0113.** Nothing below this is resolvable.

## Result

| | relL2 | above floor |
|---|---:|---:|
| **8-bit** | | |
| W8 only, per-output-channel | 0.0315 | 2.79x |
| W8 only, blockwise B=64 | 0.0392 | 3.48x |
| A8 only, per-tensor static | 0.0382 | 3.39x |
| **A8 only, blockwise B=64** | **0.0097** | **0.86x — at the floor** |
| W8A8 both coarse | 0.0560 | 4.97x |
| W8A8 both blockwise B=64 | 0.0374 | 3.32x |
| **4-bit** | | |
| W4 only, per-output-channel | 0.2604 | 23.10x |
| W4 only, blockwise B=64 | 0.2026 | 17.97x |
| W4 only, blockwise B=32 | 0.1915 | 16.99x |
| A4 only, per-tensor static | 0.5181 | 45.96x |
| **A4 only, blockwise B=64** | **0.0415** | **3.68x** |
| W4A4 both coarse | 0.5051 | 44.80x |
| W4A4 both blockwise B=64 | 0.2034 | 18.04x |

## Blocking the weights is not worth it, at either precision

**8-bit: it does nothing.** 0.0315 -> 0.0392 going from per-output-channel to blockwise B=64 --
nominally *worse*, i.e. the two are one indistinguishable blob. Blocking cannot make a quantizer
worse, so this is chaos, and the honest reading is "no effect".

**4-bit: 1.29x, and it does not solve the problem.** 0.2604 -> 0.2026 at B=64, 0.1915 at B=32.
Even fully blocked, W4 sits at 0.19-0.20, **17-18x the floor**. W4 weight error is not
granularity-limited: per-output-channel already covers only R*S*C elements per scale (1728-3456
here, far FINER than the 16384 of DeepSeek-V3's 128x128 blocks), so there is little left for
grouping to recover. Fixing W4 needs a different method -- salient-channel protection or low-rank
absorption (AWQ / GPTQ / SVDQuant), not a finer scale.

Against that 1.29x, blocking the weights permanently moves the weight scale from the FREE axis
(per-output-channel factors out of the reduction, applied once in the epilogue) into the mainloop
flush, and removes the option of ever cheapening the activation granularity independently.
**Recommendation: leave the weights per-output-channel at both precisions.**

DeepSeek-V3 blocks weights at 128x128 because FP8's E4M3 range (~2^+-8) requires fine scaling in
both directions to keep values representable. That is a format constraint, not a granularity
insight, and it does not transfer to int8/int4.

## Blocking the ACTIVATIONS is where everything is

| | per-tensor | blockwise B=64 | gain |
|---|---:|---:|---:|
| 8-bit | 0.0382 | **0.0097** (at the floor) | **3.9x, term eliminated** |
| 4-bit | 0.5181 | **0.0415** | **12.5x** |

At 8 bits, blockwise activations take the term to the floor -- it stops existing. At 4 bits it is a
**12.5x** reduction, and in the full stack W4A4 goes 0.5051 -> 0.2034, a **2.5x** end-to-end gain.

**So the blockwise conv kernel we built has its real use case at W4A4, not W8A8.** At 8 bits it
removes a 0.038 term from a stack that `docs/act_budget_2026-09-02` shows is dominated by
quantized attention at 0.1034 -- true but small. At 4 bits the activation quantizer IS the stack
(0.5181 of it), and blockwise is the only thing measured that touches it.

Note the residual: after blockwise activations, W4A4 sits at 0.2034, which is essentially W4
weights alone (0.2026). **At 4 bits, fixing the activations promotes weights to the dominant
term** -- and per the section above, granularity will not fix them.

## This corrects docs/act_budget_2026-09-02

That run reported a floor of **0.0352**; the correct floor is **0.0113**, 3x lower. Cause: the
updown resize fold (`_prequant_gn_resize_conv_modiff` -> `_conv_from_int8_o_hat`) reaches the conv
with PER-TENSOR codes, and `_conv_from_int8_o_hat` had no `_sim_guard` when act_budget ran, so **8
of 70 conv layers silently ran per-tensor in every arm of that sweep**, including its floor. The
guard added later caught it immediately here. The fold is now gated off under `MODIFF_ACT_BLOCK`
as well as `MODIFF_CONV_BLOCKK`.

Consequence for that document's conclusions:

- **Wrong:** "the conv-input quantizer contributes below the measurement floor". Against the
  correct floor, A8 per-tensor is 0.0382 = **3.39x the floor**, clearly measurable.
- **Wrong:** "W8 weights contribute nothing measurable". W8 per-channel is 0.0315 = 2.79x floor.
- **Still right:** quantized attention dominates at 8 bits. 0.1034 is ~2.7x the largest conv term
  here, so the ordering attention >> activation quantizer ~ weights > floor holds.
- **Still right, and now quantified:** granularity becomes a dominant term at 4 bits.

The qualitative conclusion of `docs/conv_blockk_e2e_2026-09-02` (MoDiff dominates
baseline+blockwise on both axes at W8A8) is unaffected -- it was measured with real kernels, not
the sim, and did not depend on the floor.

## What this says to do

1. **W8A8:** ship activation blockwise B=64 if the ~15% conv cost is acceptable; leave weights
   alone. But note the attention term is 2.7x larger and cheaper to attack.
2. **W4A4:** activation blockwise is essential -- 12.5x, and nothing else measured touches it.
   Then weights become dominant and need a method, not a granularity.
3. **Do not implement blockwise weights.** Measured 1.0x at 8 bits and 1.29x at 4 bits, against a
   permanent move of the weight scale onto the non-free axis.
