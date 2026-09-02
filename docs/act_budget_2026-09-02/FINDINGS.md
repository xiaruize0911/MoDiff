# Error budget: W8A8 latent relL2 is set by quantized ATTENTION. The conv-input quantizer

> **SUPERSEDED IN PART (2026-09-02, same day). Its floor is 3x too high and two of its
> conclusions invert.** `docs/wa_budget_2026-09-02` re-ran this with the same instrument after a
> guard was added to `_conv_from_int8_o_hat`, and measured a floor of **0.0113**, not the 0.0352
> below. Cause: the updown resize fold reaches the conv with PER-TENSOR codes, and that entry
> point had no `_sim_guard` when this document was measured, so **8 of 70 conv layers silently ran
> per-tensor in every arm here, including the floor arm**.
>
> Read against the correct floor:
> - **WRONG:** "the conv-input quantizer contributes below the measurement floor." A8 per-tensor
>   is 0.0382 = **3.39x** the correct floor, clearly measurable.
> - **WRONG:** "W8 weights contribute nothing measurable." 0.0315 = 2.79x the correct floor.
> - **STILL RIGHT:** quantized attention dominates at 8 bits (0.1034, ~2.7x the largest conv
>   term), so the ordering attention >> activation quantizer ~ weights > floor holds.
> - **STILL RIGHT, and now quantified:** granularity becomes a dominant term at 4 bits --
>   `wa_budget` measures A4 per-tensor 0.5181 vs blockwise B=64 0.0415, a 12.5x reduction.
>
> The needle control and the method here are unaffected. Use `docs/wa_budget_2026-09-02` for the
> numbers.
# contributes below the measurement floor, and blockwise has nothing to win.

LSUN-churches LDM-KL-8, A40, n=6, seed 20260805, 50 DDIM, `MODIFF_LINEAR=0`, static delta,
`int8_calibration_realckpt.pt`. Latent relL2 vs the fp16 arm. `scripts/act_budget.py`,
`data/act_budget.json`. Quality only -- the sim bypasses every fused kernel, so its time is
meaningless.

## Result

| group | arm | relL2 |
|---|---|---:|
| ref | fp16 | 0.0000 |
| **floor** | **exact A, exact W, fp16 attn** | **0.0352** |
| solo | exact A, **W8**, fp16 attn | 0.0344 |
| solo | exact A, exact W, **W8A8 attn** | **0.1034** |
| solo | exact A, W8, W8A8 attn | 0.1107 |
| sweep | A per-tensor static, exact W, fp16 attn | 0.0386 |
| sweep | A per-tensor dynamic, exact W, fp16 attn | 0.0377 |
| sweep | A blockwise B=64, exact W, fp16 attn | 0.0189 |
| sweep | A blockwise B=16, exact W, fp16 attn | 0.0222 |
| **needle** | **A static on INT4 grid, exact W, fp16 attn** | **0.5135** |
| full | A per-tensor static, W8, W8A8 attn (= shipped) | 0.1200 |
| full | A blockwise B=64, W8, W8A8 attn | 0.1135 |

## The floor is 0.0352, and it decides how to read everything else

"Everything exact" is not 0. The sim runs convs in fp32 against an fp16 reference, and 70 layers
x 50 steps of that difference diverges to **0.0352**. This harness cannot resolve any effect
smaller than that.

Three arms land *below* the floor: W8 weights (0.0344), B=64 (0.0189), B=16 (0.0222). A quantizer
cannot reduce error below the no-quantizer case, so those orderings are not real -- they are draws
from the same chaotic divergence. The sweep is non-monotonic in exactly the way that confirms it:
per-layer error says B=16 is strictly better than B=64 (0.0064 vs 0.0103), and E2E reports B=16 as
*worse* (0.0222 vs 0.0189).

So the entire sweep -- 0.0189 to 0.0386, spread 0.0197 -- is one indistinguishable blob sitting on
the floor, consistent with this tree's documented +-0.03 relL2 reproducibility.

## The needle control fired, which is what makes the flat sweep interpretable

Dropping the activation grid to int4 gives **0.5135**, 14.6x the floor. The metric responds
enormously to a genuinely coarser activation quantizer.

That rules out reading 2 from `docs/delta_dynamic_2026-09-02` ("relL2 saturates and cannot resolve
this"). Latent relL2 is a perfectly good instrument. The conv-input quantizer's contribution at
int8 is simply *small*, and int4 is where it stops being small.

## The answer: attention is the whole budget

Reading the solo arms against the 0.0352 floor:

| source, alone | relL2 | above floor |
|---|---:|---|
| W8 conv weights | 0.0344 | **nothing measurable** |
| conv-input quantizer, per-tensor static | 0.0386 | **nothing measurable** |
| W8A8 static attention | 0.1034 | **2.9x the floor -- dominant** |

Attention alone (0.1034) already accounts for the full shipped number (0.1200); in quadrature with
the floor it gives 0.109, and the remaining gap to 0.1200 is inside noise. W8 weights and the
conv-input quantizer are both invisible.

Cross-check that the sim is measuring the real path: sim-shipped is 0.1200, and the real-kernel
shipped arm in `docs/delta_dynamic_2026-09-02` is 0.0999. The 0.02 difference is the fp32 floor
folded in. The sim tracks the real path to within its own floor.

## Consequence for the blockwise conv mainloop: it is a negative result

Full stack, shipped 0.1200 -> B=64 0.1135. **A 5% relative improvement, far inside noise**, for
the 1.5-2.1x conv time measured in `docs/act_blockwise_2026-09-01` (21-37% of step time,
`docs/delta_dynamic_2026-09-02`).

The per-layer quantizer error table in `act_blockwise` is not wrong -- B=64 really is 16x better
at `||dequant(Q(v)) - v|| / ||v||`. It is **the wrong figure of merit**: the conv-input quantizer
is not a term in this model's error budget at int8, so making it 16x better changes nothing that
can be observed. Two independent measurements now agree -- 3.5x (static->dynamic, real kernels)
and 16x (static->B=64, sim) both bought zero.

`gemm_w8a8_blockk` and `conv2d_int8_blockk` should be recorded as **built, correct, and not worth
wiring in** on this model at W8A8. They stay valuable as artifacts: the mainloop is the only way
to express a sub-tensor activation scale for R,S>1, and the needle arm shows the granularity
question becomes real at int4, where the quantizer *is* a dominant term. That is the setting to
revisit them in -- W4A4, not W8A8.

## Where the quality actually is

Quantized W8A8 static attention is 2.9x the floor and everything else is invisible. The next
experiment is on the real path, not the sim: `MODIFF_STD_ATTN_BITS=0` leaves attention in fp16
math SDPA, which should take real-kernel relL2 from 0.0999 to somewhere near the floor. Its cost
is bounded by the attention bucket, 8.86 of 81.65 ms/step
(`docs/pipeline_profile_2026-08-31`, modulo the harness discrepancy noted in
`docs/delta_dynamic_2026-09-02`). `MODIFF_QUANT_ATTN_STATIC=0` (dynamic Q/K/V scales) is the
cheaper intermediate and has never been graded on this path.

That is a paired speed/quality question about a term worth 0.10, replacing a paired
speed/quality question about a term worth less than 0.035.

## Harness notes

`MODIFF_QUANT_ATTN=0` is NOT how to get fp16 attention. With `MODIFF_LINEAR=0` the gate is
`std_attn_bits in (4,8) and (not quant_lin or _force_qattn)` and `not quant_lin` is already True,
so attention stays quantized and the flag only flips it from STATIC to dynamic. It was tried
first and would have measured a different attention arm under an "fp16 attention" label.
`MODIFF_STD_ATTN_BITS=0` is the correct knob.

The sim needs **all five** fusion kill switches, not the two `_sim_guard`'s message named
(the o_hat-residual fold and the Upsample/AvgPool quantize folds have their own). The guard
message has been corrected to list all five. The guard itself worked exactly as designed --
it converted a silently-wrong measurement into a hard error, twice.

New knobs, both added for this budget: `MODIFF_ACT_SIM_EXACT_W=1` (exact instead of W8 conv
weights; must be set before the model is built, since `apply_static_scales` frees
`_orig_weight`) and `MODIFF_ACT_SIM_QMAX` (coarser activation grid, for the needle control).
`MODIFF_ACT_BLOCK=-3` is new and means activations exact.
