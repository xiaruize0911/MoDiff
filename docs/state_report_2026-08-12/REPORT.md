# Current state: what the shipped defaults actually do

Every arm below runs the configuration a user gets from `benchmark_ldm.py --mode <X>` with no flags
— nothing here passes a calibration path or overrides an environment variable.

> **W4A4 changed substantially on 2026-08-12.** Two calibration constants
> (`DELTA_CLIP_RATIO = 8`, `ACT_CLIP_RATIO = 4.5`, both in `int4_optimized.py`, both swept, neither
> needing a kernel change) took W4A4 PTQ from 0.8642 to **0.4695** and W4A4 MoDiff from 0.6122 to
> **0.3090**. The full account, including the paper reproduction that made it findable and the three
> plan items that were *deprioritised on evidence*, is
> [`docs/paper_repro_2026-08-12/FINDINGS.md`](../paper_repro_2026-08-12/FINDINGS.md).
>
> **Read the CV column.** One earlier version of §2 was measured while a second CUDA process was
> profiling, and the CV went from ≤0.23% to 38% — which is how it was caught and discarded. Any
> latency table here without CVs under ~0.5% is contended and should not be quoted.
>
> **Noise floor**: a zero-change repeat moves W8A8 arms by 1.3–5.1% and W4A4 arms by 0.05–0.6%. Do
> not read a W8A8 difference under ~5% as an effect.

## 0. The configuration under test

| | setting | resolved from |
|---|---|---|
| activation scale | static, Q-Diffusion | `CALIBRATION_PREFERENCE` → `int8_calibration_qdiff.pt` / `int4_calibration_qdiff.pt` |
| delta step size (t<T) | static, Q-Diffusion, per-layer | `DELTA_CALIBRATION_PREFERENCE` → `int8_delta_qdiff.pt` / `int4_delta_qdiff.pt` |
| `MODIFF_DELTA_MODE` | `static` | kernel default |
| `MODIFF_LINEAR` | `0` — conv-only MoDiff | kernel default |
| SmoothQuant | **off** | Q-Diffusion exports are bare floats, so the fold never happens |
| int8 weights | per-output-channel absmax | `int8_optimized.py` |
| int4 weights | per-output-channel MSE clip search | `_int4_weight_scale`, `MODIFF_INT4_WSCALE=mse` |
| int4 activation grid | **deliberately clipped, ratio 4.5** | `ACT_CLIP_RATIO` |
| int4 delta grid | **deliberately clipped, ratio 8** | `DELTA_CLIP_RATIO` |
| attention | quantized, static, flash | `QUANT_ENV` — each harness sets its own copy; `MODIFF_QUANT_ATTN=1`, `MODIFF_QUANT_ATTN_STATIC=1`, `MODIFF_QATTN_FLASH=1` |

This is the paper's `--modulate --quant_mode qdiff --cali_min_max` path (README:96).

Hardware: NVIDIA A40, torch 2.4.1+cu124, real LSUN-churches checkpoint.

## 1. Samples

![samples](plots/samples.png)

Six samples per mode, DDIM 50 steps, seed 20260805, one column = one noise vector decoded five ways.
`relL2` in each row label is the latent error against the fp16 row, measured in the same process.

* **W8A8 holds.** Both the PTQ and MoDiff rows are the same buildings as fp16, down to the spires
  and the sky. MoDiff's row is the closer of the two.
* **W4A4 now produces buildings.** After the two clip ratios landed, the MoDiff row (relL2 0.3392)
  has legible cathedrals — spires, facades, the blue tower in column 4, the yellow massing in
  column 5 — and the PTQ row (0.4680) has structure emerging where it used to be uniform pastel fog.
  Before: PTQ 0.8571 was a wash and MoDiff 0.5906 was high-frequency scribble. Still short of W8A8,
  but no longer a different category of output.

## 2. End-to-end latency

Batch 128, DDIM 200 steps, 2 warm-up runs discarded, median of 3 repeats.

| mode | ms / step | ms / sample | vs fp16 | CV |
|---|---:|---:|---:|---:|
| fp16 | 106.22 | 165.97 | 1.000x | 0.20% |
| W8A8 PTQ | 73.27 | 114.49 | **1.450x** | 0.10% |
| W8A8 MoDiff | 74.82 | 116.90 | **1.420x** | 0.15% |
| W4A4 PTQ | 59.16 | 92.44 | **1.795x** | 0.13% |
| W4A4 MoDiff | 60.51 | 94.54 | **1.756x** | 0.32% |

Every arm has CV ≤ 0.32%, so the ordering is not noise.

**Latency did not move with the quality fixes**, and it should not have: `DELTA_CLIP_RATIO` and
`ACT_CLIP_RATIO` change numbers in a table, not kernel scheduling. Measured across three separate
clean runs this session the arms reproduce to ≤1.2%. That is what makes the fidelity gains in §3
free.

**MoDiff is nearly free in time**: 74.82 against its own baseline's 73.27 at 8 bits (+2.1%), 60.51
against 59.16 at 4 bits (+2.3%). That is a property of the shipped static delta path — the step size
comes from a table, so there is no per-call absmax reduction over the activation to pay for. The
modulation costs an add and a cache read.

**Four-bit is 1.24x faster than eight-bit** (59.16 against 73.27 on the PTQ arms). Whether that is
worth having is §3's question, not this one's.

![e2e](plots/01_e2e_speed.png)

## 3. Fidelity against speed

![tradeoff](plots/02_quality_vs_speed.png)

Latent relL2 against fp16, 3 seeds {1234, 20260805, 777}, batch 8, DDIM 50 — the protocol every
A/B in this tree uses, so these are the numbers to quote:

| mode | relL2 vs fp16 | was, session start | |
|---|---:|---:|---|
| W8A8 PTQ | 0.1138 | 0.2564 | 2.25× — Q-Diffusion activation scales |
| W8A8 MoDiff | **0.0605** | 0.0393 (dynamic) | static now matches dynamic |
| W4A4 PTQ | **0.4695** | 0.8642 | **1.84× — `ACT_CLIP_RATIO`** |
| W4A4 MoDiff | **0.3090** | 0.6122 | **1.98× — `DELTA_CLIP_RATIO`** |

The scatter above uses the single-seed values measured alongside the sample grid (0.1634 / 0.0650 /
0.4680 / 0.3392); the seed-to-seed spread is why they differ from the 3-seed table.

Read together with §2:

* **W8A8 MoDiff — 0.0605.** Still the best fidelity of any quantized arm.
* **W8A8 PTQ — 0.1138.** Nearly twice the error, at essentially the same speed.
* **W4A4 MoDiff — 0.3090, and it now beats its own dynamic arm** (0.4327), which is the reversal
  worth noting: at session start static cost 1.71× against dynamic at this bit width, and it now
  wins by 0.71×. "Static Q-Diffusion is a fidelity sacrifice at W4A4" is **retracted**.
* **W4A4 PTQ — 0.4695.** Structure rather than fog, still well short of W8A8.

**One cost of `ACT_CLIP_RATIO`, stated rather than buried**: the W4A4 *dynamic* arm regressed
0.3577 → 0.4327. It reads the static activation grid at t=T too and gains nothing from clipping it.
The shipped default is static, so the trade is right — it is not free for every configuration.

## 3a. Where W4A4's damage actually comes from

§3's W4A4 numbers are one figure over four stacked things: the 4-bit activation grid, the 4-bit
weights, the MoDiff recursion, and the int4 CUTLASS datapath. Fake quantization separates them by
running the ordinary fp16 model and simulating one piece at a time.

![fake quant](plots/w4a4_fake_quant.png)

| arm | relL2 vs fp16 |
|---|---:|
| fp16 | 0.0000 |
| **fake: activations 4-bit only** (weights fp16) | **0.9060** |
| **fake: weights 4-bit only** (activations fp16) | **0.2728** |
| fake: act + weight — W4A4 PTQ simulated | 0.8885 |
| fake: act + weight + MoDiff | 0.5235 |
| real int4 kernels, W4A4 PTQ | 0.8548 |
| real int4 kernels, W4A4 MoDiff | 0.6194 |

**The activations are the whole problem.** Quantizing only the weights leaves the churches standing
— dirtier texture, but structure, sky and spires all intact, at 0.2728. Quantizing only the
activations is already total fog at 0.9060, indistinguishable from full W4A4's 0.8885. This is worth
stating against expectation: the tree documents int4 weight reconstruction at 0.1254 median relative
Frobenius, which sounds like the dominant term, and at the output it is worth 0.27 against the
activation grid's 0.91.

**The kernels are faithful.** Simulated 0.8885 against real 0.8548 (PTQ), simulated 0.5235 against
real 0.6194 (MoDiff) — same ordering, 4–18% apart. So W4A4's loss is inherent to 4-bit arithmetic
rather than something the int4 datapath adds. It is also an independent check on the `ba8b8c9`
dequant fix: before it, the real MoDiff arm read 1.0469 against the simulation's 0.5235 — twice as
bad as the arithmetic allows — and it now lands next to it.

**MoDiff's gain is real, not an implementation artifact**: 0.8885 → 0.5235 in pure simulation.

One difference the numbers hide: the simulated and real MoDiff rows have similar relL2 but do not
*look* alike — simulation gives high-contrast collage, the kernels give low-contrast structure
emerging from fog. The harness accumulates `a_hat` in fp32 where the kernels accumulate `o_hat` in
fp16, which `act_fake_quant.py` names as its one idealisation. This is what that idealisation looks
like.

**Consequence for anyone optimising W4A4.** Better weight quantization (AdaRound, group-wise) is
bounded by that 0.27. The 4-bit activation grid is where the loss is, and there is a known unexplored
lever there: the qdiff scale sized to the true range (assumed absmax 3.77, no clipping) gives 0.86,
while the shipped absmax file's accidentally-5.13x-too-large scale — which clips 43% of channel peaks
— gives 0.71. Aggressive clipping helps at 4 bits and has never been searched for deliberately.

## 3b. The W4A4 activation quantizer is leaving 1.6 bits on the table

§3a put the damage on the activation grid. This is what is wrong with it, and what fixing it buys.

**The grid is symmetric; the data is not.** Every one of the 70 convs consumes `silu(gn(x))`, and
SiLU bottoms out at −0.2785 while being unbounded above. Measured on real activations at the shipped
W4A4 scale (`probe_int4_code_use.py`, 70 convs):

| | measured |
|---|---:|
| mass on negative codes (−7..−1) | 3.62% median |
| codes ever used | 9 of 15 |
| codes carrying >0.1% of the mass | **5 of 15** |
| \|max\| / \|min\| of the activation | **19.91×** |
| effective bits | **log₂(5) = 2.32**, against a nominal 3.91 |

Seven of fifteen codes cover a range 20× narrower than the one that matters. W4A4 is not running at
4 bits; it is running at about 2.3.

**Two fixes, measured separately** (act-only fake quant, weights fp16 so the grid is the only
variable, 3 seeds). Both families were swept over clip ratio, because clipping and the zero point
are separable and a single point cannot tell them apart:

| clip ratio | symmetric | asymmetric |
|---:|---:|---:|
| 1.00 | 1.1499 | 0.6542 |
| 0.25 | — | **0.3668** |
| 0.20 | 0.6589 | 0.3884 |
| 0.15 | **0.4519** | 0.4297 |
| 0.09 | 0.7202 | 0.6281 |
| *shipped grid* | *0.9294* | — |

Clipping alone is worth **2.06×** and is pure calibration. The zero point adds **1.23×** on top and
needs a kernel change — implementable, since `Σw(a_q − z) = Σw·a_q − z·Σw` and the second term is a
per-output-channel constant that folds into the bias, but a kernel change nonetheless.

**On the real int4 kernels, clipping reproduces** (W4A4 PTQ, 3 seeds, sweep extended downward until
the minimum was bracketed rather than stopping at the edge of the range):

| | relL2 |
|---|---:|
| shipped grid | 0.8908 |
| clip ×0.15 | 0.5312 |
| **clip ×0.12** | **0.4803** |
| clip ×0.10 | 0.4917 |
| clip ×0.08 | 0.6648 |

**1.85×**, and the samples go from fog to recognisable buildings. Note the two intermediate ratios
(0.25, 0.20) are *worse* than shipped and render with a strong orange cast — the saturated regime is
not a smooth degradation, and a coarse sweep straddling it reads the wrong winner.

**MoDiff barely benefits — 1.04×** — and that is mechanism, not noise: MoDiff reads the static
activation scale only at t=T, one step in fifty. This fix is for `int4_baseline`.

### The part that does not require leaving the paper

README:87 recommends `--cali_min_max` because it is "data-efficient and computation-efficient,
resulting in comparable results compared to MSE calibration". **MSE calibration is Q-Diffusion's own
option**, and at 4 bits the "comparable" claim does not hold:

| W4A4 PTQ | scale median | relL2 | vs shipped |
|---|---:|---:|---:|
| qdiff min-max *(shipped)* | 1.857 | 0.8934 | — |
| **qdiff MSE clip-search** | 2.541 | **0.5727** | **1.56×** |
| hand clip ×0.12 | 10.516 | 0.4705 | 1.90× |

So most of the win is available by **dropping one flag from the calibration command** — no kernel
change, no departure from the paper's own menu. `qdiff_bridge` §5b rejected this exact arm ("no —
1.5203, worse"); that measurement was taken through the 18.1× unit error and is void. This is a
second casualty of that bug: it did not only make W4A4 look worse, it disqualified a correct
direction.

**And a result worth keeping.** MSE's optimum sits at scale 2.54 while the end-to-end optimum is
10.5 — it clips **4× less** than it should. qdiff's search minimises *layer-wise activation
reconstruction MSE*, and at 4 bits that objective diverges from output error: reconstruction says
keep the tail, the trajectory says sacrifice it entirely. That gap is not a bug in either, it is the
wrong loss for the regime.

## 4. Per-block attribution

![blocks](plots/04_block_kinds.png)

Same batch and step count as §2, per-layer CUDA-event timing summed by kind.

| config | wall ms/step | attributed | conv | updown | attn (score) | proj (42 linears) |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 105.8 | — | — | — | — | — |
| **W8A8 PTQ** *(shipped)* | 73.2 | 46.6 | 22.7 | 3.9 | 20.0 | — |
| **W8A8 conv-only** *(shipped MoDiff)* | 80.5 | 67.8 | 40.9 | 6.8 | 20.1 | — |
| W8A8 conv+proj | 103.1 | 90.3 | 40.4 | 6.7 | 34.4 | 8.8 |
| W8A8 conv+proj +projK4 | 100.5 | 87.7 | 40.5 | 6.8 | 32.9 | 7.5 |
| W8A8 conv+proj +projK4 +routeB | 99.6 | 87.9 | 40.5 | 6.8 | 32.3 | 8.3 |
| W8A4 conv+proj | 102.3 | 90.0 | 40.1 | 6.7 | 34.4 | 8.8 |
| W4A4 conv+proj | 95.8 | 84.9 | 28.7 | 6.3 | 22.8 | 27.1 |

**Cross-check.** This harness and §2's are independent, and they agree: fp16 105.8 against 106.22,
W8A8 PTQ 73.2 against 73.27.

**Only the first three rows are shipped configurations.** The `conv+proj` rows have
`MODIFF_LINEAR=1`, which the tree defaults to `0`. They are here because the block profiler's grid
was built to study that flag — and they show why it is off: turning it on moves wall from 80.1 to
103.1 while attributing only 8.8 ms to the projections themselves. The 23 ms it costs does not land
where the flag applies.

**MoDiff's conv cost is where the modulation lives.** conv goes 22.6 → 40.9 ms turning MoDiff on
(PTQ → conv-only) while wall goes 72.8 → 80.1. The attributed conv time nearly doubles because a
modulated step runs the delta-quantize prologue plus the o_hat-accumulate epilogue, but most of that
overlaps work the PTQ arm was doing anyway.

**W4A4 shifts the balance to the projections.** conv drops 40.4 → 28.7 (4-bit GEMMs) but proj rises
8.8 → 27.1, because those 42 linears are on the W4A4 path where the int4 GEMM has no fused
o_hat-accumulate epilogue. That is the Stage B gap this tree has open, and it is the largest single
line in the W4A4 budget.

Per-layer, in UNet depth order:

![layers](plots/05_per_layer.png)

## 5. Per-kernel

Two different measurements, because they answer different questions.

**Profile — where the wall clock goes.** From a torch-profiler pass over the same sampling loop the
e2e numbers came from, bucketed:

![kernels](plots/03_kernel_buckets.png)

Share of profiled GPU kernel time:

| mode | GEMM / conv | GroupNorm+SiLU family | attention | elementwise / copy | other |
|---|---:|---:|---:|---:|---:|
| fp16 | 46.6% | 20.2% | 11.0% | 18.6% | 3.6% |
| W8A8 PTQ | 51.2% | 25.6% | 12.6% | 7.9% | 2.8% |
| W8A8 MoDiff | 51.4% | 25.2% | 12.4% | 8.3% | 2.7% |
| W4A4 PTQ | 39.8% | 32.3% | 14.7% | 9.8% | 3.4% |
| W4A4 MoDiff | 42.1% | 29.8% | 14.5% | 10.2% | 3.3% |

Two things about how these buckets are built, because both were wrong in a first pass:

* **The GroupNorm bucket absorbs the fused quantize prologues** (`group_norm_silu_quantize_*`,
  `gn_apply_delta_quantize_*`). They normalise and emit int8/int4 in one kernel, so the time is not
  separable — filing them under "quantize" would invent an overhead the model does not separately
  pay. A standalone-quantize bucket exists and is **empty on this model**.
* **Attention is matched before GEMM.** `pytorch_flash::flash_fwd_kernel`'s template arguments
  contain `cutlass::half_t`, so a GEMM-first ordering files fp16's entire attention cost — 1896 ms,
  11% of the run — as GEMM, and the fp16 row shows no attention at all.

The shape is stable: quantizing moves work out of `elementwise / copy` (18.7% → ~8%) and into the
fused GroupNorm path, and dropping to 4 bits shrinks GEMM's share while leaving everything else to
grow proportionally. GEMM never falls below 40%, so the conv/GEMM path is still the thing to
optimise at every precision.

**Benchmark — what each kernel costs in isolation.** `kernel_suites_bench.py` intercepts the real
call arguments at the C++ entry point during a live sample and replays them, so every kernel is
timed at the shapes this model actually runs. A module-level hook cannot do this: the fused
ResBlock calls `modiff_cutlass.*` directly, bypassing `forward()`.

Per-suite, summing each captured signature's median × its calls per sample, divided by the 5
captured steps (ms per denoising step):

| mode | conv | attention | linear | norm / quantize | other | signatures |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 54.78 | 12.99 | 5.81 | 18.59 | 7.66 | 84 |
| W8A8 PTQ | 30.73 | 10.52 | 9.60 | 29.06 | 2.05 | 126 |
| W8A8 MoDiff | 54.78 | 10.52 | 9.39 | 36.98 | 1.85 | 190 |
| W4A4 PTQ | 17.15 | 10.26 | 8.73 | 29.26 | 2.01 | 126 |
| W4A4 MoDiff | 30.89 | 10.27 | 8.59 | 35.25 | 1.86 | 190 |

**Do not read the MoDiff rows as per-step totals.** They carry 190 signatures against the baselines'
126 — roughly one extra per conv — because a MoDiff layer registers *both* a first-step entry and a
modulated-step entry, and those never both run on the same step. Summing them double-counts, which
is why conv appears to double while §2 measures MoDiff at +2%. The rows are comparable *within* a
column, not as totals.

Read that way: **the conv GEMM itself is where the bit width pays.** 30.73 → 17.15 ms going W8A8 →
W4A4 on the PTQ arms, a 1.79x on the conv suite alone, which is the whole of the 1.80x e2e speedup.
Attention is flat at ~10.2–10.5 ms across every quantized arm — it is already int8-flash in all of
them, so 4-bit buys nothing there — and `norm / quantize` is essentially flat too. Everything the
4-bit path gains, it gains in the GEMM.

## 5a. Against the paper's own W4A4

The README's `--modulate --quant_mode qdiff --cali_min_max` command was run verbatim (only `-n` and
`-l` changed) with the two inputs this tree had been missing — `cali_data/church.pt` from the paper's
HF dataset and `church_w4a8_ckpt.pth` from Q-Diffusion's Drive folder.

![paper](../paper_repro_2026-08-12/paper_w4a4_samples.png)

It produces clean churches. So the method was never in question, and the gap was entirely ours.
[`docs/paper_repro_2026-08-12/FINDINGS.md`](../paper_repro_2026-08-12/FINDINGS.md) has the full
account; the parts that matter for reading this report:

* **Four configuration deviations**, two of them self-inflicted. The one worth internalising: the
  `.pt` format has no slot for a zero point, so `--a_sym` was passed to the *calibration command* —
  a limitation in the innermost layer propagated all the way out to the reference invocation, and
  then the constrained thing was measured and called "the paper's method".
* **Importing the paper's per-layer delta values is worse than sweeping our own constant** — 0.2452
  against the 0.3090 the swept constant reaches. The optimum follows the trajectory, and ours is not
  theirs (different weights, EMA, calibration set, step count).
* **Three plan items were deprioritised on evidence rather than effort**: AdaRound weight import
  (our RTN+MSE already beats AdaRound at 0.1296 vs 0.1506 on weight reconstruction), the activation
  zero point (scoped to 6 CUDA kernels, but the only instrument able to price it failed its own
  self-check twice), and the coverage alignment (the claim that 35 emb linears were unquantized was
  inferred, not measured, and is withdrawn).
* **EMA and the paper's calibration set were measured and do not help** — W4A4 PTQ +2.0%, MoDiff
  **+72.1%** — so both flags stay opt-in. The bound on that: the aligned arm kept a clip ratio swept
  on the old trajectory, and the optimum follows the trajectory, so this rules out flipping the flags
  as a free win rather than ruling out EMA itself.
* **Still not aligned**: this report's W4A4 is 0.3090 relL2 with visible structure; the paper's is
  visually indistinguishable from fp16. With EMA and the calibration set eliminated, the remaining
  gap is the **activation zero point** and the **AdaRound weights**.

## 6. Reproducing

```bash
bash docs/state_report_2026-08-12/scripts/run_all.sh     # everything below, sequential, ~45 min
```

| step | script | writes |
|---|---|---|
| 1 | `scripts/sample_grid.py` | `plots/samples.png`, `data/samples_quality.json` |
| 2 | `integration/benchmarks/report/e2e_three_mode_bench.py` | `data/e2e.json` |
| 3 | `integration/tests/profile_layers_and_model.py` | `data/profile_layers.json` |
| 4 | `integration/benchmarks/report/kernel_suites_bench.py` | `data/kernel_suites.json` |
| 5 | `scripts/make_plots.py` | `plots/0*.png` |

Sequential on purpose: a second CUDA process during a long generation run has OOM'd the VAE decode
in this tree before.

`.pt` artifacts are gitignored and regenerable; scripts, `data/*.json`, `plots/*.png` and this file
are committed.
