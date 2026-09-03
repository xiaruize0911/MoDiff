# Shipping configuration: keep blockwise `a_hat`, revert the conv-input quantizer to per-tensor

This is the configuration the 2026-09-02 measurements point to. `a_hat` blockwise is a net win
(`docs/ahat_blockwise_2026-09-01`); the conv-input blockwise quantizer is not, at W8A8
(`docs/wa_budget_2026-09-02`: the term sits at the measurement floor once blockwise;
`docs/conv_blockk_e2e_2026-09-02`: it costs 21-37% of step time to remove it).

**No code revert was needed.** Every path added on 2026-09-02 is inert at its default
(`MODIFF_CONV_BLOCKK=0`, `MODIFF_ACT_BLOCK=0`, `MODIFF_ACT_SIM_*` unset), so this is env only:

```
MODIFF_AHAT_BLOCK=32   MODIFF_CONV_BLOCKK=0   MODIFF_ACT_BLOCK=0
MODIFF_LINEAR=0  MODIFF_AHAT_BITS=16  MODIFF_IMODE=0  MODIFF_DELTA_MODE=static
```

A40, batch 128. Conv/GN buckets from a **CUDA-only** profile over 10 steps (CPU+CUDA
double-counts: both the aten op and the kernel carry self device time). E2E is wall clock,
50 DDIM, CUDA events, median of 2 after 1 warmup. `scripts/`, `data/summary.txt`.

## Result

| arm | conv ms/step | **conv vs fp16** | GN/quant ms/step | E2E ms/step | E2E vs fp16 |
|---|---:|---:|---:|---:|---:|
| fp16 | 42.54 | 1.000x | 19.62 | 101.93 | 1.000x |
| W8A8 PTQ (no `a_hat`) | 32.03 | **1.328x** | 14.36 | 72.31 | 1.410x |
| W8A8 MoDiff, fp16 `a_hat` | 32.71 | **1.301x** | 21.52 | 81.64 | 1.249x |
| **W8A8 MoDiff, `a_hat` B=32 int8** | **32.94** | **1.291x** | **20.24** | **79.71** | **1.279x** |

## The conv is unchanged by the `a_hat` setting, which is the point

`MODIFF_AHAT_BLOCK` changes how `a_hat` is *stored* and the GN / quantize / commit kernels around
it. It does not touch the conv kernel, and the measurement confirms that: conv moves **+0.23 ms**
between the two MoDiff arms, inside noise. The blockwise `a_hat` win lands where it acts:

| | conv | GN/quant | E2E |
|---|---:|---:|---:|
| `a_hat` B=32 vs fp16 `a_hat` | +0.23 ms | **-1.28 ms** | **-1.93 ms (2.36%)** |

That reproduces `docs/ahat_blockwise_2026-09-01` (1.4-1.8% E2E there, 2.4% here) and its E2E
levels: 81.64 for the fp16-`a_hat` arm against 81.30-81.74 there, 79.71 against 80.15-80.30.
`a_hat` storage is also 1.125 B/elem against 2, a **1.78x smaller cache**.

## Two conv numbers, and why they differ

**In-model conv bucket: 1.29x.** **Isolated conv kernel, frequency-weighted over the 20 real UNet
shapes: 1.575x** (`docs/conv_shape_sweep_2026-09-02`). The in-model bucket is lower because it
includes ~9.9 ms/step of convs that are byte-identical in both arms -- the attention
`ImplicitGemmConvolutionFusionPerSample` (5.35) and several unquantized fp16 `sm8x_xmma` convs --
which dilute the ratio. Netting those out gives ~23.0 ms of quantized conv against ~32.6 fp16,
i.e. **~1.42x**, and the rest of the gap to 1.575x is the shape mix (`C <= 192` layers only reach
1.23-1.46x because the reduction is too shallow to fill the tensor cores).

Quote 1.29x for "the conv segment of this model" and 1.575x for "this conv kernel"; they answer
different questions.

## What this configuration gives up

Relative to the blockwise conv-input path measured in `conv_blockk_e2e`, this configuration keeps
**1.279x** E2E instead of falling to 1.155x, and keeps the conv at 1.29x instead of 1.007x. It
gives up the activation-granularity accuracy, which at W8A8 is worth going from 0.0382 to 0.0097
on a stack whose total is ~0.10 and is dominated by quantized attention at 0.1034 -- i.e. nearly
nothing. At W4A4 the same trade is 0.5181 -> 0.0415 and would be worth taking, which is why
`conv2d_int4_blockk` stays in the tree.

Note this arm is still 0.90x of PTQ (79.71 vs 72.31): MoDiff remains a quality play at W8A8, not
a speed one. Its relL2 is ~0.09-0.11 against PTQ's 0.32.

---

# Addendum 3 — kernel 1 alone, three a_hat arms, W8A8 + W4A4, all 20 UNet conv shapes

`scripts/bench_kernel1_arms.py` → `data/kernel1_arms.json`; table via `scripts/report_kernel1_arms.py`.
A40, CUDA events, median of 25 after 8 warmup, batch 128, num_groups=32, shapes taken from
`docs/conv_shape_sweep_2026-09-02/data/shape_sweep.json`'s `unet` dump (not a hardcoded list).

Peak = `torch.cuda.max_memory_allocated()` across (allocate that arm's state + one launch), so it
includes the persistent a_hat cache and its fp32 block scales, which is the only place the arms
differ. Verified against arithmetic at the largest shape (C=576, 32x32): W8A8 baseline = 144 MiB x
+ 72 MiB int8 out = 216, measured 216; a_hat fp16 adds 144 -> 361; a_hat int8 B=32 adds 72 + 9.4
(scales) -> 298. All rows agree.

## Freq-weighted totals (sum over the 20 shapes, weighted by occurrence count)

| arm | W8A8 ms | vs baseline | vs a_hat fp16 | W4A4 ms | vs baseline | vs a_hat fp16 |
|---|---|---|---|---|---|---|
| baseline (no MoDiff)  | 13.630 | 1.000x | 0.875x | 14.157 | 1.000x | 0.813x |
| MoDiff, a_hat fp16    | 11.922 | 1.143x | 1.000x | 11.511 | 1.230x | 1.000x |
| MoDiff, a_hat i8 B=32 | **10.755** | **1.267x** | **1.108x** | 11.773 | 1.203x | 0.978x |
| MoDiff, a_hat i8 B=16 | 16.234 | 0.840x | 0.734x | 13.208 | 1.072x | 0.872x |

Peak alloc at the largest shape (C=576 32x32 / C=384 32x32), MiB:

| arm | W8A8 | W4A4 |
|---|---|---|
| baseline | 216 / 144 | 180 / 120 |
| a_hat fp16 | 361 / 241 | 325 / 217 |
| a_hat i8 B=32 | 298 / 199 | 262 / 175 |
| a_hat i8 B=16 | 307 / 205 | 271 / 181 |

int8 a_hat B=32 cuts peak by 63 MiB (0.826x W8A8, 0.806x W4A4) at the largest shape — the a_hat
cache halves (144->72) and the block scales cost back 9.4.

## Three findings

1. **B=64 a_hat is not implemented, and the guard is deliberate.** Every B=64 call raised
   `along-C block size C/ng must be even and in [2,32]`
   (`csrc/modiff/common/ahat_cache.cuh:487`). The reason is in `ahat_block_resnap2`: the fresh
   along-C amax is a `__shfl_xor_sync` over B/2 lanes with `off < 16`, i.e. one warp at 2
   channels/thread caps B at 32. A vec4 (4 ch/thread) variant would put B=64 at exactly 16 lanes
   and is reachable, but it does not exist today. B=32 is also the granularity the addressing is
   specialized for (`i>>5`, no integer divide), and B=64 would only save half of the 9.4 MiB scale
   term while coarsening the quantizer — so this is a low-value gap.

2. **int8 a_hat B=32 is 1.108x FASTER than fp16 a_hat at W8A8 but 0.978x at W4A4**, at identical
   memory saving. Cause: `blk32` / `blk32_vec4` compile-time specialization exists only in
   `group_norm_silu_delta_quantize_nhwc` (lines 2586-2589, 2631-2705). `gn_delta_pack_impl`
   (3097-3308) passes `ahat_ng` through but never sets a B32 tag, so W4A4 runs the generic
   `c/B` divide path. Porting the tag is the fix; expected to recover ~2-13%.

3. **B=16 is a large regression at W8A8 (0.734x vs fp16 a_hat)** for the same reason — it misses
   `ahat_is_b32` and falls to the generic path — and the penalty is worst exactly where a_hat
   traffic dominates (C=384 32x32: 1.128 vs 0.798 ms). Do not use B=16.

Shape-level texture: MoDiff loses to baseline on the tiny launch-bound shapes (C=768 2x2:
0.032 vs 0.022) and on the widest spatial ones where a_hat traffic dominates (C=384 32x32:
0.798 vs 0.722); it wins 2x on the mid shapes (C=1152 4x4: 0.071 vs 0.167). int8 B=32 a_hat
turns the C=384 32x32 loss into a win (0.655 vs 0.722).

## Measurement bug found and fixed mid-run

The first pass allocated NHWC via `torch.zeros(...).contiguous(memory_format=channels_last)`,
which holds the NCHW original and the NHWC copy alive simultaneously — a 2x transient that
became the peak and hid the very fp16-vs-int8 a_hat difference being measured (it reported
baseline = a_hat-i8 = 288 MiB at C=576 32x32). Fixed to `torch.empty(..., memory_format=...)`
+ `.normal_()`/`.zero_()`. The timed lambdas also allocated `retire_count` on every call, on the
MoDiff arms only; hoisted. Times moved <2%, peaks moved a lot.

---

# Addendum 4 — kernel-1 response to each shape axis (N, C, H, W)

`scripts/kernel1_axis_sweep.py` → `data/kernel1_axis_sweep.json`; figures via
`scripts/plot_kernel1_axes.py` into `plots/`.

The 20-shape model dump in Addendum 3 cannot answer this: batch is fixed at 128 in it, and
H==W in every real UNet conv shape, so it separates only C and one spatial axis. This sweeps
batch / C / H / W independently around N=128 C=384 H=16 W=16, same style as
`docs/conv_shape_sweep_2026-09-02/scripts/shape_sweep.py`.

## Figures
- `kernel1_speedup_axes.png` — speedup over the no-MoDiff baseline, 2 precisions x 4 axes
- `kernel1_peak_axes.png` — peak allocated MiB, absolute, log-log
- `kernel1_peak_ratio_axes.png` — peak relative to baseline
- `kernel1_ahat_only_axes.png` — int8 a_hat vs fp16 a_hat at identical kernel structure
- `kernel1_model_shapes.png` — where the 20 real layers land (marker area = occurrence count)

## Speedup is strongly shape-dependent; peak memory is essentially shape-INDEPENDENT

| axis | speedup, W8A8 a_hat i8 B=32 | shape |
|---|---|---|
| batch N 8→256 | 0.85 → 1.05 → 1.21 → 1.38 → 1.53 → 1.61 | monotone up, parity at N≈16 |
| C 128→1536 | 2.37 → 1.78 → 1.70 → 1.53 → 1.32 → 1.19 → 1.09 → 1.05 | monotone down |
| H 2→64 | 1.17 → **2.46** → 2.00 → 1.52 → 1.25 → 1.09 | peaks at 4 |
| W 2→64 | 1.17 → **2.47** → 2.00 → 1.52 → 1.25 → 1.10 | same curve as H |

W4A4 has the same shape with a lower ceiling and crosses below parity: C≥1024 (0.97, 0.90)
and H=W=64 (0.93).

**H and W produce numerically identical curves** (0.798 vs 0.798 ms at 64, 1.52 vs 1.52x at 16).
Expected — NHWC makes the two a single flat spatial extent — and it means there are only three
independent axes here, not four.

Peak-memory ratio is flat in N to three digits (1.694 / 1.445 / 1.401 at W8A8 for fp16 / i8 B=16
/ i8 B=32 a_hat, unchanged from N=8 to N=256) and only mildly decreasing in C and H/W
(asymptote 1.67 / 1.42 / 1.38). Everything in the working set scales as N*C*H*W, so the ratio
cannot move; the mild slope at small H/W is fixed per-tensor overhead. W4A4's ratios are HIGHER
(1.83 / 1.53 / 1.48) because the int4 output is half the bytes, so the same a_hat is a larger
fraction of a smaller total.

## The "vs baseline" ratio conflates two changes — corrected with a fifth figure

The baseline runs the group-major single-pass `group_norm_silu_quantize_nhwc`; every MoDiff arm
runs `gn_group_stats_kernel` + the flat element-major `gn_apply_delta_quantize_flat_vec2_kernel`.
So "MoDiff 2.4x faster at C=128" is mostly **group-major vs flat**, not a_hat: at C=128 CPG is 4,
and a group-major block with 4 channels coalesces badly, while the flat apply is CPG-independent.
By pure traffic MoDiff should LOSE (baseline 3 B/elem: read x fp16 + write int8; MoDiff fp16 a_hat
7 B/elem: read x + read a_hat + write a_hat + write int8), which is exactly what happens once C
and H*W are large enough for the group-major kernel to coalesce well.

`kernel1_ahat_only_axes.png` holds the kernel structure fixed by comparing the int8-a_hat arms
against the fp16-a_hat arm. That ratio is clean and much flatter:
- **W8A8, B=32: 1.13-1.22x across every axis** once N≥16 — this is just halving a_hat's bytes.
- **W4A4, B=32: 0.96-1.00x everywhere** — the missing `blk32` tag in `gn_delta_pack_impl`
  (Addendum 3, finding 2) costs the whole benefit, uniformly, at every shape.
- B=16 is 0.70-0.82x (W8A8) / 0.85-0.92x (W4A4) everywhere. Confirmed dead.

## New mechanism found: the C=192 dip

W8A8 B=32 vs fp16 a_hat dips to 1.02x at C=192 while both neighbours (C=128: 1.15x, C=256: 1.19x)
are fine. Cause: `blk32_vec4 = blk32 && (CPG % 4 == 0)` (`group_norm_silu.cu:2589`). With
num_groups=32, CPG = C/32, and 192/32 = 6 is not a multiple of 4 — so C=192 alone in the swept
set loses the vec4 path and falls back to vec2.

This predicts the real model: the only two UNet channel counts with CPG not divisible by 4 are
**C=192 (CPG 6) and C=576 (CPG 18)**, and those are exactly the two weakest rows in Addendum 3's
per-shape table (C=192 32x32: 1.123x; C=576 32x32: 0.881x, the only large shape where B=32 a_hat
still loses to baseline). Making `blk32_vec4` reachable at CPG%2==0 rather than CPG%4==0 would
target 8 of the 70 conv layers.

---

# Addendum 5 — CORRECTION: Addenda 3 and 4 used the wrong baseline kernel

**Every "vs baseline" ratio in Addenda 3 and 4 was against a kernel the model does not run.**
The data files have been regenerated; the numbers below supersede them. The a_hat-arm-vs-a_hat-arm
ratios and all peak-memory numbers are unaffected.

## What went wrong

The baseline arm called `group_norm_silu_quantize_nhwc` / `..._pack_nhwc` directly. The model
reaches those through `_gnq()` in `integration/fused_ops/fused_resblock.py:107`, which appends
`_fast` whenever `MODIFF_GN_FAST != "0"` -- and it **defaults to "1"**. The shipped kernel is
therefore the `_fast` twin, which is 1.39x-4.42x faster:

| H (C=384, W=16, N=128) | group_size | generic block_size | generic GB/s | _fast GB/s | generic/_fast |
|---|---|---|---|---|---|
| 2 | 384 | 512 | 131 | 295 | 2.26x |
| 4 | 768 | 1024 | **90** | 396 | **4.42x** |
| 8 | 1536 | 1024 | 144 | 483 | 3.34x |
| 16 | 3072 | 1024 | 220 | 561 | 2.55x |
| 32 | 6144 | 1024 | 293 | 592 | 2.02x |
| 64 | 12288 | 1024 | 348 | 483 | 1.39x |

## What tipped it off: the H=4 speedup peak

Addendum 4 reported a speedup peaking at H=W=4 (2.46x) and falling either side, while the batch
axis rose monotonically. Both axes scale total elements identically, so no property of problem
size can make one rise and the other peak -- something had to be wrong with the denominator.

It was `csrc/baseline/norm/group_norm_silu.cu:552`:

    int block_size = 32;
    while (block_size < group_size && block_size < 1024) block_size <<= 1;

block_size is the next power of two >= `group_size = CPG*H*W`, capped at 1024. With C=384 and
num_groups=32 (CPG=12), group_size crosses 512 -> 1024 between H*W=32 and H*W=64, i.e. **exactly
at H=4**. On sm_86 the SM holds 1536 threads, so 1024-thread blocks give 1 resident block/SM
where 512-thread blocks give 3 -- a 1.5x loss of memory parallelism. Measured baseline bandwidth
drops 133 -> 90 GB/s across that crossing (1.48x, matching the 1.5x prediction) and then recovers
as per-block work amortizes the lower occupancy.

Decisive test: `_fast` pins block_size to 128/256/512 by a different heuristic, and its bandwidth
curve has **no dip at all** (295 -> 396 -> 483 -> 561 -> 592 GB/s, monotone). The peak was an
artifact of the generic kernel's occupancy cliff, not a property of MoDiff.

## Corrected result: MoDiff kernel 1 is ~1.7x SLOWER than baseline, as it must be

Both kernels are two-pass (group-major stats, then apply). Per element:

| arm | bytes/elem | predicted vs baseline | measured (freq-weighted) |
|---|---|---|---|
| baseline (`_fast`) | 5 = read x (2) x2 passes + write int8 (1) | 1.00x | 1.00x |
| MoDiff, a_hat fp16 | 9 = 5 + read a_hat (2) + write a_hat (2) | 0.56x | **0.581x** (W8A8) / 0.574x (W4A4) |
| MoDiff, a_hat i8 B=32 | 7.25 = 5 + 1 + 1 + 0.25 scales | 0.69x | **0.645x** (W8A8) / 0.561x (W4A4) |
| MoDiff, a_hat i8 B=16 | 7.5 | 0.67x | 0.424x / 0.498x (misses `ahat_is_b32`) |

Freq-weighted totals over the 20 shapes: W8A8 baseline 6.857 ms, a_hat fp16 11.803, i8 B=32
10.624, i8 B=16 16.166. W4A4 baseline 6.530, fp16 11.385, B=32 11.649, B=16 13.102.

This is the expected sign. MoDiff cannot win kernel 1 -- it reads and writes a_hat that the
baseline never touches. Its payoff is that the temporal delta has a smaller range, which buys
accuracy at a given bit width; the cost is 1.7x on this kernel plus the o_hat accumulate in
kernel 2.

## What survives unchanged

- **a_hat int8 B=32 vs a_hat fp16** -- identical kernel structure on both sides, so the baseline
  error cancels. 1.21x at the default shape against a byte-model prediction of 9/7.25 = 1.24x.
  Freq-weighted 1.111x (W8A8), dragged down by C=192/576; 0.977x (W4A4), the missing `blk32` tag.
- **All peak-memory numbers** -- the baseline kernel choice does not change any allocation.
- **The C=192 / C=576 `blk32_vec4` finding** (Addendum 4) -- an arm-to-arm observation.

---

# Addendum 6 — a_hat-only blockwise: E2E per block size, W8A8 + W4A4, with samples

Scope narrowed to **a_hat blockwise only** (no blockwise activation quantizer). `MODIFF_CONV_BLOCKK=0
MODIFF_ACT_BLOCK=0`, conv-input quantizer per-tensor as shipped; the only thing blockwise is a_hat
storage (int8 codes + fp32 scales [N,H,W,C/B]).

Measured: `scripts/e2e_samples.py`, one process per arm, batch 128, 50 DDIM, CUDA events, median
of 2 after 1 warmup, fixed seed 1234 for the decoded samples.
Data: `data/e2e_ahat_blocks.jsonl`, `data/`(mem breakdown printed by `scripts/mem_breakdown_blocks.py`).
Samples: `samples/compare_blocks.png` (8 rows).

## Result — every configuration clears 80% of the shipped MoDiff speed

"原来" here is MoDiff with fp16 a_hat, i.e. what ships.

| arm | ms/step | % of fp16-a_hat | peak alloc MB | steady-state cache MB | samples |
|---|---|---|---|---|---|
| W8A8 a_hat fp16 (原来) | 81.61 | 100.0% | 7852.63 | 1403.25 | ok |
| W8A8 a_hat i8 B=16 | 88.23 | 92.5% | 8038.52 | 877.03 | ok |
| **W8A8 a_hat i8 B=32** | **80.18** | **101.8%** | **7242.27** | 789.33 | ok |
| W8A8 a_hat i8 B=64 | 90.42 | 90.2% | 7876.79 | 745.48 | ok |
| W4A4 a_hat fp16 (原来) | 79.92 | 100.0% | 7344.51 | 1403.25 | ok |
| W4A4 a_hat i8 B=16 | 88.59 | 90.2% | 7574.91 | 877.03 | ok |
| **W4A4 a_hat i8 B=32** | **83.98** | **95.2%** | **6976.21** | 789.33 | ok |
| W4A4 a_hat i8 B=64 | 88.38 | 90.4% | 7423.34 | 745.48 | ok |

B=32 wins at both precisions: only it is faster than fp16 a_hat (W8A8) and only it reduces peak.
B=16 and B=64 both land at 90-93% -- above the bar, but dominated.

Why B=32 and not B=16/B=64: `blk32` and `blk32_vec4` are compile-time specializations keyed on
`ahat_is_b32(C, ng)` (`group_norm_silu.cu:2586-2589`). B=16 and B=64 miss them and run the generic
`c/B`-divide path, which costs 8-10 ms/step.

## Peak memory is non-monotone, and the cause is a missing kernel argument

Steady state is exactly monotone in B, as the byte count says it must be (a_hat 1 B/elem + 4 B per
B channels):

| block | a_hat MB | scales MB | cache MB | allocated after warmup | peak in sample | transient |
|---|---|---|---|---|---|---|
| fp16 | 1403.25 | 0 | 1403.25 | 5943.34 | 7853.02 | 1910 |
| B=16 | 701.63 | 175.41 | 877.03 | 5420.84 | 8038.52 | **2618** |
| B=32 | 701.63 | 87.70 | 789.33 | 5332.51 | 7242.38 | 1910 |
| B=64 | 701.63 | 43.85 | 745.48 | 5286.83 | 7876.79 | **2590** |

B=64 has the smallest cache (1.88x smaller than fp16) yet the second-largest peak. The ~700 MB
extra transient at B=16/B=64 is `_pack_ahat_along_c`'s eager fallback: `ahat_pack_block_nhwc` is
hardcoded `TORCH_CHECK(block == 32)` (`delta_quantize.cu:2076`), so any other B runs the Python
path, which materializes `a.permute(...).contiguous().float()` and then `blk / scale` -- two full
fp32 copies of one layer's a_hat. The largest layer is C=576 32x32 at batch 128 = 75.5M elements
= 302 MB per fp32 copy, ~604 MB live at once, which is the observed gap. It re-runs every sample
because `_forward_first_step` replaces the cache with a fresh fp16 tensor at t=T.

Generalizing that kernel's `block` argument would make B=64's peak the smallest, but it would only
close a 45 MB steady-state gap on a configuration that is already 12% slower. Not worth it;
**B=32 is the recommendation**.

## Two real bugs fixed to get W4A4 running at all

**1. int4 blockwise a_hat held its scale across writes.**
`gn_apply_delta_quantize_pack_flat_vec2_kernel` resolved the per-block scale on load,
dequantized, then re-stored through that same scale and never refreshed it. The int8 sibling has
an `ahat_block_resnap2` branch for exactly this; the int4 kernel was missing it. a_hat grows along
the trajectory, so the t=T scale saturated every code at +-127 and the decoded samples were finite
garbage. This is verbatim the failure `docs/ahat_blockwise_2026-09-01/FINDINGS.md` already records
under "Do not": *"Do not hold a_hat scales across writes... Held scales pass isolated kernel tests
and then produce rainbow noise end to end (relL2 2.22)"*. That fix was applied to the three int8
kernels only. Fixed here, plus a host `ahat_block_shuffle_ok` TORCH_CHECK so the stale-scale path
cannot be reached silently again.

Kernel gate after the fix, 10 sequential writes, a_hat relL2 vs the fp16-a_hat trajectory:

| B | int8 | int4 |
|---|---|---|
| 16 | 0.0207 | 0.0212 |
| 32 | 0.0235 | 0.0246 |
| 64 | 0.0259 | 0.0276 |

Monotone in B, and int4 now tracks int8 -- before the fix int4 was garbage.

**2. The int4 resize kernel refuses blockwise a_hat.**
`group_norm_silu_delta_quantize_resize_nhwc` TORCH_CHECKs `!(pack && ahat_i8 && ahat_ng > 0)`
(`group_norm_silu.cu:595`): with pack=true there is no int4-aware `ahat_commit_block` to fold the
codes back on a fresh amax. `_prequant_gn_resize_conv_modiff` now declines the fusion in that
combination (`fused_resblock.py`), so the 8 updown ResBlocks take the unfused route, whose
`step1_static_quantize_pack_int4_fprop` does bind `ahat_ng`. That cost is inside the W4A4 numbers
above.

## And it was never wired for int4 in the first place

`git log -S"AHAT_BLOCK" -- integration/kernels/int4_optimized.py` is empty: the knob never existed
on the int4 path. The whole feature landed in one commit (`31de3067`) touching `int8_optimized.py`
only, and `OptimizedInt4Conv2d` is a standalone nn.Module, not a subclass, so it carried its own
per-tensor-only `_ahat_qscale`. Added here: `_ahat_block`, `_pack_ahat_along_c`, blockwise-aware
`_ahat_want_int8` / `_ensure_ahat_qscale` / `_pack_ahat_int8` / `_maybe_quantize_ahat`.

None of this was a hardware limit. The B<=32 cap in `bind_ahat_cache` was also software: at 2
channels/thread B=64 spans 32 lanes = exactly one warp, so lifting it was a loop bound
(`off < 16` -> `off < 32`, with the shuffle moved inside the warp-uniform guard so B=32 does not
pay an extra shuffle) plus two TORCH_CHECK bounds.

---

# Addendum 7 — the W4A4 blockwise number was an implementation gap, twice over

1.222x vs fp16 for W4A4 a_hat-B=32 was not credible: it made W4A4 blockwise (83.98 ms/step)
SLOWER than W8A8 blockwise (80.18) while doing half the GEMM work, when PTQ-vs-PTQ has W4A4
ahead 59.94 vs 72.90. Two causes, both found and one fixed.

## Cause 1 (fixed): the int4 apply kernel had no B=32 specialization

`ahat_b32_update2` / `ahat_b32_read2` do the whole read-delta-quantize-resnap with one
reciprocal, no `C/ng` integer division, and one byte-pair load + store.
`gn_apply_delta_quantize_flat_vec2_kernel` (int8) has selected them on a compile-time `AhatB32`
tag since 2026-09-01. `gn_apply_delta_quantize_pack_flat_vec2_kernel` (int4) did not exist in
that dispatch at all -- it always ran `ahat_resolve` + `ahat_load2` + `ahat_block_resnap2`, i.e.
the generic divide path.

Templated it on `AhatB32` and added the `blk32 = ahat_i8 && ahat_is_b32(C, ahat_ng)` dispatch in
`gn_delta_pack_impl`. Numerically equivalent -- a_hat relL2 over 10 sequential writes against the
fp16-a_hat trajectory is 0.0245 after vs 0.0246 before at B=32 (and B=16 0.0212 / B=64 0.0276
unchanged, since those keep the generic path).

**W4A4 a_hat i8 B=32: 83.98 -> 81.07 ms/step (2.91 ms recovered).**

## Cause 2 (open, and it is the whole remainder): the resize fusion had to be declined

`_prequant_gn_resize_conv_modiff` must decline int4+blockwise (Addendum 6, cause 2), so the 8
updown ResBlocks lose their fused GN+resize+delta-quantize kernel. Measured directly with a
`MODIFF_NO_RESIZE_FUSE=1` knob that forces the same decline for every arm:

| W4A4, a_hat fp16 | ms/step |
|---|---|
| resize fusion ON (default) | 79.50 |
| resize fusion OFF | 81.72 |

So the fusion is worth **2.22 ms/step**, and the blockwise arm cannot have it.

Comparing like for like -- both arms without the resize fusion:

| | ms/step | vs a_hat fp16 |
|---|---|---|
| W4A4 a_hat fp16, no resize fuse | 81.72 | 1.000x |
| W4A4 a_hat i8 B=32 (never has it) | 81.07 | **1.008x** |

**Blockwise a_hat is faster than fp16 a_hat at W4A4 too, by 1.008x**, once the resize fusion is
held equal -- consistent in sign with W8A8's 1.022x. The apparent 0.981x regression is entirely
the fusion the blockwise arm is denied, not the blockwise scheme.

Closing it needs an int4-aware `ahat_commit_block` that reads packed nibbles, so
`group_norm_silu_delta_quantize_resize_nhwc` can drop its
`TORCH_CHECK(!(pack && ahat_i8 && ahat_ng > 0))`. Worth 2.2 ms/step on W4A4.

Also still missing on the int4 path: the `blk32_vec4` specialization (int8 has a 4-channels-per-
thread B=32 kernel for CPG%4==0). For packed int4, 4 channels = 2 bytes, so the analogue exists;
it was not attempted here.

## Corrected table (all arms re-measured on the same binary)

| arm | ms/step | vs fp16 | peak alloc MB | vs fp16 | a_hat cache MB |
|---|---|---|---|---|---|
| fp16 | 102.63 | 1.000x | 4306 | 1.00x | — |
| W8A8 PTQ | 72.90 | 1.408x | 4573 | 1.06x | — |
| W8A8 MoDiff, a_hat fp16 | 81.57 | 1.258x | 7854 | 1.82x | 1403 |
| W8A8 MoDiff, a_hat i8 B=32 | 79.80 | 1.286x | 7245 | 1.68x | 789 |
| W4A4 PTQ | 59.94 | 1.712x | 4386 | 1.02x | — |
| W4A4 MoDiff, a_hat fp16 | 79.50 | 1.291x | 7339 | 1.70x | 1403 |
| W4A4 MoDiff, a_hat i8 B=32 | 81.07 | 1.266x | 6981 | 1.62x | 789 |

Note MoDiff still costs W4A4 its PTQ speed advantage (59.94 -> ~80): a_hat's read+write and the
o_hat accumulate are paid at fp16-activation resolution regardless of the conv's bit width, so
W8A8 and W4A4 MoDiff converge to nearly the same ms/step. That is a property of the scheme, not
of this change.

---

# Addendum 8 — the resize fusion's 2.22 ms recovered for int4

Addendum 7 attributed W4A4 blockwise's whole remaining deficit to the declined resize fusion.
Implemented the missing piece.

## What was missing

`group_norm_silu_delta_quantize_resize_nhwc` is group-major (one block per (sample, GN group)),
and a B=32 along-C group does not nest inside a GN group at any CPG here, so it cannot resnap
a_hat inline -- it leaves a_hat alone and a separate pass folds the delta codes in with a fresh
per-block amax. That pass, `ahat_commit_block`, could only read **int8** codes, so `pack=true`
was TORCH_CHECK'd out and `_prequant_gn_resize_conv_modiff` had to decline the fusion for the
8 updown ResBlocks.

Added `ahat_commit_block_pack4` (`ahat_cache.cuh`): the same fold reading the **packed nibble**
layout every int4 producer writes (`(i0 & 0x0F) | ((i1 & 0x0F) << 4)`, even channel in the low
nibble, signed 4-bit). Two kernels, a B=32 fast path (a 32-channel group is 16 bytes and cannot
straddle a pixel row because C%32==0) and a generic-B one. Addressing takes `Kpad` explicitly
rather than assuming `i>>1`, since the packed row stride is `Kpad/2` and `Kpad != C` whenever a
caller asks for GEMM-alignment padding. Dropped the TORCH_CHECK, made `block_commit` precision-
agnostic, removed the Python decline.

## Correctness

Resize path, int4 + blockwise a_hat, 10 sequential writes, a_hat relL2 against the fp16-a_hat
trajectory:

| B | resize UP | resize DOWN |
|---|---|---|
| 16 | 0.0215 | 0.0242 |
| 32 | 0.0250 | 0.0288 |
| 64 | 0.0279 | 0.0317 |

Same 0.02-0.03 range as the non-resize path (0.0212 / 0.0245 / 0.0276), monotone in B, both
directions. Samples indistinguishable from a_hat fp16 (`samples/compare_resizefix.png`).

## Speed

**W4A4 a_hat i8 B=32: 81.07 -> 79.88 ms/step.** Net 1.19 ms of the 2.22 ms the fusion is worth --
the remaining ~1.03 ms is the commit pass itself, which the fp16-a_hat arm does not need because
it folds a_hat inline. So the fusion is recovered but not free at blockwise; closing the last
1.03 ms would need the resize kernel split into a stats pass plus a pair-major apply, which is
exactly what `docs/ahat_blockwise_2026-09-01/FINDINGS.md` flagged as still on the table for int8.

Cumulative on the W4A4 blockwise arm: **83.98 -> 81.07 (blk32 tag) -> 79.88 (pack4 commit)**,
4.10 ms/step or 1.051x from the two fixes.

## Final table (all arms, same binary, batch 128, 50 DDIM, A40)

| arm | ms/step | vs fp16 | peak alloc MB | vs fp16 | a_hat cache MB |
|---|---|---|---|---|---|
| fp16 | 102.63 | 1.000x | 4306 | 1.00x | — |
| W8A8 PTQ | 72.90 | 1.408x | 4573 | 1.06x | — |
| W8A8 MoDiff, a_hat fp16 | 81.57 | 1.258x | 7854 | 1.82x | 1403 |
| **W8A8 MoDiff, a_hat i8 B=32** | **80.25** | **1.279x** | **7241** | 1.68x | 789 |
| W4A4 PTQ | 59.94 | 1.712x | 4386 | 1.02x | — |
| W4A4 MoDiff, a_hat fp16 | 79.55 | 1.290x | 7346 | 1.71x | 1403 |
| **W4A4 MoDiff, a_hat i8 B=32** | **79.88** | **1.285x** | **6980** | 1.62x | 789 |

Blockwise vs its own a_hat-fp16 arm: **W8A8 1.016x / -612 MB peak**, **W4A4 0.996x / -366 MB**.
W4A4 blockwise is now at parity on time and ahead on memory, and the W4A4/W8A8 ordering is no
longer inverted (79.88 vs 80.25).

MODIFF_NO_RESIZE_FUSE=1 is kept in `fused_resblock.py` as a measurement knob -- it forces the
same decline for every arm, which is how the 2.22 ms was attributed rather than inferred.

---

# Addendum 9 — 4-bit a_hat does NOT work, at any useful block size. Do not build the nibble datapath.

Asked for W4A4 with a_hat at i4 B=32. Answered by simulation before building the datapath,
because the datapath is expensive: packed nibbles need `ahat_b32_update2_i4` /
`ahat_b32_read2_i4`, an int4 `ahat_resolve`, int4 `step1_*` a_hat paths, an `[N,C/2,H,W]` cache
and the ~10 `a_hat_cache.sizes() == x.sizes()` checks that implies.

## The simulation, and why it can be trusted

`MODIFF_AHAT_SIM_BITS=b` (int4_optimized.py) keeps a_hat in fp16 but snaps it along C in groups
of `MODIFF_AHAT_BLOCK` to qmax = 2^(b-1)-1 after every write. That is exactly the arithmetic real
packed storage does -- each write resnaps from the new per-group amax. Timing is meaningless
(the snap is eager Python: 163 vs 80 ms/step); only the images matter.

Validated: **sim 8-bit B=32 is visually identical to the real int8 B=32 arm** (sample_mean
0.4882 vs 0.4884). So the sim's verdict at 4 bits is credible.

## Result

| a_hat | B/elem | vs fp16 | samples |
|---|---|---|---|
| fp16 | 2.000 | 1.00x | ok (reference) |
| int8 B=32 (real) | 1.125 | 1.78x | ok |
| sim 8-bit B=32 | 1.125 | 1.78x | ok — validates the sim |
| **sim 4-bit B=32** | 0.625 | 3.20x | **collapsed into shard texture** |
| **sim 4-bit B=16** | 0.750 | 2.67x | **still fragmented, unusable** |
| **sim 4-bit B=8** | 1.000 | 2.00x | **still fragmented, unusable** |
| sim 3-bit B=32 | — | — | pure noise |

`samples/compare_ahat_bits.png`, `samples/compare_i4_blocks.png`.

Refining the block does not rescue it. And it cannot: to beat int8 B=32's 1.125 B/elem, int4
needs B >= 8 (int4 B=4 is 1.5 B/elem, worse than int8 B=32). B=8, 16 and 32 all fail on quality,
so **every int4 configuration that would save memory is broken, and every configuration fine
enough to maybe work saves nothing.** Dominated on both axes.

## Why, and why this is not a contradiction with 4-bit activations

a_hat is a **cumulative accumulator over 50 denoising steps**, not a one-shot tensor. Each write
re-rounds the already-accumulated value, so the quantization error is fed back into the state and
compounds across steps. The conv's activation quantizer can be 4-bit because its error is
consumed once and never re-enters the state; a_hat's cannot. This is the same mechanism as the
"Do not hold a_hat scales across writes" note in `docs/ahat_blockwise_2026-09-01/FINDINGS.md` --
a_hat is the one tensor here where error accumulates.

## Better next step than i4, if a_hat memory is the goal

Make the block scales **fp16 instead of fp32**. That halves the scale term at zero quality risk
(the scale is a per-32-channel amax/127, well inside fp16 range) and needs no new datapath:

| a_hat | fp32 scales | fp16 scales |
|---|---|---|
| int8 B=32 | 1.125 B/elem (1.78x) | 1.0625 (1.88x) |
| int8 B=64 | 1.0625 (1.88x) | 1.031 (1.94x) |

Not implemented here; flagged as the cheaper remaining win.

---

# Addendum 10 — packed-int4 a_hat: built, measured. Memory wins, speed does not.

Addendum 9 argued against building this from the simulated quality. Built anyway on request
("不考虑误差先"), so the speed and memory are now measured rather than predicted.

## What was built

- `ahat_b32_update2_i4` / `ahat_b32_read2_i4` (`ahat_cache.cuh`): byte-for-byte twins of the int8
  B=32 pair, except a_hat is 4 bits per channel. One thread owns 2 consecutive channels ==
  exactly ONE byte, so load and store are single-byte and the 32-channel group is 16 bytes over
  16 lanes -- the warp geometry `ahat_group16_amax` already reduces over. Storage 0.5 B/elem +
  4 B per 32 channels = **0.625 B/elem** (int8 B=32 is 1.125, fp16 is 2.0).
- `AhatI4` template parameter on both apply kernels (`gn_apply_delta_quantize_flat_vec2_kernel`
  and `gn_apply_delta_quantize_pack_flat_vec2_kernel`), with `launch_*_i4` dispatch.
- Host detection from the shape: an int8 `[N,C/2,H,W]` channels_last cache IS the nibble layout
  (byte index = nhw*(C/2) + c/2), so no new flag. Bound by hand because `bind_ahat_cache` derives
  B from `a_hat_cache.size(1)`, which is C/2 here. `blk32` and `blk32_vec4` explicitly exclude it.
- Scoped to the two apply kernels. NOT wired for E2E: the `step1_*` a_hat paths, the resize
  commit, the Python allocation and the ~10 `a_hat_cache.sizes() == x.sizes()` checks are
  untouched, so there is no model-level arm.

Gate (10 sequential writes, a_hat relL2 vs the fp16-a_hat trajectory):

| a_hat | W8A8 | W4A4 |
|---|---|---|
| i8 B=32 | 0.0235 | 0.0245 |
| i4 B=32 | **0.1557** | **0.1499** |

6.4x more error, consistent with Addendum 9's simulated collapse.

## Speed: i4 is SLOWER than i8, and lands back at fp16-a_hat

Frequency-weighted over the 20 UNet conv shapes, time multiple vs each precision's PTQ baseline:

| arm | ms | x baseline | peak MB | x baseline | a_hat cache MB | B/elem |
|---|---|---|---|---|---|---|
| W8A8 PTQ | 6.9295 | 1.000x | 216 | 1.000x | — | — |
| W8A8 a_hat fp16 | 11.9295 | 1.722x | 361 | 1.671x | 1248 | 2.000 |
| W8A8 a_hat i8 B=32 | **10.7600** | **1.553x** | 298 | 1.380x | 702 | 1.125 |
| W8A8 a_hat i4 B=32 | 11.9619 | 1.726x | **262** | **1.213x** | **390** | 0.625 |
| W4A4 PTQ | 6.5967 | 1.000x | 180 | 1.000x | — | — |
| W4A4 a_hat fp16 | 12.0488 | 1.826x | 325 | 1.806x | 1248 | 2.000 |
| W4A4 a_hat i8 B=32 | **11.8373** | **1.794x** | 262 | 1.456x | 702 | 1.125 |
| W4A4 a_hat i4 B=32 | 12.1102 | 1.836x | **226** | **1.256x** | **390** | 0.625 |

i4/i8 time: **1.112x (W8A8) / 1.023x (W4A4)** weighted; 1.028x / 1.026x at the largest layer.
i4/fp16-a_hat: 1.002-1.005x everywhere -- i4 gives back the entire int8 gain.

The byte model predicts i4 should be 6.25/7.25 = 0.862x of i8's time. It is 1.11x. So this
kernel is not purely bandwidth-bound at 6.25 B/elem, and two things cost more than the bytes save:

1. **No vec4 variant.** int8 B=32 has `blk32_vec4` (4 channels/thread) on CPG%4==0 layers; the i4
   path is vec2-only. At the largest layer C=576 (CPG 18, vec4-ineligible for BOTH arms) i4/i8 is
   only 1.028x, versus 1.112x weighted -- so most of the weighted gap is exactly the vec4 layers.
2. **Single-byte access and branchy nibble decode.** int8 loads/stores an `unsigned short` (2 B)
   and converts with the magic-number `ahat_byte_to_f(a, 0x7640u)`; i4 does a 1-byte access plus
   `v > 7 ? v - 16 : v` sign-extension per nibble. Same total bytes, worse transactions and more ALU.

**The fix for both is the same kernel, and it is not built: 4 channels of int4 = exactly 2 bytes.**
A vec4 i4 kernel would have int8-vec2's 2-byte access AND vec4's 4 channels/thread, plus a
`__byte_perm`-based nibble decode. That is the configuration that could plausibly make i4 the
fastest arm. Without it, i4 a_hat is dominated: worse quality (Addendum 9) and slower, better
only on memory.

## Memory: the win is real

a_hat cache 1248 -> 702 (i8) -> **390 MB** (i4) over the 20 shapes, i.e. **3.2x smaller than
fp16**. Peak at the largest layer 361 -> 298 -> **262 MB** (0.73x of fp16-a_hat). Every number
matches the B/elem column exactly.
