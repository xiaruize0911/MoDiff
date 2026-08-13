# The decoder skip-concat fold: built, measured at −1.59%, on by default

`cat2_channels_last_fp16` and the GroupNorm stats pass that consumes its output are now **one kernel**.
cat2 read the two halves and wrote their concatenation (2C of traffic); the stats pass then read that
concatenation (C). Reading the halves directly and emitting the concatenation from inside the stats
kernel costs 2C, because the read was already being paid.

| arm | ms/step (batch 128, DDIM 200) |
|---|---:|
| fold off | 60.44, 60.51 → **60.48** |
| **fold on** | 59.58, 59.45 → **59.52** |

**−1.59% end to end**, two runs per arm at 5 repeats, worst between-run spread 0.20% — the effect is
**8× the noise**. W4A4 goes **1.749× → 1.778×** vs fp16. The op-level saving is **51% of cat2**, which
is what the 3C→2C argument predicted (50%).

**Default ON** (`MODIFF_CAT2_FOLD=0` to disable), because the output is bit-identical where it runs and
the kernel provably never runs anywhere else.

## Why the risky half of the original plan was not needed

The plan in `docs/cat2_fold_2026-08-13/` (earlier) required **two** changes, because the concatenated
tensor has two consumers: the GN prologue and the 1×1 skip conv. Splitting the skip conv into
`W1·a + W2·b` was the expensive, uncertain half — its measurement was dominated by rows where one GEMM
timed *slower* than two.

Having the fold **emit** the concatenation removes that requirement entirely. The skip conv and the
out-conv residual receive exactly the tensor they received before. One kernel, no GEMM split, ~80% of
the projected benefit.

## What was verified, and how each check was made non-vacuous

| gate | what it asserts |
|---|---|
[`test_cat2_gn_fold.py`](../../integration/tests/test_cat2_gn_fold.py) | concat bit-identical to `cat2`; stats bit-identical to the contiguous path via `gn_stats_fp16`; deterministic over 10 launches; within 2e-7 of an **fp64** reference. All 9 real shapes, including the 4 where a GroupNorm group straddles the two buffers |
[`verify_cat2_fold_e2e.py`](../../integration/tests/verify_cat2_fold_e2e.py) | sampled latent bit-identical with the flag on **and** the fold kernel counted as actually called (570 blocks); 0 calls with the flag off |
[`test_cat2_fold_fallback.py`](../../integration/tests/test_cat2_fold_fallback.py) | for fp16 / int8 / int4_baseline the fold kernel runs **zero** times, so the flag cannot have changed them |
`test_kernel_correctness.py` | 16/16 after the refactor of the hot prologue |

Coalescing is preserved by construction, not by luck: a warp only straddles the `C1` boundary if
`C1 % 32 != 0`, and every width this UNet concatenates is 192/384/768. The kernel `TORCH_CHECK`s that
rather than assuming it. Summation order is untouched — only the *address* a value is loaded from
changes — which is why bit-exactness is the right bar and not a tolerance.

## Five mistakes, all mine, and what each one teaches

**1. A projection I disowned on one noisy sample.** The 3-repeat pass read −0.65% against a projected
−1.65%. I called the projection "2.5× optimistic" and supplied a mechanism (the stats kernel absorbing
the write cost). Re-running at 5 repeats gave **−1.59%**. The projection was right; I had reasoned a
story onto an outlier — the exact failure I had spent the day cataloguing.

**2. An eligibility probe that allocated what the optimisation exists to avoid.** The first version
built a full `[N, C1+C2, H, W]` tensor per block per step just to ask whether the fold applied. It
would have consumed the entire saving to decide whether to take it. It also crashed
(`new_empty` takes no `memory_format` here) — which is the *only* reason I looked. A silent version
would have shipped and shown no speedup. Replaced by `can_gn_fuse_modiff_cat2`, which decides from
shapes and flags and allocates nothing.

**3. A fallback check that could not justify its verdict — twice.** Attempt 1 compared flag-off against
flag-on per mode and reported fp16 BROKEN. Attempt 2 added a same-flag control and required
`flag-diff ≤ control-diff`; with one sample each, both are draws from the same ~4–6e-3 distribution, so
it flipped between runs (pass, then fail). **A criterion that reverses on re-run measures nothing.** The
version that ships asks the question with a deterministic answer: *did the kernel run?*

**4. `pkill -f "setup.py build_ext"`, twice, matched my own shell.** The harness wraps commands in a
`bash -c` whose argv contains the pattern. The first killed a heredoc mid-write, so an edit I believed
had landed hadn't — and the grep that "confirmed" it was fooled by `gn_stats_fp16` being a substring of
`cat2_gn_stats_fp16`. Match on argv *structure*, not on a substring of the whole line.

**5. A patch anchored on a block that appears twice.** The int8 sibling is textually identical, so the
assertion fired correctly — but the header and pybind edits had already landed, leaving a declaration
with no definition. Patch all files of a change atomically.

## A pre-existing bug found on the way

`fused_resblock.py:756` aliases one `nn.Conv2d` as both `fused.in_conv` and
`fused.original.in_layers[-1]`, so the int4 conversion wraps it **twice** — 70 orphaned conv wrappers,
never called, never calibrated, carrying `modiff_enabled = True`. Unrelated to this change and still
unfixed; recorded in `docs/attn_modiff_2026-08-13/FINDINGS.md`.

## And an open question closed

**fp16 sampling is nondeterministic across processes here**, relL2 ~4–6e-3, independent of any flag —
quantized modes are bit-reproducible (fixed CUTLASS kernels) while fp16 is not (cuDNN selects its
convolution algorithm per process). That is the "unidentified second source" that stopped
`docs/attn_modiff_2026-08-13`'s A/B from reproducing a committed reference and forced `fp16_refs.py` to
pin the references to disk. It is now explained rather than merely worked around.

## Reproducing

```bash
python integration/tests/test_cat2_gn_fold.py          # op-level bit-exactness, 9 shapes
python integration/tests/verify_cat2_fold_e2e.py       # e2e latents + the fold really ran
python integration/tests/test_cat2_fold_fallback.py    # other modes: zero fold calls
python integration/tests/bench_cat2_gn_fold.py         # the op-level 51%-of-cat2 measurement

MODIFF_CAT2_FOLD=1 E2EBENCH_MODES=int4 python integration/benchmarks/report/e2e_three_mode_bench.py \
  --batch 128 --steps 200 --repeats 5 --warmups 3 --output /tmp/fold_on.json
```
