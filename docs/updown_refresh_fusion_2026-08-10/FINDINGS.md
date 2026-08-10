# The updown fusion had never fired at K=1, and what retiring the code ceiling cost

2026-08-10. Two pieces of work, in the order they happened, because the second came out of a
constraint the first exposed.

1. `_prequant_gn_resize_conv_modiff` declined on every dynamic *refresh* step. At
   `MODIFF_DELTA_REFRESH=1` — the paper's own configuration — every step is a refresh, so the
   eight updown ResBlocks' fusion fired **0 times out of 8**, not "3 of 4 steps" as its docstring
   implied. Fixed, and worth **+1.24 ms/step** at K=1 measured paired in one process.
2. The `float code_ceiling` that made W8A4 a genuine 4-bit datapath was retired in favour of a
   `bool a4` naming the datapath, and the activation surface collapsed to **W8A8 / W8A4 / W4A4**.
   The intermediate widths (A7/A6/A5/A3/A2) and `MODIFF_DELTA_CLIP` are gone.

Nothing here is a quality statement. Every number is timing or code identity.

## Part 1: the fusion that never fired

### What was wrong

`group_norm_silu_delta_quantize_resize_nhwc` took its scale as a device pointer and computed no
absmax, so it could not serve a step that has to *measure* the delta range. The caller therefore
returned `None` whenever `conv._delta_should_refresh()` was true. Its docstring framed the cost as
"one step in four falls back", which is true at the default K=4 — and completely wrong at K=1,
where every step refreshes.

The 2026-08-07 traces had recorded the consequence without anyone reading it that way
(`docs/component_attribution_2026-08-07/data/trace_buckets.json`):

| kernel | int8_ptq | modiff K=4 | modiff K=1 |
|---|---:|---:|---:|
| `group_norm_silu_[delta_]quantize_resize_nhwc` | 0.87 / 8 calls | 2.59 / 6 | **0.00 / 0** |
| `upsample_nearest2d_nhwc` (unfused) | 1.24 / 4 | 1.56 / 5 | **2.49 / 8** |
| `avg_pool2d_nhwc` (unfused) | 0.45 / 4 | 0.57 / 5 | **0.91 / 8** |
| `group_norm_silu_nhwc` (standalone) | 0 | 0.40 / 2 | **1.59 / 8** |

So the "8 convs on the unfused `_forward_modulated`" in the 2026-08-06 fusion audit were not a fixed
property of the model. They were the eight updown ResBlocks, unfused *because* K=1. Part of the
6.78 ms that separates K=4 from K=1 is this fusion dying, not only the extra absmax passes.

### The fix

The kernel now carries the same dynamic-scale contract as its non-resize sibling
`group_norm_silu_delta_quantize_nhwc`:

* **`reduce_only`** turns the same kernel body into its own reduction-only twin — identical GN, mod,
  SiLU, SmoothQuant, resize and a_hat arithmetic, storing nothing, reducing `max|delta|` and
  publishing `Q_level/absmax` through the existing `gn_report_delta_absmax`. Sharing one body rather
  than hand-writing a twin is what guarantees the measured range is the range the quantize pass then
  sees; a copy would drift the first time either changed.
* **`report_next`** so the quantizing pass can publish for a later step at no extra pass.

`_prequant_gn_resize_conv_modiff` passes `_delta_gn_dynamic_args` straight through and allocates the
four reduction buffers itself (`_ensure_delta_dyn_bufs`, added to `OptimizedInt8Conv2d`; `Int4Conv`
already had one). That buffer allocation was the other reason the old path required an unfused step
to have run first.

Two launches on a refresh step, but still one fewer full pass over `x` than the unfused route, and
no fp16 resized intermediate at all — which for the four UP layers is 4× the input.

`MODIFF_UPDOWN_FUSE_REFRESH=0` restores the old decline. It exists because the win is small enough
that cross-session drift is the same order, so the honest measurement needs both behaviours
available at once; it doubles as the revert switch.

### Kernel correctness (`data/raw_measurements.md`)

`integration/tests/test_gn_resize_delta_dynamic.py`, all 8 real updown shapes × int8/int4, PASS:

| check | result |
|---|---|
| static 14-arg form still bit-identical | ok on all 16 |
| published scale vs an fp32 reference absmax | rel err 3.7e-09 – 1.3e-07 |
| codes at that published scale | max err ≤ 1 (rounding ties) |
| a_hat advanced exactly once per call | within one code step |
| absmax / retire self-reset | ok |
| `report_next` publishes correctly | ok |

The a_hat check is the one that matters structurally: if the reduction launch also wrote a_hat or
wrote codes, that is where it shows up.

### Wiring (`data/raw_measurements.md`)

`integration/tests/test_updown_fusion_modiff_k1.py`, real checkpoint, MoDiff mode. "Per step" is
derived from a counted UNet-forward count, not assumed from `S` — the sampler runs its own number of
timesteps once warm-up and the DDIM schedule are in play.

| K | UNet forwards | fused calls | fused/step | want |
|---|---:|---:|---:|---:|
| 1 | 7 | 56 | **8.00** | 8 |
| 4 | 7 | 56 | **8.00** | 8 |

Was 0/8 at K=1 and 6/8 at K=4. A test that only checked "> 0" would have passed against the bug.

### Speed: why the paired A/B, and not the cross-session table

Measured cross-session against 2026-08-07 first (`data/differential_timing_after.json`), with
`route_check` identical on all four arms:

| arm | before | after | delta | CV% |
|---|---:|---:|---:|---:|
| `int8_ptq` | 73.61 | 73.18 | −0.43 | 0.50 |
| `modiff_conv_k1` | 84.17 | 83.16 | −1.01 | 0.20 |
| `modiff_full_k1` | 106.59 | 105.55 | −1.05 | 0.08 |
| `modiff_full_k4` | 99.81 | 99.99 | +0.19 | 0.15 |

**`int8_ptq` contains no MoDiff resize path at all and still moved −0.43 ms.** That is session drift
— different container, real checkpoint instead of the 856-byte stub — and it is the same order as
the effect. Subtracting one from the other is not a measurement, and on that table K=4 reads as a
possible small regression.

So both arms were re-measured in ONE process, alternating ON/OFF on the same model object, each ON
differenced against the OFF immediately after it (`integration/tests/ab_updown_fusion_refresh.py`,
`data/raw_measurements.md`):

| | ON (fix) | OFF (before) | paired delta | fused/step ON → OFF |
|---|---:|---:|---:|---|
| **K=1** | 105.32 | 106.56 | **+1.24 ± 0.13** | 8.00 → 0.00 |
| **K=4** | 99.60 | 99.99 | **+0.40 ± 0.20** | 8.00 → 6.00 |

All four paired repeats are positive at both K, so **K=4 is not a regression** — that reading was
drift-estimation noise. Two things make this trustworthy where the cross-session table was not: each
arm counts the fused entry, so it provably *is* the arm it claims; and both OFF arms reproduce the
historical baselines (106.56 vs 106.59, 99.99 vs 99.81), which says the revert flag is faithful.

`modiff_full_k1`: **106.59 → 105.32 ms/step.**

### The kernel microbenchmark understates it, and that is expected

`data/raw_measurements.md`, batch 128, the eight shapes, the four-kernel unfused route vs fused:

| | unfused | fused dyn | fused static (K>1 reuse) |
|---|---:|---:|---:|
| all 8 | 4.660 ms | 4.211 ms | 3.284 ms |

+0.45 ms, against +1.24 ms end to end. The microbenchmark times the kernels on one hot tensor set;
in the full model the fusion also removes the fp16 resized intermediate's allocation and its L2
footprint. The reduction launch costs 0.93 ms of the 4.21, which is why the win is a fraction of the
0.9–1.4 ms the fused-static column suggests — at K=1 the range must be measured over the
*post-resize* tensor, 4× the input for the UP layers. The largest UP shape (384×16×16) is a slight
regression standalone, 2.06 → 2.24 ms.

`MODIFF_DELTA_REPORT=1` now also fuses at K=1 and pays the fused-static cost rather than the dynamic
one — roughly 3× the win by that table. It is left off: `OptimizedInt8Conv2d.delta_report` already
records it as correct at K=1 and a quality regression at K=4, and that is a measured trade, not one
to flip here.

## Part 2: retiring the code ceiling, and the surface collapse

### W8A4 *was* the code ceiling

Asked to keep only W8A8 / W8A4 / W4A4 and delete `code_ceiling`, which turned out to be
contradictory. `docs/act_bits_2026-08-05/FINDINGS.md` states the reason:

> activations keep their int8 container and the GEMM stays W8A8 … no mainstream ISA has a mixed
> s8×s4 MMA, so W8A4 is not a speed configuration on any hardware, only a quality one.

| config | datapath | Q_b | native clamp | ceiling needed? |
|---|---|---:|---:|---|
| W8A8 | int8 | 127 | 127 | no — no-op |
| W4A4 | packed int4 | 7 | 7 | no — no-op |
| **W8A4** | **int8 container** | **7** | **127** | **yes** |

And not only at `clip < 1`: at the default K=4 a reused stale scale lets the delta outgrow it, so
codes exceed 7 there too. Deleting the ceiling outright would have re-introduced exactly the defect
commit `3be1986` fixed — *"A4 halves, A3 drops 2.6x, and the old A4/A3 rows were flattered"* — and
done it silently.

### The int4-storage route: measured, bit-identical, and refused

The alternative was to give W8A4 genuine int4 activation *storage*, where 4 bits is a property of
the format and no ceiling can be forgotten: quantize to packed int4, widen back to int8, feed the
existing int8-weight conv. `unpack_int4_to_int8_cl` was built and both halves measured
(`data/raw_measurements.md`, harness `integration/tests/bench_unpack_int4_widen.py`):

* **Equivalence: bit-identical on all 10 cases**, including the scale-4×-too-fine regime that is the
  only place the two routes could differ. `max|code|` is 7 either way, 0 differing elements. So it
  changes no number — it is a pure re-encoding of the same rule.
* **Cost: +4.74 ms/step** at batch 128 across the 70 conv layers. 4.5% of the ~105 ms/step the
  configuration runs at, and 3.8× the 1.24 ms Part 1 had just recovered.

Refused on those grounds. The kernel is kept unreferenced, which `csrc/modiff_kernels_api.h`'s
dead-code policy allows only when the reason it is unused is itself a finding — the equivalence
result is what makes the refusal defensible, and re-deriving it costs a rebuild plus a GPU hour. Its
header comment carries the numbers.

### What shipped instead

`float code_ceiling` → `bool a4`, in 8 kernels (5 in `modiff_delta_quantize.cu`, 3 in
`group_norm_silu.cu`). The saturation limit is derived inside each kernel from the datapath it was
told it is on; `clamp_code` is deleted and `csrc/common.cuh` drops from 32 lines to 13.

The point is not that a flag is fewer characters than a float. The ceiling was a *magnitude*, so its
failure mode was a plausible-but-wrong number: pass 127, or forget the argument, and a 4-bit layer
silently stayed 8-bit — which is precisely what the resize kernel did until this session. A bool has
no such value to get wrong, and on the Python side there is now exactly one place that answers "how
many bits is this activation" (`_delta_a4`), threaded from there.

Note the resize kernel had *both* defects: it never fired at K=1, and where it did fire it ignored
the ceiling. The second was invisible because of the first.

Verified in one table (`data/raw_measurements.md`, section E) — this *is* the W8A4 test:

| store | a4 | max\|code\| | want | |
|---|---|---:|---:|---|
| int8 | False | 127 | 127 | W8A8 ✓ |
| int8 | **True** | **7** | 7 | **W8A4 ✓** |
| int4 | False | 7 | 7 | W4A4, format decides ✓ |
| int4 | True | 7 | 7 | format wins over the flag ✓ |

The test failed once first, correctly: it passed `-1.0` as the old ceiling, which converts to
`a4=True`, so int8 layers saturated at 7 instead of 127 — codes off by 120. That is the refactor
surfacing its own API change.

### Retired knobs refuse rather than being ignored

* `MODIFF_ACT_Q` → `MODIFF_ACT_BITS ∈ {8, 4}`. The intermediate widths were a research instrument
  for one report, are not configurations anything ships, and each was a distinct value a call site
  could pass wrongly.
* `MODIFF_DELTA_CLIP` deleted. `Q_level` is `Q_b`, full stop.

Both raise `ValueError` on a retired value. Silently returning absmax numbers under a clip label is
the defect class this tree has paid for twice, and `docs/` still contains archived scripts that set
these.

```
 default              -> accepted        ACT_BITS=15 (old A5) -> REFUSED
 ACT_BITS=4 (W8A4)    -> accepted        DELTA_CLIP=0.40      -> REFUSED
 DELTA_CLIP=1.0       -> accepted
```

### What the clip's retirement costs

Not nothing, and it is worth stating plainly. On W8A8 the curve is monotone — any deliberate
clipping hurts, so the 1.0 default was already optimal and nothing is lost. W4A4 is flat within
noise from 0.35 to 1.0. But **on W8A4 it bought a real improvement**:
`docs/act_bits_2026-08-05` measured r=0.40 at 0.086 relL2 against r=1.0's 0.183. That option is
gone. The sweep tables are preserved in `OptimizedInt8Conv2d`'s comment as well as in
`docs/delta_clip_2026-08-06/`, so if W8A4 quality becomes the bottleneck this is the first thing to
reach back for.

## Quality: what the `a4` correction is worth, and why the answer is "nothing measurable"

The `a4` change alters numerics on the K>1 reuse path for those eight layers, which no timing run can
speak to. Measured with the paired-seed relL2 protocol
(`integration/tests/quality_updown_a4_paired.py`, batch 8, DDIM 50, real checkpoint, relL2 against
the same-seed fp16 latent, run 1 discarded per arm because the attention blocks self-calibrate).

**The two arms had to be isolated with a flag.** At K>1 the pre-commit code differed in TWO ways at
once — the fusion declined on refresh steps (6/8 fused at K=4) *and* the fused ones ignored the
ceiling. Comparing against the old code would measure both. So the fusion is held ON in both arms and
only `MODIFF_UPDOWN_A4` moves; `=0` clamps those eight layers at 127, which is what they did before.
The before-arm cannot be reproduced otherwise, since that code is gone.

**Direction was predicted before running**, and is in the script's docstring: clamping at 127 lets a
4-bit layer keep resolution a true 4-bit quantizer would discard, so the *defective* arm should look
better — the same flattering commit `3be1986` found on the other 62 layers. A correction that makes
the number worse is the correct outcome.

### Controls: both bit-identical

| control | result | why it must be |
|---|---|---|
| W8A8, K=4 | **exactly identical**, 3/3 seeds | `act_q` is 127, so `a4` is False in both arms |
| W8A4, K=1 | **exactly identical**, 3/3 seeds | scale is `Q_b/absmax`, so no code can exceed `Q_b` |

The K=1 row being *exactly* identical rather than approximately is the strongest confirmation in this
section: the ceiling only has something to clamp once a scale has gone stale.

### The effect: the 3-seed answer was noise, and the 8-seed run flipped its sign

| seeds | corrected/defective mean | corrected wins |
|---|---:|---:|
| 3 (1234, 5678, 9012) | 1.061× (worse) | 1/3 |
| **8** | **0.967× (better)** | **5/8** |

Per seed at 8 seeds, W8A4 / K=4:

| | 1234 | 5678 | 9012 | 3141 | 2718 | 1618 | 4669 | 8080 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clamp 127 | 0.1439 | 0.1309 | 0.1747 | 0.1403 | 0.1045 | 0.1976 | 0.1485 | 0.1484 | 0.1486 |
| clamp Q_b | 0.1487 | 0.1398 | 0.1695 | 0.1135 | 0.0983 | 0.1641 | 0.1778 | 0.1381 | 0.1437 |
| diff | +3.3% | +6.8% | −3.0% | −19.1% | −5.9% | −17.0% | +19.7% | −6.9% | **−2.8%** |

Paired per-seed difference: **−2.75% ± 4.49% (SEM)**, stdev 12.70%. The interval straddles zero, so
**the effect is not resolved** — this protocol cannot see it at 8 seeds, in either direction.

### The noise floor, measured rather than assumed

The two runs share three seeds and the identical configuration, so their disagreement *is* the
instrument's reproducibility:

| arm | 1234 | 5678 | 9012 |
|---|---|---|---|
| clamp 127 | 0.1581 → 0.1439 (−8.9%) | 0.1294 → 0.1309 (+1.2%) | 0.1496 → **0.1747 (+16.7%)** |
| clamp Q_b | 0.1429 → 0.1487 (+4.0%) | 0.1399 → 0.1398 (−0.0%) | 0.1810 → 0.1695 (−6.4%) |

Same config, same seed, up to **±17%** apart. The effect under test is 2.8% on the mean. That is the
whole result: the measurement is not inconclusive because too few seeds were run, it is inconclusive
because the effect is an order of magnitude below what batch 8 / DDIM 50 can resolve. `docs/act_bits_2026-08-05`
already warned of 10–30% run-to-run variation here; this quantifies it for this configuration. Eight
seeds gives SEM 4.5%, so resolving 3% would need roughly 50+ seeds, or a larger batch and step count.

### Conclusion

**The `a4` correction has no quality cost this protocol can detect**, and the change stands on
correctness grounds: those eight layers now honour the same bit-width as the other 62, which is what
`MODIFF_ACT_BITS=4` claims. The 3-seed table is left in `data/quality_a4_paired.json` deliberately —
it read 1.061× / 1-of-3 and would have supported "the correction costs 6%", which the 8-seed run
refutes. Three seeds is not enough for an effect this size, and that is worth not rediscovering.

## Regression checks

* `integration/tests/test_gn_resize_fusion.py` — the *baseline* resize kernel is untouched:
  1.57–5.71×, median 2.99×, against its docstring's 1.45–5.6×, median ~2.9×.
* Both `csrc` rebuilds clean. `ninja` had to be installed; without it torch's `BuildExtension`
  falls back to the serial backend and a full rebuild takes >20 min instead of ~1.

## Reproducing

```bash
python integration/tests/test_gn_resize_delta_dynamic.py          # kernel, ~1 min
python integration/tests/test_updown_fusion_modiff_k1.py          # wiring, ~4 min
python integration/tests/ab_updown_fusion_refresh.py --k 1        # paired A/B, ~25 min
python integration/tests/ab_updown_fusion_refresh.py --k 4        # paired A/B, ~25 min
python integration/tests/bench_gn_resize_delta_dynamic.py 128     # kernel shapes, ~1 min
python integration/tests/bench_unpack_int4_widen.py 128           # the refused route, ~1 min
```

The A/B is the one to trust for speed. Prefer it over a cross-session comparison at this effect
size — the `int8_ptq` control above is the reason.

## Open

1. **`MODIFF_DELTA_REPORT=1` at K=1** now fuses too and pays the cheaper static cost. Its quality
   cost is already measured at K=4 but not at K=1 with this fusion live -- and note from the section
   above that a batch-8 / DDIM-50 paired sweep cannot resolve anything under ~10%, so that
   measurement needs a bigger budget than the one used here.
2. **W8A4 quality without the clip.** The r=0.40 option is gone; whether anything cheaper recovers
   that 0.086 is unmeasured.
3. **The environment is only partly provisioned.** `omegaconf`, `einops`,
   `pytorch-lightning==1.4.2`, `torchmetrics==0.6.0`, `tqdm`, `ninja` were installed under a
   constraints file pinning `torch==2.4.1+cu124` so nothing could swap torch out from under the
   built extension. The rest of `requirements.txt` (`albumentations==0.4.3`, `diffusers==0.3.0`,
   CLIP, taming-transformers) is absent and was not needed for the LSUN-churches path;
   `src/taming-transformers` is still an uninitialized submodule.
