# Extending MoDiff to the attention projections: it helps, and it still lands off the frontier

`MODIFF_LINEAR=1` modulates the 42 attention qkv/proj Linears in addition to the convs. Its default
has been 0 since 2026-08-12, justified by a comment whose W4A4 half had expired: *"the 'recognisable
churches vs fog' concern was always about W4A4, where BOTH arms (0.36 / 0.42) are bad and which is not
a recommended configuration."* W4A4 MoDiff is **0.3095** now. So the flag was measured in the regime
the tree actually ships, as its own labelled arm.

| arm | relL2 | ms/step | vs fp16 |
|---|---:|---:|---:|
| `int4` shipped (conv + emb modulated) | 0.3095 | 60.53 | 1.749× |
| **`int4_linmodiff`** (+ 42 projections) | **0.2835** — 8.4% better | **90.34** — +49% | 1.172× |
| `int8` W8A8 MoDiff | 0.0605 | 74.70 | 1.417× |
| `int8_baseline` W8A8 PTQ | 0.1138 | 73.05 | 1.449× |

**The default stays 0, but for a sound reason instead of an expired one.** `int4_linmodiff` is
strictly dominated — **both** W8A8 arms are faster *and* more accurate. No objective selects it.
The frontier in [`plots/01_frontier.png`](plots/01_frontier.png) is computed by the plotting script,
which refuses to draw the annotation if exactly one arm is not dominated.

## The cost is un-fusing, not a missing epilogue

The standing explanation was that linear MoDiff would get cheap once the GEMM gained an
o_hat-accumulate epilogue. It has one, and that is not where the time goes. Profiled buckets, ms of
the window:

| bucket | int4 | linmodiff | Δ |
|---|---:|---:|---:|
| quantize (standalone) | **0** | **2890** | **+2890** |
| elementwise / copy | 1232 | 2657 | +1426 |
| GEMM / conv | 5116 | 6153 | +1037 |
| GroupNorm+SiLU | 3604 | 4137 | +532 |
| attention | 1750 | 1839 | +89 |

In the shipped path that first bucket is **exactly zero** — every quantize is folded into a GroupNorm
kernel. Modulating the projections materialises 2.89 s of standalone quantize from nothing
(`static_quantize_pack_and_update_ahat` +1131, `delta_absmax_fp16` +876). The route check sees the
other half directly: fused int8-output epilogues **21/21 → 0/21**.

## Coverage: three families, not two

The plan's fix #5 framed coverage as "the paper quantizes 168 modules, we calibrate 70", which
conflated calibration with modulation. Counted rather than inferred
([`data/module_census.json`](data/module_census.json)):

| family | class | n | modulated by default |
|---|---|---:|---|
| ResBlock convs | `OptimizedInt4Conv2d` | 70 | yes |
| emb Linears (35 `emb_layers` + 2 `time_embed`) | `OptimizedInt4Linear` | 37 | yes — `benchmark_ldm.py:668`, unconditional for the mode |
| attention qkv/proj | `QuantLinearWxAx` | 42 | **no** — the only family the flag gates |

So **107 of 149** quantized modules are modulated by default.

**And a bug the census exposed.** The census first reported **140** convs. They are 140 distinct
objects, not duplicate names: `fused_resblock.py:756` aliases one `nn.Conv2d` as both
`fused.in_conv` and `fused.original.in_layers[-1]`, so the int4 conversion wraps it twice. `forward`
uses `self.in_conv`, which is why all 70 calibrated convs are the non-nested set and none of the
nested ones are. The result is **70 orphaned int4 conv wrappers** — never called, never calibrated,
and carrying `modiff_enabled = True`. Inert today; the memory cost is unmeasured.

## The measurement defect that cost the most, and the guard that now catches it

The first run of the A/B reported the shipped arm at **0.3303** against a committed **0.3090** — 6.9%,
against a 0.6% floor. Five hypotheses were eliminated by measurement, in this order:

| hypothesis | test | result |
|---|---|---|
| the tree drifted | re-ran `static_vs_dynamic_ab.py` unmodified | **no** — all 6 arms reproduced per-seed |
| arm position in the process | same arm at positions 1 and 4 | **no** — bit-identical |
| delta table not loaded | grep both logs | **no** — loaded in both |
| references | control within the floor | **wrongly cleared, see below** |
| the `measure()` code path | 3 variants, one process, one reference set | **no** — all bit-identical |

The cause was the **fp16 references**: `H.AUTO_DELTA_TABLE` was assigned inside `measure()` instead of
before the references, so they were built with the module default `False` while every other harness
builds them `True`. That offset *every* arm in the run by a shared amount — invisible in a within-run
comparison, wrong in any cross-run table. Moving it cut the error 6.9% → 1.1%, still above the floor;
a second, unidentified source remains, so the references are now **pinned to disk**
([`scripts/fp16_refs.py`](scripts/fp16_refs.py)) and shared, which makes cross-harness comparison exact
by construction. `--rebuild` measures the residual nondeterminism instead of leaving it to surface as a
1% wobble in an unrelated A/B.

**Two process errors worth keeping, because both were mine:**

* **A control that cannot fail loudly is not a control.** The `int4_baseline` control *did* register
  the problem — 0.45% off — but 0.45% is inside the noise floor, so it was read as noise and the
  references were cleared on that basis. The fix is the **reference self-check**: an arm whose
  committed value is known, where a shared offset appears at full size. It fired on the very next run
  (+1.1%) and refused to report.
* **A plausible argument was used to eliminate a hypothesis.** The references were ruled out by a
  quadrature calculation — which models the perturbation as *independent noise* when a shared
  trajectory offset is systematic. That argument was wrong and it cost more time than any measurement.

The inflated baseline also nearly doubled the apparent benefit: the first run read the gain as 14.5%,
the corrected number is **8.4%**.

## Harness provenance fixed in passing

`e2e_three_mode_bench.py` hardcoded `int*_calibration.pt` — the stub-checkpoint file its own preference
comment grades at relL2 0.882 / 3.023 and demoted to last resort — while `run_all.sh` advertised
*"NOTHING HERE PASSES A CALIBRATION PATH."* It now resolves through `CALIBRATION_PREFERENCE`.

The stub file carries 107 entries (70 convs + 37 emb Linears) against the qdiff file's 70, so those 37
Linears ran on **static** scales there and a per-call **dynamic** absmax in the shipped path — a
different kernel route, not a different number. **Predicted the shipped path would be slower; it is
not.** Every arm moved ≤0.33%, inside the noise floor, so the emb Linears' dynamic pass does not
register at batch 128. The fix corrects provenance, not any published number.

## Reproducing

```bash
python docs/attn_modiff_2026-08-13/scripts/linear_modiff_w4a4_ab.py      # fidelity, pinned refs
E2EBENCH_MODES=fp16,int8_baseline,int8,int4_baseline,int4,int4_linmodiff \
  python integration/benchmarks/report/e2e_three_mode_bench.py \
  --batch 128 --steps 200 --repeats 3 --warmups 2 \
  --output docs/attn_modiff_2026-08-13/data/e2e_linmodiff.json            # latency
python docs/attn_modiff_2026-08-13/scripts/plot_frontier.py               # the chart
python docs/attn_modiff_2026-08-13/scripts/count_modulated_modules.py     # the census
python docs/attn_modiff_2026-08-13/scripts/fp16_refs.py --rebuild         # fp16 nondeterminism
```

Investigation scripts kept because their negative results are the evidence:
`arm_position_effect.py` (position is irrelevant), `measure_path_aa.py` (the code path is not the
cause).
