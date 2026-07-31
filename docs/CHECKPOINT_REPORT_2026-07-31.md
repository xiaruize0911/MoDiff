# Checkpoint report — FP16 / INT8 / INT4, end-to-end and per layer

**Hardware:** NVIDIA A40 (SM86) · **Model:** LSUN-churches LDM, 21 attention blocks + 21 ResBlocks
**Measured:** 2026-07-31, at commit `e0bded2` plus the INT4 layout-epilogue work

All three modes were measured **in one process per experiment**, so every column in every table
below is directly comparable. Earlier reports in this directory mix measurement sessions; those
comparisons are not reliable and are superseded here.

---

## 1. End to end

200-step DDIM, **batch 128**, median of 5 repeats after 3 warmup samples.

| mode | ms / batch | ms / sample | ms / step | vs FP16 | CV | spread |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 20487.7 | 160.060 | 102.44 | 1.000× | 0.50% | 1.21% |
| INT8 | 14679.9 | 114.686 | 73.40 | **1.396×** | 0.29% | 0.81% |
| **INT4** | **12483.7** | **97.529** | **62.42** | **1.641×** | 0.38% | 1.04% |

INT4 is **17.6% faster than INT8** end to end.

**Measurement configuration is not a detail here — it changed both the numbers and a
conclusion.** An earlier pass at batch 32 / 50 steps gave 1.300× and 1.439×, and inverted the
INT4-vs-INT8 attention comparison (see §1.3). Two separate problems:

- *Sample length.* A 50-step batch-32 sample is short enough for one scheduler hiccup to move the
  median: a 3-repeat run put INT4 at 1064 ms with 9.77% spread, 6.7% off the 997 ms that 9 repeats
  converged on. At 200 steps / batch 128 each measurement averages ~16× more work, and all three
  modes now sit at **CV < 0.5%** — noise suppressed inside the sample rather than rejected after.
- *Operating point.* Batch 32 understates quantization. At batch 128 the speedups rise from
  1.300×/1.439× to 1.396×/1.641×, because the quantized kernels' arithmetic advantage is realised
  once the model is compute-bound rather than launch-overhead-bound.

Batch 128 also matches the layer benchmark, so the two halves of this report are now measured at
the same operating point.

![e2e](final_report_2026-07-28/plots/fig_ck_e2e.png)

### One environment variable decides which code path runs — the harness verifies it

`MODIFF_QUANT_LINEAR` is not a tuning knob. It selects which attention implementation executes,
so getting it wrong does not perturb a measurement, it measures something else entirely — and it
does so without any error, warning, or implausible number.

An earlier attempt at this same benchmark produced clean-looking numbers (INT8 1.162×,
INT4 1.239×) for a configuration in which **no fused attention epilogue was active at all**.
`MODIFF_QUANT_LINEAR=1` was unset, so the attention `qkv`/`proj` stayed plain `nn.Linear`,
`_qout_eligible()` returned False, and every fused route — the INT8 layout epilogue, the INT4
layout epilogue, and the older i4values short-circuit — silently fell back to the generic score
path. Nothing errored or warned. The mistake was only visible in the kernel trace, by the absence
of `gemm_w4a4_kernel_awq_out_i8`.

`e2e_three_mode_bench.py` now asserts the route before timing and records the result in the JSON:

| mode | attention blocks | qout-eligible | expected | qkv / proj type |
|---|---:|---:|---:|---|
| INT8 | 21 | 21 | 21 | `QuantLinearWxAx` |
| INT4 | 21 | **15** | 15 | `QuantLinearWxAx` |

INT4's 15/21 is **correct, not a defect**: the six hd=96 blocks have no INT4 route (the int4
kernel caps at hd≤64, and the small-shape kernel and its scale observer are INT8-only, so
`_fq4_frozen` is never reached there). Those six run FP16 SDPA.

### Where the whole model's time goes

Panel C of the figure, profiler self-time scaled to the measured wall time (ms per batch of 128):

| stage | FP16 | INT8 | INT4 | INT4 − INT8 |
|---|---:|---:|---:|---:|
| convolution | 8662.7 | 5447.3 | 2852.8 | **−2594.5** |
| GroupNorm + quantize | 4021.6 | 3660.6 | 3798.2 | +137.6 |
| attention core | 2276.8 | 1797.6 | **1731.0** | **−66.6** |
| GEMM / projection | 703.0 | 1849.1 | 1666.0 | −183.1 |
| K/V prep + out quantize | 0.0 | 32.8 | 220.7 | +187.9 |
| elementwise / upsample / concat / pool | 4823.7 | 1892.5 | 2214.9 | +322.4 |
| **total** | **20487.7** | **14679.9** | **12483.7** | **−2196.2** |

**Convolution dominates, not attention.** That is the single most important framing in this
report: the attention core is **11.1%** of FP16's whole-model time, and INT4's 2196 ms e2e win
over INT8 is overwhelmingly convolution (−2594 ms), partly given back in GroupNorm, K/V prep and
elementwise.

Reading the rows that are easy to misread:

- **"elementwise / upsample / concat / pool"** is not one thing. FP16's 4823.7 splits into 3848.6 ms
  of elementwise and 975.1 ms of upsample/concat/avg-pool. The 4.2x elementwise drop in INT8 (3848.6 -> 926.1) is fusion: residual+bias fold into the GEMM epilogue and SiLU folds into
  GroupNorm, taking launch counts from ~8400 to ~5200. The upsample/concat/pool part is comparable in
  ALL THREE modes -- none of it is quantized, so it is a fixed floor no quantization work touches.
- **"K/V prep + out quantize"** is INT4 paying for the two shapes that were deliberately not
  ported: `aq_kv_packed_static_tiled` runs 2000x per sample (200 steps x the 10 T256/T64 blocks,
  which use nibble-packed Q/K and still need the producer) and `quant_attn_out_int4_pack` runs
  1200x (200 steps x the 6 hd=96 blocks packing their FP16-SDPA output). INT8 needs only `from_i8_kv_tiled` at
  T64 because its T1024 and T256 both emit Flash-ready layouts from the GEMM. 187.9 ms, ~1.5% of
  INT4's total, and entirely attributable to known open items 4 and 5.

Two things worth noting against the headline:

- **INT4's attention core is 66.6 ms FASTER than INT8's** (1731.0 vs 1797.6), consistent with the
  layer-level result. Both modes run the same specialized kernel
  (`flash_attn_int8_mma_kernel_t<32,8,32,true,*,false,false,24,1024>`, differing only in
  `PACK_OUT4`), and the new INT4 layout epilogue `gemm_w4a4_kernel_awq_out_i8<1>` is confirmed
  present in the trace.

  *A batch-32 measurement of this same quantity said the opposite* — INT4 8.8 ms slower, with the
  T1024 flash at 403.3 vs 363.0 us/call. At batch 128 the two kernels are within 0.3%
  (1424.6 vs 1419.9 us/call). The packed-int4 output store carries a fixed cost that is material
  at small batch and amortises away at large batch. **Conclusion: report attention-core
  comparisons at the production batch size; the batch-32 inversion was an artefact of the
  operating point, not a property of the kernels.**
- **GroupNorm+quantize is INT4's largest non-conv stage and exceeds FP16's** (279.2 vs 254.4). The
  packed-int4 GN costs more than the fp16 one it replaces.

---

## 2. Layer level

Every layer instance in the UNet, batch 128, 20 warmups, median of 5 rounds × 60 iterations.
26 distinct (kind, shape) entries covering all 42 layer instances.

### By layer kind, summed over all instances

| mode | attention | resblock_plain | resblock_updown | total |
|---|---:|---:|---:|---:|
| FP16 | 24.17 ms | 48.05 ms | 6.41 ms | 78.63 ms |
| INT8 | 20.23 ms | 34.47 ms | 5.03 ms | 59.73 ms (1.32×) |
| INT4 | **19.68 ms** | **29.47 ms** | 5.89 ms | **55.04 ms (1.43×)** |

Attention is **31% of FP16 layer time**. INT4's 4.7 ms lead over INT8 is 5.0 ms of ResBlock win
minus a 0.86 ms ResBlock-updown loss, with attention contributing only 0.55 ms.

![layers](final_report_2026-07-28/plots/fig_ck_layers.png)

### ⚠ INT4 is SLOWER THAN FP16 on five layers

![speedup matrix](final_report_2026-07-28/plots/fig_ck_speedup_matrix.png)

The heatmap is the most actionable figure here. **INT8 is never below 1.0× on any layer**
(range 1.13–1.86×). INT4 reaches 2.12× at its best but drops below FP16 on five:

| layer | INT4 vs FP16 | INT8 vs FP16 |
|---|---:|---:|
| resblk C768/2² | **0.63×** | 1.29× |
| resblk C768/2² (updown) | **0.66×** | 1.39× |
| resblk C1536/2² | **0.67×** | 1.22× |
| resblk↕ C384/8² | **0.77×** | 1.27× |
| resblk C384/4² | **0.80×** | 1.31× |

Every one is a **small-spatial** block (2², 4², 8²). The pattern is consistent: INT4's advantage
scales with problem size and inverts once the layer is small enough that fixed overhead dominates.
These five are the clearest remaining INT4 work item, and they are entirely outside attention.

### Attention layers by stage

![attention stages](final_report_2026-07-28/plots/fig_ck_attn_stages.png)

| shape | FP16 | INT8 | INT4 | best |
|---|---:|---:|---:|---|
| C192/T1024 ×5 | 3099 | 2749 | **2696** | INT4 1.15× |
| C384/T256 ×5 | 1069 | 866 | **815** | INT4 1.31× |
| C384/T64 ×5 | 411 | 230 | **225** | INT4 1.83× |
| C768/T16 ×5 | 217 | 180 | **167** | INT4 1.30× |
| C768/T4 ×1 | 191 | **103** | 161 | INT8 1.86× |

Reading the stages:

- **T1024 is attention-core-bound in every mode** (1870 / 1460 / 1450 µs). Quantizing the score
  path barely moves it; the quantized wins there come from removing FP16's residual/copy traffic
  and shrinking the projections.
- **T64's 1.8× is a GroupNorm story.** FP16 spends more time in GN than in the score path at that
  shape — the largest relative win in the table has nothing to do with the attention kernel.
- **T256 still shows INT4 carrying a K/V-prep block** (the purple segment). That shape kept the
  nibble-packed producer route deliberately; it is remaining headroom, not a regression.
- **T4 is the one attention shape INT8 clearly wins**, because INT4 falls back to FP16 SDPA there
  (hd=96 > the int4 kernel's hd≤64) and additionally pays a separate output quantize.

---

## 3. Data and reproduction

| file | contents |
|---|---|
| `final_report_2026-07-28/data/e2e_three_mode.json` | e2e latency, spreads, `route_check`, full per-kernel profile per mode |
| `final_report_2026-07-28/data/attn_three_mode_final.json` | all 26 layer entries × 3 modes, with per-kernel profiles |
| `final_report_2026-07-28/data/int4_layout_epilogue.json` | INT4 layout-epilogue bit-exactness + SASS census + A/B |
| `final_report_2026-07-28/data/int4_step0_same_session.json` | the same-session baseline that scoped the INT4 work |
| `final_report_2026-07-28/scripts/e2e_three_mode_bench.py` | e2e driver (asserts the route before timing) |
| `final_report_2026-07-28/scripts/layer_pipeline_bench.py` | layer driver |
| `final_report_2026-07-28/scripts/make_checkpoint_report_plots.py` | all four figures |

```bash
python3 docs/final_report_2026-07-28/scripts/e2e_three_mode_bench.py --batch 128 --steps 200 --repeats 5 --warmups 3
LBENCH_BATCH=128 LBENCH_MODES=fp16,int8_baseline,int4_baseline \
  LBENCH_OUT=docs/final_report_2026-07-28/data/attn_three_mode_final.json \
  python3 docs/final_report_2026-07-28/scripts/layer_pipeline_bench.py
python3 docs/final_report_2026-07-28/scripts/make_checkpoint_report_plots.py
```

Stage attribution is by kernel name; anything unmatched is routed to an explicit "other" bucket
**and printed**, so it cannot be silently folded into a neighbouring stage. That check caught five
misclassified cuBLAS kernels during development. Stacked segments are scaled to the independently
measured wall/pipeline time so they sum to the latency actually reported rather than to the
profiler's inflated total.

---

## 4. Caveats

**No result here is validated against trained weights.** `models/ldm/lsun_churches256/model.ckpt`
is an 856-byte stub whose `state_dict` is empty, loaded with `strict=False`, so every weight is
randomly initialised. Timing is unaffected — these kernels have no data-dependent control flow,
and the repo already relies on this — but **nothing in this report supports an image-quality
claim**, and the static quantization scales were calibrated against random-weight activation
statistics. Restoring real weights is the highest-value next step for confidence, independent of
any further performance work.

Nsight Compute counters are unavailable in this container (`ERR_NVGPUCTRPERM`), so all attribution
is CUDA-event timing plus profiler self-time, never hardware counters.

---

## 5. Open items, in priority order

1. **INT4's five sub-FP16 layers** (0.63–0.80×), all small-spatial ResBlocks. Largest correctness-
   of-story problem: INT4 is presented as the fastest mode while being slower than FP16 on part of
   the model.
2. **Real weights + recalibration**, to convert every timing claim here into a supportable
   end-to-end claim.
3. **INT4 GroupNorm** is now its largest non-conv stage and exceeds FP16's.
4. **T4/hd96 INT4** — the only attention shape where INT8 wins, worth ~0.06 ms weighted.
5. **T256 INT4 K/V producer** — the last fused-epilogue gap in attention.

The INT8 attention target of 1.5× weighted remains unmet at 1.20×; the analysis in
`final_report_2026-07-28/ATTENTION_SLOWER_THAN_FP16.md` and the occupancy rejection pinned in `flash_attn_int8.cu`
indicate the remaining gap is stall-bound rather than throughput-bound, which needs Nsight
counters to localise.
