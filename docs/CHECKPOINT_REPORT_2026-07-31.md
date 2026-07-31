# Checkpoint report — FP16 / INT8 / INT4, end-to-end and per layer

**Hardware:** NVIDIA A40 (SM86) · **Model:** LSUN-churches LDM, 21 attention blocks + 21 ResBlocks
**Measured:** 2026-07-31, at commit `e0bded2` plus the INT4 layout-epilogue work

All three modes were measured **in one process per experiment**, so every column in every table
below is directly comparable. Earlier reports in this directory mix measurement sessions; those
comparisons are not reliable and are superseded here.

---

## 1. End to end

50-step DDIM, batch 32, median of 9 repeats after 2 warmup samples.

| mode | ms / batch | ms / sample | ms / step | vs FP16 | run spread |
|---|---:|---:|---:|---:|---:|
| FP16 | 1435.0 | 44.843 | 28.70 | 1.000× | 2.26% |
| INT8 | 1104.2 | 34.505 | 22.08 | **1.300×** | 0.20% |
| **INT4** | **997.5** | **31.173** | **19.95** | **1.439×** | 0.79% |

INT4 is **9.7% faster than INT8** end to end.

**Repeat count mattered here.** A 3-repeat run of the same configuration put INT4 at 1064.2 ms
(1.348×) with a 9.77% spread — the median was unstable and understated it by 6.7%. Nine repeats
bring the spread to 0.79%. Do not quote e2e numbers from fewer than ~9 repeats.

![e2e](final_report_2026-07-28/plots/fig_ck_e2e.png)

### Configuration is load-bearing — check it, do not assume it

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

Panel C of the figure, profiler self-time scaled to the measured wall time (ms per batch of 32):

| stage | FP16 | INT8 | INT4 | INT4 − INT8 |
|---|---:|---:|---:|---:|
| convolution | 617.8 | 465.4 | 284.1 | **−181.3** |
| GroupNorm + quantize | 254.4 | 239.0 | 279.2 | +40.2 |
| attention core | 160.2 | 117.6 | 126.4 | +8.8 |
| GEMM / projection | 71.7 | 146.2 | 144.7 | −1.5 |
| K/V prep + out quantize | 0.0 | 1.5 | 15.1 | +13.6 |
| elementwise / copies | 330.9 | 134.6 | 148.1 | +13.5 |
| **total** | **1435.0** | **1104.2** | **997.5** | **−106.7** |

**Convolution dominates, not attention.** That is the single most important framing in this
report: the attention core is **11.2%** of FP16's whole-model time, and INT4's 106.7 ms e2e win
over INT8 is entirely convolution (−181.3 ms) partly given back everywhere else.

Two things worth noting against the headline:

- **INT4's attention core is 8.8 ms SLOWER than INT8's** end to end (126.4 vs 117.6), even though
  INT4 wins the attention *layer* comparison. The six hd=96 blocks running FP16 SDPA account for
  this — they are attention-core time that INT8 serves with a quantized kernel and INT4 does not.
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
python3 docs/final_report_2026-07-28/scripts/e2e_three_mode_bench.py --batch 32 --steps 50 --repeats 9
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
