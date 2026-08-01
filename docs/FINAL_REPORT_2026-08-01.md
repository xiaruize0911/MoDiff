# Final report — FP16 / INT8 / INT4 across five measurement levels

**Hardware:** NVIDIA A40 (SM86) · **Model:** LSUN-churches LDM, 21 AttentionBlocks + 35 ResBlocks
**Measured:** 2026-08-01 at commit `fa57f9f` · **Batch 128** everywhere · **200 DDIM steps** end to end

Five suites, from single kernels up to whole-model latency:

| # | suite | what one measurement is | measurements | data |
|---|---|---|---:|---|
| 1 | attention kernels | one call of the attention-core entry point | 31 | `kernel_suites_b128.json` |
| 2 | conv kernels | one call of the conv entry point | 117 | `kernel_suites_b128.json` |
| 3 | linear kernels | one call of the GEMM entry point | 50 | `kernel_suites_b128.json` |
| 4 | per layer | one forward of a real UNet layer module | 78 | `layers_final_b128.json` |
| 5 | end to end | one 200-step DDIM batch of 128 | 27 | `e2e_final_b128.json` |

Every suite measures all three modes **in one process**, so columns are comparable. Nothing below
is transcribed by hand — `ck_final_numbers.py` emits these tables from the JSON.

---

## 0. How stability is reported, and why it is reported this way

Each measurement is **warmup 30, then 8 rounds × 60 timed calls** (e2e: 3 warmup samples, then 9
repeats). Within a round the **median** of the 60 calls is taken, which rejects a scheduler hiccup
inside the round; the reported statistics are computed **over the 8 round medians**. So the quoted
CV is *round-to-round reproducibility* — the number that tells you whether re-running would give
the same answer — not within-round jitter, which is tracked separately.

Intervals are **Student-t 95% on the mean of the round medians**, not ±1.96σ: at 8 rounds the
normal approximation understates the interval by ~20%. Speedups carry their own delta-method
interval rather than being a bare ratio of two central values, because a 1.05× whose interval
straddles 1.0 is not a speedup.

**Every central value in this report is the mean of the round medians**, quoted with its t-based
interval, because that is the statistic a CI attaches to. The medians are also retained in the JSON
and differ from the means by under 0.4% throughout; the two give the same conclusions but must not
be mixed inside one table, which an earlier draft of §4 and §5 did.

The previous harnesses collapsed each measurement to a single median and discarded the
distribution, so a stable number and one that happened to land there once were indistinguishable.

---

## 1. Attention kernels

Aligned on **T alone, never (T, head dim)** — the quantized kernels pad hd 24 → 32 at T=1024, so a
(T, hd) key would never match FP16 there even though it is the same attention layer.

| T | FP16 µs | INT8 µs | INT4 µs | INT8 × (95% CI) | INT4 × (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 2032.3 | 1597.3 | 1566.6 | 1.272 ± 0.030 | 1.297 ± 0.018 |
| 256 | 348.7 | 256.1 | 194.1 | 1.362 ± 0.018 | **1.796 ± 0.042** |
| 64 | 91.0 | 41.5 | 32.3 | 2.194 ± 0.033 | **2.816 ± 0.042** |
| 16 | 49.0 | 65.8 | 68.8 | **0.744 ± 0.005** | **0.711 ± 0.004** |
| 4 | 48.6 | 19.1 | 26.3 | 2.547 ± 0.014 | 1.852 ± 0.047 |

Range: INT8 0.74–2.55× (median 1.36×), INT4 0.71–2.82× (median 1.80×).

**T=16 is the one place a quantized kernel genuinely loses**, and the interval is tight enough to
be certain (±0.005). The quantized small-shape kernel costs 65.8/68.8 µs against FP16 SDPA's 49.0.
It is not a routing mistake: the layer-level suite (§4) shows the same shape winning **1.40×/1.63×**
overall, because the projections around it get much cheaper. The dp4a small kernel's cost grows
with T², so closing this needs a different kernel, not a re-route.

**T=1024 carries most of the weight and gains least** — 1.27×/1.30×, and INT8 and INT4 are within
2% of each other. Quantizing the score path barely separates them.

### 1.1 The route in the first steps is not the route in the steady state

This is a measurement trap worth recording, because it silently inverts conclusions. At T=1024 the
25 calls in a 5-step window split across **three** entries — `flash_attn_int8_vt` ×10 (1843.5 µs),
`_vt_static` ×5 (1597.8), and the fused `qi8_kv_static_qout_hd24` ×10 (1597.3) — and at T=4 they
split between FP16 SDPA (59.5 µs) and the quantized small kernel (19.1 µs). Over 200 steps the
fused entry is what runs: the e2e profile shows the T=1024 kernel firing exactly **1000 times =
5 blocks × 200 steps**, one entry per block per step.

An earlier version of the table picked the slowest entry at each key and therefore reported the
startup path: 1.10× instead of 1.27× at T=1024, and **0.82× instead of 2.55× at T=4** — a reported
loss where there is a 2.5× win. The tables above select the steady-state fused entry. Per-call
times come from this suite; the call **mix** comes from the 200-step e2e profile.

---

## 2. Conv kernels

All **33 of 33** shape keys align across the three modes. But 13 of them run
`torch_conv2d_fp16` in *all three* modes — every 1×1 skip conv plus the input and output convs —
so their cross-mode ratio measures only the input dtype the surrounding pipeline handed them, not
quantization. The same entry at the same shape reads **1381.8 µs with an fp32 input in FP16 mode
and 579.5 µs with an fp16 input in the quantized modes**; 12 of 33 FP16-mode torch convs take fp32
input against 2 of 13 in the quantized modes. Those rows are flagged and excluded from the range.

Over the **20 genuinely quantized keys**:

| | range | median |
|---|---|---:|
| INT8 | 1.34–2.39× | **1.95×** |
| INT4 | 2.75–4.84× | **3.94×** |

The largest shapes, all genuinely quantized:

| (Cin, H, W, Cout, k) | FP16 µs | INT8 µs | INT4 µs | INT8 × | INT4 × |
|---|---:|---:|---:|---:|---:|
| (384, 32, 32, 384, 3) | 3791.8 | 1661.9 | 788.8 | 2.282 ± 0.010 | **4.807 ± 0.037** |
| (576, 32, 32, 192, 3) | 3273.7 | 1683.0 | 821.1 | 1.945 ± 0.011 | 3.987 ± 0.050 |
| (768, 16, 16, 384, 3) | 1880.3 | 817.6 | 388.6 | 2.300 ± 0.031 | **4.838 ± 0.123** |
| (384, 32, 32, 192, 3) | 1789.0 | 1067.7 | 508.0 | 1.676 ± 0.015 | 3.521 ± 0.076 |
| (768, 8, 8, 768, 3) | 915.0 | 399.7 | 191.1 | 2.289 ± 0.039 | 4.788 ± 0.112 |

**Conv is where INT4 actually wins, by a wide margin** — up to 4.84× on a single kernel. This is
the strongest quantization result at any level in this report, and §6 explains why only part of it
survives to the whole model.

---

## 3. Linear kernels

**Per-shape cross-mode comparison is not possible here, and that is a property of the code rather
than a gap in the measurement.** Only 4 of 29 keys align across all three modes, and all four are
the emb Linears that stayed FP16. The quantized GEMMs fuse bias and residual into the epilogue,
which changes N (e.g. 192 → 640), and the INT4 path pads K (192 → 128 packed), so the shapes
themselves differ from the FP16 Linear they replace.

Per mode, the heaviest GEMMs:

| mode | entry | (M, K, N) | µs/call ± 95% CI | CV |
|---|---|---|---:|---:|
| FP16 | `torch_linear_fp16` | (131072, 192, 192) | 200.1 ± 0.7 | 0.41% |
| INT8 | `gemm_w8a8_awq_qkv_i8_layouts` | (131072, 192, 768) | 581.8 ± 1.9 | 0.38% |
| INT8 | `gemm_w8a8_awq_bias_res` | (131072, 192, 640) | 407.9 ± 7.8 | 2.28% |
| INT4 | `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` | (131072, 128, 768) | 558.4 ± 0.2 | 0.05% |
| INT4 | `gemm_w4a4_awq_bias_res` | (131072, 128, 640) | 352.7 ± 3.3 | 1.11% |

The comparable figure is the whole-model projection stage from §5: **1785.4 / 1883.4 / 1791.5 ms**
for FP16 / INT8 / INT4. **The quantized GEMMs buy nothing end to end** — INT8 is 5% *worse* than
FP16 and INT4 is level with it. The quantized linear path is carrying its fused epilogue, not a
speedup.

---

## 4. Per layer

Every layer instance, batch 128, 8 rounds × 60 iterations.

| mode | attention | resblock_plain | resblock_updown | total | vs FP16 |
|---|---:|---:|---:|---:|---:|
| FP16 | 24.66 ms | 49.17 ms | 6.73 ms | 80.56 ms | 1.000× |
| INT8 | 20.18 ms | 35.31 ms | 5.26 ms | 60.75 ms | 1.326× |
| INT4 | **19.27 ms** | **26.37 ms** | **4.59 ms** | **50.23 ms** | **1.604×** |

(Median-based instead of mean-based, the same table reads 80.47 / 60.61 / 50.09 ms and
1.328× / 1.606× — a 0.2% difference that changes nothing.)

INT4 leads INT8 in every column. Attention is 31% of FP16 layer time.

Attention layers, with the distribution — note these are whole AttentionBlocks, so they include the
projections that rescue T=16:

| shape | n | FP16 µs ± CI | INT8 µs ± CI | INT4 µs ± CI | INT8 × | INT4 × |
|---|---:|---:|---:|---:|---:|---:|
| C192/32² (T=1024) | 5 | 3143.4 ± 24.5 | 2746.1 ± 20.0 | 2689.1 ± 38.1 | 1.14 ± 0.01 | 1.17 ± 0.02 |
| C384/16² (T=256) | 5 | 1075.2 ± 1.2 | 859.6 ± 3.9 | 778.4 ± 2.3 | 1.25 ± 0.01 | 1.38 ± 0.00 |
| C384/8² (T=64) | 5 | 411.4 ± 0.2 | 222.8 ± 0.2 | 210.3 ± 0.2 | 1.85 ± 0.00 | 1.96 ± 0.00 |
| C768/4² (T=16) | 5 | 253.3 ± 1.1 | 180.7 ± 0.1 | 155.7 ± 0.1 | 1.40 ± 0.01 | 1.63 ± 0.01 |
| C768/2² (T=4) | 1 | 246.7 ± 3.6 | 136.5 ± 8.3 | 102.1 ± 0.4 | 1.81 ± 0.11 | 2.42 ± 0.04 |

The heaviest ResBlocks:

| shape | n | FP16 µs ± CI | INT8 µs ± CI | INT4 µs ± CI | INT8 × | INT4 × |
|---|---:|---:|---:|---:|---:|---:|
| C576/32² | 1 | 6028.9 ± 10.5 | 4582.8 ± 42.1 | 3321.3 ± 27.5 | 1.32 ± 0.01 | 1.82 ± 0.02 |
| C384/32² | 2 | 4812.3 ± 9.7 | 3554.7 ± 16.1 | 2626.1 ± 26.8 | 1.35 ± 0.01 | 1.83 ± 0.02 |
| C192/32² | 2 | 3242.0 ± 10.8 | 2425.1 ± 12.2 | 1638.9 ± 7.0 | 1.34 ± 0.01 | 1.98 ± 0.01 |

**No layer is slower than FP16 in either mode** — the property established in
`CHECKPOINT_REPORT_2026-08-01.md` still holds after this round's kernel deletions.

---

## 5. End to end

200-step DDIM, batch 128, **9 repeats** after 3 warmup samples.

| mode | ms/batch (mean ± 95% CI) | ms/sample | ms/step | vs FP16 (95% CI) | CV | spread |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 20605.6 ± 22.1 | 160.981 | 103.03 | 1.000× | 0.14% | 0.43% |
| INT8 | 14693.4 ± 47.5 | 114.792 | 73.47 | 1.402 ± 0.005 | 0.42% | 1.16% |
| **INT4** | **11992.2 ± 8.3** | **93.689** | **59.96** | **1.718 ± 0.002** | 0.09% | 0.26% |

Both the ms/batch and the ms/step column are derived from the mean, and the ratios are ratios of
means. On medians the same run reads 20600.1 / 14722.4 / 11996.5 ms and 1.399× / 1.717×.

INT4 is **22.5% faster than INT8**. Both intervals exclude 1.0 by a wide margin.

Where the time goes (profiler self-time scaled to the measured wall, ms per batch). This table is
scaled to the **median** wall time, which is why its totals are the medians above rather than the
means; the scaled stage sums reproduce that wall time to ±0.00%:

| stage | FP16 | INT8 | INT4 | INT4 − INT8 |
|---|---:|---:|---:|---:|
| convolution | 7678.5 | 5467.5 | 2842.8 | −2624.7 |
| GroupNorm + quantize | 4020.4 | 3667.8 | 3785.7 | +117.9 |
| attention core | 2278.4 | 1804.7 | 1724.6 | −80.1 |
| QKV / output projection | 1785.4 | 1883.4 | 1791.5 | −91.9 |
| K/V gather + transpose | 0.0 | 0.0 | 0.0 | +0.0 |
| attention output quantize | 0.0 | 0.0 | 0.0 | +0.0 |
| elementwise / upsample / concat / pool | 4837.4 | 1899.0 | 1851.9 | −47.0 |
| **total** | **20600.1** | **14722.4** | **11996.5** | **−2725.9** |

The attention core is **11.1%** of FP16's whole-model time. INT4's win over INT8 is almost entirely
convolution (−2625 ms), partly given back in GroupNorm (+118 ms).

---

## 6. What the five levels together say

The five suites do not disagree; read together they explain the whole-model number:

| level | INT4 vs FP16 |
|---|---:|
| conv kernel, quantized shapes only | **3.94× median** |
| whole-model convolution stage | 2.70× (7678.5 → 2842.8) |
| per layer, all layers | 1.604× |
| end to end | **1.718×** |

**Why 3.94× at the kernel becomes 2.70× for the conv stage:** 13 of the 33 conv shapes are never
quantized in any mode — every 1×1 skip conv plus the in/out convs (§2). The stage total includes
them at ~1.0×, so the stage ratio is a weighted average of quantized and untouched convs.

**Why 2.70× on conv becomes 1.718× end to end:** convolution is only 37% of FP16's time.
GroupNorm+quantize is 3785.7 ms in INT4 — its largest non-conv stage, **worse than INT8's** and
barely better than FP16's 4020.4 — and the projection stage (§3) is flat across modes. Two of the
four big stages contribute nothing.

**The single highest-value target is therefore GroupNorm, not attention and not conv.** It is
31.6% of INT4's remaining time, it is the only stage where INT4 loses to INT8, and the quantized
path does strictly less output work than FP16 while being only 6% faster than it.

---

## 7. Stability

| suite | measurements | median CV | p90 CV | max CV | NOISY (CV>3%) |
|---|---:|---:|---:|---:|---:|
| attention kernels | 31 | 1.21% | 5.96% | 13.73% | 7 |
| conv kernels | 117 | 0.47% | 2.01% | 18.55% | 4 |
| linear kernels | 50 | 0.62% | 4.24% | 14.13% | 9 |
| per layer | 78 | 0.36% | 1.69% | 16.12% | 2 |
| end to end | 3 | 0.14% | 0.14% | 0.42% | 0 |

Median reproducibility is under 1.3% at every level and under 0.5% for the two suites the
headline numbers come from. The noisy tail is concentrated in the smallest shapes, which are
launch-overhead-bound: the worst offenders are T=64/T=4 attention entries and 1×1 convs whose
per-call time is tens of microseconds. Every row carries its own CV in the JSON, so a noisy row
can be identified rather than averaged in silently.

**One caveat the CI does not capture.** The 9 e2e repeats are not independent — they trend
monotonically upward as the board heats: FP16 goes 20550 → 20638 ms and INT8 14577 → 14747 across
the run, while INT4 (the shortest) flattens after two repeats at 11996–12000. A t-interval assumes
i.i.d. samples, so the ±22 ms on FP16 is **optimistic**; the honest statement is that the drift
across a run is ~0.4%, which is larger than the quoted interval and is the reason cross-session
comparisons in this project are held to ~1%. The speedup ratios are much less affected, since all
three modes drift in the same direction within one process.

---

## 8. Caveats

**No result here is validated against trained weights.** `models/ldm/lsun_churches256/model.ckpt`
is an 856-byte stub whose `state_dict` has **0 entries**, loaded with `strict=False`, so every
weight is randomly initialised. Timing is unaffected — these kernels have no data-dependent
control flow — but **nothing here supports an image-quality claim**, and the INT4 static
calibration is a known-wrong placeholder (one shared scale, ~21× too large, across all 21 layers).

Nsight Compute counters are unavailable: `ncu` 2024.1.1 is installed but returns
`ERR_NVGPUCTRPERM` (re-verified). All attribution is CUDA-event timing plus profiler self-time,
never hardware counters — which is why §6 stops at "GroupNorm is the target" without a
stall-vs-bandwidth diagnosis.

The FP16 baseline is not uniformly fp16: 12 of its 33 torch convs receive fp32 inputs where the
quantized modes hand the same convs fp16 (§2). This inflates FP16's time on those specific shapes
and is the reason the unquantized conv rows are excluded from the reported ranges.

The measurement container was rebuilt on 2026-08-01 (`omegaconf`, `einops`, `pytorch_lightning`,
`tqdm`, `matplotlib` reinstalled; torch 2.4.1+cu124 unchanged). Small-shape attention numbers moved
relative to the 07-31 sessions — see `CHECKPOINT_REPORT_2026-08-01.md` §4.1.

---

## 9. Data and reproduction

| file | contents |
|---|---|
| `final_report_2026-07-28/data/kernel_suites_b128.json` | suites 1–3: every kernel signature, distribution + per-kernel profile |
| `final_report_2026-07-28/data/layers_final_b128.json` | suite 4 |
| `final_report_2026-07-28/data/e2e_final_b128.json` | suite 5, 9 repeats |
| `integration/benchmarks/report/kernel_suites_bench.py` | suites 1–3 driver |
| `integration/benchmarks/report/layer_pipeline_bench.py` | suite 4 driver |
| `integration/benchmarks/report/e2e_three_mode_bench.py` | suite 5 driver |
| `integration/benchmarks/report/ck_bench_stats.py` | the statistics contract shared by all five |
| `integration/benchmarks/report/ck_final_numbers.py` | emits every table above |

```bash
python3 integration/benchmarks/report/kernel_suites_bench.py \
  --batch 128 --warmup 30 --iters 60 --rounds 8 \
  --output docs/final_report_2026-07-28/data/kernel_suites_b128.json
LBENCH_BATCH=128 LBENCH_MODES=fp16,int8_baseline,int4_baseline \
  LBENCH_OUT=$PWD/docs/final_report_2026-07-28/data/layers_final_b128.json \
  python3 integration/benchmarks/report/layer_pipeline_bench.py
python3 integration/benchmarks/report/e2e_three_mode_bench.py \
  --batch 128 --steps 200 --repeats 9 --warmups 3 \
  --output docs/final_report_2026-07-28/data/e2e_final_b128.json
python3 integration/benchmarks/report/ck_final_numbers.py \
  --kernels docs/final_report_2026-07-28/data/kernel_suites_b128.json \
  --layers docs/final_report_2026-07-28/data/layers_final_b128.json \
  --e2e docs/final_report_2026-07-28/data/e2e_final_b128.json
```

Shapes are **captured from a live sampling step, never listed**. The older per-report scripts
hardcode tables that no longer match this model: `conv_kernel.py` benches Cin=128/256/512 at
64×64…8×8, while this UNet runs Cin ∈ {4,192,384,576,768,1152} at 32²…2² and has no 64² conv and no
128-channel conv at all; `bench_attn_kernel.py` lists qkv as 192→576 where the real projection is
192→768. Kernels are captured at the **C++ entry point**, not the module: the fused ResBlock calls
`conv2d_int8_evt_bias_residual_fp16` and the `gemm_w8a8_awq_*` family directly, so a module hook
sees 33 conv shapes in FP16 but only the 13 FP16 leftovers in INT8. Arguments are replayed
verbatim rather than synthesized — each attention shape has its own packing convention — with the
captured tensors parked on CPU and returned one signature at a time.

---

## 10. Open items, in priority order

1. **Real weights + recalibration.** Gates every accuracy claim; the INT4 calibration is known
   wrong, not merely unvalidated.
2. **INT4 GroupNorm** (3785.7 ms, 31.6% of INT4's total, worse than INT8's). Now quantified as the
   top target from three independent directions: the stage table, the level-by-level
   reconciliation in §6, and the bandwidth spread measured in
   `CHECKPOINT_REPORT_2026-08-01.md` §2.
3. **The quantized linear path buys nothing** (§3): 1785 → 1883 → 1792 ms across modes. Either the
   fused epilogue should pay for itself or these projections should stay FP16.
4. **T=16 attention core** is the only kernel that loses to FP16 (0.744×/0.711×, §1) and the
   interval is tight. Needs a small-shape kernel whose cost does not grow with T².
5. **The 13 unquantized conv shapes** (§2) — all 1×1 skip convs plus in/out. Quantizing them is
   what would move the conv stage from 2.70× toward the 3.94× the quantized kernels already reach.
6. **INT8's attention target of 1.5× weighted remains unmet** at a 1.36× median over the five
   shapes; the remaining gap is believed stall-bound, which needs the Nsight counters this
   container cannot provide.
