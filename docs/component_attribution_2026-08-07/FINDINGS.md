# Where the K=1 full-MoDiff step time goes: differential timing, and a trace that agrees with it

2026-08-07. Replaces `docs/delta_clip_2026-08-06/scripts/component_profile.py`, which produced two
invalid runs and whose defects are recorded in that report's last section. The method here is the
one agreed there: **A, differential wall clock with no profiler attached; C, one Perfetto trace per
configuration bucketed offline**; then hold the two against each other.

They agree. On the four deltas both methods measure, the largest disagreement is **0.34 ms/step**,
and on the one that matters most — projection MoDiff — it is **0.01 ms**:

| delta | A (wall, no profiler) | C (trace GPU) | difference |
|---|---:|---:|---:|
| `int8_ptq → modiff_conv_k4` | +3.89 | +4.13 | +0.24 |
| `modiff_conv_k4 → modiff_conv_k1` | +6.67 | +6.32 | −0.34 |
| `modiff_conv_k1 → modiff_full_k1` | **+22.43** | **+22.42** | **−0.01** |
| `int8_ptq → ptq_no_projquant` | +6.66 | +6.97 | +0.30 |
| `modiff_full_k1 → base_no_conv_modiff` | +45.45 | +45.91 | +0.46 |

That is the check `component_profile.py` failed by a factor of 2.2 (235.74 reported against 106.30
measured). Nothing below is quoted from a `record_function` scope.

## A. Differential timing (`scripts/differential_timing.py`, `data/differential_timing.json`)

Batch 128, DDIM 200, 3 warm-ups + 5 repeats, CUDA events, **no profiler in the loop**. Every arm
CV ≤ 0.39%; 11 of 12 ≤ 0.26%. Twelve arms in one process, except `fp16` (see the trap below).

**It reproduces the 2026-08-06 table**, which is what makes these numbers comparable to that report
rather than only to each other:

| config | here | 2026-08-06 | |
|---|---:|---:|---|
| fp16 | 105.94 | 105.84 | +0.1% |
| int8 PTQ | 73.61 | 73.60 | +0.01% |
| K=4 conv only | 77.50 | 77.38 | +0.2% |
| K=1 conv only | 84.17 | 83.85 | +0.4% |
| K=4 conv+proj | 99.81 | 99.58 | +0.2% |
| K=1 conv+proj | 106.59 | 106.30 | +0.3% |

### The ladder: int8 PTQ → the paper's datapath

| arm | ms/step | vs fp16 | marginal | what the marginal is |
|---|---:|---:|---:|---|
| `int8_ptq` | 73.61 | 1.439x | — | |
| `modiff_conv_k4` | 77.50 | 1.367x | **+3.89** | conv MoDiff |
| `modiff_conv_k1` | 84.17 | 1.259x | **+6.67** | K=4 → K=1 |
| `modiff_full_k1` | 106.59 | 0.994x | **+22.43** | MoDiff on the 42 attention projections |
| `modiff_full_k4` | 99.81 | 1.061x | −6.78 | the projections cost +22.31 at K=4 as well |

The projections dominate: **+22.4 ms of the +32.98 ms** that separates int8 PTQ from the paper's
configuration, at both refresh settings.

### Knockouts from `modiff_full_k1`

| arm | ms/step | marginal | reading |
|---|---:|---:|---|
| `base_no_qattn` | 106.14 | **−0.45** | reverting the whole QK^T/AV to fp16 SDPA is worth nothing |
| `base_no_qlinear` | 92.44 | **−14.15** | int8+MoDiff projections vs plain fp16 `nn.Linear` |
| `base_no_gnqkv_fusion` | 109.93 | **+3.33** | the 2026-08-06 qkv GN fusion, independently confirmed |
| `ptq_no_projquant` | 80.27 | **+6.66** | the flash qout epilogue, measured where it *is* live |
| `base_no_projquant` | 106.48 | **−0.11** | control: at the base it is already 0/21, so a no-op |
| `base_no_conv_modiff` | 152.04 | +45.45 | **not an ablation** — see the second trap below |

`base_no_qattn` at −0.45 ms closes a loop with `docs/delta_clip_2026-08-06`'s ablation, which found
removing the quantized attention *math* worth 1.005x–1.019x on quality. It is worth approximately
nothing on either axis, and it is the half of attention MoDiff structurally cannot reach.

The `ptq_no_projquant` / `base_no_projquant` pair brackets the epilogue from both sides: the same
switch costs 6.66 ms where the epilogue is live and 0.11 ms (inside CV) where MoDiff has already
disabled it. A single arm would have proved neither.

### Guards

`_assert_route` checks every arm against the configuration its label claims, after warm-up, from
the modules rather than the environment. Recorded per arm in the JSON: `qout_eligible` is 21/21 on
every non-MoDiff-projection arm and 0/21 on every MoDiff one; `base_no_gnqkv_fusion` suppressed
**12,600 real fusion calls**, so its +3.33 is a knockout that fired rather than one that missed.

## C. Trace, bucketed offline (`scripts/trace_configs.py`, `scripts/bucket_traces.py`)

Seven configurations, batch 128, 8 UNet steps each, one `step/NNN` slice per step, driven directly
so no DDIM scheduler math straddles a boundary. The configurations are **imported from
`differential_timing.py`**, not restated, so A and C provably measure the same thing. Traces are in
`traces/*.json.gz`, openable at <https://ui.perfetto.dev>.

### Alignment with the profiler-free clock

| config | trace GPU ms/step | wall ms/step | trace/wall |
|---|---:|---:|---:|
| fp16 | 102.35 | 105.94 | 0.966 |
| int8_ptq | 69.75 | 73.61 | 0.948 |
| modiff_conv_k4 | 73.88 | 77.50 | 0.953 |
| modiff_conv_k1 | 80.20 | 84.17 | 0.953 |
| modiff_full_k1 | 102.62 | 106.59 | 0.963 |
| ptq_no_projquant | 76.72 | 80.27 | 0.956 |
| base_no_conv_modiff | 148.52 | 152.04 | 0.977 |

**0.948–0.977, mean 0.959.** The trace accounts for 95–98% of the wall clock everywhere, and the
shortfall is in the same direction and roughly the same size in all seven — which is what makes the
*deltas* trustworthy even though the absolute totals are 4% short. Two things are in the gap and
neither is profiler inflation: the GPU idles between kernels, and the wall-clock arms run
`sampler.sample()` (DDIM scheduler tensor math included) while the traced steps drive the UNet
directly. The ratio being flat means both cancel in a difference.

### Composition (GPU ms/step, profiler attached — read as shares, not as totals)

| bucket | fp16 | int8_ptq | conv_k4 | conv_k1 | full_k1 | ptq_no_projquant |
|---|---:|---:|---:|---:|---:|---:|
| conv | 43.85 | 27.03 | 28.30 | 28.15 | 27.85 | 27.12 |
| norm_quantize | 20.21 | 17.48 | 9.21 | 7.82 | 6.98 | 17.56 |
| elementwise | 23.40 | 7.30 | 8.16 | 9.56 | 13.57 | 9.19 |
| attention | 11.36 | 8.83 | 8.87 | 8.84 | 8.80 | 8.87 |
| linear_gemm | 2.84 | 9.10 | 9.16 | 9.15 | 15.36 | 8.03 |
| delta_quantize | 0.00 | 0.00 | 10.16 | 16.68 | 25.45 | 0.00 |
| attn_quantize | 0.00 | 0.00 | 0.00 | 0.00 | 4.59 | 4.59 |
| quantize | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.35 |
| other | 0.69 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| **total** | **102.35** | **69.75** | **73.88** | **80.20** | **102.62** | **76.72** |

`other` is ≤ 0.69 ms in every column (and 0.01 ms in all six quantized ones), so nothing large is
being guessed at. The residue is `cutlass::Kernel2`, memsets and memcpys.

Note `attention` is flat at 8.80–8.87 across all five quantized configurations. Nothing anyone did
to the projections, the conv path or the refresh rate moved the attention score kernels at all,
which is the trace-side statement of `base_no_qattn`'s −0.45 ms.

### `suite_of()` needed a conv fix before any of this was readable

`kernel_suites_bench.suite_of()` is the starting point and is still reported as its own column, but
it is wrong on raw CUDA kernel names and this report does not pretend otherwise. It was written for
*entry-point* names, where "conv2d" appears literally. The kernels a conv entry point actually
launches are `cutlass::Kernel<modiff::ImplicitGemmConvolutionEVT…>` and
`sm80_xmma_fprop_implicit_gemm_…_nhwckrsc_nhwc_…` — both contain "gemm", neither contains "conv2d" —
so it files 11.1 of the int8 trace's 27.0 ms of conv under `linear`, and leaves 43 of fp16's 98 ms
unclassified entirely. `bucket_of` tests the conv patterns first. Three further rules were needed,
each verified against `csrc/` rather than guessed:

* `gn_stats_partials_chanmajor_kernel` / `gn_stats_reduce_partials_kernel` → `norm_quantize`
  (`csrc/kernels/norm/group_norm_silu.cu`), 4.7 ms/step that had been in `other`
* `aq_*` → a dedicated `attn_quantize` bucket (`csrc/kernels/attention/attn_quant_gemm.cu`). Kept
  separate from `quantize` because these are precisely the passes a qout epilogue absorbs, so their
  presence or absence *is* the epilogue measurement
* `quant_act_int8_kernel` is spelled without the "ize" (`csrc/kernels/linear/gemm_wxax.cu`), so a
  bare `quantize` pattern silently left 1.35 ms/step of it in `other`

## By layer kind: conv / attn / lin (`scripts/layer_attribution.py`)

The bucket table above groups by what a kernel *does*. This groups by *who called it* — the question
`component_profile.py` was built for. Name-based bucketing cannot answer it, because
`gn_apply_delta_quantize_flat_vec2_kernel` serves 62 MoDiff convs **and** the 21 attention qkv, and
`static_quantize_and_update_ahat_..._vec2` serves 8 convs **and** the 21 proj.

Two facts in the traces settle it without instrumenting the measured region.

**Launch counts identify the layer sets exactly**, and they reproduce the 2026-08-06 fusion audit's
inventory (62 convs on `forward_gn_fused_modiff`, 8 on the unfused `_forward_modulated`, 21
attention blocks, 42 projections) without being told it. Calls per step:

| kernel | `conv_k1` → `full_k1` | `conv_k4` → `full_k4` | the difference is |
|---|---:|---:|---|
| `gn_apply_delta_quantize_flat_vec2` | 62 → 83 | 62 → 83 | the 21 qkv |
| `gn_delta_absmax_flat_vec2` | 62 → 83 | **15.5 → 36.5** | the 21 qkv |
| `gn_stats_partials_chanmajor` | 62 → 83 | 62 → 83 | the 21 qkv |
| `static_quantize_and_update_ahat_vec2` | 8 → 29 | **2 → 23** | the 21 proj |
| `delta_absmax_fp16` | 8 → 29 | **0 → 21** | the 21 proj |
| `gemm_w8a8_kernel_awq` | 21 → 42 | 21 → 42 | the qkv joining the proj |

`_assert_inventory` checks that every one of these deltas is exactly 21, per pair, so an attribution
built on a layer count that no longer holds fails rather than silently re-weighting. The *base*
counts are not checked because they are K-dependent — which is itself a result, below.

**The `conv_kN → full_kN` differential is the projection share.** The two configs differ in exactly
one thing, so for a shared kernel the conv part is its value in `conv_kN` and the projection part is
the difference. Splitting by call count would be wrong — a conv at `[128,C,H,W]` and a qkv at
`[128·1024,192]` do not cost the same per call.

### GPU ms/step by layer kind

| family | fp16 | int8_ptq | conv_k4 | **full_k4** | conv_k1 | **full_k1** | ptq_no_projquant |
|---|---:|---:|---:|---:|---:|---:|---:|
| conv | 87.47 | 49.75 | 53.77 | **52.93** | 60.14 | **59.75** | 49.93 |
| attn (score path) | 11.36 | 10.89 | 10.93 | **13.34** | 10.90 | **13.39** | 17.40 |
| lin (42 projections) | 2.84 | 9.10 | 9.16 | **29.37** | 9.15 | **29.46** | 9.37 |
| unattributed | 0.69 | 0.01 | 0.01 | **0.01** | 0.01 | **0.01** | 0.01 |
| *attn + lin* | 14.20 | 19.99 | 20.09 | *42.71* | 20.05 | *42.85* | 26.77 |
| **total (trace)** | 102.35 | 69.75 | 73.88 | **95.65** | 80.20 | **102.62** | 76.72 |

Shares: full MoDiff is **55.3 / 13.9 / 30.7 %** at K=4 and **58.2 / 13.0 / 28.7 %** at K=1, against
fp16's 85.5 / 11.1 / 2.8. Scaled to the profiler-free wall clock (each config by its own trace/wall
ratio):

| | conv | attn | lin | total |
|---|---:|---:|---:|---:|
| fp16 | 90.53 | 11.76 | 2.94 | 105.94 |
| int8 PTQ | 52.50 | 11.49 | 9.60 | 73.61 |
| full MoDiff K=4 | **55.23** | **13.92** | **30.65** | 99.81 |
| full MoDiff K=1 | **62.07** | **13.91** | **30.60** | 106.59 |

### `MODIFF_DELTA_REFRESH` never reaches the projections

Adding the K=4 row makes this fall out immediately. K=4 → K=1 costs 6.78 ms, and **all of it is
conv**:

| family | K=4 | K=1 | Δ |
|---|---:|---:|---:|
| conv | 55.23 | 62.07 | **−6.84** |
| attn | 13.92 | 13.91 | +0.01 |
| lin | 30.65 | 30.60 | +0.04 |

attn and lin are identical to within 0.05 ms — far inside the 2.5% conv drift on that pair. The
launch counts say why and the source confirms it: `delta_refresh` is read in
`OptimizedInt8Conv2d.__init__` and `Int4Conv.__init__` only, and `MODIFF_DELTA_REFRESH` appears
nowhere in `wxax_linear.py`. `QuantLinearWxAx.forward` calls `_mc.delta_absmax_fp16(...)` with
`report_next=False` on **every** step (`wxax_linear.py:201`), so the 42 projections recompute their
delta scale unconditionally. In the counts: the convs' `gn_delta_absmax_flat` drops 62 → 15.5
calls/step at K=4, while the projections' contribution to the same kernel is +21 at *both* refresh
settings.

Two consequences:

* It explains why the projections cost +22.3 at K=4 and +22.4 at K=1 — the knob that is supposed to
  make MoDiff cheaper simply does not apply to them.
* It is an **unexploited saving of roughly 2.5–3 ms/step**, with no new kernel: the projections'
  own scale recomputation is `gn_delta_absmax_flat` +1.94 (qkv) and `delta_absmax_fp16` +1.84
  (proj) = 3.78 ms/step, and a K=4 schedule would remove about three quarters of it, exactly as it
  does for the convs (6.30 → 1.58). It also has a known quality cost — a stale scale is what the
  code ceiling was added for on 2026-08-06 — so it is a measurement, not a free win.

Note the paper-alignment reading changes slightly too: the paper requires a dynamic per-step scale,
and the projections have been satisfying that requirement at every K all along. Only the conv path
was ever on a K schedule.

Boundary rules, stated because they are choices:

* **`lin` is the 42 attention projections**, which live *inside* the 21 attention blocks. So `attn`
  here is the score path only — QK^T, softmax, AV, and the quantize/repack passes feeding it. Both
  are given, and summed, because "attention" means both things in this project.
* **The attention block's GroupNorm follows the kernel it was fused into.** Under full MoDiff the
  qkv's GN is inside `gn_apply_delta_quantize_flat_vec2`, so it lands in `lin`; in int8 PTQ the same
  normalization is a separate `group_norm_silu_quantize_nhwc_vec2` call and lands in `attn`. That is
  ~2.06 ms (98.1 µs/call × 21, measured in `conv_k1` where those 21 calls are alone in the kernel).
* Everything else in the ResBlock / up-down path is `conv`: upsample, avgpool, the skip concat, the
  residual elementwise adds.

Error bars and exclusions:

* The conv kernels are not bit-identical across a differential pair. EVT conv total: 26.03 → 25.74
  (**1.11%**) for the K=1 pair, 26.17 → 25.50 (**2.54%**) for K=4, 24.92 → 25.00 (**0.33%**) for
  `int8_ptq → ptq_no_projquant`. That is the error on each split; ~0.65 ms at K=4.
* `unattributed` is 0.01 ms/step in every quantized config. Only fp16 has more (0.69 ms,
  `cutlass::Kernel2`).
* **`base_no_conv_modiff` is excluded from this table.** It swaps the fused EVT convs for the
  generic CUTLASS conv, so "the conv part is unchanged" — the assumption every differential split
  rests on — is false by 25 ms.
* Configs with no differential base (fp16, int8 PTQ, both conv-only arms) charge all shared
  elementwise to conv. For the quantized ones the exposure is small (2.67 ms total, and the
  attention residual is fused into `gemm_*_awq_bias_res` there), but **fp16's 23.4 ms of elementwise
  is all charged to conv** and some of it is the attention blocks' residual adds, so fp16's conv row
  is an over-estimate by an amount this data cannot bound.

**The whole of projection MoDiff's +22.42 lands in two places**: `lin` +20.31 and `attn` +2.49, with
conv flat to within its own drift (60.14 → 59.75). The `attn` +2.49 is the `aq_*` repack passes,
which are attention-side kernels that only exist because the qout epilogue went away.

Read against fp16, this is the whole speed story in three rows: quantization takes **conv from 87.47
to 59.75** (−27.7) and attn from 11.36 to 13.39, while MoDiff on the projections takes **lin from
2.84 to 29.46** (+26.6). The conv saving and the projection cost are the same size, which is why the
speedup is 0.994x.

## What the +22.43 ms of projection MoDiff is made of

The trace diff `modiff_conv_k1 → modiff_full_k1`, with `int8_ptq → ptq_no_projquant` used as an
independent instrument for the epilogue term:

| bucket | Δ ms/step |
|---|---:|
| `delta_quantize` | +8.77 |
| `linear_gemm` | +6.21 |
| `attn_quantize` | +4.59 |
| `elementwise` | +4.02 |
| `norm_quantize` | −0.84 |
| `conv` | −0.29 |
| `attention` | −0.35 |
| **total** | **+22.42** (wall +22.43) |

**The epilogue term is the same in both configurations, to 0.03 ms.** Removing the qout epilogue
from int8 PTQ and enabling projection MoDiff produce an identical kernel signature:

| kernel | PTQ → no-projquant | conv_k1 → full_k1 |
|---|---:|---:|
| `aq_vquant_trans_packed_tiled_vec2_kernel` | +1.82 | +1.81 |
| `aq_qtok_packed_static_qk_flat_kernel` | +1.79 | +1.79 |
| `aq_qtok_packed_static_qk_vec2_kernel` | +0.98 | +0.99 |
| `at::native::elementwise_kernel` | +1.79 | +1.76 |
| `flash_attn_int8_qi8packed_small_qout_kernel` | −0.35 | −0.35 |
| `gemm_w8a8_kernel_awq_out_i8` | −5.25 | −5.27 |

Two independent routes to the same six numbers is the strongest evidence in this report. It means
`ptq_no_projquant`'s clean **+6.66 ms wall** is a fair estimate of the epilogue term buried inside
the +22.43, and it did not have to be untangled from the MoDiff arm by name-based attribution — the
thing name-based attribution cannot do.

The rest separates through the GEMM. `gemm_w8a8_kernel_awq` (the plain, non-`out_i8` variant) reads
3.28 in PTQ, **7.46** once the epilogue is removed, and **14.80** under projection MoDiff. The step
from 7.46 to 14.80 is the o_hat read-modify-write with everything else held fixed:

| term | ms/step | can an epilogue remove it? |
|---|---:|---|
| attention re-quantize passes the qout epilogue used to absorb | **+6.0 to +6.7** | **yes — this is Part 3** |
| o_hat accumulate inside the projection GEMM (Eq 9 state) | **+7.3** | no |
| a_hat delta machinery on the 42 projections (`delta_quantize`) | +8.8 | partly; Part 1 already took 3.33 of this shape on the qkv |
| o_hat / residual `vectorized_elementwise` adds | +2.3 | partly |
| GEMM identity change `out_i8` → plain, second-order norm/conv | −2.2 | n/a |

### So: what the flash epilogue change is worth

**Part 3's ceiling is ~6.7 ms of the 22.4, and 6.7 is an upper bound, not an estimate.** An
a_hat-aware `flash_attn_int8_qi8packed_kv_static_qout` must additionally subtract `a_hat_proj`,
update it, and obtain a delta scale the flash kernel cannot compute at write time — none of which
today's qout epilogue pays for. Landing it perfectly would take K=1 full MoDiff from **106.59 to
about 99.9 ms/step, i.e. 0.99x → 1.06x fp16.**

That does not rescue the configuration. Conv-only MoDiff at K=1 is already 84.17 ms (1.26x), and
`docs/delta_clip_2026-08-06` measured what the projections buy over it: **0.976x relL2 at A4 r=1.0,
1.014x — worse — at the A4 clip optimum, and nothing at all at A5 and below**. The projections pay
only at A8/A7 (0.789x/0.811x, 3/3 seeds). So the honest scope for the flash epilogue work is: *it
recovers at most 6.7 ms of a 22.4 ms cost that is only worth paying at A8, in a configuration the
clip already beats on both axes.* This report does not say don't do it; it says the ceiling is 30%
of the cost, and it is now measured rather than inferred from a kernel-name profile.

This **corrects the 2026-08-06 reading** that "roughly two thirds of the +25.9 ms is collapsed
fusion, not the irreducible a_hat/o_hat bandwidth". Counting the 3.33 ms Part 1 already banked plus
the 6.7 ms ceiling here, fusion accounts for about **10 of 22.4 ms — 45%, not two thirds** — and the
single largest remaining term, the o_hat RMW at +7.3 ms, is exactly the bandwidth Stage 3.3
identified and no epilogue removes.

## Two traps, both caught by guards rather than by reading the numbers

### `MODIFF_QUANT_LINEAR` reaches modes it does not belong to, and the fp16 arm was not fp16

The first run's `fp16` arm inherited `MODIFF_QUANT_LINEAR=1` from the shared base env.
`benchmark_ldm`'s `quant_lin` block is **not gated on the mode**, so it converted **79 `nn.Linear`
to W8A8** and switched attention to token-major. It completed and reported 109.64 ms/step under the
label "fp16" — 3.5% off, entirely plausible, and wrong. The route check read `wxax: 79`.

`_assert_route` now refuses it, and the fp16 arm was re-measured alone (105.94, matching the
2026-08-06 value to 0.1%). That makes fp16 the one row in `differential_timing.json` measured in a
different process; `fp16_separate_process` records it. Nothing structural depends on it — every
arm's `delta_from` is another quantized arm, so fp16 enters only as the `speedup_vs_fp16`
denominator.

### `enable_modiff_mode(model, False)` is not an ablation of conv MoDiff

`base_no_conv_modiff` flips `modiff_enabled` at runtime, after `_setup_model` has already wired the
ResBlocks. It came out **45 ms slower**, not faster, so it was traced rather than explained. The
kernel identities settle it — this configuration is not the PTQ path and is not any shipped mode:

| kernel | full_k1 | no_conv_modiff | |
|---|---:|---:|---|
| `cutlass::Kernel<modiff::ImplicitGemmConvolutionEVT…>` (two shapes) | 14.01 + 11.73 | 0 | the fused EVT convs vanish |
| `cutlass::Kernel<cutlass::conv::kernel::ImplicitGemmConvolution…>` | 0 | **+25.11** | replaced by the generic non-EVT conv |
| `group_norm_silu_nhwc_kernel` | 1.59 | **24.54** | GN+SiLU now standalone, quantize stripped out |
| `scale_quantize_int8_kernel` | 0 | **+5.92** | the quantize reappears as its own pass |
| `gn_apply_delta_quantize_flat_vec2` + `gn_delta_absmax_flat_vec2` | 10.74 + 8.22 | 0 | |
| `cat2_channels_last_fp16_kernel` → `CatArrayBatchedCopy` | 1.93 | 3.47 | the fused concat is gone too |
| `elementwise` bucket, total | 13.57 | **53.61** | |

Turning the flag off at runtime drops the whole fused int8 datapath onto an unfused fallback. The
honest conv-MoDiff marginal is the ladder's **+3.89 ms**, where the mode switch does the reverting
at setup time. Anyone reaching for `enable_modiff_mode(..., False)` as an A/B control should read
this row first.

## Reproducing

```bash
python docs/component_attribution_2026-08-07/scripts/differential_timing.py   # A,  ~40 min
python docs/component_attribution_2026-08-07/scripts/trace_configs.py         # C,  ~12 min
python docs/component_attribution_2026-08-07/scripts/bucket_traces.py         # offline, free
python docs/component_attribution_2026-08-07/scripts/layer_attribution.py     # offline, free
```

`bucket_traces.py` touches no GPU: the bucket rules can be revised and re-run against the committed
traces for nothing, which is the reason to trace once rather than instrument per component.
`--arms` / `--configs` narrow either run. `trace_configs.py` merges into the existing manifest, so a
single configuration can be added without re-tracing the others.

The checkpoint is the 856-byte stub with an empty `state_dict`, so all weights are random. Kernel
cost here is data-independent and every shape and launch sequence is real, so the timing is sound —
**nothing in this document is a quality statement.**

## Open

1. **Part 3, with the ceiling now known.** ~6.7 ms of 22.4, upper bound. The constraints from
   `docs/delta_clip_2026-08-06` still hold: the delta scale has to come from a previous step's
   `report_next`, or a one-step-stale scale has to be accepted, and the dual-output kernel's
   `__half2` pairing needs a `TORCH_CHECK` that `n_out` is even.
2. **The `delta_quantize` +8.77 on the projections** is now the largest single term Part 3 does not
   touch, and 3.33 ms of the same shape was recovered on the qkv by Part 1. Whether the same fusion
   reaches the `proj` side is not answered here.
3. **`base_no_qattn` at −0.45 ms** says the quantized attention math earns nothing at batch 128. It
   also costs nothing, so this is not an action — but it does mean `MODIFF_QUANT_ATTN` is a knob
   with no measured benefit on either axis at this batch size.
4. **A refresh schedule for the projections**, which the layer attribution shows do not have one:
   ~3.78 ms/step of unconditional scale recomputation, of which a K=4 schedule should remove about
   three quarters. Bounded work — `QuantLinearWxAx` would need the `report_next` / `_scale_buf`
   mechanism the conv path already has — but it changes numerics on reuse steps, so it needs the
   paired-seed relL2 protocol, not just a timing run.
