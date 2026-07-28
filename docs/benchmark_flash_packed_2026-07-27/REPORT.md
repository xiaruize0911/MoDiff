# Quantize-kernel vectorization — measured results

**GPU:** NVIDIA A40 (48 GB, SM 8.6) · **PyTorch:** 2.4.1+cu124 · **CUDA:** 12.4
**Model:** LSUN-Churches LDM-8 UNet (unconditional, 256×256) · **Batch:** 128 · **Sampler:** DDIM, 200 steps
**Date:** 2026-07-28 (refreshed after the int4 ahat-cache fix and a categorization-script correction, see below) · repo: `feat/conv-attn-epilogue-fusion` @ [`c80f2b3`](https://github.com/xiaruize0911/MoDiff/commit/c80f2b3)

**5 modes:** `fp16`, `int8_baseline`, `int4_baseline`, `int8_modiff`, `int4_modiff`. int8/int4 use the
fused-flash quantized attention kernel; `_modiff` adds the temporal-delta conv cache.

**Method:** speed = wall-clock / `torch.cuda.synchronize()`, GPU clock burn-in → 30 warm → 5×150 timed
steps. Category breakdown = `torch.profiler`, CUDA-device-only events (excludes CPU-dispatcher
double-counting), 15 profiled steps. fp16 and the 4 quantized modes were measured **in the same
process** so the speedup ratios are apples-to-apples — this repo's fp16 absolute timing has drifted
across sessions before due to environment/thermal state, so cross-session speedup numbers are not
reused here. Data: `data/*.json` · figures: `plots/*.png` · scripts: `scripts/*.py`.

---

## 1. Speedup vs fp16

| mode | ms/step | speedup |
|---|--:|--:|
| fp16 | 211.23 | 1.00× |
| int8_baseline | 120.11 | 1.76× |
| **int4_baseline** | **110.21** | **1.92×** |
| int8_modiff | 126.11 | 1.68× |
| int4_modiff | 127.65 | 1.66× |

![speedup vs fp16](plots/fig_speedup.png)

`int4_baseline` is the fastest mode overall — not `_modiff` — because the temporal-caching path trades
some raw speed for its accuracy benefit (subtract/accumulate overhead on every step, without skipping
any convolution work), matching this project's own documented expectation.

**Methodology caveat:** this fp16 baseline runs the default `MATH` SDPA backend (materialized,
unfused attention — the reference every quantized-mode accuracy number in this repo has been measured
against). Switching fp16's attention to PyTorch's fused `FLASH` backend measured ~116 ms/step in
isolation (`docs/benchmark_flash_packed_2026-07-27/data/sdpa_backend_e2e.json`) — which would flip
`int8_baseline`/`int8_modiff` to a *loss* against fp16 and shrink int4's margin. The speedup numbers
above are only as fair as the fp16 baseline they're computed against; they follow this repo's existing
convention rather than the fastest possible fp16.

---

## 2. Time cost by real layer type

![UNet layer architecture](plots/unet_layer_architecture.svg)

![time cost per layer type](plots/fig_breakdown.png)

The model's actual learnable layer types are **Conv2d**, **Attention**, and **Linear/GEMM** (the
AttentionBlock's `qkv`/`proj_out` are 1×1 convs — mathematically per-token Linear, and this repo routes
them through the same GEMM kernel as `nn.Linear`). GroupNorm+SiLU and resize (avg_pool/nearest-interpolate)
are non-learnable glue between those layers, not layers themselves, and "quantize" doesn't exist in the
original fp16 model at all — it's an artifact of the int8/int4 pipeline. See the architecture diagram
above for exactly where each type sits in the 35 ResBlocks / 21 AttentionBlocks that make up this UNet.

- **Attention is the single most expensive layer type in every mode** (36-47 ms/step) and shrinks the
  least under quantization (fp16 46.6 → int8/int4 baseline 36-38 ms) because it's the pre-existing
  custom int8/int4 flash kernel — none of this round of work touched it.
- **Conv shows a strong win**: fp16 48.2 ms → int4_baseline 15.9 ms (3.0×).
- **Linear/GEMM collapses hardest**: fp16 52.7 ms → 8-9 ms across all 4 quantized modes (~6×), the
  single biggest per-layer-type speedup in this breakdown.
- **Norm/resize/quantize glue** drops from 63.6 ms (fp16) to 43-58 ms; the `_modiff` modes sit higher
  in this bucket (48-58 ms vs 43-50 ms baseline) because the temporal-delta cache path does strictly
  more GroupNorm/quantize work per step than the baseline path, trading some speed for its accuracy
  benefit (consistent with Section 1's `_modiff` vs `_baseline` note). The quantize-kernel slice of this
  bucket (5.5-6.3 ms flat across all 4 quantized modes) is what Section 3 below vectorized; Section 5
  breaks the rest of this bucket down further.

**Categorization fix (this refresh):** the categorization script originally missed several Conv/GEMM
kernel name patterns — `ImplicitGemmConvolutionFusionPerSample` (no underscore in "ImplicitGemm"),
`ampere_fp16_s1688/s16816gemm*` (cuBLAS SASS kernels, never contain the literal substring `"cublas"`),
`sm80_xmma_gemm_*`, and `cutlass::Kernel2<..._tensorop_..._gemm...>` (no `wmma_` prefix) — so they fell
into the fp16 glue bucket instead of Conv/GEMM. This mattered most for **fp16**, where fixing it moved
~13 ms from glue into Conv (+6.3 ms) and GEMM (+7.7 ms); fp16's glue share dropped from 36.5% to 30.1%
of total time. The quantized-mode numbers barely moved (these kernels are a much smaller fraction of
their profiles). Fixed in `final_speedup_and_breakdown.py`/`glue_breakdown_detail.py`'s `CATEGORY_RULES`.

---

## 3. Quantize-kernel vectorization (this round of work)

Corrected profiling showed hand-written "quantize" kernels (fp16→int8/int4 conversion) at ~25-29% of
total GPU time, most moving memory one element per thread while other kernels in the same codebase
already used `float4`/packed-`int32` vectorized loads. Widened the memory-bound, order-independent
(non-reduction) paths to `half2`/`float2` — 2 elements per instruction instead of 1:

| file | kernels vectorized | ~time share |
|---|---|--:|
| `group_norm_silu.cu` | `gn_apply_delta_quantize_flat_vec2_kernel` / `_pack_flat_vec2_kernel` (modiff path) | ~18% |
| `group_norm_silu.cu` | `group_norm_silu_quantize_nhwc_vec2_kernel` / `_pack_nhwc_vec2_kernel` (baseline path) | ~18-19% |
| `attn_quant_gemm.cu` | `aq_qtok_packed_static_qk_vec2_kernel`, `aq_vquant_trans_packed_tiled_vec2_kernel` | ~5% |
| `modiff_delta_quantize.cu` | `static_quantize_and_update_ahat_kernel_int8_half_cache_vec2` (+ `_silu`) | ~1% |

**Deliberately not vectorized:** `gn_group_stats_kernel` and the pass-1 (reduction) halves of the two
`quantize_nhwc` kernels. Vectorizing a reduction's read side reassigns which elements each thread sums,
changing fp32 addition order. This file's own code comment already documented a historical incident (a
one-line `fmaxf` reordering once flipped int8 codes via a ~1 ULP variance perturbation); a real attempt
confirmed the risk empirically — it passed a random-data correctness gate but failed the
real-activation-statistics gate with `max_code_diff=1`, and was reverted.

![vectorization before/after](plots/fig_vectorization_before_after.png)

| mode | ms/step (before → after) | quantize-kernel share (before → after) |
|---|---|---|
| int8_baseline | 122.15 → 120.12 | 24.63% → 23.35% |
| int4_baseline | 111.06 → 110.15 | 24.29% → 23.70% |
| int8_modiff | 131.20 → 126.11 | 28.73% → 25.87% |
| int4_modiff | 128.75 → 127.27 | 25.41% → 24.89% |

`int8_modiff` shows the largest win: it exercises the modiff temporal-delta kernels (Tier 1), the
single largest vectorization target and the cleanest case (flat/stride-1, no reduction involved).

A synthetic non-64-multiple-`T` test case caught a real bug before it shipped:
`aq_vquant_trans_packed_tiled_vec2_kernel`'s 4-byte-packed store assumed `T % 4 == 0` alignment, true
for every production shape (1024/256/64) but not in general — fixed by gating that path on `T % 4 == 0`
with a scalar fallback.

---

## 5. Glue-bucket sub-breakdown and the int4 ahat-cache fix

Splitting the glue bucket further, by what actually calls each kernel (backed by an independent code
survey of `integration/fused_ops/`), turns up concrete fusion-actionable categories:

![glue bucket sub-breakdown](plots/fig_glue_subbreakdown.png)

| category | int8_base | int4_base | int8_modiff | int4_modiff | status |
|---|--:|--:|--:|--:|---|
| GN+quantize fused (K1 path) | 21.9 | 19.8 | 14.7 | 13.1 | already maximally fused |
| GN stats (modiff reduction) | — | — | 11.1 | 12.2 | unfused; vectorizing it was attempted and reverted (numerically unsafe, see Section 3) |
| Updown-block GN+quantize gap | 2.5 | 5.5 | 1.8 | 6.1 | **real gap**: `_prequant_gn_conv` skips `updown=True` ResBlocks |
| Ahat-cache update (modiff) | — | — | 1.4 | 1.5 | fixed this round (see below) |
| Attention quantize (standalone) | 5.5 | 5.7 | 5.5 | 6.3 | already assessed not worth a custom kernel (FLOP analysis, earlier session) |
| Resize (avg_pool/upsample) | 3.8 | 3.8 | 3.8 | 4.2 | unfused; no existing fusion kernel covers this model's up/down path |
| Skip-connection concat | 2.6 | 2.6 | 2.6 | 2.9 | unfused; flat cost in every quantized mode |
| Residual-add/dtype-cast/other | 6.4 | 12.3 | 6.7 | 11.0 | mostly small individual PyTorch elementwise kernels |

**Fixed this round:** `static_quantize_pack_and_update_ahat_kernel_int4_half_cache[_silu]` was still
scalar — its int8 sibling was vectorized earlier this session, but this int4-pack variant was missed.
Same safe pattern (pair-major, non-reduction, gated on `num_channels % 2 == 0`) applied and verified:
16/16 capture-compare cases bit-identical, `gn_modiff_verify_kernel.py`/`_realinput.py` (int8 and int4)
`max_code_diff=0`, `test_kernel_correctness.py` `ALL PASS`, whole-model `rel_err=0.0000` on all 5 modes.
Kernel-level effect: 1.405ms → 1.342ms/step — real, but at ~1ms of a 127ms step it's below the e2e
measurement's run-to-run noise floor, as expected for a low-risk/low-payoff item.

**Not touched, not recommended without more design work:**
- **GN stats reduction** (11-12ms, ~9-10% of modiff-mode total) — single-pass (Welford-style) algorithm
  could remove the second memory pass, but touches the same fp32-summation-order-sensitive code that
  already broke once under vectorization (Section 3). High risk, needs extensive re-verification.
- **Resize** — nearest-upsample could in principle be reordered after quantize (nearest is an exact
  index-select, order-safe) for the 4 up-transition blocks, but avg_pool **cannot**: averaging int8
  codes ≠ averaging then quantizing, so downsample would have to stay fp16, and a real win needs a
  custom CUTLASS im2col reader — high engineering effort for ~1ms of payoff.

---

## 6. Other problems and gaps found (broader sweep)

Beyond the glue bucket, a broader investigation (3 parallel code-reading passes) surfaced further gaps.
None of these have been fixed yet — listed here for triage, ranked by how load-bearing each looks.

**Correctness/robustness risk:**
- **`MODIFF_SDPA_BACKEND` and the flash-vs-SDPA autotune are not independent, and this interaction is
  untested.** `QuantizedStandardAttentionBlock._autotune_flash` times the custom int8/int4 flash kernel
  against `F.scaled_dot_product_attention` under whatever backend `MODIFF_SDPA_BACKEND` currently
  selects, then **freezes** the winner for the rest of the run. If `MODIFF_SDPA_BACKEND=flash` is set,
  fp16 SDPA becomes artificially fast relative to its documented MATH baseline, and the autotune can
  freeze "use fp16 SDPA" on blocks where, under the MATH baseline every accuracy number in this repo was
  measured against, the int8/int4 flash kernel would have won. Worth a test that pins both flags
  together and checks the autotune's decision doesn't silently flip.
- **Two fp32-`a_hat_cache` kernel variants (`static_quantize_and_update_ahat_kernel_int8` /
  `static_quantize_pack_and_update_ahat_kernel_int4`) look unreachable in production** — the calibrated
  hot path always uses fp16 cache once `is_calibrated=True`, and the static-scale kernels only fire
  once calibrated. Either confirm dead and delete, or find the live path (in which case it's a missed
  vectorization target, not the dead code it appears to be).

**Test-coverage gaps (all currently untested, ranked by exposure):**
- The **dynamic (cache-free) quantize path** — `sub_absmax_scale_kernel`, `dynamic_quantize_int8_fprop`,
  `dynamic_quantize_pack_int4_fprop` — used whenever calibration is unavailable, has **zero** coverage
  in any correctness gate (only the static/calibrated path is tested).
- **No test anywhere uses batch=1.** Every kernel-correctness test uses N=16/32; production always runs
  N=128. Batch=1 stresses reduction/broadcast kernels differently and has never been exercised.
- `attn_softmax_fp16`'s `T % 8 == 0` constraint (live in production fused attention) and the *dynamic*
  variant of `quantize_attn_qkv_packed`'s `hd % 2 == 0` check are exercised only by benchmark scripts,
  never by a correctness gate — unlike their *static* siblings, which are well-covered.
- `upsample2x_quantize_pack_noahat_fprop`'s `C % 2 == 0` and the `layout_transform.cu` `C % 4 == 0`
  checks (both live in the production attention path) have no test at all, not even with production
  shapes.
- `groups > 1` conv support is confirmed **fully dead code** (the UNet never passes `groups=`) —
  informational only, not worth testing.

**Other unvectorized memory-bound kernels (same low-risk class as this session's quantize-kernel work,
but outside the 3 files already covered):**
- `csrc/kernels/conv/conv_epilogue.cu`'s `scale_accumulate_half_cache_kernel` / `_residual_half_cache` /
  `scale_store_half_kernel` — pure per-element `o = acc*scale [+ residual]`, no reduction, called once
  per quantized Conv2d layer in every ResBlock, every step — the widest-reach unvectorized kernel found.
  The same file already has a working `float4` pattern for the fp32-cache siblings to copy.
- `csrc/kernels/util/layout_transform.cu`'s NCW↔CL transpose/quantize kernels — called twice per
  AttentionBlock per step (QKV-proj input, output-proj output), still scalar.
- Lower priority: several `attn_quant_gemm.cu` kernels (`aq_vquant_trans_kernel`, `from_i8_qtok/vtrans`)
  only fire on non-default/opt-in paths (calibration warmup, `MODIFF_ROUTE1=1`); `aq_qtok_packed_static_kernel`
  and the non-tiled `aq_vquant_trans_packed_kernel` appear to be **dead code** (no launch site found).

---

## 7. Correctness

- **Kernel-level:** `integration/tests/test_kernel_correctness.py`,
  `docs/benchmark_5mode_2026-07-20/scripts/gn_modiff_verify_kernel.py` / `_realinput.py`,
  `docs/flash_attention_2026-07-19/scripts/test_packed_quant.py`, plus three new capture-vs-compare
  scripts under `scripts/vectorize_verify/` — all `ALL PASS`, zero diffs.
- **Whole-model:** `integration/tests/e2e_output_check.py --compare` (seeded DDIM output) shows
  **`rel_err = 0.0000` for every one of the 5 modes** — bit-identical end-to-end output. The speedup in
  Section 3 is free; it is not a quality tradeoff.

## Commits

- [`13df347`](https://github.com/xiaruize0911/MoDiff/commit/13df347) — SDPA backend re-read-per-call fix (the fairness issue referenced in Section 1's caveat)
- [`dad8dfb`](https://github.com/xiaruize0911/MoDiff/commit/dad8dfb) — quantize kernel vectorization (Section 3)
- [`c80f2b3`](https://github.com/xiaruize0911/MoDiff/commit/c80f2b3) — int4 ahat-cache vectorization fix (Section 5)
