# Quantize-kernel vectorization — measured results

**GPU:** NVIDIA A40 (48 GB, SM 8.6) · **PyTorch:** 2.4.1+cu124 · **CUDA:** 12.4
**Model:** LSUN-Churches LDM-8 UNet (unconditional, 256×256) · **Batch:** 128 · **Sampler:** DDIM, 200 steps
**Date:** 2026-07-28 · repo: `feat/conv-attn-epilogue-fusion` @ [`dad8dfb`](https://github.com/xiaruize0911/MoDiff/commit/dad8dfb)

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
| fp16 | 210.20 | 1.00× |
| int8_baseline | 119.90 | 1.75× |
| **int4_baseline** | **110.09** | **1.91×** |
| int8_modiff | 125.89 | 1.67× |
| int4_modiff | 127.73 | 1.65× |

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

- **Attention is the single most expensive layer type in every mode** (36-46 ms/step) and shrinks the
  least under quantization (fp16 46.4 → int8/int4 baseline 36-38 ms) because it's the pre-existing
  custom int8/int4 flash kernel — none of this round of work touched it.
- **Conv shows a strong win**: fp16 41.9 ms → int4_baseline 15.9 ms (2.6×).
- **Linear/GEMM collapses hardest**: fp16 45.0 ms → ~8-9 ms across all 4 quantized modes (~5×), the
  single biggest per-layer-type speedup in this breakdown.
- **Norm/resize/quantize glue** drops from 76.8 ms (fp16) to 43-58 ms; the `_modiff` modes sit higher
  in this bucket (48-58 ms vs 43-50 ms baseline) because the temporal-delta cache path does strictly
  more GroupNorm/quantize work per step than the baseline path, trading some speed for its accuracy
  benefit (consistent with Section 1's `_modiff` vs `_baseline` note). The quantize-kernel slice of this
  bucket (5.5-6.3 ms flat across all 4 quantized modes) is what Section 3 below vectorized.

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

## 4. Correctness

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
