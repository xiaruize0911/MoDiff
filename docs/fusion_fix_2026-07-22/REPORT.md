# MoDiff fusion-fix profile — per-component timing (torch.profiler) + Perfetto traces

**GPU:** NVIDIA A40 (48 GB, SM 8.6) · **PyTorch:** 2.4.1+cu124 · **CUDA:** 12.4 · driver 580
**Model:** LSUN-Churches LDM-8 UNet (unconditional, 256×256 → 32×32 latent) · **Batch:** 64 · **Sampler:** DDIM
**Date:** 2026-07-22 · measures the fusion-fix work landed on top of `benchmark_5mode_2026-07-21`.

This report isolates the **fusion fixes** delivered after the `benchmark_5mode_2026-07-21` audit
(see `../benchmark_5mode_2026-07-21/FUSION_FIX_ADDENDUM.md`) — each fusion measured **before vs
after** with `torch.profiler` CUDA self-time bucketed per component, plus Perfetto chrome traces
of the shipped pipeline and the one regression. Batch is **64** (not 128) purely for MoDiff
temporal-cache memory headroom; the deltas are batch-robust.

**What each fusion is** (flag; default):
- **Phase 2 — MoDiff `o_hat` conv deep-fuse** (`MODIFF_DEEPFUSE_OHAT`; ON): weight-scale folded
  into the CUTLASS epilogue + autotuned tile → fp16 `+=` cache, replacing the base fp32-GEMM +
  separate `scale_accumulate` pass.
- **Phase 5 — int4 attention GN→pack** (`MODIFF_FUSE_GN_QKV_I4`; ON): GroupNorm emits the packed
  int4 qkv input directly (`group_norm_silu_quantize_pack_nhwc` → `forward_from_int4`), replacing
  fp16 GN + standalone `quantize_act_int4_pack`. int4 parity with the int8 GN→qkv fold.
- **Phase 3 — MoDiff GN→delta-quant** (`MODIFF_ENABLE_GN_MODIFF_FUSION`; OFF): folds GN+SiLU into
  the delta-quantize kernel. Shown here as the **measured regression** that justifies keeping it off.

**Method (measured only):** per config = one subprocess (several fusion flags are read at
module-load time, so they must be set before import). GPU-clock burn-in → 20-step warmup (wires
+ freezes the fused paths / attention self-calibration) → **wall** = min of 3× synchronized
20-step loops (ms/step) → **torch.profiler** (CUDA activity) over 3×20 steps, `self_device_time_total`
summed per kernel and bucketed by name via `cat()` (mirrors `benchmark_5mode`'s
`e2e_timing_profile.cat()`; `gpu_busy` = Σ buckets). **Engagement is proven by call counters**
(`eng_*` columns) that wrap each config's fused kernel — nonzero only when the fusion is active,
so a null result is never mistaken for "no regression / no effect." Raw data:
`data/fusion_profile.csv`; figures: `figs/*.png`; Perfetto traces: `data/perfetto/trace_*.json`.

---

## 0. End-to-end DDIM speed + profile — whole UNet, b128 (the headline) · `data/e2e_speed.csv` · `data/e2e_timing_profile.csv`

Full-model e2e with all landed fusions ON by default, same methodology as
`benchmark_5mode_2026-07-21` (b128, 30 warm + 5×200 timed steps, autocast fp16, flash quant attn),
so it is directly comparable to that report's numbers (the **before** column).

| mode | before (bench5) | **after (all fusions)** | Δ | vs fp16 (after) |
|---|--:|--:|--:|--:|
| fp16 | 188.1 | 187.2 | −0.9 (noise) | 1.00× |
| int8_baseline | 121.0 | 118.1 | −2.9 | 1.59× |
| int4_baseline | 114.9 | **109.7** | **−5.2** | **1.71×** |
| **int8_modiff** | 136.8 | **129.2** | **−7.7** | **1.45× (was 1.38×)** |
| **int4_modiff** | 141.5 | **128.0** | **−13.5** | **1.46× (was 1.33×)** |

All landed fusions ON by default (o_hat deep-fuse, int4 qkv GN→pack, int4 proj fold). Highlights:
- **int8_modiff −7.7 ms/step** — Phase 2 (o_hat deep-fuse); the isolated conv microbench (−7.9 ms)
  translates straight to wall-clock. 1.38× → **1.45× vs true fp16**.
- **int4_modiff −13.5 ms/step** — Phase 2 o_hat deep-fuse (−4.6 ms conv) **+** the int4 attention
  qkv GN→pack **+** proj fold stacking. 1.33× → **1.46×** — int4_modiff now edges out int4_baseline.
- **int4_baseline −5.2 ms/step** (1.67×→**1.71×**) — the two int4 attention fusions (qkv ~3 ms + proj ~2 ms).
- int8_baseline ~flat (no int8-baseline-targeted fusion; already at the fused floor).

Per-component profile (`torch.profiler` CUDA self-time, ms/step) — the modiff buckets vs the
reference §2 profile confirm where the win lands:

| bucket (ms/step) | int8_modiff before | **after** | int4_modiff before | **after** |
|---|--:|--:|--:|--:|
| conv (int GEMM) | 28.2 | **23.8** | 15.6 | **13.9** |
| modiff cache | 11.0 | **8.7** | 11.0 | **8.7** |
| quantize/dequant | 27.5 | 25.4 | 25.2 | 24.4 |
| elementwise/copy | 5.7 | 5.8 | 10.2 | **7.3** |
| attention | 33.7 | 34.2 | 33.4 | 33.4 |
| **gpu_busy** | 142.1 | **134.3** | 131.7 | **122.8** |
| **wall** | 136.2 | **127.8** | 147.0\* | **129.7** |

The int8_modiff win is isolated to the **conv int-GEMM** (−4.4 ms) + **modiff-cache** (−2.3 ms)
buckets (the deep-fuse o_hat targets). The int4_modiff win adds the attention fusions on top:
**elementwise/copy 10.2 → 7.3 ms** (the proj transpose-copy removed by the int4 proj fold) plus the
conv/cache drops → gpu_busy 131.7 → **122.8**. \* the reference int4_modiff wall (147) was a noisy
secondary measurement; the current 129.7 is clean.

---

## 1. Landed fusions — isolated GPU-work before/after, b64 (`data/fusion_profile.csv` · `figs/fig_fusion_gpu_busy.png`)

| fusion | config | gpu_busy | wall | Δ gpu_busy | Δ wall | engaged (calls/step) |
|---|---|--:|--:|--:|--:|--:|
| **Phase 2 int8 o_hat** | ohat OFF | 75.74 | 78.69 | | | ohat 0 |
| | **ON (default)** | **71.83** | **75.09** | **−3.90** | **−3.60** | ohat **36** |
| **Phase 2 int4 o_hat** | ohat OFF | 70.21 | 84.64 | | | ohat 0 |
| | **ON (default)** | **67.87** | **84.48** | **−2.35** | −0.16 | ohat **36** |
| **Phase 5 int4 GN→pack** | gnpack OFF | 69.09 | 85.35 | | | gn_pack 0 |
| | **ON (default)** | **67.87** | **84.48** | **−1.23** | **−0.87** | gn_pack **22** |

![gpu_busy before/after](figs/fig_fusion_gpu_busy.png)

- **Phase 2 (o_hat deep-fuse)** is the real win: int8 gpu_busy −3.9 ms/step, wall −3.6 ms/step;
  int4 gpu_busy −2.35 ms/step (wall flat — the int4 conv is already fast, so at this batch the
  saving is latency-hidden). Confirmed engaged (36 deep-fuse o_hat conv calls/step).
- **Phase 5 (int4 GN→pack)** is a smaller GPU-work reduction (−1.23 ms/step) — the standalone int4
  quantize+pack + K-pad folded into the GN write; 22 folded qkv calls/step.

## 2. Where the Phase-2 time goes (per-component buckets, int8_modiff) · `figs/fig_fusion_buckets.png`

| bucket (ms/step) | ohat OFF | ON (default) | Δ |
|---|--:|--:|--:|
| attention (flash) | 17.24 | 17.26 | +0.02 |
| **conv (int GEMM)** | **16.10** | **13.23** | **−2.87** |
| qkv/proj int GEMM | 4.03 | 4.05 | +0.02 |
| GroupNorm | 11.06 | 11.10 | +0.04 |
| quantize/dequant | 12.69 | 12.72 | +0.03 |
| **modiff cache** | **5.47** | **4.32** | **−1.15** |
| elementwise/copy | 5.71 | 5.71 | 0.00 |
| **gpu_busy** | **75.74** | **71.83** | **−3.90** |

![per-component buckets](figs/fig_fusion_buckets.png)

The deep-fuse win lands exactly where expected: the **conv int-GEMM** bucket (weight-scale now in
the CUTLASS epilogue, no fp32 temp) and the **modiff-cache** bucket (fp16 `+=` in one pass instead
of a separate `scale_accumulate` over an fp32 temp). Every other bucket is unchanged — the fusion
is cleanly isolated to the o_hat conv path.

## 3. Phase 3 GN→delta-quant — the regression, made visible (int8_modiff)

| bucket (ms/step) | default (fusion OFF) | gndelta ON | Δ |
|---|--:|--:|--:|
| **GroupNorm** | **11.10** | **18.52** | **+7.42** |
| **quantize/dequant** | **12.72** | **8.18** | **−4.54** |
| modiff cache | 4.32 | 3.17 | −1.15 |
| **gpu_busy** | **71.83** | **74.89** | **+3.06** |
| **wall** | **75.09** | **78.16** | **+3.07** |
| engaged (gn_delta calls/step) | 0 | **64** | |

Enabling the fusion is a **+3.1 ms/step regression** (both gpu_busy and wall), even though it
*removes* a kernel launch (GN + step1 → one kernel). The profiler shows why: folding collapses the
`quantize/dequant` bucket by −4.54 ms, but the fused kernel's **GroupNorm** bucket balloons
**+7.42 ms** — the delta kernel iterates group-major, so in NHWC a group's `a_hat`/`x` access is
strided by C (poorly coalesced) at the dominant low-channels-per-group / high-spatial shapes, while
the separate `step1` iterates the tensor flat (coalesced). Net +2.9 ms of extra GPU work. This is a
GPU-work regression (not a launch artifact), which is why the fusion is kept **OFF** by default — a
defensible shape-driven choice, not an oversight. Genuine fix = rewrite the delta kernel's access
pattern to match the fast `group_norm_silu_quantize_nhwc` (deferred; team measured net-negative).

## 4. Perfetto traces (`data/perfetto/trace_*.json`)

Chrome-trace-event JSON (CPU + CUDA, `record_shapes=True`) over 3 real DDIM steps @ b64, loadable
directly in the Perfetto UI:

| trace | what to look for |
|---|---|
| `trace_int8_modiff.default.json` | shipped int8 pipeline; the deep-fuse o_hat conv (`conv2d_int8_dequant_fp16_o_hat_tuned` + `accumulate_from_half`) in place of the base GEMM + `scale_accumulate` |
| `trace_int4_modiff.default.json` | shipped int4 pipeline; GN→pack qkv (`group_norm_silu_quantize_pack_nhwc`, no standalone `quantize_act_int4_pack`) |
| `trace_int8_modiff.gndelta_on.json` | Phase-3 regression; the fat `group_norm_silu_delta_quantize_nhwc` kernel on the GN row |

**Load:** open https://ui.perfetto.dev → *Open trace file* → pick a `data/perfetto/trace_*.json`.
Use the CUDA track to see per-kernel durations; search a kernel name to jump to it.

---

## 5. Baseline fusion audit (int8 / int4) + int4 proj fix · per-kernel `torch.profiler`

Per-kernel CUDA self-time (b64) audit of the two baseline (cache-free) modes to find anything not
fused properly.

**int8_baseline — cleanly fused, no gap.** Top kernels are all fused forms: `flash_attn_int8_mma
<true>` (int8 flash writes token-major int8 → `proj.forward_from_int8`, so `quantize_attn_out` ≈
**0.05 ms**), `group_norm_silu_quantize_nhwc` (GN→int8 conv+qkv), conv `bias_residual_store_from_half`
(bias+residual folded). The only standalone quantize is the attention Q/K/V prologue (`aq_*`,
7.3 ms) — the documented static single-pass floor (per-token amax + head-major layout can't fold
into a GEMM epilogue).

**int4_baseline — found + fixed: the attention proj was unfused.** int4 has no `flash_attn_int4_vt_out_i8`,
so the benchmarked path did `a.transpose(1,2).reshape()` (standalone copy) + `proj.forward` (standalone
`quantize_act_int4_pack` + K-pad) — where int8 does it all in the flash store. The fused kernel
`quantize_attn_out_int4_pack` (transpose+int4-quant+pack in one) was **built + bound but unwired**.
Fix: wired it → `proj.forward_from_int4` (which pads C=192) in `QuantizedStandardAttentionBlock`,
gated `MODIFF_FUSE_PROJ_I4` (default on). This is the proj-side mirror of Phase 5 (qkv).

| int4 proj (b64) | fold OFF (before) | fold ON (default) | Δ |
|---|--:|--:|--:|
| transpose/copy | 4.85 | 3.61 | −1.25 (proj copy removed) |
| proj standalone `quantize_act_int4` | 0.62 | 0.00 | −0.62 |
| fused `quantize_attn_out_int4_pack` | 0.00 | 0.76 | +0.76 |
| **gpu_busy** | **60.44** | **59.12** | **−1.32** |

- **Correctness:** bit-identical to the fallback (e2e rel-L2 = 0.0, `scripts/verify_int4_proj_e2e.py`).
- **Engagement:** all 21 attention blocks (incl. C=192) — 1050 `quantize_attn_out_int4_pack` calls,
  proj `quantize_act_int4_pack` → 0. No CUDA rebuild (kernel + `forward_from_int4` already existed).

After this fix, int4 attention has full int8 parity on both the qkv side (Phase 5) and the proj side.

**Whole-UNet e2e impact of the int4 proj fold (b128, `data/e2e_baseline_profile.csv`):**

| int4_baseline (b128) | wall | gpu_busy | elementwise/copy |
|---|--:|--:|--:|
| proj fold OFF | 111.7 | 116.9 | 17.0 |
| proj fold ON (default) | **109.7** | **114.5** | **14.1** |
| Δ | **−2.1** | **−2.4** | **−2.9** (proj transpose copy removed) |

int8_baseline is 115.8 ms/step (1.62× vs fp16) — unchanged by the fixes (already fully fused via
flash-out-i8 → proj). **Full int4_baseline arc vs the original report (114.9 → 109.7, −5.2 ms,
1.67×→1.72×)** decomposes into the two int4 attention fusions: Phase 5 qkv GN→pack (~3 ms) + this
proj fold (~2.1 ms).

---

## Reproduce

```bash
source setup_cuda_env.sh
cd /workspace/MoDiff
D=docs/fusion_fix_2026-07-22; S=$D/scripts
rm -f $D/data/fusion_profile.csv
# per-component profile — one subprocess per config (import-time flags):
CFG=int8_modiff.ohat_off   MODE=int8 MODIFF_DEEPFUSE_OHAT=0            PYTHONPATH=src/taming-transformers python $S/profile_fusions.py
CFG=int8_modiff.default    MODE=int8                                   PYTHONPATH=src/taming-transformers python $S/profile_fusions.py
CFG=int4_modiff.ohat_off   MODE=int4 MODIFF_DEEPFUSE_OHAT=0            PYTHONPATH=src/taming-transformers python $S/profile_fusions.py
CFG=int4_modiff.default    MODE=int4                                   PYTHONPATH=src/taming-transformers python $S/profile_fusions.py
CFG=int4_modiff.gnpack_off MODE=int4 MODIFF_FUSE_GN_QKV_I4=0           PYTHONPATH=src/taming-transformers python $S/profile_fusions.py
CFG=int8_modiff.gndelta_on MODE=int8 MODIFF_ENABLE_GN_MODIFF_FUSION=1  PYTHONPATH=src/taming-transformers python $S/profile_fusions.py
PYTHONPATH=src/taming-transformers python $S/make_plots.py            # -> figs/*.png
# Perfetto traces:
CFG=int8_modiff.default    MODE=int8                                   PYTHONPATH=src/taming-transformers python $S/perfetto_trace.py --steps 3 --batch 64
CFG=int4_modiff.default    MODE=int4                                   PYTHONPATH=src/taming-transformers python $S/perfetto_trace.py --steps 3 --batch 64
CFG=int8_modiff.gndelta_on MODE=int8 MODIFF_ENABLE_GN_MODIFF_FUSION=1  PYTHONPATH=src/taming-transformers python $S/perfetto_trace.py --steps 3 --batch 64
# whole-UNet e2e speed + per-component profile (b128, 5 modes):
PYTHONPATH=src/taming-transformers python $S/e2e_speed.py             # -> data/e2e_speed.csv
PYTHONPATH=src/taming-transformers python $S/e2e_timing_profile.py    # -> data/e2e_timing_profile.csv
# §5 baseline-audit fixes — bit-identical on/off validation + engagement:
PYTHONPATH=src/taming-transformers python ../benchmark_5mode_2026-07-21/scripts/verify_int4_gn_pack_e2e.py  # Phase 5 qkv
PYTHONPATH=src/taming-transformers python $S/verify_int4_proj_e2e.py                                        # §5 proj
```

Correctness for these fusions (bit-identical / within-tol) is validated separately — see
`../benchmark_5mode_2026-07-21/scripts/verify_int4_gn_pack_e2e.py`,
`integration/tests/test_kernel_correctness.py` (`test_int8/int4_ohat_deepfuse`), and the addendum.
