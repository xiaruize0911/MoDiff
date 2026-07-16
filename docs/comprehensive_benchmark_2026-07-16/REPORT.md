# MoDiff kernel-fusion session report (2026-07-16)

Consolidates this session's kernel work on the LSUN-churches latent-diffusion UNet, with fresh,
noise-controlled speed/IO/profile numbers. **Hardware:** NVIDIA A40 (Ampere sm_86; fp16 149.7 TFLOP/s, int8
299 TOP/s, int4 599 TOP/s, DRAM 696 GB/s). **Config:** batch 32, DDIM, per-step. Raw data in [`data/`](data/),
scripts in [`scripts/`](scripts/). Supersedes / extends [`../comprehensive_benchmark_2026-07-15/`](../comprehensive_benchmark_2026-07-15/).

> ### Measurement rigor (noise control)
> A40 idles at 210 MHz, boosts to 1740 MHz, no clock-locking → **warmup dominates**. Every pipeline number:
> **≥7 s sustained `sample()` warmup** + **12 back-to-back timed runs** (median/min/stdev), **GPU-busy**
> (`torch.profiler` device self-time, throttle-robust) as the primary metric. Headline fusion numbers (§3)
> were taken across **2 independent process re-launches**; they agree to **≤0.06 ms** (every config's within-run
> stdev ≤0.10 ms), so wall and GPU-busy agree and the results are noise-free.

---

## Session summary (what changed)

| # | Work | Result |
|---|---|---|
| §1 | `gemm_wxax` **half2-epilogue** optimization | int4 qkv/proj GEMM now **beats fp16 on 5/6 shapes (≤2.13×)**, int8 on 4/6 |
| §2 | **int8-conv-out → GN** fusion (quality + kernel) | quality ~free (rel 0.0023–0.0033); GN-int8-in kernel ≤1 code, ~1.05×; conv-side blocked (needs direct-int8 epilogue) |
| §3 | **qkv-int-out → flash** fusion (int8 **and** int4) | best attention-quant config: **−1.3/−1.7 ms vs §6**, **−21% peak**; still +2.8 ms vs fp16 (flash SDPA is the wall) |
| §4 | Full 6-mode pipeline (fresh) | `int4 base` **1.16× vs fp16**, `int8 base` 1.12×; quantization Amdahl-bounded to the conv bucket |

---

## §1. GEMM `half2`-epilogue optimization

The `gemm_wxax` W8A8/W4A4 epilogue wrote fp16 outputs one scalar at a time. Vectorizing to `half2` stores
(c0 even, c0+1<N since N%64==0) flipped it from losing on every qkv/proj shape to winning on most. Per-shape
vs cuBLAS fp16 (batch 32; [`data/gemm_wxax_shapes.csv`](data/gemm_wxax_shapes.csv)):

| shape (M,K,N) | role | int8 ×fp16 | int4 ×fp16 |
|---|---|--:|--:|
| 32768,192,576 | qkv C192/T1024 | 0.41× | 0.77× |
| 32768,192,192 | proj C192 | 0.62× | **1.02×** |
| 8192,384,1152 | qkv C384 | 0.96× | **1.39×** |
| 8192,384,384 | proj C384 | **1.20×** | **1.59×** |
| 2048,768,2304 | qkv C768 | **1.02×** | **1.64×** |
| 2048,768,768 | proj C768 | **1.34×** | **2.13×** |

The lone laggard is the large-M/small-K C192 qkv (write/mainloop-bound); §3 fixes its write by emitting int8.

## §2. int8-conv-output → GroupNorm fusion

The one structural int8 handoff in the UNet: `in_conv`'s output feeds *only* the out-norm GroupNorm (skip
comes from `x`; temb is scale-shift inside GN). Making in_conv write int8 and GN read int8 avoids a fp16
write+read on that handoff.

- **Quality ~free:** fake-quantizing the in_conv output to int8 adds only **0.0033 rel-err (per-tensor) /
  0.0023 (per-channel)** on int8_baseline — far below the 0.02 gate.
- **Kernel built:** `group_norm_silu_dequant_quantize_nhwc` (int8-in GN) matches the fp16-in path to **≤1 int8
  code (100% of elements)** and is **~1.03–1.09×** faster (reads half the bytes).
- **Blocker:** the conv int8-output path (`relu_requant`) writes fp16 scratch *then* int8 → more traffic
  (handoff 0.83–0.97×). Realizing the write saving needs a direct-int8-output CUTLASS conv epilogue; projected
  e2e ~2%, deferred as low effort/payoff. Concept validated.

## §3. qkv-int-output → flash-attention fusion (int8 **and** int4)

**Motivation.** The int8 flash path (§6, opt-in) did a fp16 round-trip: qkv Linear writes fp16 `[B,T,nh,3,hd]`,
`quantize_qkv_int8` reads it back → int8. This fusion has the qkv Linear **emit int8 directly**
(`gemm_w8a8_out_int8` / `gemm_w4a4_out_int8`, reusing §1's kernels), a light **int8 transpose**
(`transpose_qkv_int8`, reusing the `quantize_qkv` layout) reorders to head-major, and the **unchanged** int8
flash consumes it. Scales are **calibrated static** (Q/K per-tensor, V per-channel-over-T) — the GEMM epilogue
can't compute per-token scales, so flash is fed constant/calibrated `sq,sk,sv`. int4 mode keeps int8 flash
(int4 scores too lossy), differing only in the qkv GEMM (`gemm_w4a4_out_int8`). Opt-in: `MODIFF_QKV_FLASH_FUSED=8|4`.

**Correctness (F0/F1;** [`scripts/fusion_kernel.py`](scripts/fusion_kernel.py)**):** `gemm_w{8a8,4a4}_out_int8`
== `round(fp16_gemm·oscale + bias·oscale)` to **≤1 int8 code (100%)**; `transpose_qkv_int8` is an **exact**
match to the head-major gather.

**Quality (F3;** [`scripts/fusion_quality.py`](scripts/fusion_quality.py)**),** latent rel-err vs fp16 attention,
C192/T1024 block:

| config | rel-err vs fp16 |
|---|--:|
| §6 per-token int8 flash | 0.0030 |
| **fused int8 (W8A8→flash)** | **0.0097** (< 0.02 gate ✓) |
| fused int4 (W4A4→flash) | 0.0840 (int4, reported) |

Static scales add ~0.007 over per-token — still safe for int8; int4 is lossy as expected.

**Speed / memory (F4;** [`data/fusion_pipeline.csv`](data/fusion_pipeline.csv)**),** int8_baseline, batch 32,
2-run confirmed (spread ≤0.06 ms):

| config | wall ms | GPU-busy | peak MiB | attn (flash) | qkv GEMM | transpose |
|---|--:|--:|--:|--:|--:|--:|
| fp16 attention | 49.89 | 49.04 | 4550 | 11.4 | — | — |
| §6 per-token flash | 54.10 | 53.51 | 3602 | 24.8 | — | — |
| **fused int8** | **52.73** | 52.0 | **3612 (−21%)** | 24.9 | 1.60 | 1.01 |
| **fused int4** | **52.41** | 51.5 | **3612 (−21%)** | 25.0 | 1.07 | 1.01 |

- **The fusion works:** it beats §6 by **−1.3 ms (int8) / −1.7 ms (int4)** by removing the fp16 round-trip, and
  the int8-output qkv GEMM (1.60/1.07 ms) is **~½ the fp16-output version (~3 ms)** — the write-halving landed;
  int4's qkv GEMM is fastest. It is the **best attention-quant config** and keeps the **−21% peak-memory** win.
- **But it does not beat fp16 e2e** (+2.8 ms): the **flash MMA + softmax (24.9 ms, unchanged) is the wall** and
  is slower than cuBLAS's fp16 SDPA (~22 ms). Exactly the predicted ceiling — the fusion attacks the qkv leg
  and the round-trip, not the attention compute. **Verdict: ship as the memory-optimal attention path
  (−21% peak, int8 quality-safe); it is not an e2e speedup on this cuBLAS-bound attention.**

---

## §4. Full pipeline (fresh, 6 modes)

Speed — GPU-busy (throttle-robust), this run's wall clean for every mode ([`data/pipeline_speed.csv`](data/pipeline_speed.csv)):

| mode | wall med | GPU-busy | vs fp16 | vs fp32 |
|---|--:|--:|--:|--:|
| fp32 | 102.67 | 101.77 | 0.54× | 1.00× |
| fp16 | 55.93 | 54.97 | 1.00× | 1.85× |
| int8 base | 50.08 | 49.19 | 1.12× | 2.07× |
| int8 modiff | 58.32 | 57.12 | 0.96× | 1.79× |
| **int4 base** | 48.13 | **47.25** | **1.16×** | **2.15×** |
| int4 modiff | 53.55 | 52.34 | 1.05× | 1.94× |

Total IO (analytical DRAM bytes/step; [`data/pipeline_io_analytic.csv`](data/pipeline_io_analytic.csv)):
fp16 7846 → int8 7362 (0.94×) → int4 7120 (0.91×) MiB. Peak memory
([`data/pipeline_io.csv`](data/pipeline_io.csv)): int4 base **4294 MiB** (lowest), fp16 4369.

**Why quantization's e2e win is modest:** only the **conv bucket (~25% of the step)** is quantized (13.7 → 9.7
int8 → 7.6 int4 ms); attention (softmax+SDPA ~22 ms) + qkv/proj GEMMs + GroupNorm run **fp16 in every mode**.
Amdahl-bounds the step to a ~1.33× ceiling even with free convs, and total IO is dominated by the
dtype-invariant fp16 attention-score traffic (~75%). Kernel profile: [`data/kernel_profile.csv`](data/kernel_profile.csv).

---

## Bottom line

- **Fastest mode:** `int4 base` **1.16× vs fp16 / 2.15× vs fp32** (int8 base 1.12× / 2.07×); lowest peak memory too.
- **§1 GEMM opt (shipped):** `gemm_wxax` beats fp16 on most qkv/proj shapes (int4 ≤2.13×).
- **§3 qkv→flash fusion (this session's main build):** correct (≤1 code), int8 quality-safe (rel 0.0097), the
  **best attention-quant config (−1.3/−1.7 ms vs §6, −21% peak)** — a **memory win, not an e2e speedup**,
  because the fp16 cuBLAS SDPA is faster than our flash MMA/softmax, which the fusion doesn't touch.
- **Recurring wall:** on this conv-dominated UNet with heavy cuBLAS-tuned attention, quantizing the non-conv
  ops (attention/linear) buys **memory/footprint, not step-time**. Beating fp16 e2e would require a
  faster-than-cuBLAS attention (flash SDPA), which remains the open problem.

*Scripts: `pipeline.py, kernel.py, io_analytic.py, linear_quant.py, attn_quant.py, quant_attn_profile.py,
fusion_kernel.py, fusion_quality.py, fusion_pipeline.py, mkplots.py`. Re-run with
`PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass` (ninja installed). Fusion opt-in:
`MODIFF_QUANT_ATTN=1 MODIFF_QKV_FLASH_FUSED=8|4`.*
