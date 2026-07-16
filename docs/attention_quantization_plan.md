# Plan: Quantize the attention path (int8 + int4, baseline + MoDiff)

## RESULTS (2026-07-16) — implemented & validated

**Shipped:** a fused, tensor-core **int8 flash-attention** score path (`mma.m16n8k32.s8`,
no T×T materialization) — `csrc/kernels/{mma_int8.cuh,flash_attn_int8.cu,quantize_qkv.cu}`,
`integration/fused_ops/quantized_attention.py`, opt-in via `MODIFF_QUANT_ATTN=1`.
Correctness gate `integration/tests/test_flash_attn.py` ALL PASS.

**Validated in the real `int8_baseline` mode (batch 32):**
| metric | fp16 attn | int8 quant attn |
|---|--:|--:|
| peak memory | 4549 MiB | **3600 MiB (−21%)** |
| speed | 45.9 ms/step | 52.7 ms/step (0.87×) |
| latent rel-err | — | 0.0038 |

- **The win is memory/IO (−21% peak), not speed.** The fused kernel avoids the
  `[N,heads,T,T]` fp16 score matrix. Speed plateaus at ~0.87× because PyTorch math SDPA
  is cuBLAS-backed and hard to beat at these small shapes even though it materializes T×T.
  Perf trajectory: naive dp4a 0.11× → tensor-core mma 0.53× → multi-warp 0.82× → fused
  packed quantize 0.89× (fp16-mode e2e). The T=1024 score kernel matches fp16 SDPA (2.2 vs
  2.0 ms); smaller-T blocks stay ~2× (overhead-bound).

**int4 findings (why not pursued further):**
- **int4 scores:** ~0 additional peak-memory benefit — the −21% comes from T×T avoidance,
  which is mode-independent (int8 already banks it). int4 only shrinks the small Q/K/V int8
  buffers (~24→12 MB vs a 3600 MB peak). Not worth the separate `m16n8k64.s4` kernel + quality risk.
- **int4 proj weights (naive per-channel):** **quality-breaking** — e2e latent rel-err jumps
  0.008 → **0.155**. Viable only with AWQ group-wise + SmoothQuant, a large effort for a tiny
  IO target (proj weights). Available default-off via `MODIFF_ATTN_PROJ_BITS=4` for experiments.

**Bottom line:** int8 fused attention is the deliverable — a −21%-peak-memory, quality-neutral
capability. int4 (scores or proj) is low-value at these diffusion-attention shapes.

---

## Original plan (below)


Status: design proposal (2026-07-15). Target: churches LSUN LDM UNet, A40 sm_86.
Goal: extend int8/int4 quantization into the attention block for **all four** compute modes —
`int8_baseline`, `int4_baseline`, `int8` (MoDiff), `int4` (MoDiff) — so attention stops being the
dtype-invariant fp16 wall that bounds both speed (§4) and DRAM IO (§5) in the comprehensive benchmark.

---

## 0. Why (motivation, from the benchmark)

- Attention is ~42% of the fp16 step (~23 ms) and **dtype-invariant today** — quantization only touches the
  conv bucket, so speed is capped at a ~1.32× Amdahl ceiling and total DRAM IO barely moves (int8 0.94×,
  int4 0.91× of fp16).
- The single dominant item is the **QKᵀ/AV score path**: on the C192/T1024 block the math-SDPA score
  matrix is ~512 MiB (fp16) and 4060 µs. That is ~75% of attention DRAM IO and the biggest kernel in the UNet.
- Therefore the high-value target is the **score matmul**, not the qkv/proj projections. Quantizing the
  projections alone (which is all the existing `*_attn_modiff` path does) moves <10% of attention IO.

## 1. Current state (what exists, what's missing)

| Piece | Exists? | Where | Gap |
|---|---|---|---|
| qkv/proj int8 + MoDiff delta | ✅ | `integration/kernels/modiff_attention.py` (`convert_attention_to_modiff`) | Only in `*_attn_modiff` modes; **channel-major** (bypasses token-major + fused GN→qkv); int4 only via slow fallback; not in baseline or the plain int8/int4 modes |
| Fused GN→qkv (fp16) | ✅ | `csrc/kernels/fused_gn_qkv.cu`, `TokenMajorAttentionBlock` | fp16 only — does not emit int8/int4 Q/K/V |
| QKᵀ/AV score-matmul quant | ❌ | — | Never done; scores are fp16 in every mode |
| int8/int4 CUTLASS GEMM primitives | ✅ | `conv2d_int8/int4.cu`, `w8a8` mma in AWQ | Built for conv (1×1 = Linear); no fused attention |
| MoDiff delta-quantize kernels | ✅ | `csrc/kernels/modiff_delta_quantize.cu` (`step1_*`, `sub_absmax_scale`, `*_o_hat`) | Linear-only accumulation identity (see §3.3) |
| Calibration infra | ✅ | `integration/calibration/`, `apply_*_static_scales` | No attention-specific scales (Q/K/V/P) |

**Key constraint from evidence:** int8 W8A8 Linear is **2–3× slower than fp16 cuBLAS** at the projection
shapes (K = 192/384/768, below the int8 crossover ~2048 — report §1, and `int8_linear.py` gates K≥2048). So
quantizing qkv/proj for *speed* will regress; their only win is *IO* (weight-only int4, AWQ-style). The score
matmul is the piece that pays off on both axes — **but only if fused (flash-style) so the T×T matrix is never
materialized.** Quantization on top of fusion then halves the operand bytes.

## 2. Scope & sub-targets

Three independently-gated layers (each an env kill-switch + calibration entry):

- **A — Projection quant** (`MODIFF_QUANT_ATTN_PROJ`): qkv (C→3C) and proj (C→C).
- **B — Score-matmul quant** (`MODIFF_QUANT_ATTN_SCORE`): QKᵀ and AV, ideally as one fused flash-int8 kernel.
- **C — MoDiff temporal delta** on attention (auto-on in the `int8`/`int4` MoDiff modes), applied only where
  the accumulation identity is valid (§3.3).

Mode matrix:

| mode | proj (A) | score (B) | MoDiff delta (C) |
|---|---|---|---|
| int8_baseline | int8 W8A8 (or fp16 if slower) | int8 fused attn | no (dynamic/static quant each step) |
| int4_baseline | int4 weight-only (AWQ) | int8 fused attn (int4 scores experimental) | no |
| int8 (modiff) | int8 + MoDiff delta on qkv/proj | int8 fused attn, static-calibrated | yes on qkv/proj only |
| int4 (modiff) | int4 + MoDiff delta on qkv/proj | int8 fused attn | yes on qkv/proj only |

Note: score matmul uses **int8 even in the int4 modes** by default — int4 softmax scores are too aggressive
(accuracy). int4 on scores is an optional experimental flag, validated by FID before adoption.

## 3. Design

### 3.1 Layer A — projection quantization (qkv, proj)

qkv/proj are 1×1 convs ≡ Linear, K = C. Two paths:

- **int8 modes:** reuse the existing `OptimizedInt8Conv2d` path (already used by MoDiff attn), but keep it in
  the **token-major** block. Because W8A8 is slower than cuBLAS at these K, gate it: only emit int8 if a
  per-shape micro-benchmark says it wins; otherwise keep fp16 qkv/proj and rely on Layer B for the win. The IO
  metric still credits the int8 operand reduction.
- **int4 modes:** weight-only **W4A16 (AWQ)** for the projection weights — activation stays fp16, weights int4,
  dequant-in-GEMM. Reference: `/workspace/llm-awq` `gemm_forward_cuda_new` + `pack_intweight` + the
  `dequantize_s4_to_fp16x2` PTX. **Caveat (must resolve):** AWQ's new kernel hard-codes group size 128 and
  needs `OC%128==0`, `K%64==0`; K=192 fails %64-group divisibility and OC=192/576 partially fail %128. Options:
  (a) use AWQ's *legacy* g64 kernel (allows G=64, tiles K by 32 → K=192 ok), (b) per-tensor int4 weight scale
  (no groups), or (c) a bespoke small-shape tile. Recommend (a)/(b) first; measure.

**Fusion:** extend `fused_gn_qkv.cu` with an int8-output variant `fused_gn_qkv_int8` that folds the qkv weight
quant + per-token activation quant into the CUTLASS epilogue, emitting **int8 Q/K/V + per-token scales**
directly (no fp16 qkv write). This feeds Layer B with zero intermediate fp16 traffic — the cleanest fusion
win. Keeps the per-sample GN scale iterator and SHIFT/ReLU-absorption trick; the epilogue changes from
fp16-store to quantize-store (reuse `scale_quantize_int8` logic in-epilogue). Tile constraint T%128 still
applies (C192/T1024, C384/T256 qualify; smaller blocks use a non-fused GN→int8 path).

### 3.2 Layer B — score-matmul quantization (the payoff), fused flash-int8

**This is the core new kernel.** A custom fused, flash-style, int8 attention kernel (SageAttention/
FlashAttention-int8 pattern), replacing the math-SDPA call in `TokenMajorAttentionBlock`:

```
per (batch, head), tile over T:
  load Q_i (int8, per-row scale sq_i)
  for each K/V tile j:
    S_ij = Q_i · K_jᵀ        # int8 mma m16n8k32.s8.s8.s32 -> int32
    S_ij = S_ij * (sq_i * sk_j)          # dequant to fp32
    online softmax (running max m_i, sum l_i, fp32)   # numerically safe path stays fp32
    P_ij = quantize(exp(S_ij - m_i))     # per-row P scale sp_i ; P in [0,1] -> int8 is ample
    O_i += P_ij · V_j        # int8 mma -> int32
  O_i = O_i * (sp_i * sv) / l_i          # final dequant + normalize -> fp16
```

Wins: (1) **T×T scores never materialized** — this alone recovers most of the IO/latency the math backend
lost vs flash; (2) **int8 operands** halve Q/K/V/P bytes and use the 2× int8 tensor-core rate. This is why the
plan targets a *fused quantized* kernel rather than replacing the two discrete cuBLAS GEMMs (which would keep
the T×T materialization and, at hd=24/48, likely lose to cuBLAS).

**Building blocks / AWQ reference:** the int8 tensor-core mma (`mma.sync.m16n8k32.s8.s8.s32`) and per-token
activation-quant kernel come straight from AWQ's QServe-derived W8A8 path
(`/workspace/llm-awq/awq/kernels/csrc/w8a8/w8a8_gemm_cuda.cu`, `quantization.cu`) and its `cp.async`
pipelining. AWQ has **no** attention kernel, so the flash tiling + online softmax + P-requant is new code
(reference: SageAttention). We can prototype the tiling in Triton first (fast to iterate), then port the hot
path to CUTLASS/CUDA if Triton can't beat fused fp16 flash (recall the Triton GN→qkv dead-end: Triton lost to
cuBLAS at small K — so budget for a CUDA/CUTLASS final kernel).

**Small-K caveat (QKᵀ):** the QKᵀ contraction dim is head_dim = 24 / 48 / 96. int8 mma needs K a multiple of
32 → hd=24 pads to 32 (33% waste), hd=48 pads to 64, hd=96 is clean (3×32). AV contracts over T (1024/256/64
→ clean). So expect the int8 QKᵀ to underperform its theoretical 2× at hd=24/48; the fusion (no T×T write) is
what guarantees the net win there. Provide a per-shape fp16-flash fallback and only enable int8-fused where it
measurably wins.

**Scale granularity:** per-row (per-token) scales for Q and P, per-tensor or per-head for K and V (start
per-tensor, refine to per-head if FID needs it). Softmax max-subtraction and the exp/sum stay fp32 for
stability regardless of int8 operands.

### 3.3 Layer C — MoDiff temporal delta on attention

MoDiff's delta identity `o_hat_t = A(Q(a_t − a_hat_{t+1})) + o_hat_{t+1}` is only valid when **A is linear**.
qkv/proj are linear → MoDiff delta applies (reuse `step1_static_quantize_fprop` + `conv2d_int8/4_fprop_o_hat`
exactly as `modiff_attention.py` does). The **score matmul is bilinear (both Q and K move) and sits under a
nonlinear softmax → no clean linear accumulation**, so MoDiff delta does **not** apply to QKᵀ/AV. In the
MoDiff modes:

- qkv/proj: MoDiff delta-quantized (int8/int4), producing fp16 Q/K/V (`o_hat`).
- score matmul: those fp16 Q/K/V are freshly (statically-calibrated) quantized to int8 each step and fed to
  the fused flash-int8 kernel — **no delta caching on scores.**

This matches the existing design (MoDiff attn only ever quantized the linear projections) and is the
mathematically honest boundary. Document it explicitly so "int8 modiff" isn't mistaken for delta-caching the
whole attention.

### 3.4 Wiring & unification

- Replace the two divergent paths (`TokenMajorAttentionBlock` fp16 vs `convert_attention_to_modiff`
  channel-major) with **one** `QuantizedTokenMajorAttentionBlock` parameterized by `proj_bits ∈ {16,8,4}`,
  `score_bits ∈ {16,8}`, `modiff ∈ {False,True}`. Keeps token-major layout + fused GN→(int8)qkv for all modes;
  removes the channel-major duplication.
- `_setup_model` (`benchmark_ldm.py`): route `int8_baseline`→(8,8,F), `int4_baseline`→(4,8,F),
  `int8`→(8,8,T), `int4`→(4,8,T). Preserve `MODIFF_DISABLE_TOKEN_MAJOR_ATTN` and add
  `MODIFF_QUANT_ATTN_PROJ`, `MODIFF_QUANT_ATTN_SCORE` kill-switches (default on for the quant modes).
- **Calibration:** add attention scales to the `.pt` files — per-layer qkv/proj activation scales (already
  produced by the conv/linear calibrators once qkv/proj are quantized modules) plus Q/K/V/P score scales under
  an `attn:` key prefix, gathered in the existing `_calibrate_int8/_calibrate_int4` short-DDIM calibration
  loop. Reset MoDiff attn state between calibration samples (existing `reset_attention_modiff`).
- **CUDA-graph gotcha:** if graph-capturing the attention hot path (as MoDiff conv does), `reset_cache()` must
  destroy the graph — stale baked-in tensor addresses cause illegal-memory-access on the next DDIM sample.

## 4. Kernel fusion design (summary)

Three fused kernels, in priority order:

1. **`fused_flash_attn_int8`** (Layer B, highest value): GN-normalized int8 Q/K/V → flash-tiled int8 QKᵀ →
   fp32 online softmax → int8 P → int8 AV → fp16 O. One kernel, no T×T materialization. Prototype in Triton,
   final in CUDA/CUTLASS.
2. **`fused_gn_qkv_int8`** (Layer A fusion): extend the existing per-sample GN→qkv CUTLASS kernel to
   quantize-store int8 Q/K/V + per-token scales in the epilogue (no fp16 qkv write). Feeds kernel 1 directly.
3. **qkv/proj weight-quant GEMM** (Layer A): int8 (reuse conv path) or int4 W4A16 (AWQ legacy-g64 / per-tensor
   adaptation). Only the non-fused (T%128≠0) blocks need this standalone; the fused blocks fold it into (2).

End-to-end fused pipeline for a qualifying block (C192/T1024): `raw x → [fused_gn_qkv_int8] → int8 Q/K/V →
[fused_flash_attn_int8] → fp16 O → proj → residual`. Only two custom kernels, no fp16 score matrix, no
intermediate fp16 qkv.

## 5. Testing plan

Extend `integration/tests/test_kernel_correctness.py` (the existing 12-test gate, `UPDATE_GOLDEN` flow, rel-err
reporting). New gates, each vs an fp32 reference and a golden:

1. `attn_qkv_int8` / `attn_qkv_int4` — fused GN→int8/int4 qkv vs `GroupNorm→Linear` fp32. Target rel err:
   int8 ≲ 0.02, int4 ≲ 0.15 (in line with existing conv gates 0.012 / 0.22).
2. `attn_score_int8` — `fused_flash_attn_int8` vs fp32 `F.scaled_dot_product_attention`, **all five shapes**
   (C192/T1024, C384/T256, C384/T64, C768/T16, C768/T4). Target rel err ≲ 0.03. Include a causal=False,
   scale-correctness check and a numerical-stability case (large logits).
3. `attn_score_int4` (experimental, non-blocking) — same, target ≲ 0.1; report only.
4. `attn_modiff_lifecycle` — build / stable-across-steps / no-cache-growth / reset / reboot, mirroring the
   existing `int8_modiff_conv` lifecycle test; assert `reset` destroys any CUDA graph.
5. `attn_block_e2e` — full `QuantizedTokenMajorAttentionBlock` (proj+score+residual) vs fp32 block, per mode.
   Target end-to-end block rel err ≲ 0.02 (int8), matching the fused-GN→qkv 0.0016 bar where possible.

Correctness harness must run green before any perf claim: `python integration/tests/test_kernel_correctness.py`
→ ALL PASS (now 12 → ~17 tests).

## 6. Validation plan

1. **End-to-end numerical:** UNet output rel err vs fp16 for each of the 4 modes (existing A/B harness,
   `integration/benchmarks/ab_benchmark.py`), gate ≲ 0.01–0.02.
2. **Accuracy (the real gate for quantization):** FID on LSUN churches via
   `integration/benchmarks/eval_fid_lsun.py` for fp16, int8_baseline, int4_baseline, int8, int4 — **with and
   without** attention quantization — reporting ΔFID. Attention int8 should be near-neutral; int4 scores (if
   enabled) must clear a ΔFID budget or stay disabled. This directly answers "does quantizing attention hurt
   quality."
3. **Speed + IO re-benchmark:** re-run `docs/comprehensive_benchmark_2026-07-15/scripts/{pipeline,kernel,
   io_analytic}.py`. Expected deltas to verify:
   - attention (softmax+SDPA) bucket **drops** (fused, no T×T) — this is the headline;
   - conv-vs-attention Amdahl ceiling **rises above 1.32×** (attention no longer fully dtype-invariant);
   - analytical total IO: the 5867 MiB fp16 attention term **falls** (int8 scores and/or fused-away),
     finally moving int8/int4 total IO below fp16 — the outcome you flagged.
4. **Ablations:** proj-only vs score-only vs both; fused-flash-int8 vs discrete-int8-GEMM (to justify the fused
   kernel); int8-fused-attn vs revert-to-fp16-flash (show the quantized-fused path wins on speed **and** IO).
5. Update `REPORT.md` §1/§3/§4/§5 and `io_analytic.py` (attention operands now dtype-dependent) with the new
   numbers.

## 7. Risks & fallbacks

- **Small-K QKᵀ (hd=24/48)** underperforms int8 → keep a per-shape fp16-flash fallback; enable int8-fused only
  where it wins. Non-negotiable: never ship a slower default.
- **int8 qkv/proj slower than cuBLAS** at these K → gate by shape; treat projection quant as an IO play, lean
  on Layer B for latency.
- **int4 score accuracy** → default score_bits=8; int4 scores gated behind an FID check.
- **Triton loses at small K** (proven by the GN→qkv dead-end) → budget for a CUDA/CUTLASS final kernel; use
  Triton only for prototyping the flash tiling logic.
- **MoDiff on attention is accuracy-only** and MoDiff has diverged on CNNs before → FID-gate the MoDiff modes;
  keep baseline modes as the safe default.
- **CUDA-graph reset** must destroy graphs on `reset_cache` (illegal-memory-access otherwise).
- **Calibration drift:** attention score scales are activation-dependent across timesteps → validate static
  scales hold across the full DDIM schedule, else fall back to dynamic per-step scales for the score kernel.

## 8. Phasing (each phase independently landable & validated)

- **P0 — scaffolding:** unify into `QuantizedTokenMajorAttentionBlock` (fp16 behavior unchanged), add flags +
  kill-switches + mode routing. Gate: existing 12 tests still pass, e2e rel err unchanged.
- **P1 — Layer A int8 projections** (correctness-first, discrete GEMM): reuse conv int8 path in token-major +
  MoDiff delta for the modiff modes. Gate: `attn_qkv_int8`, `attn_modiff_lifecycle`, FID neutral.
- **P2 — Layer B discrete int8 QKᵀ/AV** (still materializes T×T): correctness scaffold for the score quant +
  calibration of Q/K/V/P scales. Gate: `attn_score_int8` rel err, FID neutral. (May be slower — that's fine;
  it's the correctness stepping stone.)
- **P3 — Layer B fused flash-int8 kernel:** the real perf/IO win. Gate: `attn_score_int8` still green, speed &
  IO re-benchmark shows attention bucket + IO drop, net e2e speedup up.
- **P4 — fused GN→int8-qkv** (kernel 2): fold projection quant into the CUTLASS GN→qkv epilogue for the
  T%128 blocks. Gate: e2e rel err, further IO drop.
- **P5 — int4 projections (AWQ W4A16)** + optional int4 scores (experimental, FID-gated).

Recommended stop-and-review after **P3** — that delivers the headline result (attention IO/latency drops,
int8/int4 total finally beats fp16) with the least risk; P4/P5 are incremental polish.

## 9. Open decisions for you

1. **Aggressiveness of Layer B:** go straight for the fused flash-int8 kernel (P3, bigger CUDA effort, real
   win), or ship the discrete-GEMM scaffold (P2) first? (I recommend P2→P3 staged.)
2. **int4 scores:** attempt at all, or keep scores int8 in every mode and use int4 only for projection weights?
   (I recommend int8 scores everywhere; int4 scores experimental only.)
3. **Prototype language for the flash kernel:** Triton-first for iteration (risk: loses at small K), or commit
   to CUDA/CUTLASS from the start? (I recommend Triton prototype for the softmax/tiling logic, CUDA for ship.)
4. **MoDiff-on-attention scope:** confirm MoDiff delta stays on qkv/proj only (score matmul uses plain static
   quant), per §3.3 — or do you want to explore a delta scheme for Q/K/V feeding the score kernel?
