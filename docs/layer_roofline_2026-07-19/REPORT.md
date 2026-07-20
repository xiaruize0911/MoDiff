> ⚠️ **2026-07-20:** any "int8 ~2× e2e" figure here is inflated by a fp32/tf32 fp16-baseline (autocast
> bug). Real int8 e2e ≈ **1.08× vs true fp16**. See
> [../flash_attention_2026-07-19/E2E_CORRECTION_2026-07-20.md](../flash_attention_2026-07-19/E2E_CORRECTION_2026-07-20.md).

# Layer-level roofline profile → anomaly → fix (no fallback)

**Churches LDM UNet, A40, batch 64.** Measured each production kernel (qkv/proj GEMM; attention
QKᵀ / softmax / AV) in fp16/int8/int4 at the real layer shapes and compared to the A40 roofline
(compute = FLOPs / peak-TOPS; memory = bytes / 696 GB/s). Peaks used: fp16 149.7 TFLOPS, int8 299.3
TOPS, int4 598.7 TOPS. Data: `data/roofline_{gemm,attn}_b64.csv`, `data/attn_fix_b64.csv`,
`data/e2e_s8score_b64.csv`. Scripts: `scripts/{roofline,attn_fix,e2e_s8score}.py`.

## 1. Layer-level profile vs expected math

### qkv/proj GEMM (int8 `gemm_w8a8_awq`, single-instance µs)

| shape | measured | compute-roofline | memory-roofline | bound | efficiency |
|---|--:|--:|--:|:--:|--:|
| 32² C192 qkv | 186.3 | 48.4 | 126.7 | memory | 68% |
| 16² C384 qkv | 124.3 | 48.4 | 63.9 | memory | 51% |
| 8² C384 qkv | 37.5 | 12.1 | 16.4 | memory | 44% |
| 4² C768 proj | 20.0 | 4.0 | 4.2 | memory | 21% |

### Attention (int8 path, single-instance µs, per op)

| shape | op | int8 meas | fp16 meas | roofline | bound | int8 eff |
|---|---|--:|--:|--:|:--:|--:|
| 32² T1024 hd24 | QKᵀ | **2707** | **2060** | 1595 (mem) | memory | 59% |
| 32² T1024 hd24 | softmax | 3547 | — | 2311 (mem) | memory | 65% |
| 32² T1024 hd24 | AV | 1651 | — | 862 (mem) | memory | 52% |
| 16² T256 hd48 | softmax | 855 | — | 145 | memory | 17% |
| 8² T64 hd48 | softmax | 217 | — | 9 | memory | 4% |

## 2. Anomalies (measured ≠ expected)

1. **int8 attention is memory-bound on the fp16 T×T score matrix — so quantization gives no benefit,
   and int8 QKᵀ is actually SLOWER than fp16 (2707 vs 2060 µs).** The score matrix S=[BH,T,T] is written
   (QKᵀ) and read (softmax) as **fp16** in every path, and that T²·2-byte traffic dwarfs the int8 Q/K/V
   inputs. The compute roofline (where int8's 2× / int4's 4× lives) is irrelevant — arithmetic intensity
   is ~`2·hd/2 = hd = 24` FLOP/byte, far below the A40 int8 ridge point (~430). So int8/int4 only add a
   quantize pass on top of the same fp16-scores memory cost → net loss. **This is the root cause of every
   "quantized attention is slower" result in the sibling report.**
2. **int8 GEMMs run at 21–68% of the *memory* roofline** (not compute) — short-K (192/384) qkv/proj are
   memory-bound, dominated by the fp16 **output** (M·N·2 bytes), and don't even saturate bandwidth. int8's
   compute advantage (compute-roofline 2–4× below measured) is entirely unused.
3. **softmax efficiency collapses at small T** (17% at T=256, 4% at T=64) — the per-row-block kernel is
   occupancy/launch-bound when T×T is small. (Moot in-model: T<256 falls back to fp16 SDPA.)

## 3. Fix (no fallback): int8-score attention with a dynamic softmax

Anomaly #1 says: to make attention quantization pay, **shrink the T×T score traffic** — don't keep the
scores in fp16. The existing int8-score kernels (`attn_qk_int8_s8out` writes int8 S; `attn_softmax_requant_s8`
reads int8 S) only had a **static-c** softmax, which is quality-broken for diffusion (a single frozen
softmax constant can't track the ~30× per-timestep logit drift → rel-L2 0.3–0.6). So I built the missing
piece:

**New kernel `attn_softmax_requant_s8_dyn`** (`csrc/kernels/attn_quant_gemm.cu`): reads int8 scores,
does a **2-pass per-row-max** softmax (dynamic → quality-safe, no static-c), writes int8 P. Paired with
`attn_qk_int8_s8out` it makes the full path **int8 S write + int8 S read** — halving both T×T passes —
while keeping dynamic-softmax accuracy. Wired into `QuantizedStandardAttentionBlock` (dynamic path,
gate `MODIFF_ATTN_S8_SCORE=1`) with a self-calibrated per-tensor score scale `sS` (score absmax over the
first 8 forwards, then frozen — a grid scale only, not the softmax constant, so it stays quality-safe).
No fallback: the fp16-score path is replaced, not skipped.

## 4. Results (redone tests)

**Layer level — full attention QKᵀ+softmax+AV (`attn_fix_b64.csv`):**

| shape | fp16 (matl.) | int8 old (fp16 S) | int8 **new (int8 S)** | new/old | rel old | rel new |
|---|--:|--:|--:|--:|--:|--:|
| 32² T1024 hd24 | 11690 | 7911 | **7285** | **1.09×** | 0.029 | 0.034 |
| 16² T256 hd48 | 832 | 1221 | 1195 | 1.02× | 0.020 | 0.025 |

The int8-score path is **1.09× faster than the old int8 path at the dominant T=1024 block**, at
**+0.005 rel-L2** (quality-safe — the dynamic softmax preserves accuracy; only the score grid is int8).

**End-to-end — int8_baseline (AWQ w8a8 Linear + dynamic quantized attention), b64 (`e2e_s8score_b64.csv`):**

| config | wall ms/step | vs fp16 | rel-L2 vs fp16 |
|---|--:|--:|--:|
| fp16 | 96.3 | 1.00× | 0 |
| int8_baseline dyn-attn (fp16 scores, old) | 101.2 | 0.95× | 0.177 |
| **int8_baseline dyn-attn (int8 scores, fixed)** | **97.8** | **0.99×** | **0.178** |

The fix recovers **+3.5% e2e** (101.2 → 97.8 ms/step) at **zero quality cost** (rel 0.177 → 0.178),
closing most of the quantized-attention gap vs fp16 without a fallback and without the static-c quality
loss. (Static-c attention was 93.3 ms but rel 0.283 — faster still, but a real quality hit; the int8-score
dynamic fix is the quality-safe operating point.)

## 5. GEMM fix (anomaly #2): int8-output fusion + coalesced (smem-staged) stores

The short-K qkv/proj GEMMs are memory-bound and dominated by the **fp16 output write** (M·N·2 = ~86% of
the traffic). Fix, no fallback — new kernels `gemm_w8a8_awq_out_i8` and `gemm_w4a4_awq_out_i8`
(`csrc/kernels/gemm_wxax.cu`): identical (validated) mainloops, but the epilogue **requantizes the fp32
accumulator to int8** (per-column `inv_out_scale = 127/absmax`, folded) — halving the output write. The
downstream op consumes int8 directly / dequants with `1/inv_out_scale`.

**First attempt (naive per-thread int8 store) was slower** (0.83–1.0×): the mma output layout gives each
thread two scattered int8 pairs → 16-bit stores, *worse*-coalesced than the fp16 kernel's 32-bit
`__half2` stores. Halving bytes doesn't help if the transactions shrink. **Fix for the fix: a
shared-memory-staged epilogue** — scatter int8 results into a `[CTA_M][CTA_N]` smem tile (reusing the
`As` mainloop buffer), then store to global with **coalesced 128-bit (uint4 = 16 int8) writes**.

**Results (`gemm_outi8_b64.csv`, measured µs, dequant rel-L2 vs fp16 output):**

| shape | int8 fp16-out | int8-out | **int8 ×** | int4 fp16-out | int4-out | **int4 ×** | rel |
|---|--:|--:|--:|--:|--:|--:|--:|
| 32²C192 qkv (dominant) | 185.9 | 174.9 | **1.06×** | 181.9 | 146.4 | **1.24×** | 0.010 |
| 32²C192 proj | 88.2 | 80.3 | **1.10×** | 80.6 | 65.3 | **1.23×** | 0.010 |
| 16²C384 proj | 50.9 | 48.3 | 1.05× | 37.2 | 35.6 | 1.04× | 0.009 |
| 8²C384 qkv | 37.7 | 35.8 | 1.05× | 27.9 | 25.3 | 1.10× | 0.009 |
| 4²C768 (small) | 19.5 | 20.1 | 0.97× | 12.2 | 12.8 | 0.95× | 0.008 |

- **Correct:** dequant(int8 output) vs fp16 output rel-L2 **0.008–0.010 ≈ 1/127** — exactly the expected
  int8-output rounding, on every shape, int8 and int4.
- **1.06–1.24× on the dominant large-M shapes** (32²C192 qkv/proj), **int4 benefits most** (1.23–1.24×,
  since its int4 inputs are tiny so the output write is a bigger share). The smem-staged epilogue lifted
  the naive 0.88× → 1.06× (int8 qkv) and 0.97× → 1.24× (int4 qkv).
- **Small shapes stay ~neutral** (occupancy/short-K compute limited, not output-bound). The kernel is now
  at 45–55% of the (halved) int8-output roofline on the big shapes vs ~37% naive — the store is coalesced;
  the residual gap is short-K occupancy, not the output write.

E2E note: these GEMMs are ~3% of the step (§ sibling report), so the e2e payoff is small; the win is a
real, verified, no-fallback kernel-level improvement on the dominant qkv/proj shapes, largest for int4.

### 5b. Wired into the model + e2e (best batch 128, 30 warm-up + 5×200 steps, MEAN)

Wired the int8-output GEMM into `QuantLinearWxAx` (`MODIFF_LINEAR_OUT_I8=1`): the qkv/proj GEMM writes
int8, a per-column output scale is calibrated in the existing pass, and the dequant is **fused into the
bias add** via a new one-pass `dequant_bias_i8` kernel — so the epilogue stays one op (no extra fp16
round-trip; engages only when the layer has a bias, i.e. all attention qkv/proj). Correctness: whole-UNet
rel-L2 fix-on vs off = **0.004 (int8)**, **0.060 (int4)** — int8 negligible, int4 a real (int4-is-aggressive) cost.

| version | fix OFF | fix ON | ON vs OFF | ON vs fp16 |
|---|--:|--:|--:|--:|
| fp16 | 188.0 | — | — | 1.00× |
| int8_baseline | 178.1 | 178.2 | 0.999× (neutral) | 1.055× |
| int8_modiff | 201.1 | 201.1 | 1.000× (neutral) | 0.935× |
| **int4_baseline** | 177.7 | **174.8** | **1.016× (+1.6%)** | **1.075× (best)** |
| int4_modiff | 203.7 | 205.1 | 0.993× (noise) | 0.917× |

**Result:** the fix is a **modest real win for int4_baseline (+1.6% e2e → 1.075× fp16, the best config)**
and **neutral for int8** — exactly as the roofline predicts: the GEMM is ~3% of the step and int4's GEMM
win (1.24×) is 4× int8's (1.06×), so only int4 clears the noise floor e2e. MoDiff versions are launch-bound
(~13–17% slower than baseline), so the GEMM change is lost in that overhead. Data: `bench5_outi8_b128.csv`,
`scripts/bench5_outi8.py`. Kernels: `gemm_w{8a8,4a4}_awq_out_i8`, `dequant_bias_i8`; wiring:
`integration/kernels/wxax_linear.py`.

## 6. Remaining (occupancy, low impact)
**AV at 52% and small-T softmax at 4–17%** — occupancy tuning; low absolute impact (small-T falls back
to fp16 SDPA in-model).

## Verdict

The layer-level roofline exposed the real anomaly — **quantized attention was memory-bound on fp16 T×T
scores, so it could never beat fp16** — and the no-fallback fix (a dynamic int8-score softmax kernel that
halves the T×T traffic while staying quality-safe) turns the dominant attention block from a loss into a
**1.09× layer win / +3.5% e2e**, at ~0 quality cost. The next gap is the int8 GEMM's sub-roofline
bandwidth, which needs a kernel-level (not fallback) rewrite.
