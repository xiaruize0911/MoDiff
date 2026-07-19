# Fused int8/int4 attention applied to baseline & MoDiff — 5-version benchmark

**Churches LDM UNet, A40, batch 64, DDIM.** Wired the fused int8 flash-attention kernel
(`modiff_cutlass.flash_attn_int8`: QKᵀ / softmax / AV in one kernel, **online (running-register)
softmax** so no `[N,H,T,T]` score matrix is materialized and no static-c) into the shared attention
block `TokenMajorAttentionBlock` (`_attn_from_qkv`, gate `MODIFF_FLASH_ATTN`, `MODIFF_FLASH_MIN_T=512`
→ the dominant T=1024 block). Because the attention block is shared, the fused kernel applies to **both
baseline and MoDiff** automatically. Inputs are quantized by the fused `quantize_qkv_int8` (per-token
Q/K, per-channel V, head_dim padded to 32).

## The 5 versions (wall ms/step, min-of-6; × = speed vs fp16)

| version | attention | wall (flash OFF) | wall (flash ON) | flash Δ |
|---|---|--:|--:|--:|
| fp16 | fp16 SDPA | 96.1 (1.00×) | — | — |
| **int8_baseline** | int8 flash | **91.1 (1.05×)** | 100.0 (0.96×) | **+8.9 ms** |
| int8_modiff | int8 flash | 104.2 (0.92×) | 114.0 (0.84×) | +9.8 ms |
| **int4_baseline** | int8 flash | **92.3 (1.04×)** | 102.7 (0.94×) | +10.5 ms |
| int4_modiff | int8 flash | 115.3 (0.83×) | 129.9 (0.74×) | +14.7 ms |

(int4 versions use the **int8** flash kernel — there is no int4 flash kernel; attention precision is
decoupled from the int4 conv/linear. int4 scores are extremely lossy anyway.)

## Findings

1. **Correctness — the fused flash attention is quality-safe.** vs fp32 SDPA rel 0.007–0.025 (kernel
   test); vs fp16 SDPA the whole-UNet forward is rel-L2 ~0.074 (per-block ~0.01 at T=1024), well within
   an acceptable band. Online softmax tracks diffusion's per-timestep logit drift (no static-c problem).

2. **Speed — the fused attention is a consistent net LOSS (+9 to +15 ms/step).** The dispatched kernel is
   `flash_attn_int8_mma_kernel` (tensor-core, not the dp4a fallback), yet at **45.0 ms** it is *slower*
   than the fp16 SDPA it replaces (QKᵀ+softmax+AV ≈ 40.8 ms), and `quantize_qkv_int8` adds ~7 ms. Why:
   the churches attention has **tiny head_dim (hd=24)**, where flash tiling is inefficient and there is
   little T×T-traffic to save relative to the extra int8 quantize + the strength of cuBLAS fp16 bmm on
   A40. Flash fuses the bmm into itself (qkv/proj bucket 25→5 ms) but the fused kernel is bigger than the
   sum it replaces.

3. **Best config is int8/int4 baseline WITHOUT flash: 0.95×/0.96× fp16 (faster).** The quantization win
   is the conv (int8 conv 12.6 vs fp16 22.4 ms); attention should stay fp16 SDPA on this model.

4. **MoDiff caching costs ~13–24 ms/step over baseline** (int8: 104 vs 91; int4: 115 vs 92) — the
   a_hat/o_hat temporal-delta quantize + accumulate. For int4_modiff, GPU-busy (98–108) ≪ wall (115–130):
   it is launch-bound (many small MoDiff cache kernels).

## Verdict

The fused int8 attention kernel was **implemented, wired (baseline + MoDiff, int8 + int4), correct, and
measured** — but it does **not** speed up this UNet: fp16 SDPA-math is a strong baseline and the attention
here is small-head-dim. The e2e win remains **int8/int4 conv/linear quantization with fp16 SDPA attention
(baseline)**. A fused-attention win would need (a) a head_dim-24-tuned flash kernel, or (b) larger-head-dim
models. int4 attention would additionally need a native int4 flash kernel (does not exist).

## Combined config: AWQ w8a8 / modified w4a4 Linear + quantized W8A8/W4A4 attention

Per request, wired the **combined** config — AWQ `gemm_w8a8_awq` / modified `gemm_w4a4_awq` Linear
**and** quantized standard attention (`quantized_std_attention`: QKᵀ int-GEMM → requant softmax → AV
int-GEMM, materialized) in one model — via the new opt-in `MODIFF_QUANT_ATTN=1` (benchmark_ldm setup;
default off, so quant_lin keeps fp16 SDPA attention). 21/21 attention blocks convert to
`QuantizedStandardAttentionBlock`; their qkv/proj Linears are then the AWQ int8/int4 ports. Attention
quant is **STATIC by default** (`MODIFF_QUANT_ATTN_STATIC=1`: calibrated per-tensor Q/K, per-channel V,
single softmax-c, frozen after 8 forwards — consistent with the static conv/linear quant); `=0` → dynamic.
Script `scripts/bench_combined.py` → `fig_combined.png`, `data/combined_{speed,buckets}{,_dynamic}_b64.csv`.

Speed context: conv and Linear were already static; the attention is now static too. wall ms/step, b64:

| version | STATIC quant attn | dynamic quant attn | fp16 attn | rel-L2 vs fp16 (static / dynamic) |
|---|--:|--:|--:|--:|
| fp16 | 96.3 (1.00×) | 96.3 | 96.1 | 0 |
| int8_baseline | **93.3 (1.03×)** | 101.5 | 91.1 | 0.283 / 0.177 |
| int8_modiff | 106.5 (0.91×) | 114.1 | 104.2 | 0.35\* / 0.48\* |
| int4_baseline | **86.5 (1.11×)** | 97.7 | 92.3 | 0.306 / 0.208 |
| int4_modiff | 109.7 (0.88×) | 122.1 | 115.3 | 0.33\* / 0.29\* |

\* MoDiff single-forward rel-L2 is unreliable (stale a_hat/o_hat cache from a different warmup timestep).

**Findings:**
- **STATIC attention is ~8–11 ms/step faster than dynamic** (no runtime per-token/per-row absmax
  reductions), which *flips the speed verdict*: **int4_baseline static = 86.5 ms = 1.11× fp16 — the
  fastest config measured**, and int8_baseline static (93.3) also beats fp16. So with static quant, the
  int4 QKᵀ/AV (4-bit, packed) is genuinely faster than the fp16 SDPA bmm.
- **Quality cost: static-c softmax raises rel-L2 to 0.28–0.35 vs fp16** (from 0.18–0.21 dynamic) — a
  single frozen softmax-c can't track diffusion's ~30× per-timestep logit drift. This is the classic
  static-vs-dynamic trade (see the static-vs-dynamic study): static is faster, dynamic is more accurate.
  MoDiff is meant to compensate the static error across DDIM steps (needs a full-trajectory FID to confirm).
- **Net:** static quantized attention makes int4 (and marginally int8) a speed win, but at a real
  single-step quality cost. Whether it's worth it depends on the FID after MoDiff compensation — a
  trajectory-level quality eval is the open follow-up.

## Removed (2026-07-19)

Given the measured net loss, the flash-attention wiring (and the earlier materialized int8-score
experiment) was **removed from the live attention path** `TokenMajorAttentionBlock`
(`integration/fused_ops/token_major_attention.py`): the block is back to fp16 SDPA / materialized
attention + the int8 qkv/proj **Linear**-quantize fusions (which are correct wins and stay).

**Update 2026-07-19: flash fully DELETED from the code** (not just unwired). Removed `flash_attn_int8.cu`
from the build (`setup.py`), its pybind binding + api decl (+ the `mma_smoke` debug kernel it carried),
and deleted `integration/fused_ops/quantized_attention.py` and `integration/tests/test_flash_attn.py`.
Reason (measured, `scripts/flash_why` in the layer-roofline folder): flash_attn_int8 is slower than fp16
SDPA at **every** churches head-dim (0.73× at hd=24, 0.42× at hd=48, 0.34× at hd=64) and ~100× slower via
its naive fallback at hd=96 — a correctness-only v1, never perf-tuned. Quantized attention, when enabled
(`MODIFF_QUANT_ATTN=1`), now uses only the materialized **"our kernel"** path in
`quantized_std_attention.py` (`attn_qk_int8` / `attn_softmax_requant[_s8_dyn]` / `attn_av_int8`). Verified
post-removal: build clean, `flash_attn_int8` absent from the module, model runs (fp16 SDPA default and our
quantized attention both finite).

## Files

- `scripts/bench5.py` — 5-version benchmark + torch.profiler bucket profile (`BENCH5_NOFLASH=1` for the
  flash-OFF reference pass). `scripts/mkplot.py` → `fig_bench5.png`.
- `data/bench5_speed{,_noflash}_b64.csv`, `bench5_buckets{,_noflash}_b64.csv`,
  `bench5_topkernels{,_noflash}_b64.csv`.
- Wiring: `integration/fused_ops/token_major_attention.py` (`_attn_from_qkv`, `_flash*` flags); kernel:
  `csrc/kernels/flash_attn_int8.cu`; correctness gate: `integration/tests/test_flash_attn.py`.
