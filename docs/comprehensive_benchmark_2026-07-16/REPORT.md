# MoDiff kernel report (2026-07-16) — the flash-SDPA breakthrough + kernel session

LSUN-churches latent-diffusion UNet, **A40** (sm_86; fp16 149.7 TFLOP/s, int8 299 TOP/s, int4 599 TOP/s,
DRAM 696 GB/s), batch 32, DDIM, per-step. Data in [`data/`](data/), scripts in [`scripts/`](scripts/).

> ### ⚡ Headline: use PyTorch **flash SDPA**, not the forced math backend
> The pipeline had been **forcing the math SDPA backend** so QKᵀ/AV stayed interceptable cuBLAS GEMMs for
> attention-score quantization. That made attention **~9× slower than PyTorch's flash SDPA** and dominated the
> step. Switching the fp16 attention to **flash SDPA** (`MODIFF_SDPA_MATH=1` to revert) gives, quality-identical
> (latent rel-err **0.0022**), applied to **every mode**:
> - **fp16: 55.9 → 32.9 ms/step (1.70×)**, peak **4369 → 3407 MiB (−22%)**.
> - **`int4 base`: 48.1 → 25.7 ms (1.87×)**; **1.35× vs fp16-flash, 2.34× vs the originally-shipped fp16-math**.
> - It **obsoletes the whole attention-quantization direction** (§6 int8-flash, §3 qkv-fusion): flash SDPA is
>   ~2× faster than our best int8 attention path *and* matches its −21% memory (see §3).
>
> Noise control: ≥7 s warmup, 12 timed runs + GPU-busy, headline confirmed across re-launches (stdev ≤0.16 ms).

---

## §1. The SDPA-backend finding (the real attention lever)

Per-call microbench of the dominant attention block (C192, T=1024, nh=8, hd=24; batch 32;
[`scripts/fusion_kernel.py`](scripts/fusion_kernel.py) sibling probe):

| SDPA backend | µs | vs math |
|---|--:|--:|
| math (was forced) | 4069 | 1.0× |
| mem-efficient | 957 | 4.3× |
| **flash** | **467** | **8.7×** |
| our int8 flash (§6) | 4429 | 0.9× |

The math backend materializes the `[N,H,T,T]` score matrix (~512 MB on this block) — that was the "single
biggest kernel." Flash SDPA tiles it. Our hand-written int8 flash is a naive kernel, **~10× slower than flash
SDPA** and not competitive. e2e (fp16, batch 32, 2-run confirmed): math 55.7 ms / 4369 MiB → **flash 32.5 ms /
3408 MiB**, latent rel-err **0.0022**. `token_major_attention.py` now defaults to flash (math via `MODIFF_SDPA_MATH=1`).

## §2. Full pipeline — flash SDPA default (6 modes)

Speed ([`data/pipeline_speed.csv`](data/pipeline_speed.csv), GPU-busy throttle-robust; wall clean, stdev ≤0.22):

| mode | wall | GPU-busy | vs fp16 | vs fp32 | peak MiB |
|---|--:|--:|--:|--:|--:|
| fp32 | 88.06 | 87.24 | 0.36× | 1.00× | 2920 |
| fp16 | 32.93 | 31.59 | 1.00× | 2.76× | 3407 |
| int8 base | 27.05 | 25.59 | 1.23× | 3.41× | 3602 |
| int8 modiff | 35.79 | 33.28 | 0.95× | 2.62× | 4060 |
| **int4 base** | 25.72 | **23.44** | **1.35×** | **3.72×** | **3362** |
| int4 modiff | 32.97 | 28.59 | 1.11× | 3.05× | 3797 |

- **`int4 base` = 1.35× vs fp16 / 3.72× vs fp32** (and **2.34× vs the originally-shipped fp16-math 54.97**).
- **Conv quantization's e2e win finally shows** (1.16×→1.34× for int4): with flash SDPA the attention bucket
  collapses, so the quantized-conv bucket is now the dominant lever instead of being drowned by attention.

Kernel profile shift ([`data/kernel_profile.csv`](data/kernel_profile.csv), fp16 ms/step): attention (softmax
+SDPA) **11.4 → 3.43**, GEMM (qkv/proj + QKᵀ·AV) **13.4 → 1.56** — the **conv bucket (13.55) is now the top
cost**. Total analytical IO ([`data/pipeline_io_analytic.csv`](data/pipeline_io_analytic.csv)): attention
collapses 5867 → **406 MiB** (no T×T), so total drops fp16 2385 → int4 **1659 MiB (0.70×)** — quantization's
IO benefit now shows (was 0.91× when the fp16 T×T dominated).

## §3. Obsoleted: attention-score quantization (§6 int8-flash, qkv→flash fusion)

This session built a correct fused **qkv-int-output → int8-flash** path (int8 & int4): `gemm_w8a8_out_int8` /
`gemm_w4a4_out_int8` (int8 output = `round(acc·a_scale·w_scale[c]·oscale[c]+bias)`, ≤1 code) + `transpose_qkv_int8`
(exact) + calibrated static scales; quality-safe (fused int8 latent rel-err **0.0097**). Under the old *math*
default it was the best attention-quant config (−1.3/−1.7 ms vs §6, −21% peak). **But flash SDPA obsoletes it**
([`data/fusion_pipeline.csv`](data/fusion_pipeline.csv), int8_baseline mode):

| attention path | wall ms | peak MiB |
|---|--:|--:|
| **fp16, flash SDPA** | **26.6** | 3604 |
| §6 per-token int8 flash | 51.8 | 3603 |
| fused int8 (W8A8→flash) | 50.5 | 3612 |
| fused int4 (W4A4→flash) | 50.0 | 3612 |

Flash SDPA is **~2× faster** than our best int8 attention path (our int8 flash kernel is ~10× slower than
flash SDPA, §1) **and matches its peak memory** (flash already avoids the T×T matrix). So the fused kernels are
kept in-tree, correct and opt-in (`MODIFF_QKV_FLASH_FUSED=8|4`), but are **not** on any recommended path — the
attention lever is the backend, not quantization. ![obsoleted](08_qkv_flash_fusion.png)

## §4. What still helps

- **Conv int8/int4 quantization** — the real e2e win now that attention is cheap: `int4 base` 1.34×, `int8 base`
  1.23× vs fp16 (§2). int8 quality-safe; int4 base FID-acceptable, int4-conv rel ~0.22 (see the linear-quant doc).
- **`gemm_wxax` `half2`-epilogue** (this session): the W8A8/W4A4 qkv/proj GEMM beats fp16 on 5/6 shapes (int4
  ≤2.13×) — [`data/gemm_wxax_shapes.csv`](data/gemm_wxax_shapes.csv). Relevant for the (opt-in) §7 Linear quant.

## §5. int8-conv-output → GroupNorm (validated, deferred)

Quality is ~free (fake-quant rel-err 0.0023–0.0033) and the int8-in `group_norm_silu_dequant_quantize_nhwc`
kernel is correct (≤1 code) + ~1.05×, but realizing the conv-side write saving needs a direct-int8-output
CUTLASS conv epilogue (the `relu_requant` path writes fp16 scratch first). Projected e2e ~2%; deferred.

---

## Bottom line

1. **Biggest win by far: flash SDPA.** Not forcing the math backend gives fp16 **1.71× + −22% peak**,
   quality-identical, on every mode — dwarfing every quantization effort. `int4 base` is now **1.35× vs fp16 /
   3.72× vs fp32 / 2.34× vs the originally-shipped fp16-math**.
2. **Attention-score quantization is obsolete.** Flash SDPA beats our int8 flash by ~2× on speed and ties on
   memory; the entire premise (force math SDPA so QKᵀ/AV are quantizable) was net-negative. Kept opt-in only.
3. **Conv quantization is the remaining real lever** (int4 1.34×), now visible because attention is cheap.
4. Lesson: the attention bottleneck was a **software-config artifact** (forced math SDPA), not a kernel gap —
   the fix was one backend switch, not a faster-than-cuBLAS kernel (which we also confirmed we can't write:
   our int8 flash is 10× off PyTorch's flash SDPA).

*Fusion opt-in: `MODIFF_QUANT_ATTN=1 MODIFF_QKV_FLASH_FUSED=8|4`. Revert to math SDPA: `MODIFF_SDPA_MATH=1`.
Env: `PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass`, ninja installed.*
