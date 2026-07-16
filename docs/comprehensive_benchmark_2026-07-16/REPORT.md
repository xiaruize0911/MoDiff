# MoDiff kernel report (2026-07-16) — the flash-SDPA breakthrough + kernel session

LSUN-churches latent-diffusion UNet, **A40** (sm_86; fp16 149.7 TFLOP/s, int8 299 TOP/s, int4 599 TOP/s,
DRAM 696 GB/s), batch 32, DDIM, per-step. Data in [`data/`](data/), scripts in [`scripts/`](scripts/).

> ### ⚡ Headline: use PyTorch **flash SDPA**, not the forced math backend
> The pipeline had been **forcing the math SDPA backend** so QKᵀ/AV stayed interceptable cuBLAS GEMMs for
> attention-score quantization. That made attention **~9× slower than PyTorch's flash SDPA** and dominated the
> step. Switching the fp16 attention to **flash SDPA** (`MODIFF_SDPA_MATH=1` to revert) gives, quality-identical
> (latent rel-err **0.0022**), applied to **every mode**:
> - **fp16: 55.9 → 32.9 ms/step (1.70×)**, peak **4369 → 3407 MiB (−22%)**.
> - **`int4 base`: 48.1 → 25.7 ms (1.87×)**; **1.34× vs fp16-flash, 2.34× vs the originally-shipped fp16-math**.
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
SDPA** and not competitive. e2e (fp16, batch 32): math 55.9 ms / 4369 MiB → **flash 33.0 ms / 3407 MiB**,
latent rel-err **0.0022**. `token_major_attention.py` now defaults to flash (math via `MODIFF_SDPA_MATH=1`).

## §2. Full pipeline — flash SDPA default (6 modes)

Speed ([`data/pipeline_speed.csv`](data/pipeline_speed.csv), GPU-busy throttle-robust; wall clean, stdev ≤0.22).
![pipeline speed](01_pipeline_speed.png)

| mode | wall | GPU-busy | vs fp16 | vs fp32 | peak MiB |
|---|--:|--:|--:|--:|--:|
| fp32 | 88.25 | 87.34 | 0.36× | 1.00× | 2920 |
| fp16 | 33.04 | 31.58 | 1.00× | 2.77× | 3407 |
| int8 base | 27.05 | 25.62 | 1.23× | 3.41× | 3602 |
| int8 modiff | 35.76 | 33.30 | 0.95× | 2.62× | 4060 |
| **int4 base** | 25.70 | **23.48** | **1.34×** | **3.72×** | **3365** |
| int4 modiff | 33.33† | 28.59 | 1.10× | 3.05× | 3800 |

†`int4 modiff` wall was clock-throttled this run (stdev 2.7 ms, median 35.18); min 33.33 and GPU-busy 28.59
are the reliable values. Speedups use GPU-busy (throttle-robust).

- **`int4 base` = 1.34× vs fp16 / 3.72× vs fp32** (and **2.34× vs the originally-shipped fp16-math 54.97**).
- **Conv quantization's e2e win finally shows** (1.16×→1.34× for int4): with flash SDPA the attention bucket
  collapses, so the quantized-conv bucket is now the dominant lever instead of being drowned by attention.

**Per-operation GPU-busy profile** (ms/step, flash SDPA; [`data/kernel_profile.csv`](data/kernel_profile.csv)).
![kernel profile](03_kernel_profile.png)

| bucket | fp32 | fp16 | int8 base | int8 modiff | int4 base | int4 modiff |
|---|--:|--:|--:|--:|--:|--:|
| **conv (GEMM)** | 27.42 | 13.57 | 9.53 | 11.76 | **7.37** | **7.42** |
| attention (softmax+SDPA) | 37.23† | 3.43 | 3.42 | 3.38 | 3.42 | 3.40 |
| GEMM (qkv/proj + QKᵀ·AV) | 6.40 | 1.55 | 1.55 | 1.53 | 1.53 | 1.54 |
| GroupNorm | 7.54 | 5.76 | 5.55 | 5.43 | 5.26 | 5.42 |
| conv store epilogue | 0 | 1.82 | 1.58 | 1.33 | 2.47 | 1.33 |
| quantize / MoDiff delta | 0 | 0 | 0.20 | 2.96 | 0.18 | 2.61 |
| elementwise / copy | 6.50 | 4.09 | 2.20 | 3.29 | 1.66 | 3.28 |
| upsample / concat | 1.80 | 1.07 | 1.30 | 1.03 | 1.28 | 1.03 |
| other | 0.35 | 0.29 | 0.27 | 2.56 | 0.27 | 2.55 |
| **GPU-busy total** | **87.24** | **31.59** | **25.59** | **33.28** | **23.44** | **28.59** |

†fp32 keeps math SDPA (PyTorch flash is fp16-only) — that 37 ms T×T term is why fp32 is 87 ms; fp16/int8/int4
all get flash (attention 3.4 ms). **With attention + QKᵀ·AV now ~5 ms (was ~22 ms under math), the conv bucket
(13.6 fp16 / 7.4 int4) is the dominant cost** — so conv quantization drives the e2e win (int4 conv 7.4 vs fp16
13.6 ms). MoDiff modes add ~2.6–3.0 ms quantize-delta + ~2.5 ms other (accuracy machinery, not speed).

**Total IO usage** — analytical DRAM bytes/step, flash model (no T×T);
[`data/pipeline_io_analytic.csv`](data/pipeline_io_analytic.csv). ![pipeline IO](02_pipeline_io.png)

| precision | conv | qkv/proj lin | attention | **total** | vs fp16 |
|---|--:|--:|--:|--:|--:|
| fp32 | 2298 | 1297 | 812 | **4406** | 1.85× |
| fp16 | 1330 | 648 | 406 | **2385** | 1.00× |
| int8 | **847** | 648 | 406 | **1901** | 0.80× |
| int4 | **605** | 648 | 406 | **1659** | **0.70×** |

Flash collapses attention IO to 406 MiB (was 5867 under math's T×T), so the total is now **conv-dominated** and
quantization's IO win shows: int8 0.80×, int4 **0.70×** of fp16 (was 0.94×/0.91× when the fp16 T×T dominated).

## §3. Obsoleted: attention-score quantization (§6 int8-flash, qkv→flash fusion)

This session built a correct fused **qkv-int-output → int8-flash** path (int8 & int4): `gemm_w8a8_out_int8` /
`gemm_w4a4_out_int8` (int8 output = `round(acc·a_scale·w_scale[c]·oscale[c]+bias)`, ≤1 code) + `transpose_qkv_int8`
(exact) + calibrated static scales; quality-safe (fused int8 latent rel-err **0.0097**). Under the old *math*
default it was the best attention-quant config (−1.3/−1.7 ms vs §6, −21% peak). **But flash SDPA obsoletes it**
([`data/fusion_pipeline.csv`](data/fusion_pipeline.csv), int8_baseline mode):

| attention path | wall ms | peak MiB |
|---|--:|--:|
| **fp16, flash SDPA** | **26.6** | 3603 |
| §6 per-token int8 flash | 52.1 | 3603 |
| fused int8 (W8A8→flash) | 50.6 | 3612 |
| fused int4 (W4A4→flash) | 50.2 | 3612 |

Flash SDPA is **~2× faster** than our best int8 attention path (our int8 flash kernel is ~10× slower than
flash SDPA, §1) **and matches its peak memory** (flash already avoids the T×T matrix). So the fused kernels are
kept in-tree, correct and opt-in (`MODIFF_QKV_FLASH_FUSED=8|4`), but are **not** on any recommended path — the
attention lever is the backend, not quantization. ![obsoleted](08_qkv_flash_fusion.png)

## §4. What still helps — the kernel-level wins

**Conv quantization** is now the dominant e2e lever (int4 base 1.34×, int8 base 1.23× vs fp16). Per-conv kernel
speed, top shapes by cost ([`data/kernel_conv_speed.csv`](data/kernel_conv_speed.csv), µs, batch 32).
![conv speed](04_kernel_conv_speed.png) ![kernel IO](06_kernel_io.png)

| conv (3×3) | fp16 | int8 | int4 | int8 ×fp16 | int4 ×fp16 |
|---|--:|--:|--:|--:|--:|
| 384→384 32² | 759.7 | 447.8 | **396.5** | 1.70× | 1.92× |
| 576→192 32² | 587.2 | 329.2 | 338.1 | 1.78× | 1.74× |
| 384→192 32² | 395.2 | 227.7 | 244.3 | 1.74× | 1.62× |
| 768→384 16² | 378.1 | 204.9 | **160.6** | 1.84× | 2.35× |
| 192→192 32² | 210.4 | **116.8** | 189.2 | 1.80× | 1.11× |
| 384→384 16² | 195.6 | 103.7 | 106.7 | 1.89× | 1.83× |

int8 is ~1.7–1.9× fp16 on every compute-bound 3×3; int4 wins where channels are large (768→384: 2.35×) and
loses to int8 on small-channel shapes (192→192; CUTLASS int4 needs warp-K≥128, and its config set is exhausted).
The aggregate conv bucket (§2): int4 7.4 ms vs int8 9.5 vs fp16 13.6.

**Attention GN→qkv fusion** ([`data/kernel_attn.csv`](data/kernel_attn.csv)): the fused GroupNorm→qkv CUTLASS
kernel is **1.11×** (C192/T1024, 236→213 µs) / **1.26×** (C384/T256) vs GroupNorm+cuBLAS — on for all modes.

**Projection-linear (qkv/proj) quantization — opt-in §7** (`MODIFF_QUANT_LINEAR=1`). Two W8A8 GEMM backends:
our `gemm_wxax` (`half2`-epilogue, beats fp16 on 5/6 shapes, int4 ≤2.13× — [`data/gemm_wxax_shapes.csv`](data/gemm_wxax_shapes.csv))
and **AWQ's `w8a8_gemm_forward_cuda`, now the default** where it fits (out%128 & in%64): AWQ is **1.28–1.44×
faster than cuBLAS** on the eligible qkv/proj shapes — better than our kernel there — and numerically matches
(rel 4e-4); shapes it can't take (e.g. the dominant C192 qkv, N=576) fall back to `gemm_wxax`.
`MODIFF_WXAX_NO_AWQ=1` forces our kernel. **But §7 is still e2e near-neutral-to-slightly-negative under flash
SDPA** ([`data/linear_quant_speed.csv`](data/linear_quant_speed.csv)): int8 base 0.90×, int4 base 0.95×, int8
modiff 0.94×, int4 modiff 0.97× — the projections are only ~1.5 ms of the flash-cheap ~27 ms step, activation-
quant adds fixed overhead, and the biggest qkv (N=576) can't use AWQ. Real benefit is peak memory (−~100 MiB).

## §5. int8-conv-output → GroupNorm (validated, deferred)

Quality is ~free (fake-quant rel-err 0.0023–0.0033) and the int8-in `group_norm_silu_dequant_quantize_nhwc`
kernel is correct (≤1 code) + ~1.05×, but realizing the conv-side write saving needs a direct-int8-output
CUTLASS conv epilogue (the `relu_requant` path writes fp16 scratch first). Projected e2e ~2%; deferred.

---

## Bottom line

1. **Biggest win by far: flash SDPA.** Not forcing the math backend gives fp16 **1.71× + −22% peak**,
   quality-identical, on every mode — dwarfing every quantization effort. `int4 base` is now **1.34× vs fp16 /
   3.72× vs fp32 / 2.34× vs the originally-shipped fp16-math**.
2. **Attention-score quantization is obsolete.** Flash SDPA beats our int8 flash by ~2× on speed and ties on
   memory; the entire premise (force math SDPA so QKᵀ/AV are quantizable) was net-negative. Kept opt-in only.
3. **Conv quantization is the remaining real lever** (int4 1.34×), now visible because attention is cheap.
4. Lesson: the attention bottleneck was a **software-config artifact** (forced math SDPA), not a kernel gap —
   the fix was one backend switch, not a faster-than-cuBLAS kernel (which we also confirmed we can't write:
   our int8 flash is 10× off PyTorch's flash SDPA).

*Fusion opt-in: `MODIFF_QUANT_ATTN=1 MODIFF_QKV_FLASH_FUSED=8|4`. Revert to math SDPA: `MODIFF_SDPA_MATH=1`.
Env: `PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass`, ninja installed.*
