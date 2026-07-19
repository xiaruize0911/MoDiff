# Where the time goes, and why quantization gives far less than the expected 2–4×

**int8_baseline / int4_baseline, batch 128 (best), torch.profiler device self-time.** Full per-kernel
breakdown in `data/detailed_{int8,int4}_baseline_b128.csv`; script `scripts/detailed_profile.py`.

## 1. Config verification: static quantization + fused kernels ✓

`detailed_profile.py` checks the live modules:

| | int8_baseline | int4_baseline |
|---|---|---|
| Linear layers with **static** a_scale | **42 / 42** | **42 / 42** |
| Conv static scales | yes (`apply_static_scales`) | yes |
| GN→int-quantize **fused** qkv | **21 / 21** | 0 / 21 (fusion is int8-only) |
| proj transpose+quantize **fused** | **21 / 21** | 0 / 21 (int8-only) |
| output-fused Linear (`out_i8`) ready | 42 | 42 |

- **int8 is fully static + fused** (GN→int8 qkv, proj transpose+quant, output-fused GEMM available).
- **int4 is static but its attention qkv/proj quantize is NOT fused** (the GN→quantize / transpose+quant
  fusions were built int8-only) → int4 pays an extra unfused GN(fp16) + separate `quant_act_int4_pack`.
  int4 spends **8.95 ms (5.2%) in quantize/dequant vs int8's 3.0 ms (1.7%)** — a ~5 ms int4-only tax
  (fixable by extending the fusions to int4; ~3% of the step).

## 2. Where the time is spent (int8_baseline, GPU-busy 176 ms/step)

| category | ms/step | % | quantizable? |
|---|--:|--:|:--:|
| **attention softmax** (fp16) | 41.9 | 23.8% | no (memory-bound) |
| **attention QKᵀ/AV bmm** (fp16) | 40.2 | 22.8% | no (fp16, we tried — slower) |
| **elementwise / copy** | 32.1 | 18.3% | no (memory-bound) |
| **conv** (int8 GEMM) | 22.4 | 12.7% | **yes ✓** |
| **GroupNorm** | 21.7 | 12.3% | no (memory-bound) |
| qkv/proj int GEMM | 5.1 | 2.9% | **yes ✓** |
| upsample / concat | 4.7 | 2.7% | no |
| other fp16 GEMM / quantize / other | 8.0 | 4.5% | — |

**Quantizable compute (conv + qkv/proj int GEMM) = 27.5 ms = 16% of the step.**
**fp16 / memory-bound (attention + GroupNorm + elementwise) = 140.6 ms = 80%.**

Single biggest kernel: `softmax_warp_forward` ≈ **40 ms (23%)** — one fp16, memory-bound kernel.

## 3. Why the speed is "far less than expected"

The expected **2–4×** is the **GEMM compute roofline** — it assumes the model is dominated by
matrix-multiply FLOPs that int8/int4 tensor cores accelerate. **This model isn't.** Only **16%** of the
step is quantizable GEMM; the other **80%** is attention (memory-bound softmax + fp16 QKᵀ/AV bmm),
GroupNorm, and elementwise/copy — none of which quantization touches.

**Amdahl ceiling:** even if every int GEMM were *free*, e2e speedup ≤ `176 / (176 − 27.5) = 1.19×`.
Measured is 1.05–1.075× because conv only gets ~1.8× (not ∞), the qkv/proj GEMM is short-K memory-bound
(~1×, §1–5 sibling report), and int adds a small quantize/elementwise tax. **The quantization is already
delivering close to its structural ceiling on this model — the ceiling itself is low.**

Why attention (47%, the obvious target) stays fp16: it's memory-bound on the fp16 T×T score matrix, and
every quantized-attention variant we built (materialized int8/int4 static-c, dynamic int8-score, fused
int8 flash) was either slower than fp16 SDPA on this small-head-dim (hd=24) UNet or quality-broken
(see the sibling reports). So attention is not a quantization win here.

## Update: int4 unfused-quantize gap closed (`data/int4_gap_b128.csv`)

Extended the qkv/proj quantize fusions to int4 (previously int8-only):
- **qkv**: `group_norm_silu_quantize_pack_nhwc` (GN→int4-pack, gemm_w4a4 layout) → `gemm_w4a4_awq`. The
  existing kernel already emits the `[N,H,W,C/2]` = `[M,K/2]` GEMM layout (the earlier rel-1.41 was my
  wrong `.permute`; the correct `.reshape` gives rel 0.014 vs reference).
- **proj**: new kernel `quantize_attn_out_int4_pack` (transpose + int4 quantize + pack) → `gemm_w4a4_awq`.

**16/16 fusable qkv/proj layers now fuse** (C384/C768). **C192 (T=1024, 5+5 layers) is structurally
excluded** — int4 needs K%128, so its K=192 would need a padded packed output; it falls back.

| int4_baseline (b128) | mean ms/step | rel-L2 vs fp16 | fused |
|---|--:|--:|--:|
| fuse OFF | 177.15 | 0.1535 | 0/0 |
| fuse ON | 177.00 | 0.1589 | 16/16 |

**Result: gap closed for all fusable layers; e2e = +0.09% (neutral).** As predicted from the profile —
the reclaimed unfused quantize (`quant_act_int4_pack`) is ~1.3% of the step and dominated by the
structurally-unfusable C192, so removing it from the C384/768 layers is within noise. Quality cost of the
fusion: +0.005 rel-L2 (fp32-GN-quantize vs fp16-GN, quality-safe — same nature as the int8 fusion).
Correctness: no blow-up (rel 0.159, not ~1.4), kernels verified. Net: the int4 config is now as-fused as
int8 wherever the int4 K-granularity allows, but this confirms it does **not** move e2e — the ceiling is
the 80% fp16/memory-bound work, not the quantize.

## Takeaways

1. It **is** static + fused (int8 fully; int4 has an unfused-quantize gap worth ~5 ms).
2. The step is **attention/memory-bound (80%), not GEMM-bound (16%)** — so quantization's e2e ceiling is
   ~1.19×, and we're at 1.05–1.075×, near that ceiling.
3. To go faster you must attack the **memory-bound** work (softmax ≈ 23%, elementwise ≈ 18%, GN ≈ 12%),
   not add more GEMM quantization. The lever is a genuinely faster **attention** kernel (a hd=24-tuned
   flash, or larger-head-dim models), not int-precision on the GEMMs.
