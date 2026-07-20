# MoDiff CUDA kernels (`csrc/`)

Kernels are grouped by the **operation** they serve and the **precision** (W8A8 = int8 weight+activation, W4A4 = int4). Each kernel file carries a header comment documenting its inputs, outputs, what it computes, what it fuses, and its measured speed vs the fp16 equivalent.

## Directory layout

```
csrc/
├── pybind.cpp                 # Python bindings (all m.def entrypoints)
├── modiff_kernels_api.h       # C++ declarations for every exported kernel
├── common.cuh                 # shared device helpers (cp.async, smem ptr, etc.)
└── kernels/
    ├── common/
    │   └── mma_int8.cuh        # int8/int4 tensor-core MMA + async-copy primitives
    │                           #   (shared by linear/ and attention/)
    ├── linear/                 # Linear-layer GEMM  (qkv, proj, etc.)
    │   ├── gemm_wxax.cu         #   W8A8 / W4A4 AWQ-tiling GEMM (own port)
    │   └── awq_w8a8_gemm_cuda.cu#   vendored llm-awq W8A8 GEMM
    ├── conv/                   # Conv2d  (ResBlock / down / up)
    │   ├── conv2d_int8.cu       #   W8A8 CUTLASS implicit-GEMM conv (+fused variants)
    │   ├── conv2d_int4.cu       #   W4A4 CUTLASS implicit-GEMM conv (+fused variants)
    │   └── conv_epilogue.cu     #   shared conv epilogue helpers
    ├── attention/              # Quantized attention (fused flash) + its quantize prologue
    │   ├── attn_quant_gemm.cu   #   quantize prologue: fp16 Q/K/V -> int8/int4 Q/K + int8 V (feeds flash)
    │   └── flash_attn_int8.cu   #   FUSED int8/int4 flash attention (QKᵀ+softmax+AV, scores in SRAM)
    ├── norm/                   # GroupNorm + fusions
    │   ├── group_norm_silu.cu   #   GroupNorm(+SiLU)(+quantize/pack) NHWC
    │   └── fused_gn_qkv.cu      #   fused GroupNorm -> qkv projection
    ├── quantize/               # Quantization + MoDiff temporal caching
    │   ├── quantize.cu          #   fp16 -> int8 / packed-int4 activation quantize
    │   └── modiff_delta_quantize.cu # MoDiff cross-timestep delta quantize/accumulate
    └── util/
        └── layout_transform.cu  # NCHW<->NHWC / packing layout transforms
```

## Speed vs fp16 — summary (LSUN-churches shapes, batch 128, A40)

"vs fp16" = this kernel (family) ÷ the PyTorch fp16 op it replaces. Ratios >1 are faster.

| Op family | kernel(s) | vs fp16 | what fp16 it's compared to | notes |
|---|---|---|---|---|
| **Linear GEMM** | `gemm_w8a8_awq` / `gemm_w4a4_awq` | **W8A8 1.46× / W4A4 1.83×** | `F.linear` | *fused-quant* (activation quantize folded into upstream GroupNorm). With a **standalone** quantize the win is erased (int8 0.99×, int4 0.78×). Per-shape: wins at K≥384 (int4 up to 2.66×), weakest at K=192. |
| **Attention — fused flash (int8/int4)** | `flash_attn_int8_vt` / `flash_attn_int4_vt` (+ `quantize_attn_qkv*` prologue) | int8 flash **2.75× vs fp16 MATH** at the dominant hd24/T1024 block (~2.5× weighted over eligible blocks) | `F.scaled_dot_product_attention` **MATH** (what the fp16 baseline runs) | **Sole quantized-attention path** — the materialized QKᵀ/softmax/AV int-GEMM path (`attn_qk_int8`/`attn_softmax_requant`/`attn_av_int8` …) was **removed**. Eligible blocks (hd≤48, T%64==0) run flash; ineligible fall back to fp16 SDPA. Lifts int8 e2e to ~1.5× vs true fp16. int4 uses int4 Q/K + int8 V/P (no native int4 flash). |
| **GroupNorm(+SiLU)** | `group_norm_silu_nhwc` | **1.58–2.11×** | `F.group_norm(+F.silu)` | fp16-vs-fp16 win from reading/writing channels_last directly (avoids `F.group_norm`'s forced NCHW copy). `+quantize`/`+pack` variants fold the activation quantize in for free. |
| **GN→qkv fusion** | `fused_gn_qkv` / `_int8` | (fusion) | GroupNorm + Linear (2 kernels) | Collapses GroupNorm + qkv projection (+int8 output requant) into one CUTLASS kernel; removes the fp16 qkv materialization + reshape copy. |
| **Conv2d** | `conv2d_int8_*` / `conv2d_int4_*` | 0.48–1.16× **standalone** | cuDNN fp16 conv | cuDNN fp16 conv is highly optimized; standalone int8 (per-call quantize + fp16 dequant) loses on large-spatial, wins only high-channel/low-spatial. In the **fused int8→int8 chain** (deepfuse relu+requant, quantize once, no fp16 round-trip) conv is faster than standalone, but the whole-model e2e gain is only **~1.08× vs true fp16** (see the corrected e2e note below). |
| **Quantize / layout / MoDiff** | `quantize_act_*`, `modiff_delta_quantize`, `layout_transform` | n/a | — | Support ops (no fp16 equivalent); these are the overhead that fusion aims to hide. |

### End-to-end (whole UNet, DDIM, b128) — vs a TRUE fp16 baseline

| mode | ms/step | vs **true** fp16 |
|---|---|---|
| **fp16 (true, autocast on)** | 189.6 | 1.00× |
| int8 — conv+linear quant, attention fp16 | 175.2 | 1.08× |
| int4 — conv+linear quant, attention fp16 | 173.9 | 1.09× |
| **int8 — DEFAULT (packed quantize + residual/bias GEMM-epilogue fusion)** | **123.0** | **1.54×** ✅ |
| **int4 — DEFAULT (packed quantize + residual/bias GEMM-epilogue fusion)** | **117.1** | **1.62×** ✅ |
| int8_modiff / int4_modiff | ~199 | 0.95× (slower) |

> **2026-07-20 — residual + bias GEMM-epilogue fusion (glue optimization 2).** The fp16 residual/bias
> adds (~11.8 ms: qkv bias, proj bias + attention residual `x+proj(a)`, resblock skips) were separate
> elementwise kernels. Added optional `bias[n]` + `residual[m,n]` to the AWQ GEMM epilogue
> (`gemm_w8a8_awq_bias_res` / `gemm_w4a4_awq_bias_res`; `QuantLinearWxAx.forward(x, residual=)` uses them
> by default) — dequant + bias + residual in one store. Add traffic −79%, rel-L2 ~2e-4 vs separate adds,
> e2e int8 1.47→**1.54×**, int4 1.53→**1.62×**. Cumulative glue work: int8 1.42→1.54×, int4 1.47→1.62×.

> **2026-07-20 — packed-qkv quantize (glue optimization).** The fused attention fed its quantize kernel
> a contiguous buffer via `q/k/v.transpose(1,2).reshape().contiguous()` — a **1.2 GB/step** fp16 copy (the
> #1 "elementwise/copy glue" cost). New `quantize_attn_qkv_packed`/`_packed_static` kernels read the
> interleaved qkv `[b,T,nh,3,hd]` directly (decode row→(b,h,t) offset, write head-major int8/int4) —
> folding the transpose into the quantize sweep. Bit-identical output (verified `torch.equal`), copy traffic
> −74%, e2e int8 1.42→**1.47×**, int4 1.47→**1.53×**.

> **2026-07-20 — fused-flash quantized attention is now the DEFAULT for int8/int4.** Quantized attention
> was a loss when *materialized* (0.79× — softmax-requant + `[BH,T,T]` HBM round-trip), but the **fused
> flash** path (QKᵀ+softmax+AV in one kernel, scores in SRAM; kernel quantize; static single-pass scales;
> `flash_attn_int8_vt` V-pre-transposed) makes int8 attention *faster than fp16* — lifting int8 e2e from
> 1.08× → **1.54×** (int4 **1.62×**). It's quality-transparent (+~0.004 sampled-latent rel-L2 over fp16
> attention; the ~0.35 rel-L2 of int8 is from the linear/conv quant, not attention). The materialized
> int path was **removed** — flash is the sole quantized-attention path; `MODIFF_QUANT_ATTN=0` reverts
> to fp16 SDPA attention. Scripts: `docs/flash_attention_2026-07-19/`.

> ⚠️ **Correction (2026-07-20).** Earlier reports here (and `bench5`) claimed **int8 ≈ 2.00×**.
> That was a benchmark-harness artifact: the sampling loop used `autocast(enabled=quant)`, so the
> **fp16 baseline ran fp32/tf32** (kernels `softmax<float>`, `s1688gemm`/`s1688fprop_tf32`) at ~half
> the FP16 rate. The "2×" = **~1.85× (fp32/tf32→fp16 precision)** × **~1.08× (int8 quantization)**.
> Against a **true fp16** baseline (autocast on), int8 is **~1.08×** and the modiff variants are
> *slower* than fp16. Authoritative scripts: `docs/flash_attention_2026-07-19/scripts/true_fp16_vs_int8.py`,
> `e2e_true_fp16_table.py`. The per-kernel ratios below used explicit fp16 tensors and are unaffected.

The e2e win comes from the **conv + linear** int-GEMM chain **plus fused-flash quantized attention**,
which is now the default int8/int4 path: `flash_attn_int8_vt/int4_vt` beat fp16 MATH (2.75× at the
dominant block), lifting int8 e2e to ~1.5× vs true fp16. Ineligible (hd=96 / tiny-T) blocks stay fp16 SDPA.

> Measurement scripts: `docs/flash_attention_2026-07-19/scripts/` (flash_opt_bench, kernel_bench_all_shapes, linear_gemm_only, e2e_int8_int4_vs_fp16, …). Per-kernel I/O and fusion details are in each kernel file's header comment.
