# Diffusion UNet — bottleneck of our method (int8_baseline)

Companion to `REPORT.md`. Profiles the LSUN-churches latent-diffusion UNet in our shipping quantized
mode (`int8_baseline`, ~13–18% faster than fp16) to locate the bottleneck. nsys, A40, batch 32, per-DDIM-step
GPU time (capture region = one `sample()` = 12 steps). Kernels bucketed by operation.

![per-op fp16 vs int8](10_ldm_breakdown.png)
![int8 bottleneck](11_ldm_bottleneck.png)

## Per-operation GPU time (ms / DDIM step)

| operation | fp16 | int8_baseline | what our method did |
|---|--:|--:|---|
| conv (GEMM) | 11.70 (38%) | **7.84 (31%)** | int8 tensor-core convs (fp16→int8) |
| GroupNorm+SiLU (+quant) | 6.24 (20%) | 6.07 (24%) | quantize **fused into GN** (writes int8 directly) |
| attention (flash + proj) | 5.92 (19%) | 5.89 (23%) | unchanged (fp16) |
| elementwise | 5.92 (19%) | 2.81 (11%) | skip-add fused into conv store epilogue |
| upsample / concat | 1.03 | 1.27 | unchanged |
| conv store epilogue | 0 | 1.00 | fp16 residual store (int8 path) |
| quantize (standalone) | 0 | 0.20 | ~eliminated (hidden in GN) |
| **total** | **31.14** | **25.42** | **−18% GPU** |

## What each bucket contains (actual kernels measured)

The buckets group nsys device kernels by role. The three "glue" buckets — which together are ~18% of the
step and are **almost entirely fp16, memory-bound, and dtype-invariant** (int8 can't touch them) — break
down as follows (int8_baseline, ms/DDIM step):

**`elementwise` (2.81 ms)** — the pointwise "glue" between convs/attention:
- **residual & skip-connection adds + timestep-embedding adds** (`vectorized_elementwise … CUDAFunctor_add`, 0.62) — ResBlock/attention residual sums and the `h += emb` timestep injection not folded into a conv epilogue.
- **scale-shift modulation + other fused pointwise** (`elementwise_kernel … gpu_kernel_impl_nocast`, ~1.18) — the `norm·(1+scale)+shift` FiLM modulation and assorted pointwise math.
- **dtype / layout copies** (`unrolled_elementwise … direct_copy`, ~0.58) — fp16↔fp32 casts and channels_last↔contiguous `.contiguous()` copies at op boundaries.
- **SiLU activations outside the GN-fused path** (`silu_kernel`, ~0.18) — e.g. on the time-embedding MLP.
- misc (~0.14): buffer fills (`FillFunctor`), the initial noise draw (`distribution_…`), small reductions.

**`upsample/concat` (1.27 ms)** — the U-Net resampling structure:
- **nearest-neighbor upsampling in the decoder** (`upsample_nearest2d_nhwc`, 0.64) — each decoder stage ×2 spatial.
- **U-Net skip-connection concatenation** (`CatArrayBatchedCopy`, 0.63) — encoder feature maps concatenated onto the decoder path.

**`other` (0.33 ms)** — small structural ops:
- **encoder avg-pool downsampling** (`avg_pool2d_out_cuda_frame_nhwc`, 0.25).
- **native GroupNorm statistics** for norms not on the fused kernel (`RowwiseMomentsCUDAKernel` + `ComputeFusedParamsCUDAKernel`, 0.07) — mean/var passes.
- **cuDNN channel padding + timestep-embedding table lookup** (`nhwcAddPaddingKernel`, `indexSelectLargeIndex`, ~0.01).

(For reference, the other buckets: `conv (GEMM)` = the int8/fp16 CUTLASS `ImplicitGemmConvolution` kernels;
`GroupNorm+SiLU (+quant)` = our fused `group_norm_silu[_quantize]_nhwc` kernels; `attention (flash+proj)` =
`pytorch_flash::flash_fwd_kernel` + the fp16 QKV/proj GEMMs; `conv store epilogue` =
`bias_residual_store_half_from_half`; `quantize` = any standalone activation quantize kernel.)

## The bottleneck

After int8-accelerating the convs, the UNet time splits roughly three ways — **conv 31%, GroupNorm 24%,
attention 23%** — with a long memory-bound tail (elementwise + resample + store). The decisive number:

> **Only ~36% of the step (conv + store + quantize) is quantization-accelerable; ~64% (GroupNorm +
> attention + elementwise + upsample/concat) is dtype-invariant, fp16, memory-bound work that int8 cannot
> touch.**

So the bottleneck of our method is **not** the conv kernels (already int8 and fast) — it is the **fp16,
memory-bound normalization + attention** that make up ~half the UNet. This is textbook Amdahl: with ~40%
of the work quantizable and even that only ~1.5–2× faster, the end-to-end ceiling is the ~13–18% we
measure. Our fusions already claimed the cheap wins on the quantizable side (quantize folded into
GroupNorm → ~0.2 ms standalone; skip-add folded into the conv epilogue → elementwise 5.9→2.8 ms).

## Where further speedup would have to come from
- **Attention (23%)** — quantize the attention QKV/proj GEMMs to int8 (currently fp16), or a faster flash
  variant. Biggest single untapped block.
- **GroupNorm (24%)** — already fused with quant/SiLU; it's memory-bound, so gains need fewer passes
  (e.g. fusing GN into the preceding conv store) rather than dtype.
- Convs are near their int8 ceiling (see `KERNEL_BENCHMARK.md`); little left there.

*(MoDiff modes on this UNet trade the above speed for temporal-accuracy quality; see the diffusion results
in the prior handoff. This profile is the baseline int8 path.)*
