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
