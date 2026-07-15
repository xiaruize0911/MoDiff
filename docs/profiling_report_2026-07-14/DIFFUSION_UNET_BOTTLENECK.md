# Diffusion UNet — bottleneck of our method (int8_baseline)

Companion to `REPORT.md`. Profiles the LSUN-churches latent-diffusion UNet in our shipping quantized
mode (`int8_baseline`, ~13–18% faster than fp16) to locate the bottleneck. nsys, A40, batch 32, per-DDIM-step
GPU time (capture region = one `sample()` = 12 steps). Kernels bucketed by operation.

![per-op fp16 vs int8](10_ldm_breakdown.png)
![int8 bottleneck](11_ldm_bottleneck.png)

## Per-operation GPU time (ms / DDIM step)

| operation | fp16 | int8_baseline | int4_baseline |
|---|--:|--:|--:|
| conv (GEMM) | 11.70 | **7.84** | **5.76** |
| conv store epilogue | 0 | 1.00 | 1.94 |
| GroupNorm+SiLU (+quant) | 6.24 | 6.07 | 5.91 |
| attention (flash + proj) | 5.92 | 5.89 | 5.90 |
| elementwise | 5.92 | 2.81 | 2.25 |
| upsample / concat | 1.03 | 1.27 | 1.27 |
| quantize (standalone) | 0 | 0.20 | 0.18 |
| other | 0.34 | 0.33 | 0.34 |
| **GPU-sum total** | **31.14** | **25.42** | **23.56** |
| **wall-clock / step** | **32.1** | **27.2** | **25.8** |

int8_baseline is what our fusions target: int8 convs (11.7→7.8), quantize folded into GroupNorm
(standalone ~0.2 ms), skip-add folded into the conv store epilogue (elementwise 5.9→2.8).

> **Wall-clock corrected (2026-07-15).** An earlier version of this table reported wall-clock 38.5 / 31.8 /
> **38.1** ms and concluded int4 "doesn't pay off" (a 14.5 ms overhead tail). Those wall numbers were an
> **nsys/under-warmed measurement artifact** — the A40 idles at 210 MHz and clocks can't be locked here, so
> without heavy warmup the int4 path was timed mid clock-ramp. Re-measured with ≥6 s sustained warmup + 12
> low-variance runs (stdev <1%, see `../comprehensive_benchmark_2026-07-15/`), the **GPU-sum numbers were
> already correct** (they agree to <1%), but the true wall-clock is **32.1 / 27.2 / 25.8 ms**: int4's overhead
> tail is ~2.4 ms, not 14.5 ms. **int4_baseline is actually the fastest mode (1.25× vs fp16), int8_baseline
> close behind (1.18×).** The dtype-invariant tail (GroupNorm + attention, ~half the step) still caps the
> end-to-end speedup at ~1.2–1.3× — the central bottleneck argument below is unchanged.

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

## Inside the attention blocks (detailed)

The UNet has **21 `TokenMajorAttentionBlock`s** (self-attention: `GroupNorm → qkv Linear(C,3C) →
scaled_dot_product_attention (flash) → proj Linear(C,C)`; no cross-attention or FF — churches is
unconditional). Micro-benchmarking each block's components (all fp16, usage-weighted per step):

![attention detail](13_attention_detail.png)

| component | ms/step | share | note |
|---|--:|--:|---|
| pre-attention **GroupNorm** | 3.50 | **40%** | the norm before each block — the biggest attention cost |
| **flash SDPA** (QKᵀ·softmax·V) | 2.74 | 31% | dominated by the high-res blocks (quadratic in tokens) |
| **qkv** Linear (C→3C) | 1.31 | 15% | fp16 GEMM |
| **proj** Linear (C→C) | 0.99 | 11% | fp16 GEMM |
| reshape/permute (token-major views) | 0.26 | 3% | mostly free views (token-major opt already removed the copies) |
| **total** | **8.81** | | (the norm 3.5 ms is counted in the GN bucket above; the rest ≈ the attention bucket) |

Per resolution (each config runs 5× per step except the last):

| C | H×W | tokens T | norm µs | qkv µs | SDPA µs | proj µs |
|--:|--|--:|--:|--:|--:|--:|
| 192 | 32² | **1024** | 340 | 107 | **437** | 77 |
| 384 | 16² | 256 | 185 | 87 | 61 | 60 |
| 384 | 8² | 64 | 79 | 33 | 23 | 27 |
| 768 | 4² | 16 | 80 | 29 | 23 | 28 |
| 768 | 2² | 4 | 84 | 28 | 24 | 33 |

**Findings:**
- **The highest-resolution block (C192, 32², T=1024) dominates SDPA** — 437 µs vs 23–61 µs elsewhere,
  because attention is O(T²) and T=1024 there. Its GroupNorm (340 µs) is also the largest.
- **The pre-attention GroupNorm is the single biggest attention cost (40%)** — a memory-bound reduction,
  not the matmuls.
- **The qkv+proj GEMMs are 26% of attention (~2.3 ms/step) and are fp16** — this is the one clearly
  int8-quantizable slice of attention. SDPA and GroupNorm are much harder to quantize.

## Prototype: int8 for the attention qkv/proj GEMMs — measured, **negative**

The qkv/proj are plain fp16 GEMMs (~26% of attention, ~2.3 ms/step), so they looked like the one
int8-quantizable slice of attention. Prototyped with `OptimizedInt8Linear(backend="int_gemm")` and, to
rule out fallback, the true W8A8 kernel forced directly. Result: **int8 is 2–3× *slower*, not faster.**

![int8 attention crossover](14_attn_int8_crossover.png)

| K = channels | fp16 µs | int8 µs | int8/fp16 |
|--:|--:|--:|--:|
| 192 (attn) | 108 | 212 | **0.51×** |
| 384 (attn) | 86 | 165 | **0.52×** |
| 768 (attn) | 32 | 103 | **0.31×** |
| 1536 | 257 | 330 | 0.78× |
| 4096 | 1813 | 1822 | ~1.00× |

**Why:** an Ampere INT8 tensor-core GEMM only out-throughputs fp16 cuBLAS once the **contraction dim K
(= in_features = channels) is large** — the measured crossover is **K ≈ 2048–4096** (below it, IMMA can't
reach peak and the int8 quantize/dequant overhead dominates). The churches UNet's attention channels are
**192–768**, far below the crossover, so int8 loses badly. The codebase already encodes this as a K-gate
(`K_INT8_GATE = 2048` in `int8_linear.py`, which silently falls back to fp16) — the prototype confirms the
gate is correct and this lever is **not worth pursuing on this UNet.** (It *would* help a larger UNet with
K ≥ 2048 attention, e.g. SDXL-scale — the code comment references exactly that regime.)

## Fused GroupNorm→qkv — first a Triton dead-end, then a **working custom CUTLASS kernel** (+1.4% e2e)

GroupNorm is the biggest single attention cost, and it is immediately followed by the qkv Linear, so
fusing the GN *normalize* into the qkv GEMM (skip writing the normalized tensor) is the obvious lever.
**A Triton fused GEMM cannot win here** (§ below), but a **custom CUTLASS per-sample mainloop-fusion conv
does** — it is the shipping result:

### The win: `fused_gn_qkv` (custom CUTLASS, `csrc/kernels/fused_gn_qkv.cu`)

The qkv Linear is a 1×1 convolution, so it maps onto CUTLASS's **fprop mainloop fusion** (example 25),
which applies a per-channel `scale·x+bias` to the activations *inside* the mainloop — exactly the GN
normalize, with GN's γ folded into the conv weight and GN's β + qkv bias into the epilogue bias. Two
custom pieces made it correct and fast:
- **Per-sample scale/bias.** GroupNorm's scale/bias depend on the activation's sample `n`; the stock
  fusion shares one `[1,C]` vector across the batch. `ImplicitGemmConvolutionFusionPerSample` offsets the
  scale/bias pointer by `sample·C` per threadblock (valid when the M-tile stays within one sample, i.e.
  tokens `T` is a multiple of the tile's `kM=128`; smaller blocks fall back to cuBLAS+GN).
- **ReLU absorption.** The stock fusion does scale+bias+**ReLU**; GN→qkv must not ReLU the sign-bearing
  normalized activations. A constant `SHIFT=16` added to the bias makes the pre-ReLU value always ≥0
  (normalized activations are ~unit variance), and the induced per-output-channel constant `SHIFT·Σ_c Wf`
  is subtracted back in the (static) epilogue bias.
- Per-(sample,group) stats come from a two-pass CUDA kernel (coalesced per-channel reduce with token-tiled
  atomics for occupancy).

**Measured (int8_baseline UNet, batch 32, A40), numerically correct (output rel err 0.0016):**

| | baseline (GN+cuBLAS) | fused CUTLASS | result |
|---|--:|--:|--:|
| wall-clock (interleaved A/B) | 24.56 ms/step | 24.20 ms/step | **1.015× (+1.5%)** |
| GPU-busy (torch.profiler self-time) | 23.90 ms/step | 23.56 ms/step | **1.014× (+1.4%)** |

Per-block: **C384/T256 1.23×**, **C192/T1024 1.10×** (the dominant block is capped by the CUTLASS-vs-cuBLAS
gap below; its GN cost is real but its matmul can't beat cuBLAS). This is the **first** approach to win
end-to-end — the Triton attempt lost 11%. **On by default** (kill-switch `MODIFF_FUSE_GN_QKV=0`); gate `test_fused_gn_qkv` PASS.

### Why Triton could not do it (the dead-end that motivated CUTLASS)

We first built the same fusion as a Triton GEMM (normalize `(x−mean_{n,g})·rstd_{n,g}` on the A operand,
GN affine folded into the qkv weights, per-(sample,group) stats from a channels_last `var_mean`). It is
**numerically correct** (rel err 0.0013) but **end-to-end slower** — and slower on **GPU-busy time**, not
just launch overhead:

| int8_baseline UNet, batch 32 | baseline | fused | result |
|---|--:|--:|--:|
| wall-clock (interleaved A/B) | 24.3 ms/step | 27.0 ms/step | **0.90× (−11%)** |
| GPU-busy (torch.profiler self-time) | 23.7 ms/step | 26.4 ms/step | **0.90× (−12%)** |

Per-block, in-context (real UNet inputs, vs the **actual** baseline `_group_norm_silu` + cuBLAS): the fused
path loses on **every** block — 0.80× (C192/32²), 0.75× (C384/16²), 0.71× (C384/8²), 0.33× (C768/≤4²).

**Root cause (three measured facts):**
1. **cuBLAS is fast on these shapes; Triton cannot match it.** For the dominant `[M=32768, K=192, N=576]`
   qkv GEMM, cuBLAS = **108 µs**; the *exhaustively* autotuned pure Triton GEMM (no normalize) bottoms out
   at **152 µs = 1.41× slower** (C384 shape: 86 vs 105 µs, 1.23×). Small-K, fat-M GEMMs are exactly cuBLAS's
   strength and Triton's weakness (too few K-iterations to pipeline the tensor cores).
2. **The matmul penalty exceeds the whole GN cost.** GN (`_group_norm_silu`) ≈ 126 µs and cuBLAS qkv ≈ 108 µs
   on the big block. Fusing *deletes* the GN write but *replaces* the 108 µs cuBLAS matmul with a ≥152 µs
   Triton one — the +44 µs matmul penalty is larger than the intermediate-write saving, so it is net-negative
   before the normalize overhead is even added.
3. **The earlier "1.1× win" was a benchmarking artifact.** It compared against *native* `nn.GroupNorm`, which
   is **1.35–1.43× slower** than the pipeline's real `_group_norm_silu` kernel (measured 180 vs 126 µs), and
   ran in a tight loop where L2 residency flattered the Triton matmul. Against the real baseline on real
   inputs, the win disappears.

**Also tried (also negative):** keep cuBLAS untouched and just beat the *standalone* GroupNorm with a
single-launch Triton GN (per-(sample,group) reduce+normalize). Correct (rel 0.0000) but it loses on the
dominant C192/T=1024 block (**0.54×**; wins only on C384/16²). Channels_last GN with a small per-group
channel count (`Cg=6`) reduces over 1024 tokens **strided by C** — scattered 12-byte reads, memory-access-
limited — and the native kernel already handles that access pattern well.

**Root cause of the Triton dead-end:** for the dominant `[M=32768, K=192, N=576]` qkv GEMM, cuBLAS = 108 µs
but the *exhaustively* autotuned Triton GEMM bottoms out at 152 µs (1.41× slower); since the GN cost we'd
save (~126 µs) ≈ the cuBLAS matmul cost, swapping to a 1.4× slower matmul is net-negative. (An earlier
"1.1× win" was an artifact of comparing against *native* `nn.GroupNorm`, 1.4× slower than the pipeline's
`_group_norm_silu`, plus L2 residency in a tight loop.) A standalone Triton GN also lost on the dominant
block (channels_last scatter). The **CUTLASS kernel wins where Triton can't** because it *matches* cuBLAS's
mainloop and fuses the normalize for free — but even it is capped by the same small-K gap: a plain CUTLASS
GEMM is also 1.37× off cuBLAS on `K=192` (157 vs 115 µs), so the dominant block only reaches 1.10×, while
the larger-K C384 blocks (where CUTLASS ≈ or beats cuBLAS) reach 1.23×. Net **+1.4% end-to-end** — a real,
correct win, bounded by the fact that no kernel beats cuBLAS on the smallest-K block.

## Where further speedup would have to come from
- **GroupNorm — the largest lever, ~24% of the step and ~40% of attention — is now partially captured.**
  The fused GN→qkv is **shipped as a custom CUTLASS per-sample mainloop-fusion conv** (`fused_gn_qkv`,
  on by default, kill-switch `MODIFF_FUSE_GN_QKV=0`): **+1.4% end-to-end**, correct (rel 0.0016). The remaining GroupNorm cost is
  bounded by the small-K CUTLASS-vs-cuBLAS gap on the dominant C192/T=1024 block (1.10× there vs 1.23× on
  C384); closing it further would need a matmul that beats cuBLAS at `K=192`, which neither CUTLASS nor Triton
  achieves. A Triton version of the same fusion was **−11% end-to-end** and is not used (see § above).
- **Attention qkv/proj GEMMs — int8 does NOT help here (prototyped, 2–3× slower; small-K crossover).** The
  remaining attention cost is SDPA (flash) and GroupNorm, both hard to quantize; the O(T²) SDPA is
  concentrated in the single 32²/T=1024 block (a faster flash variant or lower-res attention would help).
- **Convs are near their int8 ceiling** (see `KERNEL_BENCHMARK.md`). **int4 base is actually the fastest mode**
  (1.25× wall vs fp16, vs int8's 1.18×; corrected 2026-07-15 — the earlier "int4 not worth it / wall ≈ fp16"
  claim was an under-warmed measurement artifact, see the note in the table above). int4 wins on the
  large-channel 3×3 convs that dominate cost and loses only on ≤192-channel / 1×1 convs; net faster.

*(MoDiff modes on this UNet trade the above speed for temporal-accuracy quality; see the diffusion results
in the prior handoff. This profile is the baseline int8 path.)*
