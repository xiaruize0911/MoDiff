# int8 / int4 speedup vs fp16 — MoDiff churches UNet — 2026-07-16

**Question this report answers:** does quantizing the pipeline to int8 / int4 actually make it faster
than fp16, and where does the speedup come from? Baseline is **precision-isolated fp16**: both fp16 and
the int modes use the *same* materialized attention (bmm→softmax→bmm), so only the numeric precision
differs. A40, LSUN-churches LDM UNet, batch 32, DDIM, GPU-busy = torch.profiler device self-time
(throttle-robust); e2e numbers are the 20-run-averaged clean measurements (`clean_speed.csv`).

## TL;DR

- **Yes — int8 and int4 are faster than fp16: int8 1.33×, int4 1.51× e2e** (GPU-busy). Fastest overall
  **int4 = 44 ms/step** vs fp16 67 ms.
- **The speedup comes almost entirely from the conv layers.** Conv matmul: fp16 13.5 → int8 9.6 (1.4×)
  → int4 7.4 (1.8×). The **qkv/proj-linear + attention matmul does *not* speed up** (fp16 13.6 → int8
  14.5 → int4 12.4) — the qkv/proj linears are **short-K (K=192/384/768) memory-bound** shapes where the
  int GEMM can't beat cuBLAS fp16, and the attention QKᵀ/AV is materialized (memory-bound). Attention
  *in isolation* is faster in int (1.45× int8 / 1.71× int4 at T=1024), but bundled with the slow int
  linears the bucket is flat.
- **The e2e win is Amdahl-capped.** Matmul is only ~40% of GPU time; the rest is softmax, GroupNorm,
  elementwise, quantize overhead — so even free matmul would cap at ~1.7×, and quantization only
  accelerates the conv slice of that 40%. This is why we see ~1.3–1.5×, not the textbook 2× / 4×.

(Quality is out of scope for this speed report — see the static-vs-dynamic report for rel-err numbers.)

## Why the profile's "attn QKᵀ/AV" bucket is separate for int but not fp16 (a caveat, not a result)

In the nsys/profiler bucketing, int8/int4 attention uses named kernels (`bmm_qk_s8/s4`, `bmm_av_s8/s4`)
so it lands in a separate **attn QKᵀ/AV** bucket, while fp16 attention is `torch.bmm` (cuBLAS) and gets
lumped into **conv/linear GEMM**. So fp16's attention matmul is *hidden* inside its conv/linear bar,
making the fp16 GEMM bar look bigger and the int bars look smaller. **It is not that quantization made
the GEMM shrink for free** — you must add the two matmul buckets. All tables below use the merged
matmul view (torch.profiler's `GEMM (qkv/proj + attn QK·AV)` already merges attention with qkv/proj,
and `conv (GEMM)` is separate), which *is* comparable across precisions.

## 1. E2E speedup (GPU-busy ms/step, `clean_speed.csv`)

| config | ms/step | speedup vs fp16 |
|---|--:|--:|
| fp16 (materialized) | 66.9 | 1.00× |
| int8 static (fastest) | 50.3 | **1.33×** |
| int8 balanced (dyn attn) | 55.0 | 1.22× |
| int4 static (fastest) | 44.3 | **1.51×** |
| int4 balanced (dyn attn) | 50.6 | 1.32× |

![E2E speedup vs fp16](01_speedup_vs_fp16.png)

> Deployment aside: fp16 here is *materialized* (67 ms) to isolate precision. The fastest real fp16
> uses flash/SDPA-math (~56 ms); against that, int8-static is ~1.12× and int4-static ~1.27×. We use the
> materialized (precision-isolated) baseline as the headline per the study's design.

## 2. Where the speedup comes from — matmul by type (`kernel_profile.csv`, batch 32)

| matmul bucket | fp16 | int8 | int4 |
|---|--:|--:|--:|
| **conv (GEMM)** | 13.5 | 9.6 (1.4×) | **7.4 (1.8×)** |
| **qkv/proj + attn QKᵀ/AV** | 13.6 | 14.5 (0.93×) | 12.4 (1.1×) |
| combined matmul | 27.0 | 24.1 | 19.8 |

**The merged `qkv/proj + attn QKᵀ/AV` bucket is misleading** — it lumps two ops that move in opposite
directions. Split apart (profiler now separates `attn QKᵀ/AV (int GEMM)` via the `bmm_qk/av` kernels):

- **conv (GEMM): quantization delivers** — int8 1.4×, int4 1.8×.
- **qkv/proj linear: quantization *hurts*** — the linears are short-K (K=192–768), memory-bound, and
  don't beat cuBLAS fp16 (§5 kernel wall, 0.4–1.0×).
- **attention QKᵀ/AV: quantization *delivers*** — the gain was hidden by the slow int linear in the
  merged bucket.

![Matmul breakdown](02_matmul_breakdown.png)

**Attention op-by-op, isolated (fp16 vs int8 vs int4, T=1024, `attn_ops.csv`):**

| attention op | fp16 µs | int8 | int4 |
|---|--:|--:|--:|
| QKᵀ (writes fp16 S) | 2906 | 1356 (**2.14×**) | 1386 (**2.10×**) |
| softmax | 1867 | 1806 (1.03×) | 1792 (1.04×) |
| AV (reads T×T P) | 976 | 824 (1.18×) | 463 (**2.11×**) |

So the attention **matmuls do speed up 2–4×** (QKᵀ 2.1×; AV 1.2× int8 / 2.1× int4 — int4's packed P
halves the T×T read again). Only **softmax is precision-neutral** (~1.0×): it is bandwidth-bound on the
**fp16 score matrix S**, which stays fp16 for numerical range regardless of Q/K/V precision, so quantizing
the operands doesn't shrink its dominant traffic (§7). QKᵀ's 2.1× is partly the scale-fold (fp16 pays a
separate `S*scale` pass) and partly int8 operands; AV's win is the smaller int P read. Net: quantization
**does** accelerate attention — the earlier "no benefit" was the merged bucket masking it behind the slow
int qkv/proj linear.

![Attention op-by-op speedup](02b_attn_ops.png)

## 3. Why not 2× / 4×? — Amdahl (`kernel_profile.csv`)

Matmul (conv + qkv/proj + attn) is only ~40% of fp16 GPU time (27 of 67 ms). The remaining ~60% —
softmax, GroupNorm, elementwise, quantize/absmax — is memory/elementwise-bound and largely
precision-invariant (quantization even *adds* a quantize/absmax bucket). So the e2e ceiling from
matmul alone is ~1.7×, and since quantization only accelerates the **conv** part of the matmul, the
realized 1.33× (int8) / 1.51× (int4) is consistent with the roofline, not a shortfall of the kernels.

![Amdahl breakdown](03_amdahl.png)

## 4. Memory & IO (static configs)

| precision | analytical DRAM MiB/step | peak MiB |
|---|--:|--:|
| fp16 | 13307 | 4422 |
| int8 | 9769 (0.73×) | 4386 |
| int4 | 7999 (0.60×) | 3997 |

Quantization cuts analytical IO 27–40% and peak memory (int4 −10%).

![IO and peak memory](05_io_mem.png)

## 5. Can the linear kernels be optimized? Is AWQ faster? (`linear_backends.csv`, `linear_kernel_only.csv`)

Two measurements on the real qkv shapes — the **deployed op** (quantize + GEMM + fp16-dequant) vs the
**GEMM kernel alone** (pre-quantized inputs), each vs fp16 cuBLAS `F.linear`:

| qkv shape (M,K,N) | our w8a8 dep / **kern** | AWQ w8a8 dep / **kern** | w4a4 dep / **kern** |
|---|--:|--:|--:|
| C192 (32768,192,576) | 0.24× / **0.41×** | 0.36× / **0.95×** | 0.35× / **0.76×** |
| C384 (8192,384,1152) | 0.48× / **0.96×** | 0.54× / **1.22×** | 0.58× / **1.40×** |
| C768 (2048,768,2304) | 0.64× / **1.03×** | 0.75× / **1.41×** | 0.84× / **1.62×** |

**The kernels are not the bottleneck — the quantize/dequant plumbing is.** Kernel-only, **AWQ w8a8 and
w4a4 beat fp16 cuBLAS for K≥384** (AWQ up to 1.41×, w4a4 up to 1.62×), and AWQ is ~parity even at the
hard short-K C192 (0.95×). But the **deployed** op is 0.24–0.84× because each call pays: (1) an
activation-quantize pass (fp16→int8), (2) an fp16-output dequant write, and (3) for AWQ, a per-call
`asc` allocation + output-slice from the N-padded buffer. That overhead is larger than the kernel's win.

**So yes, there is real optimization headroom, and it is fusion, not a faster GEMM:**
- **Fuse the activation-quantize into the producer** (write int8 directly out of the preceding op's
  epilogue — same trick that made conv fast via `group_norm_silu_quantize`). Removes pass (1).
- **int8/int4 output** where the consumer re-quantizes anyway — the qkv-linear output feeds attention,
  which *immediately re-quantizes Q/K/V*; emitting int8 and consuming it directly removes passes (2) and
  the attention's own quantize. Biggest single lever.
- **Use AWQ (not our gemm_w8a8) for W8A8** — it is faster on every shape (kernel 0.95–1.41× vs our
  0.41–1.03×); precompute its `asc` and avoid the output slice (use N%128 or a fused slice).
- **C192 (K=192) is fundamentally hard** — even the AWQ kernel is only ~parity (short-K, memory-bound);
  don't expect a matmul win there regardless.

Attention QKᵀ/AV is the same story: the int kernels are faster in isolation (1.45×/1.71× at T=1024,
§2) but the materialization IO + surrounding quantize eat it; the fix is the same (int-output fusion),
short of a quantized flash kernel (out of scope).

## 6. Prototype: int8-output qkv→attention fusion (`fusion_qkv_attn.py`, `fusion_qkv_attn.csv`)

Built and validated the fusion the §5 analysis pointed to. The qkv-linear emits **int8** directly
(`gemm_w8a8_out_int8`, per-column `oscale`), and a new kernel **`quantize_attn_qkv_from_i8`** consumes
that int8 straight into the attention's per-head int8 Q/K/V — dequant-on-the-fly (reciprocal-multiply),
**no fp16 round-trip and no reshape copy**. Compared to the current path (int8 linear → fp16 → reshape →
`quantize_attn_qkv`), on the qkv-linear + quantize step:

| block | path A (fp16 round-trip) µs | path B (int8 fused) µs | speedup | rel-err A / B |
|---|--:|--:|--:|--:|
| C192 T1024 | 1027 | **903** | **1.14×** | 0.019 / 0.022 |
| C384 T256 | 413 | **378** | 1.09× | 0.028 / 0.031 |
| C768 T64 | 317 | **290** | 1.09× | 0.036 / 0.040 |

Component breakdown (C192 T1024, µs): path A = gemm-fp16-out 270 + **reshape copy 187** + quantize 571
= 1027; path B = gemm-int8-out 286 + `from_i8` 617 = 903. **The fusion eliminates the 187µs reshape copy**
and folds the quantize into one int8-reading kernel; correctness is preserved (int8-output adds only
~0.003 extra rel-err — quality-safe). `gemm_w8a8_out_int8` is ~neutral vs fp16-out (286 vs 270).

![int8-output qkv→attn fusion](06_fusion_qkv_attn.png)

**Micro verdict:** correct, and ~1.1–1.14× on the qkv-linear+quantize sub-step — but that micro baseline
used a plain `gemm_w8a8` (no GroupNorm fusion).

### 6a. Wired into the live block — two iterations, honest e2e

**Iteration 1 (naive): net-NEGATIVE.** First wiring used a *separate* GN + `gemm_w8a8_out_int8`, which
gives up the block's `fused_gn_qkv` (GroupNorm fused into the qkv conv, **206 µs** on C192 T1024). The
separate GN + int8 gemm is **443 µs (2.15×) on the qkv-production side** — the 237 µs loss outweighs the
140 µs saved on the quantize side → static_int8 **49.3 → 51.7 ms (0.95×, slower)**. Lesson: a fusion
that discards a stronger existing fusion loses, even if the micro (vs an unfused gemm) looked +1.14×.

**Iteration 2: built `fused_gn_qkv_int8`.** New CUTLASS kernel = the fp16 GN→qkv mainloop with an
**int8-clamp epilogue**; the per-column requant `oscale` is folded into the (fp16) conv weight + epilogue
bias offline (bias stored int8, consistent with the output grid), so the GEMM emits int8 directly and
**keeps the GN fusion**. Standalone: **1.07–1.10× vs fp16 `fused_gn_qkv`** and emits int8 (dequant
rel-vs-fp16 **0.0038**). Re-wired the block to `fused_gn_qkv_int8` → `quantize_attn_qkv_from_i8`:

| static_int8 (batch 32, GPU-busy) | ms/step |
|---|--:|
| fusion **off** (baseline) | 49.58 |
| fusion **on** (fused_gn_qkv_int8) | **49.57** |

**Result: e2e-neutral (within noise).** The regression is gone (the int8 path now keeps GN fusion and
is 1.07–1.10× on the qkv side), and the qkv+quantize sub-step is ~1.2× faster in isolation — but that
sub-step is a small fraction of the 50 ms pipeline, so the win is **Amdahl-diluted to ~0** e2e. The
fused int8 path is confirmed active in the profile (`aq_*_from_i8` + `fused_gn_qkv_int8` kernels run;
output finite, quality-safe). Kernels: `fused_gn_qkv_int8` (`fused_gn_qkv.cu`) + `quantize_attn_qkv_from_i8`
(`attn_quant_gemm.cu`); flag `MODIFF_FUSE_QKV_I8`, default **off**.

**Conclusion:** the fusion is now correct and non-regressing, but delivers **no e2e speedup** — it
confirms (again) that the pipeline is memory/Amdahl-bound, not qkv-quantize-bound. Not worth enabling by
default; the machinery stays in-tree. Further attention speedup needs the softmax/elementwise memory
traffic or a quantized flash kernel, not more qkv-side fusion.

## 7. Softmax / elementwise memory-traffic profile (`softmax_mem.csv`)

The pipeline's memory-bound tail is dominated by **passes over the `[BH,T,T]` attention score matrix**
(512 MiB at T=1024). Top pipeline kernels (torch.profiler, ms/step): static_int8 — softmax 8.7, QKᵀ 7.9,
AV 5.0, GN-quantize 4.3, quantize 2.6, elementwise 3.4; fp16 — softmax 13.4 **+ a 10.8 ms elementwise
`AUnaryFunctor`** which is the `S*scale` multiply over the whole T×T matrix (the int8 path folds that
into the QKᵀ epilogue, so it is absent there). ncu counters are blocked, so achieved bandwidth is
analytical bytes ÷ measured time vs A40 peak (696 GB/s), T=1024:

| kernel | µs | DRAM MiB | GB/s | % peak |
|---|--:|--:|--:|--:|
| score `*scale` (fp16 elementwise) | 1881 | 1024 | 571 | **82%** |
| softmax int8 **static** (1-pass) | 1373 | 768 | 586 | **84%** |
| softmax int8 dynamic (2-pass) | 1775 | 768 | 454 | 65% |
| softmax fp16 dynamic (2-pass) | 1857 | 1024 | 578 | 83% |
| QKᵀ int8 (writes fp16 S) | 1352 | 528 | 409 | 59% |
| AV int8 (reads P) | 823 | 304 | 387 | 56% |

![Softmax/score memory roofline](07_softmax_mem_roofline.png)

**Findings:**
- The softmax and score-scale kernels run at **82–84% of peak DRAM bandwidth** → they are essentially at
  the **bandwidth roofline**. There is no meaningful kernel-tuning headroom left; the only lever is
  **moving fewer bytes** over the T×T matrix.
- The **fp16 path pays an extra full T×T pass** (`S*scale`, 1024 MiB, 82% peak) that the int8 path avoids
  by folding the scale into the QKᵀ epilogue — a real, already-captured int8 advantage.
- **int8 static softmax (1-pass, 84% peak) beats the 2-pass dynamic** (65%): the 2nd score read + the
  block reduction drop it off the roofline. This is the concrete payoff of the static single-pass softmax.
- The QKᵀ/AV matmuls sit at **56–59% of peak** — bound by the fp16 score **write** (QKᵀ) and the int8 P
  **read** (AV), i.e. also T×T-memory-bound, not compute-bound.
- Small-T blocks (T=256/64) run softmax at only 8–30% of peak (launch/latency-bound) but are cheap in
  absolute terms.

**Implication:** the attention tail is DRAM-bandwidth-bound on the materialized T×T matrix, and the
kernels are already near roofline. Further speedup requires **fewer T×T bytes**, not faster kernels:
(a) a **quantized flash** kernel that never materializes S (out of scope — the reason we chose materialized
standard attention), or (b) **lower-precision scores** (int8/fp8 S instead of fp16) to shrink every T×T
pass — but softmax needs enough logit precision, so this trades quality. No amount of GEMM/quantize
fusion helps here, consistent with §6a's neutral e2e result.

## Verdict

- **int8/int4 do beat fp16** (1.33× / 1.51× e2e, precision-isolated) — the desired speedup is real.
- **It's a conv win, capped by Amdahl.** Conv quantization gives 1.4×/1.8×; the qkv/proj-linear +
  materialized-attention matmul doesn't improve *as deployed*, and matmul is only ~40% of the pipeline
  — so e2e lands at ~1.3–1.5×, not 2×/4×.
- **The kernels aren't the bottleneck; the plumbing is (§5).** Kernel-only, AWQ w8a8 and w4a4 already
  beat fp16 for K≥384 (up to 1.4–1.6×); the deployed linears lose only to per-call quantize/fp16-dequant
  overhead. Switch W8A8 to AWQ (faster than our gemm_w8a8 on every shape). C192 (K=192) stays hard.
- **Fusion must preserve existing fusions, and even then it's Amdahl-bound (§6/§6a).** The int8-output
  qkv→attention fusion was built, validated, and wired in twice: (1) naive (separate GN + int8 gemm) was
  net-negative (49.3→51.7 ms) because it discarded the `fused_gn_qkv` kernel; (2) building
  **`fused_gn_qkv_int8`** (GN + int8-clamp epilogue, oscale folded into the weight — keeps GN fusion,
  1.07–1.10× standalone, rel 0.0038) removed the regression → **e2e-neutral** (49.57 vs 49.58 ms). The
  qkv+quantize sub-step is ~1.2× faster but too small a slice to move e2e. Net: no e2e win; the machinery
  is in-tree, flag off. Real attention headroom is the memory-bound softmax/elementwise, not qkv fusion.

## Files
`scripts/mkplots.py` (reads `../static_vs_dynamic_2026-07-16/data/` measured CSVs). Plots `01`…`05`.
Speed/profile/IO data are the same measured runs as the static-vs-dynamic study; this report re-frames
them around the int8/int4-vs-fp16 speed question (quality is out of scope here).
