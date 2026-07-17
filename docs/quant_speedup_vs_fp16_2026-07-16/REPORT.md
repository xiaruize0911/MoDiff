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

**Conv quantization delivers; qkv/proj+attn does not.** The int qkv/proj linears are short-K,
memory-bound, and don't beat cuBLAS fp16 (documented kernel wall); the materialized attention is
memory-bound. So the matmul win is conv-only.

![Matmul breakdown](02_matmul_breakdown.png)

Attention *in isolation* IS faster in int (from `attn_kernel_speed.csv`, T=1024): fp16 5807 µs →
int8 4016 (1.45×) → int4 3398 (1.71×) — the tensor-core advantage is real, it just gets masked in the
bucket by the slow int qkv/proj linears next to it (and disappears at small T where quant overhead
dominates).

![Attention kernel micro](04_attn_micro.png)

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

## Verdict

- **int8/int4 do beat fp16** (1.33× / 1.51× e2e, precision-isolated) — the desired speedup is real.
- **It's a conv win, capped by Amdahl.** Conv quantization gives 1.4×/1.8×; the qkv/proj-linear +
  materialized-attention matmul doesn't improve (short-K / memory-bound), and matmul is only ~40% of
  the pipeline — so e2e lands at ~1.3–1.5×, not 2×/4×. To go further, the leverage is the qkv/proj
  short-K linear GEMM and the memory-bound softmax/elementwise, not more precision reduction.

## Files
`scripts/mkplots.py` (reads `../static_vs_dynamic_2026-07-16/data/` measured CSVs). Plots `01`…`05`.
Speed/profile/IO data are the same measured runs as the static-vs-dynamic study; this report re-frames
them around the int8/int4-vs-fp16 speed question (quality is out of scope here).
