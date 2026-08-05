# Attention under MoDiff — profile, and where the projection headroom actually is

**2026-08-04 · A40 · LSUN-churches LDM · batch 128 · five modes, time-aligned**
*Revised 2026-08-04 after the second pass; see §6 for what changed and why.*

Trigger: `MEASUREMENT_REPORT_2026-08-01`'s stage table shows *QKV / output projection* going
**1773.5 ms (fp16) → 1850.5 ms (int8)** — reading as if quantizing the projections made them slower.
That was carried into the 08-04 report without being chased, and separately I described attention as
"structurally done", which conflated two different claims: *MoDiff cannot apply to attention* (true)
with *attention is optimized* (not true).

> ### Corrections to this document's first draft
>
> 1. **"Out projection is 1.8× SLOWER at int8" is withdrawn.** It compared int8's out-projection
>    GEMM, which has bias **and the residual add** in its epilogue, against fp16's GEMM, which has
>    neither — fp16 pays the residual in a separate `vectorized_elementwise_kernel`. Paired correctly:
>    **fp16 459.8 µs vs int8 345.7 µs, i.e. int8 1.33× faster** (§2).
> 2. **The whole per-stage fp16-vs-int8 table is withdrawn, not just that row** — the fp16 column does
>    not balance. A byte count proves it cannot contain the work attributed to it (§2). `plots/attn_stage_ratio.png`
>    is deleted.
> 3. **"% of peak TOPS" was the wrong denominator.** At K=C these GEMMs are bound by memory (out proj,
>    all shapes) or by tensor-core throughput at tiny M (qkv, T ≤ 256) — not by peak TOPS. The
>    "int8 reaches 9.3% of peak where fp16 reaches 33.7%" framing overstated the gap by ~4×.
>    `plots/attn_gemm_efficiency.png` is replaced by `plots/attn_gemm_roofline.png` (§3).
> 4. **The headroom estimate moves from −0.85 to −2.8 ms/step, and it moves from the out projection to
>    qkv.** Bigger prize, different owner (§4).
>
> §1 (MoDiff does nothing inside attention) and the TOTAL-level fp16/int8 ratios are unaffected.

## Artifacts

| file | what |
|---|---|
| `traces/attention_modiff_5mode_aligned.json` | Perfetto trace, 5 modes × 5 shapes, time-aligned per shape. Open at ui.perfetto.dev |
| `plots/attn_stage_breakdown.png` | stacked stage time, per shape, per mode. **Totals are comparable; stages are not** |
| `plots/attn_fusion_accounting.png` | why the per-stage fp16/int8 ratio was withdrawn (§2) |
| `plots/attn_gemm_roofline.png` | achieved % of each projection GEMM's own roofline (§3) — the root cause |
| `plots/attn_modiff_delta.png` | MoDiff / baseline inside attention (§1) |
| `data/attn_modiff_buckets.json` | raw per-kernel µs per forward |
| `data/attn_stages_rebucketed.json` | the stage attribution |
| `data/attn_roofline.json` | the roofline model, per shape per projection |

Reading the trace: six process tracks per mode (CPU + GPU), one time slot per attention shape, all
modes starting at the same timestamp inside a slot — so the same forward reads straight down the page
across precisions. Same format as the 08-03 traces, so they are directly comparable.

All µs figures below are **per attention forward at batch 128**. To reach a whole-model µs/step,
multiply by the instance count: C192/T1024, C384/T256, C384/T64 and C768/T16 appear **5×** each in
the UNet, C768/T4 appears **1×**.

## 1. MoDiff does nothing inside attention — measured, not asserted

| shape | INT8 | INT8+MoDiff | INT4 | INT4+MoDiff |
|---|---|---|---|---|
| C192/T1024 | 2550.6 | 2594.7 (1.017×) | 2524.5 | 2587.5 (1.025×) |
| C384/T256 | 831.3 | 844.9 (1.016×) | 770.6 | 779.5 (1.012×) |
| C384/T64 | 221.8 | 222.3 (1.002×) | 205.2 | 206.6 (1.007×) |
| C768/T16 | 178.1 | 178.1 (1.000×) | 151.0 | 150.1 (0.994×) |

1.00–1.03×, and the residual 1–2% is the shared GroupNorm kernel now also serving the conv path's
delta work. The structural exclusion holds: the qkv epilogue exists to emit quantized *codes*, and
under MoDiff the GEMM produces an accumulator increment, so there are none to emit. **Any attention
work is orthogonal to MoDiff — it helps the baseline and MoDiff equally.** This section is unchanged
from the first draft; it is an int8-vs-int8 comparison and so is unaffected by the corrections.

## 2. Why the per-stage fp16-vs-int8 comparison is invalid (withdrawn)

The raw kernels at C192/T1024, which is where the time is:

| FP16 | µs | INT8 | µs |
|---|---|---|---|
| `pytorch_flash::flash_fwd_kernel` | 1781.8 | `flash_attn_int8_mma_kernel_t` | 1345.4 |
| `cutlass ImplicitGemmConvolution` (qkv, a 1×1 conv) | 617.5 | `gemm_w8a8_kernel_awq_out_i8` (qkv) | 604.9 |
| `vectorized_elementwise_kernel` | 268.2 | — *(none at all)* | 0.0 |
| `ampere_fp16_s1688gemm` (out proj) | 191.6 | `gemm_w8a8_kernel_awq` (out proj **+bias+residual**) | 345.7 |
| `gn_accum` + `gn_finalize` | 111.0 | `group_norm_silu_quantize_nhwc_vec2` | 254.6 |
| **TOTAL** | **2970.1** | **TOTAL** | **2550.6** |

**int8 has zero elementwise kernels.** Bias, the residual add and the GroupNorm apply are all inside
fused kernels. fp16 pays them in one `vectorized_elementwise_kernel` entry — and the profiler
aggregates by kernel name, truncated at 60 chars, so every elementwise functor collapses into that
single 268.2 µs number and **cannot be split per stage**.

It is worse than merely unsplittable: it cannot even contain the work it would have to contain.

```
GN apply     read x + write normalized                    = 100.7 MB
residual add read proj_out + read x + write               = 151.0 MB
                                                    total = 251.7 MB
251.7 MB in 268.2 us = 939 GB/s,  against an A40 peak of 696 GB/s.   IMPOSSIBLE
```

So at least one fp16 stage is fused somewhere the attribution does not see, or is absent. Consistent
with that: `gn_accum` reads 50.3 MB in 108.4 µs = 464 GB/s, i.e. a **read-only reduction** — fp16's
GroupNorm rows are statistics only, there is no apply kernel anywhere in the fp16 column.
**A column that does not balance cannot be compared stage by stage.**

The specific damage, and the one pairing that is defensible:

| | fp16 | int8 | |
|---|---|---|---|
| out-proj GEMM alone | 191.6 | 345.7 | 0.55× — **the mis-pairing that produced "1.8× slower"** |
| residual add | 268.2 | 0 (fused) | |
| **out proj + the residual it adds** | **459.8** | **345.7** | **1.33× — int8 faster** |

Strictly, the residual add is the *largest* thing the 268.2 µs can be (151 MB at 563 GB/s = 81% of
peak fits; adding the GN apply does not), so 459.8 is an upper bound on fp16's out-proj-plus-residual.
Even taking the other extreme — all 268.2 µs is GN apply and the residual add is simply missing from
the trace — the comparison still is not "1.8× slower", it is "unaccounted". Either way the claim goes.

The only fp16/int8 ratios that survive are the **TOTALs**: 1.16× (C192/T1024), 1.26×, 1.84×, 1.19×,
1.35×. Those are unaffected because every kernel is inside them regardless of how it is bucketed.

## 3. What replaces it: each int8 GEMM against its own roofline

int8-only, so nothing has to be attributed across precisions. A40: 696 GB/s, 299.4 TOPS int8 dense.
`qkv` reads A int8 (M·C) and writes int8 Q/K/Vᵗ (M·3C); `proj` reads A int8 (M·C), reads the residual
fp16 (2M·C) and writes fp16 (2M·C) — the trace's total absence of int8 elementwise kernels is the
evidence that the residual really is in that epilogue.

| shape | kernel | M | K | N | MB | mem µs | compute µs | **roofline** | measured | **% of roofline** | bound by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C192/T1024 | qkv | 131072 | 192 | 576 | 96.0 | 144.6 | 96.8 | **144.6** | 604.9 | **23.9%** | memory |
| C192/T1024 | proj | 131072 | 192 | 192 | 120.0 | 180.8 | 32.3 | **180.8** | 345.7 | **52.3%** | memory |
| C384/T256 | qkv | 32768 | 384 | 1152 | 48.0 | 72.3 | 96.8 | **96.8** | 302.7 | **32.0%** | compute |
| C384/T256 | proj | 32768 | 384 | 384 | 60.0 | 90.4 | 32.3 | **90.4** | 185.2 | **48.8%** | memory |
| C384/T64 | qkv | 8192 | 384 | 1152 | 12.0 | 18.1 | 24.2 | **24.2** | 89.1 | **27.2%** | compute |
| C384/T64 | proj | 8192 | 384 | 384 | 15.0 | 22.6 | 8.1 | **22.6** | 65.1 | **34.7%** | memory |
| C768/T16 | qkv | 2048 | 768 | 2304 | 6.0 | 9.0 | 24.2 | **24.2** | 54.1 | **44.7%** | compute |
| C768/T16 | proj | 2048 | 768 | 768 | 7.5 | 11.3 | 8.1 | **11.3** | 44.1 | **25.6%** | memory |
| C768/T4 | qkv | 512 | 768 | 2304 | 1.5 | 2.3 | 6.1 | **6.1** | 20.1 | **30.1%** | compute |
| C768/T4 | proj | 512 | 768 | 768 | 1.9 | 2.8 | 2.0 | **2.8** | 29.5 | **9.6%** | memory |

fp16 reference, at the one shape where the trace separates the two projections by kernel name:

| | roofline | measured | % of its own roofline |
|---|---|---|---|
| fp16 qkv (1×1 conv, fp16→3×fp16) | 289.3 | 617.5 | **46.8%** |
| fp16 out proj (cuBLAS, no residual) | 144.6 | 191.6 | **75.5%** |

So the honest statement of the root cause is: **the hand-written AWQ GEMM reaches 10–52% of its own
roofline where cuBLAS fp16 reaches 75% of its.** Why: `GWQ_CTA_K = 64`, so K=192 is **three** K-tiles
against a **3-stage** software pipeline — the pipeline is all prologue, with essentially no steady
state to amortize the shared-memory staging, the swizzle setup and the epilogue over. The kernel has
one fixed tile config (`GWQ_CTA_M/N/K = 128/128/64`, `csrc/kernels/linear/gemm_wxax.cu:104-106`) tuned
for the large-K LLM shapes it was ported from. At the tiny shapes (C768/T4, M=512) the cause is
different and not fixable by tiling: 4 M-tiles × 6 N-tiles = 24 CTAs on 84 SMs.

## 4. How much is on the table, and where

Target = the fraction of its own roofline that **fp16 already achieves on the same shape**
(46.8% qkv, 75.5% proj) — a demonstrated-reachable bar, not 100%.

| shape | kernel | measured | target | × instances | saving |
|---|---|---|---|---|---|
| C192/T1024 | qkv | 604.9 | 309.0 | 5 | **1479 µs/step** |
| C192/T1024 | proj | 345.7 | 239.5 | 5 | 531 µs/step |
| C384/T256 | qkv | 302.7 | 206.9 | 5 | 479 µs/step |
| C384/T256 | proj | 185.2 | 119.7 | 5 | 327 µs/step |
| | | | | **TOTAL** | **2817 µs/step = 2.82 ms/step** |

**≈ 4.0% of the 71 ms/step int8 baseline.** The three small shapes would add another 0.55 ms/step at
the same target, but they are occupancy-bound rather than tile-bound and no tile choice reaches it —
**not counted, and not to be quoted.**

Two things changed versus the first draft: the number is 3.3× larger (−2.82 vs −0.85 ms/step), and
**the prize is in qkv (1.96 ms/step), not the out projection (0.86 ms/step)** — the opposite of what
the withdrawn "out projection is 1.8× slower" claim pointed at.

## 5. What to do about it — the fix is dictated by the epilogue, not by the tile

Two candidate mechanisms were considered: **(1)** add tile candidates + an autotuner to the
hand-written AWQ GEMM, copying conv's `num_tuned_configs` / `_tuned(config_id)` pattern; **(2)** route
the projections to CUTLASS and let it pick the tile.

**They converge, and both advertised advantages are false.**

- conv's "8 tile configs" **are 8 CUTLASS instantiations** — `DequantFp16ConvConfig<GemmShape<…>,
  GemmShape<…>, Stages>` typedefs (`csrc/kernels/conv/conv2d_int8.cu:453-461`) behind a
  `switch (config_id)` (`:502-506`), with the timing loop in Python
  (`integration/kernels/int8_optimized.py:1005-1035`). So option 1 does not avoid CUTLASS.
- CUTLASS's **device** API — the only one in this tree — instantiates one tile per template. "Let it
  pick per problem" happens in the CUTLASS profiler or in a heuristic you write yourself. So option 2
  does not avoid writing an autotuner.

Both therefore land on the same artifact: *a tile family plus an autotuner*. The only real choice is
whether the family holds hand-written AWQ instantiations or CUTLASS ones — **and that is decided by
whether CUTLASS can express the epilogue:**

| target | epilogue | expressible in CUTLASS/EVT? | route | prize |
|---|---|---|---|---|
| **out proj** `gemm_w8a8_kernel_awq` | dequant + bias + residual → fp16 | **yes** — this is exactly `DequantFp16ConvConfig`'s epilogue | CUTLASS | 0.86 ms/step |
| **qkv** `gemm_w8a8_kernel_awq_out_i8` | emit int8 Q/K/**Vᵗ** in flash layouts | **no** — Vᵗ is a layout transform, not an elementwise op | must stay hand-written → option 1 | 1.96 ms/step |

### The cheap route for the out projection

The fp16 path already runs its qkv as a 1×1 conv (`cutlass ImplicitGemmConvolution` in the trace) —
and `conv2d_int8_dequant_fp16_tuned` (`conv2d_int8.cu:476`) takes a **channels-last** `[N,C,H,W]`
(`CHECK_CONTIGUOUS` is `is_contiguous(ChannelsLast)`, `csrc/common.cuh:11`). Viewing `[B·T, C]` as
`[B, C, 1, T]` channels-last is byte-identical — a **metadata-only reshape, zero copy**. R=S=1,
K=192 satisfies `K % 8 == 0`. So the tuned tile family and its autotuner are reachable with no new
CUDA kernel. Costs: a one-off weight repack to `[K,1,1,C]` (`qweight` is `[K, Cin]` today,
`integration/kernels/wxax_linear.py:69`), and — the real one — **the tuned family has no
bias+residual variant**, so it needs extending to the EVT bias+residual config (≈8 more typedefs plus
a switch arm, at the cost of compile time and `.so` size). Without that the residual falls back to an
eager add, which gives back exactly the 1.33× int8 currently wins by (§2).

### The expensive route for qkv, and its risk

Templating `GWQ_CTA_M/N/K` (`gemm_wxax.cu:104-106`) is not a three-line change:

- the 4 staging macros `GWQ_LOAD` (`:212`), `GWQ_LOAD_O` (`:308`), `GWQ4_LOAD` (`:856`),
  `GWQ4O_LOAD` (`:1095`) reference the tile macros directly and must be parameterized;
- smem is **exactly 48 KiB** at the current config (`2·3·128·64`, per the in-source comment at `:109`)
  — any larger tile needs `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` and dynamic smem, a code
  path CUTLASS handles for conv and this kernel does not have;
- `int acc[CTA_M/16][WARP_N/16][8]` is already 128 accumulator registers per thread, so candidate
  tiles are register-budget-constrained, not free;
- the swizzle `ld_col ^ ((ld_row / 2) & 3)` (`:125,137`) stays **correct** under any CTA_K — writer and
  reader share the formula and XOR is an involution — but it is only bank-conflict-free *by design* at
  CTA_K=64 (four 16 B chunks); other CTA_K needs re-checking under ncu.

And the honest risk: **the fix may not be in the tile space at all.** Three K-iterations against a
3-stage pipeline is a *structural* mismatch; the shapes that would help (CTA_K = K, one shot,
STAGES = 1–2) are not members of the AWQ tile family. That would be a new kernel structure, not a new
number in a list. Unverified either way.

### Order

1. **Out projection via the tuned conv path** — cheapest, and the only cheap way to test the premise
   that the roofline gap is recoverable at all. 0.86 ms/step.
2. **qkv tile family + autotuner inside the AWQ kernel** — the actual prize, 1.96 ms/step, and the
   only route to it, because its layout-emitting epilogue must stay hand-written.
3. **GroupNorm (110.9 vs 254.6):** left as an open item, but note it inherits §2's problem — fp16's
   111.0 µs is a stats-only pass while int8's 254.6 µs does stats + apply + SmoothQuant + quantize +
   int8 write. It is not a 2.3× regression; it is not a like-for-like pair at all, and the fp16 apply
   cannot be located in this trace. Needs a direct kernel A/B, not this data.
4. **Nothing for the core.** 1.32× at T=1024 for a hand-written int8 flash kernel against PyTorch's
   flash is a reasonable place to stop.

## 6. Corrections log

**First bucketing put fp16's attention core into "out projection"** because
`pytorch_flash::flash_fwd_kernel` does not contain the string `flash_attn`. That made fp16's
projection read 2399 µs and would have supported the opposite conclusion — that int8's projections are
2.5× *faster*. Buckets are now keyed on the kernel names actually observed, listed at the top of
`scripts/analyse_attn.py`.

**Second pass, the four items in the box at the top.** The common cause of all four: comparing a
*fused* int8 kernel against an *unfused* fp16 one, stage by stage, in a table whose fp16 column had
never been checked for balance. The lesson that generalizes — when one side fuses and the other does
not, per-stage attribution is not a measurement, and the sanity check that catches it is a byte count
against peak bandwidth, which takes about a minute.

**A limitation that remains.** In fp16 both projections can be 1×1 convs sharing the
`cutlass ImplicitGemmConvolution` name, so at some shapes they cannot be separated by name (C384/T256
shows fp16 qkv = 428.4 and out = 121.2 for this reason; C384/T64 and C768/T16 put both fp16 GEMMs into
the "out projection" row and 0 into qkv). C192/T1024 does separate them — different kernels — and it
is the only shape any fp16 comparison above rests on.

**Not measured here.** Calibration used the un-suffixed (stub-derived) scale files, deliberately: this
is a timing run, scale *values* do not affect kernel selection or duration, and no accuracy number is
reported. Any quality claim needs the `_realckpt` files.
