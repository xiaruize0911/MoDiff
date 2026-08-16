# Open items, 2026-08-16

Standing list, three sections: **values that are not what a committed doc says they are**, **questions
with no answer yet**, and **headroom with a known lever**. Every row cites the file it comes from, and
every number is read from committed data rather than restated from memory.

Provenance: sections B and C are carried forward from the `Open` sections of
[`profile_kernels_layers_2026-08-11`](profile_kernels_layers_2026-08-11/FINDINGS.md),
[`aq_fusion_2026-08-12`](aq_fusion_2026-08-12/FINDINGS.md),
[`fid_2026-08-05`](fid_2026-08-05/FINDINGS.md),
[`static_qdiff_2026-08-12`](static_qdiff_2026-08-12/FINDINGS.md) and
[`zp_coverage_2026-08-13`](zp_coverage_2026-08-13/README.md). Section A rows marked **[2026-08-16]** were
found by re-reading `bench_report_2026-08-13_postzp/data/kernel_suites.json` against the prose it
generated; A1–A6 are **fixed** as of this doc and the corrected numbers are in
[`KERNEL_SPEEDUP.md`](bench_report_2026-08-13_postzp/KERNEL_SPEEDUP.md) and
[`SUMMARY.md`](bench_report_2026-08-13_postzp/SUMMARY.md).

---

## A. Not the expected value

Measured numbers that contradicted a claim a committed doc made about them. Ordered by how much of the
published story they move.

| # | claim as published | what the data says | state |
|---|---|---|---|
| **A1** **[2026-08-16]** | linear reads 0.61×/0.67× because "in fp16 the attention projections are 1×1 convs" counted in the conv suite, and `conv + linear` cancels it | The conv suite matches **100% three ways** — 33/33 records, fp16 265.72 = matched 265.72, unmatched **0.00** in all three arms. Nothing moves conv↔linear. fp16's T=1024 and T=256 qkv run `fused_gn_qkv`, which `suite_of()` dropped into **`other`**: **31.96 ms/sample**. `conv+linear` 1.50×/2.28× was computed against an fp16 side missing that work | **fixed** |
| **A2** **[2026-08-16]** | `other` 3.77×, `norm_quantize` 0.64× | Same misbucketing, mirrored: fp16's fused kernel also absorbs the GroupNorm whose quantized counterpart lives in `norm_quantize`. All three ratios are accounting artifacts | **fixed** |
| **A3** **[2026-08-16]** | attention T=1024 nets 1.19× (int8) / 1.16× (int4) because "the flash kernels take a padded head dim" | 1.21× / **1.21×** call-weighted. Padding explains int8. It does not explain int4 at all: every operand in the int4 arm is `torch.int8`, the dominant hd24 route's profiled kernel is `flash_attn_int8_mma_kernel_t`, V stays int8 (`gemm_w4a4_awq_qkv_i4qk_i8v_layouts`), and hd=24 pads to **64 int4 = the same 32 B/row** as int8's pad-to-32 | **fixed** |
| **A4** **[2026-08-16]** | T≤16 "falls back to `torch_sdpa_fp16` in every arm — correctly" at 1.00×/0.99× | **0.88× / 0.86×**. Only 15 of 25 calls fall back; the other 10 run `flash_attn_int8_qi8packed_small_qout` at 64.7 µs against sdpa's 47.8 µs | **fixed** |
| **A5** **[2026-08-16]** | "5 of 8 blocks matched in all three arms" | `attn_key` keyed on Q, and the `_qout`/hd24 kernels take **token-major** Q `[N,T,H,hd]` → 3 phantom keys. The T=1024 row compared fp16's 25 calls against int8's **15**, silently omitting `flash_attn_int8_qi8_kv_static_qout_hd24` — the kernel the same document describes as ~31% of the suite. µs were also an unweighted mean over heterogeneous kernels | **fixed** |
| **A6** **[2026-08-16]** | int4 attention's one real win, T=256 at 1.66× | Both contributing records carry `stability: NOISY` | **fixed** (flagged in the table) |
| **A15** **[2026-08-16]** | `other` compared arm-to-arm | fp16 is missing two `cat2_channels_last_fp16` signatures the quantized arms both have (2.64 + 0.34 ms) and captured 5 calls where int8 captured 10 on `[128,384,16,16]²`. So `other` is not a clean comparison even after A1 | **open** — capture-coverage asymmetry, needs a GPU re-capture |
| **A7** | MoDiff's temporal machinery costs 2.8% (W8A8) / 1.1% (W4A4) | Arm **order** alone moves W4A4 MoDiff by 28% and PTQ by 7–9%, and both committed values are second-arm values ([FINDINGS_NOISE_FLOOR](zp_coverage_2026-08-13/FINDINGS_NOISE_FLOOR.md)) | **open** — not resolvable on this axis; read 1–3% as "roughly free", not as a figure |
| **A8** | the GroupNorm+SiLU family should track precision | 1.15× at W8A8 and **1.13× at W4A4** — slightly *slower* with fewer bits, while growing to 32.2% of the run | **open**, unexplained |
| **A9** | MoDiff is the quality answer | True at W8A8 (97.3% of the quantization error removed, FID 7.802 vs fp16's 7.803). At W4A4 it removes **31.3%** and FID is 200.1 vs PTQ's 278.0 — the dominant error is in the **weights**, which an activation method cannot reach ([fid_2026-08-05](fid_2026-08-05/FINDINGS.md)) | **diagnosed**, not fixed → B5 |
| **A10** | fix #2 (activation zero point) is a quality lever | Reversed twice, closed **negative**: 1.06× ceiling against a 1.15× bar. The obstacle is per-output-**pixel** zero padding, unfoldable into a per-channel bias ([zp_coverage](zp_coverage_2026-08-13/FINDINGS.md)) | **closed** |
| **A11** | fix #4 (weight zero point / AdaRound) was deprioritised | Deprioritised on ‖W−Q(W)‖, the one metric AdaRound is willing to lose. On conv output error it wins 1.35×; end to end, weight-only, **1.58×** ([FINDINGS_WEIGHT_ZP](zp_coverage_2026-08-13/FINDINGS_WEIGHT_ZP.md)) | **open** — reprioritise, → C6 |
| **A12** | the csrc/ split bought −1.3 ms | The offset appears on `int8_ptq`, an arm containing no MoDiff code the change could reach. Cross-session container drift, not code ([postsplit](postsplit_benchmark_2026-08-12/FINDINGS.md)) | **closed** as artifact; same-session references reproduce to 0.21 ms |
| **A13** | the `aq_*` fusion is blocked because "the non-qout siblings were deleted" | Three non-`_qout` int8 entry points are still exported and declared, including `flash_attn_int8_packed_vt` — exactly the configuration the note said no longer exists | **closed**; blocker was stale → C2 |
| **A14** | hd=24 wants an 8-byte loader | Built and **refuted**. The remaining 3.13 ms needs a gather that beats the mma kernel at T=1024 — a new kernel, not a wider load | **closed** → C3 |

---

## B. Unsolved

1. **W4A4: per-step vs constant delta.** Settled *flat* at W8A8 — 255 levels absorb a 6.5× step-to-step
   swing and the paper's single scalar is not a compromise. At W4A4 the same measurement was standing on
   a bug and was never redone. [static_qdiff_2026-08-12 §4](static_qdiff_2026-08-12/FINDINGS.md)
2. **FID for W8A4 + MoDiff** — the one row directly comparable to the paper's table. ~30 min.
3. **FID for W4A4 + MoDiff with the new weight scale** — −7.5% relL2 at a point where the relL2→FID curve
   is very steep; the effect on FID is unknown and could be either sign.
4. **FID at 50k**, for publication-grade absolute numbers. ~1.5 h/mode, and 68–78% of that is not the
   UNet. [FID_50K_ESTIMATE](bench_report_2026-08-13_postzp/FID_50K_ESTIMATE.md)
5. **A weight-side method for W4A4.** No candidate has landed. This is what gates W4A4 being usable at
   all — the kernels are already there (3.83× median per conv). Follows from A9.
6. **Quality of the qkv-i8 fusion.** ±2.5% per-seed swings at 3 seeds. Making it default rather than
   flag-gated needs more seeds; the 8-seed lesson from `act_bits_2026-08-05` is that a 3-seed mean can
   reverse sign.
7. **Part 3, the a_hat-aware flash qout epilogue: the first gate is a numerics decision, not code.** The
   delta scale must either come from a previous step's `report_next` or be accepted one step stale, and
   the relL2 cost of stale is unmeasured. Ceiling 6.7 ms, and it only pays at A8/A7 — at A4 the
   projections are already a 0.976×/1.014× proposition.
8. **The dual-output GEMM's signature.** Whether `gemm_w8a8_awq_o_hat_out_i8` accepts a per-column out
   scale or only the scalar `a_scale` its `TORCH_CHECK`s mention. This decides whether C2 is wiring or a
   kernel change, and it is the next thing to read.
9. **Per-layer profile coverage gap, 12–36%.** Closing it needs the ResBlock's own `forward` timed as a
   residual bucket.
10. **The PTQ attn/proj split** — needs `_flash_proj_qout` instrumented to separate the projection GEMM
    from the score path.
11. **Whether to build the per-output-pixel windowed reduction epilogue.** One epilogue unlocks both
    remaining quality levers; the measurements say build it for fix #4 (1.58×) and not for fix #2
    (1.06×).

---

## C. Headroom with a known lever

| # | target | size | lever | blocked on |
|---|---|--:|---|---|
| **C1** | GroupNorm+SiLU family | **32.2% of W4A4 at 1.13×** | The next real lever, and the ceiling has moved to it: at W4A4 the matmuls are 39.9%, so even a *free* conv only takes 57.85 → 34.8 ms/step | no design has landed |
| **C2** | the `aq_*` trio | **~4.60 ms** + part of the elementwise bucket | `gemm_w8a8_awq_o_hat_out_i8` → reshape (free — the qkv weight is already stored `(nh,3,hd)`) → `flash_attn_int8_packed_vt`; the three scales express as one 3C-long per-column vector | B8 only. **No CUTLASS work.** Largest actionable item |
| **C3** | attention T=1024 / hd=24 | 15.6 ms/sample, ~31% of the attention suite, at 1.21× | Needs a gather that beats the mma kernel at T=1024. The padding is structural to the MMA fragment layout, not a missing optimization, and the 8-byte loader is refuted (A14) | new kernel, no design |
| **C4** | int4 attention | zero gain today | V is int8 and the MMA is the int8 path in every arm (A3). Nothing to win until there is a real int4 datapath | design |
| **C5** | GN-stats epilogue, Stage C | — | Needs a reduction that wins on `768×4×4` (still 1.54×) rather than only in the weighted average — a real epilogue pays this on top of its existing work, not instead of a separate launch | measurement |
| **C6** | weight zero point (fix #4) | **1.58×** end-to-end, weight-only | per-output-pixel windowed reduction epilogue | B11 |
| **C7** | MoDiff warm-up | +663 ms (W8A8) / +615 ms (W4A4) per **cold sample** = 4–5% at 200 steps, **17–20% at 50** | `_forward_first_step` runs 5 convs where a steady step runs 1. Every quality harness pays it 70× because a stale `a_hat` cache produces NaN latents and it must reset | design |
| **C8** | 70 orphaned int4 wrappers | **114 MiB** | Measured to 0.7% by two independent methods; not reclaimed | — |
| **C9** | `_qout` under MoDiff | all 21 blocks report `qout_eligible == 0` | Mutually exclusive with MoDiff's fp16 o_hat state | subsumed by C2 / B7 |

---

## What is settled and should not be re-litigated

So the list above is read against a fixed background, not an open one:

- **W8A8 + MoDiff is the result.** FID 7.802 against fp16's 7.803 at 1.41×, 1.95/255 mean paired pixel
  distance, 97.3% of the quantization error removed.
- **Nearly a third of the end-to-end gain is fusion, not low precision.** The elementwise/copy bucket
  falls 3.42× and is *identical* at W8A8 and W4A4.
- **Going 8→4 bits buys exactly one bucket**: 2643 of the 2674 ms saved is GEMM/conv, 98.8%.
- **Per-conv-layer int4 is 3.83× median** (14 layers, arithmetic-only subset), with 7 unquantized
  controls at 1.00× proving the layout normalization matches real layers.
- **The paper reproduces**, and two swept calibration constants (`DELTA_CLIP_RATIO = 8`,
  `ACT_CLIP_RATIO = 4.5`) closed most of the W4A4 gap without a kernel change.
- **Suite totals are not a valid speedup denominator** (A1/A2). The load-bearing tables are the
  per-layer matched conv table and the full-run profile buckets.
