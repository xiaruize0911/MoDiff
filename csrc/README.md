# MoDiff CUDA kernels (`csrc/`)

`csrc/` is being split into **two trees, one per datapath**, so each can be read, edited and profiled
without the other in the way:

```
csrc/
├── baseline/     W8A8 / W4A4 post-training quantization. No temporal state. 11 .cu, 110 exports.
│   ├── common/{common.cuh, mma_int8.cuh}      duplicated device headers
│   ├── attention/{flash_attn_int8, attn_quant_gemm}.cu
│   ├── conv/{conv2d_int8, conv2d_int4, conv2d_evt, conv_epilogue}.cu (+ 2 duplicated headers)
│   ├── linear/gemm_wxax.cu   norm/{group_norm_silu, fused_gn_qkv}.cu (+ 2 duplicated headers)
│   └── quantize/quantize.cu  util/layout_transform.cu
├── modiff/       MoDiff: the temporal delta datapath (a_hat / o_hat across timesteps). 8 .cu, 29 exports.
│   ├── common/{common.cuh, mma_int8.cuh}      duplicated device headers
│   ├── conv/{conv2d_int8, conv2d_int4, conv2d_evt, conv_epilogue}.cu (+ 2 duplicated headers)
│   ├── linear/gemm_wxax.cu   norm/group_norm_silu.cu (+ 2 duplicated headers)
│   └── quantize/delta_quantize.cu  util/layout_transform.cu
├── pybind.cpp            every Python entry point -- the ONE file where both datapaths meet
└── modiff_kernels_api.h  C++ declarations for all of them (deliberately NOT split; see below)
```

**Migration status (2026-08-12): COMPLETE. All six families migrated; `csrc/kernels/` is deleted.**
Every `.cu` now lives in exactly one tree, and each tree compiles against its own copy of the shared
device headers. Verified by poisoning `baseline/common/mma_int8.cuh` with an `#error` and confirming
that only the two baseline translation units that include it failed, while `modiff/linear/gemm_wxax.cu`
compiled clean against its own copy — the copies are load-bearing, not decoration.

## What separates the two datapaths

The distinction is **not** precision (both trees have int8 and int4) and **not** the operation. It is
whether the kernel touches MoDiff's cross-timestep state:

| | reads/writes `a_hat` or `o_hat` | example |
|---|---|---|
| **`modiff/`** | yes | `gemm_w8a8_awq_o_hat` — GEMM whose epilogue accumulates `o_hat_t = A·Q(δ) + o_hat_{t+1}` |
| **`baseline/`** | no | `conv2d_int8_fprop_no_ohat` — the same conv with the accumulate deleted |

`a_hat` is the previous step's quantized activation (so this step can quantize only the delta);
`o_hat` is the previous step's output accumulator (Eq 9). A kernel that carries either belongs to
`modiff/`. Most of the tree already encodes this in its names: `*_o_hat` versus `*_no_ohat` /
`*_noahat`.

### The dual-purpose kernels, and why they get copied

38 exported kernels touch that state. Which of them are genuinely dual was settled from the **call
sites**, not the signatures (verified 2026-08-12):

| kernel | MoDiff caller | baseline caller | verdict |
|---|---|---|---|
| `step1_static_quantize_fprop` | `wxax_linear.py:243` (`self.a_hat`), `int8_optimized.py:1519` (`self.a_hat_cache`) | `int8_optimized.py:1362` — **passes `self._zero_ahat_buf`** | **dual** |
| `step1_static_quantize_pack_int4_fprop` | `wxax_linear.py`, `int4_optimized.py` | same zero-buffer pattern | **dual** |
| `step1_quantize_fprop`, `_pack_int4_fprop` | `int8_optimized.py`, `int4_optimized.py` | — | check at migration |
| `step1_static_quantize_fprop_silu`, `_pack_int4_fprop_silu` | `int8/int4_optimized.py` | — | check at migration |
| `dequant_accumulate_and_return_int8` / `_int4` | `int8_linear.py`, `int4_linear.py` | — | check at migration |
| `dequant_accumulate_int8` / `_int4` | none found | none found | **no Python caller at all** |

**How the baseline fakes it, and what that means for the copy.** The baseline conv path does not pass an
*empty* state tensor — it passes a **zero** `a_hat` buffer and lets the kernel subtract it. So the
baseline copy could drop the parameter and the subtraction entirely, which would also stop it reading a
zero buffer it does not need. **That is an optimization, not a move**, and it would change that kernel's
SASS, so the migration keeps the copy byte-identical and leaves the saving as a separate, measurable
change. Recorded here rather than done silently.

Two of the four `dequant_accumulate*` exports have **no Python caller anywhere** in `integration/` or
`docs/`. They are candidates for deletion rather than duplication; confirm against `pybind.cpp` before
removing, since the export manifest will flag the change either way.

Those are **duplicated, one copy per tree, with the tree in the name** (`modiff_*` / `baseline_*`).
That is a deliberate trade:

* **for**: no hidden coupling. A call site says which datapath it is on, and an un-migrated call site
  fails loudly at import (`AttributeError`) instead of silently taking the other path.
* **against**: two copies can diverge, and every A/B in `docs/` compares the two datapaths, so a
  numerical edit applied to one twin and not the other invalidates those comparisons. Duplicated
  files carry a banner naming their twin and a `diff` command; keep them passing it.
* **also against**: compile time and `.so` size grow roughly with the duplicated families. The build
  is ~2–3 min with `ninja`; without `ninja` torch falls back to a serial backend and it is >20 min.

## Build-cost baseline (2026-08-12, pre-migration)

Recorded so the `conv/` family's duplicated CUTLASS instantiations can be costed against something
rather than asserted:

| | value |
|---|---|
| clean `python setup.py build_ext --inplace`, `ninja` present | **246 s** pre-split (12 TUs) → **480 s** post-split (20 TUs), **1.95×** |
| `modiff_cutlass...so` | 26,480,696 B pre-split → 27,116,888 B post-split (**+2.4%**) |
| `build/` | 83 MB → 92 MB |
| device kernels in the binary | 279 unique (289 `Function` entries; 10 templates instantiated in two TUs) |

Without `ninja` the same build is >20 min: torch falls back to a serial backend. Check for it first.

## Classification: 130 exported kernels

| class | count | rule | destination |
|---|--:|---|---|
| baseline-only | ~92 | never sees `a_hat`/`o_hat` | `baseline/` |
| MoDiff-only | ~26 | named for or requires the state | `modiff/` |
| dual-purpose | ~12 | optional state tensor; both paths call it | **both**, renamed |

The authoritative list is generated, not maintained by hand — the classifier reads `pybind.cpp` for
what is callable and `modiff_kernels_api.h` for whether the signature carries state. Regenerate with
the snippet in "Regenerating the map" below rather than editing a table here, because a hand-kept list
of 130 names goes stale silently. (The first version of that classifier called
`conv2d_int8_fprop_o_hat` "baseline" because the header names that parameter differently; the rule now
also reads the kernel's own name.)

## Per-family migration plan

Ordered so the build stays green after every step, and so the family holding most of the dual set goes
first while the pattern is still cheap to change.

| # | family | now | split | notes |
|--:|---|---|---|---|
| 1 | `quantize/` | `quantize.cu` (10 host fns), `modiff_delta_quantize.cu` (19) | `baseline/quantize/`, `modiff/quantize/` | already nearly split by file; **holds most of the dual set** (`step1_*`) |
| 2 | `linear/` | **DONE** — `baseline/linear/gemm_wxax.cu` (1508 L, 16 exports), `modiff/linear/gemm_wxax.cu` (795 L, 3 exports) | the 3 `*_o_hat*` host fns moved; the 3 GEMM kernels + `GWQ_*` constants + `gwq_s2r_A/B`/`gwq_store2` **copied** as `static` (the kernels are dual-purpose — baseline launches them with `o_hat == nullptr`) |
| 3 | `norm/` | **DONE** — `baseline/norm/{group_norm_silu,fused_gn_qkv}.cu` (1551 + 436 L, 10 exports), `modiff/norm/group_norm_silu.cu` (2033 L, 4 exports) | **zero shared kernels**: 16 kernels + `gn_launch_group_stats` reach only from the delta entry points, 6 kernels + the two `*_impl` helpers only from baseline. Only the small dtype-dispatch helpers (`gn_load`/`gn_load2`/`gn_store2`, `gns_silu`, `gn_report_delta_absmax`) are copied. SASS gate passed with **no re-baseline** |
| 4 | `conv/` | **DONE** — `baseline/conv/` (4 files, 2325 L) + `modiff/conv/` (4 files, 1033 L) | the 8 `*_o_hat*` exports moved; **the int8 and int4 CUTLASS conv Op instantiations, `conv2d_intX_fprop`, the EVT preamble and `make_problem` are COPIED**. Cost measured: clean build **246 s → 480 s (1.95×)**, `.so` 26,480,696 → 27,116,888 B (+2.4%), `build/` 83 → 92 MB |
| 5 | `attention/` | **DONE** — both files whole to `baseline/attention/` (2777 + 1144 L, 36 exports) | **0 of 36 host fns reference `a_hat`/`o_hat`** — attention is stateless in both datapaths, so nothing was duplicated and nothing went to `modiff/`. Each file carries a DATAPATH NOTE explaining that MoDiff's involvement is *which entry point it may call* (`_qout` unusable under MoDiff; `packed_vt` is route (b)'s), decided in Python. SASS gate passed with **no re-baseline** |
| 6 | `util/` | **DONE** — `baseline/util/layout_transform.cu` (460 L, 4 exports), `modiff/util/layout_transform.cu` (324 L, 1 export) | `fp16_ncw_delta_to_int8_cl` + its 4 delta kernels moved; nothing shared but the `TILE_T` #define. SASS gate passed with **no re-baseline** — a true pure move |

**`modiff_kernels_api.h` is deliberately NOT split.** It was attempted and reverted. The partition
itself is exact — scanning the definitions in each tree gives 110 declarations owned by `baseline/`,
29 by `modiff/`, and zero overlap — but it is a declaration-only header included by `pybind.cpp`, and
`pybind.cpp` is already the single place where both datapaths' entry points meet. Splitting it buys
documentation while risking a silently dropped declaration, which would remove an export without any
compile error. The `test_export_manifest.py` gate would catch that, but the trade is still poor: no
code moves, and one more file to keep in sync. `csrc/common.cuh` at the top level WAS deleted, since
after the migration nothing included it (every `.cu` uses its own tree's `common/common.cuh`).

## Family 1 work order: `quantize/`

Generated from the sources (host function → the `__global__` kernels it launches), so the split is a
move-list rather than a judgement call. `STATE` = the body references `a_hat`/`o_hat`.

**`kernels/quantize/quantize.cu` → `baseline/quantize/quantize.cu`** — all 10 host functions, verbatim:
`quantize_and_pack`, `scale_quantize_and_pack`, `scale_quantize_int8`, `dequant_bias_i8`,
`quantize_attn_out_int8`, `quantize_attn_out_int4_pack`, and the four `dequant_accumulate*`.
The `dequant_accumulate*` four are STATE by body but **two of them have no Python caller at all**
(`_int8`, `_int4`); the two `_and_return_` ones are called from `int8_linear.py` / `int4_linear.py`.

**`kernels/quantize/modiff_delta_quantize.cu` (19 host fns, 24 kernels) splits in two.** To `modiff/`:

| host fn | launches |
|---|---|
| `sub_absmax_scale`, `compute_dynamic_scale` | `sub_absmax_scale_kernel` |
| `delta_absmax_fp16` | `delta_absmax_fp16_kernel` |
| `dynamic_quantize_int8_fprop`, `dynamic_quantize_pack_int4_fprop` | (compose the above) |
| `step1_quantize_fprop` | `quantize_and_update_ahat_kernel` |
| `step1_static_quantize_fprop` **(dual)** | `static_quantize_and_update_ahat_kernel_int8*` |
| `step1_static_quantize_fprop_silu` | `..._int8_half_cache_silu` |
| `step1_quantize_pack_int4_fprop` | `quantize_pack_and_update_ahat_kernel_int4` |
| `step1_static_quantize_pack_int4_fprop` **(dual)** | `static_quantize_pack_and_update_ahat_kernel_int4*` |
| `step1_static_quantize_pack_int4_fprop_silu` | `..._int4_half_cache_silu` |

**Family 1b: DONE (2026-08-12).** All eight `*_noahat` / `*_no_ahat` functions now live in
`baseline/quantize/quantize.cu`, with their private kernels moved and every shared dependency
**copied** rather than referenced across trees:

*Moved verbatim, no shared dependency (6):* `step1_static_quantize_noahat_fprop`
(`static_quantize_int8_noahat[_vec2]_kernel`), `step1_static_quantize_pack_int4_noahat_fprop`
(`static_quantize_pack_int4_noahat[_vec2]_kernel`), `upsample2x_quantize_noahat_fprop` and `_pack_`,
`avgpool2x_quantize_noahat_fprop` and `_pack_`.

*Moved, with a copied dependency (2):* `step1_quantize_no_ahat_fprop` and
`step1_quantize_pack_int4_no_ahat_fprop`. Each takes an `a_hat_cache` argument and calls the MoDiff
host function `sub_absmax_scale`. Chosen resolution — option (a) from the prior note, per the
explicit instruction to duplicate every reused function rather than leave a cross-tree reference:
`sub_absmax_scale` and `sub_absmax_scale_kernel` are **copied** into `baseline/quantize.cu` as
`static` (file-local) twins; the originals stay in `modiff/delta_quantize.cu`, unchanged and still
exported under their own names. `load_as_float`/`load_as_float2` (generic device helpers used by
both MoDiff-side and baseline-side kernels) and `avgpool4_as_stored` are copied the same way.

Verified: both dual functions checked bit-exact against a hand-computed reference
(`x - a_hat_cache` → absmax scale → quantize) with a **real, non-zero** `a_hat_cache` — this is
stronger than the SASS-identity gate alone, since it exercises the copied `sub_absmax_scale` path
with actual state rather than only proving the move didn't touch instruction bytes.

**A parser lesson from getting the SASS gate right on this family, twice.** First, `cuobjdump`
attributes per-fatbin boundary text to the preceding kernel — fixed by keeping only lines that carry
the `/*addr*/`/`/*encoding*/` comment columns. Second (found while landing 1b): `cuobjdump`
**right-pads every instruction's comment column to a width shared across the whole dump**, so adding
code anywhere in the file can shift an unrelated, byte-identical kernel's padding by one space. Fixed
by collapsing whitespace runs before hashing. Both are recorded in `test_sass_golden.py`'s comments
so a third instance of "kernel X changed" is diagnosed by rereading that file first, not re-derived.

**A consequence worth acting on later, not now.** The baseline already *has*
`step1_static_quantize_noahat_fprop`, yet `int8_optimized.py:1362` calls the a_hat variant with a zero
buffer instead. If that call site simply used the existing no-ahat kernel, the dual set for this family
would drop to zero and no duplication would be needed at all. That is a behaviour change (a different
kernel runs, with different SASS), so it is out of scope for a migration gated on SASS identity — but it
is the correct end state and it is cheap to verify, since both compute the same thing when `a_hat == 0`.

## Speed, from the current committed measurements

**End to end, batch 128, A40, 200 steps × 5 repeats, profiler-free**
(`docs/profile_kernels_layers_2026-08-11/data/`, `docs/aq_fusion_2026-08-12/data/`):

| arm | ms/step | vs fp16 |
|---|---:|---:|
| fp16 (autocast on) | 106.09 | 1.000× |
| **W8A8 PTQ — baseline tree only** | **73.31** | **1.447×** |
| MoDiff conv only, K=4 | 77.33 | 1.372× |
| MoDiff conv only, K=1 | 83.01 | 1.278× |
| MoDiff conv+proj, K=1 (the paper's datapath) | 105.42 | 1.006× |
| MoDiff conv+proj, K=4 | 99.73 | 1.064× |
| … + projection refresh schedule (opt-in) | 95.64 | 1.109× |
| … + route (b) qkv int8→flash (opt-in) | **94.88** | **1.118×** |

**Per bucket, batch 128** (bucketed Perfetto trace, `docs/aq_fusion_2026-08-12/data/trace_buckets_qkvi8.json`,
last arm above, 91.55 ms/step of GPU time):

| bucket | ms/step | dominant kernels |
|---|---:|---|
| conv | 27.3 | two CUTLASS `ImplicitGemmConvolutionEVT` instantiations, 35 calls each |
| delta_quantize | 15.8 | `gn_apply_delta_quantize_flat_vec2` (83), `static_quantize_and_update_ahat` (21) |
| linear_gemm | 15.2 | `gemm_w8a8_kernel_awq` (32), `gemm_w8a8_kernel_awq_out_i8` (10) |
| elementwise | 11.6 | `at::native::(vectorized_)elementwise_kernel`, ~190 calls — **the largest unfused item left** |
| attention | 9.8 | `flash_attn_int8_mma_kernel_t` (5), `flash_attn_int8_packed_mma_kernel` (10) |
| norm_quantize | 8.9 | `gn_stats_partials_chanmajor` (83), `group_norm_silu_delta_quantize_resize_nhwc` (10) |
| attn_quantize | 2.9 | the surviving `aq_*` passes on the 5 hd=24 blocks |

Read shares within a column and named kernels within a capture — **not** totals across captures.
Two captures of the same arm minutes apart drift ~1 ms on buckets nothing touched.

**Post-migration re-measurement (2026-08-12), the evidence that the split changed no behaviour:**
the route (b) paired A/B reads **+0.71 ms/step** (stdev 0.073) against the pre-split **+0.79 ± 0.14**,
and the ON arm sits at **94.60 ms/step** against the recorded **94.88** — both inside run-to-run noise,
with all three kernel counters exact (10 / 5 / 10 ON, 0 / 15 / 0 OFF). Together with the per-kernel SASS
gate (279/279 identical) and the export manifest (130/130), that is the whole claim: same code, same
numbers, in two trees instead of one.

> Superseded numbers: this file used to quote 123.0 ms/step and 1.54× from 2026-07-20. Five reports
> have corrected that since (the harness's fp16 baseline was running fp32/tf32; the delta quantizer
> gained a code ceiling; the warm-up was fixed). Do not resurrect them — the table above and the
> reports it cites are current.

## What is fused, and what is not

Of the 91.55 ms/step above, roughly 78 sits in kernels that each do several operations in one launch:
conv+EVT epilogue, GN+delta-subtract+quantize+`a_hat` write, GEMM+`o_hat`+bias+residual,
QKᵀ+softmax+AV with scores never leaving SRAM. The two genuinely unfused items are
`gn_stats_partials_chanmajor` (4.7 ms) and the elementwise glue (6.7 ms).

Fusion candidates that were **built and refuted by measurement** are recorded where they were
measured, not deleted, so they are not re-proposed: `docs/aq_fusion_2026-08-12/FINDINGS.md` (route (a)
18 ms slower; hd=24 8-byte gather 2.11× against a 1.44× break-even; GN-stats in the conv epilogue
0.96×, viable mechanism but too thin a margin) and `docs/profile_kernels_layers_2026-08-11/FINDINGS.md`.

## Regenerating the map

```bash
python - <<'EOF'
import re
names = re.findall(r'm\.def\("([a-z0-9_]+)"', open("csrc/pybind.cpp").read())
api = open("csrc/modiff_kernels_api.h").read()
for chunk in api.split(";"):
    flat = " ".join(chunk.split())
    m = re.search(r"\b([a-z0-9_]+)\s*\(", flat)
    if not m or m.group(1) not in names:
        continue
    n = m.group(1)
    state = re.search(r"o_hat|a_hat|ohat|ahat|cache", flat) or re.search(r"o_hat|ohat|modiff|delta", n)
    print(f"{'MODIFF/DUAL' if state else 'BASELINE  '}  {n}")
EOF
```

Every kernel file carries a header comment with its inputs, outputs, what it fuses, and its measured
speed against the fp16 op it replaces. Those headers are the per-kernel documentation; this file is
only the map.
