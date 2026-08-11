# MoDiff CUDA kernels (`csrc/`)

`csrc/` is being split into **two trees, one per datapath**, so each can be read, edited and profiled
without the other in the way:

```
csrc/
├── baseline/     W8A8 / W4A4 post-training quantization. No temporal state.
├── modiff/       MoDiff: the temporal delta datapath (a_hat / o_hat carried across timesteps).
├── kernels/      NOT YET MIGRATED -- the original mixed tree. Shrinks as families move out.
├── pybind.cpp            every Python entry point
└── modiff_kernels_api.h  C++ declarations for all of them
```

**Migration status (2026-08-12): skeleton and shared headers in place, no kernel family moved yet.**
`kernels/` is still the tree that builds. The per-family plan and the classification that drives it are
below; nothing here is aspirational except where it says NOT YET MIGRATED.

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
| clean `python setup.py build_ext --inplace`, 12 CUDA TUs, `ninja` present | **246 s** |
| `modiff_cutlass...so` | 26,480,696 B (25.3 MiB) |
| `build/` | 83 MB |
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
| 2 | `linear/` | `gemm_wxax.cu` (19) | both | `*_o_hat`, `*_o_hat_out_i8` → modiff; plain/`bias_res`/`out_i8` → baseline |
| 3 | `norm/` | `group_norm_silu.cu` (11), `fused_gn_qkv.cu` (3) | both | `*_delta_quantize_*` and `gn_stats_from_tiles` → modiff; `group_norm_silu*`, `fused_gn_qkv` → baseline |
| 4 | `conv/` | `conv2d_int8.cu` (13), `conv2d_int4.cu` (12), `conv2d_evt.cu` (9) | both | `_o_hat*` → modiff, `_no_ohat*` → baseline; the CUTLASS instantiations get copied, which is where compile time doubles |
| 5 | `attention/` | `flash_attn_int8.cu` (25), `attn_quant_gemm.cu` (11) | mostly baseline | no flash kernel carries state. MoDiff's involvement is which *entry point* it may use: `_qout` variants are unusable under MoDiff (the epilogue and the o_hat state are mutually exclusive), and `flash_attn_int8_packed_vt` is what route (b) feeds int8 into |
| 6 | `util/` | `layout_transform.cu` (5) | both | `fp16_ncw_delta_to_int8_cl` → modiff, rest → baseline |

`setup.py`'s source list and `pybind.cpp` change with each step; `modiff_kernels_api.h` splits into
`baseline/api.h` + `modiff/api.h` last, once nothing else moves.

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
