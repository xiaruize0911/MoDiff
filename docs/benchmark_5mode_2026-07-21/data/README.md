# Datasets — MoDiff 5-mode per-step kernel benchmark (2026-07-21)

All raw measured results. `dataset.json` bundles every CSV below under its filename key.

| file | rows | what it holds |
|---|---|---|
| `kernel_shapes.csv` | mode × family × distinct shape | **Ground truth.** Every conv/linear/attention shape the real model dispatches, with `count_per_step` (steady-state calls per DDIM step) and `count_first_step`, `kernel_class` (which kernel runs), `n_layers`. Source for all shape lists + counts. |
| `e2e_speed.csv` | 5 modes | DDIM wall ms/step, min ms, speedup vs fp16 (b128, 5×200 timed steps). |
| `e2e_timing_profile.csv` | 5 modes | `torch.profiler` CUDA self-time bucketed by kernel (ms/step) + `gpu_busy` + independent `wall`. |
| `conv_kernel_speed.csv` | 33 geometries + TOTAL | Per-call µs and per-step µs (`_us_per_step = count × µs`) for each conv geometry in all 5 modes, `quant_eligible`, `*_vs_fp16`. TOTAL_PER_STEP row = summed per-step time per mode. |
| `linear_kernel_speed.csv` | 14 shapes + TOTAL | qkv/proj + time-embed linear GEMMs: fp16 vs int8/int4 `full` (quantize+GEMM) and `gemm` (GEMM-only), per-step counts, speedups. No modiff variant. |
| `attn_kernel_speed.csv` | 5 blocks + TOTAL | Attention block (GroupNorm + quantize + attention): per-call GN/quant/attn µs, fp16/int8/int4 totals, per-step totals, rel-L2 vs fp32. |
| `perstep_summary.csv` | 3 families + SUM | Per-DDIM-step time (ms) of each kernel family in all 5 modes (`count/step × µs/call`). |
| `fuse_gn_qkv_quant.csv` | 5 qkv blocks + TOTAL | §7 microbench: qkv front-end (GN+quantize+GEMM) non-fused vs GN→quantize-fused, per-call + per-step, with rel-L2 correctness. |
| `fuse_gn_qkv_e2e.csv` | 1 (int8_baseline) | §7 in-model: fused GN→qkv-quantize (`MODIFF_FUSE_GN_QKV_I8`) OFF vs ON — output rel-L2 and e2e ms/step. |
| `dataset.json` | — | All of the above bundled as JSON. |
| `*.log` | — | Raw stdout of each run (enumerate / conv / e2e). |

**Modes:** `fp16`, `int8_baseline`, `int4_baseline`, `int8_modiff`, `int4_modiff`.
`int8/int4` use the latest fused kernels (deep-fuse CUTLASS conv, fused-flash quantized attention, W8A8/W4A4
AWQ linear). `_modiff` adds the temporal-delta conv cache. Linear and attention have no modiff variant
(static W/A quant in every mode) → `int8_baseline == int8_modiff`, `int4_baseline == int4_modiff` for those.

GPU: NVIDIA A40 · torch 2.4.1+cu124 · b128 · LSUN-Churches LDM-8 UNet · DDIM.
