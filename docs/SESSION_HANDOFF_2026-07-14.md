# Session handoff — MoDiff INT8/INT4 conv-speed optimization (2026-07-14)

Everything below is **committed and pushed to `origin/main`** (HEAD `b8a8811`). Rebuild
the CUDA extension before running anything:
`CUTLASS_PATH=/workspace/cutlass MAX_JOBS=$(nproc) python setup.py build_ext --inplace`
(the `conv2d_int8.cu` unit is ~5–8 min; run in the background). Always set
`PYTHONPATH=/workspace/MoDiff/src/taming-transformers:$PYTHONPATH`. Also see the memory
notes `modiff-optimization-findings.md` and `modiff-resnet-int8-chaining.md`.

## Goal
Make INT8/INT4 quantized inference actually beat fp16 — first on the LSUN-churches
latent-diffusion UNet, then (as the conv-bound testbed) on ResNet-50.

## What shipped this session (commits oldest→newest)

**Diffusion UNet fusions (churches, A40, batch 32, 100 DDIM steps):**
- `958a7e7` GN→INT8 quantize fusion (baseline): GroupNorm+SiLU emits int8 directly, removing the
  per-conv quantize K1. int8_baseline −5.6%, first time beating fp16.
- `8086696` INT4 (packed) mirror. int4_baseline −8.6%.
- `68107d4` Fold `use_scale_shift_norm` modulation into the GN→quantize kernel → coverage 39%→89%.
  int8_baseline −13.0%, int4 −16.6% cumulative.
- `900b944` Fuse ResBlock skip-add(+bias) into the conv store epilogue (baseline). int8 −15.2%,
  int4 −19.6% cumulative.
- `94d42ee` **Token-major AttentionBlock** (biggest single win, ALL modes incl fp16): run attention
  `[N,T,C]` with nn.Linear qkv/proj so the channel-major↔flash layout copies become free views.
  fp16 −12%, int8_baseline −13.8%, int4 −16.1%.
- `92e1abd` Fold scale-shift modulation into the **fp16** GN kernel (MoDiff + fp16 fallback paths).
  MoDiff int8 −14.8%, int4 −11.7% vs unfused; fp16 also −1.7ms.

**Cumulative diffusion result (all fusions on vs original unfused):** int8_baseline 36→~25.7 ms
(~15% faster than fp16), int4_baseline ~33→~22 ms, MoDiff int8 ~37→31.5, int4 ~32→28.
Kill-switch `MODIFF_DISABLE_GN_INT8_FUSION=1`, `MODIFF_DISABLE_TOKEN_MAJOR_ATTN=1`.

**ResNet-50 / kernel-quality arc (A40, batch 64):**
- `d6819ff` ResNet-50 all-modes benchmark (`integration/benchmarks/benchmark_resnet50.py`), BN folded.
  Found int8 end-to-end 0.63× (SLOWER) — a CNN has no norm to hide the per-conv quantize.
- `4c5f3bb` **INT8-native conv→conv chaining** (`Int8ChainedBottleneck` in
  `integration/fused_ops/chained_bottleneck.py` + `forward_to_int8`/`quantize_input` on
  `OptimizedInt8Conv2d` + `scale_bias_relu_requant_store_int8` epilogue): keep activations int8
  across conv1→conv2→conv3 (ReLU folds into requantize), fp16 only at the residual add. 1.17× vs
  unchained int8, but still 0.79× vs fp16 → gap is kernel quality, not plumbing.
- `e07f3ed` Deep-fuse the chaining requantize (reuse `Int8DequantScaleSource`, no fp32 temp). 0.82×.
- `c2d8db5` **Tile-parametric int8 deep-fuse conv + microbench** (`microbench_conv_tuned.py`):
  `conv2d_int8_dequant_fp16_tuned(config_id)` + 5 tile configs. Kernel best-of-5 = **1.37× vs cuDNN
  fp16** (the losing 0.63× shape → 2.01× with a small 64³ tile). Proves per-shape tiling is the fix.
- `3b9d2e9` **Autotuner + dispatch (A+B+C):** `OptimizedInt8Conv2d._ensure_tuned_config` lazily times
  all tiles per shape and caches the winner (kill-switch `MODIFF_DISABLE_CONV_AUTOTUNE=1`); wired into
  `_conv_from_int8` (baseline) and `forward_to_int8` (chaining). Diffusion int8_baseline +1.9%;
  ResNet int8_chained 0.82→0.85×.
- `b8a8811` **Tune+deep-fuse the conv3 residual path** (`conv2d_int8_fprop_deepfuse_bias_residual_fp16`
  + `bias_residual_store_half_from_half`) **+ 3 wide-N tiles** for the 1×1 expands (8 configs total).
  **Kernel int8 = 1.47× vs cuDNN fp16; ResNet-50 int8_chained 0.82→0.96× (near parity), 1.36× vs
  unchained int8.** Parity chained-vs-int8 rel 0.057; golden PASS.

## Current state / headline numbers
- **Diffusion (the shipping model): int8_baseline ~15% faster than fp16, int4 more.** MoDiff modes
  ~12–15% faster than their own unfused baseline (but slower than *_baseline — temporal-cache cost;
  MoDiff's value is quality, not speed, on this UNet).
- **ResNet-50 int8_chained: 0.96× vs fp16 (near parity), int8 kernel 1.47× vs cuDNN fp16.** The
  multi-tile autotune (built in CUTLASS, reusable across all modes incl MoDiff) closed the gap.
- All int8 conv paths are per-shape tile-autotuned; exact int32 GEMM → no numerical change.

## Key infra added (reusable)
- CUTLASS: `conv2d_int8_dequant_fp16_tuned` + 8 tile configs (`TCfg0..7`) in `csrc/kernels/conv2d_int8.cu`;
  `conv2d_int8_num_tuned_configs()`. Epilogue is a template param → orthogonal to the tile, so the
  tile set backs every epilogue variant (baseline, chaining int8-out, residual fp16-out, and — when
  routed — MoDiff `o_hat`).
- Chaining epilogues in `conv_epilogue.cuh`: `scale_bias_relu_requant_store_int8_from_half`,
  `bias_residual_store_half_from_half`.
- Python: `OptimizedInt8Conv2d._ensure_tuned_config` (autotuner), `forward_to_int8`, `quantize_input`,
  `output_requant_scale`/`fuse_output_relu`; `chained_bottleneck.py`.

## Verified findings / gotchas
- The single 128³ tile fit churches' large 3×3 convs (int8 already won there) but NOT ResNet's 23
  diverse shapes (K=64→4608). Per-shape tile selection is the fix; wide-N tiles fixed the 1×1 expands.
- **Calibration:** the module `begin/end_calibration` runs convs in a mode whose output magnitude
  differs ~10× from the fast path → mis-scaled downstream convs. For ResNet use **fp16-hook PTQ**
  (`benchmark_resnet50.build_quantized` does this: hook the fp16 model, set_static_scale(127/absmax)).
- Random-weight models are useless for quantization parity — use pretrained + realistic input. The
  chaining correctness gate is **chained-vs-unchained-int8** (isolates chaining), not vs fp16.
- Triton was rejected for this: it forks the kernel ecosystem; MoDiff's o_hat/delta fusions are
  CUTLASS. Autotune was built in CUTLASS so it serves all modes.
- INT4 is still **single-tile** (0.41× overall — wins big-K, loses 1×1s). Needs the same tile-
  parametrization mirror on `conv2d_int4.cu`.

## Next steps (prioritized)
1. **INT4 tile autotune** — mirror `conv2d_int8_dequant_fp16_tuned` + config set onto `conv2d_int4.cu`
   (int4 packed output; instruction shape 16×8×64). Highest-value remaining kernel work.
2. **Route the MoDiff `conv2d_int8_fprop_o_hat` epilogue through the tuned tile** — the autotuner
   already exists; the o_hat epilogue just needs the same tile-parametric treatment so MoDiff modes
   get the kernel win too.
3. **#3 (deferred): fold per-block entry-quantize into the prior block's conv3 store** to cross fp16
   on ResNet — needs a STAGE-LEVEL refactor to thread int8 between blocks (nn.Sequential passes one
   tensor); ~2% marginal, likely still <fp16. Only if a ResNet fp16-beating demo is required.
4. Optionally: TensorRT as an alt int8 backend for the baseline UNet (would autotune convs + fuse
   attention; big end-to-end but a separate integration).

## Commands
- Diffusion A/B: `python integration/benchmarks/ab_benchmark.py --modes fp16 int8_baseline --repeats 6
  --warmups 2 --steps 100 --batch_size 32 --calibration integration/calibration/int8_calibration.pt`
  (prefix `MODIFF_DISABLE_CONV_AUTOTUNE=1` for the before/fixed-tile side).
- ResNet: `python integration/benchmarks/benchmark_resnet50.py --batch 64 --repeats 8`.
- Kernel microbench: `python integration/benchmarks/microbench_conv_tuned.py --batch 64`.
- Correctness gate: `python integration/tests/test_kernel_correctness.py` (must stay ALL PASS).
