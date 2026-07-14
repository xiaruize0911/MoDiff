# Session handoff — ResNet INT8+INT4 block-entry-quantize fusion, deep-fuse, weight-layout bug fix (2026-07-14, session 2)

Follow-on to `SESSION_HANDOFF_2026-07-14.md`. Rebuild before running:
`CUTLASS_PATH=/workspace/cutlass MAX_JOBS=$(nproc) python setup.py build_ext --inplace`
(int4 now compiles 6 fp32 + 6 fp16-deepfuse tile configs — the longest unit; run in background.)
Always `PYTHONPATH=/workspace/MoDiff/src/taming-transformers:$PYTHONPATH`. The LDM stack deps
(omegaconf==2.1.1, einops==0.3.0, tqdm, pytorch-lightning==1.4.2, torchmetrics==0.6.0) were
reinstalled this session (they'd been wiped from the env).

**All work is on branch `resnet-block-entry-quantize-fusion` (6 commits), COMMITTED but NOT PUSHED**
— this env has no push credentials (SSH remote, no private key / gh / token). Push from a machine
with GitHub access: `git push -u origin resnet-block-entry-quantize-fusion`.

## TL;DR — ResNet-50, A40, batch 64 (median ms/iter, random weights)
```
    fp16            21.71   1.00x
    int8_fullchain  15.22   1.43x  WIN
    int4_fullchain  14.02   1.55x  WIN  (fastest; 1.09x vs int8_fullchain)
```
Both quantized modes went from *slower than fp16* at the start of the session (int8 0.71x, int4
0.65x) to beating it. Timed set is fp16 / int8_fullchain / int4_fullchain (plain int8 & int8_chained
dropped — int8_fullchain dominates them; the code stays for reference).

## Headline results

**1. INT8 block-entry-quantize fusion (deferred step #3) — SHIPPED.**
Whole-net int8 threading (`Int8FullyChainedResNet`): the per-block entry quantize is fused into the
PREVIOUS block's conv3 store (a dual fp16+int8 epilogue), so the net quantizes to int8 exactly ONCE
(after maxpool) and stays int8 through every conv until avgpool. Removes 15 of 16 standalone
quantize kernels. **int8_fullchain = 1.43x vs fp16** (was int8_chained 0.96x — a loss).

  nsys: standalone quantize 58.0 -> 14.5 ms, ReLU 28.5 -> 4.1 ms, total GPU 225.8 -> 151.5 ms.
  Accuracy (pretrained ResNet50 IMAGENET1K_V2, 1000 real ImageNet images, 1/class):
    fp16 92.5% top1 | int8 92.9% | int8_fullchain 92.6%  -> ~lossless, and fullchain == plain int8.

**2. CRITICAL pre-existing bug found + fixed: channels_last corrupts packed int8/int4 weights.**
`build_quantized` (ResNet) did `convert -> m.to(memory_format=channels_last)`. The channels_last
conversion reformats the 4D `weight_int8` [K,R,S,C] (and int4 `weight_packed` [K,R,S,C/2]) buffer
to a channels_last stride, which **for 3x3 convs silently transposes the physical layout the
CUTLASS conv kernel reads -> garbage** (1x1 convs immune: channels_last == contiguous there).
Effect: ResNet int8 top1 0.2% (random) instead of ~92%. **Invisible to every prior check** —
random-weight consistency passes (both sides share the garbage), the correctness gate is
chained-vs-unchained (not vs fp16), and speed benchmarks don't look at values. Only real
accuracy-vs-fp16 (never run before) exposed it.
  Fix: `OptimizedInt8Conv2d._apply` / `OptimizedInt4Conv2d._apply` re-contiguate the packed weight
  after any tensor transform (no-op when already contiguous; setup-time only, no per-forward cost).
  Verified: int8 0.2% -> 92.9%. Robust regardless of build order.

**3. Diffusion int8 was NOT affected — its 13%-faster-than-fp16 result stands.**
Diffusion `_setup_model` does the OPPOSITE order (`.to(channels_last)` at benchmark_ldm.py:326
BEFORE `convert` at :454), so `weight_int8` is created after and never reformatted. Measured:
0/140 int8 conv weights non-contiguous; int8_baseline-vs-fp16 UNet eps rel L2 = 0.17 (normal quant
error). Bug simulation (forcing channels_last) would have pushed it to 0.40. So the diffusion int8
speed numbers were on numerically-correct kernels.

**4. INT4 — the full arc: fusion -> tile autotune -> deep-fuse. int4_fullchain now BEATS int8.**
Same optimizations as int8, applied in three commits, each with its own speed step (batch 64 vs fp16):
  - `5b11944` **block-entry-quantize fusion** (`Int4FullyChainedResNet`, dual fp16+packed-int4 conv3
    store; new `scale_bias_relu_requant_pack_int4` chaining kernel matching quantize.cu's nibble
    convention): **0.65x -> 1.20x**.
  - `fd8257e` **per-shape tile autotune** (`Int4ConvConfig` + 6 tiles + `conv2d_int4_fprop_tuned` +
    `_ensure_tuned_config`; closes the prior handoff's "int4 single-tile" #1): 1.20x -> **1.22x** only,
    because the real bottleneck was the fp32 conv_out temp, not the GEMM tile.
  - `9664a0c` **deep-fuse epilogue** (`Int4DequantScaleSource`, int4 twin of Int8DequantScaleSource:
    per-channel weight_scale folded into the CUTLASS GEMM -> fp16 out, no fp32 temp; chaining reads
    fp16 via `*_from_half` pack kernels): **1.22x -> 1.55x**, and **1.09x vs int8_fullchain**. Removing
    the memory-bound fp32 round-trip let int4's raw edge (half the bytes, 2x tensor-core throughput)
    finally show. THIS was the lever, not the tile.

  Correctness: fusion is faithful (int4_fullchain == plain int4 at 99.9% top1 agreement); gate
  `test_int4_dual_store` fp16-exact (rel 2e-4, nibble mean|Δ| 1e-4).
  **End-to-end int4 accuracy on ResNet stays ~random** — inherent naive 4-bit PTQ error (single-conv
  rel 0.225 compounds over 52 layers), NOT a kernel bug. Usable int4 accuracy needs MoDiff temporal
  caching or better-than-naive PTQ; the kernels are speed-correct (golden + dual-store gates pass).
  The int4 layout bug (below) is also fixed via `OptimizedInt4Conv2d._apply`.

## What shipped (all committed this session, branch `resnet-block-entry-quantize-fusion`)
INT8 (commit `e9c6e96`, benchmark trim `9575523`):
- CUDA: `bias_residual_relu_dual_store_from_half_kernel` (conv_epilogue.cuh) +
  `conv2d_int8_fprop_deepfuse_bias_residual_dual` (conv2d_int8.cu). Dual = fp16 (post-ReLU,
  residual-added) + int8 (requantized to next block's conv1 scale).
- Python: `OptimizedInt8Conv2d.forward_from_int8_dual()`; `_apply` weight-contiguity guard (int8+int4).
- `Int8FullyChainedResNet` + `build_fully_chained()` (chained_bottleneck.py).

INT4 (commits `5b11944` fusion, `fd8257e` tile autotune, `9664a0c` deep-fuse):
- CUDA (conv_epilogue.cuh): `scale_bias_relu_requant_pack_int4` + `bias_residual_relu_dual_store_pack_int4`
  (fp32-in, kept) and their `*_from_half` twins (fp16-in, used by the deep-fuse path).
- CUDA (conv2d_int4.cu): `Int4ConvConfig`/`conv2d_int4_fprop_tuned` (fp32-out tuned) and
  `Int4DequantScaleSource`/`Int4DequantFp16Config`/`conv2d_int4_dequant_fp16_tuned` (deep-fuse, fp16-out)
  + `conv2d_int4_num_tuned_configs`; the two chaining launchers now deep-fuse.
- Python: `OptimizedInt4Conv2d` gains `quantize_input`, `forward_to_int4`, `forward_from_int4_dual`,
  `_ensure_tuned_config` (times the deep-fuse kernel), `weight_scale_channel_half` (fp16 epilogue
  source, kept in sync); `Int4FullyChainedResNet` + `build_fully_chained_int4`.

Shared:
- `benchmark_resnet50.py`: `int8_fullchain` + `int4_fullchain` modes; optional `weights=` (pretrained
  accuracy). Timed set = fp16 / int8_fullchain / int4_fullchain.
- `test_kernel_correctness.py`: `test_int8_conv_channels_last` (layout-bug regression),
  `test_int8_dual_store`, `test_int4_dual_store`. Gate is ALL PASS.
- Autotune kill-switch (both int8+int4): `MODIFF_DISABLE_CONV_AUTOTUNE=1`.

## Verified findings / gotchas
- The layout bug is layout-only: `weight_int8` VALUES reconstruct fp16 fine (rel 0.019); only the
  physical stride was wrong. A FRESH `OptimizedInt8Conv2d` from the same weights was always correct
  (rel 0.0004 self-consistent) — that's how it was isolated.
- **int4's real bottleneck was the fp32 conv_out temp, not the GEMM tile** — tile autotune alone gave
  +2% (1.20->1.22x); the deep-fuse (no fp32 temp) gave +27% (1.22->1.55x). int8 always deep-fused, so
  its tile autotune paid off (1.29->1.43x); int4 didn't until `9664a0c`.
- Correctness gate for a fusion is fullchain-vs-chained / vs-plain (same scales) = ~100% top1 agree.
  Accuracy-vs-fp16 needs PRETRAINED weights (random weights hide quantization errors).
- `*_fullchain` fp16 output of each conv3 is ALREADY ReLU'd (fused) — next block must not re-ReLU.
- `quantize_input` (int8 and int4) uses `static_input_scale` directly, NOT the cacheable
  `_cached_scale_tensor` (which can go stale if populated before `set_static_scale` — this bit the
  int4 dual-store gate until fixed).
- int4 packed tensors are PLAIN-contiguous `[N,H,W,C/2]` (not channels_last) — the packed launchers
  use `is_contiguous()`, NOT the `CHECK_CONTIGUOUS` macro (which demands channels_last).

## Next steps (prioritized)
1. **Push the branch** (blocked here on credentials) and open a PR.
2. **Route the MoDiff `conv2d_int8_fprop_o_hat` epilogue through the tuned tile** (prior handoff #2) —
   and consider an `o_hat` deep-fuse; the int8 & int4 deep-fuse/tuned machinery now exists to reuse.
3. **Usable int4 accuracy**: needs MoDiff temporal caching or better-than-naive PTQ (per-channel-act /
   GPTQ-style). The int4 *kernels* are speed-correct; only naive PTQ quality is the gap.
4. **Audit any prior INT8/INT4 accuracy/quality claim** for the layout bug — only the ResNet
   `convert -> to(channels_last)` path was affected (diffusion order was safe), but worth a sweep.
5. Diffusion analogue of the block-entry dual store: N/A while GN hides the quantize (only relevant if
   a conv->conv path there ever lacks a norm).

## Commands
- ResNet all modes + fusion: `python integration/benchmarks/benchmark_resnet50.py --batch 64 --repeats 8`
- Correctness gate: `python integration/tests/test_kernel_correctness.py` (must stay ALL PASS)
- Scratch validation scripts (accuracy, layout root-cause) live in the session scratchpad, not the repo.
