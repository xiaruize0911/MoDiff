# Fusion fix pass — addendum to REPORT.md (2026-07-22)

Follow-up to the fusion-gap audit of this benchmark. Goal: close genuine + stranded fusion
gaps, each behind a `MODIFF_*` flag, validated (correctness) and profiled (speed). Plan:
`/root/.claude/plans/based-on-your-research-jiggly-parasol.md`. GPU A40, torch 2.4.1+cu124.

Headline: the two genuine e2e wins (Phases 1, 2) landed + validated; Phase 5 (int4 attention
parity) landed as a GPU-work reduction; the remaining audited gaps turned out, on measurement, to
be **already at the fused floor** (latency-hidden per §7/§2) or a **measured regression** — the
front-end was more optimized than the trace-only audit implied.

## Delivered + validated

### 1. CUDA-graph capture/replay for the per-step UNet (gap: launch overhead)
Wired `install_cuda_graph_replay_pytorch_int8` into `BenchmarkRunner._generate_samples`
(`--cuda_graph` / `MODIFF_CUDA_GRAPH=1`); capture after calibration + cuDNN/autotune settle,
in-place-only state reset between replays, fixed batch + final-batch slice.
- **Correctness:** graph replay bit-identical to eager, all 5 modes (rel-L2 = 0.0,
  `scripts/verify_cuda_graph_e2e.py`).
- **Speed (int8_baseline):** b8 **2.16 → 1.25 ms/step-per-sample (1.73×)**; b32 1.01 → 0.97
  (1.04×); b128 OOMs (no capture headroom). CUDA graphs are a **launch-bound / small-batch**
  win — at b128 the GPU is already saturated and launch overhead is hidden. Compounds with the
  GPU-work reductions below in that regime.

### 2. MoDiff `o_hat` conv deep-fuse + autotune (gap: MoDiff conv was a second-class path)
The old `o_hat` conv ran the base fp32-output GEMM + a separate `scale_accumulate` pass (no
weight-scale-in-epilogue, no autotuned tile) — so MoDiff conv ≈ fp16 (0.99×) while the baseline
got 1.39×/2.00×. New kernels: `accumulate_from_half[_residual]_kernel` +
`conv2d_int8/int4_dequant_fp16_o_hat[_residual]_tuned` (deep-fuse into fp16 scratch, no fp32
temp, autotuned tile → fp16 `+=` cache). Python dispatch `_o_hat_conv[_residual]` behind
`MODIFF_DEEPFUSE_OHAT` (**default ON**, opt-out `=0`).
- **Correctness:** kernel deep-fuse-vs-base agreement 9e-4 (`test_int8/int4_ohat_deepfuse` in
  `test_kernel_correctness.py`); in-model e2e latent rel-L2 < 5e-5 vs the deep-fuse-OFF golden
  (within the 0.02 bar). The fp16-before-add cache divergence is negligible and does not
  accumulate over 50 DDIM steps.
- **Speed (conv microbench, b128):** int8_modiff **46.05 → 38.19 ms/step (1.21×, was 0.99× vs
  fp16)**; int4_modiff **33.78 → 29.17 (1.16×, 1.35→1.58× vs fp16)**. Baselines unchanged
  (33.16 / 23.22) — confirming the change is isolated to the modiff path.

### 5. int4 attention GN→pack fold (gap: int4 attention less fused than int8)
int4 attention (unlike int8) ran standalone GroupNorm + int4 qkv (`quantize_act_int4_pack` +
K=192→256 `F.pad`). Added `QuantLinearWxAx.can_from_int4()` / `forward_from_int4()` and an int4
branch in `QuantizedStandardAttentionBlock.forward` (`MODIFF_FUSE_GN_QKV_I4`, default on) that
emits the packed int4 qkv input straight out of `group_norm_silu_quantize_pack_nhwc`, mirroring
the int8 `forward_from_int8` fold. The C=192 K-pad is handled in Python by zero-padding the
**packed** activation (96→128 bytes) — the weight's padded K-channels are zero, so padded nibbles
contribute 0. **No CUDA rebuild** (reuses existing kernels).
- **Engagement:** `forward_from_int4` fires for all attention blocks (in_features 192/384/768),
  incl. 25 C=192 padded calls / warmup — confirmed by a call counter.
- **Correctness:** bit-identical to the fallback (e2e rel-L2 = 0.0, `scripts/verify_int4_gn_pack_e2e.py`):
  int4 buckets (~0.7 apart) absorb the fp16-vs-fp32 GN-rounding difference. `test_wxax` +
  `test_kernel_correctness` still green.
- **Speed:** GPU-work reduction (b32 gpu_busy 33.01 → 32.42 ms/step, ~2%); e2e-neutral wall like
  the int8 fold (§7) — a GPU-work win that pays off in the small-batch + CUDA-graph regime.

### 8. Hygiene
Corrected stale "production use" comments on the vendored `awq_w8a8_gemm` (the model uses the
own-port `gemm_w8a8_awq`); annotated `fused_gn_qkv_int8` as a dead path (its consumer
`quantize_attn_qkv_from_i8` was never implemented); marked `modiff_fused/` (Triton prototype) as
non-shipped.

## Assessed — NOT built (evidence-backed)

The remaining audited gaps were re-examined against the per-kernel e2e data in this report and
found not worth building at the reported b128 (each would reduce GPU work but not wall — the
seams are latency-hidden — while adding real kernel complexity/risk to delicate paths). They
would help only in the small-batch + CUDA-graph regime; recommend a dedicated session if that
regime is a target.

- **3. GN→delta-quant fusion — KEPT OFF (measured regression).** Verified bit-identical but a
  ~2-3 ms/step loss: the fused kernel iterates group-major, so NHWC `a_hat`/`x` access is strided
  by C (poorly coalesced) at the dominant low-CPG/high-spatial shapes, vs the coalesced separate
  `step1`. This is a defensible shape-driven choice. Genuine fix = rewrite the delta kernel's
  access pattern to match the fast `group_norm_silu_quantize_nhwc` (speculative; team measured
  net-negative).
- **4. Fold attention Q/K quantize into the qkv GEMM — already at floor.** The qkv GEMM's fp16
  output is read *directly* (strided views, no copy) by a single `quantize_attn_qkv_packed_static`
  kernel that does quantize + head-major layout transform + V-transpose, feeding flash (whose
  epilogue already emits int8 for proj). Folding Q/K into the GEMM epilogue needs a custom
  head-major-*scatter* int8 epilogue (GEMM emits `[M,3C]` row-major; flash needs `[b,nh,T,hd]`;
  V is channel-major) — and §7 already measured these attention-quantize folds as e2e-neutral.
  High cost, ~0 e2e gain.
- **6. conv→GN int8 handoff — HBM-traffic only, complex.** `group_norm_silu_dequant_quantize_nhwc`
  (int8-in GN) is built but unused; wiring it needs an int8-output conv on the diffusion path +
  int8-in GN on clean straight-through chains (no residual/concat between). Benefit is HBM traffic
  only; §2 shows elementwise/copy is already small in quant modes and CUDA graphs remove the
  launch component.

## Reproduce
```
cd /workspace/MoDiff && source setup_cuda_env.sh && python setup.py build_ext --inplace
python integration/tests/test_kernel_correctness.py                    # +test_int8/int4_ohat_deepfuse
PYTHONPATH=src/taming-transformers python docs/benchmark_5mode_2026-07-21/scripts/verify_cuda_graph_e2e.py
for m in int8 int4; do MODIFF_DEEPFUSE_OHAT=1 PYTHONPATH=src/taming-transformers \
  python integration/tests/e2e_output_check.py --mode $m --compare --tol 0.02; done
python docs/benchmark_5mode_2026-07-21/scripts/verify_int4_gn_pack_e2e.py   # int4 GN->pack fold (Phase 5)
PYTHONPATH=src/taming-transformers python integration/benchmarks/benchmark_ldm.py \
  --mode int8_baseline --batch_size 8 --steps 100 --num_samples 8 --cuda_graph \
  --calibration integration/calibration/int8_calibration.pt --skip_calibration   # graph speed
```
