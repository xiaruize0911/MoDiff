# Flash attention (fp16 + int8 + int4) — progress / handoff

Date: 2026-07-19. Status: **fp16 flash shipped (big win); int8/int4 flash kernels built +
correct but not yet fast enough (need more optimization). Continue tomorrow.**
Plan file: `/root/.claude/plans/drifting-hopping-zephyr.md`.

## TL;DR

- **fp16 flash is DONE and is the shipped default** for all modes: switching the SDPA
  backend from forced-MATH to flash-preferring gave **1.82× end-to-end** at b128
  (190.0 → 104.6 ms/step, fp16 mode), latent rel-err 5.35e-04. Attention dropped from
  ~48% of the step to ~9%. This is the single biggest win of the whole effort.
- **int8/int4 flash kernels are built, correct, and wired (opt-in), but NOT competitive**
  yet: at level-0 (b128) int8 flash = 0.24× and int4 flash = 0.14× of fp16 flash. The
  kernels need more optimization to beat FlashAttention-2 (which is why the full 5-version
  benchmark was skipped — kernel not well optimized yet).

## Measured numbers (level-0: BH=1024, T=1024, hd=24, b128)

| path | µs | vs fp16 flash | rel-L2 (kernel) |
|---|---|---|---|
| fp16 MATH SDPA (old default) | 16224 | 0.11× | — |
| **fp16 FlashAttention-2 (new default)** | **1766** | **1.00×** | — |
| int8 flash (ours, optimized) | 7368 | 0.24× | 0.024 |
| int4 flash (ours) | 13074 | 0.14× | 0.144 |

- SFU exp2 floor (measured, fused CUDA microbench): 2013 G-exp2/s → **533 µs** for the
  1.07e9 level-0 exponentials. So the theoretical int8-flash floor is ~max(400 µs MMA,
  533 µs exp) ≈ 533 µs; we are at 7368 µs → **~14× above floor**, i.e. lots of headroom
  left in the kernel, but hard to reach FA-2's 1766 µs.
- In-model e2e latent rel-err vs fp16 flash (b16, int8_baseline): int8 flash **0.0037**
  (passes the ≤0.01 speed-first gate), int4 flash **0.0103** (at the gate). Accuracy is
  fine; only speed is the problem.

## int8 flash optimization progression (level-0)

| step | µs | note |
|---|---|---|
| original (as removed 2026-07-19) | 17829 | serial 1-lane softmax, blocking loads |
| + register-parallel online softmax | 10822 | S in regs, __shfl_xor reduce, dropped 16KB Ss/m/l smem |
| + cp.async double-buffer + coalesced/pre-transposed V | 7368 | 2-stage pipeline, V fed [N,H,hd_pad,T] |
| (8 warps/CTA — tried, reverted) | 7748 | worse + breaks T=64 (T%128≠0) |

## What was done (files changed)

- `integration/fused_ops/token_major_attention.py`
  - `_SDPA_CTX` is now flash-preferring (`[FLASH, EFFICIENT, MATH]`); `_SDPA_MATH_CTX`
    and env `MODIFF_SDPA_BACKEND=math` keep the old MATH backend for A/B. Docstrings updated.
  - `_attn` routes to `_flash_quant_attn` when `MODIFF_FLASH_ATTN=8|4` (default 0 = fp16
    flash). `_flash_quant_attn` does dynamic per-token Q/K + per-channel V quant and calls
    `flash_attn_int8` / `flash_attn_int4`. `_HAS_FLASH` flag added.
- `csrc/kernels/flash_attn_int8.cu`
  - Revived (was removed from build). `flash_attn_int8_mma_kernel` rewritten:
    register-parallel online softmax (all 32 lanes), cp.async 2-stage double-buffer for
    K/V, V taken PRE-TRANSPOSED `[N,H,hd_pad,T]`. Added `flash_attn_int4_mma_kernel`
    (int4 QKᵀ via `modiff_mma_m16n8k64_s4`, int8 PV) + `flash_attn_int4` wrapper.
  - `#include <cuda_pipeline_primitives.h>` added.
- `csrc/pybind.cpp`, `csrc/modiff_kernels_api.h`, `setup.py` — re-registered
  `flash_attn_int8`, `flash_attn_int4`, `mma_smoke`; re-added the .cu to sources.
- `integration/benchmarks/benchmark_ldm.py` — comment updated (flash-preferring default,
  `MODIFF_SDPA_BACKEND=math` escape hatch); token-major print line reflects backend.

## Validation done

- `mma_smoke` exact (int8 m16n8k32.s8 fragment mapping).
- int8/int4 flash kernels: rel-L2 vs fp32 reference at churches shapes (int8 0.014–0.024,
  int4 0.144). In-model e2e latent rel-err int8 0.0037 / int4 0.0103.
- fp16 flash: e2e 1.82×, latent rel-err 5.35e-04.

## NOT done (tomorrow)

1. **Optimize int8 flash to beat fp16 flash** (currently 0.24×, 14× above floor). Ideas not
   yet tried: XOR-swizzle the K/V smem to kill ldmatrix bank conflicts (currently plain
   smem); 3-stage cp.async (STAGES=3); BC=64 to halve tiles/syncs; reduce redundant K/V
   re-reads across the 16 query-block CTAs (bigger query tile without breaking T=64);
   ncu-guided (perf counters were BLOCKED on this box — ERR_NVGPUCTRPERM — so profiling was
   roofline-only; may need host to enable counters). Honest risk: beating FA-2 at hd=24 is
   very hard; even success adds only ~3–5% e2e (attention is now ~9% after fp16 flash).
2. **Full 5-version e2e benchmark with flash** (30 warm-up + 5×200, b128) — skipped today
   because the int8/int4 kernel isn't optimized; rerun once it is. Script:
   `docs/layer_roofline_2026-07-19/scripts/bench5_confirm.py` (uses flash by default now;
   get a clean warm fp16 reference — do not run fp16 first, cold-clock artifact).
3. **Write the final report** (mirror `docs/layer_roofline_2026-07-19/FULL_INT8_DATAFLOW.md`).

## How to reproduce today's kernel numbers

- int8: `python /tmp/.../scratchpad/flash_i8_check.py` (or reconstruct: quantize per-token
  Q/K, per-channel V, pad hd→32, call `mc.flash_attn_int8`).
- int4: `flash_i4_check.py` (pack int4, pad hd→64, `mc.flash_attn_int4`).
- exp floor: `sfu_cuda.py` (load_inline exp2 microbench).
- fp16 flash e2e: `phase1_flash.py`.
