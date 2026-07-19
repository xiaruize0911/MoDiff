# Status: AWQ-split wins at 14/15 real shapes once its ascale/out buffers are cached (2026-07-18)

## History (short version — see REPORT.md §11-§14 for full detail)

1. §11/§12: naive int8 split loses to fp16-**fused** (`fused_gn_qkv`) — an architecture mismatch, not
   a precision one. §12 predicted a GN-prologue GEMM fuse would close the gap (1.6–2.1×).
2. §13: built that fuse on two GEMMs (our own, and a fork of AWQ's `dense_kernel0_fuse_bias`) — both
   lost badly (0.25–0.55×). Structural: `cp.async` pipelines are fast because the loader does zero
   compute; GN+quantize needs real per-element math, which can't ride `cp.async`. Confirmed by AWQ's
   own TinyChat pipeline, which also never fuses quantize into its GEMM. **Reverted** both fuse
   attempts from `csrc/` (`git checkout` + delete) — confirmed clean.
3. §13 (cont.): the fair comparison is split-vs-split (identical op count: norm[+quant] → GEMM →
   bias-add on every path). On 3 synthetic qkv shapes, AWQ-split won at C384/C768, tied at C192.
4. **§14 (this update)**: extended to all **15 real AWQ-eligible GEMM shapes** in the UNet (see
   below) — found and fixed a real AWQ-calling-convention inefficiency, using `nsys` (not `ncu` —
   blocked in this environment, see below).

## The 15 real shapes (corrects an earlier error: C768 was benchmarked at synthetic T=64 — the real
model runs C768 at T=16/T=4; T=64 belongs to C384's second occurrence)

- qkv + proj at 5 (C,T) combos: C192 T1024, C384 T256, C384 T64, C768 T16, C768 T4 (10 shapes).
- 5 time-embedding MLPs at **M = batch_size only** (no token multiplication): `time_embed[0]`,
  `time_embed[2]`, and `ResBlock.emb_layers` at each of C192/C384/C768 — all AWQ-eligible per
  `wxax_linear.py`'s `_eligible()`, a totally different tiny-M regime from qkv/proj.

## The finding: AWQ's calling convention costs an extra kernel launch at tiny M

First pass: AWQ lost badly (0.47–0.94×) at every tiny-M shape, despite its GEMM kernel itself only
taking ~3.5µs there (confirmed via `nsys --trace=cuda,nvtx` + `cuda_gpu_kern_sum`). Root cause: AWQ's
`w8a8_gemm_forward_cuda` needs a per-token `ascale` tensor, materialized via `torch.full(...)` — **a
whole extra kernel launch** (median CPU dispatch ~3.5µs, from `cuda_api_sum`) that fp16 and our own
`gemm_w8a8` don't need (they take a scalar Python float). At tiny M, kernel *count* dominates wall
time more than kernel *speed*: fp16 = 2 kernels/call, ours = 3, AWQ (as normally called) = 4.

**Fix, validated by direct A/B**: `a_scale` is static (calibrated) and `M` is constant per layer in
this UNet (unlike AWQ's original target: dynamic per-token LLM decoding) — so `ascale` and the output
buffer can be allocated **once** and reused, like the weight padding already is. Measured 1.80×
speedup at `time_embed[0]` from just this, no kernel change.

## Current numbers (A40, µs, split-stage: norm[+quant] → GEMM → bias-add)

| shape | fp16-split | our int8×fp16 | AWQ×fp16 (cached) |
|---|--:|--:|--:|
| qkv C192 T1024 | 376 | 0.71× | 0.99× |
| proj C192 T1024 | 232 | 0.91× | 1.02× |
| qkv C384 T256 | 247 | 1.06× | 1.16× |
| proj C384 T256 | 137 | 1.06× | 1.03× |
| qkv C384 T64 | 88 | 1.02× | 1.09× |
| proj C384 T64 | 60 | 1.02× | 1.03× |
| qkv C768 T16 | 43 | 0.94× | 1.09× |
| proj C768 T16 | 36 | 1.12× | 1.08× |
| qkv C768 T4 | 36 | 1.33× | 1.49× |
| proj C768 T4 | 40 | 1.51× | 1.65× |
| time_embed[0] | 22 | 1.12× | 1.28× |
| time_embed[2] | 26 | 1.30× | 1.11× |
| emb_layers Cch192 | 32 | 1.26× | 1.40× |
| emb_layers Cch384 | 33 | 1.26× | 1.44× |
| emb_layers Cch768 | 27 | 1.10× | 1.25× |

rel-err vs fp32 reference: 0.008–0.013 across all shapes, under the 0.02 gate.

**AWQ beats or ties fp16 at 14/15 shapes** (loses only 0.99× at qkv C192, a rounding error).

## Recommendation — DONE

**Use `group_norm_silu_quantize_nhwc` (or plain `quantize_act_int8` for the no-norm time-embed MLPs)
→ AWQ's native `w8a8_gemm_forward_cuda`, WITH the ascale/output buffer cached per layer.** Patched
into `integration/kernels/wxax_linear.py`'s `QuantLinearWxAx._gemm` (2026-07-18): `_awq_asc`/
`_awq_out` are now plain scratch attributes, reallocated only on M/device change, refilled only when
`a_scale` actually changed. Verified: 5 repeated static-scale calls bit-identical; correct when
`a_scale` or `M` change mid-run; MoDiff's dynamic delta-scale lifecycle unaffected (still correct,
just doesn't get the speedup, no regression either); a bias-less-layer aliasing hazard fixed with a
defensive `.clone()`. Real-module latency at `time_embed[0]`: ~32.5µs (was ~54µs uncached).

## Open items if this is picked up further
- `ncu` (Nsight Compute) is installed (`/usr/local/cuda/bin/ncu`) but **blocked**:
  `ERR_NVGPUCTRPERM` — GPU perf-counter access needs a host/driver permission this container lacks,
  even as root. `nsys` works (binary at
  `/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys`, not on `PATH` by default) and was
  sufficient for this finding (`--trace=cuda,nvtx --stats=true`, `cuda_gpu_kern_sum`/`cuda_api_sum`
  reports). If `ncu` becomes available later, occupancy/roofline metrics on the AWQ vs our-own GEMM
  kernels at large-M would be a natural follow-up (is AWQ's large-M win compute-bound tiling, or
  something else?).
- Take a real e2e number with this patch active (run `benchmark_ldm.py` with `MODIFF_QUANT_LINEAR=1`
  and calibration/`finalize_wxax_ascale` run first, so `a_scale` is static and the caching actually
  triggers). This report's numbers are isolated per-layer latency, not e2e.

## Files
- `scripts/split_stage.py` — 3-shape split-vs-split (§13, superseded in coverage by split_stage_full.py).
- `scripts/split_stage_full.py` / `data/split_stage_full.csv` — all 15 real shapes + the caching fix
  (§14, current).
- `csrc/kernels/group_norm_silu.cu` — `group_norm_silu_quantize_nhwc` (norm+quant fused kernel).
- `integration/kernels/wxax_linear.py` — **patched** with the ascale/out caching fix (§14).
- Build: `source setup_cuda_env.sh; MAX_JOBS=$(nproc) CUTLASS_PATH=/workspace/cutlass python3.11 setup.py build_ext --inplace`
- Profiling: `export PATH="/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64:$PATH"` then `nsys profile --trace=cuda,nvtx --stats=true -o <out> python3.11 <script>`.

## §15: optimizing our OWN GEMM (`gemm_w8a8`) toward AWQ — in progress (2026-07-18)

**Not about replacing AWQ** (it remains the recommended production backend, per §14) — this is about
closing the gap in our own fallback kernel, which AWQ can't cover for every case (no true AWQ W4A4
kernel exists — llm-awq only ships W4A16, weight-only). Worst-measured gap: `qkv C192/T1024`
(M=32768, K=192), our kernel at 0.71× fp16 vs AWQ's 0.99×. Full plan: `/root/.claude/plans/plan-on-how-to-moonlit-pearl.md`.

### Diagnosed root cause
Our kernel's K-tile was fixed at 32 elements/main-loop-iteration (`gemm_wxax.cu`, `nkt = K/32`), so
K=192 took 6 iterations — mainloop-bound (documented earlier at
`docs/comprehensive_benchmark_2026-07-15/REPORT.md:240-243`). AWQ uses `CTA_K=64` throughout (2×
`m16n8k32` mma calls per main-loop step). Two other AWQ advantages identified but NOT yet attempted
(higher risk/effort, deferred as optional per the plan): `ldmatrix`+XOR-swizzle shared-memory reads
(we use plain scalar pointer casts, no bank-conflict mitigation — the `ldmatrix` wrapper already
exists unused in `mma_int8.cuh`), and a 128-wide N-tile (ours is fixed at `GW_BN=64`).

### Stage 1 — double the int8 K-tile to 64 (matching AWQ): DONE, mixed result
Implemented in `gemm_w8a8_kernel` (int4's `gemm_w4a4_kernel` untouched — its native `m16n8k64` mma
already covers 64 logical K/iteration, so it didn't need this): new `GW_LDS8=64` smem row stride,
load two 32-wide sub-tiles per iteration, 2 `mma.m16n8k32` calls per (mi,nt) instead of 1. `K%64==0`
now required (all real shapes already satisfy this). Correctness: `test_kernel_correctness.py` all
pass, `int8_linear` matches its golden reference exactly.

**Measured (bare GEMM, 5 repeated trials, low-noise since M is large)**:
| shape | before | after | change |
|---|--:|--:|--:|
| C192 qkv (target) | 270µs | 209.3µs | **1.29× faster** ✅ |
| C384 qkv (control) | ~89-95µs | 102.9µs | **~7-15% slower** ⚠️ |

**Full split-stage (repeated, stable at large-M; tiny-M shapes M≤128 swing up to 2× run-to-run —
noise, not attributable to this change)**:
| shape | before | after |
|---|--:|--:|
| qkv C192 T1024 | 0.71× fp16 | **0.80× fp16** (target was ≥0.85×, plan's go/no-go bar) |
| proj C192 T1024 | 0.91× | 0.95–0.96× |
| qkv C384 T256 | 1.06× | 1.00–1.01× (small regression) |
| proj C384 T256 | 1.06× | 1.01× (small regression) |
| qkv C384 T64 | 1.02× | 0.97× (small regression) |

**Root cause of the regression**: C384 qkv uses the same `MT=1` tiling as C192 (ruling out an
MT=2-specific register-pressure theory) — so the cost (holding 2× the A/B fragments live at once,
`a0`/`a1`/`b0`/`b1` instead of `a`/`b`) is paid on every shape equally, but the benefit (fewer
mainloop iterations) only outweighs it when K was short enough that iteration overhead dominated.
K=192 (3 iterations after the fix) clears that bar; K=384 (6 iterations) doesn't.

**Decision (user confirmed)**: proceed to shape-gate the fix rather than accept the regression or
revert — add a compile-time toggle so K<384 uses the new 64-wide-K kernel and K≥384 keeps the
original 32-wide-K kernel, capturing the C192 win with no cost elsewhere.

### Shape-gating: DONE, regression eliminated, C192 win preserved

Implemented via a third template parameter `WideK` (`gemm_w8a8_kernel<MT, OUT_I8, WideK>`) with
`if constexpr` branching the mainloop body — one compiled kernel per `(MT, OUT_I8, WideK)`
combination, selected at the host level by a new `gw_pick_widek(K)` (`K < 384`, override via
`MODIFF_GW_WIDEK` env var). Both `gemm_w8a8` and `gemm_w8a8_out_int8` updated; `gemm_w4a4` untouched
(never needed this). Rebuilt, `test_kernel_correctness.py` all pass (`int8_linear`, K=4096, uses the
narrow path, matches golden exactly — confirms the untouched path is bit-identical to pre-Stage-1).

**Measured (bare GEMM, 5 repeated trials each)**:
| shape | Stage 1 (unconditional wide) | shape-gated | vs original (pre-Stage-1) |
|---|--:|--:|--:|
| C192 qkv (wide path) | 209.3µs | **209.2µs** (unchanged — win preserved) | 270µs → **1.29× faster** |
| C384 qkv (narrow path) | 102.9µs (regressed) | **93.6µs** (regression gone) | ~89-95µs → **back to baseline** |

**Full split-stage pipeline** — C384/proj-C384 all returned to their exact pre-Stage-1 numbers
(qkv C384 T256: 1.06× fp16, matching the original 1.06× exactly; proj C384 T256/T64 likewise). The
C192 win holds: **0.71× → 0.80× fp16** (full pipeline includes the GN+quant kernel's fixed cost, so
the bare-GEMM's 1.29× speedup translates to a smaller full-pipeline gain).

**Honest gap remaining**: 0.80× is still short of the plan's ≥0.85× stretch target, and the bare GEMM
(209µs) is still ~1.85× slower than AWQ's (113µs) at this exact shape. Closing that further needs
Stage 3 (`ldmatrix`+XOR-swizzle for bank-conflict-free shared-memory reads, plus AWQ's 128-wide
N-tile vs our fixed 64) — both deferred as optional/higher-risk in the original plan, not started.
Given AWQ remains the recommended production backend regardless (see §14), the net win here is a
better, regression-free fallback kernel for shapes AWQ can't cover (e.g. int4), not parity with AWQ.

### Stage 2 — deeper pipeline (`GW_STAGES` 3→4): DONE, small further win, no regression

User confirmed doing Stage 2 and Stage 3 ("Stage 2和Stage 3也做一下吧"). Changed
`#define GW_STAGES 3` → `4` globally in `gemm_wxax.cu` (affects both int8 paths — WideK/narrow — and
int4). Rebuilt (~1hr, zero errors), ran full verification.

**Correctness**: `test_kernel_correctness.py` — all pass, including `int8_linear`/`int4_conv` golden
checks.

**Measured (bare GEMM, 5 repeated trials)**:
| shape | Stage 1 (shape-gated) | Stage 2 (+deeper pipeline) | change |
|---|--:|--:|--:|
| C192 qkv (wide, nkt=3=STAGES-1) | 209.2µs | **190.4µs** | further **9% faster** |
| C384 qkv (narrow, nkt=12) | 93.6µs | **93.6µs** | unchanged, no regression |
| C768 qkv (narrow, nkt=24, new datapoint) | — | 79.4µs | — |

**Full 15-shape split-stage sweep**: C192 qkv full-pipeline speedup improved **0.80× → 0.83× fp16**;
every other shape matches Stage-1 numbers within run-to-run noise (C384 qkv/proj still ~1.03–1.06×,
C768/time-embed shapes unchanged). No regressions anywhere.

**Why C192 improved further while C384 didn't move**: C192's WideK path has `nkt=3` main-loop
iterations — with `GW_STAGES=4` the whole K dimension now prefetches during the prologue before
compute starts at all (`GW_STAGES-1 == nkt`), removing the last synchronization stall. C384's narrow
path has `nkt=12`, already well past 3 stages before this change — going to 4 stages doesn't remove
any additional stall there, consistent with the unchanged latency.

**Decision**: keep `GW_STAGES=4`. C192 now at **0.83× fp16** full-pipeline (bare GEMM 190.4µs, still
~1.68× behind AWQ's 113µs at this shape — down from ~1.85× after Stage 1 alone).

### Stage 3 — AWQ-tiling-scheme port (`ldmatrix`+XOR-swizzle, N-tile=128): in progress

User's guidance mid-design: our kernel's warp-to-tile assignment is a mirror image of AWQ's large-M
config (we partition M across warps + share B redundantly across warps; AWQ partitions N across warps
+ shares A redundantly) — so AWQ's exact swizzle/`ldmatrix` read-side address formulas can't be
copy-pasted; either a re-derivation (real correctness risk, `ncu` unavailable to independently verify
bank-conflict elimination) or a full restructure to AWQ's scheme is needed. User decided: **"先在
int8 上验证，通过了再搬到 int4"** (validate on int8 first, then port to int4) — build a new,
standalone kernel `gemm_w8a8_kernel_awq` that fully replicates AWQ's tiling
(`CTA_M=128,CTA_N=128,CTA_K=64,WARP_M=128,WARP_N=32`, 4 warps, `STAGES=3`) with real
`ldmatrix`+swizzle, on top of our existing plain pre-quantized-int8 A/B contract (no compute-in-loader
— avoids the `cp.async`-breaking issue that killed the earlier `fused_gn_qkv_awq.cu` attempt). Kept
separate from the existing Stage-1/2 kernel (safe fallback unaffected); dispatched only for
validation, not wired into the production `gemm_w8a8` path yet. Not yet written to any file as of this
update — design (constants, swizzle formula, per-warp read offsets for A/B) reasoned through, next
step is implementation + correctness + bare-GEMM timing vs AWQ's own kernel at the same shape.

### Stage 3 result: DONE, validation PASSED — beats AWQ's own kernel at 4/6 real shapes

Implemented `gemm_w8a8_kernel_awq` (new standalone kernel, `gemm_w8a8_awq` Python entry point, NOT
wired into `gemm_w8a8`'s dispatch) exactly as designed: `CTA_M=CTA_N=128, CTA_K=64, WARP_N=32`, 4
warps (each owns a distinct 32-wide N-slice, redundantly reads the full M-range of A — matching
AWQ's large-M warp-tiling exactly), real `ldmatrix.m8n8.x4` fragment reads with the XOR swizzle
`col ^ ((row/2)&3)` applied identically on write and read (transcribed directly from AWQ's
`share_to_reg_one_stage_A/B` in `w8a8_gemm_cuda.cu`). Simplified vs AWQ: skips AWQ's register
ping-pong prefetch-overlap across the two `INTRIN_K=32` sub-iters within a `CTA_K=64` tile (load then
compute per sub-iter, no overlap) — a deliberate correctness-first scope reduction for a validation
kernel. Requires `N%128==0` (pad B/w_scale at the call site, exactly like AWQ's own kernel requires
its callers to do). Rebuilt clean (zero errors).

**Correctness**: bit-identical (`rel_err=0.0, max_abs_diff=0.0`) vs our own golden-validated
`gemm_w8a8` across M∈{37,256,300,2048} (multiple-of-128 and not), K∈{192,384,768}, N∈{128,384,2304}.
Full `test_kernel_correctness.py` suite still all-pass (no regression to the untouched kernels).

**Bare-GEMM benchmark (real qkv/proj shapes, N padded to next 128-multiple where needed;
repeated 5x, stable)**:
| shape | M,K,N | ours (gemm_w8a8) | ours-awq-tiling | AWQ reference | vs ours | vs AWQ |
|---|---|--:|--:|--:|--:|--:|
| C192 qkv | 32768,192,576 | 190.1µs | **98.2µs** | 116.7µs | **1.94×** | **1.19×** |
| C192 proj | 32768,192,192 | 65.4µs | **47.1µs** | 54.6µs | **1.39×** | **1.16×** |
| C384 qkv | 8192,384,1152 | 86.9µs | **63.0µs** | 70.5µs | **1.38×** | **1.12×** |
| C384 proj | 8192,384,384 | 33.8µs | 33.1µs | 34.4µs | 1.02× | 1.04× |
| C768 qkv | 2048,768,2304 | 75.8µs | 57.6µs | 54.0µs | 1.32× | 0.94× |
| C768 proj | 2048,768,768 | 27.6µs | 27.8µs | 24.1µs | 1.00× | 0.90× |

**This beats AWQ's own reference kernel outright at 4/6 shapes** (C192 qkv/proj, C384 qkv/proj) —
not just closes the gap. It loses modestly (0.90–0.94×) only at the two longest-K shapes (C768,
K=768). Consistent with the one simplification made: skipping AWQ's register-prefetch overlap costs
more as iteration count grows (K=768/64=12 iterations vs K=192/64=3), since there's more opportunity
for that overlap to hide latency; at short K it barely matters and our simpler mainloop code has less
overhead. This is a strong signal the ldmatrix+swizzle+128-wide-N mechanism itself is the dominant
win (explains beating our own `gemm_w8a8` everywhere, 1.00–1.94×), and AWQ's remaining edge at long-K
is a smaller, secondary optimization (prefetch overlap) that could be added later if needed.

**Verdict: Stage 3 validation PASSED** on int8, per the plan's own bar ("先在int8上验证，通过了再搬到int4").
Next: port this validated mechanism to `gemm_w4a4` (Stage 3's actual target — no AWQ W4A4 kernel
exists to compete against, so the bar there is simply "beat our own existing `gemm_w4a4`", which
given the int8 result (1.00–1.94× win over our own analogous kernel) is a reasonable expectation).

### int4 port + production merge + e2e/IO: DONE (2026-07-18 overnight) → see SESSION_REPORT_2026-07-18.md

`gemm_w4a4_awq` built, **bit-identical on first try**, beats our own `gemm_w4a4` 6/6 (1.02–2.10×) and
fp16 6/6 (1.15–2.29×). Both kernels merged into `wxax_linear.py` behind `MODIFF_WXAX_AWQTILE`
(default OFF). **e2e within noise** on this conv-bound UNet (Linears ~9% of a step); flag-on adds
+0.7% D2D I/O and loses the AWQ ascale/out caching at tiny-M layers. **Follow-up before default-on:**
a buffer-caching kernel variant (write into caller-provided output, cache per-layer) to remove the
per-call alloc and stop the tiny-M regression — then int4 (no alternative, biggest win) is a clear
default-on candidate. Full detail + all tables: `SESSION_REPORT_2026-07-18.md`; run trace: `OVERNIGHT_LOG.md`.

### Consolidation: ports are now the SOLE Linear backend (2026-07-18) → see SESSION_REPORT_2026-07-18_consolidate_ports.md

Per user request, retired the hand-written `gemm_w8a8`/`gemm_w4a4`(+`*_out_int8`, shared template) —
moved to `csrc/kernels/backup/` (non-compiled) — and made `gemm_w8a8_awq`/`gemm_w4a4_awq` the
unconditional int8/int4 Linear backend (no `MODIFF_WXAX_AWQTILE` flag, no `awq_inference_engine`
runtime dependency; AWQ-ref kept as a benchmark baseline only). The test-only fused-int8-qkv→flash
prototype in `quantized_attention.py` was refactored off `*_out_int8`. Rebuilt (ninja, parallel);
all correctness gates pass. Re-ran the full suite: kernel bench (int8 vs fp16 6/6, vs AWQ-ref 4/6;
int4 vs fp16 6/6 up to 2.33×), nsys (int8 98.2µs / AWQ-ref 115.5µs / int4 93.7µs on C192 qkv), e2e
(30 warm-up + 5 runs × 200 steps; **port-only slower than fp16, batch-sensitive to a ~9% floor**:
int8 +21% @ b16 → +9.0% @ b128, int4 ~+8.5% @ b128 — batch part = CTA_M=128 tile under-fill at
M=batch time-embed layers, floor = `quantize_act` O(M·K) pass + Amdahl), and total I/O (+~1%, all D2D).
**Still-open follow-ups (ordered by the batch sweep):** (1) buffer-caching port variant — removes
per-call alloc/ascale, helps **small batch** but NOT the ~9% large-batch floor; (2) **fuse
`quantize_act` into the preceding op's epilogue** (or a small-M kernel for the M=batch layers) to kill
the O(M·K) quantize pass — needed to close the ~9% floor. Full tables + batch sweep:
`SESSION_REPORT_2026-07-18_consolidate_ports.md`.
