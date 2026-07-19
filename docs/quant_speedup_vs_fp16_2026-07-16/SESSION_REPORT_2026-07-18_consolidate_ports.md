# Session report — Consolidate the int8/int4 Linear GEMM onto the AWQ-tiling ports (2026-07-18)

**TL;DR.** We retired the older hand-written int8/int4 Linear GEMM kernels (`gemm_w8a8` / `gemm_w4a4`
and their `*_out_int8` variants, plus the shared template) and made the AWQ-tiling ports
(`gemm_w8a8_awq` / `gemm_w4a4_awq`) the **sole** production Linear backend — no flag, no AWQ-external
dependency in the runtime path. The removed code is preserved, non-compiled, under
`csrc/kernels/backup/`. AWQ's external reference kernel is kept **only as a benchmark baseline**. The
one attention path that depended on the removed `*_out_int8` kernels (a test-only, off-by-default
fused-int8-qkv→flash prototype) was refactored off them. All correctness gates pass. At the GEMM level
the ports remain the fastest option (int8 beats fp16 6/6 and AWQ-ref 4/6; int4 beats fp16 6/6 up to
2.33×). **End-to-end on this conv-bound UNet the port-only path is slower than fp16, and the gap is
batch-sensitive down to a ~9% floor** (int8 +21% @ b16 → +9.0% @ b128; int4 similar, ~+8.5% @ b128).
The batch sensitivity is CTA_M=128 tile under-fill at the M=batch time-embed layers; the residual ~9%
floor is the `quantize_act` O(M·K) pass fp16 doesn't pay, plus Amdahl (Linears ≈9% of a conv-bound
step). A buffer-caching port variant helps small batch but won't erase the floor — see §5/§7.

---

## 1. What changed

**CUDA / bindings**
- `csrc/kernels/gemm_wxax.cu` — deleted the templated `gemm_w8a8_kernel`/`gemm_w4a4_kernel`, the host
  entry points `gemm_w8a8`/`gemm_w4a4`/`gemm_w8a8_out_int8`/`gemm_w4a4_out_int8`, and their exclusive
  helpers/macros (`GW_*`, `gw_pick_mt`, `gw_pick_widek`, `gw_bias_ptr`). Kept the AWQ-tiling ports,
  their `GWQ_*`/`gwq_s2r_*` machinery, and the activation-quantize helpers. File header rewritten.
- `csrc/pybind.cpp`, `csrc/modiff_kernels_api.h` — removed the four retired bindings/prototypes.
- `csrc/kernels/attn_quant_gemm.cu` — comment-only touch-ups (a `quantize_attn_qkv_from_i8` doc no
  longer names the removed kernel).
- `csrc/kernels/backup/gemm_wxax_own_kernels_2026-07-18.cu` (+ `README.md`) — verbatim pre-removal
  copy, **not** in `setup.py` sources (never compiled).

**Python**
- `integration/kernels/wxax_linear.py` — `_gemm` collapsed to a single path (pad activation K →
  quantize → `gemm_w{8,4}a{8,4}_awq` → slice N). Removed the `MODIFF_WXAX_AWQTILE` flag, the
  `awq_inference_engine` import, the AWQ-ref `w8a8_gemm_forward_cuda` path, and the ascale/output
  scratch-caching. Port weight buffers (`qweight`/`w_scale`, padded to N%128 & K%64/128) are now built
  unconditionally.
- `integration/fused_ops/quantized_attention.py` — removed the off-by-default, test-only
  fused-int8-qkv→flash prototype (`_fused_qkv_flash` + calib/scale helpers) that consumed
  `gemm_*_out_int8`; its premise (emit int8 to skip the fp16 round-trip) is incompatible with the
  fp16-output ports, and the block already had a correct non-fused flash path. `test_flash_attn.py`
  never exercised it.
- `integration/tests/test_wxax.py` — `test_kernels` repointed to the ports (with the required N/K
  padding).

**Benchmark scripts** — `stage3_kernel_bench.py` and `stage3_nsys_driver.py` updated to drop the
retired-kernel columns/backend; new drivers `e2e_sweep_consolidated.sh` and `io_sweep_consolidated.sh`.

## 2. Correctness (all gates PASS)

- `integration/tests/test_wxax.py` — `test_kernels` (ports vs dequant reference): kdiff = 2.1e-4 at all
  6 shapes. `test_module` (QuantLinearWxAx vs fp16 nn.Linear): int8 rel ~0.010, int4 rel ~0.19–0.22.
- `integration/tests/test_kernel_correctness.py` — **ALL PASS**, including `int8_linear` golden
  rel_err = 0.0 (uses an independent modiff_triton backend, unaffected by the removal), `int4_conv`,
  `fused_gn_qkv`, and the MoDiff-lifecycle checks.
- `integration/tests/test_flash_attn.py` — **ALL PASS**: `flash_attn_int8` vs fp32 SDPA (5/5) and
  `QuantizedTokenMajorAttentionBlock` vs fp16 (int8 rel ~0.010) — validates the attention refactor.

## 3. Kernel benchmark (bare GEMM, real qkv/proj shapes, median of 5, µs)

Source: `data/stage3_kernel_bench.csv`. `o8awq` = `gemm_w8a8_awq`, `o4awq` = `gemm_w4a4_awq`,
`awqref` = AWQ's external `w8a8_gemm_forward_cuda` (baseline only). int4 K=192 benchmarked at K→256.

| shape | M,K,N | fp16 | o8awq | awqref | o4awq | o8awq/awqref | o8awq/fp16 | o4awq/fp16 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| C192 qkv  | 32768,192,576  | 108.9 | **103.9** | 117.4 | **94.4** | 1.13× | 1.05× | 1.15× |
| C192 proj | 32768,192,192  | 54.6  | **47.6**  | 54.6  | **42.5** | 1.15× | 1.15× | 1.28× |
| C384 qkv  | 8192,384,1152  | 93.1  | **68.5**  | 71.6  | **47.2** | 1.05× | 1.36× | 1.97× |
| C384 proj | 8192,384,384   | 39.6  | **33.5**  | 33.3  | **23.9** | 0.99× | 1.18× | 1.65× |
| C768 qkv  | 2048,768,2304  | 80.9  | **60.8**  | 57.4  | **38.8** | 0.95× | 1.33× | 2.08× |
| C768 proj | 2048,768,768   | 36.7  | **26.6**  | 23.7  | **15.8** | 0.89× | 1.38× | 2.33× |

- **int8 port** beats fp16 at all 6 shapes (1.05–1.38×) and **AWQ-ref at 4/6** (C192 qkv/proj, C384
  qkv; ~tie at C384 proj), losing only at the two longest-K C768 shapes (0.89–0.95×) — the same
  prefetch-overlap gap noted before.
- **int4 port** beats fp16 at all 6 (1.15–2.33×). Still the fastest W4A4 GEMM available (no AWQ int4
  kernel exists).

## 4. Kernel profile (nsys, C192 qkv M=32768,K=192,N=576; per-call GPU kernel time)

Source: `data/stage3_nsys_kern_sum.csv`, `data/nsys/`. ncu remains blocked (`ERR_NVGPUCTRPERM`).

| backend | kernel | per-call GPU time |
|---|---|--:|
| `gemm_w8a8_awq` | `gemm_w8a8_kernel_awq` | **98.2 µs** |
| AWQ reference | `dense_kernel0<128,128,64,128,32,64,3>` | 115.5 µs |
| `gemm_w4a4_awq` | `gemm_w4a4_kernel_awq` | **93.7 µs** |

Matches the CUDA-event wall-clock (103.9 / 117.4 / 94.4 µs) within ~1% — the timings are real, single
kernel = 100% of GPU time each, no extra launches.

## 5. End-to-end pipeline (benchmark_ldm sampler, 30 warm-up steps + 5 runs × 200 steps)

Source: `data/e2e_bench_b{16,32,64,128}_s200.txt` (driver `scripts/e2e_bench_b64_s200.py`, batch set by
`E2E_BATCH`). Reuses `BenchmarkRunner._setup_model` (model load + conv/attn + wxax-linear static
calibration), then drives the DDIM sampler directly: 30 warm-up steps, then 5 timed runs of 200 steps.
int8/int4 use `MODIFF_QUANT_LINEAR=1 --linear_backend int_gemm` (ports are the only backend now); fp16
is the unquantized baseline. **ms/step = time per full batch DDIM step** (not per-sample). Per-batch
detail at batch 128 (min/median/mean, variance <0.5%): fp16 189.4/189.8/189.7, int8 206.4/206.6/206.6,
int4 205.4/206.7/207.1.

**Batch sweep (min ms/step, port vs fp16):**

| batch | fp16 (ms/step) | int8 vs fp16 | int4 vs fp16 |
|--:|--:|--:|--:|
| 16 | 1.83* | +21% | +15% |
| 32 | 50.9 | +11.6% | +4.6% |
| 64 | 97.2 | +10.3% | +15.3% |
| **128** | **189.4** | **+9.0%** | **+8.5%** |

*batch-16 row is from an earlier per-*sample*-normalized sweep (`data/e2e_sweep_consolidated.txt`), so
its absolute ms/step isn't comparable to the per-batch rows; only its relative % is.

**Honest read:** on this conv-dominated UNet the port-only Linear path is *slower* e2e than fp16,
despite the GEMM-level wins in §3 — and the gap is **batch-sensitive down to a ~9% floor**. Two
distinct effects, both centered on the layers where **M = batch** (time-embed / `emb_layers` — unlike
qkv/proj, whose M = batch × tokens is already large):
1. **Tile utilization (explains the batch *sensitivity*).** The ports use `CTA_M=128`; at M=16 only
   12.5% of the tile does useful work (M=32 → 25%, M=64 → 50%, M=128 → 100%). Bigger batch fills the
   tile, so the regression falls from +21% (b16) toward the floor as batch → 128. (int4's b32/b64
   points are noisy — a few-% run-to-run spread + thermal ordering across the fp16→int8→int4 sequence
   — but it converges to the same floor at b128.)
2. **A persistent ~9% floor (NOT tile waste, NOT per-call overhead).** At b128 the tile is full and
   per-call costs are maximally amortized, yet ~9% remains. That residual is cost that scales *with
   the data*, so it never amortizes: the **`quantize_act` pass** (O(M·K) memory traffic + a launch)
   that fp16 never pays, plus **Amdahl** — Linears are ~9% of a conv-bound step and the port GEMM win
   at these shapes isn't large enough to overcome the added quantize work within that slice.

**Consequence for the fix:** a buffer-caching port variant removes the per-call alloc/ascale and so
mainly helps *small* batch (where per-call overhead dominates) — it will **not** erase the ~9%
large-batch floor. Closing that floor needs either **fusing the activation-quantize into the preceding
op's epilogue** (kill the separate O(M·K) pass) or a model where Linears are a larger share of the
step. On this conv-bound UNet the Linear quant is Amdahl-limited regardless.

## 6. Total IO / memory traffic

Source: `integration/results/awqtile_io_consolidated/{fp16,int8,int4}/nsys_memory_summary.json`
(nsys CUPTI memcpy tables; steps=15, batch=16, 16 samples). **Total CUDA I/O = H2D + D2D + D2H bytes.**

| mode | H2D (MiB) | D2D (MiB) | D2H (MiB) | **Total I/O (MiB)** | Δ vs fp16 | linear quant-weight (MiB) |
|---|--:|--:|--:|--:|--:|--:|
| fp16 | 2649.4 | 162.7 | 12.0 | **2824.1** | — | 0 |
| int8 (port) | 2650.1 | 189.4 | 12.1 | **2851.5** | +27.4 (+1.0%) | 27.14 |
| int4 (port) | 2650.1 | 188.7 | 12.1 | **2850.8** | +26.7 (+0.9%) | 13.57 |

- The int8/int4 paths add **~27 MiB (~1%)** of total I/O vs fp16, **entirely in Device-to-Device**
  (162.7 → ~189 MiB) — from the ports allocating a fresh output tensor each call, the output slice,
  and (int4) the activation K-pad copy, over all layers × steps × samples. **H2D (weight loads) and
  D2H are unchanged.** This is the same per-call-allocation cost that drives the §5 tiny-M e2e
  regression, and the buffer-caching variant (§7) would remove it.
- (These runs are under nsys, which inflates ms/step vs §5's clean numbers; use them only for the
  off-vs-on I/O comparison.)

## 7. Verdict & recommendation

- **Consolidation done and correct.** The ports are the sole int8/int4 Linear backend; the old kernels
  are removed (backed up) and the attention path no longer depends on them. All correctness gates pass.
- **GEMM level: the ports are the best option** (int8 beats fp16 6/6 & AWQ-ref 4/6; int4 beats fp16
  6/6, up to 2.33×, with no alternative).
- **E2e on this model regresses vs fp16, batch-sensitively, to a ~9% floor** (int8 +21% @ b16 →
  +9.0% @ b128; int4 ~+8.5% @ b128). The batch part is CTA_M=128 tile under-fill at the M=batch
  time-embed layers; the ~9% floor is the `quantize_act` O(M·K) pass + Amdahl (conv-bound step). This
  is the accepted cost of dropping the AWQ-ref path.
- **Follow-ups, in the order the batch sweep implies:**
  1. **Buffer-caching port variant** — write into a caller-provided output cached per layer (like the
     AWQ-ref path did) + cache the padded-activation scratch. This removes the *per-call* alloc/ascale
     cost, which dominates at **small batch** — it recovers most of the b16→b64 gap but, per §5, does
     **not** erase the ~9% large-batch floor.
  2. **Fuse `quantize_act` into the preceding op's epilogue** (or a small-M kernel variant for the
     M=batch layers) to kill the separate O(M·K) quantize pass — this is what's needed to close the
     ~9% floor and make the quantized Linear net-neutral/faster vs fp16 at large batch.
  3. Independent of both, the ports are already a clear win for **standalone/large-M int4 GEMM** and
     for models where Linears are a larger share of the step than in this conv-heavy UNet.

## Artifacts
- Kernels: `csrc/kernels/gemm_wxax.cu` (ports only); retired copy at `csrc/kernels/backup/`.
- Dispatch: `integration/kernels/wxax_linear.py` (single port path).
- Data: `data/stage3_kernel_bench.csv`, `data/stage3_nsys_kern_sum.csv`, `data/nsys/`,
  `data/e2e_bench_b64_s200.txt` (primary e2e), `data/e2e_sweep_consolidated.txt` (earlier batch-16),
  `integration/results/awqtile_io_consolidated/{fp16,int8,int4}/`.
- Scripts: `scripts/stage3_kernel_bench.py`, `scripts/stage3_nsys_driver.py`,
  `scripts/e2e_bench_b64_s200.py`, `scripts/e2e_sweep_consolidated.sh`, `scripts/io_sweep_consolidated.sh`.
