# Session report — AWQ-tiling int8/int4 GEMM kernels: build, validate, benchmark, profile, merge (2026-07-18)

**TL;DR.** We finished optimizing our own int8/int4 Linear GEMM by porting AWQ's tiling scheme
(`ldmatrix` + XOR bank-swizzle + 128-wide N-tile) into two new kernels, `gemm_w8a8_awq` and
`gemm_w4a4_awq`. Both are **bit-identical** to the existing golden kernels. At the GEMM level the new
int8 kernel **beats AWQ's own reference at 4/6 real shapes** (up to 1.17×) and our previous kernel
everywhere (1.0–1.9×); the new int4 kernel — which has **no AWQ competitor at all** — beats our
previous `gemm_w4a4` at all 6 shapes (1.0–2.1×) and fp16 at all 6 (1.15–2.29×). We merged both into
the production module behind an opt-in flag (`MODIFF_WXAX_AWQTILE`, default OFF). **End-to-end, on this
conv-dominated UNet, the speedup is within run-to-run noise** — expected, since the quantized Linear
layers are only ~9% of a diffusion step (Amdahl). The kernels' value is as faster standalone GEMMs
(especially int4) rather than an e2e win on this particular model.

---

## 1. What we did this session

Continuing the plan `/root/.claude/plans/plan-on-how-to-moonlit-pearl.md` (closing the gap between our
own Linear GEMM and AWQ's), across three optimization stages plus an int4 port and a production merge:

- **Stage 1 — K-tile 32→64, shape-gated (`K<384`).** Our int8 mainloop consumed 32 K-elements/iter
  vs AWQ's 64; the short-K qkv shape (K=192) was mainloop-bound. Doubling it won on C192 but cost
  register pressure at long K, so it's gated (`gw_pick_widek(K)=K<384`). C192 bare GEMM 270→209µs.
- **Stage 2 — deeper pipeline (`GW_STAGES` 3→4).** For C192 (nkt=3) the whole K now prefetches in the
  prologue. C192 209→190µs; no regression elsewhere.
- **Stage 3 — AWQ-tiling port (`gemm_w8a8_awq`).** A from-scratch kernel replicating AWQ's large-M
  tiling (`CTA_M=CTA_N=128, CTA_K=64, WARP_N=32`, 4 warps, STAGES=3) with real
  `ldmatrix.m8n8.x4` + XOR swizzle (`col ^ ((row/2)&3)`) shared→register loads — the mechanisms our
  earlier kernel lacked. Operates on plain pre-quantized int8 (no compute-in-loader, so `cp.async`
  stays a zero-compute pipeline).
- **int4 port (`gemm_w4a4_awq`).** The identical tiling/swizzle/`ldmatrix` machinery, reused verbatim
  — only the mma primitive (`m16n8k64.s4` vs `m16n8k32.s8`) and the packed-K stride differ. The
  `ldmatrix`/swizzle operate on dtype-agnostic 16-byte chunks, and `m16n8k64.s4`'s per-warp fragment
  footprint equals `m16n8k32.s8`'s, so the same helpers carried over. **Bit-identical on the first
  try** — no debugging needed.
- **Merge.** Wired both kernels into `integration/kernels/wxax_linear.py` behind
  `MODIFF_WXAX_AWQTILE={int8,int4,both}` (default OFF → zero behavior change), with one-time load-time
  weight padding (N%128; K%64 int8 / %128 int4) and matching activation zero-pad at call time.

## 2. Correctness

- `integration/tests/test_kernel_correctness.py`: **ALL PASS** (no regression to any existing kernel).
- Bespoke bit-identical check vs golden `gemm_w8a8`/`gemm_w4a4` across M∈{37,128,256,300,512,2048}
  (multiple-of-128 and not), K∈{128,192,256,384,768}, N∈{128,384,768,2304}: **rel_err = 0.0,
  max_abs_diff = 0.0** for both int8 and int4.
- `integration/tests/test_wxax.py` (module-level, vs fp16 `nn.Linear`): ALL PASS both flag-off and
  flag=both, identical rel-err (int8 ~0.010, int4 ~0.19–0.22 — expected quantization error, not kernel
  error).

## 3. Kernel benchmark (bare GEMM, real qkv/proj shapes, median of 5, µs)

Source: `data/stage3_kernel_bench.csv`. `o8awq` = new `gemm_w8a8_awq`; `o4awq` = new `gemm_w4a4_awq`;
`ours8`/`ours4` = previous kernels; `awqref` = AWQ's `w8a8_gemm_forward_cuda`. (int4 has no AWQ
equivalent; int4 K=192 shapes are benchmarked at K padded→256, the kernel's requirement.)

| shape | M,K,N | fp16 | ours8 | **o8awq** | awqref | ours4 | **o4awq** | o8awq/ours8 | o8awq/awq | o8awq/fp16 | o4awq/ours4 | o4awq/fp16 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| C192 qkv  | 32768,192,576  | 108.2 | 189.6 | **98.2** | 115.1 | 197.4 | **94.0** | 1.93× | **1.17×** | 1.10× | 2.10× | 1.15× |
| C192 proj | 32768,192,192  | 53.8  | 66.3  | **47.1** | 53.8  | 58.2  | **42.3** | 1.41× | **1.14×** | 1.14× | 1.38× | 1.27× |
| C384 qkv  | 8192,384,1152  | 89.2  | 95.5  | **66.1** | 70.2  | 66.0  | **46.1** | 1.45× | **1.06×** | 1.35× | 1.43× | 1.93× |
| C384 proj | 8192,384,384   | 38.0  | 33.4  | **32.7** | 33.5  | 24.1  | **23.7** | 1.02× | **1.03×** | 1.16× | 1.02× | 1.60× |
| C768 qkv  | 2048,768,2304  | 78.8  | 79.4  | **59.3** | 54.7  | 48.0  | **38.8** | 1.34× | 0.92× | 1.33× | 1.24× | 2.03× |
| C768 proj | 2048,768,768   | 36.1  | 26.7  | **26.6** | 23.7  | 16.6  | **15.8** | 1.01× | 0.89× | 1.36× | 1.05× | 2.29× |

- **int8 `gemm_w8a8_awq`**: beats fp16 at all 6 shapes (1.10–1.36×), beats our previous kernel at all 6
  (1.01–1.93×), and **beats AWQ's own kernel at 4/6** (1.03–1.17×), losing only at the two longest-K
  C768 shapes (0.89–0.92×).
- **int4 `gemm_w4a4_awq`**: beats our previous `gemm_w4a4` at all 6 (1.02–2.10×) and fp16 at all 6
  (1.15–2.29×). This is the most valuable result — llm-awq ships no W4A4 kernel, so there was no faster
  alternative before this.

## 4. Profiler results (nsys; `ncu` blocked in this container)

Per-call GPU kernel time on the biggest-win shape C192 qkv (M=32768,K=192,N=576), single kernel = 100%
of GPU time each (`data/stage3_nsys_kern_sum.csv`, `data/nsys/`):

| backend | kernel | per-call GPU time |
|---|---|--:|
| `gemm_w8a8_awq` | `gemm_w8a8_kernel_awq` | **97.9 µs** |
| AWQ reference | `dense_kernel0<128,128,64,128,32,64,3>` | 115.6 µs |
| our previous `gemm_w8a8` | `gemm_w8a8_kernel<1,0,1>` | 190.0 µs |
| `gemm_w4a4_awq` | `gemm_w4a4_kernel_awq` | **93.7 µs** |

These match the CUDA-event wall-clock benchmark within <1% (98.2 / 115.1 / 189.6 / 94.0 µs),
confirming the timings are real and not measurement artifacts. `cuda_api_sum` was dominated by
`cudaDeviceSynchronize` — no per-call launch overhead, no extra kernels.

**Interpretation — why we beat AWQ at short-K but lose at long-K.** The new kernel is faster than our
old one everywhere because `ldmatrix`+swizzle removes shared-memory bank conflicts and the 128-wide
N-tile improves reuse. Against AWQ specifically, we win at short/medium K (C192/C384) because our
mainloop is simpler (less per-iteration bookkeeping), but we deliberately omitted AWQ's register
ping-pong prefetch-overlap across the two INTRIN_K=32 sub-iterations — an optimization whose benefit
grows with iteration count. At C768 (K=768 = 12 iterations) that missing overlap costs us ~8–11%; at
C192 (3 iterations) it's negligible and our simpler code wins.

## 5. Pipeline speedup (e2e, `benchmark_ldm.py`, steps=30, batch=16, 16 samples, min of 3 reps)

Source: `data/e2e_sweep.txt`. Mode = `MODIFF_QUANT_LINEAR=1` (qkv/proj/time-embed Linears quantized),
`--linear_backend int_gemm`.

| mode | flag OFF (ms/step) | flag ON (ms/step) | Δ |
|---|--:|--:|--:|
| int8 | 1.792 | 1.805 | −0.7% (within noise) |
| int4 | 1.753 | 1.784 | −1.8% (within noise) |

**The e2e delta is within run-to-run noise — no measurable e2e speedup (marginally slower on).** This
is the expected result on this **conv-dominated** UNet: the quantized Linear layers are only ~9% of a
diffusion step, so a 1.1–2.3× GEMM speedup is ~2% e2e at best — below the ~1–3% run noise. Two
compounding effects explain the marginal *slowdown* with the flag on:
1. **Amdahl** — the Linear portion is too small a fraction to move the total.
2. **Small per-call overheads the flag-on path adds at tiny-M layers**: the int8 flag-OFF path uses
   AWQ with cached ascale/output buffers (the §14 fix, which specifically helps the M=16 time-embed
   layers); the flag-ON path calls `gemm_w8a8_awq`/`gemm_w4a4_awq`, which allocate a fresh output each
   call, and the int4 path adds an activation K-pad. At batch=16 those tiny-M layers slightly regress,
   offsetting the big-M attention layers' GEMM win.

## 6. Total IO / memory traffic

Source: `integration/results/awqtile_io/{off,on}/nsys_memory_summary.json` (nsys memcpy tables via
`run_nsys_memory_redo.sh` + `analyze_nsys_memory.py`; steps=15, batch=16, 16 samples). **Total CUDA
I/O = H2D + D2D + D2H memcpy bytes.**

| flag | mode | H2D (MiB) | D2D (MiB) | D2H (MiB) | **Total I/O (MiB)** | linear quant-weight (MiB) |
|---|---|--:|--:|--:|--:|--:|
| OFF | int8 | 2650.1 | 165.0 | 12.1 | **2827.1** | 27.14 |
| ON  | int8 | 2650.1 | 184.9 | 12.1 | **2847.1** | 27.14 |
| OFF | int4 | 2650.1 | 164.2 | 12.1 | **2826.4** | 13.57 |
| ON  | int4 | 2650.1 | 183.5 | 12.1 | **2845.7** | 13.57 |

- Flag-on adds **~20 MiB (~0.7%)** of total I/O, entirely in **Device-to-Device** (165→185 MiB int8;
  164→184 int4) — from `gemm_*_awq` allocating a fresh output tensor each call, the output slice, and
  (int4) the activation K-pad copy, accumulated over all layers × steps × samples. **H2D (weight
  loads) and D2H are unchanged**, and the tracked quantized-weight footprint is identical (27.14 MiB
  int8 / 13.57 int4) — the small N/K weight padding doesn't materially change memory.
- These extra D2D copies are part of why the flag-on e2e was marginally slower; a buffer-caching
  kernel variant (§7) would remove them.
- (Note: the ms/step here differs from §5 because it's measured *under nsys*, which distorts timing;
  use §5's clean numbers for the speedup claim. These runs are only for the off-vs-on I/O comparison.)

## 7. Verdict & recommendation

- **Kernel work: success.** Both new kernels are bit-identical and faster than every prior option at
  the GEMM level; the int8 kernel even beats AWQ's own kernel at the shapes that dominate this model.
  The int4 kernel is the standout — it's the fastest W4A4 GEMM available here (nothing to compare
  against previously) and beats fp16 by up to 2.3×.
- **Production default: keep `MODIFF_WXAX_AWQTILE` OFF for now.** On this conv-bound UNet the e2e
  numbers don't justify swapping the battle-tested AWQ int8 path, and the flag-on tiny-M regression
  needs a fix first. The right follow-up before flipping the default is a **buffer-caching variant**
  of `gemm_w8a8_awq`/`gemm_w4a4_awq` (write into a caller-provided output, cache it per layer like the
  AWQ path does) so the tiny-M time-embed layers stop regressing; then int4 in particular (no
  alternative, biggest GEMM win) becomes a clear default-on candidate.
- **Where this kernel pays off today:** any standalone/large-M int4 GEMM use, and models where Linear
  layers are a larger share of the step than in this conv-heavy UNet.

## Artifacts
- Kernels: `csrc/kernels/gemm_wxax.cu` (`gemm_w8a8_kernel_awq`/`gemm_w8a8_awq`,
  `gemm_w4a4_kernel_awq`/`gemm_w4a4_awq`); bound in `csrc/pybind.cpp`.
- Merge: `integration/kernels/wxax_linear.py` (`MODIFF_WXAX_AWQTILE`).
- Data: `data/stage3_kernel_bench.csv`, `data/stage3_nsys_kern_sum.csv`, `data/nsys/`,
  `data/e2e_sweep.txt`, `data/io_runs.log`, `integration/results/awqtile_io/{off,on}/`.
- Progress log: `OVERNIGHT_LOG.md`. Prior detail: `NEXT_STEPS.md`.
