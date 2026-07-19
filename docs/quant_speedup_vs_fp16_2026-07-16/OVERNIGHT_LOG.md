# Overnight autonomous run log (2026-07-18)

Plan: `/root/.claude/plans/plan-on-how-to-moonlit-pearl.md`. Steps: verify int4 → benchmark →
profile → sanity → merge → pipeline speedup → total IO → report → memory → close pod.

Append-only, timestamped. Each step logs START then END (with outcome).

---

- [08:59] Build confirmed: `.so` relinked 08:59:52, zero errors, `gemm_w8a8_awq`+`gemm_w4a4_awq` import OK.
- [STEP 1 START] int4 correctness gating — test_kernel_correctness.py + bit-identical gemm_w4a4_awq vs golden gemm_w4a4.
- [STEP 1 END ✅ PASS] int4 `gemm_w4a4_awq` **bit-identical** (rel_err=0, max_abs=0) to golden `gemm_w4a4`
  on first try across 6 boundary shapes (M∈{37,128,256,300,512,2048}, K∈{128,256,384,768}, N∈{128,384,768,2304}).
  int8 `gemm_w8a8_awq` re-confirmed bit-identical too. Full `test_kernel_correctness.py` = ALL PASS
  (no regression). Gate passed → int4 is included in the merge. No debug/rebuild needed.
- [STEP 2 START] Kernel benchmark: int8+int4 AWQ-tiling vs own vs AWQ-ref vs fp16, 6 shapes ×5 repeats.
- [STEP 2 END ✅] `data/stage3_kernel_bench.csv` written. Median-of-5, µs. Highlights:
  - **int8 `gemm_w8a8_awq`**: vs own `gemm_w8a8` 1.01–1.93×; vs AWQ ref 0.89–1.17× (beats AWQ 4/6,
    loses at long-K C768); vs fp16 1.10–1.36×. Reproduces the earlier Stage-3 finding.
  - **int4 `gemm_w4a4_awq`**: vs own `gemm_w4a4` **1.02–2.10×**; vs fp16 **1.15–2.29×** — beats both at
    ALL 6 shapes. No AWQ int4 competitor exists, so this is the clearest, most valuable win.
  - (int4 K=192 shapes benchmarked at K padded→256 since int4 kernel needs K%128.)
- [STEP 3 START] nsys profile: C192 qkv through gemm_w8a8_awq / AWQ ref / gemm_w8a8; cuda_gpu_kern_sum + cuda_api_sum.
- [STEP 3 END ✅] nsys reports under `data/nsys/`, summary CSV `data/stage3_nsys_kern_sum.csv`.
  Per-call GPU kernel time (C192 qkv, M=32768,K=192,N=576), single kernel = 100% of GPU time each:
  ours8awq **97.9µs** | awqref **115.6µs** | ours8 **190.0µs** | ours4awq **93.7µs**. Matches the
  CUDA-event benchmark (98.2 / 115.1 / 189.6 / 94.0µs) within <1%. cuda_api_sum is dominated by
  `cudaDeviceSynchronize` (waiting on the 200 batched launches) — no per-call launch overhead / no
  extra kernels. Confirms gemm_w8a8_awq is 1.18× AWQ and 1.94× our own at this shape, at the GPU level.
- [STEP 4 END ✅] Sanity pass: profiler per-kernel times match wall-clock benchmark within <1% across
  all 4 backends → no measurement bug, no NaN/anti-speedup. Results trustworthy. (Steps 3+4 merged.)
- [STEP 5 START] Merge into wxax_linear.py behind MODIFF_WXAX_AWQTILE (default OFF).
- [STEP 5 END ✅] Merged. Module-level `_AWQTILE`/`_awqtile_on(bits)`; `__init__` builds padded
  `qweight_awqt`+`w_scale_awqt` (N%128, K%64 int8 / %128 int4, zero-pad weight + 1.0-pad scale) when
  the flag selects the layer's dtype; `_gemm` routes to `gemm_w8a8_awq`/`gemm_w4a4_awq` (zero-pads
  activation K, slices output back to out_features). Verified: flag UNSET → `_awqtile=False`, zero
  behavior change. flag=both → int8 rel_err_vs_fp16 ~0.010, int4 ~0.19 (both match expected
  quant accuracy, not kernel error), output shapes correct, int4 K=192→256 pad handled.
- [STEP 6 START] Pipeline: test_wxax.py (flag on) + benchmark_ldm.py e2e int8/int4, flag off vs on.
- [STEP 6 note] Installed missing LDM deps (omegaconf, einops, pytorch-lightning, kornia, torchmetrics).
- [STEP 6 END ✅ with honest caveat] `test_wxax.py` ALL PASS both flag-off and flag=both (module routes
  through new kernels, identical rel-err). E2e (`benchmark_ldm.py`, steps=30, batch=16, 16 samples, min
  of 3 reps, ms/step) in `data/e2e_sweep.txt`:
    int8  OFF **1.792**  |  ON **1.805**  (ON −0.7%, within noise; reps overlap)
    int4  OFF **1.753**  |  ON **1.784**  (ON −1.8%, within noise)
  **Finding: e2e delta is within run-to-run noise — no e2e speedup, marginally slower on.** Expected
  per the conv-bound Amdahl caveat (Linears ~9% of a step). Two compounding reasons the GEMM win
  (1.1–2.3× at the kernel level) doesn't surface e2e: (a) Amdahl — a 1.3× speedup on 9% of the step is
  ~2% e2e, below the ~1–3% run noise here; (b) the flag-on int8 path loses the AWQ ascale/output-buffer
  caching (§14 fix) that helps tiny-M time-embed layers (M=batch=16), and `gemm_w8a8_awq`/`_w4a4_awq`
  allocate a fresh output each call + the int4 path adds an activation K-pad — small per-call overheads
  that offset the big-M attention-layer GEMM win at this batch size. Honest takeaway for the report:
  the new kernel is a real standalone-GEMM win (esp. int4, the only option), but wiring it as the
  default in THIS conv-bound pipeline is not justified by e2e numbers; a buffer-caching kernel variant
  would be needed to also win the tiny-M layers. Flag stays default-OFF.
- [STEP 7 START] Total IO via run_nsys_memory_redo.sh + analyze_nsys_memory.py, flag off vs on.
- [STEP 7 END ✅] Total CUDA I/O (H2D+D2D+D2H MiB), nsys, steps=15/batch=16/16 samples:
  int8 OFF **2827.1** → ON **2847.1** (+0.7%); int4 OFF **2826.4** → ON **2845.7** (+0.7%). Extra is
  all D2D (fresh output alloc + slice + int4 activation K-pad); H2D/D2H + weight footprint unchanged.
  Data: `integration/results/awqtile_io/{off,on}/nsys_memory_summary.json`, `data/io_runs.log`.
- [STEP 8 END ✅] Wrote `SESSION_REPORT_2026-07-18.md` (what-we-did, kernel bench, profiler, e2e, IO,
  verdict + buffer-caching follow-up recommendation, flag stays default-OFF).
- [STEP 9 END ✅] Updated `modiff-quant-speedup-report.md` + `MEMORY.md` (int4 result + merge state +
  buffer-caching follow-up). Added continuity pointer to `NEXT_STEPS.md`.
- [STEP 10 START] Close pod: `runpodctl config --apiKey <held-in-session>` then `stop pod 5ddublyzz556ki`.
  All report artifacts confirmed on disk first (see checklist below). This is the last action — the
  container terminates on stop, so any line after this may not be written.
- [STEP 10 ⚠️ BLOCKED] `runpodctl config --apiKey <...>; runpodctl stop pod 5ddublyzz556ki` was
  **blocked by the Claude Code auto-mode permission classifier** (state-changing infra command held
  for explicit human approval). Did NOT work around it. Pod is STILL RUNNING. The user must either:
  (a) run `runpodctl config --apiKey <their key>` then `runpodctl stop pod 5ddublyzz556ki` in an
  interactive shell, (b) add a Bash permission rule allowing `runpodctl`, or (c) stop the pod from the
  RunPod web console. All other work (steps 1–9) is COMPLETE and on disk. Chose `stop` (reversible
  pause), never `remove`.
- [RUN COMPLETE] All kernel/report/merge/e2e/IO work done; only the pod-close remains, blocked as above.
